"""Duplicate detection.

Three tiers, cheapest first, because a full SHA-256 pass over a large model library is
hours of disk I/O and almost all of it is wasted:

**Tier 0 — physical identity.** Files sharing a ``(device, inode)`` pair are one file with
two names. No bytes are read, and — critically — the second name is *not* reclaimable
space. Reporting it as such would send the user deleting files that free nothing.

**Tier 1 — quick signature.** Only files that share an exact byte size with another file
are read at all, and only their head and tail. A file with a unique size cannot have a
duplicate, so the overwhelming majority of a library is eliminated for the cost of a
``stat`` that already happened during the walk.

**Tier 2 — full SHA-256.** Reserved for quick-signature collisions. Persisted, so the
work is done once and reused by verification and by later scans.

Assets are then grouped by the multiset of their payload hashes, which is what turns
"these 400 files are identical" into the answer the user actually wants: "these two model
folders are the same download".
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import Asset, AssetFile, DuplicateGroup, DuplicateMember
from ai_asset_manager.backend.models.enums import DuplicateKind
from ai_asset_manager.backend.utils.hashing import (
    HashCancelled,
    combine_hashes,
    quick_signature,
    sha256_file,
)
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Files below this size are ignored. Duplicate configs and READMEs are inevitable and
#: uninteresting; the question being answered is "what is wasting real space?".
DEFAULT_MIN_FILE_BYTES = 1024 * 1024

#: Two assets sharing at least this fraction of their payload hashes are near-duplicates
#: — the same download with a README or a `.gitattributes` added.
NEAR_DUPLICATE_THRESHOLD = 0.9

#: Upper bound on a "defining" configuration file read during signature refinement.
#: Real configs are kilobytes; anything larger is data, not definition.
MAX_DEFINING_FILE_BYTES = 4 * 1024 * 1024

#: Filenames that describe an asset without defining it. A copy differing only in these
#: is the same download, so they are kept out of the identity signature.
_DOCUMENTATION_NAMES = frozenset(
    {
        ".gitattributes", ".gitignore", ".gitmodules", "license", "licence",
        "notice", "notice.txt", "license.txt", "licence.txt", "authors",
        "contributing", "code_of_conduct", "citation.cff", "changelog",
    }
)

_DOCUMENTATION_SUFFIXES = (".md", ".rst", ".html", ".pdf", ".log")


def _is_documentation(relpath: str) -> bool:
    """Report whether a file documents an asset rather than defining it."""
    name = os.path.basename(relpath).lower()
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (
        name in _DOCUMENTATION_NAMES
        or stem in _DOCUMENTATION_NAMES
        or name.endswith(_DOCUMENTATION_SUFFIXES)
    )


@dataclass(slots=True)
class DuplicateStats:
    """What a duplicate pass did and found."""

    files_considered: int = 0
    size_collision_candidates: int = 0
    quick_hashed: int = 0
    fully_hashed: int = 0
    bytes_read: int = 0
    file_groups: int = 0
    asset_groups: int = 0
    near_duplicate_groups: int = 0
    wasted_bytes: int = 0
    #: Bytes that look duplicated but share storage, so deleting frees nothing.
    shared_storage_bytes: int = 0
    errors: int = 0

    def summary(self) -> str:
        """Return a one-line summary for logs."""
        return (
            f"{self.file_groups} file group(s), {self.asset_groups} asset group(s), "
            f"{self.wasted_bytes} reclaimable byte(s); "
            f"hashed {self.quick_hashed} quick / {self.fully_hashed} full"
        )


ProgressHook = Callable[[str, int, int], None]


class DuplicateDetector:
    """Finds duplicate files and assets in the catalogue."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressHook | None = None,
    ) -> None:
        """Initialise the detector.

        Args:
            session: Open database session.
            settings: Configuration; defaults to the singleton.
            cancel_event: Cooperative cancellation flag, checked between files.
            on_progress: Called with ``(phase, completed, total)``.
        """
        self.session = session
        self.settings = settings or get_settings()
        self.cancel_event = cancel_event
        self.on_progress = on_progress
        self.stats = DuplicateStats()

    # -- entry point --------------------------------------------------------

    def detect(
        self,
        *,
        min_file_bytes: int = DEFAULT_MIN_FILE_BYTES,
        include_near_duplicates: bool = True,
    ) -> DuplicateStats:
        """Run a full duplicate pass and persist the results.

        Args:
            min_file_bytes: Ignore files smaller than this.
            include_near_duplicates: Also report assets that overlap heavily without
                being identical.

        Returns:
            Statistics describing the pass.
        """
        self.stats = DuplicateStats()
        self._clear_previous_groups()

        candidates = self._size_collision_candidates(min_file_bytes)
        if not candidates:
            logger.info("No files share a size; nothing can be duplicated")
            self.session.commit()
            return self.stats

        by_quick = self._tier1_quick_signatures(candidates)
        by_content = self._tier2_full_hashes(by_quick)

        self._record_file_groups(by_content)
        self._record_asset_groups(include_near_duplicates=include_near_duplicates)

        self.session.commit()
        logger.info("Duplicate pass complete: %s", self.stats.summary())
        return self.stats

    # -- tier 0/1 -----------------------------------------------------------

    def _size_collision_candidates(self, min_file_bytes: int) -> list[AssetFile]:
        """Return files whose byte size is shared with at least one other file.

        The cheapest possible filter, and the one that does most of the work: a file
        whose size is unique in the catalogue cannot be a duplicate of anything, and is
        eliminated without reading a byte.
        """
        shared_sizes = select(AssetFile.size_bytes).where(
            AssetFile.size_bytes >= min_file_bytes
        ).group_by(AssetFile.size_bytes).having(func.count() > 1)

        rows = self.session.scalars(
            select(AssetFile)
            .join(Asset, Asset.id == AssetFile.asset_id)
            .where(
                AssetFile.size_bytes >= min_file_bytes,
                AssetFile.size_bytes.in_(shared_sizes),
                Asset.is_missing.is_(False),
            )
        ).all()

        self.stats.files_considered = self.session.scalar(
            select(func.count()).select_from(AssetFile).where(
                AssetFile.size_bytes >= min_file_bytes
            )
        ) or 0
        self.stats.size_collision_candidates = len(rows)
        logger.debug(
            "%d of %d file(s) share a size and need hashing",
            len(rows), self.stats.files_considered,
        )
        return list(rows)

    def _tier1_quick_signatures(
        self, candidates: Sequence[AssetFile]
    ) -> dict[str, list[AssetFile]]:
        """Compute or reuse quick signatures, grouping files that share one."""
        groups: dict[str, list[AssetFile]] = defaultdict(list)
        total = len(candidates)

        for index, record in enumerate(candidates):
            self._check_cancelled()
            self._report("quick-hash", index, total)

            signature = record.quick_sig
            if signature is None:
                signature = self._compute_quick(record)
                if signature is None:
                    continue
                record.quick_sig = signature
                self.stats.quick_hashed += 1

            groups[signature].append(record)

        # Flush the newly computed signatures so a cancelled or failed later phase does
        # not throw away I/O that has already been paid for.
        self.session.commit()

        return {
            signature: files for signature, files in groups.items() if len(files) > 1
        }

    def _compute_quick(self, record: AssetFile) -> str | None:
        """Compute one quick signature, tolerating unreadable files."""
        path = self._absolute_path(record)
        if path is None:
            return None
        try:
            self.stats.bytes_read += min(
                record.size_bytes, self.settings.quick_hash_chunk_bytes * 2
            )
            return quick_signature(
                path,
                size=record.size_bytes,
                chunk_bytes=self.settings.quick_hash_chunk_bytes,
                min_full_hash_bytes=self.settings.quick_hash_min_file_bytes,
                cancel_event=self.cancel_event,
            )
        except HashCancelled:
            raise
        except OSError as exc:
            logger.debug("Cannot hash %s: %s", path, exc)
            self.stats.errors += 1
            return None

    # -- tier 2 -------------------------------------------------------------

    def _tier2_full_hashes(
        self, by_quick: dict[str, list[AssetFile]]
    ) -> dict[str, list[AssetFile]]:
        """Resolve quick-signature collisions with full SHA-256 digests."""
        groups: dict[str, list[AssetFile]] = defaultdict(list)
        pending = [record for files in by_quick.values() for record in files]
        total = len(pending)

        for index, record in enumerate(pending):
            self._check_cancelled()
            self._report("sha256", index, total)

            digest = record.sha256
            if digest is None:
                digest = self._compute_sha256(record)
                if digest is None:
                    continue
                record.sha256 = digest
                self.stats.fully_hashed += 1

            groups[digest].append(record)

        self.session.commit()
        return {digest: files for digest, files in groups.items() if len(files) > 1}

    def _compute_sha256(self, record: AssetFile) -> str | None:
        """Compute one full digest, tolerating unreadable files."""
        path = self._absolute_path(record)
        if path is None:
            return None

        limit = self.settings.full_hash_max_bytes
        if limit and record.size_bytes > limit:
            logger.debug("Skipping full hash of %s: above the configured limit", path)
            return None

        try:
            digest = sha256_file(path, cancel_event=self.cancel_event)
            self.stats.bytes_read += record.size_bytes
            return digest
        except HashCancelled:
            raise
        except OSError as exc:
            logger.debug("Cannot hash %s: %s", path, exc)
            self.stats.errors += 1
            return None

    # -- grouping -----------------------------------------------------------

    def _record_file_groups(self, by_content: dict[str, list[AssetFile]]) -> None:
        """Persist file-level duplicate groups."""
        for digest, files in by_content.items():
            distinct, shared = self._split_by_physical_identity(files)
            unit_size = files[0].size_bytes

            # Only extents that occupy their own storage can be reclaimed, and one copy
            # is always kept, so the waste is every distinct extent beyond the first.
            wasted = unit_size * max(0, len(distinct) - 1)
            self.stats.wasted_bytes += wasted
            self.stats.shared_storage_bytes += unit_size * len(shared)

            keeper = self._choose_keeper(files)
            group = DuplicateGroup(
                group_hash=digest,
                kind=DuplicateKind.FILE.value,
                member_count=len(files),
                unit_size_bytes=unit_size,
                wasted_bytes=wasted,
                similarity=1.0,
                keeper_asset_id=keeper.asset_id if keeper else None,
            )
            self.session.add(group)
            self.session.flush()

            distinct_ids = {id(record) for record in distinct}
            for record in files:
                self.session.add(
                    DuplicateMember(
                        group_id=group.id,
                        asset_id=record.asset_id,
                        file_id=record.id,
                        path=self._absolute_path(record) or record.relpath,
                        size_bytes=record.size_bytes,
                        occupies_own_storage=id(record) in distinct_ids,
                        is_keeper=record is keeper,
                    )
                )
            self.stats.file_groups += 1

    def _split_by_physical_identity(
        self, files: Sequence[AssetFile]
    ) -> tuple[list[AssetFile], list[AssetFile]]:
        """Split files into distinct physical extents and shared aliases.

        Two catalogue rows pointing at the same ``(device, inode)`` are one file on disk.
        Only the first is counted as occupying storage; the rest are aliases whose
        deletion frees nothing.
        """
        distinct: list[AssetFile] = []
        shared: list[AssetFile] = []
        seen: set[tuple[int, int]] = set()

        for record in files:
            if record.is_symlink:
                shared.append(record)
                continue
            identity = (
                (record.device, record.inode)
                if record.inode and record.device is not None
                else None
            )
            if identity is not None and record.nlink > 1:
                if identity in seen:
                    shared.append(record)
                    continue
                seen.add(identity)
            distinct.append(record)

        return distinct, shared

    def _choose_keeper(self, files: Sequence[AssetFile]) -> AssetFile | None:
        r"""Pick the copy to keep.

        Newest wins, then the shortest path — which in practice favours a deliberate
        ``D:\\Models\\llama`` over a stray ``D:\\Downloads\\tmp\\llama (1)``.
        """
        if not files:
            return None
        return max(
            files,
            key=lambda record: (
                record.modified_at.timestamp() if record.modified_at else 0.0,
                -len(record.relpath),
            ),
        )

    def _record_asset_groups(self, *, include_near_duplicates: bool) -> None:
        """Group whole assets whose payload content matches."""
        signatures = self._asset_signatures()
        if not signatures:
            return

        exact: dict[str, list[int]] = defaultdict(list)
        for asset_id, (combined, _hashes) in signatures.items():
            exact[combined].append(asset_id)

        grouped_assets: set[int] = set()
        for combined, asset_ids in exact.items():
            if len(asset_ids) < 2:
                continue
            self._persist_asset_group(
                combined, asset_ids, similarity=1.0, kind=DuplicateKind.ASSET
            )
            grouped_assets.update(asset_ids)
            self.stats.asset_groups += 1

        if include_near_duplicates:
            self._record_near_duplicates(signatures, grouped_assets)

    def _asset_signatures(self) -> dict[int, tuple[str, frozenset[str]]]:
        """Return each asset's content signature.

        Built in two steps. Payload files come first, since they are already hashed and
        eliminate almost every asset from consideration. Assets whose payloads match are
        then compared on their *defining* small files as well.

        That second step is not optional. Two models can share byte-identical weights and
        still be different models — a re-uploaded checkpoint with a corrected
        ``config.json``, a base model and a variant with a different rope scaling. Judging
        on weights alone would tell the user to delete one of them.

        Documentation is excluded throughout: a copy that differs only by a README is the
        same download.
        """
        rows = self.session.execute(
            select(AssetFile.asset_id, AssetFile.sha256)
            .join(Asset, Asset.id == AssetFile.asset_id)
            .where(
                AssetFile.sha256.is_not(None),
                AssetFile.is_payload.is_(True),
                Asset.is_missing.is_(False),
            )
        ).all()

        by_asset: dict[int, set[str]] = defaultdict(set)
        for asset_id, digest in rows:
            by_asset[asset_id].add(digest)

        payload_only = {
            asset_id: (combine_hashes(sorted(hashes)), frozenset(hashes))
            for asset_id, hashes in by_asset.items()
            if hashes
        }
        return self._refine_by_defining_files(payload_only)

    def _refine_by_defining_files(
        self, payload_signatures: dict[int, tuple[str, frozenset[str]]]
    ) -> dict[int, tuple[str, frozenset[str]]]:
        """Fold defining configuration into the signatures of colliding assets.

        Only assets whose payload signature already collides are examined, so the extra
        hashing is proportional to real candidates rather than to catalogue size. The
        files involved are kilobytes each.
        """
        collisions: dict[str, list[int]] = defaultdict(list)
        for asset_id, (combined, _hashes) in payload_signatures.items():
            collisions[combined].append(asset_id)

        refined = dict(payload_signatures)
        for asset_ids in collisions.values():
            if len(asset_ids) < 2:
                continue
            for asset_id in asset_ids:
                extra = self._defining_file_hashes(asset_id)
                if not extra:
                    continue
                combined, hashes = refined[asset_id]
                merged = hashes | extra
                refined[asset_id] = (combine_hashes(sorted(merged)), frozenset(merged))

        return refined

    def _defining_file_hashes(self, asset_id: int) -> set[str]:
        """Hash an asset's small configuration files.

        These are never hashed during the tiered pass — they fall below the size floor —
        so they are read here, on demand, for the handful of assets that need them.
        """
        records = self.session.scalars(
            select(AssetFile).where(
                AssetFile.asset_id == asset_id,
                AssetFile.is_payload.is_(False),
            )
        ).all()

        hashes: set[str] = set()
        for record in records:
            if _is_documentation(record.relpath) or record.size_bytes > MAX_DEFINING_FILE_BYTES:
                continue

            digest = record.sha256
            if digest is None:
                digest = self._compute_sha256(record)
                if digest is None:
                    continue
                record.sha256 = digest
                self.stats.fully_hashed += 1

            # Mixing the relative path in means a config moved to a different name is a
            # different asset, which is what "defining" has to mean here.
            hashes.add(combine_hashes([record.relpath, digest]))

        return hashes

    def _record_near_duplicates(
        self,
        signatures: dict[int, tuple[str, frozenset[str]]],
        already_grouped: set[int],
    ) -> None:
        """Report assets that overlap heavily without matching exactly.

        Comparison is pairwise, which is quadratic, so it is restricted to assets that
        share at least one payload hash. Assets with nothing in common cannot clear the
        similarity threshold and never need comparing.
        """
        by_hash: dict[str, list[int]] = defaultdict(list)
        for asset_id, (_combined, hashes) in signatures.items():
            if asset_id in already_grouped:
                continue
            for digest in hashes:
                by_hash[digest].append(asset_id)

        checked: set[tuple[int, int]] = set()
        for sharing in by_hash.values():
            if len(sharing) < 2:
                continue
            for left in sharing:
                for right in sharing:
                    if left >= right:
                        continue
                    pair = (left, right)
                    if pair in checked:
                        continue
                    checked.add(pair)

                    left_hashes = signatures[left][1]
                    right_hashes = signatures[right][1]
                    union = left_hashes | right_hashes
                    if not union:
                        continue
                    similarity = len(left_hashes & right_hashes) / len(union)
                    if similarity < NEAR_DUPLICATE_THRESHOLD or similarity >= 1.0:
                        continue

                    self._persist_asset_group(
                        combine_hashes(sorted(left_hashes & right_hashes)),
                        [left, right],
                        similarity=similarity,
                        kind=DuplicateKind.NEAR_ASSET,
                    )
                    self.stats.near_duplicate_groups += 1

    def _persist_asset_group(
        self,
        group_hash: str,
        asset_ids: Sequence[int],
        *,
        similarity: float,
        kind: DuplicateKind,
    ) -> None:
        """Persist one asset-level duplicate group."""
        assets = self.session.scalars(select(Asset).where(Asset.id.in_(asset_ids))).all()
        if len(assets) < 2:
            return

        keeper = max(
            assets,
            key=lambda asset: (
                asset.modified_at.timestamp() if asset.modified_at else 0.0,
                asset.size_bytes,
            ),
        )
        unit_size = max(asset.physical_size_bytes or asset.size_bytes for asset in assets)
        wasted = sum(
            asset.physical_size_bytes or asset.size_bytes
            for asset in assets
            if asset.id != keeper.id
        )

        group = DuplicateGroup(
            group_hash=group_hash,
            kind=kind.value,
            member_count=len(assets),
            unit_size_bytes=unit_size,
            wasted_bytes=wasted,
            similarity=round(similarity, 4),
            keeper_asset_id=keeper.id,
        )
        self.session.add(group)
        self.session.flush()

        for asset in assets:
            self.session.add(
                DuplicateMember(
                    group_id=group.id,
                    asset_id=asset.id,
                    path=asset.root_path,
                    size_bytes=asset.physical_size_bytes or asset.size_bytes,
                    occupies_own_storage=True,
                    is_keeper=asset.id == keeper.id,
                )
            )

    # -- helpers ------------------------------------------------------------

    def _clear_previous_groups(self) -> None:
        """Remove groups from an earlier pass.

        Members cascade, and hashes live on ``files`` rather than here, so nothing that
        cost I/O is discarded.
        """
        self.session.execute(delete(DuplicateMember))
        self.session.execute(delete(DuplicateGroup))
        self.session.flush()

    def _absolute_path(self, record: AssetFile) -> str | None:
        """Reconstruct a file's absolute path from its asset root."""
        asset = self.session.get(Asset, record.asset_id)
        if asset is None:
            return None
        if asset.is_single_file:
            return asset.root_path
        return os.path.normpath(os.path.join(asset.root_path, record.relpath))

    def _check_cancelled(self) -> None:
        """Raise if cancellation has been requested."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise HashCancelled("Duplicate detection cancelled")

    def _report(self, phase: str, completed: int, total: int) -> None:
        """Emit a progress update."""
        if self.on_progress is not None and (completed % 25 == 0 or completed == total - 1):
            self.on_progress(phase, completed, total)


@dataclass(slots=True)
class DuplicateSummary:
    """A duplicate group prepared for display."""

    group: DuplicateGroup
    members: list[DuplicateMember] = field(default_factory=list)

    @property
    def reclaimable(self) -> int:
        """Return the bytes that would be freed by keeping one copy."""
        return self.group.wasted_bytes


def list_duplicate_groups(
    session: Session,
    *,
    kind: str | None = None,
    min_wasted: int = 0,
    limit: int = 100,
) -> list[DuplicateSummary]:
    """Return persisted duplicate groups, largest waste first."""
    statement = select(DuplicateGroup).where(DuplicateGroup.wasted_bytes >= min_wasted)
    if kind:
        statement = statement.where(DuplicateGroup.kind == kind)

    groups = session.scalars(
        statement.order_by(DuplicateGroup.wasted_bytes.desc()).limit(limit)
    ).all()
    return [DuplicateSummary(group=group, members=list(group.members)) for group in groups]


def total_wasted_bytes(session: Session) -> int:
    """Return the total reclaimable space across all duplicate groups.

    File-level groups are excluded from the total when an asset-level group already
    covers the same bytes; counting both would double the figure shown to the user.
    """
    asset_waste = session.scalar(
        select(func.coalesce(func.sum(DuplicateGroup.wasted_bytes), 0)).where(
            DuplicateGroup.kind.in_([DuplicateKind.ASSET.value, DuplicateKind.NEAR_ASSET.value])
        )
    ) or 0

    grouped_assets = set(
        session.scalars(
            select(DuplicateMember.asset_id)
            .join(DuplicateGroup, DuplicateGroup.id == DuplicateMember.group_id)
            .where(
                DuplicateGroup.kind.in_(
                    [DuplicateKind.ASSET.value, DuplicateKind.NEAR_ASSET.value]
                )
            )
        ).all()
    )

    file_waste = 0
    file_groups = session.scalars(
        select(DuplicateGroup).where(DuplicateGroup.kind == DuplicateKind.FILE.value)
    ).all()
    for group in file_groups:
        if any(member.asset_id in grouped_assets for member in group.members):
            continue
        file_waste += group.wasted_bytes

    return int(asset_waste) + int(file_waste)
