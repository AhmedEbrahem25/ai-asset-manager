r"""The same model, installed by several applications.

Distinct from the duplicate detector next door, and the difference is what it costs. That
one answers "are these bytes identical?" and pays for the answer in disk reads. This one
answers "is this the same model shipped six times?" and pays nothing at all: it works
entirely from rows the scanner already wrote.

The observation that makes it possible is that applications do not repackage the models
they embed. Chrome, Edge, VS Code and Cursor all ship Chromium's optimisation-guide model,
and every copy has the same byte size, the same file count and the same file names, because
every copy came from the same build. Those three facts are already in the catalogue, and
together they are a signature strong enough to group on.

Where SHA-256 digests exist — the duplicate pass computes them, and they persist — they are
used instead and the group is reported as *verified*. So the answer improves for free after
a duplicate run without needing a different command or a different table.

Nothing here deletes, moves or suggests a shell command. It reports what is installed where
and what one copy weighs; what to do about six copies of a model that Chrome will silently
re-download is the user's call.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.identity import identify
from ai_asset_manager.backend.models import Asset, AssetFile
from ai_asset_manager.backend.utils.hashing import combine_hashes
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Installations smaller than this are not worth a row. A 40 KB model shipped nine times
#: is a curiosity; the report is about space.
DEFAULT_MIN_UNIT_BYTES = 256 * 1024

#: How many copies before it is worth calling an installation group.
MIN_INSTALLS = 2

#: Cap on file names folded into a metadata signature. Two assets agreeing on their size,
#: their file count and their first sixty-four file names are the same download.
MAX_SIGNATURE_NAMES = 64


@dataclass(slots=True, frozen=True)
class Installation:
    """One copy of a model, and what put it there."""

    asset_id: int
    name: str
    path: str
    size_bytes: int
    #: Short source identifier from the identity layer: ``"chrome"``, ``"vscode"``, …
    source: str
    #: Product name, when one was determined: ``"Chrome"``, ``"VS Code"``.
    product: str | None = None
    vendor: str | None = None


@dataclass(slots=True)
class InstallationGroup:
    """One model found installed in several places."""

    #: Stable identifier for the group, derived from whatever evidence was used.
    key: str
    installs: list[Installation] = field(default_factory=list)
    #: Size of a single copy.
    unit_size_bytes: int = 0
    #: True when the grouping was confirmed by content hashes rather than metadata.
    verified_by_hash: bool = False

    @property
    def install_count(self) -> int:
        """Return how many copies were found."""
        return len(self.installs)

    @property
    def reclaimable_bytes(self) -> int:
        """Return the bytes that would be freed by keeping one copy.

        Every copy but one. This is an upper bound and is reported as such: an application
        that finds its bundled model missing will usually fetch it again, so the space is
        recoverable rather than reclaimable in any permanent sense.
        """
        return self.unit_size_bytes * max(0, self.install_count - 1)

    @property
    def sources(self) -> list[str]:
        """Return the distinct applications holding a copy, in display order."""
        seen: dict[str, None] = {}
        for install in self.installs:
            seen.setdefault(install.product or install.source, None)
        return sorted(seen)

    @property
    def spans_applications(self) -> bool:
        """Report whether the copies belong to different applications.

        The interesting case. Six copies inside one application's own cache are that
        application's business; six copies across six applications are a fact about the
        machine.
        """
        return len(self.sources) > 1

    @property
    def display_name(self) -> str:
        """Return the most informative name any copy carries."""
        named = [install.name for install in self.installs if install.name]
        if not named:
            return "(unnamed)"
        # The longest name is the one that survived renaming and says the most.
        return max(named, key=len)


def find_duplicate_installations(
    session: Session,
    *,
    min_unit_bytes: int = DEFAULT_MIN_UNIT_BYTES,
    across_applications_only: bool = False,
) -> list[InstallationGroup]:
    """Return models present in more than one place, largest reclaim first.

    Args:
        session: Open database session.
        min_unit_bytes: Ignore models whose single copy is smaller than this.
        across_applications_only: Report only groups whose copies belong to different
            applications, dropping the case of one program keeping several copies of its
            own model.

    Returns:
        One group per model found more than once. Reads only the catalogue; no file is
        opened and no hash is computed.
    """
    assets = list(
        session.scalars(
            select(Asset).where(
                Asset.is_missing.is_(False),
                Asset.size_bytes >= min_unit_bytes,
            )
        ).all()
    )
    if len(assets) < MIN_INSTALLS:
        return []

    file_rows = _file_rows(session, [asset.id for asset in assets])

    grouped: dict[tuple[bool, str], list[Asset]] = defaultdict(list)
    for asset in assets:
        signature, verified = _signature(asset, file_rows.get(asset.id, []))
        if signature is None:
            continue
        grouped[(verified, signature)].append(asset)

    groups: list[InstallationGroup] = []
    for (verified, signature), members in grouped.items():
        if len(members) < MIN_INSTALLS:
            continue

        group = InstallationGroup(
            key=signature,
            installs=[_installation(asset) for asset in members],
            unit_size_bytes=max(
                asset.physical_size_bytes or asset.size_bytes for asset in members
            ),
            verified_by_hash=verified,
        )
        if across_applications_only and not group.spans_applications:
            continue
        groups.append(group)

    groups.sort(key=lambda item: (-item.reclaimable_bytes, -item.install_count))
    logger.info(
        "Found %d duplicated installation(s) across the catalogue", len(groups)
    )
    return groups


def total_reclaimable(groups: list[InstallationGroup]) -> int:
    """Return the summed upper bound on recoverable space."""
    return sum(group.reclaimable_bytes for group in groups)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _file_rows(
    session: Session, asset_ids: list[int]
) -> dict[int, list[tuple[str, int, str | None]]]:
    """Return each asset's ``(relpath, size, sha256)`` triples.

    Fetched in chunks because SQLite binds at most 999 parameters per statement, and a
    catalogue with thousands of assets would otherwise fail on the first query.
    """
    collected: dict[int, list[tuple[str, int, str | None]]] = defaultdict(list)
    chunk = 500

    for start in range(0, len(asset_ids), chunk):
        rows = session.execute(
            select(
                AssetFile.asset_id,
                AssetFile.relpath,
                AssetFile.size_bytes,
                AssetFile.sha256,
            ).where(AssetFile.asset_id.in_(asset_ids[start : start + chunk]))
        ).all()
        for asset_id, relpath, size_bytes, sha256 in rows:
            collected[asset_id].append((relpath or "", size_bytes or 0, sha256))

    return collected


def _signature(
    asset: Asset, files: list[tuple[str, int, str | None]]
) -> tuple[str | None, bool]:
    """Return an asset's identity signature and whether hashes backed it.

    Prefers content digests when the duplicate pass has computed them, because two files
    of the same size are not necessarily the same file. Falls back to the shape of the
    asset — total bytes, file count, and the names and sizes within — which is what makes
    this usable straight after a scan with no hashing at all.
    """
    if not files:
        return None, False

    digests = sorted(digest for _relpath, _size, digest in files if digest)
    if digests and len(digests) == len(files):
        return combine_hashes(digests), True

    shape = sorted(
        (_basename(relpath), size) for relpath, size, _digest in files
    )[:MAX_SIGNATURE_NAMES]
    parts = [
        str(asset.size_bytes),
        str(len(files)),
        str(asset.format),
        *(f"{name}:{size}" for name, size in shape),
    ]
    return _digest(parts), False


def _digest(parts: list[str]) -> str:
    """Hash an ordered list of strings into a stable key.

    Not :func:`~ai_asset_manager.backend.utils.hashing.combine_hashes`, which encodes as
    ASCII because it combines hex digests. These parts are filenames, and a model shipped
    with a non-ASCII filename must not crash the report.
    """
    combined = hashlib.blake2b(digest_size=16)
    for part in parts:
        combined.update(part.encode("utf-8", errors="replace"))
        combined.update(b"\x00")
    return combined.hexdigest()


def _basename(relpath: str) -> str:
    """Return a relative path's filename, lower-cased and separator-agnostic."""
    return relpath.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _installation(asset: Asset) -> Installation:
    """Describe one copy, including which application it belongs to.

    The identity is read from the asset's stored evidence when the scanner recorded it and
    re-derived from the path otherwise, so a catalogue written before the identity layer
    existed still reports sources rather than blanks.
    """
    stored = asset.evidence.get("identity") if asset.evidence else None
    if isinstance(stored, dict):
        source = str(stored.get("source") or "unknown")
        product = stored.get("product")
        vendor = stored.get("vendor")
    else:
        derived = identify(
            asset.root_path, name=asset.name, is_single_file=asset.is_single_file
        )
        source, product, vendor = derived.source, derived.product, derived.vendor

    return Installation(
        asset_id=asset.id,
        name=asset.display_name or asset.name,
        path=asset.root_path,
        size_bytes=asset.physical_size_bytes or asset.size_bytes,
        source=source,
        product=str(product) if product else None,
        vendor=str(vendor) if vendor else None,
    )
