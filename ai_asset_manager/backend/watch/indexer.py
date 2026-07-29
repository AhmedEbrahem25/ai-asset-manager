"""Turning changed paths into the smallest scan that will make the catalogue correct.

The whole value of live indexing is in this decision. Rescanning everything on every change
is correct and useless; rescanning nothing is fast and wrong. What is wanted is the
narrowest subtree that certainly contains the change.

Two cases, and they are genuinely different:

*The path is inside an asset we already catalogued.* Then the asset root is the answer.
Rescanning it re-fingerprints one directory and touches nothing else.

*The path is not inside any known asset.* Then something new has appeared, and there is no
way to know where its boundary is — a new HuggingFace repo is three directories below the
cache root, and its manifest and its weights live in different subtrees. So the managed
root is rescanned instead. That sounds expensive and is not: an incremental scan skips any
asset whose fingerprint is unchanged, so the cost is one walk plus the parse of whatever
actually changed.

Deletions need no special case. A targeted rescan of a vanished asset finds nothing and
marks it missing, which is exactly right — and it marks rather than deletes, so unplugging
a drive does not destroy its catalogue.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import Asset
from ai_asset_manager.backend.services.scan_service import ScanService
from ai_asset_manager.backend.utils.paths import normalize_path
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Returns a context-managed session. ``session_scope`` satisfies it; so does a
#: test's factory, which is what lets the watcher be driven without a real database.
SessionFactory = Callable[[], AbstractContextManager[Session]]

#: Above this many distinct targets in one batch, rescan the managed roots instead. A
#: change touching hundreds of assets at once is a bulk operation - an unpack, a move, a
#: sync - and one root walk beats hundreds of directory scans.
MAX_TARGETS_PER_BATCH = 24


@dataclass(slots=True)
class IndexResult:
    """What one batch of changes did to the catalogue."""

    targets: list[str] = field(default_factory=list)
    assets_created: int = 0
    assets_updated: int = 0
    assets_unchanged: int = 0
    assets_missing: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        """Return how many assets were actually created, updated or lost."""
        return self.assets_created + self.assets_updated + self.assets_missing

    def describe(self) -> str:
        """Return a one-line summary for a log or a status display."""
        if not self.targets:
            return "nothing to do"
        parts = []
        if self.assets_created:
            parts.append(f"{self.assets_created} new")
        if self.assets_updated:
            parts.append(f"{self.assets_updated} updated")
        if self.assets_missing:
            parts.append(f"{self.assets_missing} gone")
        summary = ", ".join(parts) or "no change"
        return f"{summary} across {len(self.targets)} location(s) in {self.duration_seconds:.1f}s"


@dataclass(slots=True)
class IndexerStats:
    """Running totals for the life of a watcher."""

    batches: int = 0
    paths_seen: int = 0
    assets_created: int = 0
    assets_updated: int = 0
    assets_missing: int = 0
    errors: int = 0
    last_result: IndexResult | None = None
    last_run_at: float | None = None


class LiveIndexer:
    """Applies filesystem changes to the catalogue, one batch at a time."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        settings: Settings | None = None,
    ) -> None:
        """Create an indexer.

        Args:
            session_factory: Returns a context-managed session. Each batch gets its own,
                so a failure cannot poison the next one.
            settings: Configuration; the process settings when omitted.
        """
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self.stats = IndexerStats()

    def handle(self, paths: set[str]) -> IndexResult:
        """Bring the catalogue up to date with a batch of changed paths."""
        started = time.monotonic()
        result = IndexResult()

        if not paths:
            return result

        with self._session_factory() as session:
            roots = [root.path for root in ScanService(session).list_roots(enabled_only=True)]
            if not roots:
                logger.debug("Ignoring %d change(s): no managed roots", len(paths))
                return result

            targets = self.resolve_targets(session, paths, roots)
            if not targets:
                return result

            result.targets = targets
            service = ScanService(session, settings=self._settings)

            for target in targets:
                try:
                    run = service.scan(
                        [target],
                        incremental=True,
                        # A targeted scan only sees one subtree, so pruning is scoped to
                        # it: a deleted asset is marked missing, and nothing outside the
                        # target is touched.
                        prune_missing=True,
                    )
                except Exception as exc:
                    logger.exception("Live index of %s failed", target)
                    result.errors.append(f"{target}: {exc}")
                    continue

                result.assets_created += run.assets_created
                result.assets_updated += run.assets_updated
                result.assets_unchanged += run.assets_unchanged
                result.assets_missing += run.assets_missing

        result.duration_seconds = round(time.monotonic() - started, 3)
        self._record(paths, result)
        return result

    def resolve_targets(
        self, session: Session, paths: set[str], roots: Sequence[str]
    ) -> list[str]:
        """Return the smallest set of directories that covers every changed path."""
        catalogued = self._asset_roots(session)
        targets: set[str] = set()
        needs_root_scan: set[str] = set()

        for raw in paths:
            path = normalize_path(raw)
            owning_root = _containing(path, roots)
            if owning_root is None:
                # Outside every managed root. Watchdog can deliver these when a root is a
                # symlink or when a parent directory is renamed; there is nothing useful
                # to do with them.
                continue

            asset_root = _containing(path, catalogued)
            if asset_root is not None:
                targets.add(asset_root)
            else:
                needs_root_scan.add(owning_root)

        # A root-level scan subsumes every asset beneath it, so drop the redundant ones.
        targets = {
            target
            for target in targets
            if _containing(target, list(needs_root_scan)) is None
        }
        targets |= needs_root_scan

        if len(targets) > MAX_TARGETS_PER_BATCH:
            logger.debug(
                "%d targets in one batch; falling back to %d root scan(s)",
                len(targets), len(roots),
            )
            return sorted(set(roots))

        return sorted(targets)

    def _asset_roots(self, session: Session) -> list[str]:
        """Return every catalogued asset root, longest first.

        Longest first because assets nest: a LoRA can live inside a model directory, and
        the innermost match is the one that owns a changed file.
        """
        rows = session.scalars(select(Asset.root_path)).all()
        return sorted(rows, key=len, reverse=True)

    def _record(self, paths: set[str], result: IndexResult) -> None:
        """Fold a batch into the running totals."""
        self.stats.batches += 1
        self.stats.paths_seen += len(paths)
        self.stats.assets_created += result.assets_created
        self.stats.assets_updated += result.assets_updated
        self.stats.assets_missing += result.assets_missing
        self.stats.errors += len(result.errors)
        self.stats.last_result = result
        self.stats.last_run_at = time.time()

        if result.changed:
            logger.info("Live index: %s", result.describe())
        else:
            logger.debug("Live index: %s", result.describe())


def _containing(path: str, candidates: Sequence[str]) -> str | None:
    r"""Return the longest candidate that contains ``path``, or ``None``.

    Compared case-insensitively on Windows and with a separator guard, so that
    ``D:\\Models2`` is never treated as living inside ``D:\\Models``.
    """
    import os

    normalised = path.replace("\\", "/")
    if os.name == "nt":
        normalised = normalised.lower()

    best: str | None = None
    for candidate in candidates:
        prepared = candidate.replace("\\", "/")
        if os.name == "nt":
            prepared = prepared.lower()

        if normalised == prepared or normalised.startswith(prepared.rstrip("/") + "/"):
            if best is None or len(candidate) > len(best):
                best = candidate
    return best
