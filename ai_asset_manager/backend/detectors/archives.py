r"""Archive detector.

Sits in the same priority band as the loose-weights rule, and for the same reason: an
archive is a *file* that happens to be an asset, not a directory that happens to contain
one. Both bands emit at once, so a folder holding ``qwen.gguf`` beside ``coco.zip`` yields
two assets rather than whichever detector happened to rank higher.

The cost model is what constrains everything here. Detection runs once per directory across
a whole-machine scan, and an archive is the only asset type whose examination requires
opening a file. Three limits keep that honest:

* archives below :data:`MIN_ARCHIVE_BYTES` are ignored — a 30 KB zip is an attachment;
* at most :data:`MAX_ARCHIVES_PER_DIR` are opened in any one directory, so a backup folder
  with four hundred of them costs a bounded amount rather than a proportional one;
* the reader itself caps entries and decompressed bytes.

Nothing is ever extracted. See :mod:`ai_asset_manager.backend.archives.reader`.
"""

from __future__ import annotations

from ai_asset_manager.backend.archives import (
    ArchiveListing,
    classify_listing,
    inspect_archive,
    is_archive_name,
)
from ai_asset_manager.backend.detectors.base import (
    PRIORITY_ARCHIVE,
    BaseDetector,
    DetectionResult,
)
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import FileEntry
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Archives smaller than this are not catalogued. Source tarballs, mailed spreadsheets and
#: browser downloads live below this line; models and datasets do not.
MIN_ARCHIVE_BYTES = 1024 * 1024

#: How many archives are opened in one directory. Beyond this the rest are skipped rather
#: than the scan being allowed to grow with the size of somebody's backup folder.
MAX_ARCHIVES_PER_DIR = 25

#: Multi-volume markers. Only the first part is opened; the rest belong to the same asset
#: and opening them individually would both fail and waste the I/O.
_CONTINUATION_MARKERS: tuple[str, ...] = (
    ".z0", ".z1", ".z2", ".z3", ".z4", ".z5", ".z6", ".z7", ".z8", ".z9",
    ".r0", ".r1", ".r2", ".r3", ".r4", ".r5", ".r6", ".r7", ".r8", ".r9",
)


class ArchiveDetector(BaseDetector):
    """Catalogues archive files from their table of contents."""

    name = "archive"
    priority = PRIORITY_ARCHIVE

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one asset per archive file in this directory."""
        candidates = self._candidates(ctx)
        if not candidates:
            return []

        results: list[DetectionResult] = []
        for entry in candidates[:MAX_ARCHIVES_PER_DIR]:
            listing = inspect_archive(entry.path, entry.size)
            results.append(self._result_for(entry, listing))

        skipped = len(candidates) - len(results)
        if skipped > 0:
            logger.debug(
                "Listed %d of %d archive(s) in %s; the rest are above the per-directory cap",
                len(results), len(candidates), ctx.path,
            )
        return results

    def _candidates(self, ctx: DirectoryContext) -> list[FileEntry]:
        """Return the archives in this directory worth opening, largest first.

        Largest first because the cap is a budget: if only twenty-five archives in a
        directory can be opened, they should be the twenty-five that account for the space.
        """
        found = [
            entry
            for entry in ctx.files
            if entry.size >= MIN_ARCHIVE_BYTES
            and is_archive_name(entry.name)
            and not self._is_continuation(entry.name)
        ]
        found.sort(key=lambda entry: -entry.size)
        return found

    def _is_continuation(self, name: str) -> bool:
        """Report whether a filename is a later volume of a split archive."""
        lowered = name.lower()
        return (
            lowered.endswith(_CONTINUATION_MARKERS)
            or (".part" in lowered and not lowered.endswith((".zip", ".rar", ".7z")))
        )

    def _result_for(self, entry: FileEntry, listing: ArchiveListing) -> DetectionResult:
        """Turn one listing into a detection result."""
        verdict = classify_listing(listing)

        evidence: dict[str, object] = {
            "archive": True,
            "archive_format": listing.format,
            "archive_label": verdict.label,
            "members_listed": len(listing.file_entries),
            "extracted": False,
            "signals": verdict.signals,
        }
        evidence.update(verdict.extra)
        if listing.metadata:
            # Recorded so it is visible that these were read *in memory* and which ones.
            evidence["metadata_read"] = sorted(listing.metadata)

        return DetectionResult(
            kind=verdict.kind,
            name=self._display_name(entry.name, listing),
            root_path=entry.path,
            detector=self.name,
            subkind=verdict.subkind,
            is_single_file=True,
            format=verdict.format,
            framework=verdict.framework,
            confidence=verdict.confidence,
            evidence=evidence,
            explicit_files=[entry.path],
            claims_subtree=False,
        )

    def _display_name(self, filename: str, listing: ArchiveListing) -> str:
        """Return the name to catalogue an archive under.

        The archive's own filename, minus its compression suffix. An archive whose members
        all sit under one folder is named for that folder instead when the filename is
        uninformative — ``download.zip`` holding ``UNSW-NB15/`` is the UNSW-NB15 dataset,
        and cataloguing it as "download" helps nobody.
        """
        stem = filename
        lowered = stem.lower()
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"):
            if lowered.endswith(suffix):
                return stem[: -len(suffix)]

        stem = stem.rsplit(".", 1)[0] if "." in stem else stem
        if stem.lower() in _UNINFORMATIVE_NAMES:
            roots = {name for name in listing.top_level if name and "." not in name}
            if len(roots) == 1:
                return next(iter(roots))
        return stem


#: Filenames that say nothing about what was downloaded.
_UNINFORMATIVE_NAMES: frozenset[str] = frozenset(
    {"archive", "download", "data", "dataset", "file", "files", "new folder", "tmp", "temp"}
)
