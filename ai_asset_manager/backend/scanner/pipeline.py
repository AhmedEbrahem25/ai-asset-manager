"""The scan pipeline: walk, detect, parse, merge.

Produces :class:`AssetRecord` objects and stops there. Persistence is a separate concern
handled by the service layer, which keeps this module free of any database session and
therefore safe to run entirely on worker threads.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from ai_asset_manager.backend.detectors.base import DetectionResult
from ai_asset_manager.backend.detectors.registry import DetectorRegistry
from ai_asset_manager.backend.metadata.merge import build_asset_record
from ai_asset_manager.backend.metadata.records import AssetRecord
from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.parsers.base import FactSet, MetadataParser
from ai_asset_manager.backend.parsers.ollama import (
    OllamaParser,
    find_ollama_root,
    parse_manifest,
)
from ai_asset_manager.backend.parsers.registry import ParserRegistry
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.progress import ScanContext, ScanPhase
from ai_asset_manager.backend.scanner.types import DirectoryTree, DirNode, FileEntry
from ai_asset_manager.backend.scanner.walker import walk_tree
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Called with an asset's root path to decide whether it needs re-parsing. Returning the
#: stored fingerprint lets the pipeline skip unchanged assets without a database import.
FingerprintLookup = Callable[[str], str | None]


class ScanPipeline:
    """Turns a directory tree into asset records."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        detectors: DetectorRegistry | None = None,
        parsers: ParserRegistry | None = None,
    ) -> None:
        """Initialise the pipeline with its registries.

        Registries are injected rather than constructed inline so tests can supply a
        single detector and callers can extend the set without subclassing.
        """
        self.settings = settings or get_settings()
        self.detectors = detectors or DetectorRegistry()
        self.parsers = parsers or ParserRegistry()

    def scan_root(
        self,
        root: str | os.PathLike[str],
        *,
        context: ScanContext | None = None,
        fingerprint_lookup: FingerprintLookup | None = None,
    ) -> list[AssetRecord]:
        """Walk one root and return a record for every asset found.

        Args:
            root: Directory to scan.
            context: Progress and cancellation context.
            fingerprint_lookup: Returns the previously stored fingerprint for a root
                path, if any. When the fingerprint still matches, detection and parsing
                are skipped for that asset — the optimisation that makes rescans fast.

        Returns:
            One record per asset. Unchanged assets are represented by a record carrying
            only the fingerprint and root path, flagged via ``evidence["unchanged"]``.
        """
        tree = walk_tree(root, settings=self.settings, context=context)
        return self.process_tree(
            tree, context=context, fingerprint_lookup=fingerprint_lookup
        )

    def process_tree(
        self,
        tree: DirectoryTree,
        *,
        context: ScanContext | None = None,
        fingerprint_lookup: FingerprintLookup | None = None,
    ) -> list[AssetRecord]:
        """Detect and parse every asset in an already-walked tree."""
        if context is not None:
            context.set_phase(ScanPhase.DETECTING, message=f"Classifying {tree.root}")

        detections = self.detectors.detect_tree(tree)

        if context is not None:
            context.set_phase(
                ScanPhase.PARSING, total=len(detections), message="Reading metadata"
            )
            context.increment(assets_found=len(detections))

        records: list[AssetRecord] = []
        for index, detection in enumerate(detections):
            if context is not None:
                context.raise_if_cancelled()
                context.update(completed=index, current_path=detection.root_path)

            try:
                record = self.build_record(
                    tree, detection, fingerprint_lookup=fingerprint_lookup
                )
            except Exception as exc:
                logger.warning("Failed to build record for %s: %s", detection.root_path, exc)
                if context is not None:
                    context.increment(errors=1)
                continue

            if record is not None:
                records.append(record)

        return records

    def build_record(
        self,
        tree: DirectoryTree,
        detection: DetectionResult,
        *,
        fingerprint_lookup: FingerprintLookup | None = None,
    ) -> AssetRecord | None:
        """Gather an asset's files, parse its metadata and merge the result.

        Returns:
            The completed record, or ``None`` when the asset has no files at all — which
            happens when a detector matched on structure that has since been deleted.
        """
        files = self._collect_files(tree, detection)
        if not files:
            logger.debug("No files found for detected asset at %s", detection.root_path)
            return None

        if fingerprint_lookup is not None:
            record = self._try_skip_unchanged(detection, files, fingerprint_lookup)
            if record is not None:
                return record

        facts = self._parse(tree, detection, files)
        return build_asset_record(detection, facts, files)

    def _collect_files(
        self, tree: DirectoryTree, detection: DetectionResult
    ) -> list[FileEntry]:
        """Return the files belonging to a detected asset.

        Three cases, in order of specificity: an explicit file list (an Ollama model,
        whose blobs live in a directory shared with every other model), a single file
        (a standalone ``.gguf``), or the whole subtree beneath the asset's file root.
        """
        if detection.explicit_files is not None:
            wanted = set(detection.explicit_files)
            root_dir = os.path.dirname(detection.root_path)
            found = [
                entry
                for entry in tree.iter_subtree_files(
                    find_ollama_root(root_dir) or root_dir
                )
                if entry.path in wanted
            ]
            # For an Ollama model the identifying path is the manifest, which lives
            # outside the blob directory and so is not in the explicit list. Guard
            # against re-adding it when the asset *is* the listed file, as it is for
            # loose weights — double-counting it would double the reported size.
            if detection.root_path not in wanted:
                manifest = self._entry_for_path(tree, detection.root_path)
                if manifest is not None:
                    found.append(manifest)
            return found

        if detection.is_single_file:
            entry = self._entry_for_path(tree, detection.root_path)
            return [entry] if entry is not None else []

        return tree.iter_subtree_files(detection.file_root)

    def _scoped_node(self, node: DirNode, files: list[FileEntry]) -> DirNode:
        """Return a copy of a directory node exposing only the given files.

        Sidecar configuration that sits beside a standalone weight file is kept, since a
        lone ``config.json`` in a folder of GGUF files genuinely describes all of them,
        while the weight files themselves are restricted to this asset's own.
        """
        own_paths = {entry.path for entry in files}
        sidecars = [
            entry
            for entry in node.files
            if entry.path not in own_paths and not entry.is_payload
        ]
        return DirNode(
            path=node.path,
            depth=node.depth,
            parent=node.parent,
            files=[*files, *sidecars],
            child_dirs=list(node.child_dirs),
        )

    def _entry_for_path(self, tree: DirectoryTree, path: str) -> FileEntry | None:
        """Find the walked entry for a single file path."""
        node = tree.get(os.path.dirname(path))
        if node is None:
            return None
        for entry in node.files:
            if entry.path == path:
                return entry
        return None

    def _try_skip_unchanged(
        self,
        detection: DetectionResult,
        files: list[FileEntry],
        lookup: FingerprintLookup,
    ) -> AssetRecord | None:
        """Return a stub record when the asset is unchanged since the last scan.

        The fingerprint covers every file's relative path, size and modification time, so
        an unchanged value means nothing about the asset can have changed in a way any
        parser would notice.
        """
        from ai_asset_manager.backend.utils.hashing import fingerprint_entries
        from ai_asset_manager.backend.utils.paths import safe_relpath

        current = fingerprint_entries(
            (safe_relpath(entry.path, detection.root_path), entry.size, entry.mtime)
            for entry in files
        )
        if lookup(detection.root_path) != current:
            return None

        return AssetRecord(
            kind=detection.kind,
            name=detection.name,
            root_path=detection.root_path,
            fingerprint=current,
            file_count=len(files),
            detector=detection.detector,
            evidence={"unchanged": True},
        )

    def _parse(
        self,
        tree: DirectoryTree,
        detection: DetectionResult,
        files: list[FileEntry] | None = None,
    ) -> FactSet:
        """Run the parser registry against an asset's content."""
        extra: list[MetadataParser] = []

        content_dir = (
            detection.file_root
            if not detection.is_single_file
            else os.path.dirname(detection.root_path)
        )
        node = tree.get(content_dir)
        if node is None:
            node = tree.get(detection.root_path)
        if node is None:
            return FactSet()

        if detection.is_single_file and files:
            # A directory holding twenty unrelated GGUF files holds twenty models. Parsing
            # the directory would hand every one of them the union of all twenty headers,
            # so each single-file asset is parsed against a view containing only its own
            # files.
            ctx = DirectoryContext(tree, self._scoped_node(node, files))
        else:
            ctx = DirectoryContext(tree, node)

        # An Ollama model is described by its manifest, not by anything in the blob
        # directory it happens to share with every other model in the store.
        if detection.detector == "ollama_store":
            ollama_root = find_ollama_root(os.path.dirname(detection.root_path))
            if ollama_root:
                extra.append(
                    OllamaParser(
                        parse_manifest(
                            detection.root_path,
                            os.path.join(ollama_root, "blobs"),
                            manifests_root=os.path.join(ollama_root, "manifests"),
                        )
                    )
                )

        facts = self.parsers.collect(ctx, extra=extra)

        # A cache directory's own name is authoritative for identity, but the snapshot
        # holds the configs. Both must be parsed, and the cache facts must not be lost.
        if detection.content_root and detection.content_root != detection.root_path:
            root_node = tree.get(detection.root_path)
            if root_node is not None:
                facts.merge_from(
                    self.parsers.collect(DirectoryContext(tree, root_node))
                )

        return facts


def summarise(records: list[AssetRecord]) -> dict[str, int]:
    """Return per-kind counts for a set of records, for logging and CLI output."""
    counts: dict[str, int] = {}
    for record in records:
        key = record.kind.value if isinstance(record.kind, AssetKind) else str(record.kind)
        counts[key] = counts.get(key, 0) + 1
    return counts
