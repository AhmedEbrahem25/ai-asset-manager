"""The view of a directory that detectors and parsers are given.

A single object wrapping "this directory, its files, and everything beneath it", with
cached accessors. Detectors ask dozens of cheap structural questions per directory, so
every answer that costs a syscall or a file read is memoised here rather than in each of
the forty-odd detectors.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from fnmatch import fnmatch
from functools import cached_property
from typing import Any

from ai_asset_manager.backend.scanner.types import DirectoryTree, DirNode, FileEntry
from ai_asset_manager.backend.utils.paths import classify_extension
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Upper bound on text/JSON sidecars read into memory. Real configs are kilobytes;
#: anything larger is either not what we think it is or not worth parsing eagerly.
MAX_SIDECAR_BYTES = 8 * 1024 * 1024


class DirectoryContext:
    """Read-only view of one directory within a walked tree."""

    __slots__ = ("__dict__", "_json_cache", "_text_cache", "node", "tree")

    def __init__(self, tree: DirectoryTree, node: DirNode) -> None:
        """Wrap a node from a walked tree.

        Args:
            tree: The tree the node belongs to, used for subtree queries.
            node: The directory being examined.
        """
        self.tree = tree
        self.node = node
        self._json_cache: dict[str, Any | None] = {}
        self._text_cache: dict[str, str | None] = {}

    # -- identity -----------------------------------------------------------

    @property
    def path(self) -> str:
        """Return the absolute path of this directory."""
        return self.node.path

    @property
    def name(self) -> str:
        """Return this directory's own name."""
        return self.node.name

    @property
    def depth(self) -> int:
        """Return the directory's depth below the scan root."""
        return self.node.depth

    @property
    def parent_name(self) -> str:
        """Return the parent directory's name, or an empty string at the root."""
        return os.path.basename(self.node.parent) if self.node.parent else ""

    # -- immediate contents -------------------------------------------------

    @property
    def files(self) -> list[FileEntry]:
        """Return files directly inside this directory."""
        return self.node.files

    @cached_property
    def file_names(self) -> set[str]:
        """Return immediate filenames."""
        return {entry.name for entry in self.node.files}

    @cached_property
    def lower_file_names(self) -> set[str]:
        """Return immediate filenames, lower-cased, for case-insensitive membership."""
        return {name.lower() for name in self.file_names}

    @cached_property
    def child_dir_names(self) -> set[str]:
        """Return immediate subdirectory names."""
        return {os.path.basename(path) for path in self.node.child_dirs}

    @cached_property
    def lower_child_dir_names(self) -> set[str]:
        """Return immediate subdirectory names, lower-cased."""
        return {name.lower() for name in self.child_dir_names}

    @cached_property
    def extensions(self) -> set[str]:
        """Return the set of extensions present directly in this directory."""
        return {entry.extension for entry in self.node.files if entry.extension}

    # -- subtree ------------------------------------------------------------

    @cached_property
    def subtree_files(self) -> list[FileEntry]:
        """Return every file at or beneath this directory."""
        return self.tree.iter_subtree_files(self.node.path)

    @cached_property
    def subtree_dir_count(self) -> int:
        """Return the number of directories at or beneath this one."""
        return len(self.tree.subtree_paths(self.node.path))

    @cached_property
    def total_size(self) -> int:
        """Return the summed apparent size of every file in the subtree."""
        return sum(entry.size for entry in self.subtree_files)

    @cached_property
    def total_file_count(self) -> int:
        """Return the number of files in the subtree."""
        return len(self.subtree_files)

    @cached_property
    def subtree_extension_counts(self) -> dict[str, int]:
        """Return a histogram of file extensions across the subtree.

        The workhorse behind most dataset detectors, which classify on the *shape* of a
        tree — thousands of ``.jpg`` beside thousands of ``.txt`` — rather than on names.
        """
        counts: dict[str, int] = {}
        for entry in self.subtree_files:
            ext = entry.extension
            if ext:
                counts[ext] = counts.get(ext, 0) + 1
        return counts

    @cached_property
    def content_class_counts(self) -> dict[str, int]:
        """Return a histogram of coarse content classes across the subtree."""
        counts: dict[str, int] = {}
        for entry in self.subtree_files:
            klass = classify_extension(entry.name)
            if klass:
                counts[klass] = counts.get(klass, 0) + 1
        return counts

    def count_extension(self, *extensions: str) -> int:
        """Return how many subtree files carry any of the given extensions."""
        counts = self.subtree_extension_counts
        return sum(counts.get(ext.lower(), 0) for ext in extensions)

    @property
    def image_count(self) -> int:
        """Return the number of image files in the subtree."""
        return self.content_class_counts.get("image", 0)

    @property
    def video_count(self) -> int:
        """Return the number of video files in the subtree."""
        return self.content_class_counts.get("video", 0)

    @property
    def audio_count(self) -> int:
        """Return the number of audio files in the subtree."""
        return self.content_class_counts.get("audio", 0)

    # -- queries ------------------------------------------------------------

    def has(self, *names: str) -> bool:
        """Report whether *every* named file is present directly in this directory."""
        return all(name.lower() in self.lower_file_names for name in names)

    def has_any(self, *names: str) -> bool:
        """Report whether *any* named file is present directly in this directory."""
        return any(name.lower() in self.lower_file_names for name in names)

    def has_dir(self, *names: str) -> bool:
        """Report whether every named subdirectory is present."""
        return all(name.lower() in self.lower_child_dir_names for name in names)

    def has_any_dir(self, *names: str) -> bool:
        """Report whether any named subdirectory is present."""
        return any(name.lower() in self.lower_child_dir_names for name in names)

    def glob(self, pattern: str) -> list[FileEntry]:
        """Return immediate files matching a glob pattern, case-insensitively."""
        lowered = pattern.lower()
        return [entry for entry in self.node.files if fnmatch(entry.name.lower(), lowered)]

    def glob_subtree(self, pattern: str, *, limit: int = 0) -> list[FileEntry]:
        """Return subtree files matching a glob pattern.

        Args:
            pattern: Glob applied to the filename only.
            limit: Stop after this many matches; ``0`` means no limit. Detectors that
                only need to know "does one of these exist" should pass ``limit=1``.
        """
        lowered = pattern.lower()
        matches: list[FileEntry] = []
        for entry in self.subtree_files:
            if fnmatch(entry.name.lower(), lowered):
                matches.append(entry)
                if limit and len(matches) >= limit:
                    break
        return matches

    def first(self, pattern: str) -> FileEntry | None:
        """Return the first immediate file matching a glob, or ``None``."""
        matches = self.glob(pattern)
        return matches[0] if matches else None

    def entry(self, name: str) -> FileEntry | None:
        """Return a named immediate file, matched case-insensitively."""
        target = name.lower()
        for item in self.node.files:
            if item.name.lower() == target:
                return item
        return None

    def child(self, name: str) -> DirectoryContext | None:
        """Return a context for a named immediate subdirectory."""
        target = name.lower()
        for child_path in self.node.child_dirs:
            if os.path.basename(child_path).lower() == target:
                child_node = self.tree.get(child_path)
                if child_node is not None:
                    return DirectoryContext(self.tree, child_node)
        return None

    def children(self) -> Iterator[DirectoryContext]:
        """Yield contexts for every immediate subdirectory."""
        for child_path in self.node.child_dirs:
            child_node = self.tree.get(child_path)
            if child_node is not None:
                yield DirectoryContext(self.tree, child_node)

    # -- reads --------------------------------------------------------------

    def read_text(self, name: str, *, max_bytes: int = MAX_SIDECAR_BYTES) -> str | None:
        """Read a small text file from this directory, cached.

        Returns:
            The decoded text, or ``None`` if the file is absent, too large, or unreadable.
            Decoding is lenient: a model card with a stray byte should not break a scan.
        """
        if name in self._text_cache:
            return self._text_cache[name]

        result: str | None = None
        target = self.entry(name)
        if target is not None and target.size <= max_bytes:
            try:
                with open(target.path, encoding="utf-8", errors="replace") as handle:
                    result = handle.read()
            except OSError as exc:
                logger.debug("Cannot read %s: %s", target.path, exc)

        self._text_cache[name] = result
        return result

    def read_json(self, name: str, *, max_bytes: int = MAX_SIDECAR_BYTES) -> Any | None:
        """Read and parse a JSON file from this directory, cached.

        Returns:
            The parsed value, or ``None`` when the file is missing or malformed. Half-
            downloaded configs are common, so a parse failure is expected rather than
            exceptional and is reported by the health checker instead.
        """
        if name in self._json_cache:
            return self._json_cache[name]

        result: Any | None = None
        text = self.read_text(name, max_bytes=max_bytes)
        if text is not None:
            try:
                result = json.loads(text)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug("Malformed JSON in %s/%s: %s", self.path, name, exc)

        self._json_cache[name] = result
        return result

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<DirectoryContext {self.path!r} files={len(self.node.files)}>"
