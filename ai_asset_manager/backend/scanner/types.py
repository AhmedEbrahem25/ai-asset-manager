"""Value types produced by the filesystem walk.

These use ``slots=True`` deliberately: a scan of a large library holds hundreds of
thousands of :class:`FileEntry` objects in memory at once, and slots cut their footprint
by roughly half compared with ``__dict__``-backed instances.
"""

from __future__ import annotations

import os
import stat as stat_module
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ai_asset_manager.backend.utils.paths import classify_extension, is_payload_file


@dataclass(slots=True, frozen=True)
class FileEntry:
    """A single file observed during the walk.

    Captured from one ``stat`` call so that later phases never need to touch the
    filesystem again for size or timestamp information.
    """

    name: str
    path: str
    size: int
    mtime: float
    ctime: float
    atime: float
    inode: int
    device: int
    nlink: int
    is_symlink: bool

    @property
    def extension(self) -> str:
        """Return the lower-cased file extension, including the leading dot."""
        return Path(self.name).suffix.lower()

    @property
    def stem(self) -> str:
        """Return the filename without its final extension."""
        return Path(self.name).stem

    @property
    def is_payload(self) -> bool:
        """Report whether this file holds weights or dataset content."""
        return is_payload_file(self.name)

    @property
    def content_class(self) -> str | None:
        """Return the coarse content class, e.g. ``"image"`` or ``"payload"``."""
        return classify_extension(self.name)

    @property
    def modified_at(self) -> datetime:
        """Return the modification time as an aware UTC datetime."""
        return datetime.fromtimestamp(self.mtime, tz=UTC)

    @property
    def identity(self) -> tuple[int, int] | None:
        """Return the ``(device, inode)`` pair, or ``None`` if unavailable.

        Used to recognise that two paths are the same physical file before any content
        is hashed.
        """
        if self.inode == 0 and self.device == 0:
            return None
        return (self.device, self.inode)

    @classmethod
    def from_stat(cls, name: str, path: str, st: os.stat_result, *, is_symlink: bool) -> FileEntry:
        """Build an entry from a ``stat`` result."""
        return cls(
            name=name,
            path=path,
            size=st.st_size,
            mtime=st.st_mtime,
            ctime=st.st_ctime,
            atime=st.st_atime,
            inode=st.st_ino,
            device=st.st_dev,
            nlink=st.st_nlink,
            is_symlink=is_symlink,
        )


@dataclass(slots=True)
class DirNode:
    """One directory in the walked tree, with its immediate files and children."""

    path: str
    depth: int
    parent: str | None
    files: list[FileEntry] = field(default_factory=list)
    child_dirs: list[str] = field(default_factory=list)
    #: Set when the directory could not be fully read (permissions, disconnected drive).
    error: str | None = None

    @property
    def name(self) -> str:
        """Return the directory's own name."""
        return os.path.basename(self.path.rstrip("\\/")) or self.path

    @property
    def file_names(self) -> set[str]:
        """Return the set of immediate filenames, for cheap membership tests."""
        return {entry.name for entry in self.files}

    @property
    def child_names(self) -> set[str]:
        """Return the set of immediate subdirectory names."""
        return {os.path.basename(child) for child in self.child_dirs}


@dataclass(slots=True)
class DirectoryTree:
    """The complete result of a walk: every directory reached, keyed by path."""

    root: str
    nodes: dict[str, DirNode] = field(default_factory=dict)
    total_files: int = 0
    total_bytes: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def get(self, path: str) -> DirNode | None:
        """Return the node for a path, if it was walked."""
        return self.nodes.get(path)

    def iter_descending(self, start: str | None = None) -> list[DirNode]:
        """Return nodes ordered parent-before-child.

        Detection relies on this order: a directory claimed as an asset must be visited
        before its children so that the children can be suppressed.
        """
        origin = start or self.root
        ordered: list[DirNode] = []
        stack = [origin]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            ordered.append(node)
            stack.extend(reversed(node.child_dirs))
        return ordered

    def iter_subtree_files(self, path: str) -> list[FileEntry]:
        """Return every file at or beneath ``path``.

        This is how an asset's total size, file count and fingerprint are computed once
        a detector has claimed a directory.
        """
        collected: list[FileEntry] = []
        stack = [path]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            collected.extend(node.files)
            stack.extend(node.child_dirs)
        return collected

    def subtree_paths(self, path: str) -> list[str]:
        """Return every directory path at or beneath ``path``, including itself."""
        collected: list[str] = []
        stack = [path]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            collected.append(current)
            stack.extend(node.child_dirs)
        return collected


def is_regular_file(st: os.stat_result) -> bool:
    """Report whether a ``stat`` result describes a regular file.

    Filters out FIFOs, sockets and device nodes, which appear on POSIX systems and would
    otherwise be hashed or sized as if they were real files.
    """
    return stat_module.S_ISREG(st.st_mode)
