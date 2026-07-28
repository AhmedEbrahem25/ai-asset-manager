"""HuggingFace cache decoder.

The hub cache stores repositories under a flattened, unreadable name::

    models--meta-llama--Meta-Llama-3-8B-Instruct/
      refs/main                      -> commit sha
      blobs/<sha>                    -> content, addressed by hash
      snapshots/<commit>/config.json -> the readable tree

This module recovers the original repository id, resolves the snapshot that holds the
real content, and works out how the snapshot relates to the blob store.

That last part is not cosmetic. The same cache appears three different ways depending on
platform and privileges:

- **symlinks** — the usual POSIX layout; snapshot entries point into ``blobs/``.
- **hardlinks** — snapshot and blob are one file with two names.
- **plain copies** — Windows without developer mode, where ``blobs/`` may be empty and
  the snapshot holds the only copy. This is what is on the machine this was built on.

Counting a symlinked or hardlinked pair twice would double every size in the catalogue
and invent duplicates that free nothing when deleted.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum

from ai_asset_manager.backend.models.enums import AssetKind, FactSource
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Prefixes the hub uses, mapped to the kind of asset they hold.
REPO_TYPE_PREFIXES: dict[str, AssetKind] = {
    "models": AssetKind.MODEL,
    "datasets": AssetKind.DATASET,
    "spaces": AssetKind.UNKNOWN,
}

#: ``<type>--<org>--<name>``. The org segment is optional for canonical repos such as
#: ``models--gpt2``. Repository names may themselves contain ``--``.
CACHE_DIR_RE = re.compile(r"^(?P<type>models|datasets|spaces)--(?P<rest>.+)$")

#: A 40-character git commit sha, the snapshot directory naming convention.
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class LinkLayout(StrEnum):
    """How a cache's snapshot relates to its blob store."""

    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class CacheRepoInfo:
    """A decoded HuggingFace cache directory."""

    repo_id: str
    kind: AssetKind
    author: str | None
    repo_name: str
    #: Absolute path of the snapshot holding the content, when one was found.
    snapshot_path: str | None = None
    revision: str | None = None
    layout: LinkLayout = LinkLayout.UNKNOWN
    #: Superseded revisions still on disk, as ``(revision, bytes)`` pairs. Pure waste
    #: unless the layout links snapshots to shared blobs.
    stale_snapshots: tuple[tuple[str, int], ...] = ()

    @property
    def stale_bytes(self) -> int:
        """Return the total size of superseded revisions."""
        return sum(size for _, size in self.stale_snapshots)


def decode_cache_dir_name(name: str) -> tuple[AssetKind, str, str | None, str] | None:
    """Decode a hub cache directory name into a repository identifier.

    Args:
        name: Directory name such as ``models--meta-llama--Meta-Llama-3-8B-Instruct``.

    Returns:
        ``(kind, repo_id, author, repo_name)``, or ``None`` if the name is not a cache
        directory.

    Examples:
        >>> decode_cache_dir_name("models--Qwen--Qwen2.5-0.5B-Instruct")[1]
        'Qwen/Qwen2.5-0.5B-Instruct'
        >>> decode_cache_dir_name("datasets--openai--gsm8k")[1]
        'openai/gsm8k'
        >>> decode_cache_dir_name("models--gpt2")[1]
        'gpt2'
        >>> decode_cache_dir_name("not-a-cache-dir") is None
        True
    """
    match = CACHE_DIR_RE.match(name)
    if not match:
        return None

    kind = REPO_TYPE_PREFIXES.get(match.group("type"), AssetKind.UNKNOWN)
    rest = match.group("rest")

    # Only the first separator splits owner from name: repository names legitimately
    # contain "--" (e.g. "some-org--my--model"), so splitting on all of them would
    # mangle them.
    author, separator, repo_name = rest.partition("--")
    if not separator:
        return kind, rest, None, rest
    return kind, f"{author}/{repo_name}", author, repo_name


def is_cache_repo_dir(name: str) -> bool:
    """Report whether a directory name looks like a hub cache repository."""
    return CACHE_DIR_RE.match(name) is not None


def find_snapshot(ctx: DirectoryContext) -> tuple[str | None, str | None]:
    """Locate the snapshot directory holding a cache repository's content.

    Prefers the commit named by ``refs/main``. Falls back to the most recently modified
    snapshot, which matters when a repo was fetched by revision and has no ``main`` ref.

    Returns:
        ``(snapshot_path, revision)``, either of which may be ``None``.
    """
    snapshots = ctx.child("snapshots")
    if snapshots is None:
        return None, None

    candidates = {
        os.path.basename(path): path
        for path in snapshots.node.child_dirs
    }
    if not candidates:
        return None, None

    refs = ctx.child("refs")
    if refs is not None:
        for ref_name in ("main", "master"):
            ref_text = refs.read_text(ref_name)
            if ref_text:
                revision = ref_text.strip()
                if revision in candidates:
                    return candidates[revision], revision

    # No usable ref: take the newest snapshot directory.
    newest_name, newest_path = max(
        candidates.items(),
        key=lambda item: _safe_mtime(item[1]),
    )
    newest_revision = newest_name if COMMIT_SHA_RE.match(newest_name) else None
    return newest_path, newest_revision


def _safe_mtime(path: str) -> float:
    """Return a path's modification time, or ``0.0`` if it cannot be read."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def detect_link_layout(ctx: DirectoryContext, snapshot_path: str | None) -> LinkLayout:
    """Determine how a cache's snapshot files relate to its blob store.

    Inspects the largest snapshot file, since that is the one whose double-counting would
    matter, and only that one: stat-ing every file of every repo to answer a question
    that is uniform within a cache would be wasted I/O.
    """
    if snapshot_path is None:
        return LinkLayout.UNKNOWN

    snapshot_node = ctx.tree.get(snapshot_path)
    if snapshot_node is None:
        return LinkLayout.UNKNOWN

    files = ctx.tree.iter_subtree_files(snapshot_path)
    if not files:
        return LinkLayout.UNKNOWN

    largest = max(files, key=lambda entry: entry.size)
    if largest.is_symlink:
        return LinkLayout.SYMLINK
    if largest.nlink > 1:
        return LinkLayout.HARDLINK
    return LinkLayout.COPY


def find_stale_snapshots(
    ctx: DirectoryContext, active_snapshot: str | None
) -> tuple[tuple[str, int], ...]:
    """Return superseded snapshot revisions still occupying disk.

    Pulling a repository twice at different revisions leaves both on disk; the hub never
    garbage-collects the old one. On a linked cache the two share their blobs and cost
    almost nothing, but on a copy-based cache — Windows without developer mode — each
    revision is a full second copy. Distinguishing the two is why the layout is checked
    rather than assumed.

    Returns:
        ``(revision, bytes)`` pairs for every snapshot that is not the active one.
    """
    snapshots = ctx.child("snapshots")
    if snapshots is None or active_snapshot is None:
        return ()

    stale: list[tuple[str, int]] = []
    for path in snapshots.node.child_dirs:
        if path == active_snapshot:
            continue
        size = sum(entry.size for entry in ctx.tree.iter_subtree_files(path))
        stale.append((os.path.basename(path), size))

    return tuple(sorted(stale, key=lambda item: -item[1]))


def inspect_cache_repo(ctx: DirectoryContext) -> CacheRepoInfo | None:
    """Decode a cache repository directory in full.

    Args:
        ctx: Context positioned on the ``models--*`` or ``datasets--*`` directory.

    Returns:
        The decoded repository, or ``None`` if this is not a cache directory.
    """
    decoded = decode_cache_dir_name(ctx.name)
    if decoded is None:
        return None

    kind, repo_id, author, repo_name = decoded
    snapshot_path, revision = find_snapshot(ctx)

    return CacheRepoInfo(
        repo_id=repo_id,
        kind=kind,
        author=author,
        repo_name=repo_name,
        snapshot_path=snapshot_path,
        revision=revision,
        layout=detect_link_layout(ctx, snapshot_path),
        stale_snapshots=find_stale_snapshots(ctx, snapshot_path),
    )


class HfCacheParser(BaseParser):
    """Recovers repository identity from a hub cache directory."""

    name = "hf_cache"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether this directory is a hub cache repository."""
        return is_cache_repo_dir(ctx.name)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract the repository id, author, revision and storage layout.

        Facts are attributed to the directory name because that is literally their
        source, which keeps them below anything a config file inside the snapshot says.
        The one exception is ``repo_id``: the cache path is authoritative for it, whereas
        ``config.json``'s ``_name_or_path`` is frequently a stale local path from
        whichever machine trained the model.
        """
        facts = self._new_facts()
        info = inspect_cache_repo(ctx)
        if info is None:
            return facts

        facts.add("repo_id", info.repo_id, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("name", info.repo_name, source=FactSource.DIRECTORY_NAME, confidence=0.9,
                  origin=self.name)
        facts.add("display_name", info.repo_id, source=FactSource.DIRECTORY_NAME,
                  origin=self.name)
        facts.add("author", info.author, source=FactSource.DIRECTORY_NAME, origin=self.name)
        facts.add("kind", info.kind.value, source=FactSource.DIRECTORY_NAME, origin=self.name)
        facts.add("revision", info.revision, source=FactSource.DIRECTORY_NAME, origin=self.name)
        facts.add("is_hf_cache", True, source=FactSource.DIRECTORY_NAME, origin=self.name)
        facts.add("cache_layout", info.layout.value, source=FactSource.DIRECTORY_NAME,
                  origin=self.name)

        if info.snapshot_path:
            facts.add("content_root", info.snapshot_path, source=FactSource.DIRECTORY_NAME,
                      origin=self.name)
        else:
            facts.warn("cache directory has no readable snapshot")

        if info.stale_snapshots:
            facts.add("stale_revisions", [rev for rev, _ in info.stale_snapshots],
                      source=FactSource.DIRECTORY_NAME, origin=self.name)
            facts.add("stale_bytes", info.stale_bytes, source=FactSource.DIRECTORY_NAME,
                      origin=self.name)
            # Only real waste when each revision holds its own copy of the weights. On a
            # linked cache the revisions share blobs and deleting one frees nothing.
            if info.layout is not LinkLayout.SYMLINK:
                facts.warn(
                    f"{len(info.stale_snapshots)} superseded revision(s) still on disk "
                    f"({info.stale_bytes / 1024 ** 3:.2f} GiB reclaimable): "
                    + ", ".join(rev[:8] for rev, _ in info.stale_snapshots)
                )

        return facts
