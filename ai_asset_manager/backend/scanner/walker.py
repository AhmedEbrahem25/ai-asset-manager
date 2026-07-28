"""Parallel filesystem walk.

One pass over the tree collecting nothing but ``stat`` metadata: no file is opened and
nothing is hashed here. Everything later phases need about sizes and timestamps comes
from this single traversal.

Threads rather than processes: the work is dominated by syscall latency, ``os.scandir``
releases the GIL, and a process pool would have to pickle every entry back to the parent.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from ai_asset_manager.backend.scanner.progress import ScanCancelled, ScanContext, ScanPhase
from ai_asset_manager.backend.scanner.types import DirectoryTree, DirNode, FileEntry
from ai_asset_manager.backend.utils.paths import (
    is_excluded_dir,
    matches_any_glob,
    normalize_path,
)
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Cap on recorded errors. The full set still goes to the log; this only bounds memory
#: when scanning a drive where thousands of directories are unreadable.
MAX_RECORDED_ERRORS = 200


class WalkError(RuntimeError):
    """Raised when a scan root itself cannot be walked."""


def _scan_directory(
    path: str,
    depth: int,
    parent: str | None,
    settings: Settings,
) -> DirNode:
    """Read a single directory. Runs on a worker thread.

    Errors are captured on the node rather than raised: one unreadable directory must not
    abort a scan of an entire drive.
    """
    node = DirNode(path=path, depth=depth, parent=parent)
    try:
        with os.scandir(path) as iterator:
            for entry in iterator:
                try:
                    is_symlink = entry.is_symlink()
                    if entry.is_dir(follow_symlinks=settings.follow_symlinks):
                        if is_excluded_dir(entry.name, settings.excluded_dirs):
                            continue
                        if is_symlink and not settings.follow_symlinks:
                            continue
                        node.child_dirs.append(entry.path)
                        continue

                    if not entry.is_file(follow_symlinks=False) and not is_symlink:
                        continue
                    if matches_any_glob(entry.name, settings.excluded_file_globs):
                        continue

                    # follow_symlinks=False describes the link itself, which is what we
                    # want: its target is counted where the target actually lives.
                    st = entry.stat(follow_symlinks=False)
                    node.files.append(
                        FileEntry.from_stat(entry.name, entry.path, st, is_symlink=is_symlink)
                    )
                except OSError as exc:
                    logger.debug("Skipping entry %s: %s", entry.path, exc)
    except (OSError, ValueError) as exc:
        node.error = str(exc)
        logger.debug("Cannot read directory %s: %s", path, exc)
    return node


def walk_tree(
    root: str | os.PathLike[str],
    *,
    settings: Settings | None = None,
    context: ScanContext | None = None,
) -> DirectoryTree:
    """Walk ``root`` and return every directory reached.

    Args:
        root: Directory to walk. Must exist and be a directory.
        settings: Scan configuration; defaults to the singleton.
        context: Progress and cancellation context.

    Returns:
        A :class:`DirectoryTree` holding one node per directory visited.

    Raises:
        WalkError: If ``root`` does not exist or is not a directory.
        ScanCancelled: If cancellation is requested mid-walk.
    """
    settings = settings or get_settings()
    root_path = normalize_path(root)

    if not os.path.exists(root_path):
        raise WalkError(f"Scan root does not exist: {root_path}")
    if not os.path.isdir(root_path):
        raise WalkError(f"Scan root is not a directory: {root_path}")

    tree = DirectoryTree(root=root_path)
    if context is not None:
        context.set_phase(ScanPhase.WALKING, message=f"Walking {root_path}")

    # Directory identities already visited. Guards against symlink loops when
    # follow_symlinks is on, and against junction points on Windows that would
    # otherwise send the walk around the same subtree forever.
    seen_identities: set[tuple[int, int]] = set()

    def should_descend(path: str) -> bool:
        """Return whether a directory is new, recording its identity if so."""
        if not settings.follow_symlinks:
            return True
        try:
            st = os.stat(path)
        except OSError:
            return False
        if st.st_ino == 0 and st.st_dev == 0:
            return True
        identity = (st.st_dev, st.st_ino)
        if identity in seen_identities:
            logger.debug("Already visited %s; skipping to avoid a cycle", path)
            return False
        seen_identities.add(identity)
        return True

    with ThreadPoolExecutor(
        max_workers=settings.scan_workers, thread_name_prefix="aam-walk"
    ) as pool:
        pending: dict[Future[DirNode], str] = {
            pool.submit(_scan_directory, root_path, 0, None, settings): root_path
        }

        try:
            while pending:
                if context is not None and context.cancelled:
                    raise ScanCancelled

                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future, None)
                    node = future.result()
                    tree.nodes[node.path] = node

                    if node.error and len(tree.errors) < MAX_RECORDED_ERRORS:
                        tree.errors.append((node.path, node.error))

                    node_bytes = sum(entry.size for entry in node.files)
                    tree.total_files += len(node.files)
                    tree.total_bytes += node_bytes

                    if context is not None:
                        context.increment(
                            directories_seen=1,
                            files_seen=len(node.files),
                            bytes_seen=node_bytes,
                            errors=1 if node.error else 0,
                        )
                        context.update(current_path=node.path)

                    if node.depth >= settings.max_depth:
                        # Record the children as known but do not descend: a runaway
                        # tree (a recursive junction, a mounted backup) must terminate.
                        if node.child_dirs:
                            logger.debug(
                                "Depth limit %d reached at %s; not descending",
                                settings.max_depth,
                                node.path,
                            )
                        node.child_dirs.clear()
                        continue

                    for child in node.child_dirs:
                        if not should_descend(child):
                            continue
                        pending[
                            pool.submit(_scan_directory, child, node.depth + 1, node.path, settings)
                        ] = child
        except ScanCancelled:
            for future in pending:
                future.cancel()
            raise

    # Children that were pruned (depth limit, cycle) must not remain as dangling
    # references, or tree traversal would report directories that were never walked.
    for node in tree.nodes.values():
        node.child_dirs = [child for child in node.child_dirs if child in tree.nodes]

    logger.info(
        "Walked %s: %d directories, %d files, %d error(s)",
        root_path,
        len(tree.nodes),
        tree.total_files,
        len(tree.errors),
    )
    return tree


def walk_roots(
    roots: list[str] | tuple[str, ...],
    *,
    settings: Settings | None = None,
    context: ScanContext | None = None,
) -> list[DirectoryTree]:
    """Walk several roots, isolating failures.

    A root that cannot be reached — an unplugged external drive, a revoked network share
    — is logged and skipped so the remaining roots still produce a catalogue.
    """
    settings = settings or get_settings()
    trees: list[DirectoryTree] = []
    for root in roots:
        try:
            trees.append(walk_tree(root, settings=settings, context=context))
        except ScanCancelled:
            raise
        except (WalkError, OSError) as exc:
            logger.warning("Skipping root %s: %s", root, exc)
            if context is not None:
                context.increment(errors=1)
    return trees
