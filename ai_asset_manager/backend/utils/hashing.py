"""Content hashing primitives.

Three tiers, cheapest first, because full SHA256 over a large model library is hours of
disk I/O:

1. :func:`file_identity` — filesystem identity. Two paths sharing a ``(device, inode)``
   pair are one physical file; no bytes are read at all.
2. :func:`quick_signature` — a cheap head/tail digest used to separate same-size files.
3. :func:`sha256_file` — the full digest, computed only for quick-signature collisions.

BLAKE2b backs the quick signature because it is markedly faster than SHA-256 and the
signature is only ever compared against other signatures, never published. SHA-256 is
kept for the authoritative digest since users expect to be able to cross-check it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Streaming read size. 1 MiB measurably beats both 64 KiB and 8 MiB on spinning disks.
READ_CHUNK_BYTES = 1024 * 1024

#: Digest length for quick signatures, in bytes. 16 bytes (128 bits) makes accidental
#: collisions negligible while keeping the index small; a collision only costs a
#: needless tier-3 hash anyway, never a wrong answer.
QUICK_DIGEST_BYTES = 16


class HashCancelled(RuntimeError):
    """Raised when a cancellation event fires partway through hashing a file."""


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    """Raise :class:`HashCancelled` if the caller has asked to stop."""
    if cancel_event is not None and cancel_event.is_set():
        raise HashCancelled("Hashing cancelled")


def file_identity(stat_result: os.stat_result) -> tuple[int, int] | None:
    """Return the ``(device, inode)`` identity of a file, if the platform reports one.

    Python populates both fields on Windows via ``GetFileInformationByHandle``, so this
    works for NTFS hardlinks as well as POSIX ones. Some filesystems (notably certain
    network mounts) report ``0``, in which case identity is unavailable and callers must
    fall back to content hashing.

    Returns:
        The identity pair, or ``None`` when the filesystem does not supply one.
    """
    if stat_result.st_ino == 0 and stat_result.st_dev == 0:
        return None
    return (stat_result.st_dev, stat_result.st_ino)


def quick_signature(
    path: str | os.PathLike[str],
    *,
    size: int | None = None,
    chunk_bytes: int = 4 * 1024 * 1024,
    min_full_hash_bytes: int = 1024 * 1024,
    cancel_event: threading.Event | None = None,
) -> str:
    """Compute a cheap content signature from a file's size, head and tail.

    Files at or below ``min_full_hash_bytes`` are digested in full, because reading two
    chunks from a small file costs the same as reading all of it and a full digest is
    strictly more discriminating.

    The size is mixed into the digest so that two files whose head and tail match but
    whose lengths differ cannot collide.

    Args:
        path: File to read.
        size: Known file size; queried via ``stat`` when omitted.
        chunk_bytes: Bytes to read from each of the head and tail.
        min_full_hash_bytes: Threshold below which the whole file is hashed.
        cancel_event: Optional cooperative cancellation flag.

    Returns:
        A hex digest.

    Raises:
        HashCancelled: If ``cancel_event`` fires during the read.
        OSError: If the file cannot be read.
    """
    _check_cancelled(cancel_event)
    file_size = os.path.getsize(path) if size is None else size

    digest = hashlib.blake2b(digest_size=QUICK_DIGEST_BYTES)
    digest.update(file_size.to_bytes(8, "little"))

    with open(path, "rb") as handle:
        if file_size <= min_full_hash_bytes or file_size <= chunk_bytes * 2:
            while chunk := handle.read(READ_CHUNK_BYTES):
                _check_cancelled(cancel_event)
                digest.update(chunk)
        else:
            digest.update(handle.read(chunk_bytes))
            _check_cancelled(cancel_event)
            handle.seek(-chunk_bytes, os.SEEK_END)
            digest.update(handle.read(chunk_bytes))

    return digest.hexdigest()


def sha256_file(
    path: str | os.PathLike[str],
    *,
    cancel_event: threading.Event | None = None,
    progress: object = None,
) -> str:
    """Compute the full SHA-256 digest of a file.

    Reserved for quick-signature collisions and explicit verification requests.

    Args:
        path: File to read.
        cancel_event: Optional cooperative cancellation flag, checked every chunk so a
            cancelled scan does not have to finish hashing a 40 GB file first.
        progress: Optional callable invoked with the byte count of each chunk read.

    Returns:
        A 64-character hex digest.

    Raises:
        HashCancelled: If ``cancel_event`` fires during the read.
        OSError: If the file cannot be read.
    """
    _check_cancelled(cancel_event)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            _check_cancelled(cancel_event)
            digest.update(chunk)
            if callable(progress):
                progress(len(chunk))
    return digest.hexdigest()


def hash_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of an in-memory buffer."""
    return hashlib.sha256(payload).hexdigest()


def fingerprint_entries(entries: Iterable[tuple[str, int, float]]) -> str:
    """Build an asset fingerprint from its files' metadata.

    The fingerprint answers "has anything about this asset changed since last scan?"
    without reading a single byte of content. An unchanged fingerprint lets a rescan skip
    detection and parsing entirely, which is what makes the second scan of a large
    library fast.

    Modification times are rounded to whole seconds: filesystems disagree about
    sub-second precision, and a fingerprint that flickers between scans would defeat the
    optimisation it exists to enable.

    Args:
        entries: Tuples of ``(relative_path, size_bytes, mtime_seconds)``.

    Returns:
        A hex digest that is stable regardless of iteration order.
    """
    digest = hashlib.blake2b(digest_size=32)
    for relpath, size, mtime in sorted(entries):
        digest.update(relpath.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(int(size).to_bytes(8, "little", signed=False))
        digest.update(int(mtime).to_bytes(8, "little", signed=True))
        digest.update(b"\n")
    return digest.hexdigest()


def combine_hashes(hashes: Sequence[str]) -> str:
    """Combine per-file content hashes into one asset-level digest.

    Sorted before combining so that two asset directories holding identical content hash
    the same regardless of the order their files happened to be walked in.
    """
    digest = hashlib.blake2b(digest_size=32)
    for item in sorted(hashes):
        digest.update(item.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_sha256(path: str | os.PathLike[str], expected: str) -> bool:
    """Check a file against an expected SHA-256 digest.

    Used to validate Ollama blobs, whose filenames embed their own digest.
    """
    try:
        return sha256_file(path).lower() == expected.lower().removeprefix("sha256:")
    except OSError as exc:
        logger.debug("Cannot verify %s: %s", Path(path).name, exc)
        return False
