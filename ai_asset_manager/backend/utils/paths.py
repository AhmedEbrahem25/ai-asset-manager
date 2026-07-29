"""Path normalisation and classification helpers.

Windows is a first-class target here: drive letters, case-insensitive comparison and
extended-length paths all need handling that ``pathlib`` does not do on its own.
"""

from __future__ import annotations

import os
import re
from fnmatch import fnmatch
from pathlib import PurePath

#: Extensions that hold model weights or dataset payloads rather than configuration.
PAYLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".safetensors",
        ".gguf",
        ".ggml",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".engine",
        ".plan",
        ".pb",
        ".h5",
        ".keras",
        ".tflite",
        ".mlmodel",
        ".mlpackage",
        ".npz",
        ".npy",
        ".msgpack",
        ".params",
        ".pdparams",
        ".xml",
        ".model",
        ".gguf.part",
    }
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".ppm", ".pgm"}
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
)

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
)

TEXT_DATA_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".jsonl", ".csv", ".tsv", ".txt", ".parquet", ".arrow", ".feather", ".xml"}
)

POINT_CLOUD_EXTENSIONS: frozenset[str] = frozenset({".pcd", ".ply", ".las", ".laz", ".bin"})

#: Markers left behind by interrupted downloads.
INCOMPLETE_SUFFIXES: tuple[str, ...] = (
    ".incomplete",
    ".part",
    ".partial",
    ".download",
    ".crdownload",
    ".tmp",
    ".aria2",
)

_WINDOWS_EXTENDED_PREFIX = "\\\\?\\"
_DRIVE_RE = re.compile(r"^([A-Za-z]):")


def normalize_path(path: str | os.PathLike[str]) -> str:
    r"""Return a canonical absolute path string suitable for use as a database key.

    Resolves ``.``/``..``, expands ``~``, makes the path absolute, and upper-cases the
    Windows drive letter so ``c:\\models`` and ``C:\\Models`` do not become two rows.
    Symlinks are deliberately *not* resolved: the user asked about the path they gave.

    Args:
        path: Any path-like value.

    Returns:
        A normalised absolute path string.
    """
    text = os.fspath(path)
    if text.startswith(_WINDOWS_EXTENDED_PREFIX):
        text = text[len(_WINDOWS_EXTENDED_PREFIX) :]
    text = os.path.abspath(os.path.expanduser(text))
    match = _DRIVE_RE.match(text)
    if match:
        text = match.group(1).upper() + text[1:]
    return text


def get_drive(path: str | os.PathLike[str]) -> str | None:
    """Return the drive or mount identifier a path lives on.

    On Windows this is the drive letter (``"C:"``) or the UNC share. On POSIX it is the
    first path component, which is a good enough proxy for grouping storage by device in
    the dashboard.
    """
    text = normalize_path(path)
    drive, _ = os.path.splitdrive(text)
    if drive:
        return drive.upper()
    parts = PurePath(text).parts
    if len(parts) >= 2:
        return f"/{parts[1]}"
    return "/"


def is_excluded_dir(
    name: str, excluded: frozenset[str], path: str | None = None
) -> bool:
    r"""Report whether a directory should be pruned from the walk.

    Two kinds of entry, and the difference matters:

    A bare name (``node_modules``) prunes that directory wherever it appears. Compared
    case-insensitively, because Windows treats ``Node_Modules`` and ``node_modules`` as
    the same directory.

    An entry containing a separator (``appdata\local\temp``) prunes only that *location*,
    so a folder legitimately called ``temp`` inside a dataset is still walked. This needs
    the full path, and silently did nothing until it got one: the entry sat in the default
    list looking effective while the temp directory -- 138,000 files on the development
    machine -- was walked on every scan.
    """
    if name.lower() in excluded:
        return True
    if path is None:
        return False

    normalised = path.replace("/", "\\").lower()
    return any(
        normalised.endswith("\\" + item)
        for item in excluded
        if "\\" in item
    )


def matches_any_glob(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """Report whether a filename matches any of the given glob patterns."""
    return any(fnmatch(name, pattern) for pattern in patterns)


def is_incomplete_marker(name: str) -> bool:
    """Report whether a filename looks like an interrupted download.

    Matches both trailing markers (``model.safetensors.incomplete``) and HuggingFace's
    blob-staging convention of appending the suffix mid-name.
    """
    lowered = name.lower()
    return any(suffix in lowered for suffix in INCOMPLETE_SUFFIXES)


def is_payload_file(name: str) -> bool:
    """Report whether a filename looks like a weight or data payload."""
    return os.path.splitext(name)[1].lower() in PAYLOAD_EXTENSIONS


def classify_extension(name: str) -> str | None:
    """Return a coarse content class for a filename.

    Returns:
        One of ``"image"``, ``"video"``, ``"audio"``, ``"text"``, ``"point_cloud"``,
        ``"payload"``, or ``None`` when the extension is unrecognised.
    """
    # `os.path.splitext` rather than `Path(...).suffix`: this runs once per file per
    # content census, and building a PurePath to read the end of a string is the most
    # expensive way to do it.
    suffix = os.path.splitext(name)[1].lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in PAYLOAD_EXTENSIONS:
        return "payload"
    if suffix in TEXT_DATA_EXTENSIONS:
        return "text"
    if suffix in POINT_CLOUD_EXTENSIONS:
        return "point_cloud"
    return None


def safe_relpath(path: str | os.PathLike[str], start: str | os.PathLike[str]) -> str:
    """Return ``path`` relative to ``start``, falling back to the absolute path.

    ``os.path.relpath`` raises on Windows when the two paths sit on different drives,
    which happens whenever an asset root and one of its files are reached through
    different mount points.
    """
    try:
        return os.path.relpath(os.fspath(path), os.fspath(start)).replace("\\", "/")
    except ValueError:
        return os.fspath(path).replace("\\", "/")


def path_depth(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> int:
    """Return how many directory levels ``path`` sits below ``root``."""
    relative = safe_relpath(path, root)
    if relative in (".", ""):
        return 0
    return relative.count("/") + 1


def shorten_path(path: str, max_length: int = 60) -> str:
    """Truncate a path in the middle for fixed-width display.

    Examples:
        >>> shorten_path("/a/very/long/path/to/a/model/file.safetensors", 30)
        '/a/very/lo...el/file.safetensors'
    """
    if len(path) <= max_length:
        return path
    keep = max_length - 3
    head = keep // 3
    tail = keep - head
    return f"{path[:head]}...{path[-tail:]}"
