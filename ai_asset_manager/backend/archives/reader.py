r"""Getting a table of contents out of an archive without unpacking it.

Three levels of access, and the boundaries between them are the whole design:

**Level 1 — the listing.** Name, size and the entry names inside. For a zip this is the
central directory, a few hundred kilobytes at the end of the file whatever the archive
weighs. For a tar it is the member headers. Nothing is decompressed that does not have to
be, and nothing is written anywhere.

**Level 2 — named metadata, in memory.** A ``config.json`` or a ``data.yaml`` inside the
archive is read into a bytes object and parsed, provided it is small enough to be
configuration. It never touches the filesystem. :data:`MAX_METADATA_FILES` of them at most,
:data:`MAX_METADATA_BYTES` each.

**Level 3 — never.** Images, video, weights, checkpoints, parquet, arrow, ONNX, GGUF,
safetensors: not read at any size, for any reason. :func:`_is_readable_metadata` is the one
gate, and it works from an allow-list of names rather than a deny-list of extensions, so a
format nobody here has heard of is refused by default.

Cost is bounded twice over — by entry count and by bytes decompressed — because an archive
is an untrusted input and "how long does listing this take" must have an answer that does
not depend on what is inside it.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Literal

from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Filename suffixes recognised as archives. Ordered longest-first so ``.tar.gz`` is
#: matched before ``.gz`` would be.
ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
    ".tgz",
    ".tbz2",
    ".txz",
    ".tar",
    ".zip",
    ".7z",
    ".rar",
)

#: Suffix to the reader that handles it.
_SUFFIX_FORMATS: tuple[tuple[str, str], ...] = (
    (".tar.gz", "tar.gz"),
    (".tgz", "tar.gz"),
    (".tar.bz2", "tar.bz2"),
    (".tbz2", "tar.bz2"),
    (".tar.xz", "tar.xz"),
    (".txz", "tar.xz"),
    (".tar.zst", "tar.zst"),
    (".tar", "tar"),
    (".zip", "zip"),
    (".7z", "7z"),
    (".rar", "rar"),
)

#: Tar modes by format. ``r:`` is seekable and costs nothing to walk; the compressed modes
#: decompress forward, which is why they carry a byte budget.
#:
#: Typed as the literals ``tarfile.open`` accepts rather than as ``str``: the mode decides
#: which overload applies, and a plain string would make every call unresolvable.
_TarMode = Literal["r:", "r:gz", "r:bz2", "r:xz"]

_TAR_MODES: dict[str, _TarMode] = {
    "tar": "r:",
    "tar.gz": "r:gz",
    "tar.bz2": "r:bz2",
    "tar.xz": "r:xz",
}

#: Stop listing after this many entries. A dataset archive with a million images tells us
#: everything we need in its first few thousand names, and the shape of the tree is already
#: obvious by then.
MAX_ENTRIES = 4000

#: Prefix of the error set when no reader exists for a format at all.
NO_READER_PREFIX = "no reader"

#: Markers identifying a failure caused by a missing reader rather than a bad archive.
#: Kept beside the strings that produce them so the two cannot drift apart — a health rule
#: guessing at this from prose is how the advice "your archive may be damaged" ended up on
#: fifteen perfectly good msys2 packages.
_MISSING_READER_MARKERS = (NO_READER_PREFIX, "not installed")

#: Stop walking a *compressed* tar after this many decompressed bytes. Uncompressed tars
#: and zips are exempt: their listings are seeks, not decompression.
MAX_DECOMPRESSED_SCAN_BYTES = 256 * 1024 * 1024

#: Largest file read into memory from inside an archive. Real configs are kilobytes.
MAX_METADATA_BYTES = 256 * 1024

#: How many metadata files are read from one archive, at most.
MAX_METADATA_FILES = 8

#: Basenames worth reading into memory, and the only ones that ever are. An allow-list,
#: not a deny-list: anything not named here is left in the archive whatever its size.
METADATA_NAMES: frozenset[str] = frozenset(
    {
        "config.json",
        "adapter_config.json",
        "model_index.json",
        "dataset_info.json",
        "dataset_infos.json",
        "dataset_dict.json",
        "data.yaml",
        "data.yml",
        "dataset.yaml",
        "metadata.json",
        "readme.md",
        "meta.json",
        "manifest.json",
    }
)


def is_archive_name(name: str) -> bool:
    """Report whether a filename looks like an archive this module can describe."""
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def is_missing_reader(error: str | None) -> bool:
    """Report whether a listing error means "no reader" rather than "bad archive".

    Examples:
        >>> is_missing_reader("no reader for tar.zst")
        True
        >>> is_missing_reader("py7zr not installed")
        True
        >>> is_missing_reader("BadZipFile: File is not a zip file")
        False
    """
    if not error:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in _MISSING_READER_MARKERS)


def archive_format(name: str) -> str | None:
    """Return the container format implied by a filename, or ``None``.

    Examples:
        >>> archive_format("dataset.tar.gz")
        'tar.gz'
        >>> archive_format("weights.safetensors") is None
        True
    """
    lowered = name.lower()
    for suffix, fmt in _SUFFIX_FORMATS:
        if lowered.endswith(suffix):
            return fmt
    return None


@dataclass(slots=True, frozen=True)
class ArchiveEntry:
    """One member of an archive, as named in its table of contents."""

    #: Path inside the archive, forward-slashed and lower-cased. Everything matches on
    #: this, because archives written on three operating systems do not agree on case.
    name: str
    #: Uncompressed size in bytes, or ``0`` for directories and unknown members.
    size: int
    is_dir: bool = False
    #: The path exactly as the archive spells it. Only used for display: an archive whose
    #: sole root folder is ``UNSW-NB15`` should not be catalogued as ``unsw-nb15``.
    raw: str = ""

    @property
    def basename(self) -> str:
        """Return the member's filename without its directory path."""
        return self.name.rsplit("/", 1)[-1]

    @property
    def extension(self) -> str:
        """Return the member's extension, including the leading dot."""
        base = self.basename
        stem, dot, tail = base.rpartition(".")
        return f".{tail}" if dot and stem else ""


@dataclass(slots=True)
class ArchiveListing:
    """What an archive says about itself, without being unpacked."""

    path: str
    size: int
    #: One of the values :func:`archive_format` returns, or ``"unknown"``.
    format: str
    entries: list[ArchiveEntry] = field(default_factory=list)
    #: Metadata files read into memory, keyed by lower-cased basename.
    metadata: dict[str, bytes] = field(default_factory=dict)
    #: True when the listing stopped at a limit rather than at the end of the archive.
    truncated: bool = False
    #: Set when the archive could not be read at all — encrypted, corrupt, or a format
    #: with no reader installed. The asset is still catalogued; only its contents are
    #: unknown.
    error: str | None = None

    @property
    def missing_reader(self) -> bool:
        """Report whether the failure was a missing reader rather than a bad archive.

        The distinction is the whole of what the user should do about it: a missing reader
        is fixed by installing one, a damaged archive is not.
        """
        return is_missing_reader(self.error)

    @property
    def name(self) -> str:
        """Return the archive's own filename."""
        return os.path.basename(self.path)

    @property
    def is_listed(self) -> bool:
        """Report whether any table of contents was recovered."""
        return bool(self.entries)

    @property
    def file_entries(self) -> list[ArchiveEntry]:
        """Return members that are files rather than directory markers."""
        return [entry for entry in self.entries if not entry.is_dir]

    @property
    def basenames(self) -> frozenset[str]:
        """Return every member's filename, lower-cased."""
        return frozenset(entry.basename for entry in self.entries)

    @property
    def directories(self) -> frozenset[str]:
        """Return every directory name appearing anywhere in a member path.

        A tar records directories explicitly and a zip often does not, so these are
        derived from the member paths themselves rather than trusted from the headers.
        """
        found: set[str] = set()
        for entry in self.entries:
            segments = entry.name.split("/")
            found.update(segments[:-1] if not entry.is_dir else segments)
        found.discard("")
        return frozenset(found)

    @property
    def top_level(self) -> frozenset[str]:
        """Return the first path segment of every member.

        An archive whose members all sit under one folder is the same dataset as one whose
        members sit at the root; this is what lets both be recognised.

        Spelled as the archive spells it, because this is the one place a member name is
        used as a name rather than matched against a marker.
        """
        return frozenset(
            (entry.raw or entry.name).split("/", 1)[0]
            for entry in self.entries
            if entry.name
        )

    @property
    def extension_counts(self) -> dict[str, int]:
        """Return a histogram of member extensions."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.is_dir:
                continue
            suffix = entry.extension
            if suffix:
                counts[suffix] = counts.get(suffix, 0) + 1
        return counts

    def count(self, *extensions: str) -> int:
        """Return how many members carry any of these extensions."""
        counts = self.extension_counts
        return sum(counts.get(extension.lower(), 0) for extension in extensions)

    def has_name(self, *names: str) -> bool:
        """Report whether any member has one of these basenames."""
        present = self.basenames
        return any(name.lower() in present for name in names)

    def has_dir(self, *names: str) -> bool:
        """Report whether any of these directory names appears in the tree."""
        present = self.directories
        return any(name.lower() in present for name in names)

    def matching(self, *fragments: str) -> int:
        """Return how many member paths contain any of these substrings."""
        lowered = tuple(fragment.lower() for fragment in fragments)
        return sum(
            1
            for entry in self.entries
            if any(fragment in entry.name for fragment in lowered)
        )

    def json_metadata(self, name: str) -> Any | None:
        """Return a metadata file parsed as JSON, or ``None``.

        A half-written config inside an archive is no more exceptional than one on disk, so
        a parse failure is reported as absence rather than raised.
        """
        import json

        raw = self.metadata.get(name.lower())
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            logger.debug("Malformed JSON in %s!%s", self.path, name)
            return None


def _is_readable_metadata(name: str, size: int) -> bool:
    """Report whether a member may be read into memory.

    Two conditions, both required: the basename is on the allow-list, and the member is
    small enough to be configuration. Level 3 of the module docstring is enforced here and
    nowhere else.
    """
    return name in METADATA_NAMES and 0 < size <= MAX_METADATA_BYTES


def _normalise(name: str) -> str:
    """Return a member path in the canonical form the classifier expects."""
    return name.replace("\\", "/").lstrip("./").lower()


def inspect_archive(path: str, size: int) -> ArchiveListing:
    """Return an archive's table of contents and its small metadata files.

    Args:
        path: Absolute path to the archive.
        size: The archive's own size on disk, already known from the walk.

    Returns:
        A listing. Never raises: an unreadable archive comes back with ``error`` set and
        no entries, because a corrupt zip in a Downloads folder must not end a scan.
    """
    fmt = archive_format(path) or "unknown"
    listing = ArchiveListing(path=path, size=size, format=fmt)

    try:
        if fmt == "zip":
            _read_zip(path, listing)
        elif fmt in _TAR_MODES:
            _read_tar(path, listing, mode=_TAR_MODES[fmt])
        elif fmt == "7z":
            _read_7z(path, listing)
        elif fmt == "rar":
            _read_rar(path, listing)
        else:
            listing.error = f"{NO_READER_PREFIX} for {fmt}"
    except Exception as exc:  # an archive is untrusted input; nothing it does ends a scan
        listing.error = f"{type(exc).__name__}: {exc}"
        logger.debug("Cannot list archive %s: %s", path, exc)

    return listing


def _record(
    listing: ArchiveListing, name: str, size: int, *, is_dir: bool
) -> ArchiveEntry | None:
    """Append one member to a listing, honouring the entry cap.

    Returns:
        The recorded entry, or ``None`` when the cap has been reached.
    """
    if len(listing.entries) >= MAX_ENTRIES:
        listing.truncated = True
        return None
    cleaned = name.replace("\\", "/").lstrip("./")
    entry = ArchiveEntry(
        name=cleaned.lower(), size=size, is_dir=is_dir, raw=cleaned
    )
    listing.entries.append(entry)
    return entry


def _read_zip(path: str, listing: ArchiveListing) -> None:
    """List a zip from its central directory.

    The central directory sits at the end of the file and is proportional to the number of
    members, not to their size: a 40 GB zip is listed by reading a few hundred kilobytes.
    """
    with zipfile.ZipFile(path) as archive:
        wanted: list[zipfile.ZipInfo] = []
        for info in archive.infolist():
            entry = _record(
                listing, info.filename, info.file_size, is_dir=info.is_dir()
            )
            if entry is None:
                break
            if (
                len(wanted) < MAX_METADATA_FILES
                and not entry.is_dir
                and _is_readable_metadata(entry.basename, info.file_size)
            ):
                wanted.append(info)

        for info in wanted:
            basename = _normalise(info.filename).rsplit("/", 1)[-1]
            if basename in listing.metadata:
                continue
            try:
                with archive.open(info) as handle:
                    listing.metadata[basename] = handle.read(MAX_METADATA_BYTES)
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                # RuntimeError is what zipfile raises for an encrypted member.
                logger.debug("Cannot read %s!%s: %s", path, info.filename, exc)


def _read_tar(path: str, listing: ArchiveListing, *, mode: _TarMode) -> None:
    """List a tar, reading metadata members as they are passed.

    Members are visited in stored order and metadata is read at the moment its header is
    reached, so a compressed tar is walked exactly once. Compressed modes stop at
    :data:`MAX_DECOMPRESSED_SCAN_BYTES`; an uncompressed tar seeks over member data and is
    capped by entry count alone.
    """
    budgeted = mode != "r:"

    with tarfile.open(path, mode) as archive:
        for member in archive:
            entry = _record(
                listing, member.name, member.size, is_dir=member.isdir()
            )
            if entry is None:
                break

            if (
                member.isfile()
                and len(listing.metadata) < MAX_METADATA_FILES
                and _is_readable_metadata(entry.basename, member.size)
                and entry.basename not in listing.metadata
            ):
                try:
                    handle = archive.extractfile(member)
                    if handle is not None:
                        with handle:
                            listing.metadata[entry.basename] = handle.read(
                                MAX_METADATA_BYTES
                            )
                except (OSError, tarfile.TarError) as exc:
                    logger.debug("Cannot read %s!%s: %s", path, member.name, exc)

            if budgeted and archive.offset > MAX_DECOMPRESSED_SCAN_BYTES:
                listing.truncated = True
                break


def _read_7z(path: str, listing: ArchiveListing) -> None:
    """List a 7z archive, if a reader is installed.

    ``py7zr`` is an optional dependency (``pip install ai-asset-manager[archives]``).
    Without it the archive is still catalogued from its name and size — a
    ``CICIDS2017.7z`` is recognisable without opening it — but its contents stay unknown,
    which the listing says rather than guesses.
    """
    try:
        import py7zr
    except ImportError:
        listing.error = "py7zr not installed"
        return

    with py7zr.SevenZipFile(path, mode="r") as archive:
        for info in archive.list():
            if _record(
                listing,
                info.filename,
                int(getattr(info, "uncompressed", 0) or 0),
                is_dir=bool(info.is_directory),
            ) is None:
                break

    # Reading a member from a 7z means decompressing the solid block it lives in, which
    # can mean decompressing the whole archive. The listing alone is worth having; the
    # metadata is not worth that price.


def _read_rar(path: str, listing: ArchiveListing) -> None:
    """List a RAR archive, if a reader is installed.

    ``rarfile`` is an optional dependency and additionally needs an ``unrar`` binary on the
    path for some archive versions. Both absences are reported, not raised.
    """
    try:
        import rarfile
    except ImportError:
        listing.error = "rarfile not installed"
        return

    with rarfile.RarFile(path) as archive:
        for info in archive.infolist():
            if _record(
                listing, info.filename, info.file_size or 0, is_dir=info.is_dir()
            ) is None:
                break
