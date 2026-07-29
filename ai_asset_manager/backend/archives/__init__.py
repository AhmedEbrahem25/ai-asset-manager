"""Read-only archive inspection.

An archive is catalogued from its *table of contents*, never from its contents. Nothing is
unpacked to disk, no temporary directory is created, and the only bytes that reach memory
are those of a handful of named metadata files small enough to be configuration rather than
data.

The split is deliberate: :mod:`~ai_asset_manager.backend.archives.reader` knows how to get
a listing out of a container format and nothing about AI, while
:mod:`~ai_asset_manager.backend.archives.classify` knows what a listing means and nothing
about zip headers.
"""

from __future__ import annotations

from ai_asset_manager.backend.archives.classify import (
    ArchiveVerdict,
    classify_listing,
)
from ai_asset_manager.backend.archives.reader import (
    ARCHIVE_SUFFIXES,
    ArchiveEntry,
    ArchiveListing,
    archive_format,
    inspect_archive,
    is_archive_name,
)

__all__ = [
    "ARCHIVE_SUFFIXES",
    "ArchiveEntry",
    "ArchiveListing",
    "ArchiveVerdict",
    "archive_format",
    "classify_listing",
    "inspect_archive",
    "is_archive_name",
]
