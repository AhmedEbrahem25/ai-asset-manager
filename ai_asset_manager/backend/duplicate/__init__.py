"""Finding the same content in more than one place.

Two questions that look alike and cost very differently:

:mod:`~ai_asset_manager.backend.duplicate.detector` answers *are these bytes identical?* It
reads files, in three tiers, and persists what it learns so the reading is done once.

:mod:`~ai_asset_manager.backend.duplicate.installations` answers *is this the same model
shipped by several applications?* It reads nothing at all — the catalogue already records
every size, file count and filename, and applications that embed a model ship it verbatim.
When the first module has run, the second uses its digests and says so.
"""

from __future__ import annotations

from ai_asset_manager.backend.duplicate.detector import (
    DuplicateDetector,
    DuplicateStats,
    DuplicateSummary,
    list_duplicate_groups,
    total_wasted_bytes,
)
from ai_asset_manager.backend.duplicate.installations import (
    Installation,
    InstallationGroup,
    find_duplicate_installations,
    total_reclaimable,
)

__all__ = [
    "DuplicateDetector",
    "DuplicateStats",
    "DuplicateSummary",
    "Installation",
    "InstallationGroup",
    "find_duplicate_installations",
    "list_duplicate_groups",
    "total_reclaimable",
    "total_wasted_bytes",
]
