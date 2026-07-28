"""Inventory Engine.

Builds a categorised inventory of everything in the catalogue, reading only from the
database. Independent of search: this module answers "show me everything I own", not
"find the thing I am thinking of".
"""

from __future__ import annotations

from ai_asset_manager.backend.inventory.categories import (
    CATEGORY_ALIASES,
    CATEGORY_INFO,
    InventoryCategory,
    InventorySection,
    classify_asset,
    classify_dataset,
    classify_model,
    known_aliases,
    label_of,
    resolve_alias,
)
from ai_asset_manager.backend.inventory.engine import (
    GROUP_BY_FIELDS,
    SORT_FIELDS,
    CategoryCount,
    InventoryEngine,
    InventoryGroup,
    InventoryItem,
    InventoryReport,
    InventorySummary,
)
from ai_asset_manager.backend.inventory.export import (
    available_formats,
    export_report,
    format_parameters,
    get_exporter,
    suggest_filename,
)

__all__ = [
    "CATEGORY_ALIASES",
    "CATEGORY_INFO",
    "GROUP_BY_FIELDS",
    "SORT_FIELDS",
    "CategoryCount",
    "InventoryCategory",
    "InventoryEngine",
    "InventoryGroup",
    "InventoryItem",
    "InventoryReport",
    "InventorySection",
    "InventorySummary",
    "available_formats",
    "classify_asset",
    "classify_dataset",
    "classify_model",
    "export_report",
    "format_parameters",
    "get_exporter",
    "known_aliases",
    "label_of",
    "resolve_alias",
    "suggest_filename",
]
