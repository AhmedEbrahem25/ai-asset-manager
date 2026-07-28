"""Inventory Engine.

Builds a categorised, classified inventory of everything in the catalogue, reading only
from the database. What each asset *is for* — its task, family and health — comes from the
plugin taxonomy in :mod:`ai_asset_manager.backend.taxonomy`, so this package knows no model
families and no dataset formats of its own.

Independent of search: this answers "show me everything I own", not "find the thing I am
thinking of".
"""

from __future__ import annotations

from ai_asset_manager.backend.inventory.categories import (
    all_categories,
    all_sections,
    categories_in_section,
    category_info,
    domain_label,
    known_aliases,
    label_of,
    order_of,
    resolve_alias,
    section_info,
    section_label,
    section_of,
    section_order,
    task_label,
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
from ai_asset_manager.backend.inventory.profile import build_profile, load_file_summaries
from ai_asset_manager.backend.inventory.tree import (
    DEFAULT_LEVELS,
    TREE_LEVELS,
    TreeNode,
    build_tree,
    flatten,
)

__all__ = [
    "DEFAULT_LEVELS",
    "GROUP_BY_FIELDS",
    "SORT_FIELDS",
    "TREE_LEVELS",
    "CategoryCount",
    "InventoryEngine",
    "InventoryGroup",
    "InventoryItem",
    "InventoryReport",
    "InventorySummary",
    "TreeNode",
    "all_categories",
    "all_sections",
    "available_formats",
    "build_profile",
    "build_tree",
    "categories_in_section",
    "category_info",
    "domain_label",
    "export_report",
    "flatten",
    "format_parameters",
    "get_exporter",
    "known_aliases",
    "label_of",
    "load_file_summaries",
    "order_of",
    "resolve_alias",
    "section_info",
    "section_label",
    "section_of",
    "section_order",
    "suggest_filename",
    "task_label",
]
