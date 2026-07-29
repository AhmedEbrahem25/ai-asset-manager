"""Extensible taxonomy for AI assets.

Classifies catalogued assets into categories, tasks, domains and families; derives display
statistics; and judges asset health. Everything it knows comes from plugins, so supporting
a new AI domain never means editing this package.

See :mod:`ai_asset_manager.backend.taxonomy.registry` for how plugins are found, and
:mod:`ai_asset_manager.backend.taxonomy.plugins` for how one is written.
"""

from __future__ import annotations

from ai_asset_manager.backend.taxonomy.registry import (
    ENTRY_POINT_GROUP,
    UNCLASSIFIED,
    TaxonomyRegistry,
    default_registry,
    reset_default_registry,
)
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Classifier,
    DatasetFacts,
    Domain,
    FileSummary,
    Finding,
    HealthReport,
    HealthRule,
    Modality,
    ModelFacts,
    Section,
    StatisticProvider,
    Task,
)

__all__ = [
    "CONFIDENCE_CERTAIN",
    "CONFIDENCE_STRONG",
    "CONFIDENCE_WEAK",
    "ENTRY_POINT_GROUP",
    "UNCLASSIFIED",
    "AssetProfile",
    "Category",
    "Classification",
    "Classifier",
    "DatasetFacts",
    "Domain",
    "FileSummary",
    "Finding",
    "HealthReport",
    "HealthRule",
    "Modality",
    "ModelFacts",
    "Section",
    "StatisticProvider",
    "Task",
    "TaxonomyRegistry",
    "default_registry",
    "reset_default_registry",
]
