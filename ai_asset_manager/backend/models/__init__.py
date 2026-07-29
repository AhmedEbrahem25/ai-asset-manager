"""ORM models.

Every model class is imported here so that ``Base.metadata`` is fully populated by a
single ``import ai_asset_manager.backend.models``. Alembic autogeneration and
``create_all`` both depend on that; importing individual modules piecemeal would produce
a partial schema and unresolvable relationship strings.
"""

from __future__ import annotations

from ai_asset_manager.backend.database.base import Base
from ai_asset_manager.backend.models.asset import Asset, AssetFile, DatasetDetails, ModelDetails
from ai_asset_manager.backend.models.duplicate import DuplicateGroup, DuplicateMember
from ai_asset_manager.backend.models.enums import (
    AssetFormat,
    AssetKind,
    DatasetFormat,
    DuplicateKind,
    FactSource,
    Framework,
    HealthStatus,
    Modality,
    ModelType,
    Precision,
    ScanStatus,
    Severity,
)
from ai_asset_manager.backend.models.health import HealthFinding
from ai_asset_manager.backend.models.link import AssetLink, LinkRelation
from ai_asset_manager.backend.models.scan import ScanRoot, ScanRun
from ai_asset_manager.backend.models.tag import BUILTIN_TAGS, Tag, asset_tags

__all__ = [
    "BUILTIN_TAGS",
    "Asset",
    "AssetFile",
    "AssetFormat",
    "AssetKind",
    "AssetLink",
    "Base",
    "DatasetDetails",
    "DatasetFormat",
    "DuplicateGroup",
    "DuplicateKind",
    "DuplicateMember",
    "FactSource",
    "Framework",
    "HealthFinding",
    "HealthStatus",
    "LinkRelation",
    "Modality",
    "ModelDetails",
    "ModelType",
    "Precision",
    "ScanRoot",
    "ScanRun",
    "ScanStatus",
    "Severity",
    "Tag",
    "asset_tags",
]
