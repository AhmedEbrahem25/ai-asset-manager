"""The Inventory Engine.

Answers one question: *what AI assets do I have, and where are they?*

Strictly read-only and strictly database-bound. It opens no files, walks no directories
and parses no metadata — every value it reports was recorded by the scanner during M1.
That is what makes it instant regardless of library size, and it is also why it can never
disagree with the catalogue: there is no second code path that could drift.

Deliberately independent of search. Search answers "find me the thing I am thinking of";
inventory answers "show me everything I own". They share the database and nothing else.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.inventory.categories import (
    InventoryCategory,
    InventorySection,
    classify_asset,
    label_of,
    order_of,
    section_of,
)
from ai_asset_manager.backend.models import Asset, DatasetDetails, ModelDetails
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Fields a report may be grouped by.
GROUP_BY_FIELDS = ("category", "section", "framework", "drive", "architecture",
                   "dataset_type", "format")

#: Fields a report may be sorted by.
SORT_FIELDS = ("name", "size", "date", "category", "framework", "files")


@dataclass(slots=True)
class InventoryItem:
    """One asset as it appears in the inventory."""

    asset_id: int
    name: str
    category: InventoryCategory
    section: InventorySection
    subcategory: str | None
    framework: str
    format: str
    size_bytes: int
    file_count: int
    path: str
    root_folder: str
    drive: str | None
    modified_at: datetime | None

    # -- model specifics ---------------------------------------------------
    architecture: str | None = None
    param_count: int | None = None
    param_count_is_exact: bool = False
    quantization: str | None = None
    precision: str | None = None
    context_length: int | None = None
    repo_id: str | None = None
    author: str | None = None
    license: str | None = None

    # -- dataset specifics -------------------------------------------------
    dataset_type: str | None = None
    num_images: int = 0
    num_videos: int = 0
    num_annotations: int = 0
    num_classes: int | None = None
    splits: dict[str, int] = field(default_factory=dict)

    tags: list[str] = field(default_factory=list)
    is_missing: bool = False

    @property
    def category_label(self) -> str:
        """Return the human-readable category name."""
        return label_of(self.category)

    @property
    def is_model(self) -> bool:
        """Report whether this item is a model rather than a dataset."""
        return self.section is InventorySection.MODELS

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Used by the export layer and, later, by the REST API, so that both describe an
        inventory item identically.
        """
        payload: dict[str, Any] = {
            "id": self.asset_id,
            "name": self.name,
            "category": self.category.value,
            "category_label": self.category_label,
            "section": self.section.value,
            "subcategory": self.subcategory,
            "framework": self.framework,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "path": self.path,
            "root_folder": self.root_folder,
            "drive": self.drive,
            "last_modified": self.modified_at.isoformat() if self.modified_at else None,
            "tags": self.tags,
        }
        if self.is_model:
            payload.update(
                architecture=self.architecture,
                parameters=self.param_count,
                parameters_exact=self.param_count_is_exact,
                quantization=self.quantization,
                precision=self.precision,
                context_length=self.context_length,
                repo_id=self.repo_id,
                author=self.author,
                license=self.license,
            )
        else:
            payload.update(
                dataset_type=self.dataset_type,
                images=self.num_images,
                videos=self.num_videos,
                annotations=self.num_annotations,
                classes=self.num_classes,
                splits=self.splits,
            )
        return payload


@dataclass(slots=True)
class InventoryGroup:
    """A set of items sharing a grouping key."""

    key: str
    label: str
    items: list[InventoryItem] = field(default_factory=list)
    #: Preserves taxonomy ordering when grouping by category.
    order: int = 0

    @property
    def count(self) -> int:
        """Return the number of items in the group."""
        return len(self.items)

    @property
    def total_bytes(self) -> int:
        """Return the group's total size."""
        return sum(item.size_bytes for item in self.items)


@dataclass(slots=True)
class CategoryCount:
    """Per-category totals for the summary."""

    category: InventoryCategory
    label: str
    count: int
    total_bytes: int


@dataclass(slots=True)
class InventorySummary:
    """Headline totals for an inventory report."""

    total_assets: int = 0
    total_bytes: int = 0
    #: Bytes actually occupied, counting shared extents once.
    physical_bytes: int = 0
    total_files: int = 0
    by_category: list[CategoryCount] = field(default_factory=list)
    by_section: dict[str, int] = field(default_factory=dict)
    by_drive: dict[str, int] = field(default_factory=dict)
    by_framework: dict[str, int] = field(default_factory=dict)
    missing_assets: int = 0

    def category_count(self, category: InventoryCategory) -> int:
        """Return how many assets fall in a category."""
        for entry in self.by_category:
            if entry.category is category:
                return entry.count
        return 0


@dataclass(slots=True)
class InventoryReport:
    """The complete result of an inventory query."""

    items: list[InventoryItem] = field(default_factory=list)
    groups: list[InventoryGroup] = field(default_factory=list)
    summary: InventorySummary = field(default_factory=InventorySummary)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: What was asked for, echoed back so an export is self-describing.
    query: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Report whether anything matched."""
        return not self.items

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the whole report."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "query": self.query,
            "summary": {
                "total_assets": self.summary.total_assets,
                "total_bytes": self.summary.total_bytes,
                "physical_bytes": self.summary.physical_bytes,
                "total_files": self.summary.total_files,
                "missing_assets": self.summary.missing_assets,
                "by_category": [
                    {
                        "category": entry.category.value,
                        "label": entry.label,
                        "count": entry.count,
                        "total_bytes": entry.total_bytes,
                    }
                    for entry in self.summary.by_category
                ],
                "by_section": self.summary.by_section,
                "by_drive": self.summary.by_drive,
                "by_framework": self.summary.by_framework,
            },
            "items": [item.as_dict() for item in self.items],
        }


class InventoryEngine:
    """Builds categorised inventories from the catalogue.

    Read-only by construction: the session is used for ``SELECT`` only, and the engine
    exposes no method that writes, moves or deletes anything.
    """

    def __init__(self, session: Session) -> None:
        """Bind the engine to a database session."""
        self.session = session

    def build(
        self,
        categories: Sequence[InventoryCategory] | None = None,
        *,
        drives: Sequence[str] | None = None,
        frameworks: Sequence[str] | None = None,
        group_by: str | None = None,
        sort: str = "size",
        descending: bool = True,
        include_missing: bool = False,
        limit: int | None = None,
    ) -> InventoryReport:
        """Build an inventory report.

        Args:
            categories: Restrict to these categories; ``None`` means everything.
            drives: Restrict to these drives, e.g. ``["F:"]``.
            frameworks: Restrict to these frameworks.
            group_by: One of :data:`GROUP_BY_FIELDS`, or ``None`` for a flat list.
            sort: One of :data:`SORT_FIELDS`.
            descending: Sort direction.
            include_missing: Include assets no longer present on disk.
            limit: Cap the number of items returned. The summary still reflects
                everything that matched, so a truncated table never misreports totals.

        Returns:
            The completed :class:`InventoryReport`.
        """
        rows = self._fetch(drives=drives, frameworks=frameworks, include_missing=include_missing)
        items = [self._to_item(row) for row in rows]

        if categories is not None:
            wanted = set(categories)
            items = [item for item in items if item.category in wanted]

        # The summary is computed before any limit is applied: a table showing the top 20
        # assets must still report the true total for the library.
        summary = self._summarise(items)

        items = self._sort(items, sort=sort, descending=descending)
        if limit is not None and limit > 0:
            items = items[:limit]

        report = InventoryReport(
            items=items,
            summary=summary,
            query={
                "categories": [category.value for category in categories] if categories else None,
                "drives": list(drives) if drives else None,
                "frameworks": list(frameworks) if frameworks else None,
                "group_by": group_by,
                "sort": sort,
                "descending": descending,
                "include_missing": include_missing,
                "limit": limit,
            },
        )

        if group_by:
            report.groups = self._group(items, group_by)

        logger.debug(
            "Inventory built: %d item(s) across %d categor(y/ies)",
            len(items), len(summary.by_category),
        )
        return report

    # -- data access --------------------------------------------------------

    def _base_query(self, *, include_missing: bool) -> Select[Any]:
        """Return the single query the whole report is built from.

        One statement with two outer joins, rather than a query per asset. A library of
        several thousand assets would otherwise issue several thousand round trips to
        answer a question the database can answer once.
        """
        statement = (
            select(Asset, ModelDetails, DatasetDetails)
            .outerjoin(ModelDetails, ModelDetails.asset_id == Asset.id)
            .outerjoin(DatasetDetails, DatasetDetails.asset_id == Asset.id)
        )
        if not include_missing:
            statement = statement.where(Asset.is_missing.is_(False))
        return statement

    def _fetch(
        self,
        *,
        drives: Sequence[str] | None,
        frameworks: Sequence[str] | None,
        include_missing: bool,
    ) -> Sequence[Any]:
        """Run the base query with optional column filters applied in SQL."""
        statement = self._base_query(include_missing=include_missing)

        if drives:
            normalised = [drive.upper() if drive.endswith(":") else f"{drive.upper()}:"
                          for drive in drives]
            statement = statement.where(Asset.drive.in_(normalised))
        if frameworks:
            statement = statement.where(Asset.framework.in_([f.lower() for f in frameworks]))

        return self.session.execute(statement).all()

    def _to_item(self, row: Any) -> InventoryItem:
        """Convert one joined row into an inventory item."""
        asset: Asset = row[0]
        model: ModelDetails | None = row[1]
        dataset: DatasetDetails | None = row[2]

        category = classify_asset(
            asset.kind,
            name=asset.display_name or asset.name,
            model_type=model.model_type if model else None,
            architecture=model.architecture if model else None,
            dataset_format=dataset.dataset_format if dataset else None,
            task=dataset.task if dataset else None,
            class_names=list(dataset.class_names) if dataset else None,
            num_images=dataset.num_images if dataset else 0,
            num_videos=dataset.num_videos if dataset else 0,
            num_audio_files=dataset.num_audio_files if dataset else 0,
        )

        item = InventoryItem(
            asset_id=asset.id,
            name=asset.display_name or asset.name,
            category=category,
            section=section_of(category),
            subcategory=asset.subkind,
            framework=asset.framework,
            format=asset.format,
            size_bytes=asset.size_bytes,
            file_count=asset.file_count,
            path=asset.root_path,
            root_folder=_parent_of(asset.root_path),
            drive=asset.drive,
            modified_at=asset.modified_at,
            tags=[tag.name for tag in asset.tags],
            is_missing=asset.is_missing,
        )

        if model is not None:
            item.architecture = model.architecture
            item.param_count = model.param_count
            item.param_count_is_exact = model.param_count_is_exact
            item.quantization = model.quantization
            item.precision = model.precision
            item.context_length = model.context_length
            item.repo_id = model.repo_id
            item.author = model.author
            item.license = model.license

        if dataset is not None:
            item.dataset_type = dataset.dataset_format
            item.num_images = dataset.num_images
            item.num_videos = dataset.num_videos
            item.num_annotations = dataset.num_annotations
            item.num_classes = dataset.num_classes
            item.splits = dict(dataset.splits)
            item.license = item.license or dataset.license

        return item

    # -- shaping ------------------------------------------------------------

    def _summarise(self, items: Sequence[InventoryItem]) -> InventorySummary:
        """Compute headline totals over the matched items."""
        summary = InventorySummary(
            total_assets=len(items),
            total_bytes=sum(item.size_bytes for item in items),
            total_files=sum(item.file_count for item in items),
            missing_assets=sum(1 for item in items if item.is_missing),
        )

        per_category: dict[InventoryCategory, list[InventoryItem]] = defaultdict(list)
        for item in items:
            per_category[item.category].append(item)

        summary.by_category = [
            CategoryCount(
                category=category,
                label=label_of(category),
                count=len(group),
                total_bytes=sum(entry.size_bytes for entry in group),
            )
            for category, group in sorted(per_category.items(), key=lambda pair: order_of(pair[0]))
        ]

        for item in items:
            summary.by_section[item.section.value] = (
                summary.by_section.get(item.section.value, 0) + 1
            )
            drive = item.drive or "unknown"
            summary.by_drive[drive] = summary.by_drive.get(drive, 0) + item.size_bytes
            summary.by_framework[item.framework] = (
                summary.by_framework.get(item.framework, 0) + item.size_bytes
            )

        summary.by_drive = dict(
            sorted(summary.by_drive.items(), key=lambda pair: -pair[1])
        )
        summary.by_framework = dict(
            sorted(summary.by_framework.items(), key=lambda pair: -pair[1])
        )

        if items:
            asset_ids = [item.asset_id for item in items]
            summary.physical_bytes = int(
                self.session.scalar(
                    select(func.coalesce(func.sum(Asset.physical_size_bytes), 0)).where(
                        Asset.id.in_(asset_ids)
                    )
                )
                or 0
            )

        return summary

    def _sort(
        self, items: list[InventoryItem], *, sort: str, descending: bool
    ) -> list[InventoryItem]:
        """Sort items by the requested field."""
        if sort == "name":
            # Names read best ascending, so the direction flag is inverted for them
            # unless the caller explicitly asked otherwise.
            return sorted(items, key=lambda item: item.name.lower(), reverse=not descending)

        if sort == "date":
            def date_key(item: InventoryItem) -> float:
                return item.modified_at.timestamp() if item.modified_at else 0.0

            return sorted(items, key=date_key, reverse=descending)

        if sort == "category":
            return sorted(
                items,
                key=lambda item: (order_of(item.category), -item.size_bytes),
            )

        if sort == "framework":
            return sorted(
                items,
                key=lambda item: (item.framework, -item.size_bytes),
            )

        if sort == "files":
            return sorted(items, key=lambda item: item.file_count, reverse=descending)

        return sorted(items, key=lambda item: item.size_bytes, reverse=descending)

    def _group(self, items: Sequence[InventoryItem], group_by: str) -> list[InventoryGroup]:
        """Bucket items by a field."""
        buckets: dict[str, InventoryGroup] = {}

        for item in items:
            key, label, order = _grouping_key(item, group_by)
            group = buckets.get(key)
            if group is None:
                group = InventoryGroup(key=key, label=label, order=order)
                buckets[key] = group
            group.items.append(item)

        return sorted(
            buckets.values(), key=lambda group: (group.order, -group.total_bytes)
        )

    # -- convenience --------------------------------------------------------

    def categories_present(self, *, include_missing: bool = False) -> list[CategoryCount]:
        """Return every category that currently holds at least one asset."""
        return self.build(include_missing=include_missing).summary.by_category

    def locate(self, name: str, *, limit: int = 20) -> list[InventoryItem]:
        """Return where assets matching a name are stored.

        A direct answer to "where did I put it?" without going through search.
        """
        needle = name.strip().lower()
        if not needle:
            return []
        report = self.build()
        matches = [
            item
            for item in report.items
            if needle in item.name.lower() or needle in item.path.lower()
        ]
        return matches[:limit]


def _parent_of(path: str) -> str:
    """Return the containing folder of an asset root."""
    parent = os.path.dirname(path.rstrip("\\/"))
    return parent or path


def _grouping_key(item: InventoryItem, group_by: str) -> tuple[str, str, int]:
    """Return ``(key, label, order)`` for an item under a grouping field."""
    if group_by == "category":
        return item.category.value, item.category_label, order_of(item.category)
    if group_by == "section":
        return item.section.value, item.section.value.title(), 0
    if group_by == "framework":
        return item.framework, item.framework.replace("_", " ").title(), 0
    if group_by == "drive":
        drive = item.drive or "unknown"
        return drive, drive, 0
    if group_by == "architecture":
        architecture = item.architecture or "unknown"
        return architecture, architecture, 0
    if group_by == "dataset_type":
        dataset_type = item.dataset_type or "—"
        return dataset_type, dataset_type.replace("_", " ").title(), 0
    if group_by == "format":
        return item.format, item.format.upper(), 0
    return "all", "All", 0
