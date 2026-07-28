"""The Inventory Engine.

Answers one question: *what AI assets do I have, what are they for, and where are they?*

Strictly read-only and strictly database-bound. It opens no files, walks no directories
and parses no metadata — every value it reports was recorded by the scanner. That is what
makes it instant regardless of library size, and it is also why it can never disagree with
the catalogue: there is no second code path that could drift.

What it adds on top of the catalogue is judgement, and all of that comes from the plugin
taxonomy: the category an asset belongs on, the task it serves, the family it comes from,
the statistics worth showing and whether anything is missing. The engine itself knows no
model families and no dataset formats. It fetches rows, hands them to the taxonomy, and
shapes the answers.

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
    domain_label,
    label_of,
    order_of,
    section_label,
    section_of,
    section_order,
    task_label,
)
from ai_asset_manager.backend.inventory.profile import build_profile, load_file_summaries
from ai_asset_manager.backend.models import Asset, DatasetDetails, ModelDetails
from ai_asset_manager.backend.taxonomy import HealthReport, TaxonomyRegistry, default_registry
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Fields a report may be grouped by.
GROUP_BY_FIELDS = ("category", "section", "task", "domain", "family", "framework",
                   "drive", "architecture", "dataset_type", "format", "health")

#: Fields a report may be sorted by.
SORT_FIELDS = ("name", "size", "date", "category", "framework", "files", "health", "task")


@dataclass(slots=True)
class InventoryItem:
    """One asset as it appears in the inventory."""

    asset_id: int
    name: str
    #: Registered category id. A plain string so plugins can introduce their own.
    category: str
    #: Registered section id.
    section: str
    subcategory: str | None
    framework: str
    format: str
    size_bytes: int
    file_count: int
    path: str
    root_folder: str
    drive: str | None
    modified_at: datetime | None

    # -- what the taxonomy concluded ---------------------------------------
    task: str | None = None
    domain: str | None = None
    family: str | None = None
    modalities: tuple[str, ...] = ()
    confidence: float = 1.0
    #: Why it was classified this way, for when the answer is surprising.
    evidence: str = ""

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

    #: Plugin-contributed statistics, keyed by whatever the plugin chose.
    stats: dict[str, Any] = field(default_factory=dict)
    health: HealthReport | None = None

    tags: list[str] = field(default_factory=list)
    is_missing: bool = False

    @property
    def category_label(self) -> str:
        """Return the human-readable category name."""
        return label_of(self.category)

    @property
    def section_label(self) -> str:
        """Return the human-readable section name."""
        return section_label(self.section)

    @property
    def task_label(self) -> str:
        """Return the human-readable task name, or an empty string."""
        return task_label(self.task)

    @property
    def domain_label(self) -> str:
        """Return the human-readable domain name, or an empty string."""
        return domain_label(self.domain)

    @property
    def is_model(self) -> bool:
        """Report whether this item belongs on a model shelf."""
        return self.section == "models"

    @property
    def is_dataset(self) -> bool:
        """Report whether this item belongs on a dataset shelf."""
        return self.section == "datasets"

    @property
    def health_score(self) -> int | None:
        """Return the 0-100 health score, or ``None`` when health was not evaluated."""
        if self.health is None or not self.health.evaluated:
            return None
        return self.health.score

    @property
    def health_status(self) -> str:
        """Return ``ok``, ``warning``, ``error`` or ``unknown``."""
        return self.health.status if self.health else "unknown"

    @property
    def is_incomplete(self) -> bool:
        """Report whether anything above informational was found wrong with this asset."""
        return self.health_status in ("warning", "error")

    def stat(self, key: str, default: Any = None) -> Any:
        """Return one plugin-contributed statistic."""
        return self.stats.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Used by the export layer and, later, by the REST API, so that both describe an
        inventory item identically.
        """
        payload: dict[str, Any] = {
            "id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "category_label": self.category_label,
            "section": self.section,
            "subcategory": self.subcategory,
            "task": self.task,
            "task_label": self.task_label,
            "domain": self.domain,
            "family": self.family,
            "modalities": list(self.modalities),
            "framework": self.framework,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "path": self.path,
            "root_folder": self.root_folder,
            "drive": self.drive,
            "last_modified": self.modified_at.isoformat() if self.modified_at else None,
            "tags": self.tags,
            "statistics": self.stats,
        }

        if self.health is not None and self.health.evaluated:
            payload["health"] = {
                "score": self.health.score,
                "status": self.health.status,
                "findings": [
                    {
                        "code": finding.code,
                        "severity": finding.severity.value,
                        "message": finding.message,
                        "fix_hint": finding.fix_hint,
                    }
                    for finding in self.health.findings
                ],
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

    category: str
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
    by_task: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, int] = field(default_factory=dict)
    by_family: dict[str, int] = field(default_factory=dict)
    by_drive: dict[str, int] = field(default_factory=dict)
    by_framework: dict[str, int] = field(default_factory=dict)
    missing_assets: int = 0
    #: Assets carrying at least one warning or error.
    unhealthy_assets: int = 0
    #: Mean health score across everything that was evaluated, or ``None``.
    average_health: int | None = None

    def category_count(self, category: str) -> int:
        """Return how many assets fall in a category."""
        for entry in self.by_category:
            if entry.category == category:
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
                "unhealthy_assets": self.summary.unhealthy_assets,
                "average_health": self.summary.average_health,
                "by_category": [
                    {
                        "category": entry.category,
                        "label": entry.label,
                        "count": entry.count,
                        "total_bytes": entry.total_bytes,
                    }
                    for entry in self.summary.by_category
                ],
                "by_section": self.summary.by_section,
                "by_task": self.summary.by_task,
                "by_domain": self.summary.by_domain,
                "by_family": self.summary.by_family,
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

    def __init__(self, session: Session, registry: TaxonomyRegistry | None = None) -> None:
        """Bind the engine to a database session and a taxonomy.

        Args:
            session: Open database session.
            registry: Taxonomy to classify with. Defaults to the process-wide one; tests
                pass their own to prove that a plugin can extend the inventory without
                touching this class.
        """
        self.session = session
        self.registry = registry or default_registry()

    def build(
        self,
        categories: Sequence[str] | None = None,
        *,
        drives: Sequence[str] | None = None,
        frameworks: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        group_by: str | None = None,
        sort: str = "size",
        descending: bool = True,
        include_missing: bool = False,
        only_unhealthy: bool = False,
        limit: int | None = None,
    ) -> InventoryReport:
        """Build an inventory report.

        Args:
            categories: Restrict to these category ids; ``None`` means everything.
            drives: Restrict to these drives, e.g. ``["F:"]``.
            frameworks: Restrict to these frameworks.
            tasks: Restrict to these task ids.
            domains: Restrict to these domain ids.
            group_by: One of :data:`GROUP_BY_FIELDS`, or ``None`` for a flat list.
            sort: One of :data:`SORT_FIELDS`.
            descending: Sort direction.
            include_missing: Include assets no longer present on disk.
            only_unhealthy: Keep only assets with a warning or an error.
            limit: Cap the number of items returned. The summary still reflects
                everything that matched, so a truncated table never misreports totals.

        Returns:
            The completed :class:`InventoryReport`.
        """
        rows = self._fetch(drives=drives, frameworks=frameworks, include_missing=include_missing)

        # The file list is loaded for every matched asset, not only when a detailed view
        # asks for it. Classification depends on it — a training run is recognised by its
        # event files — and an asset must not change category depending on which command
        # you happened to run. It is one indexed query, and it is what dataset
        # intelligence is built from.
        summaries = load_file_summaries(self.session, [row[0].id for row in rows])
        items = [self._to_item(row, summaries) for row in rows]

        if categories is not None:
            wanted = set(categories)
            items = [item for item in items if item.category in wanted]
        if tasks:
            wanted_tasks = set(tasks)
            items = [item for item in items if item.task in wanted_tasks]
        if domains:
            wanted_domains = set(domains)
            items = [item for item in items if item.domain in wanted_domains]
        if only_unhealthy:
            items = [item for item in items if item.is_incomplete]

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
                "categories": list(categories) if categories else None,
                "drives": list(drives) if drives else None,
                "frameworks": list(frameworks) if frameworks else None,
                "tasks": list(tasks) if tasks else None,
                "domains": list(domains) if domains else None,
                "group_by": group_by,
                "sort": sort,
                "descending": descending,
                "include_missing": include_missing,
                "only_unhealthy": only_unhealthy,
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

    def _to_item(self, row: Any, summaries: dict[int, Any]) -> InventoryItem:
        """Convert one joined row into a classified, measured inventory item."""
        asset: Asset = row[0]
        model: ModelDetails | None = row[1]
        dataset: DatasetDetails | None = row[2]

        profile = build_profile(asset, model, dataset, summaries.get(asset.id))
        verdict = self.registry.classify(profile)

        item = InventoryItem(
            asset_id=asset.id,
            name=asset.display_name or asset.name,
            category=verdict.category,
            section=section_of(verdict.category),
            subcategory=asset.subkind,
            framework=asset.framework,
            format=asset.format,
            size_bytes=asset.size_bytes,
            file_count=asset.file_count,
            path=asset.root_path,
            root_folder=_parent_of(asset.root_path),
            drive=asset.drive,
            modified_at=asset.modified_at,
            task=verdict.task,
            domain=verdict.domain,
            family=verdict.family,
            modalities=verdict.modalities,
            confidence=verdict.confidence,
            evidence=verdict.evidence,
            stats=self.registry.statistics(profile),
            health=self.registry.check_health(profile),
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

        # Statistics are the richer source for datasets whose counts the scanner never
        # recorded — a plain image folder has no manifest to parse, so the file tally is
        # the only number there is.
        item.num_images = item.num_images or int(item.stat("images", 0) or 0)
        item.num_videos = item.num_videos or int(item.stat("videos", 0) or 0)

        return item

    # -- shaping ------------------------------------------------------------

    def _summarise(self, items: Sequence[InventoryItem]) -> InventorySummary:
        """Compute headline totals over the matched items."""
        summary = InventorySummary(
            total_assets=len(items),
            total_bytes=sum(item.size_bytes for item in items),
            total_files=sum(item.file_count for item in items),
            missing_assets=sum(1 for item in items if item.is_missing),
            unhealthy_assets=sum(1 for item in items if item.is_incomplete),
        )

        per_category: dict[str, list[InventoryItem]] = defaultdict(list)
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
            summary.by_section[item.section] = summary.by_section.get(item.section, 0) + 1
            if item.task:
                summary.by_task[item.task] = summary.by_task.get(item.task, 0) + 1
            if item.domain:
                summary.by_domain[item.domain] = summary.by_domain.get(item.domain, 0) + 1
            if item.family:
                summary.by_family[item.family] = summary.by_family.get(item.family, 0) + 1
            drive = item.drive or "unknown"
            summary.by_drive[drive] = summary.by_drive.get(drive, 0) + item.size_bytes
            summary.by_framework[item.framework] = (
                summary.by_framework.get(item.framework, 0) + item.size_bytes
            )

        summary.by_task = _sorted_by_value(summary.by_task)
        summary.by_domain = _sorted_by_value(summary.by_domain)
        summary.by_family = _sorted_by_value(summary.by_family)
        summary.by_drive = _sorted_by_value(summary.by_drive)
        summary.by_framework = _sorted_by_value(summary.by_framework)

        scores = [item.health_score for item in items if item.health_score is not None]
        if scores:
            summary.average_health = round(sum(scores) / len(scores))

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

        if sort == "task":
            return sorted(
                items,
                key=lambda item: (item.task_label or "~", -item.size_bytes),
            )

        if sort == "framework":
            return sorted(
                items,
                key=lambda item: (item.framework, -item.size_bytes),
            )

        if sort == "files":
            return sorted(items, key=lambda item: item.file_count, reverse=descending)

        if sort == "health":
            # Worst first regardless of direction: a health listing exists to surface
            # problems, and burying them under the healthy assets defeats it.
            return sorted(
                items,
                key=lambda item: (item.health_score if item.health_score is not None else 101,
                                  -item.size_bytes),
            )

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


def _sorted_by_value(mapping: dict[str, int]) -> dict[str, int]:
    """Return a mapping ordered by descending value, then by key."""
    return dict(sorted(mapping.items(), key=lambda pair: (-pair[1], pair[0])))


def _grouping_key(item: InventoryItem, group_by: str) -> tuple[str, str, int]:
    """Return ``(key, label, order)`` for an item under a grouping field."""
    if group_by == "category":
        return item.category, item.category_label, order_of(item.category)
    if group_by == "section":
        return item.section, item.section_label, section_order(item.section)
    if group_by == "task":
        return item.task or "unknown", item.task_label or "Unknown task", 0
    if group_by == "domain":
        return item.domain or "unknown", item.domain_label or "Unknown domain", 0
    if group_by == "family":
        return item.family or "unknown", item.family or "Unknown family", 0
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
    if group_by == "health":
        status = item.health_status
        # Worst first: this grouping exists to put the broken assets at the top.
        order = {"error": 0, "warning": 1, "unknown": 2, "ok": 3}.get(status, 4)
        return status, status.title(), order
    return "all", "All", 0
