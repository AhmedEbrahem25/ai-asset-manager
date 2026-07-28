"""Asset querying.

Shared by the CLI and the HTTP API so both offer exactly the same filters and ordering.
Filtering lives here rather than in either caller, which is what keeps ``aam list`` and
``GET /models`` from drifting apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, selectinload

from ai_asset_manager.backend.models import (
    Asset,
    AssetFile,
    DatasetDetails,
    ModelDetails,
    Tag,
)
from ai_asset_manager.backend.models.enums import AssetKind

#: Columns exposed for ordering, mapped to their ORM attribute.
SORT_COLUMNS: dict[str, Any] = {
    "name": Asset.name,
    "size": Asset.size_bytes,
    "modified": Asset.modified_at,
    "created": Asset.created_at,
    "files": Asset.file_count,
    "kind": Asset.kind,
    "scanned": Asset.last_scanned,
}


@dataclass(slots=True)
class AssetFilter:
    """Filters applied to an asset query.

    Every field is optional; unset fields impose no restriction.
    """

    text: str | None = None
    kinds: list[str] = field(default_factory=list)
    model_types: list[str] = field(default_factory=list)
    dataset_formats: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    drives: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    quantizations: list[str] = field(default_factory=list)
    min_size: int | None = None
    max_size: int | None = None
    year: int | None = None
    health_status: str | None = None
    include_missing: bool = False
    #: Assets to exclude outright, used by negated search terms such as ``-tag:archived``.
    excluded_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Page:
    """One page of query results."""

    items: Sequence[Asset]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        """Report whether more results exist beyond this page."""
        return self.offset + len(self.items) < self.total


class AssetService:
    """Reads assets from the catalogue."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a session."""
        self.session = session

    def get(self, asset_id: int) -> Asset | None:
        """Return one asset by id, with its files eagerly loaded."""
        return self.session.scalar(
            select(Asset)
            .where(Asset.id == asset_id)
            .options(selectinload(Asset.files), selectinload(Asset.tags))
        )

    def get_by_path(self, root_path: str) -> Asset | None:
        """Return one asset by its root path."""
        return self.session.scalar(select(Asset).where(Asset.root_path == root_path))

    def list(
        self,
        filters: AssetFilter | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
        sort: str = "size",
        descending: bool = True,
    ) -> Page:
        """Return a filtered, sorted page of assets.

        Args:
            filters: Restrictions to apply.
            limit: Maximum rows to return.
            offset: Rows to skip.
            sort: Column name from :data:`SORT_COLUMNS`.
            descending: Sort direction.

        Returns:
            A :class:`Page` carrying the rows and the unpaginated total.
        """
        filters = filters or AssetFilter()
        statement = self._apply_filters(select(Asset), filters)

        total = self.session.scalar(
            self._apply_filters(select(func.count()).select_from(Asset), filters)
        ) or 0

        column = SORT_COLUMNS.get(sort, Asset.size_bytes)
        statement = statement.order_by(column.desc() if descending else column.asc())
        # A stable tiebreaker, so paging through equal-sized assets cannot repeat or skip.
        statement = statement.order_by(Asset.id.asc())

        rows = self.session.scalars(
            statement.limit(limit).offset(offset).options(selectinload(Asset.tags))
        ).all()
        return Page(items=rows, total=total, limit=limit, offset=offset)

    def _apply_filters(self, statement: Select[Any], filters: AssetFilter) -> Select[Any]:
        """Attach filter clauses to a statement."""
        if not filters.include_missing:
            statement = statement.where(Asset.is_missing.is_(False))
        if filters.excluded_ids:
            statement = statement.where(Asset.id.not_in(filters.excluded_ids))

        if filters.kinds:
            statement = statement.where(Asset.kind.in_(filters.kinds))
        if filters.formats:
            statement = statement.where(Asset.format.in_(filters.formats))
        if filters.frameworks:
            statement = statement.where(Asset.framework.in_(filters.frameworks))
        if filters.drives:
            statement = statement.where(Asset.drive.in_(filters.drives))
        if filters.health_status:
            statement = statement.where(Asset.health_status == filters.health_status)
        if filters.min_size is not None:
            statement = statement.where(Asset.size_bytes >= filters.min_size)
        if filters.max_size is not None:
            statement = statement.where(Asset.size_bytes <= filters.max_size)
        if filters.year is not None:
            statement = statement.where(
                func.strftime("%Y", Asset.modified_at) == str(filters.year)
            )

        # Detail-table filters use EXISTS-style correlated subqueries rather than joins,
        # so combining a model filter with a dataset filter cannot multiply rows.
        if filters.model_types:
            statement = statement.where(
                Asset.id.in_(
                    select(ModelDetails.asset_id).where(
                        ModelDetails.model_type.in_(filters.model_types)
                    )
                )
            )
        if filters.quantizations:
            statement = statement.where(
                Asset.id.in_(
                    select(ModelDetails.asset_id).where(
                        ModelDetails.quantization.in_(filters.quantizations)
                    )
                )
            )
        if filters.authors:
            statement = statement.where(
                Asset.id.in_(
                    select(ModelDetails.asset_id).where(
                        ModelDetails.author.in_(filters.authors)
                    )
                )
            )
        if filters.licenses:
            statement = statement.where(
                Asset.id.in_(
                    select(ModelDetails.asset_id).where(
                        ModelDetails.license.in_(filters.licenses)
                    )
                )
            )
        if filters.dataset_formats:
            statement = statement.where(
                Asset.id.in_(
                    select(DatasetDetails.asset_id).where(
                        DatasetDetails.dataset_format.in_(filters.dataset_formats)
                    )
                )
            )
        if filters.tags:
            statement = statement.where(
                Asset.id.in_(
                    select(Asset.id).join(Asset.tags).where(Tag.name.in_(filters.tags))
                )
            )

        if filters.text:
            pattern = f"%{filters.text.strip()}%"
            statement = statement.where(
                or_(
                    Asset.name.ilike(pattern),
                    Asset.display_name.ilike(pattern),
                    Asset.root_path.ilike(pattern),
                    Asset.id.in_(
                        select(ModelDetails.asset_id).where(
                            or_(
                                ModelDetails.repo_id.ilike(pattern),
                                ModelDetails.architecture.ilike(pattern),
                                ModelDetails.author.ilike(pattern),
                                ModelDetails.description.ilike(pattern),
                            )
                        )
                    ),
                )
            )

        return statement

    # -- aggregates ---------------------------------------------------------

    def counts_by_kind(self) -> dict[str, int]:
        """Return the number of catalogued assets per kind."""
        rows = self.session.execute(
            select(Asset.kind, func.count())
            .where(Asset.is_missing.is_(False))
            .group_by(Asset.kind)
        ).all()
        return {str(kind): int(count) for kind, count in rows}

    def total_size(self, *, physical: bool = False) -> int:
        """Return the total catalogued size in bytes.

        Args:
            physical: Count bytes actually occupied on disk, which excludes hardlinked
                and symlinked copies that share storage with another asset.
        """
        column = Asset.physical_size_bytes if physical else Asset.size_bytes
        return self.session.scalar(
            select(func.coalesce(func.sum(column), 0)).where(Asset.is_missing.is_(False))
        ) or 0

    def size_by(self, column_name: str) -> dict[str, int]:
        """Return total size grouped by a column such as ``drive`` or ``framework``."""
        column = {
            "drive": Asset.drive,
            "framework": Asset.framework,
            "format": Asset.format,
            "kind": Asset.kind,
        }.get(column_name, Asset.drive)

        rows = self.session.execute(
            select(column, func.coalesce(func.sum(Asset.size_bytes), 0), func.count())
            .where(Asset.is_missing.is_(False))
            .group_by(column)
            .order_by(func.sum(Asset.size_bytes).desc())
        ).all()
        return {str(key or "unknown"): int(total) for key, total, _count in rows}

    def largest(self, limit: int = 10) -> Sequence[Asset]:
        """Return the largest catalogued assets."""
        return self.session.scalars(
            select(Asset)
            .where(Asset.is_missing.is_(False))
            .order_by(Asset.size_bytes.desc())
            .limit(limit)
        ).all()

    def newest(self, limit: int = 10) -> Sequence[Asset]:
        """Return the most recently modified assets."""
        return self.session.scalars(
            select(Asset)
            .where(Asset.is_missing.is_(False), Asset.modified_at.is_not(None))
            .order_by(Asset.modified_at.desc())
            .limit(limit)
        ).all()

    def oldest(self, limit: int = 10) -> Sequence[Asset]:
        """Return the least recently modified assets."""
        return self.session.scalars(
            select(Asset)
            .where(Asset.is_missing.is_(False), Asset.modified_at.is_not(None))
            .order_by(Asset.modified_at.asc())
            .limit(limit)
        ).all()

    def file_count(self) -> int:
        """Return the total number of catalogued files."""
        return self.session.scalar(select(func.count()).select_from(AssetFile)) or 0

    def distinct_values(self, column_name: str) -> Sequence[str]:
        """Return the distinct values of a filterable column, for UI facets."""
        column = {
            "drive": Asset.drive,
            "framework": Asset.framework,
            "format": Asset.format,
            "kind": Asset.kind,
        }.get(column_name)
        if column is None:
            detail_column = {
                "model_type": ModelDetails.model_type,
                "quantization": ModelDetails.quantization,
                "author": ModelDetails.author,
                "license": ModelDetails.license,
                "dataset_format": DatasetDetails.dataset_format,
            }.get(column_name)
            if detail_column is None:
                return []
            rows = self.session.scalars(
                select(detail_column).distinct().where(detail_column.is_not(None))
            ).all()
        else:
            rows = self.session.scalars(
                select(column).distinct().where(column.is_not(None))
            ).all()
        return sorted(str(value) for value in rows if value)


def kind_label(kind: str) -> str:
    """Return a display label for an asset kind."""
    try:
        return AssetKind(kind).value.replace("_", " ").title()
    except ValueError:
        return kind.title()
