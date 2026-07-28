"""Search orchestration.

Combines the free-text backend with the structured filters from
:mod:`ai_asset_manager.backend.services.asset_service`, so one query string can say both
"about llama" and "on drive F, over 10 GB".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, select
from sqlalchemy.orm import Session, selectinload

from ai_asset_manager.backend.models import Asset, Tag
from ai_asset_manager.backend.search.backend import (
    SearchBackend,
    create_search_backend,
    order_by_ids,
)
from ai_asset_manager.backend.search.query import ParsedQuery, describe, parse_query
from ai_asset_manager.backend.services.asset_service import AssetService, Page
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Ceiling on ids pulled from the full-text index before structured filters narrow them.
#: Without a cap, a one-letter query would load the entire catalogue into an IN clause.
FTS_CANDIDATE_LIMIT = 2000


@dataclass(slots=True)
class SearchResult:
    """A page of search results, with the interpretation that produced it."""

    page: Page
    parsed: ParsedQuery
    backend: str
    #: How the query was understood, surfaced so a user can see why they got these rows.
    interpretation: dict[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> list[Asset]:
        """Return the matched assets."""
        return list(self.page.items)

    @property
    def total(self) -> int:
        """Return the total number of matches."""
        return self.page.total


class SearchService:
    """Runs combined full-text and structured searches."""

    def __init__(self, session: Session, backend: SearchBackend | None = None) -> None:
        """Initialise the service.

        Args:
            session: Open database session.
            backend: Search backend; selected from the session's engine when omitted.
        """
        self.session = session
        self.assets = AssetService(session)
        # get_bind() may hand back a Connection when the session is bound to one, as it
        # is inside a test transaction; the backend needs the Engine behind it.
        bind = session.get_bind()
        engine = bind.engine if isinstance(bind, Connection) else bind
        self.backend = backend or create_search_backend(engine)

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        sort: str = "relevance",
    ) -> SearchResult:
        """Run a search.

        Args:
            query: The raw query string.
            limit: Rows per page.
            offset: Rows to skip.
            sort: ``"relevance"`` to keep the full-text ranking, or any column accepted
                by :class:`AssetService`. Relevance only applies when free text was
                given; a purely structured query has nothing to rank by and falls back
                to size.

        Returns:
            A :class:`SearchResult`.
        """
        parsed = parse_query(query)

        if parsed.is_empty:
            return SearchResult(
                page=self.assets.list(limit=limit, offset=offset, sort="size"),
                parsed=parsed,
                backend=self.backend.name,
                interpretation=describe(parsed),
            )

        if parsed.excluded_tags:
            self._exclude_tagged(parsed)

        if not parsed.has_text:
            page = self.assets.list(
                parsed.filters,
                limit=limit,
                offset=offset,
                sort="size" if sort == "relevance" else sort,
            )
            return SearchResult(
                page=page,
                parsed=parsed,
                backend=self.backend.name,
                interpretation=describe(parsed),
            )

        ranked_ids = self.backend.search_ids(
            self.session, parsed.text, limit=FTS_CANDIDATE_LIMIT
        )

        if not ranked_ids:
            # The index found nothing. Fall through to the structured path so that
            # `type:llm nonsenseword` still reports zero rather than silently ignoring
            # the text and returning every LLM.
            page = Page(items=[], total=0, limit=limit, offset=offset)
            return SearchResult(
                page=page,
                parsed=parsed,
                backend=self.backend.name,
                interpretation=describe(parsed),
            )

        return self._page_from_ids(ranked_ids, parsed, limit=limit, offset=offset, sort=sort)

    def _page_from_ids(
        self,
        ranked_ids: list[int],
        parsed: ParsedQuery,
        *,
        limit: int,
        offset: int,
        sort: str,
    ) -> SearchResult:
        """Apply structured filters to a ranked id list and page the result."""
        # The text filter has already been satisfied by the index; leaving it on would
        # re-apply it as a LIKE and drop rows the index matched on a field the LIKE
        # clause does not cover.
        filters = parsed.filters
        filters.text = None

        statement = self.assets._apply_filters(
            select(Asset).where(Asset.id.in_(ranked_ids)), filters
        )
        matched = self.session.scalars(
            statement.options(selectinload(Asset.tags))
        ).all()

        if sort == "relevance":
            ordered = order_by_ids(matched, ranked_ids)
        else:
            reverse = sort != "name"
            ordered = sorted(
                matched,
                key=lambda asset: _sort_key(asset, sort),
                reverse=reverse,
            )

        window = ordered[offset : offset + limit]
        return SearchResult(
            page=Page(items=window, total=len(ordered), limit=limit, offset=offset),
            parsed=parsed,
            backend=self.backend.name,
            interpretation=describe(parsed),
        )

    def _exclude_tagged(self, parsed: ParsedQuery) -> None:
        """Remove assets carrying a tag the user negated, e.g. ``-tag:archived``."""
        excluded = self.session.scalars(
            select(Asset.id).join(Asset.tags).where(Tag.name.in_(parsed.excluded_tags))
        ).all()
        parsed.filters.excluded_ids.extend(excluded)

    def reindex(self) -> int:
        """Rebuild the full-text index."""
        count = self.backend.reindex(self.session)
        self.session.commit()
        return count

    def suggest(self, prefix: str, *, limit: int = 10) -> list[str]:
        """Return asset names beginning with a prefix, for type-ahead."""
        if not prefix.strip():
            return []
        rows = self.session.scalars(
            select(Asset.name)
            .where(Asset.name.ilike(f"{prefix}%"), Asset.is_missing.is_(False))
            .order_by(Asset.size_bytes.desc())
            .limit(limit)
        ).all()
        return list(dict.fromkeys(rows))


def _sort_key(asset: Asset, sort: str) -> Any:
    """Return the in-Python sort key for a column name.

    Sorting happens in Python here because the rows have already been narrowed to a
    ranked id set; re-issuing an ORDER BY would mean a second query for no benefit.
    """
    if sort == "name":
        return (asset.display_name or asset.name).lower()
    if sort == "modified":
        return asset.modified_at.timestamp() if asset.modified_at else 0.0
    if sort == "created":
        return asset.created_at.timestamp() if asset.created_at else 0.0
    if sort == "files":
        return asset.file_count
    return asset.size_bytes
