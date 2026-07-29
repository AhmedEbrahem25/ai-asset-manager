"""Full-text search backends.

Two implementations behind one protocol: SQLite FTS5 where available, and a ``LIKE``
scan where it is not. The fallback matters — some Python distributions ship SQLite
without FTS5 compiled in, and a catalogue that refuses to search on those builds would
be broken through no fault of the user.

The protocol also leaves room for a PostgreSQL ``tsvector`` backend without touching any
caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import Asset
from ai_asset_manager.backend.search.query import build_fts_match
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Name of the FTS5 virtual table.
FTS_TABLE = "assets_fts"

#: Columns indexed for full-text search, in the order they are inserted.
FTS_COLUMNS = ("name", "display_name", "repo_id", "author", "architecture",
               "description", "tags", "path")

#: The virtual table. `content=''` makes it contentless: FTS5 stores only the index and
#: we keep `rowid` aligned with `assets.id`, which avoids duplicating every description
#: into a shadow table.
_CREATE_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
    {", ".join(FTS_COLUMNS)},
    content='',
    tokenize='unicode61 remove_diacritics 2'
)
"""


@runtime_checkable
class SearchBackend(Protocol):
    """Finds asset ids matching a free-text query."""

    name: str

    def is_available(self) -> bool:
        """Report whether this backend can be used on the current database."""
        ...

    def reindex(self, session: Session) -> int:
        """Rebuild the index from the catalogue. Returns the number of rows indexed."""
        ...

    def search_ids(self, session: Session, query: str, *, limit: int = 500) -> list[int]:
        """Return matching asset ids, best match first."""
        ...


class Fts5Backend:
    """SQLite FTS5 search.

    The index is rebuilt rather than incrementally maintained by triggers. Triggers would
    have to fire on ``assets``, ``model_details``, ``dataset_details`` and the tag
    association table to keep one denormalised row correct, and a scan rewrites all four;
    a rebuild after a scan is both simpler and faster than 4N trigger invocations.
    """

    name = "fts5"

    def __init__(self, engine: Engine) -> None:
        """Bind the backend to an engine."""
        self.engine = engine
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Report whether this SQLite build provides FTS5."""
        if self._available is not None:
            return self._available

        if self.engine.dialect.name != "sqlite":
            self._available = False
            return False

        with self.engine.connect() as connection:
            try:
                connection.execute(text("CREATE VIRTUAL TABLE temp._fts_probe USING fts5(x)"))
                connection.execute(text("DROP TABLE temp._fts_probe"))
                self._available = True
            except Exception:
                logger.info("SQLite build has no FTS5; falling back to LIKE search")
                self._available = False
        return self._available

    def ensure_table(self, session: Session) -> None:
        """Create the virtual table if it does not exist."""
        session.execute(text(_CREATE_FTS))

    def reindex(self, session: Session) -> int:
        """Rebuild the whole index from the catalogue."""
        if not self.is_available():
            return 0

        self.ensure_table(session)
        session.execute(text(f"DELETE FROM {FTS_TABLE}"))

        rows = session.execute(
            text(
                """
                SELECT a.id,
                       a.name,
                       COALESCE(a.display_name, ''),
                       COALESCE(m.repo_id, d.repo_id, ''),
                       COALESCE(m.author, ''),
                       COALESCE(m.architecture, d.dataset_format, ''),
                       COALESCE(m.description, d.description, ''),
                       COALESCE((
                           SELECT GROUP_CONCAT(t.name, ' ')
                           FROM asset_tags at JOIN tags t ON t.id = at.tag_id
                           WHERE at.asset_id = a.id
                       ), ''),
                       a.root_path
                FROM assets a
                LEFT JOIN model_details m ON m.asset_id = a.id
                LEFT JOIN dataset_details d ON d.asset_id = a.id
                """
            )
        ).all()

        placeholders = ", ".join(f":{column}" for column in FTS_COLUMNS)
        statement = text(
            f"INSERT INTO {FTS_TABLE} (rowid, {', '.join(FTS_COLUMNS)}) "
            f"VALUES (:rowid, {placeholders})"
        )

        for row in rows:
            session.execute(
                statement,
                {"rowid": row[0], **dict(zip(FTS_COLUMNS, row[1:], strict=True))},
            )

        logger.debug("Reindexed %d asset(s) for full-text search", len(rows))
        return len(rows)

    def search_ids(self, session: Session, query: str, *, limit: int = 500) -> list[int]:
        """Return asset ids matching the query, ranked by FTS5's relevance score."""
        if not self.is_available():
            return []

        match = build_fts_match(query)
        if not match:
            return []

        self.ensure_table(session)
        try:
            rows = session.execute(
                text(
                    f"SELECT rowid FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :match "
                    "ORDER BY rank LIMIT :limit"
                ),
                {"match": match, "limit": limit},
            ).all()
        except Exception as exc:
            logger.warning("Full-text query %r failed: %s", match, exc)
            return []

        return [row[0] for row in rows]


class LikeBackend:
    """Fallback search using ``LIKE`` scans.

    Slower and unranked, but correct, and it keeps the feature working on SQLite builds
    without FTS5 and on any other dialect.
    """

    name = "like"

    def __init__(self, engine: Engine) -> None:
        """Bind the backend to an engine."""
        self.engine = engine

    def is_available(self) -> bool:
        """Always available."""
        return True

    def reindex(self, session: Session) -> int:
        """No index to maintain."""
        return 0

    def search_ids(self, session: Session, query: str, *, limit: int = 500) -> list[int]:
        """Return asset ids whose indexed text contains every term."""
        terms = [term for term in query.split() if term]
        if not terms:
            return []

        statement = """
            SELECT a.id FROM assets a
            LEFT JOIN model_details m ON m.asset_id = a.id
            LEFT JOIN dataset_details d ON d.asset_id = a.id
            WHERE 1=1
        """
        params: dict[str, object] = {"limit": limit}
        for index, term in enumerate(terms):
            key = f"term{index}"
            params[key] = f"%{term}%"
            statement += f"""
                AND (
                    a.name LIKE :{key} OR a.display_name LIKE :{key}
                    OR a.root_path LIKE :{key}
                    OR m.repo_id LIKE :{key} OR m.author LIKE :{key}
                    OR m.architecture LIKE :{key} OR m.description LIKE :{key}
                    OR d.dataset_format LIKE :{key} OR d.description LIKE :{key}
                )
            """
        statement += " ORDER BY a.size_bytes DESC LIMIT :limit"

        rows = session.execute(text(statement), params).all()
        return [row[0] for row in rows]


def create_search_backend(engine: Engine) -> SearchBackend:
    """Return the best search backend available for this database."""
    fts = Fts5Backend(engine)
    return fts if fts.is_available() else LikeBackend(engine)


def order_by_ids(assets: Sequence[Asset], ids: Sequence[int]) -> list[Asset]:
    """Reorder loaded assets to match a ranked id list.

    A SQL ``IN`` clause discards ordering, so the relevance ranking the backend computed
    has to be reapplied after loading.
    """
    position = {asset_id: index for index, asset_id in enumerate(ids)}
    return sorted(assets, key=lambda asset: position.get(asset.id, len(position)))
