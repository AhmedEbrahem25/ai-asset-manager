"""Schema creation and first-run seeding."""

from __future__ import annotations

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session

# Importing the models package (not individual modules) guarantees every table is
# registered on Base.metadata before create_all runs.
from ai_asset_manager.backend.models import BUILTIN_TAGS, Base, Tag
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)


def init_database(engine: Engine, *, seed: bool = True) -> None:
    """Create any missing tables and seed built-in rows.

    Safe to call on every startup: ``create_all`` only issues DDL for tables that do not
    yet exist, and seeding is idempotent.

    Args:
        engine: Engine bound to the target database.
        seed: Whether to insert the built-in tag set.
    """
    existing = set(inspect(engine).get_table_names())
    Base.metadata.create_all(engine)
    created = set(inspect(engine).get_table_names()) - existing
    if created:
        logger.info("Created %d table(s): %s", len(created), ", ".join(sorted(created)))

    if seed:
        with Session(engine) as session:
            seed_builtin_tags(session)
            session.commit()


def seed_builtin_tags(session: Session) -> int:
    """Insert the built-in tags that are missing.

    Args:
        session: Open session; the caller owns the transaction.

    Returns:
        Number of tags inserted.
    """
    present = set(session.scalars(select(Tag.name)).all())
    inserted = 0
    for name, color, description in BUILTIN_TAGS:
        if name in present:
            continue
        session.add(Tag(name=name, color=color, description=description, is_builtin=True))
        inserted += 1
    if inserted:
        logger.debug("Seeded %d built-in tag(s)", inserted)
    return inserted


def has_fts5(engine: Engine) -> bool:
    """Report whether this SQLite build ships the FTS5 extension.

    Some distributions compile it out. The search layer falls back to ``LIKE`` scans when
    this returns ``False``, so the feature degrades instead of crashing at query time.
    """
    if engine.dialect.name != "sqlite":
        return False
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)"))
            conn.execute(text("DROP TABLE temp._fts5_probe"))
        except Exception:
            return False
    return True
