"""Engine and session construction.

The scanner writes from worker threads while the API serves reads from the same file, so
the SQLite pragmas applied here are load-bearing rather than cosmetic.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _apply_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """Configure a freshly opened SQLite connection.

    - ``WAL`` lets the API read while a scan writes, instead of raising "database is
      locked" the moment both touch the file.
    - ``foreign_keys`` is OFF by default in SQLite, so cascades would silently not fire.
    - ``busy_timeout`` makes concurrent writers wait rather than fail immediately.
    """
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        # ~64 MiB page cache; the negative form means kibibytes rather than pages.
        cursor.execute("PRAGMA cache_size=-65536")
    finally:
        cursor.close()


def create_db_engine(settings: Settings | None = None, *, url: str | None = None) -> Engine:
    """Build a configured :class:`~sqlalchemy.Engine`.

    Args:
        settings: Settings to read connection options from; defaults to the singleton.
        url: Explicit override for the database URL, used by tests.

    Returns:
        A new engine. Callers that want the shared instance should use :func:`get_engine`.
    """
    settings = settings or get_settings()
    database_url = url or settings.resolved_database_url

    kwargs: dict[str, Any] = {"echo": settings.echo_sql, "future": True}

    if database_url.startswith("sqlite"):
        # check_same_thread=False is required because the scanner hands sessions between
        # worker threads; SQLAlchemy's pool still serialises access per connection.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15.0}
        if ":memory:" in database_url:
            # An in-memory database vanishes when its connection closes, so every session
            # in a test must share one connection.
            kwargs["poolclass"] = StaticPool
        else:
            settings.ensure_data_dir()
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20

    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite"):
        event.listen(engine, "connect", _apply_sqlite_pragmas)

    logger.debug("Created engine for %s", _redact(database_url))
    return engine


def _redact(url: str) -> str:
    """Strip credentials from a database URL before logging it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def configure_engine(url: str, *, settings: Settings | None = None) -> Engine:
    """Point the process-wide engine at a specific database and return it.

    Used by the CLI's ``--database`` flag and by tests. Building an engine without
    installing it here would leave :func:`session_scope` still talking to the default
    database, so the two must always be set together.
    """
    global _engine, _session_factory
    reset_engine()
    _engine = create_db_engine(settings, url=url)
    _session_factory = None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope.

    Commits on clean exit, rolls back on exception, and always closes.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Iterator[Session]:
    """Yield a session for FastAPI's dependency injection.

    FastAPI closes the generator after the response is sent, which is when the session is
    released back to the pool.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose of the cached engine and session factory.

    Used by tests and by any command that changes the configured database URL mid-process.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
