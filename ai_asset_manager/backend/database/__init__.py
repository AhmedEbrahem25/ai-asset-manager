"""Database engine and session management.

Deliberately does *not* re-export :mod:`ai_asset_manager.backend.database.schema`. Schema
creation needs the ORM, the ORM needs :class:`Base` from this package, and re-exporting it
here would make importing any model pull the whole cycle. Import it directly instead::

    from ai_asset_manager.backend.database.schema import init_database
"""

from __future__ import annotations

from ai_asset_manager.backend.database.base import Base, utcnow
from ai_asset_manager.backend.database.engine import (
    create_db_engine,
    get_db_session,
    get_engine,
    get_session_factory,
    reset_engine,
    session_scope,
)

__all__ = [
    "Base",
    "create_db_engine",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "reset_engine",
    "session_scope",
    "utcnow",
]
