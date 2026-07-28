"""Scan root registration and scan run history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ai_asset_manager.backend.database.base import Base, utcnow
from ai_asset_manager.backend.models.enums import ScanStatus


class ScanRoot(Base):
    """A folder the user has registered for scanning.

    Roots are unlimited and may live on different drives; each is scanned independently
    so one unplugged external drive cannot fail the whole run.
    """

    __tablename__ = "scan_roots"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    label: Mapped[str | None] = mapped_column(String(128), default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    #: Additional glob patterns excluded for this root only.
    exclude_globs: Mapped[list[str]] = mapped_column(JSON, default=list)

    last_scanned: Mapped[datetime | None] = mapped_column(default=None)
    last_asset_count: Mapped[int] = mapped_column(Integer, default=0)
    added_at: Mapped[datetime] = mapped_column(default=utcnow)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<ScanRoot {self.path!r} enabled={self.enabled}>"


class ScanRun(Base):
    """History and live state of one scan invocation.

    The API's progress stream reads this row, so it is updated during the scan rather
    than only written at the end.
    """

    __tablename__ = "scan_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default=ScanStatus.RUNNING, index=True)
    roots: Mapped[list[str]] = mapped_column(JSON, default=list)

    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    duration_seconds: Mapped[float | None] = mapped_column(default=None)

    files_seen: Mapped[int] = mapped_column(BigInteger, default=0)
    bytes_seen: Mapped[int] = mapped_column(BigInteger, default=0)
    directories_seen: Mapped[int] = mapped_column(BigInteger, default=0)

    assets_found: Mapped[int] = mapped_column(Integer, default=0)
    assets_created: Mapped[int] = mapped_column(Integer, default=0)
    assets_updated: Mapped[int] = mapped_column(Integer, default=0)
    #: Assets whose fingerprint was unchanged, so detection and parsing were skipped.
    assets_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    assets_missing: Mapped[int] = mapped_column(Integer, default=0)

    error_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Bounded sample of errors; the full set goes to the log file.
    errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    message: Mapped[str | None] = mapped_column(Text, default=None)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<ScanRun {self.id} {self.status} assets={self.assets_found}>"
