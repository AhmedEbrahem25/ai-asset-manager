"""Health finding table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_asset_manager.backend.database.base import Base, utcnow
from ai_asset_manager.backend.models.enums import Severity

if TYPE_CHECKING:
    from ai_asset_manager.backend.models.asset import Asset


class HealthFinding(Base):
    """A single problem detected on an asset.

    Findings are replaced wholesale each time an asset is health-checked, so ``code``
    plus ``asset_id`` is effectively unique within a check run without needing a
    constraint that would fight the delete-then-insert refresh.
    """

    __tablename__ = "health_findings"
    __table_args__ = (
        Index("ix_health_asset_severity", "asset_id", "severity"),
        Index("ix_health_code", "code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), index=True
    )

    #: Stable machine-readable identifier, e.g. ``"model.missing_config"``.
    code: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16), default=Severity.WARNING, index=True)
    message: Mapped[str] = mapped_column(Text)
    #: Actionable remediation shown next to the finding in the dashboard.
    fix_hint: Mapped[str | None] = mapped_column(Text, default=None)
    #: Paths or values that triggered the rule, for drill-down.
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    detected_at: Mapped[datetime] = mapped_column(default=utcnow)

    asset: Mapped[Asset] = relationship(back_populates="health_findings")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<HealthFinding {self.code} {self.severity} asset={self.asset_id}>"
