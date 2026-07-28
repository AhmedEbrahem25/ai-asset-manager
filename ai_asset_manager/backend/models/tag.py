"""User-applied tags and the asset association table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, ForeignKey, String, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_asset_manager.backend.database.base import Base, utcnow

if TYPE_CHECKING:
    from ai_asset_manager.backend.models.asset import Asset

#: Association table. Declared as a Core table rather than an ORM class because it
#: carries no attributes of its own; the CASCADE keeps rows from outliving either side.
asset_tags = Table(
    "asset_tags",
    Base.metadata,
    Column("asset_id", ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class Tag(Base):
    """A user-defined label applicable to any asset."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #: Hex colour used by the dashboard chips, e.g. ``"#f59e0b"``.
    color: Mapped[str | None] = mapped_column(String(16), default=None)
    description: Mapped[str | None] = mapped_column(String(256), default=None)
    #: Built-in tags are seeded on first run and cannot be deleted, only unassigned.
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    assets: Mapped[list[Asset]] = relationship(secondary=asset_tags, back_populates="tags")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<Tag {self.name!r}>"


#: Seeded on first run. Colours are chosen for AA contrast on the dashboard's dark surface.
BUILTIN_TAGS: tuple[tuple[str, str, str], ...] = (
    ("Favorite", "#f59e0b", "Assets you reach for most often"),
    ("Archived", "#64748b", "Kept for reference, not in active use"),
    ("Research", "#8b5cf6", "Experimental or paper-related"),
    ("Production", "#10b981", "Serving live workloads"),
    ("Training", "#3b82f6", "Currently used for training runs"),
    ("Benchmark", "#ec4899", "Reserved for evaluation"),
    ("Need Download", "#ef4444", "Incomplete; re-download required"),
)
