"""Duplicate grouping tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_asset_manager.backend.database.base import Base, utcnow
from ai_asset_manager.backend.models.enums import DuplicateKind


class DuplicateGroup(Base):
    """A set of assets or files judged to hold the same content.

    ``wasted_bytes`` counts only physically distinct extents: hardlinked or symlinked
    copies share storage and are not waste, so reporting them as such would send the user
    deleting files that free nothing.
    """

    __tablename__ = "duplicate_groups"
    __table_args__ = (Index("ix_dupe_groups_kind_wasted", "kind", "wasted_bytes"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    #: Content hash shared by the group's members, or a composite hash for asset groups.
    group_hash: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), default=DuplicateKind.FILE, index=True)

    member_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Size of one copy.
    unit_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    #: Reclaimable bytes: total occupied minus the single copy that is kept.
    wasted_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    #: Jaccard-style overlap of member file hashes; 1.0 for exact groups.
    similarity: Mapped[float] = mapped_column(default=1.0)

    #: Suggested copy to keep, chosen by the resolution policy (newest, then largest).
    keeper_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL"), default=None
    )
    detected_at: Mapped[datetime] = mapped_column(default=utcnow)

    members: Mapped[list[DuplicateMember]] = relationship(
        back_populates="group", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<DuplicateGroup {self.kind} n={self.member_count} waste={self.wasted_bytes}>"


class DuplicateMember(Base):
    """One member of a :class:`DuplicateGroup`."""

    __tablename__ = "duplicate_members"
    __table_args__ = (Index("ix_dupe_members_asset", "asset_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("duplicate_groups.id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), default=None
    )
    file_id: Mapped[int | None] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), default=None
    )

    path: Mapped[str] = mapped_column(String(1024))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    #: False when this member shares storage with another member (hardlink/symlink), in
    #: which case deleting it reclaims nothing.
    occupies_own_storage: Mapped[bool] = mapped_column(default=True)
    is_keeper: Mapped[bool] = mapped_column(default=False)

    group: Mapped[DuplicateGroup] = relationship(back_populates="members")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<DuplicateMember {self.path!r}>"
