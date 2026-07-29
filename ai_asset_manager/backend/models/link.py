"""Relationships between assets.

The catalogue answers "what do I have". The graph answers "what was this *for*" — which
project produced a checkpoint, which run produced it, which base model an adapter patches.
Those questions are the ones a person actually asks when deciding whether something can be
deleted, and none of them can be answered from a flat list.

Derived, never entered by hand: every edge is recomputed from what the scan found, so a
graph rebuilt after a rescan is correct rather than merely old. That is why edges carry the
rule that produced them — an edge nothing derives any more should disappear, and it can
only be recognised as stale if it says where it came from.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ai_asset_manager.backend.database.base import Base, utcnow


class LinkRelation(StrEnum):
    """How one asset relates to another.

    Directional, and the direction is always *from the dependent to what it depends on*:
    a checkpoint points at the run that produced it, not the other way round. Stated once
    here so queries never have to guess which way an edge runs.
    """

    #: Source lives inside the target project.
    BELONGS_TO = "belongs_to"
    #: Source was produced by the target training run.
    PRODUCED_BY = "produced_by"
    #: Source is an adapter for the target base model.
    ADAPTS = "adapts"
    #: Source is a quantised or converted form of the target.
    DERIVED_FROM = "derived_from"
    #: Source was trained or evaluated on the target dataset.
    TRAINED_ON = "trained_on"


class AssetLink(Base):
    """One directed edge between two assets."""

    __tablename__ = "asset_links"
    __table_args__ = (
        UniqueConstraint(
            "source_id", "target_id", "relation", name="uq_asset_links_edge"
        ),
        Index("ix_asset_links_target", "target_id", "relation"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    source_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), index=True
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), index=True
    )

    relation: Mapped[str] = mapped_column(String(32), index=True)
    #: 0..1. Containment is certain; a name-based match between a quantisation and its
    #: source is a guess, and the display should be able to tell them apart.
    confidence: Mapped[float] = mapped_column(default=1.0)
    #: The rule that derived this edge, so a stale one can be recognised and replaced.
    derived_by: Mapped[str] = mapped_column(String(64), default="")

    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<AssetLink {self.source_id} -{self.relation}-> {self.target_id}>"
