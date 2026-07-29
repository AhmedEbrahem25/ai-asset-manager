"""Core catalogue tables: assets, their format-specific details, and their files."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_asset_manager.backend.database.base import Base, utcnow
from ai_asset_manager.backend.models.enums import (
    AssetFormat,
    AssetKind,
    DatasetFormat,
    Framework,
    HealthStatus,
    ModelType,
    Precision,
)

if TYPE_CHECKING:
    from ai_asset_manager.backend.models.health import HealthFinding
    from ai_asset_manager.backend.models.tag import Tag


class Asset(Base):
    """A single discovered asset: a model directory, a dataset tree or a lone weight file.

    ``root_path`` is the natural key. Rescans upsert on it, which is what lets the
    catalogue survive a database rebuild without losing user tags.
    """

    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_kind_subkind", "kind", "subkind"),
        Index("ix_assets_drive_kind", "drive", "kind"),
        Index("ix_assets_size", "size_bytes"),
        Index("ix_assets_fingerprint", "fingerprint"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # -- identity -----------------------------------------------------------
    kind: Mapped[str] = mapped_column(String(32), default=AssetKind.UNKNOWN, index=True)
    subkind: Mapped[str | None] = mapped_column(String(64), default=None)
    name: Mapped[str] = mapped_column(String(512), index=True)
    display_name: Mapped[str | None] = mapped_column(String(512), default=None)

    #: Absolute, normalised path to the asset root. Unique across the catalogue.
    root_path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    drive: Mapped[str | None] = mapped_column(String(16), default=None, index=True)
    is_single_file: Mapped[bool] = mapped_column(Boolean, default=False)

    # -- storage ------------------------------------------------------------
    #: Sum of the apparent size of every file in the asset.
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    #: Bytes actually occupied on disk, counting each shared extent (hardlink or
    #: resolved symlink) exactly once. Diverges from ``size_bytes`` in linked caches.
    physical_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)

    # -- classification -----------------------------------------------------
    format: Mapped[str] = mapped_column(String(32), default=AssetFormat.UNKNOWN, index=True)
    framework: Mapped[str] = mapped_column(String(32), default=Framework.UNKNOWN, index=True)

    # -- filesystem timestamps ---------------------------------------------
    created_at: Mapped[datetime | None] = mapped_column(default=None)
    modified_at: Mapped[datetime | None] = mapped_column(default=None, index=True)
    accessed_at: Mapped[datetime | None] = mapped_column(default=None)

    # -- catalogue bookkeeping ---------------------------------------------
    #: blake2b over the sorted (relpath, size, mtime) triples of the asset's files.
    #: An unchanged fingerprint lets a rescan skip detection and parsing entirely.
    fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    health_status: Mapped[str] = mapped_column(
        String(16), default=HealthStatus.UNKNOWN, index=True
    )
    detection_confidence: Mapped[float] = mapped_column(default=1.0)
    detector: Mapped[str | None] = mapped_column(String(64), default=None)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    first_seen: Mapped[datetime] = mapped_column(default=utcnow)
    last_scanned: Mapped[datetime] = mapped_column(default=utcnow)
    is_missing: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # -- relationships ------------------------------------------------------
    model_details: Mapped[ModelDetails | None] = relationship(
        back_populates="asset", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )
    dataset_details: Mapped[DatasetDetails | None] = relationship(
        back_populates="asset", cascade="all, delete-orphan", uselist=False, lazy="selectin"
    )
    files: Mapped[list[AssetFile]] = relationship(
        back_populates="asset", cascade="all, delete-orphan", lazy="select"
    )
    health_findings: Mapped[list[HealthFinding]] = relationship(
        back_populates="asset", cascade="all, delete-orphan", lazy="selectin"
    )
    tags: Mapped[list[Tag]] = relationship(
        secondary="asset_tags", back_populates="assets", lazy="selectin"
    )

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<Asset {self.id} {self.kind}:{self.name!r} @ {self.root_path!r}>"


class ModelDetails(Base):
    """Model-specific metadata, stored 1:1 alongside its :class:`Asset`."""

    __tablename__ = "model_details"
    __table_args__ = (Index("ix_model_details_repo", "repo_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), unique=True, index=True
    )

    model_type: Mapped[str] = mapped_column(String(32), default=ModelType.UNKNOWN, index=True)
    architecture: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    #: Exact parameter count when derivable from tensor shapes, else an estimate.
    param_count: Mapped[int | None] = mapped_column(BigInteger, default=None)
    param_count_is_exact: Mapped[bool] = mapped_column(Boolean, default=False)
    quantization: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    precision: Mapped[str] = mapped_column(String(16), default=Precision.UNKNOWN)
    context_length: Mapped[int | None] = mapped_column(Integer, default=None)
    hidden_size: Mapped[int | None] = mapped_column(Integer, default=None)
    num_layers: Mapped[int | None] = mapped_column(Integer, default=None)
    vocab_size: Mapped[int | None] = mapped_column(Integer, default=None)
    tensor_count: Mapped[int | None] = mapped_column(Integer, default=None)

    # -- provenance ---------------------------------------------------------
    repo_id: Mapped[str | None] = mapped_column(String(256), default=None)
    revision: Mapped[str | None] = mapped_column(String(64), default=None)
    author: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    license: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    base_model: Mapped[str | None] = mapped_column(String(256), default=None)
    pipeline_tag: Mapped[str | None] = mapped_column(String(64), default=None)
    library_name: Mapped[str | None] = mapped_column(String(64), default=None)
    #: Free-form tags declared by the model card, distinct from user-applied tags.
    card_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    asset: Mapped[Asset] = relationship(back_populates="model_details")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<ModelDetails asset={self.asset_id} arch={self.architecture!r}>"


class DatasetDetails(Base):
    """Dataset-specific metadata, stored 1:1 alongside its :class:`Asset`."""

    __tablename__ = "dataset_details"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), unique=True, index=True
    )

    dataset_format: Mapped[str] = mapped_column(
        String(32), default=DatasetFormat.UNKNOWN, index=True
    )
    task: Mapped[str | None] = mapped_column(String(64), default=None, index=True)

    num_images: Mapped[int] = mapped_column(Integer, default=0)
    num_videos: Mapped[int] = mapped_column(Integer, default=0)
    num_audio_files: Mapped[int] = mapped_column(Integer, default=0)
    num_text_files: Mapped[int] = mapped_column(Integer, default=0)
    num_annotations: Mapped[int] = mapped_column(BigInteger, default=0)
    num_classes: Mapped[int | None] = mapped_column(Integer, default=None)

    class_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    #: Mapping of split name to item count, e.g. ``{"train": 118287, "val": 5000}``.
    splits: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    modalities: Mapped[list[str]] = mapped_column(JSON, default=list)

    has_bounding_boxes: Mapped[bool] = mapped_column(Boolean, default=False)
    has_masks: Mapped[bool] = mapped_column(Boolean, default=False)
    has_keypoints: Mapped[bool] = mapped_column(Boolean, default=False)
    has_lidar: Mapped[bool] = mapped_column(Boolean, default=False)
    has_radar: Mapped[bool] = mapped_column(Boolean, default=False)
    has_depth: Mapped[bool] = mapped_column(Boolean, default=False)
    has_thermal: Mapped[bool] = mapped_column(Boolean, default=False)

    repo_id: Mapped[str | None] = mapped_column(String(256), default=None)
    license: Mapped[str | None] = mapped_column(String(64), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    asset: Mapped[Asset] = relationship(back_populates="dataset_details")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<DatasetDetails asset={self.asset_id} format={self.dataset_format!r}>"


class AssetFile(Base):
    """One file belonging to an asset.

    Hash columns are populated lazily: ``quick_sig`` only for size-collision candidates
    and ``sha256`` only for quick-signature collisions. See
    :mod:`ai_asset_manager.backend.duplicate`.
    """

    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("asset_id", "relpath", name="uq_files_asset_relpath"),
        Index("ix_files_quick_sig", "quick_sig"),
        Index("ix_files_sha256", "sha256"),
        Index("ix_files_size", "size_bytes"),
        Index("ix_files_identity", "device", "inode"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), index=True
    )

    relpath: Mapped[str] = mapped_column(String(1024))
    extension: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    modified_at: Mapped[datetime | None] = mapped_column(default=None)

    quick_sig: Mapped[str | None] = mapped_column(String(64), default=None)
    sha256: Mapped[str | None] = mapped_column(String(64), default=None)

    #: Filesystem identity, used to avoid counting a shared extent twice.
    inode: Mapped[int | None] = mapped_column(BigInteger, default=None)
    device: Mapped[int | None] = mapped_column(BigInteger, default=None)
    nlink: Mapped[int] = mapped_column(Integer, default=1)
    is_symlink: Mapped[bool] = mapped_column(Boolean, default=False)
    #: True when this file is a weight/data payload rather than a config or doc.
    is_payload: Mapped[bool] = mapped_column(Boolean, default=False)

    asset: Mapped[Asset] = relationship(back_populates="files")

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<AssetFile {self.relpath!r} {self.size_bytes}B>"
