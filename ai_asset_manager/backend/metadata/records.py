"""The record produced by the pipeline and consumed by persistence.

A plain dataclass rather than an ORM object, so that detection, parsing and merging can
run on worker threads with no session attached and no risk of a partially built entity
being flushed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_asset_manager.backend.models.enums import (
    AssetFormat,
    AssetKind,
    DatasetFormat,
    Framework,
    ModelType,
    Precision,
)


@dataclass(slots=True)
class ModelFacts:
    """Resolved model-specific metadata."""

    model_type: ModelType = ModelType.UNKNOWN
    architecture: str | None = None
    param_count: int | None = None
    param_count_is_exact: bool = False
    quantization: str | None = None
    precision: Precision = Precision.UNKNOWN
    context_length: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    tensor_count: int | None = None
    repo_id: str | None = None
    revision: str | None = None
    author: str | None = None
    license: str | None = None
    description: str | None = None
    base_model: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    card_tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetFacts:
    """Resolved dataset-specific metadata."""

    dataset_format: DatasetFormat = DatasetFormat.UNKNOWN
    task: str | None = None
    num_images: int = 0
    num_videos: int = 0
    num_audio_files: int = 0
    num_text_files: int = 0
    num_annotations: int = 0
    num_classes: int | None = None
    class_names: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=list)
    has_bounding_boxes: bool = False
    has_masks: bool = False
    has_keypoints: bool = False
    has_lidar: bool = False
    has_radar: bool = False
    has_depth: bool = False
    has_thermal: bool = False
    repo_id: str | None = None
    license: str | None = None
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileRecord:
    """One file belonging to an asset, as gathered during the walk."""

    relpath: str
    abspath: str
    size: int
    mtime: float
    inode: int
    device: int
    nlink: int
    is_symlink: bool
    is_payload: bool
    extension: str | None = None

    @property
    def identity(self) -> tuple[int, int] | None:
        """Return the ``(device, inode)`` pair, or ``None`` when unavailable."""
        if self.inode == 0 and self.device == 0:
            return None
        return (self.device, self.inode)


@dataclass(slots=True)
class AssetRecord:
    """Everything known about one asset after detection, parsing and merging."""

    # -- identity -----------------------------------------------------------
    kind: AssetKind
    name: str
    root_path: str
    display_name: str | None = None
    subkind: str | None = None
    drive: str | None = None
    is_single_file: bool = False

    # -- storage ------------------------------------------------------------
    size_bytes: int = 0
    physical_size_bytes: int = 0
    file_count: int = 0

    # -- classification -----------------------------------------------------
    format: AssetFormat = AssetFormat.UNKNOWN
    framework: Framework = Framework.UNKNOWN

    # -- filesystem timestamps ---------------------------------------------
    created_at: datetime | None = None
    modified_at: datetime | None = None
    accessed_at: datetime | None = None

    # -- provenance ---------------------------------------------------------
    fingerprint: str | None = None
    detector: str | None = None
    detection_confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    # -- details ------------------------------------------------------------
    model: ModelFacts | None = None
    dataset: DatasetFacts | None = None
    files: list[FileRecord] = field(default_factory=list)

    #: Problems noticed during parsing, promoted to health findings on persist.
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<AssetRecord {self.kind.value}:{self.name!r} files={self.file_count}>"
