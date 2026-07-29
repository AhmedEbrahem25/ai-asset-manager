"""Persisted domain vocabulary.

These enums are stored in the database as strings rather than native SQL enums so that
adding a new format or dataset type never requires a migration, and so the schema stays
portable to PostgreSQL without type-creation DDL.
"""

from __future__ import annotations

from enum import StrEnum


class AssetKind(StrEnum):
    """Top-level classification of a catalogued asset."""

    MODEL = "model"
    DATASET = "dataset"
    ADAPTER = "adapter"
    CHECKPOINT = "checkpoint"
    #: A codebase that trains or serves models. A container of assets rather than an
    #: asset's contents, which is why its detector never claims its own subtree.
    PROJECT = "project"
    #: One training or evaluation run: a TensorBoard log directory, a W&B run, an MLflow
    #: run, an Ultralytics ``runs/detect/train`` folder.
    EXPERIMENT = "experiment"
    #: A labelling project: CVAT, Label Studio, Roboflow, Supervisely.
    ANNOTATION_PROJECT = "annotation_project"
    PAPER = "paper"
    TRAINING_CONFIG = "training_config"
    UNKNOWN = "unknown"


class ModelType(StrEnum):
    """What a model is *for*.

    Derived from config architectures, pipeline tags and filename evidence rather than
    from the storage format.
    """

    LLM = "llm"
    VISION_LANGUAGE = "vision_language"
    VISION = "vision"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    IMAGE_GENERATION = "image_generation"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO = "audio"
    OCR = "ocr"
    MULTIMODAL = "multimodal"
    LORA = "lora"
    UNKNOWN = "unknown"


class AssetFormat(StrEnum):
    """On-disk storage format of the weights."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GGML = "ggml"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    OPENVINO = "openvino"
    MLX = "mlx"
    COREML = "coreml"
    TFLITE = "tflite"
    PADDLE = "paddle"
    NUMPY = "numpy"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Framework(StrEnum):
    """Ecosystem that produced or consumes the asset."""

    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    PEFT = "peft"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    ULTRALYTICS = "ultralytics"
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    MLX = "mlx"
    TIMM = "timm"
    PADDLE = "paddle"
    UNKNOWN = "unknown"


class DatasetFormat(StrEnum):
    """Recognised dataset layouts, identified by directory structure."""

    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    IMAGENET = "imagenet"
    KITTI = "kitti"
    WAYMO = "waymo"
    NUSCENES = "nuscenes"
    BDD100K = "bdd100k"
    CITYSCAPES = "cityscapes"
    MOT = "mot"
    CROWDHUMAN = "crowdhuman"
    OPEN_IMAGES = "open_images"
    ADE20K = "ade20k"
    LVIS = "lvis"
    HF_DATASET = "hf_dataset"
    IMAGE_CLASSIFICATION = "image_classification"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    VIDEO = "video"
    AUDIO = "audio"
    NLP = "nlp"
    TABULAR = "tabular"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    """Sensor or data modality present in a dataset."""

    RGB = "rgb"
    DEPTH = "depth"
    LIDAR = "lidar"
    RADAR = "radar"
    THERMAL = "thermal"
    AUDIO = "audio"
    TEXT = "text"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"


class Precision(StrEnum):
    """Numeric precision of stored weights."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class HealthStatus(StrEnum):
    """Aggregate health verdict for an asset."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Severity of an individual health finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Return an ordinal usable for max()-style aggregation."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}


class ScanStatus(StrEnum):
    """Lifecycle state of a scan run."""

    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class DuplicateKind(StrEnum):
    """Granularity at which a duplicate group was found."""

    FILE = "file"
    ASSET = "asset"
    NEAR_ASSET = "near_asset"


class FactSource(StrEnum):
    """Where a metadata fact came from.

    Ordering of :data:`FACT_SOURCE_PRIORITY` is the single place the precedence rule
    lives; :mod:`ai_asset_manager.backend.metadata.merge` is its only consumer.
    """

    EXPLICIT_CONFIG = "explicit_config"
    BINARY_HEADER = "binary_header"
    SIDECAR_DOC = "sidecar_doc"
    FILENAME = "filename"
    DIRECTORY_NAME = "directory_name"
    DEFAULT = "default"


FACT_SOURCE_PRIORITY: dict[FactSource, int] = {
    FactSource.EXPLICIT_CONFIG: 100,
    FactSource.BINARY_HEADER: 80,
    FactSource.SIDECAR_DOC: 60,
    FactSource.FILENAME: 40,
    FactSource.DIRECTORY_NAME: 20,
    FactSource.DEFAULT: 0,
}
