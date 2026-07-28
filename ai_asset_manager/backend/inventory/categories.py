"""The inventory taxonomy.

Answers "what *kind* of thing is this?" in the vocabulary a person uses when taking stock
of their library — "OCR models", "detection datasets" — rather than in the vocabulary the
scanner uses to identify formats.

The two are deliberately separate. :class:`ModelType` describes what a file *is*
(a checkpoint whose architecture ends in ``ForCausalLM``); an inventory category
describes what a user *keeps it for*. Several model types collapse into one category, and
one dataset format can land in different categories depending on what it contains.

Classification here reads nothing but columns already in the database. No file is opened
and no path is walked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ai_asset_manager.backend.models.enums import AssetKind, DatasetFormat, ModelType


class InventorySection(StrEnum):
    """Top-level division of the inventory."""

    MODELS = "models"
    DATASETS = "datasets"
    PAPERS = "papers"
    OTHER = "other"


class InventoryCategory(StrEnum):
    """A category in the inventory taxonomy."""

    # -- models ------------------------------------------------------------
    LLM = "llm"
    VISION_LANGUAGE = "vision_language"
    OCR = "ocr"
    VISION = "vision"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    SPEECH = "speech"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO = "audio"
    DIFFUSION = "diffusion"
    ADAPTER = "adapter"
    OTHER_MODEL = "other_model"

    # -- datasets ----------------------------------------------------------
    IMAGE_DATASET = "image_dataset"
    VIDEO_DATASET = "video_dataset"
    OCR_DATASET = "ocr_dataset"
    DETECTION_DATASET = "detection_dataset"
    SEGMENTATION_DATASET = "segmentation_dataset"
    TRACKING_DATASET = "tracking_dataset"
    AUDIO_DATASET = "audio_dataset"
    NLP_DATASET = "nlp_dataset"
    OTHER_DATASET = "other_dataset"

    # -- future ------------------------------------------------------------
    PAPER = "paper"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class CategoryInfo:
    """Display metadata for a category."""

    category: InventoryCategory
    label: str
    section: InventorySection
    #: Sort position within its section, so the report reads in a sensible order.
    order: int


#: Display metadata, in the order categories should appear.
CATEGORY_INFO: dict[InventoryCategory, CategoryInfo] = {
    info.category: info
    for info in (
        CategoryInfo(InventoryCategory.LLM, "LLM", InventorySection.MODELS, 10),
        CategoryInfo(InventoryCategory.VISION_LANGUAGE, "Vision-Language",
                     InventorySection.MODELS, 20),
        CategoryInfo(InventoryCategory.OCR, "OCR Model", InventorySection.MODELS, 30),
        CategoryInfo(InventoryCategory.OBJECT_DETECTION, "Object Detection",
                     InventorySection.MODELS, 40),
        CategoryInfo(InventoryCategory.SEGMENTATION, "Segmentation",
                     InventorySection.MODELS, 50),
        CategoryInfo(InventoryCategory.TRACKING, "Tracking", InventorySection.MODELS, 60),
        CategoryInfo(InventoryCategory.CLASSIFICATION, "Classification",
                     InventorySection.MODELS, 70),
        CategoryInfo(InventoryCategory.VISION, "Vision Model", InventorySection.MODELS, 80),
        CategoryInfo(InventoryCategory.DIFFUSION, "Diffusion", InventorySection.MODELS, 90),
        CategoryInfo(InventoryCategory.EMBEDDING, "Embedding", InventorySection.MODELS, 100),
        CategoryInfo(InventoryCategory.RERANKER, "Reranker", InventorySection.MODELS, 110),
        CategoryInfo(InventoryCategory.SPEECH, "Speech", InventorySection.MODELS, 120),
        CategoryInfo(InventoryCategory.TEXT_TO_SPEECH, "Text-to-Speech",
                     InventorySection.MODELS, 130),
        CategoryInfo(InventoryCategory.AUDIO, "Audio Model", InventorySection.MODELS, 140),
        CategoryInfo(InventoryCategory.ADAPTER, "Adapter / LoRA", InventorySection.MODELS, 150),
        CategoryInfo(InventoryCategory.OTHER_MODEL, "Other Model",
                     InventorySection.MODELS, 900),
        CategoryInfo(InventoryCategory.DETECTION_DATASET, "Detection Dataset",
                     InventorySection.DATASETS, 200),
        CategoryInfo(InventoryCategory.SEGMENTATION_DATASET, "Segmentation Dataset",
                     InventorySection.DATASETS, 210),
        CategoryInfo(InventoryCategory.TRACKING_DATASET, "Tracking Dataset",
                     InventorySection.DATASETS, 220),
        CategoryInfo(InventoryCategory.IMAGE_DATASET, "Image Dataset",
                     InventorySection.DATASETS, 230),
        CategoryInfo(InventoryCategory.VIDEO_DATASET, "Video Dataset",
                     InventorySection.DATASETS, 240),
        CategoryInfo(InventoryCategory.OCR_DATASET, "OCR Dataset",
                     InventorySection.DATASETS, 250),
        CategoryInfo(InventoryCategory.AUDIO_DATASET, "Audio Dataset",
                     InventorySection.DATASETS, 260),
        CategoryInfo(InventoryCategory.NLP_DATASET, "NLP Dataset",
                     InventorySection.DATASETS, 270),
        CategoryInfo(InventoryCategory.OTHER_DATASET, "Other Dataset",
                     InventorySection.DATASETS, 910),
        CategoryInfo(InventoryCategory.PAPER, "Research Paper", InventorySection.PAPERS, 300),
        CategoryInfo(InventoryCategory.UNKNOWN, "Unclassified", InventorySection.OTHER, 999),
    )
}

#: Model types mapped to their inventory category.
MODEL_TYPE_CATEGORIES: dict[ModelType, InventoryCategory] = {
    ModelType.LLM: InventoryCategory.LLM,
    ModelType.VISION_LANGUAGE: InventoryCategory.VISION_LANGUAGE,
    ModelType.MULTIMODAL: InventoryCategory.VISION_LANGUAGE,
    ModelType.OCR: InventoryCategory.OCR,
    ModelType.OBJECT_DETECTION: InventoryCategory.OBJECT_DETECTION,
    ModelType.SEGMENTATION: InventoryCategory.SEGMENTATION,
    ModelType.CLASSIFICATION: InventoryCategory.CLASSIFICATION,
    ModelType.POSE: InventoryCategory.VISION,
    ModelType.VISION: InventoryCategory.VISION,
    ModelType.IMAGE_GENERATION: InventoryCategory.DIFFUSION,
    ModelType.EMBEDDING: InventoryCategory.EMBEDDING,
    ModelType.RERANKER: InventoryCategory.RERANKER,
    ModelType.SPEECH_RECOGNITION: InventoryCategory.SPEECH,
    ModelType.TEXT_TO_SPEECH: InventoryCategory.TEXT_TO_SPEECH,
    ModelType.AUDIO: InventoryCategory.AUDIO,
    ModelType.LORA: InventoryCategory.ADAPTER,
    ModelType.UNKNOWN: InventoryCategory.OTHER_MODEL,
}

#: Dataset formats mapped to their inventory category.
DATASET_FORMAT_CATEGORIES: dict[DatasetFormat, InventoryCategory] = {
    DatasetFormat.COCO: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.YOLO: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.PASCAL_VOC: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.KITTI: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.WAYMO: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.NUSCENES: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.BDD100K: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.CROWDHUMAN: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.OPEN_IMAGES: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.LVIS: InventoryCategory.DETECTION_DATASET,
    DatasetFormat.CITYSCAPES: InventoryCategory.SEGMENTATION_DATASET,
    DatasetFormat.ADE20K: InventoryCategory.SEGMENTATION_DATASET,
    DatasetFormat.SEGMENTATION: InventoryCategory.SEGMENTATION_DATASET,
    DatasetFormat.MOT: InventoryCategory.TRACKING_DATASET,
    DatasetFormat.TRACKING: InventoryCategory.TRACKING_DATASET,
    DatasetFormat.IMAGENET: InventoryCategory.IMAGE_DATASET,
    DatasetFormat.IMAGE_CLASSIFICATION: InventoryCategory.IMAGE_DATASET,
    DatasetFormat.VIDEO: InventoryCategory.VIDEO_DATASET,
    DatasetFormat.AUDIO: InventoryCategory.AUDIO_DATASET,
    DatasetFormat.NLP: InventoryCategory.NLP_DATASET,
    DatasetFormat.TABULAR: InventoryCategory.NLP_DATASET,
    DatasetFormat.HF_DATASET: InventoryCategory.NLP_DATASET,
    DatasetFormat.CUSTOM: InventoryCategory.OTHER_DATASET,
    DatasetFormat.UNKNOWN: InventoryCategory.OTHER_DATASET,
}

#: Words that mark a dataset as OCR-related. Checked against the declared task, the
#: dataset name and its class labels, since no dataset *format* is OCR-specific — an OCR
#: corpus is usually COCO-shaped or a plain image folder.
_OCR_MARKERS = ("ocr", "text-recognition", "text_recognition", "textocr", "scene-text",
                "handwriting", "htr", "iam", "receipt", "docvqa", "funsd", "sroie")

#: Names that mark a model as tracking-related. Tracking has no distinct architecture —
#: trackers are detectors plus association — so the name is the only signal available.
_TRACKING_MARKERS = ("bytetrack", "botsort", "deepsort", "ocsort", "strongsort",
                     "fairmot", "jde", "sort_", "tracker", "trackformer")

#: Image-classification backbone families, distributed as bare checkpoints by torchvision
#: and timm with no configuration file to identify them.
_BACKBONE_MARKERS = ("resnet", "resnext", "mobilenet", "efficientnet", "vgg", "densenet",
                     "inception", "shufflenet", "squeezenet", "convnext", "regnet",
                     "swin", "vit_", "deit", "alexnet", "googlenet", "mnasnet")


def _matches(haystack: str, markers: tuple[str, ...]) -> bool:
    """Report whether any marker appears in a lower-cased haystack."""
    return any(marker in haystack for marker in markers)


def classify_model(
    model_type: str | None,
    *,
    name: str = "",
    architecture: str | None = None,
) -> InventoryCategory:
    """Return the inventory category for a model.

    Args:
        model_type: Stored :class:`ModelType` value.
        name: Asset name, used for signals no architecture carries.
        architecture: Stored architecture string.

    Returns:
        The category to file this model under.

    Examples:
        >>> classify_model("llm")
        <InventoryCategory.LLM: 'llm'>
        >>> classify_model("object_detection", name="bytetrack_x_mot17")
        <InventoryCategory.TRACKING: 'tracking'>
        >>> classify_model(None, name="surya_ocr")
        <InventoryCategory.OCR: 'ocr'>
    """
    haystack = f"{name} {architecture or ''}".lower()

    # Trackers are detectors with association on top, so the detector mapping would
    # swallow them. Name is the only thing that distinguishes them.
    if _matches(haystack, _TRACKING_MARKERS):
        return InventoryCategory.TRACKING

    resolved = _coerce(model_type, ModelType)
    if resolved is not None and resolved is not ModelType.UNKNOWN:
        return MODEL_TYPE_CATEGORIES.get(resolved, InventoryCategory.OTHER_MODEL)

    # Nothing classified it; fall back to the name, which is how loose weight files with
    # no config beside them get categorised at all.
    if _matches(haystack, _OCR_MARKERS):
        return InventoryCategory.OCR
    if _matches(haystack, ("whisper", "wav2vec", "speech")):
        return InventoryCategory.SPEECH
    if _matches(haystack, ("yolo", "detr", "rcnn", "ssd", "retinanet")):
        return InventoryCategory.OBJECT_DETECTION
    if _matches(haystack, ("sam", "segformer", "maskformer", "unet")):
        return InventoryCategory.SEGMENTATION
    if _matches(haystack, ("embed", "bge", "gte", "e5", "minilm")):
        return InventoryCategory.EMBEDDING
    # Torchvision and timm backbones ship as bare .pth files with no config beside them,
    # so the family name is the only thing that identifies them.
    if _matches(haystack, _BACKBONE_MARKERS):
        return InventoryCategory.CLASSIFICATION
    if _matches(haystack, ("stable-diffusion", "sdxl", "flux", "controlnet")):
        return InventoryCategory.DIFFUSION

    return InventoryCategory.OTHER_MODEL


def classify_dataset(
    dataset_format: str | None,
    *,
    name: str = "",
    task: str | None = None,
    class_names: list[str] | None = None,
    num_videos: int = 0,
    num_audio_files: int = 0,
    num_images: int = 0,
) -> InventoryCategory:
    """Return the inventory category for a dataset.

    The format alone is often not enough. A HuggingFace dataset is just Parquet, and an
    image folder is just images; what distinguishes an OCR corpus from a classification
    set is its task and content, both of which the scanner already recorded.

    Examples:
        >>> classify_dataset("coco")
        <InventoryCategory.DETECTION_DATASET: 'detection_dataset'>
        >>> classify_dataset("image_classification", name="TextOCR-2024")
        <InventoryCategory.OCR_DATASET: 'ocr_dataset'>
        >>> classify_dataset("hf_dataset", num_audio_files=5000)
        <InventoryCategory.AUDIO_DATASET: 'audio_dataset'>
    """
    haystack = f"{name} {task or ''} {' '.join(class_names or [])}".lower()

    # OCR is a task, not a layout, so it is checked before the format mapping — a
    # TextOCR corpus in COCO format is an OCR dataset, not a detection dataset.
    if _matches(haystack, _OCR_MARKERS):
        return InventoryCategory.OCR_DATASET

    resolved = _coerce(dataset_format, DatasetFormat)
    if resolved is not None and resolved not in (
        DatasetFormat.UNKNOWN, DatasetFormat.CUSTOM, DatasetFormat.HF_DATASET
    ):
        return DATASET_FORMAT_CATEGORIES.get(resolved, InventoryCategory.OTHER_DATASET)

    # Generic containers are classified by what is actually inside them.
    if num_audio_files > 0 and num_audio_files >= max(num_images, num_videos):
        return InventoryCategory.AUDIO_DATASET
    if num_videos > 0 and num_videos >= num_images:
        return InventoryCategory.VIDEO_DATASET
    if num_images > 0:
        return InventoryCategory.IMAGE_DATASET
    if resolved is DatasetFormat.HF_DATASET:
        return InventoryCategory.NLP_DATASET

    return InventoryCategory.OTHER_DATASET


def classify_asset(
    kind: str,
    *,
    name: str = "",
    model_type: str | None = None,
    architecture: str | None = None,
    dataset_format: str | None = None,
    task: str | None = None,
    class_names: list[str] | None = None,
    num_images: int = 0,
    num_videos: int = 0,
    num_audio_files: int = 0,
) -> InventoryCategory:
    """Return the inventory category for any catalogued asset."""
    resolved_kind = _coerce(kind, AssetKind)

    if resolved_kind is AssetKind.DATASET:
        return classify_dataset(
            dataset_format,
            name=name,
            task=task,
            class_names=class_names,
            num_images=num_images,
            num_videos=num_videos,
            num_audio_files=num_audio_files,
        )

    if resolved_kind is AssetKind.ADAPTER:
        return InventoryCategory.ADAPTER

    if resolved_kind is AssetKind.PAPER:
        return InventoryCategory.PAPER

    if resolved_kind in (AssetKind.MODEL, AssetKind.CHECKPOINT):
        return classify_model(model_type, name=name, architecture=architecture)

    return InventoryCategory.UNKNOWN


def _coerce[E: StrEnum](value: str | None, enum_cls: type[E]) -> E | None:
    """Convert a stored string into an enum member, tolerating unknown values.

    Unknown values return ``None`` rather than raising: the catalogue may hold a value
    written by an older version, and an inventory that crashes on it would be worse than
    one that files it as unclassified.
    """
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def section_of(category: InventoryCategory) -> InventorySection:
    """Return the section a category belongs to."""
    info = CATEGORY_INFO.get(category)
    return info.section if info else InventorySection.OTHER


def label_of(category: InventoryCategory) -> str:
    """Return the display label for a category."""
    info = CATEGORY_INFO.get(category)
    return info.label if info else category.value.replace("_", " ").title()


def order_of(category: InventoryCategory) -> int:
    """Return the display sort position for a category."""
    info = CATEGORY_INFO.get(category)
    return info.order if info else 999


def categories_in_section(section: InventorySection) -> list[InventoryCategory]:
    """Return every category belonging to a section, in display order."""
    return sorted(
        (info.category for info in CATEGORY_INFO.values() if info.section is section),
        key=order_of,
    )


#: What a user may type on the command line, mapped to the categories it selects.
#: Aliases are precise: ``ocr`` means OCR *models*, matching "every OCR model installed".
#: Dataset categories are reached through ``datasets`` or an explicit ``*-datasets`` name.
CATEGORY_ALIASES: dict[str, tuple[InventoryCategory, ...]] = {
    "all": tuple(CATEGORY_INFO),
    "models": tuple(categories_in_section(InventorySection.MODELS)),
    "datasets": tuple(categories_in_section(InventorySection.DATASETS)),
    "papers": (InventoryCategory.PAPER,),
    "llm": (InventoryCategory.LLM,),
    "llms": (InventoryCategory.LLM,),
    "ocr": (InventoryCategory.OCR,),
    "vlm": (InventoryCategory.VISION_LANGUAGE,),
    "vision-language": (InventoryCategory.VISION_LANGUAGE,),
    "vision": (
        InventoryCategory.VISION,
        InventoryCategory.VISION_LANGUAGE,
        InventoryCategory.OBJECT_DETECTION,
        InventoryCategory.SEGMENTATION,
        InventoryCategory.CLASSIFICATION,
        InventoryCategory.TRACKING,
    ),
    "detection": (InventoryCategory.OBJECT_DETECTION,),
    "segmentation": (InventoryCategory.SEGMENTATION,),
    "tracking": (InventoryCategory.TRACKING,),
    "classification": (InventoryCategory.CLASSIFICATION,),
    "speech": (
        InventoryCategory.SPEECH,
        InventoryCategory.TEXT_TO_SPEECH,
        InventoryCategory.AUDIO,
    ),
    "asr": (InventoryCategory.SPEECH,),
    "tts": (InventoryCategory.TEXT_TO_SPEECH,),
    "embeddings": (InventoryCategory.EMBEDDING, InventoryCategory.RERANKER),
    "embedding": (InventoryCategory.EMBEDDING, InventoryCategory.RERANKER),
    "diffusion": (InventoryCategory.DIFFUSION,),
    "image-generation": (InventoryCategory.DIFFUSION,),
    "adapters": (InventoryCategory.ADAPTER,),
    "adapter": (InventoryCategory.ADAPTER,),
    "lora": (InventoryCategory.ADAPTER,),
    "image-datasets": (InventoryCategory.IMAGE_DATASET,),
    "video-datasets": (InventoryCategory.VIDEO_DATASET,),
    "audio-datasets": (InventoryCategory.AUDIO_DATASET,),
    "nlp-datasets": (InventoryCategory.NLP_DATASET,),
    "ocr-datasets": (InventoryCategory.OCR_DATASET,),
    "detection-datasets": (InventoryCategory.DETECTION_DATASET,),
    "segmentation-datasets": (InventoryCategory.SEGMENTATION_DATASET,),
    "tracking-datasets": (InventoryCategory.TRACKING_DATASET,),
}


def resolve_alias(value: str) -> tuple[InventoryCategory, ...] | None:
    """Resolve a user-typed category name.

    Accepts an alias, a bare category value, or ``None`` for an unrecognised name so the
    caller can report the valid options rather than silently returning nothing.

    Examples:
        >>> resolve_alias("llm")
        (<InventoryCategory.LLM: 'llm'>,)
        >>> resolve_alias("nonsense") is None
        True
    """
    key = value.strip().lower().replace("_", "-")
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]

    underscored = key.replace("-", "_")
    if underscored in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[underscored]

    try:
        return (InventoryCategory(underscored),)
    except ValueError:
        return None


def known_aliases() -> list[str]:
    """Return every accepted category name, for help text and error messages."""
    return sorted(CATEGORY_ALIASES)
