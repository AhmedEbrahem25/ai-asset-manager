"""OCR and document AI.

Ranked above the language and vision-language plugins on purpose. Modern OCR models are
vision-language models architecturally — ``PaddleOCRVLForConditionalGeneration`` and
``Qwen2VLForConditionalGeneration`` differ by a prefix — so any rule reading the
architecture alone will file an OCR model as a chat model. Reading the *name* first, and
only then falling through to architecture, is what keeps "which OCR models do I have?"
answerable.

Nor is OCR a dataset layout. An OCR corpus is usually COCO-shaped or a plain image folder,
so the dataset rule here also outranks the vision plugin's.
"""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    declared_format,
    family_of,
    is_dataset,
    is_model,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: OCR engines and document models, by family.
OCR_FAMILIES = (
    ("Surya", ("surya",)),
    ("PaddleOCR", ("paddleocr", "paddle-ocr", "pp-ocr", "ppocr", "pp-structure")),
    ("EasyOCR", ("easyocr",)),
    ("RapidOCR", ("rapidocr",)),
    ("TrOCR", ("trocr",)),
    ("GOT-OCR", ("got-ocr", "got_ocr")),
    ("dots.ocr", ("dots.ocr", "dots-ocr")),
    ("MinerU", ("mineru",)),
    ("olmOCR", ("olmocr",)),
    ("Nougat", ("nougat",)),
    ("Donut", ("donut",)),
    ("docTR", ("doctr",)),
    ("Tesseract", ("tesseract",)),
    ("LayoutLM", ("layoutlm",)),
    ("Kosmos", ("kosmos-2.5", "kosmos_2_5")),
    ("Qari", ("qari",)),
    ("Marker", ("marker-pdf", "marker_pdf")),
)

#: Words that mark OCR work anywhere in a name, architecture, task or class list. No
#: single one of these is decisive on its own, which is why they are tested together and
#: the matched marker is reported as evidence.
OCR_MARKERS = (
    "ocr", "text-recognition", "text_recognition", "textocr", "scene-text", "scenetext",
    "handwriting", "handwritten", "htr", "text-detection", "text_detection",
    "document-understanding", "docvqa", "funsd", "sroie", "cord-v2", "publaynet",
    "doclaynet", "iam-handwriting", "receipt", "invoice", "table-recognition",
)

#: Corpora people actually download for OCR work.
OCR_DATASET_FAMILIES = (
    ("TextOCR", ("textocr",)),
    ("ICDAR", ("icdar",)),
    ("IAM", ("iam-handwriting", "iam_handwriting", "iamdb")),
    ("FUNSD", ("funsd",)),
    ("SROIE", ("sroie",)),
    ("CORD", ("cord-v2", "cord_v2")),
    ("DocVQA", ("docvqa",)),
    ("PubLayNet", ("publaynet",)),
    ("DocLayNet", ("doclaynet",)),
    ("SynthText", ("synthtext",)),
    ("MJSynth", ("mjsynth",)),
)

TASKS = (
    Task(id="ocr", label="OCR", domain="document_ai", order=10),
    Task(id="document_understanding", label="Document Understanding",
         domain="document_ai", order=20),
    Task(id="document_layout_analysis", label="Document Layout Analysis",
         domain="document_ai", order=30),
    Task(id="handwriting_recognition", label="Handwriting Recognition",
         domain="document_ai", order=40),
    Task(id="table_recognition", label="Table Recognition", domain="document_ai", order=50),
    Task(id="key_information_extraction", label="Key Information Extraction",
         domain="document_ai", order=60),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register OCR categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="ocr", label="OCR Model", section="models", order=30,
                 domain="document_ai", aliases=("ocr-models", "document-ai"),
                 description="Reads text out of images and documents.")
    )
    registry.add_category(
        Category(id="ocr_dataset", label="OCR Dataset", section="datasets", order=250,
                 domain="document_ai", aliases=("ocr-datasets",))
    )

    registry.add_classifier(_ocr_model, name="ocr.model", priority=800)
    registry.add_classifier(_ocr_dataset, name="ocr.dataset", priority=790)


def _task_for(haystack: str) -> str:
    """Return the most specific document task the name supports."""
    if "handwriting" in haystack or "handwritten" in haystack or "htr" in haystack:
        return "handwriting_recognition"
    if "layout" in haystack or "publaynet" in haystack or "doclaynet" in haystack:
        return "document_layout_analysis"
    if "table" in haystack:
        return "table_recognition"
    if "docvqa" in haystack or "document-understanding" in haystack:
        return "document_understanding"
    if "funsd" in haystack or "sroie" in haystack or "cord" in haystack:
        return "key_information_extraction"
    return "ocr"


def _ocr_model(profile: AssetProfile) -> Classification | None:
    """Claim OCR and document models before any language rule sees them."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, OCR_FAMILIES)
    marker = profile.matches(OCR_MARKERS)

    if declared != ModelType.OCR and family is None and marker is None:
        return None

    if declared == ModelType.OCR:
        evidence = "declared an OCR model"
        confidence = CONFIDENCE_CERTAIN
    elif family is not None:
        evidence = f"{family} OCR engine"
        confidence = CONFIDENCE_STRONG
    else:
        evidence = f"name contains {marker!r}"
        confidence = CONFIDENCE_STRONG

    return Classification(
        category="ocr", task=_task_for(profile.haystack), domain="document_ai",
        family=family, modalities=("document", "text"), confidence=confidence,
        evidence=evidence,
    )


def _ocr_dataset(profile: AssetProfile) -> Classification | None:
    """Claim OCR corpora, whatever layout they happen to use.

    Outranks the detection-dataset rule because an OCR corpus in COCO format is an OCR
    corpus: the boxes are around words, and filing it under detection would put it on a
    shelf with COCO and Pascal VOC where nobody looking for text data would find it.
    """
    if not is_dataset(profile):
        return None

    family = family_of(profile, OCR_DATASET_FAMILIES)
    marker = profile.matches(OCR_MARKERS)
    if family is None and marker is None:
        return None

    layout = declared_format(profile)
    return Classification(
        category="ocr_dataset", task=_task_for(profile.haystack), domain="document_ai",
        family=family, modalities=("document", "text", "rgb"),
        confidence=CONFIDENCE_STRONG,
        evidence=f"{family} corpus" if family
        else f"{marker!r} in {layout or 'name'}",
    )
