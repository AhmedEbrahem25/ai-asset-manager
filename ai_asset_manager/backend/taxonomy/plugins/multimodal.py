"""Vision-language models.

Sits between OCR and the vision plugins. A vision-language model and an OCR model share an
architecture, so OCR is checked first; a vision-language model and a detector share
nothing but a name prefix, so this runs before vision to keep ``Qwen2-VL`` off the
detection shelf.
"""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import (
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

#: Vision-language families. Contrastive models such as CLIP belong here too: they are
#: image-and-text models, and someone taking stock of what they can caption or retrieve
#: with wants to see them alongside the generative ones.
VLM_FAMILIES = (
    ("LLaVA", ("llava",)),
    ("Qwen-VL", ("qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwenvl")),
    ("InternVL", ("internvl",)),
    ("Florence", ("florence",)),
    ("PaliGemma", ("paligemma",)),
    ("MiniCPM-V", ("minicpm-v", "minicpm_v")),
    ("SmolVLM", ("smolvlm",)),
    ("Pixtral", ("pixtral",)),
    ("Idefics", ("idefics",)),
    ("Molmo", ("molmo",)),
    ("Moondream", ("moondream",)),
    ("CogVLM", ("cogvlm", "cogagent")),
    ("Fuyu", ("fuyu",)),
    ("Janus", ("janus",)),
    ("Ovis", ("ovis",)),
    ("Aria", ("aria-",)),
    ("BLIP", ("blip",)),
    ("SigLIP", ("siglip",)),
    ("CLIP", ("clip",)),
    ("Kosmos", ("kosmos",)),
)

#: Vision-language corpora.
VLM_DATASET_FAMILIES = (
    ("LAION", ("laion",)),
    ("COCO Captions", ("coco-captions", "coco_captions")),
    ("Visual Genome", ("visual-genome", "visual_genome")),
    ("VQAv2", ("vqav2", "vqa-v2")),
    ("TextVQA", ("textvqa",)),
    ("GQA", ("gqa",)),
    ("Flickr30k", ("flickr30k", "flickr8k")),
    ("Conceptual Captions", ("conceptual-captions", "cc3m", "cc12m")),
    ("ShareGPT4V", ("sharegpt4v",)),
    ("LLaVA-Instruct", ("llava-instruct",)),
)

TASKS = (
    Task(id="image_captioning", label="Image Captioning", domain="multimodal", order=10),
    Task(id="visual_question_answering", label="Visual Question Answering",
         domain="multimodal", order=20),
    Task(id="image_text_retrieval", label="Image-Text Retrieval",
         domain="multimodal", order=30),
    Task(id="visual_grounding", label="Visual Grounding", domain="multimodal", order=40),
    Task(id="video_understanding", label="Video Understanding", domain="multimodal", order=50),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register vision-language categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="vision_language", label="Vision-Language", section="models", order=20,
                 domain="multimodal", aliases=("vlm", "vlms", "vision-language"),
                 description="Models that read images and write or match text.")
    )
    registry.add_category(
        Category(id="multimodal_dataset", label="Multimodal Dataset", section="datasets",
                 order=245, domain="multimodal", aliases=("multimodal-datasets",))
    )

    registry.add_classifier(_vlm_model, name="multimodal.model", priority=700)
    registry.add_classifier(_vlm_dataset, name="multimodal.dataset", priority=690)


def _vlm_model(profile: AssetProfile) -> Classification | None:
    """Claim vision-language models."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, VLM_FAMILIES)
    marker = profile.matches(("vision-language", "image-text-to-text", "multimodal", "-vl-"))

    if declared not in (ModelType.VISION_LANGUAGE, ModelType.MULTIMODAL) \
            and family is None and marker is None:
        return None

    if family in ("CLIP", "SigLIP"):
        task = "image_text_retrieval"
    elif family == "BLIP" or "caption" in profile.haystack:
        task = "image_captioning"
    else:
        task = "visual_question_answering"

    return Classification(
        category="vision_language", task=task, domain="multimodal", family=family,
        modalities=("rgb", "text"),
        confidence=CONFIDENCE_CERTAIN
        if declared in (ModelType.VISION_LANGUAGE, ModelType.MULTIMODAL)
        else CONFIDENCE_STRONG,
        evidence=f"{family} family" if family
        else ("declared a vision-language model"
              if declared else f"name contains {marker!r}"),
    )


def _vlm_dataset(profile: AssetProfile) -> Classification | None:
    """Claim image-and-text corpora."""
    if not is_dataset(profile):
        return None

    family = family_of(profile, VLM_DATASET_FAMILIES)
    marker = profile.matches(("caption", "vqa", "image-text", "visual-question"))
    if family is None and marker is None:
        return None

    task = "visual_question_answering" if (marker and "vqa" in marker) else "image_captioning"
    return Classification(
        category="multimodal_dataset", task=task, domain="multimodal", family=family,
        modalities=("rgb", "text"), confidence=CONFIDENCE_STRONG,
        evidence=f"{family} corpus" if family else f"name contains {marker!r}",
    )
