"""Image and video generation models."""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import Framework, ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import family_of, is_model
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Generative families. Auxiliary pieces — VAEs, ControlNets, IP-Adapters — are included
#: because they occupy real disk space and a user taking stock of a generation setup needs
#: to see them, even though none of them generates anything alone.
DIFFUSION_FAMILIES = (
    ("FLUX", ("flux",)),
    ("SDXL", ("sdxl", "stable-diffusion-xl", "stable_diffusion_xl")),
    ("SD3", ("sd3", "stable-diffusion-3")),
    ("Stable Diffusion", ("stable-diffusion", "stable_diffusion", "sd-v1", "sd15", "sd-1.5")),
    ("ControlNet", ("controlnet",)),
    ("IP-Adapter", ("ip-adapter", "ip_adapter")),
    ("Kandinsky", ("kandinsky",)),
    ("PixArt", ("pixart",)),
    ("Playground", ("playground-v2", "playgroundai")),
    ("Wan", ("wan2", "wan-2", "wanx")),
    ("HunyuanVideo", ("hunyuanvideo", "hunyuan-video")),
    ("LTX-Video", ("ltx-video", "ltxvideo")),
    ("CogVideo", ("cogvideo",)),
    ("AnimateDiff", ("animatediff",)),
    ("Mochi", ("mochi-1", "genmo/mochi")),
    ("VAE", ("-vae", "_vae", "vae-ft")),
    ("Real-ESRGAN", ("real-esrgan", "realesrgan", "esrgan")),
    ("GFPGAN", ("gfpgan", "codeformer")),
)

TASKS = (
    Task(id="image_generation", label="Image Generation", domain="generative", order=10),
    Task(id="video_generation", label="Video Generation", domain="generative", order=20),
    Task(id="image_editing", label="Image Editing", domain="generative", order=30),
    Task(id="inpainting", label="Inpainting", domain="generative", order=40),
    Task(id="conditioning", label="Conditioning", domain="generative", order=50),
    Task(id="face_restoration", label="Face Restoration", domain="generative", order=60),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register generative categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="diffusion", label="Diffusion", section="models", order=90,
                 domain="generative",
                 aliases=("image-generation", "diffusion-models", "txt2img"),
                 description="Image and video generation pipelines and their parts.")
    )

    registry.add_classifier(_diffusion_model, name="generative.diffusion", priority=450)


def _diffusion_model(profile: AssetProfile) -> Classification | None:
    """Claim generation pipelines and the components that serve them."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, DIFFUSION_FAMILIES)
    diffusers = profile.framework == Framework.DIFFUSERS

    if declared != ModelType.IMAGE_GENERATION and family is None and not diffusers:
        return None

    if family in ("ControlNet", "IP-Adapter", "VAE"):
        task = "conditioning"
    elif family in ("Real-ESRGAN", "GFPGAN"):
        task = "face_restoration"
    elif family in ("Wan", "HunyuanVideo", "LTX-Video", "CogVideo", "AnimateDiff", "Mochi"):
        task = "video_generation"
    elif "inpaint" in profile.haystack:
        task = "inpainting"
    else:
        task = "image_generation"

    return Classification(
        category="diffusion", task=task, domain="generative", family=family,
        modalities=("rgb", "text"),
        confidence=CONFIDENCE_CERTAIN if declared == ModelType.IMAGE_GENERATION
        else CONFIDENCE_STRONG,
        evidence=f"{family} pipeline" if family
        else ("diffusers pipeline" if diffusers else "declared an image generator"),
    )
