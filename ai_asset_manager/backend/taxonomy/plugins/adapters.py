"""Adapters, LoRAs and other parameter-efficient fine-tunes.

Ranked near the top of the priority order because this is a structural fact rather than an
inference: the catalogue already recorded that the asset is an adapter. What matters for an
adapter is what it was trained *onto*, so the base model becomes its family — a LoRA is
useless without the model it patches, and "which base does this need?" is the question a
user actually has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import AssetKind, ModelType, Severity
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    AssetProfile,
    Category,
    Classification,
    Finding,
    Task,
)

TASKS = (
    Task(id="fine_tuning", label="Fine-Tuning", domain="general", order=10),
    Task(id="style_transfer", label="Style Transfer", domain="generative", order=70),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the adapter category, task and rules."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="adapter", label="Adapter / LoRA", section="models", order=150,
                 domain="general", aliases=("adapters", "lora", "loras", "peft"),
                 description="Small weight deltas applied on top of a base model.")
    )

    registry.add_classifier(_adapter, name="adapter", priority=900)
    registry.add_statistic(_adapter_statistics, name="adapter")
    registry.add_health_rule(_orphaned_adapter, name="adapter.no_base_model")


def _is_adapter(profile: AssetProfile) -> bool:
    """Report whether the catalogue recorded this asset as an adapter."""
    if profile.kind == AssetKind.ADAPTER:
        return True
    return bool(profile.model and profile.model.model_type == ModelType.LORA)


def _adapter(profile: AssetProfile) -> Classification | None:
    """Claim adapters, naming the base model as the family."""
    if not _is_adapter(profile):
        return None

    base = profile.model.base_model if profile.model else None
    haystack = profile.haystack
    task = "style_transfer" if ("style" in haystack or ("lora" in haystack
                                and "diffusion" in haystack)) else "fine_tuning"

    return Classification(
        category="adapter", task=task, domain="general",
        family=_short_base_name(base), confidence=CONFIDENCE_CERTAIN,
        evidence=f"adapter for {base}" if base else "catalogued as an adapter",
    )


def _short_base_name(base_model: str | None) -> str | None:
    """Return the repository's model name without its owner prefix."""
    if not base_model:
        return None
    return base_model.rsplit("/", 1)[-1]


def _adapter_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return what an adapter attaches to and how big the delta is."""
    if not _is_adapter(profile) or profile.model is None:
        return {}

    stats: dict[str, Any] = {}
    if profile.model.base_model:
        stats["base_model"] = profile.model.base_model

    extra = profile.model.extra or {}
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "peft_type"):
        if key in extra:
            stats[key] = extra[key]

    return stats


def _orphaned_adapter(profile: AssetProfile) -> Sequence[Finding]:
    """Report an adapter that does not say what it patches.

    Without a base model an adapter is unusable: the weights are a delta against a tensor
    layout, and applying them to the wrong model fails in confusing ways or, worse,
    silently produces nonsense.
    """
    if not _is_adapter(profile):
        return ()
    if profile.model is not None and profile.model.base_model:
        return ()

    return (
        Finding(
            code="adapter.no_base_model",
            severity=Severity.WARNING,
            message="No base model recorded",
            fix_hint="Check adapter_config.json for base_model_name_or_path.",
        ),
    )
