r"""Shelves for what the scanner recognised structurally.

Most classifiers here work from an asset's *contents*: a config naming an architecture, a
directory of event files, a manifest. That is the right default, because contents are what
distinguish an OCR model from a chat model.

Some things are not distinguished by their contents at all. A file called
``dapt_epoch3.pt`` is a training checkpoint because of where it sits and what it is named,
and there is nothing inside it that says so more clearly. A codebase is a project because
of its shape. The scanner already worked all of that out, recorded it as the asset's kind,
and this plugin is what stops that conclusion being thrown away.

Registered above the catch-all and below everything content-based, so it only answers when
the specific rules have all declined -- ten checkpoints from one run should still be filed
as "Checkpoint" rather than as "Unclassified", but a checkpoint that a vision plugin can
recognise properly is better served by the vision plugin.
"""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Above every content-based classifier. Reserved for kinds whose identity *is* their
#: structure and which content sniffing therefore gets wrong: a project's file list is full
#: of the runs and results it produced, so a rule looking for ``results.json`` in a file
#: list will call a codebase an evaluation. On the development machine it did exactly that
#: to ``thorn-nlp``, an 8.5 GB project, and to the service inside it.
PRIORITY_CONTAINER = 900

#: Below every content-based classifier. For kinds where content really is more
#: informative: ``yolov8n.pt`` is a checkpoint, but "object detection model" is the more
#: useful thing to say about it, so the vision plugin should get first refusal.
PRIORITY_DERIVED = 50

TASKS = (
    Task(id="checkpointing", label="Checkpointing", domain="mlops", order=40),
    Task(id="development", label="Development", domain="mlops", order=50),
)

#: Kinds claimed outright, before anything looks at their contents.
_CONTAINER_KINDS: dict[str, tuple[str, str | None]] = {
    AssetKind.PROJECT.value: ("ai_project", "development"),
    AssetKind.ANNOTATION_PROJECT.value: ("annotation_project", None),
}

#: Kinds claimed only when no content-based rule wanted them.
_DERIVED_KINDS: dict[str, tuple[str, str | None]] = {
    AssetKind.CHECKPOINT.value: ("checkpoint", "checkpointing"),
    AssetKind.EXPERIMENT.value: ("experiment_log", "experiment_tracking"),
    AssetKind.MODEL.value: ("model", None),
}


def register(registry: TaxonomyRegistry) -> None:
    """Register the structural shelves and the classifier that fills them."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(
            id="checkpoint", label="Checkpoint", section="experiments", order=430,
            domain="mlops", aliases=("checkpoints", "ckpt"),
            description="A weight file saved partway through training.",
        )
    )
    registry.add_category(
        Category(
            id="ai_project", label="AI Project", section="experiments", order=390,
            domain="mlops", aliases=("projects", "repos", "codebases"),
            description="A codebase that trains, serves or evaluates models.",
        )
    )

    registry.add_category(
        Category(
            id="model", label="Model", section="models", order=500,
            domain="general", aliases=("unidentified", "generic-models"),
            description="A weight file whose purpose could not be determined.",
        )
    )

    registry.add_classifier(_container, name="structural.container", priority=PRIORITY_CONTAINER)
    registry.add_classifier(_derived, name="structural.derived", priority=PRIORITY_DERIVED)


def _container(profile: AssetProfile) -> Classification | None:
    """Claim the kinds whose identity is structural, before contents are consulted."""
    return _shelve(profile, _CONTAINER_KINDS)


def _derived(profile: AssetProfile) -> Classification | None:
    """Claim the kinds no content-based rule wanted."""
    return _shelve(profile, _DERIVED_KINDS)


def _shelve(
    profile: AssetProfile, table: dict[str, tuple[str, str | None]]
) -> Classification | None:
    """Classify from the kind the scanner recorded, if this table covers it."""
    match = table.get(str(profile.kind))
    if match is None:
        return None

    category, task = match
    return Classification(
        category=category,
        task=task,
        domain="mlops" if category != "model" else "general",
        confidence=CONFIDENCE_STRONG,
        evidence=f"detected as {profile.detector or profile.kind}",
    )
