"""Annotation projects.

Half-finished labelling work: a CVAT export, a Label Studio project, a Roboflow download.
Worth a shelf of its own because it is neither a finished dataset nor disposable — it is
usually the most expensive thing on the disk per byte, since somebody's time made it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import image_count
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    AssetProfile,
    Category,
    Classification,
    Finding,
    Task,
)

#: Annotation tools and the files their exports contain.
#: ``(label, required filenames, required directories)`` — a tool matches when any listed
#: filename is present, or any listed directory is.
_TOOLS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("CVAT", ("annotations.xml", "task.json", "index.json"), ("task_", ".cvat")),
    ("Label Studio", ("label_config.xml", "tasks.json", "project.json"), ("label-studio",)),
    ("Roboflow", ("readme.roboflow.txt", "readme.dataset.txt"), ()),
    ("Supervisely", ("meta.json", "key_id_map.json"), ("ann", "img")),
    ("FiftyOne", ("samples.json", "fiftyone.yml"), ("fiftyone",)),
    ("VoTT", (".vott",), ()),
    ("makesense.ai", ("labels_my-project-name.zip",), ()),
    ("Doccano", ("doccano.jsonl",), ()),
)

TASKS = (
    Task(id="annotation", label="Annotation", domain="general", order=20),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the annotation-project shelf and its rules."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="annotation_project", label="Annotation Project", section="datasets",
                 order=290, domain="general",
                 aliases=("annotations", "labelling", "labeling"),
                 description="Labelling work in progress or exported from a tool.")
    )

    registry.add_classifier(_annotation_project, name="annotation.project", priority=830)
    registry.add_statistic(_annotation_statistics, name="annotation")
    registry.add_health_rule(_unlabelled_images, name="annotation.incomplete")


def _tool_of(profile: AssetProfile) -> str | None:
    """Return the annotation tool that produced this directory, if recognisable."""
    files = profile.files
    for label, names, directories in _TOOLS:
        if names and files.has_name(*names):
            return label
        if directories and files.has_dir(*directories):
            return label
    return None


def _annotation_project(profile: AssetProfile) -> Classification | None:
    """Claim exports and working directories from annotation tools.

    Supervisely and CVAT are matched on distinctive filename pairs rather than single
    names: ``meta.json`` alone appears in half the model repositories on a machine, and
    claiming those as labelling projects would be worse than missing a few real ones.
    """
    if not profile.files.loaded:
        return None

    tool = _tool_of(profile)
    if tool is None:
        return None

    # A directory with both a tool fingerprint and no images at all is more likely to be a
    # configuration folder than a project.
    if tool in ("Supervisely", "CVAT") and not image_count(profile):
        return None

    return Classification(
        category="annotation_project", task="annotation", domain="general", family=tool,
        modalities=("rgb",), confidence=CONFIDENCE_CERTAIN,
        evidence=f"{tool} project files",
    )


def _annotation_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return the tool and the size of the labelling job."""
    if not profile.files.loaded:
        return {}

    tool = _tool_of(profile)
    if tool is None:
        return {}

    stats: dict[str, Any] = {"annotation_tool": tool}
    images = image_count(profile)
    if images:
        stats["images"] = images
    return stats


def _unlabelled_images(profile: AssetProfile) -> Sequence[Finding]:
    """Report a labelling project with far more images than annotations.

    Says how much of the job is left, which is the one thing a half-finished annotation
    project should be able to tell you.
    """
    if not profile.files.loaded or _tool_of(profile) is None:
        return ()

    details = profile.dataset
    images = image_count(profile)
    annotations = details.num_annotations if details else 0

    if not images or not annotations or annotations >= images * 0.9:
        return ()

    remaining = images - annotations
    return (
        Finding(
            code="annotation.incomplete",
            severity=Severity.INFO,
            message=f"{remaining:,} of {images:,} image(s) still unlabelled",
            fix_hint="Labelling is unfinished; do not treat this as a finished dataset.",
        ),
    )
