"""Knowledge every dataset shares, whatever domain it belongs to.

Storage-format recognition, split and content statistics, version detection, and the
health rules that apply to any collection of data. Domain-specific dataset knowledge —
what a KITTI calibration file is, why a YOLO ``data.yaml`` matters — lives in the plugin
for that domain.

Everything here is derived from the file list the scanner recorded. Nothing is opened.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    ANNOTATION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    audio_count,
    has_licence,
    has_readme,
    image_count,
    is_dataset,
    splits_present,
    video_count,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Finding,
)

#: How data is physically stored, tested in order. This is orthogonal to the *layout*
#: (COCO, YOLO) the scanner records: a COCO dataset can be a folder of JPEGs or a stack of
#: WebDataset shards, and which it is decides whether a training script can stream it.
#: Each entry is a name and a predicate over the asset's file list.
_STORAGE_TESTS: tuple[tuple[str, str], ...] = (
    ("lmdb", "data.mdb"),
    ("tfrecord", ".tfrecord"),
    ("hdf5", ".h5"),
    ("parquet", ".parquet"),
    ("arrow", ".arrow"),
    ("sqlite", ".sqlite"),
    ("jsonl", ".jsonl"),
    ("csv", ".csv"),
    ("tsv", ".tsv"),
)

#: A WebDataset is a set of numbered tar shards. One tar file is an archive; a run of them
#: named ``shard-000123.tar`` is a streaming format, and telling them apart matters.
_WEBDATASET_SHARD = re.compile(r"[-_]\d{3,}\.tar$")

#: Directory or file names carrying an explicit dataset version, e.g. nuScenes'
#: ``v1.0-trainval`` or a bare ``v2``.
_VERSION_PATTERN = re.compile(r"^v(\d+(?:\.\d+)*)")


def register(registry: TaxonomyRegistry) -> None:
    """Register generic dataset categories, statistics and health rules."""
    registry.add_category(
        Category(
            id="other_dataset",
            label="Other Dataset",
            section="datasets",
            order=910,
            domain="general",
            aliases=("other-datasets", "custom-datasets"),
            description="A dataset whose purpose no plugin could determine.",
        )
    )

    registry.add_classifier(_unrecognised_dataset, name="dataset-fallback", priority=10)
    registry.add_statistic(_dataset_statistics, name="dataset")
    registry.add_health_rule(_documentation, name="dataset.documentation")
    registry.add_health_rule(_splits, name="dataset.splits")
    registry.add_health_rule(_annotations, name="dataset.annotations")


def _unrecognised_dataset(profile: AssetProfile) -> Classification | None:
    """Claim any dataset no domain plugin recognised.

    Sits just above the universal fallback so that an unfamiliar dataset is still filed
    under datasets rather than disappearing into "unclassified".
    """
    if not is_dataset(profile):
        return None
    return Classification(
        category="other_dataset",
        task=(profile.dataset.task if profile.dataset else None),
        domain="general",
        confidence=CONFIDENCE_WEAK,
        evidence="catalogued as a dataset, purpose undetermined",
    )


def _dataset_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return what a dataset contains: samples, classes, splits and documentation."""
    if not is_dataset(profile):
        return {}

    stats: dict[str, Any] = {}
    details = profile.dataset

    for key, value in (
        ("images", image_count(profile)),
        ("videos", video_count(profile)),
        ("audio_files", audio_count(profile)),
    ):
        if value:
            stats[key] = value

    if details is not None:
        if details.num_annotations:
            stats["annotations"] = details.num_annotations
        if details.num_text_files:
            stats["text_files"] = details.num_text_files
        if details.num_classes:
            stats["classes"] = details.num_classes
        if details.class_names:
            stats["class_names"] = list(details.class_names[:32])
        if details.splits:
            stats["split_counts"] = dict(details.splits)
        if details.modalities:
            stats["modalities"] = list(details.modalities)

    splits = splits_present(profile)
    if splits:
        stats["splits"] = sorted(splits)

    if profile.files.loaded:
        stats["has_readme"] = has_readme(profile)
        stats["has_license"] = has_licence(profile)

        storage = _storage_format(profile)
        if storage:
            stats["storage_format"] = storage

        annotation_files = _annotation_file_count(profile)
        if annotation_files:
            stats["annotation_files"] = annotation_files

        # Average bytes per image is the closest honest proxy for resolution available
        # without decoding a single file. It separates a thumbnail set from a set of
        # 4K frames, which is usually the question being asked.
        image_files = profile.files.count(*IMAGE_EXTENSIONS)
        if image_files:
            stats["avg_image_bytes"] = profile.files.bytes_in(*IMAGE_EXTENSIONS) // image_files

    version = _version_of(profile)
    if version:
        stats["version"] = version

    return stats


def _storage_format(profile: AssetProfile) -> str | None:
    """Return how the dataset's samples are physically stored."""
    files = profile.files

    if any(_WEBDATASET_SHARD.search(relpath) for relpath in files.relpaths):
        return "webdataset"

    if files.has_name("dataset_info.json", "dataset_infos.json", "state.json"):
        return "hf_dataset"

    for label, needle in _STORAGE_TESTS:
        if needle.startswith("."):
            if files.count(needle):
                return label
        elif files.has_name(needle):
            return label

    if files.count(*IMAGE_EXTENSIONS):
        return "imagefolder"

    return None


def _annotation_file_count(profile: AssetProfile) -> int:
    """Return how many files look like labels rather than samples.

    Counted from paths rather than extensions alone: a ``.txt`` under ``labels/`` is an
    annotation, while a ``.txt`` at the root is probably a note.
    """
    marked = sum(
        1
        for relpath in profile.files.relpaths
        if ("label" in relpath or "annotation" in relpath or "/gt" in relpath)
        and relpath.endswith(ANNOTATION_EXTENSIONS)
    )
    return marked


def _version_of(profile: AssetProfile) -> str | None:
    """Return an explicitly declared dataset version, if one is visible in the layout."""
    for name in sorted(profile.files.top_level):
        match = _VERSION_PATTERN.match(name)
        if match:
            return f"v{match.group(1)}"
    return None


def _documentation(profile: AssetProfile) -> Sequence[Finding]:
    """Report missing documentation and licensing.

    Informational rather than warnings: an undocumented dataset still trains a model. It
    is the thing you regret six months later, not today, and the score should say so.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()

    findings: list[Finding] = []

    if not has_readme(profile):
        findings.append(
            Finding(
                code="dataset.no_readme",
                severity=Severity.INFO,
                message="No README or dataset card",
                fix_hint="Add a README recording where this came from and what it labels.",
            )
        )

    if not has_licence(profile):
        findings.append(
            Finding(
                code="dataset.no_license",
                severity=Severity.INFO,
                message="No licence file",
                fix_hint="Record the licence before using this in anything you ship.",
            )
        )

    return findings


def _splits(profile: AssetProfile) -> Sequence[Finding]:
    """Report missing train/validation/test splits.

    A dataset with no validation split cannot tell you whether training worked, so this is
    a warning rather than a note. A dataset with no splits at all is not necessarily
    broken — plenty are meant to be split at load time — so that is only reported when the
    layout shows the author intended splits and one is absent.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()

    present = splits_present(profile)
    if not present:
        return ()

    findings: list[Finding] = []

    if "train" not in present:
        findings.append(
            Finding(
                code="dataset.no_train_split",
                severity=Severity.WARNING,
                message=f"Split layout present ({', '.join(sorted(present))}) but no train split",
                fix_hint="Check the download completed; the training data may be missing.",
            )
        )

    if "val" not in present:
        findings.append(
            Finding(
                code="dataset.no_val_split",
                severity=Severity.WARNING,
                message="No validation split",
                fix_hint="Hold out part of the training set, or fetch the official split.",
            )
        )

    if "test" not in present:
        findings.append(
            Finding(
                code="dataset.no_test_split",
                severity=Severity.INFO,
                message="No test split",
                fix_hint="Many datasets withhold theirs; ignore if that is the case here.",
            )
        )

    return findings


def _annotations(profile: AssetProfile) -> Sequence[Finding]:
    """Report samples with nothing labelling them.

    Only fires when the dataset holds images and the catalogue recorded no annotations and
    no label files exist. An unlabelled image folder is a legitimate thing to own — for
    self-supervised training, or as a scrape awaiting labelling — so this is a warning
    that says what is missing, not an error.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()

    images = image_count(profile)
    if images < 1:
        return ()

    details = profile.dataset
    if details is not None and (details.num_annotations or details.num_classes):
        return ()

    if _annotation_file_count(profile) or profile.files.has_name(
        "data.yaml", "dataset.yaml", "annotations.json"
    ):
        return ()

    return (
        Finding(
            code="dataset.no_annotations",
            severity=Severity.WARNING,
            message=f"{images:,} image(s) with no annotations found",
            fix_hint="Unlabelled data. Fine for pretraining, useless for supervised work.",
        ),
    )
