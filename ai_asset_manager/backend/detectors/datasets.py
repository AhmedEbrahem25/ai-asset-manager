"""Dataset detectors.

Identification is by *layout*, not by name. A folder called ``coco`` proves nothing — it
might hold three sample images — while a folder holding ``annotations/instances_train.json``
with ``images``, ``annotations`` and ``categories`` keys is a COCO dataset whatever it is
called. Every detector here keys on structure that the dataset's own tooling requires.

Large annotation files are inspected by streaming their opening bytes rather than by
parsing them: a COCO ``instances_train2017.json`` is ~450 MB, and ``json.load`` on one per
candidate directory would dominate the scan.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ai_asset_manager.backend.detectors.base import (
    PRIORITY_DATASET_GENERIC,
    PRIORITY_DATASET_SPECIFIC,
    BaseDetector,
    DetectionResult,
)
from ai_asset_manager.backend.models.enums import AssetKind, DatasetFormat, Modality
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Bytes read from the head of a large JSON file when probing for marker keys.
JSON_PROBE_BYTES = 64 * 1024

#: Directory names conventionally used for dataset splits.
SPLIT_NAMES = frozenset(
    {"train", "val", "valid", "validation", "test", "testing", "training", "eval", "dev"}
)

#: A minimum image count before a directory of pictures is called a dataset. Below this
#: it is far more likely to be a folder of screenshots or sample outputs.
MIN_DATASET_IMAGES = 20


def probe_json_keys(
    path: str, keys: tuple[str, ...], *, probe_bytes: int = JSON_PROBE_BYTES
) -> bool:
    """Report whether a JSON file's opening bytes mention all the given keys.

    A substring probe, not a parse. Annotation files reach hundreds of megabytes and
    their top-level keys always appear near the start, so this answers the question at a
    fraction of the cost. A false positive only means a slightly wrong dataset label.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(probe_bytes)
    except OSError:
        return False
    text = head.decode("utf-8", errors="replace")
    return all(f'"{key}"' in text for key in keys)


def load_json_head(path: str, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any] | None:
    """Parse a JSON file only when it is small enough to be worth parsing whole."""
    try:
        if os.path.getsize(path) > max_bytes:
            return None
        with open(path, encoding="utf-8", errors="replace") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def find_split_dirs(ctx: DirectoryContext) -> list[str]:
    """Return immediate subdirectory names that are dataset splits.

    Uses the same exact-match rule as the metadata parsers, so a directory named
    ``latest`` or ``pretrained`` is not mistaken for a split.
    """
    from ai_asset_manager.backend.parsers.dataset_meta import canonical_split

    return sorted(name for name in ctx.child_dir_names if canonical_split(name))


class CocoDetector(BaseDetector):
    """Detects a COCO dataset by its annotation schema."""

    name = "coco"
    priority = PRIORITY_DATASET_SPECIFIC

    #: Keys every COCO annotation file carries.
    MARKER_KEYS = ("images", "annotations", "categories")

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when COCO-shaped annotations are present."""
        annotation_dir = ctx.child("annotations")
        candidates = (
            annotation_dir.glob("*.json") if annotation_dir else ctx.glob("*.json")
        )
        matched = [
            entry for entry in candidates if probe_json_keys(entry.path, self.MARKER_KEYS)
        ]
        if not matched:
            return []

        # Distinguish COCO proper from LVIS, which shares the schema but adds its own keys.
        is_lvis = any(
            probe_json_keys(entry.path, ("images", "annotations", "categories", "synset"))
            for entry in matched
        )

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.LVIS.value if is_lvis else DatasetFormat.COCO.value,
                confidence=0.95,
                evidence={
                    "annotation_files": [entry.name for entry in matched][:20],
                    "in_annotations_dir": annotation_dir is not None,
                    "images": ctx.image_count,
                },
            )
        ]


class YoloDetector(BaseDetector):
    """Detects a YOLO dataset: a ``data.yaml`` plus parallel ``images``/``labels`` trees."""

    name = "yolo_dataset"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when the YOLO layout is present."""
        has_manifest = ctx.has_any("data.yaml", "data.yml", "dataset.yaml")
        has_parallel_dirs = ctx.has_dir("images") and ctx.has_dir("labels")

        if not has_manifest and not has_parallel_dirs:
            return []

        # A manifest alone could be any YAML; require label files to confirm. Labels are
        # `.txt` files sitting beside images, which is the format's defining trait.
        label_count = ctx.count_extension(".txt")
        if label_count == 0 and not has_parallel_dirs:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.YOLO.value,
                confidence=0.95 if has_manifest and has_parallel_dirs else 0.75,
                evidence={
                    "manifest": has_manifest,
                    "parallel_dirs": has_parallel_dirs,
                    "labels": label_count,
                    "images": ctx.image_count,
                },
            )
        ]


class PascalVocDetector(BaseDetector):
    """Detects a Pascal VOC dataset by its ``Annotations``/``ImageSets`` directories."""

    name = "pascal_voc"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when the VOC directory layout is present."""
        has_annotations = ctx.has_any_dir("Annotations")
        has_imagesets = ctx.has_any_dir("ImageSets")
        has_jpeg = ctx.has_any_dir("JPEGImages")

        markers = sum((has_annotations, has_imagesets, has_jpeg))
        if markers < 2:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.PASCAL_VOC.value,
                confidence=0.9 if markers == 3 else 0.75,
                evidence={
                    "annotations": has_annotations,
                    "imagesets": has_imagesets,
                    "jpegimages": has_jpeg,
                    "xml_files": ctx.count_extension(".xml"),
                },
            )
        ]


class CityscapesDetector(BaseDetector):
    """Detects Cityscapes by its paired image and ground-truth directories."""

    name = "cityscapes"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when the Cityscapes directory pairing is present."""
        if not ctx.has_any_dir("leftImg8bit"):
            return []

        ground_truth = [
            name for name in ctx.child_dir_names if name.lower().startswith("gt")
        ]
        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.CITYSCAPES.value,
                confidence=0.95 if ground_truth else 0.8,
                evidence={"ground_truth_dirs": ground_truth, "images": ctx.image_count},
            )
        ]


class KittiDetector(BaseDetector):
    """Detects a KITTI dataset by its sensor-directory naming."""

    name = "kitti"
    priority = PRIORITY_DATASET_SPECIFIC

    #: KITTI's directory names for each sensor stream.
    SENSOR_DIRS = ("image_2", "image_3", "velodyne", "calib", "label_2", "oxts")

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when at least two KITTI sensor directories are present."""
        present = [name for name in self.SENSOR_DIRS if ctx.has_any_dir(name)]
        if len(present) < 2:
            return []

        modalities = [Modality.RGB.value]
        if "velodyne" in present:
            modalities.append(Modality.LIDAR.value)

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.KITTI.value,
                confidence=0.9,
                evidence={"sensor_dirs": present, "modalities": modalities},
            )
        ]


class NuScenesDetector(BaseDetector):
    """Detects nuScenes by its versioned metadata directory."""

    name = "nuscenes"
    priority = PRIORITY_DATASET_SPECIFIC

    #: The table files nuScenes ships in every release.
    TABLE_FILES = ("sample.json", "scene.json", "sample_data.json", "ego_pose.json")

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when a ``v1.0-*`` metadata directory is present."""
        version_dirs = [
            name for name in ctx.child_dir_names if name.lower().startswith("v1.0")
        ]
        has_sensor_dirs = ctx.has_any_dir("samples", "sweeps")

        if not version_dirs and not has_sensor_dirs:
            return []

        tables_found: list[str] = []
        for version in version_dirs:
            child = ctx.child(version)
            if child is not None:
                tables_found = [name for name in self.TABLE_FILES if child.has(name)]
                if tables_found:
                    break

        if not tables_found and not has_sensor_dirs:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.NUSCENES.value,
                confidence=0.9 if tables_found else 0.6,
                evidence={
                    "version_dirs": version_dirs,
                    "tables": tables_found,
                    "modalities": [Modality.RGB.value, Modality.LIDAR.value,
                                   Modality.RADAR.value],
                },
            )
        ]


class MotDetector(BaseDetector):
    """Detects a MOT-challenge tracking dataset by its ``seqinfo.ini`` sequence files."""

    name = "mot"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when MOT sequence descriptors are present."""
        sequences = ctx.glob_subtree("seqinfo.ini", limit=64)
        if not sequences:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.MOT.value,
                confidence=0.95,
                evidence={"sequences": len(sequences), "images": ctx.image_count},
            )
        ]


class Bdd100kDetector(BaseDetector):
    """Detects BDD100K by its label and image directory naming."""

    name = "bdd100k"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when BDD100K markers are present."""
        lowered = ctx.name.lower()
        has_marker_dirs = ctx.has_any_dir("bdd100k") or (
            ctx.has_any_dir("images") and ctx.has_any_dir("labels")
            and "bdd" in lowered
        )
        if not has_marker_dirs and "bdd100k" not in lowered:
            return []
        if ctx.image_count < MIN_DATASET_IMAGES and not ctx.has_any_dir("bdd100k"):
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.BDD100K.value,
                confidence=0.7,
                evidence={"images": ctx.image_count},
            )
        ]


class HfDatasetCacheDetector(BaseDetector):
    """Detects a HuggingFace dataset by its Arrow/Parquet payload and metadata."""

    name = "hf_dataset"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when Arrow or Parquet shards sit beside dataset metadata."""
        parquet = ctx.count_extension(".parquet")
        arrow = ctx.count_extension(".arrow")
        has_metadata = ctx.has_any(
            "dataset_info.json", "dataset_infos.json", "state.json", "README.md"
        )

        if parquet + arrow == 0:
            return []
        if not has_metadata and parquet + arrow < 2:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.HF_DATASET.value,
                confidence=0.85,
                evidence={"parquet": parquet, "arrow": arrow, "metadata": has_metadata},
            )
        ]


class ImageClassificationDetector(BaseDetector):
    """Detects a directory-per-class image dataset.

    The convention behind ``ImageFolder`` and ImageNet: every immediate subdirectory is a
    class label and holds that class's images. Recognised by shape, since the class names
    are arbitrary.
    """

    name = "image_classification"
    priority = PRIORITY_DATASET_GENERIC

    #: A dataset needs several classes; two directories of pictures is not a taxonomy.
    MIN_CLASS_DIRS = 3

    #: WordNet ids (``n01440764``), which mark a directory as ImageNet specifically.
    WNID_RE = re.compile(r"^n\d{8}$")

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when a class-per-directory layout holds enough images."""
        if ctx.image_count < MIN_DATASET_IMAGES:
            return []

        split_dirs = find_split_dirs(ctx)
        search_roots = (
            [child for name in split_dirs if (child := ctx.child(name)) is not None]
            if split_dirs
            else [ctx]
        )

        class_dirs: list[str] = []
        for root in search_roots:
            for candidate in root.children():
                if candidate.image_count > 0:
                    class_dirs.append(candidate.name)

        unique_classes = sorted(set(class_dirs))
        if len(unique_classes) < self.MIN_CLASS_DIRS:
            return []

        is_imagenet = sum(
            1 for name in unique_classes if self.WNID_RE.match(name)
        ) >= max(3, len(unique_classes) // 2)

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=(
                    DatasetFormat.IMAGENET.value
                    if is_imagenet
                    else DatasetFormat.IMAGE_CLASSIFICATION.value
                ),
                confidence=0.85 if is_imagenet else 0.65,
                evidence={
                    "classes": len(unique_classes),
                    "class_names": unique_classes[:100],
                    "splits": split_dirs,
                    "images": ctx.image_count,
                },
            )
        ]


class MediaCollectionDetector(BaseDetector):
    """Detects bulk video, audio or text corpora with no more specific structure.

    Last resort before a directory is left uncatalogued. Deliberately conservative: it
    requires a large, homogeneous body of media so that ordinary media folders — a music
    library, a directory of holiday photos — are not filed as training data.
    """

    name = "media_collection"
    priority = PRIORITY_DATASET_GENERIC

    MIN_VIDEOS = 10
    MIN_AUDIO = 50
    MIN_TEXT_RECORDS = 2

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset for a sufficiently large single-modality collection."""
        videos = ctx.video_count
        audio = ctx.audio_count
        images = ctx.image_count
        jsonl = ctx.count_extension(".jsonl")
        parquet = ctx.count_extension(".parquet")

        if videos >= self.MIN_VIDEOS and videos > images:
            return [self._emit(ctx, DatasetFormat.VIDEO, Modality.VIDEO, {"videos": videos})]

        if audio >= self.MIN_AUDIO and audio > images:
            return [self._emit(ctx, DatasetFormat.AUDIO, Modality.AUDIO, {"audio": audio})]

        if jsonl + parquet >= self.MIN_TEXT_RECORDS and images < MIN_DATASET_IMAGES:
            return [
                self._emit(
                    ctx, DatasetFormat.NLP, Modality.TEXT, {"jsonl": jsonl, "parquet": parquet}
                )
            ]

        if images >= MIN_DATASET_IMAGES and not ctx.child_dir_names:
            # A flat pile of images with no labels anywhere: real, but unstructured.
            return [
                self._emit(
                    ctx, DatasetFormat.CUSTOM, Modality.RGB, {"images": images}, confidence=0.4
                )
            ]

        return []

    def _emit(
        self,
        ctx: DirectoryContext,
        dataset_format: DatasetFormat,
        modality: Modality,
        evidence: dict[str, int],
        *,
        confidence: float = 0.6,
    ) -> DetectionResult:
        """Build a generic-collection result."""
        return self._result(
            ctx,
            kind=AssetKind.DATASET,
            subkind=dataset_format.value,
            confidence=confidence,
            evidence={**evidence, "modalities": [modality.value]},
        )
