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
from ai_asset_manager.backend.detectors.boundary import (
    MIN_UNSTRUCTURED_IMAGES,
    looks_like_dataset_root,
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


def _direct_class_counts(ctx: DirectoryContext) -> dict[str, int]:
    """Return a histogram of content classes for files directly in this directory."""
    counts: dict[str, int] = {}
    for entry in ctx.files:
        klass = entry.content_class
        if klass:
            counts[klass] = counts.get(klass, 0) + 1
    return counts


def _direct_extension_count(ctx: DirectoryContext, *extensions: str) -> int:
    """Return how many files directly in this directory carry any of these extensions."""
    wanted = {ext.lower() for ext in extensions}
    return sum(1 for entry in ctx.files if entry.extension in wanted)


#: Directories that hold supervision for the media beside them.
_LABEL_DIRS = ("labels", "annotations", "masks", "gt", "ground_truth", "captions",
               "transcripts", "metadata")


def _assembled(ctx: DirectoryContext) -> bool:
    """Report whether a pile of media shows signs of having been assembled deliberately.

    Three things count, and each is an act by whoever built the dataset rather than a
    by-product of downloading files into a folder: a split layout, a manifest naming the
    contents, or a directory of labels sitting beside the media.
    """
    if looks_like_dataset_root(ctx):
        return True
    if ctx.has_any_dir(*_LABEL_DIRS):
        return True
    return bool(ctx.glob("*.csv") or ctx.glob("*.jsonl") or ctx.glob("*.parquet"))


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
        """Emit one dataset when a ``v1.0-*`` metadata directory is present.

        ``samples/`` alone is not evidence. Plenty of directories keep a folder of samples,
        and on the development machine a folder of security logs with one ``samples/``
        child was confidently filed as an autonomous-driving dataset. nuScenes ships
        ``samples`` *and* ``sweeps`` together, so requiring the pair costs nothing real.
        """
        version_dirs = [
            name for name in ctx.child_dir_names if name.lower().startswith("v1.0")
        ]
        has_sensor_dirs = ctx.has_dir("samples", "sweeps")

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
        """Emit one dataset when MOT sequence descriptors sit close beneath this directory.

        A subtree-wide search would let any ancestor claim the dataset — and since the
        first match up the tree wins, that ancestor would be whichever container the
        sequences happen to live in. MOT's own layout puts ``seqinfo.ini`` exactly one
        level below the dataset root (``MOT17/train/MOT17-02-DPM/seqinfo.ini`` relative to
        ``MOT17/train``), so looking one level down finds the boundary rather than an
        arbitrary ancestor of it.
        """
        sequences = [
            child.name for child in ctx.children() if child.has("seqinfo.ini")
        ]
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
        """Emit one dataset when its own Arrow/Parquet shards sit beside metadata.

        Shards must be direct children of the candidate directory.  Counting an entire
        subtree lets a drive root with a README and a few unrelated Parquet files claim
        every dataset below it as one giant HuggingFace dataset.
        """
        parquet = sum(1 for entry in ctx.files if entry.extension == ".parquet")
        arrow = sum(1 for entry in ctx.files if entry.extension == ".arrow")
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


class IdxUbyteDetector(BaseDetector):
    """Detects an MNIST-family dataset stored in IDX binary format.

    MNIST, Fashion-MNIST, KMNIST and EMNIST all ship as four ``*-idx?-ubyte`` files and
    nothing else — no images on disk, no annotations, no manifest. Every heuristic in this
    module keys on structure that simply is not there, so torchvision's most-downloaded
    dataset was invisible until this existed.
    """

    name = "idx_ubyte"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset for a directory of IDX files."""
        idx_files = [
            entry.name
            for entry in ctx.files
            if "-idx" in entry.name.lower() and "ubyte" in entry.name.lower()
        ]
        if len(idx_files) < 2:
            return []

        # torchvision unpacks into `<name>/raw`; the dataset's identity is the parent.
        name = ctx.name
        if name.lower() in {"raw", "processed"} and ctx.parent_name:
            name = ctx.parent_name

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                name=name,
                subkind=DatasetFormat.IMAGE_CLASSIFICATION.value,
                confidence=0.9,
                evidence={
                    "idx_files": sorted(idx_files)[:12],
                    "compressed": ctx.count_extension(".gz"),
                    "modalities": [Modality.RGB.value],
                },
            )
        ]


class TabularDatasetDetector(BaseDetector):
    """Detects a corpus of tabular or record files.

    Requires the records to be what the directory *is*, measured over its own files rather
    than its subtree, and requires them to be substantial: two stray CSVs beside a hundred
    scripts is a project that reads data, not a dataset.
    """

    name = "tabular_dataset"
    priority = PRIORITY_DATASET_GENERIC

    #: Extensions that hold records rather than prose.
    RECORD_EXTENSIONS = (".csv", ".tsv", ".parquet", ".arrow", ".feather", ".jsonl")

    MIN_FILES = 2
    #: A real corpus is measured in megabytes. This filters out the sample files that
    #: accompany code without excluding a genuinely small labelled set.
    MIN_BYTES = 512 * 1024

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset for a directory of record files."""
        records = [
            entry for entry in ctx.files if entry.extension in self.RECORD_EXTENSIONS
        ]
        if len(records) < self.MIN_FILES:
            return []
        total = sum(entry.size for entry in records)
        if total < self.MIN_BYTES:
            return []
        if len(records) / max(1, len(ctx.files)) < 0.5:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                subkind=DatasetFormat.TABULAR.value,
                confidence=0.6,
                evidence={
                    "records": len(records),
                    "bytes": total,
                    "modalities": [Modality.TEXT.value],
                },
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

    #: ...unless the classes sit under named splits. ``train/{ants,bees}`` beside
    #: ``val/{ants,bees}`` is unambiguously an ImageFolder dataset — nobody arranges
    #: holiday photos that way — so two classes suffice once splits are present. Without
    #: this, the canonical two-class PyTorch tutorial set is claimed as four separate
    #: datasets, one per class folder, at the wrong boundary.
    MIN_CLASS_DIRS_WITH_SPLITS = 2

    #: Average images per class below which this is a folder-per-thing layout rather than
    #: a labelled dataset.
    MIN_IMAGES_PER_CLASS = 10

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

        # A class directory holds images and nothing else. Requiring that — rather than
        # merely "contains images somewhere below" — is what stops a folder that simply
        # *contains* several datasets from being claimed as one giant classification set,
        # swallowing every real dataset beneath it. A parent of datasets has children
        # with their own structure; a class folder is a leaf.
        class_dirs: list[str] = []
        class_images = 0
        for root in search_roots:
            for candidate in root.children():
                if candidate.direct_image_count > 0 and candidate.is_leaf:
                    class_dirs.append(candidate.name)
                    class_images += candidate.direct_image_count

        unique_classes = sorted(set(class_dirs))
        minimum = self.MIN_CLASS_DIRS_WITH_SPLITS if split_dirs else self.MIN_CLASS_DIRS
        if len(unique_classes) < minimum:
            return []

        # A class with one example in it is not a class. Any folder-per-thing layout looks
        # like this one — 110 Office add-in directories each holding a single icon, a
        # notes vault whose chapters each hold a screenshot — and only the density of
        # examples separates those from a training set.
        #
        # Counted over the class folders themselves rather than the subtree: a directory
        # can hold thousands of images that are not in any class folder, and dividing
        # those by the class count would let the layout pass on the strength of images
        # that have nothing to do with it.
        if class_images / len(class_dirs) < self.MIN_IMAGES_PER_CLASS:
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
    r"""Detects bulk video, audio or text corpora with no more specific structure.

    Last resort before a directory is left uncatalogued, and historically the source of the
    worst misclassifications this project has produced. Two rules keep it honest.

    *Counts are of files sitting directly here, never of the subtree.* A subtree count
    describes everything below a directory, which at ``F:\`` means the whole disk; since
    detection runs parents first and a claim suppresses its descendants, the rule fired at
    the top of the tree and swallowed everything real underneath. On the development
    machine two ``.jsonl`` files five levels down were enough to file an entire 372-
    directory project as one NLP dataset.

    *The media must be what the directory is.* A folder is a corpus when its contents are
    overwhelmingly one kind of thing, not when it merely contains ten of them somewhere.
    """

    name = "media_collection"
    priority = PRIORITY_DATASET_GENERIC

    MIN_VIDEOS = 10
    MIN_AUDIO = 50
    MIN_TEXT_RECORDS = 2

    #: Share of this directory's own files that must belong to the claimed modality. A
    #: project folder with a dozen sample clips beside its source code fails this; a
    #: directory of nothing but clips passes.
    DOMINANCE = 0.6

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset for a sufficiently large single-modality collection.

        Bulk alone is never enough. A folder of a thousand images and a folder of a
        thousand screenshots are the same shape, and so are a video dataset and a course
        download — on the development machine this rule claimed a Udemy course, three
        screen-recording folders and 1,070 screenshots, all of them correctly described as
        "lots of media" and none of them a dataset. Something must say that the media was
        *assembled* rather than merely accumulated: a split layout, a manifest, or labels
        beside it. That is the "strong structural evidence" a dataset has and a folder
        does not.
        """
        if not _assembled(ctx):
            return []

        direct = _direct_class_counts(ctx)
        total = len(ctx.files)
        videos = direct.get("video", 0)
        audio = direct.get("audio", 0)
        images = direct.get("image", 0)
        jsonl = _direct_extension_count(ctx, ".jsonl")
        parquet = _direct_extension_count(ctx, ".parquet")

        def dominant(count: int) -> bool:
            return total > 0 and count / total >= self.DOMINANCE

        if videos >= self.MIN_VIDEOS and dominant(videos):
            return [self._emit(ctx, DatasetFormat.VIDEO, Modality.VIDEO, {"videos": videos})]

        if audio >= self.MIN_AUDIO and dominant(audio):
            return [self._emit(ctx, DatasetFormat.AUDIO, Modality.AUDIO, {"audio": audio})]

        if jsonl + parquet >= self.MIN_TEXT_RECORDS and dominant(jsonl + parquet):
            return [
                self._emit(
                    ctx, DatasetFormat.NLP, Modality.TEXT, {"jsonl": jsonl, "parquet": parquet}
                )
            ]

        if images >= MIN_UNSTRUCTURED_IMAGES and dominant(images):
            return [
                self._emit(
                    ctx, DatasetFormat.CUSTOM, Modality.RGB, {"images": images}, confidence=0.5
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
