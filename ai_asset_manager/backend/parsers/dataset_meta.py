"""Dataset metadata parsers.

Recovers class lists, annotation counts and split sizes from the manifests each dataset
format ships: a YOLO ``data.yaml``, a COCO ``categories`` block, a VOC ``ImageSets``
index.

Annotation files are the difficulty. A COCO ``instances_train2017.json`` is around
450 MB, and ``json.load`` on one per candidate directory would dominate a scan. Small
files are parsed properly; large ones are counted by streaming for marker substrings,
which gives an accurate count for a fixed, small amount of memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import yaml

from ai_asset_manager.backend.models.enums import DatasetFormat, FactSource, Modality
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import FileEntry
from ai_asset_manager.backend.utils.paths import safe_relpath
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Files up to this size are parsed as JSON; larger ones are stream-counted.
FULL_PARSE_LIMIT = 48 * 1024 * 1024

#: Read size when stream-counting a large annotation file.
STREAM_CHUNK_BYTES = 4 * 1024 * 1024

#: Directory names conventionally used for splits, mapped to a canonical label.
SPLIT_ALIASES: dict[str, str] = {
    "train": "train", "training": "train", "train2017": "train", "train2014": "train",
    "val": "val", "valid": "val", "validation": "val", "val2017": "val", "val2014": "val",
    "test": "test", "testing": "test", "test2017": "test", "eval": "val", "dev": "val",
}


def canonical_split(name: str) -> str | None:
    """Map a directory or filename to a canonical split label.

    Matching is exact, after dropping any file extension and any trailing year digits, so
    that ``train2017`` and ``val.txt`` resolve but ``test_yolo0``, ``latest`` and
    ``pretrained`` do not. Substring matching was tried and is actively wrong here: it
    lets an unrelated ancestor directory name capture every file beneath it.

    Examples:
        >>> canonical_split("train2017")
        'train'
        >>> canonical_split("instances_val2017.json")
        'val'
        >>> canonical_split("latest") is None
        True
        >>> canonical_split("test_yolo0") is None
        True
    """
    stem = name.lower().rsplit(".", 1)[0] if "." in name else name.lower()

    for candidate in (stem, stem.rstrip("0123456789")):
        if candidate in SPLIT_ALIASES:
            return SPLIT_ALIASES[candidate]

    # Annotation files are named `instances_train2017.json`; the split is the last
    # underscore-delimited token, checked on its own rather than against the whole name.
    if "_" in stem:
        tail = stem.rsplit("_", 1)[1]
        for candidate in (tail, tail.rstrip("0123456789")):
            if candidate in SPLIT_ALIASES:
                return SPLIT_ALIASES[candidate]

    return None


def count_occurrences(path: str, needle: bytes) -> int:
    """Count occurrences of a byte sequence in a file, streaming.

    Chunks overlap by ``len(needle) - 1`` bytes so a match straddling a boundary is not
    missed — the classic off-by-one in this pattern, and one that silently undercounts.
    """
    total = 0
    overlap = len(needle) - 1
    try:
        with open(path, "rb") as handle:
            tail = b""
            while chunk := handle.read(STREAM_CHUNK_BYTES):
                buffer = tail + chunk
                total += buffer.count(needle)
                tail = buffer[-overlap:] if overlap else b""
    except OSError as exc:
        logger.debug("Cannot stream %s: %s", path, exc)
        return 0
    return total


class CocoMetadataParser(BaseParser):
    """Extracts categories, image and annotation counts from COCO annotation files."""

    name = "coco_meta"

    MARKER_KEYS = ("images", "annotations", "categories")

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether COCO-shaped annotation files are present."""
        return bool(self._annotation_files(ctx))

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Read class names and per-split counts across every annotation file."""
        facts = self._new_facts()
        files = self._annotation_files(ctx)
        if not files:
            return facts

        class_names: list[str] = []
        splits: dict[str, int] = {}
        total_images = 0
        total_annotations = 0
        has_masks = False
        has_keypoints = False

        for entry in files:
            summary = self._summarise(entry.path, entry.size)
            if summary is None:
                continue

            images, annotations, categories, flags = summary
            total_images += images
            total_annotations += annotations
            has_masks = has_masks or flags.get("segmentation", False)
            has_keypoints = has_keypoints or flags.get("keypoints", False)

            if categories and not class_names:
                class_names = categories

            split = canonical_split(entry.name)
            if split and images:
                splits[split] = splits.get(split, 0) + images

        facts.add("dataset_format", DatasetFormat.COCO.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("num_images", total_images or None, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("num_annotations", total_annotations or None,
                  source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("class_names", class_names or None, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("num_classes", len(class_names) or None, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("splits", splits or None, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("has_bounding_boxes", True, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("has_masks", has_masks or None, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("has_keypoints", has_keypoints or None, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("modalities", [Modality.RGB.value], source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        return facts

    def _annotation_files(self, ctx: DirectoryContext) -> list[FileEntry]:
        """Return JSON files whose opening bytes carry the COCO marker keys."""
        from ai_asset_manager.backend.detectors.datasets import probe_json_keys

        annotation_dir = ctx.child("annotations")
        candidates = annotation_dir.glob("*.json") if annotation_dir else ctx.glob("*.json")
        return [
            entry for entry in candidates if probe_json_keys(entry.path, self.MARKER_KEYS)
        ][:12]

    def _summarise(
        self, path: str, size: int
    ) -> tuple[int, int, list[str], dict[str, bool]] | None:
        """Return ``(images, annotations, categories, flags)`` for one annotation file."""
        if size <= FULL_PARSE_LIMIT:
            return self._parse_fully(path)
        return self._stream_count(path)

    def _parse_fully(self, path: str) -> tuple[int, int, list[str], dict[str, bool]] | None:
        """Parse a small annotation file exactly."""
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, ValueError, MemoryError) as exc:
            logger.debug("Cannot parse %s: %s", path, exc)
            return None
        if not isinstance(data, dict):
            return None

        images = data.get("images")
        annotations = data.get("annotations")
        categories = data.get("categories")

        names: list[str] = []
        if isinstance(categories, list):
            names = [
                str(item["name"])
                for item in categories
                if isinstance(item, dict) and "name" in item
            ]

        flags = {"segmentation": False, "keypoints": False}
        if isinstance(annotations, list):
            for annotation in annotations[:200]:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("segmentation"):
                    flags["segmentation"] = True
                if annotation.get("keypoints"):
                    flags["keypoints"] = True

        return (
            len(images) if isinstance(images, list) else 0,
            len(annotations) if isinstance(annotations, list) else 0,
            names,
            flags,
        )

    def _stream_count(self, path: str) -> tuple[int, int, list[str], dict[str, bool]]:
        """Count a large annotation file without loading it.

        ``"image_id"`` appears once per annotation and ``"file_name"`` once per image, so
        counting them yields the totals directly. Category names are read from the head
        of the file, where COCO writers place them.
        """
        annotations = count_occurrences(path, b'"image_id"')
        images = count_occurrences(path, b'"file_name"')
        has_segmentation = count_occurrences(path, b'"segmentation": [[') > 0
        has_keypoints = count_occurrences(path, b'"keypoints"') > 0

        names: list[str] = []
        try:
            with open(path, "rb") as handle:
                head = handle.read(256 * 1024).decode("utf-8", errors="replace")
                tail_marker = head.find('"categories"')
                window = head[tail_marker:] if tail_marker != -1 else head
                names = re.findall(r'"name"\s*:\s*"([^"]{1,64})"', window)[:1000]
        except OSError:
            pass

        return images, annotations, names, {
            "segmentation": has_segmentation,
            "keypoints": has_keypoints,
        }


class YoloDatasetMetadataParser(BaseParser):
    """Extracts classes and split sizes from a YOLO ``data.yaml`` and label tree."""

    name = "yolo_meta"

    MANIFEST_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a YOLO manifest or label tree is present."""
        return ctx.has_any(*self.MANIFEST_NAMES) or (
            ctx.has_dir("images") and ctx.has_dir("labels")
        )

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Read class names from the manifest and count labels from the tree."""
        facts = self._new_facts()
        facts.add("dataset_format", DatasetFormat.YOLO.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)

        manifest = self._read_manifest(ctx)
        if manifest:
            names = manifest.get("names")
            class_names = self._normalise_names(names)
            if class_names:
                facts.add("class_names", class_names, source=FactSource.EXPLICIT_CONFIG,
                          origin=self.name)
            declared = manifest.get("nc")
            facts.add(
                "num_classes",
                declared if isinstance(declared, int) else (len(class_names) or None),
                source=FactSource.EXPLICIT_CONFIG,
                origin=self.name,
            )

        label_files = [
            entry for entry in ctx.subtree_files
            if entry.extension == ".txt" and "label" in entry.path.lower()
        ]
        if label_files:
            facts.add("num_annotations", len(label_files), source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        splits = self._count_splits(ctx)
        if splits:
            facts.add("splits", splits, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        facts.add("has_bounding_boxes", True, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("modalities", [Modality.RGB.value], source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("task", "object-detection", source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        return facts

    def _read_manifest(self, ctx: DirectoryContext) -> dict[str, Any] | None:
        """Parse the YOLO manifest, whichever of its spellings is present."""
        for filename in self.MANIFEST_NAMES:
            text = ctx.read_text(filename)
            if not text:
                continue
            try:
                parsed = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                logger.debug("Malformed %s in %s: %s", filename, ctx.path, exc)
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _normalise_names(self, names: Any) -> list[str]:
        """Coerce the ``names`` field, which may be a list or an index-keyed mapping."""
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            # Keys are integer indices; sort numerically so class order is preserved.
            try:
                return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
            except (ValueError, TypeError):
                return [str(value) for value in names.values()]
        return []

    def _count_splits(self, ctx: DirectoryContext) -> dict[str, int]:
        """Count images per split from the directory layout.

        Only path components *below* the dataset root are considered. Components above it
        belong to the user's own folder hierarchy and have no bearing on the dataset's
        splits, so letting them match would misattribute every file at once.
        """
        counts: Counter[str] = Counter()
        for entry in ctx.subtree_files:
            if entry.content_class != "image":
                continue
            for part in safe_relpath(entry.path, ctx.path).split("/")[:-1]:
                split = canonical_split(part)
                if split:
                    counts[split] += 1
                    break
        return dict(counts)


class VocMetadataParser(BaseParser):
    """Extracts classes and split sizes from a Pascal VOC tree."""

    name = "voc_meta"

    #: Number of annotation files sampled for class names. Reading every XML of a
    #: 20,000-image dataset to learn a 20-item class list would be pure waste.
    SAMPLE_SIZE = 200

    _NAME_RE = re.compile(r"<name>\s*([^<]{1,64})\s*</name>")

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a VOC annotation directory is present."""
        return ctx.has_any_dir("Annotations")

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Sample annotation XML for class names and read the split index files."""
        facts = self._new_facts()
        facts.add("dataset_format", DatasetFormat.PASCAL_VOC.value,
                  source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        annotations = ctx.child("Annotations")
        if annotations is not None:
            xml_files = annotations.glob("*.xml")
            facts.add("num_annotations", len(xml_files) or None,
                      source=FactSource.EXPLICIT_CONFIG, origin=self.name)

            found: set[str] = set()
            for entry in xml_files[: self.SAMPLE_SIZE]:
                try:
                    with open(entry.path, encoding="utf-8", errors="replace") as handle:
                        found.update(self._NAME_RE.findall(handle.read(64 * 1024)))
                except OSError:
                    continue
            if found:
                class_names = sorted(found)
                facts.add("class_names", class_names, source=FactSource.EXPLICIT_CONFIG,
                          confidence=0.8, origin=self.name)
                facts.add("num_classes", len(class_names), source=FactSource.EXPLICIT_CONFIG,
                          confidence=0.8, origin=self.name)

        splits = self._read_image_sets(ctx)
        if splits:
            facts.add("splits", splits, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        facts.add("has_bounding_boxes", True, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("has_masks", ctx.has_any_dir("SegmentationClass") or None,
                  source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("modalities", [Modality.RGB.value], source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        return facts

    def _read_image_sets(self, ctx: DirectoryContext) -> dict[str, int]:
        """Count entries in each ``ImageSets`` index file."""
        image_sets = ctx.child("ImageSets")
        if image_sets is None:
            return {}

        counts: dict[str, int] = {}
        for source in [image_sets, *image_sets.children()]:
            for entry in source.glob("*.txt"):
                split = canonical_split(entry.name)
                if split is None or entry.size > 8 * 1024 * 1024:
                    continue
                try:
                    with open(entry.path, encoding="utf-8", errors="replace") as handle:
                        counts[split] = sum(1 for line in handle if line.strip())
                except OSError:
                    continue
        return counts


class ImageFolderMetadataParser(BaseParser):
    """Extracts class names from a directory-per-class image dataset."""

    name = "imagefolder_meta"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether the directory holds a class-per-subdirectory image layout."""
        return ctx.image_count > 0 and bool(ctx.child_dir_names)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Derive class names and per-split counts from the directory names."""
        facts = self._new_facts()

        split_children = [
            child for child in ctx.children() if canonical_split(child.name) is not None
        ]
        search_roots = split_children or [ctx]

        class_names: set[str] = set()
        splits: dict[str, int] = {}

        for root in search_roots:
            split = canonical_split(root.name) if split_children else None
            images_in_split = 0
            for candidate in root.children():
                if candidate.image_count > 0:
                    class_names.add(candidate.name)
                    images_in_split += candidate.image_count
            if split and images_in_split:
                splits[split] = splits.get(split, 0) + images_in_split

        if not class_names:
            return facts

        ordered = sorted(class_names)
        facts.add("class_names", ordered, source=FactSource.DIRECTORY_NAME, confidence=0.8,
                  origin=self.name)
        facts.add("num_classes", len(ordered), source=FactSource.DIRECTORY_NAME, confidence=0.8,
                  origin=self.name)
        if splits:
            facts.add("splits", splits, source=FactSource.DIRECTORY_NAME, origin=self.name)
        facts.add("task", "image-classification", source=FactSource.DIRECTORY_NAME,
                  confidence=0.7, origin=self.name)
        facts.add("modalities", [Modality.RGB.value], source=FactSource.DIRECTORY_NAME,
                  origin=self.name)
        return facts
