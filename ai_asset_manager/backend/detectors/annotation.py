r"""Annotation-project detection.

A labelling project is neither a model nor a dataset: it is work in progress, and knowing
it exists is the difference between re-labelling three thousand images and finding the
export you already made. Each tool below leaves a file it writes and nothing else does,
which is what makes these detectors short.

Roboflow is the exception worth stating: its exports *are* datasets — YOLO or COCO,
complete and usable — and are already recognised as such. What marks them additionally as
Roboflow work is the receipt the exporter drops beside them, so that detector runs above
the dataset band and records provenance the plain layout cannot.
"""

from __future__ import annotations

from ai_asset_manager.backend.detectors.base import BaseDetector, DetectionResult
from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.scanner.context import DirectoryContext

#: Above the specific-dataset band, so a Roboflow export is reported as the annotation
#: project it came from rather than only as the YOLO tree it looks like.
PRIORITY_ANNOTATION = 62


class CvatProjectDetector(BaseDetector):
    """Detects a CVAT task or project export."""

    name = "cvat"
    priority = PRIORITY_ANNOTATION

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one annotation project for a CVAT export."""
        has_task = ctx.has_any("task.json", "project.json")
        has_annotations = ctx.has_any("annotations.xml") or ctx.has_any_dir("annotations")

        if not (has_task and has_annotations):
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.ANNOTATION_PROJECT,
                subkind="cvat",
                confidence=0.9,
                evidence={
                    "task_manifest": has_task,
                    "annotations": has_annotations,
                    "images": ctx.image_count,
                },
            )
        ]


class LabelStudioDetector(BaseDetector):
    """Detects a Label Studio project directory."""

    name = "label_studio"
    priority = PRIORITY_ANNOTATION

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one annotation project for a Label Studio workspace."""
        database = ctx.has_any("label_studio.sqlite3")
        config = ctx.has_any("label_config.xml", "project_config.xml")

        if not database and not config:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.ANNOTATION_PROJECT,
                subkind="label_studio",
                confidence=0.95 if database else 0.7,
                evidence={
                    "database": database,
                    "label_config": config,
                    "exports": ctx.has_any_dir("export", "exports"),
                },
            )
        ]


class RoboflowExportDetector(BaseDetector):
    """Detects a Roboflow export by the receipt its exporter writes."""

    name = "roboflow"
    priority = PRIORITY_ANNOTATION

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one annotation project for a Roboflow-exported dataset."""
        receipt = [
            name
            for name in ("README.roboflow.txt", "README.dataset.txt")
            if ctx.has_any(name)
        ]
        if not receipt:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.ANNOTATION_PROJECT,
                subkind="roboflow",
                confidence=0.9,
                evidence={
                    "receipt": receipt,
                    "yolo_manifest": ctx.has_any("data.yaml"),
                    "coco_annotations": bool(ctx.glob("*_annotations.coco.json")),
                    "images": ctx.image_count,
                },
            )
        ]


class SupervisleyProjectDetector(BaseDetector):
    """Detects a Supervisely project by its ``meta.json`` and per-dataset ``ann`` folders."""

    name = "supervisely"
    priority = PRIORITY_ANNOTATION

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one annotation project for a Supervisely export."""
        if not ctx.has("meta.json"):
            return []

        # Supervisely nests `<dataset>/ann` and `<dataset>/img` beneath the project root.
        datasets = [
            child.name
            for child in ctx.children()
            if child.has_any_dir("ann") and child.has_any_dir("img")
        ]
        if not datasets:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.ANNOTATION_PROJECT,
                subkind="supervisely",
                confidence=0.9,
                evidence={"datasets": datasets[:20], "images": ctx.image_count},
            )
        ]
