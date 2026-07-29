r"""Training-run and experiment detection.

Every tracker here writes a *container* holding many runs — ``runs/detect/`` for
Ultralytics, ``wandb/`` for Weights & Biases, ``mlruns/<experiment>/`` for MLflow — and the
thing worth cataloguing is one run inside it, not the container. Rooting at the container
would report "1 experiment" for a folder holding forty, and would suppress each run's
checkpoints along with it.

So every detector in this module claims the *child* and leaves the parent alone. The
container is caught on the way past: it matches nothing itself, the walk descends into it,
and each run beneath it is claimed in its own right.
"""

from __future__ import annotations

import os

from ai_asset_manager.backend.detectors.base import BaseDetector, DetectionResult
from ai_asset_manager.backend.models.enums import AssetKind, Framework
from ai_asset_manager.backend.scanner.context import DirectoryContext

#: Above the specific-dataset band: a run directory full of ``.json`` predictions must be
#: recognised as a run before a generic rule decides it is a corpus.
PRIORITY_EXPERIMENT = 65


def _run(
    detector: str,
    ctx: DirectoryContext,
    *,
    subkind: str,
    confidence: float,
    evidence: dict[str, object],
    framework: Framework = Framework.UNKNOWN,
    root_path: str | None = None,
    name: str | None = None,
) -> DetectionResult:
    """Build an experiment result, rooted at a run rather than at its container."""
    path = root_path or ctx.path
    return DetectionResult(
        kind=AssetKind.EXPERIMENT,
        name=name or os.path.basename(path) or ctx.name,
        root_path=path,
        detector=detector,
        subkind=subkind,
        framework=framework,
        confidence=confidence,
        evidence=evidence,
    )


class TensorBoardDetector(BaseDetector):
    """Detects a TensorBoard log directory by its event files."""

    name = "tensorboard"
    priority = PRIORITY_EXPERIMENT

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one experiment when event files sit directly in this directory.

        Directly, not anywhere below: ``events.out.tfevents.*`` files are written into the
        run directory itself, so a subtree search would only ever find an ancestor of the
        answer.
        """
        events = [
            entry for entry in ctx.files if entry.name.startswith("events.out.tfevents")
        ]
        if not events:
            return []

        return [
            _run(
                self.name,
                ctx,
                subkind="tensorboard",
                confidence=0.95,
                framework=Framework.TENSORFLOW,
                evidence={
                    "event_files": len(events),
                    "bytes": sum(entry.size for entry in events),
                    "checkpoints": ctx.has_any_dir("checkpoints", "ckpt"),
                },
            )
        ]


class WandbRunDetector(BaseDetector):
    """Detects a Weights & Biases run directory."""

    name = "wandb"
    priority = PRIORITY_EXPERIMENT

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one experiment for a ``run-<timestamp>-<id>`` directory."""
        metadata = ctx.child("files")
        has_metadata = (metadata is not None and metadata.has("wandb-metadata.json")) or (
            ctx.has_any("wandb-metadata.json")
        )
        looks_like_run = ctx.name.startswith(("run-", "offline-run-"))

        if not has_metadata and not (looks_like_run and ctx.has_any_dir("files", "logs")):
            return []

        return [
            _run(
                self.name,
                ctx,
                subkind="wandb",
                confidence=0.95 if has_metadata else 0.7,
                evidence={"metadata": has_metadata, "run_dir": looks_like_run},
            )
        ]


class MlflowRunDetector(BaseDetector):
    """Detects an MLflow run by its ``meta.yaml`` and metric store."""

    name = "mlflow"
    priority = PRIORITY_EXPERIMENT

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one experiment for an MLflow run directory."""
        if not ctx.has("meta.yaml"):
            return []
        # An MLflow *experiment* also carries meta.yaml; only a run has the metric,
        # parameter and tag stores beside it.
        stores = ctx.lower_child_dir_names & {"metrics", "params", "tags", "artifacts"}
        if len(stores) < 2:
            return []

        return [
            _run(
                self.name,
                ctx,
                subkind="mlflow",
                confidence=0.95,
                evidence={"stores": sorted(stores)},
            )
        ]


class UltralyticsRunDetector(BaseDetector):
    """Detects an Ultralytics training run by its resolved arguments and results."""

    name = "ultralytics_run"
    priority = PRIORITY_EXPERIMENT

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one experiment for a ``runs/<task>/<name>`` directory.

        ``args.yaml`` is written by Ultralytics at the start of every run and by nothing
        else, which makes it a reliable marker even when the run was interrupted before it
        produced results or weights — the case on the development machine, where four runs
        exist with empty ``weights`` directories.
        """
        if not ctx.has("args.yaml"):
            return []

        weights = ctx.child("weights")
        weight_files = (
            [entry.name for entry in weights.files if entry.extension in {".pt", ".onnx"}]
            if weights is not None
            else []
        )

        return [
            _run(
                self.name,
                ctx,
                subkind="ultralytics",
                confidence=0.9,
                framework=Framework.ULTRALYTICS,
                evidence={
                    "results": ctx.has_any("results.csv"),
                    "weights": weight_files,
                    "plots": ctx.count_extension(".png", ".jpg"),
                    "task": os.path.basename(os.path.dirname(ctx.path)),
                },
            )
        ]


class LightningLogsDetector(BaseDetector):
    """Detects a PyTorch Lightning ``version_N`` run directory."""

    name = "lightning"
    priority = PRIORITY_EXPERIMENT

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one experiment for a Lightning version directory."""
        if not ctx.name.lower().startswith("version_"):
            return []
        if not ctx.has_any("hparams.yaml", "metrics.csv"):
            return []

        return [
            _run(
                self.name,
                ctx,
                subkind="lightning",
                confidence=0.9,
                framework=Framework.PYTORCH,
                evidence={
                    "hparams": ctx.has_any("hparams.yaml"),
                    "metrics": ctx.has_any("metrics.csv"),
                    "checkpoints": ctx.has_any_dir("checkpoints"),
                },
            )
        ]
