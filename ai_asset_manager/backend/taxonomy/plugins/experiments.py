"""Training runs, experiment logs and evaluation artefacts.

The output of training rather than its input, and the part of a machine that grows without
anyone noticing: a directory of runs accumulates checkpoints, event files and metric
dumps for months. Recognising them means ``aam inventory experiments`` can say how much
disk the last six months of training is holding.

Ranked high, because a run directory usually contains weights and would otherwise be
classified by what it contains rather than by what it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Finding,
    Task,
)

#: Tracking tools, and the fingerprint each leaves in a directory listing.
_TRACKERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (label, filename stems, directory names)
    ("TensorBoard", ("events.out.tfevents",), ()),
    ("Weights & Biases", ("wandb-summary.json", "wandb-metadata.json"), ("wandb",)),
    ("MLflow", ("mlmodel", "meta.yaml"), ("mlruns", "artifacts")),
    ("Ultralytics", ("results.csv", "args.yaml"), ("weights",)),
    ("Lightning", ("hparams.yaml",), ("lightning_logs", "checkpoints")),
    ("Hydra", ("hydra.yaml", "overrides.yaml"), (".hydra",)),
    ("DVC", ("dvc.lock", "dvc.yaml"), (".dvc",)),
    ("Aim", ("run_metadata.db",), (".aim",)),
)

#: Files that are results rather than a run: a report someone produced and kept.
_EVALUATION_FILES = ("results.json", "eval_results.json", "metrics.json",
                     "predictions.json", "confusion_matrix.png", "all_results.json",
                     "test_results.json", "evaluation.json")

TASKS = (
    Task(id="training", label="Training", domain="mlops", order=10),
    Task(id="evaluation", label="Evaluation", domain="mlops", order=20),
    Task(id="experiment_tracking", label="Experiment Tracking", domain="mlops", order=30),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the experiment shelves, their classifier and their statistics."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="training_run", label="Training Run", section="experiments", order=400,
                 domain="mlops", aliases=("runs", "training-runs"),
                 description="A directory produced by a training job.")
    )
    registry.add_category(
        Category(id="experiment_log", label="Experiment Log", section="experiments", order=410,
                 domain="mlops", aliases=("logs", "experiment-logs"))
    )
    registry.add_category(
        Category(id="evaluation", label="Evaluation", section="experiments", order=420,
                 domain="mlops", aliases=("evaluations", "results", "benchmarks"))
    )

    registry.add_classifier(_experiment, name="experiments", priority=850)
    registry.add_statistic(_experiment_statistics, name="experiment")
    registry.add_health_rule(_run_without_checkpoint, name="experiment.no_checkpoint")


def _tracker_of(profile: AssetProfile) -> str | None:
    """Return the tracking tool whose fingerprint is in this asset's file list."""
    files = profile.files
    for label, stems, directories in _TRACKERS:
        if stems and files.has_stem(*stems):
            return label
        if directories and files.has_dir(*directories):
            return label
    return None


def _experiment(profile: AssetProfile) -> Classification | None:
    """Claim training runs, logs and evaluation output."""
    if not profile.files.loaded:
        return None

    tracker = _tracker_of(profile)
    files = profile.files
    has_evaluation = files.has_name(*_EVALUATION_FILES)

    if tracker is None and not has_evaluation:
        return None

    # A run keeps its weights; a log directory is only the record of one. The distinction
    # matters because deleting the second is safe and deleting the first is not.
    has_checkpoints = bool(
        files.count(".pt", ".pth", ".ckpt", ".safetensors", ".bin")
        or files.has_dir("weights", "checkpoints")
    )

    if has_checkpoints:
        category, task = "training_run", "training"
    elif tracker is not None:
        category, task = "experiment_log", "experiment_tracking"
    else:
        category, task = "evaluation", "evaluation"

    return Classification(
        category=category, task=task, domain="mlops", family=tracker,
        confidence=CONFIDENCE_CERTAIN if tracker else CONFIDENCE_STRONG,
        evidence=f"{tracker} artefacts" if tracker else "evaluation results present",
    )


def _experiment_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return which tool produced a run and how much of it is checkpoints."""
    if not profile.files.loaded:
        return {}

    tracker = _tracker_of(profile)
    if tracker is None and not profile.files.has_name(*_EVALUATION_FILES):
        return {}

    stats: dict[str, Any] = {}
    if tracker:
        stats["tracker"] = tracker

    checkpoints = profile.files.count(".pt", ".pth", ".ckpt", ".safetensors")
    if checkpoints:
        stats["checkpoints"] = checkpoints
        # Nearly all of a run's size is usually its checkpoints, and that is the number
        # that decides whether pruning old runs is worth the effort.
        stats["checkpoint_bytes"] = profile.files.bytes_in(
            ".pt", ".pth", ".ckpt", ".safetensors"
        )

    events = profile.files.matching(r"events\.out\.tfevents")
    if events:
        stats["event_files"] = events

    return stats


def _run_without_checkpoint(profile: AssetProfile) -> Sequence[Finding]:
    """Report a training run whose checkpoints are gone.

    Common and quietly expensive: someone deletes the weights to reclaim space and leaves
    the logs, so the run still looks complete in a file browser but nothing can be resumed
    or evaluated from it.
    """
    if not profile.files.loaded:
        return ()
    if _tracker_of(profile) is None:
        return ()
    if profile.files.count(".pt", ".pth", ".ckpt", ".safetensors", ".bin"):
        return ()

    return (
        Finding(
            code="experiment.no_checkpoint",
            severity=Severity.INFO,
            message="Logs kept but no checkpoint remains",
            fix_hint="Nothing to resume or evaluate from. Safe to archive the logs.",
        ),
    )
