r"""AI project detection.

A project is the codebase that *makes* assets: it trains models, holds their configs, and
keeps its datasets and runs nearby. Cataloguing it answers a question the asset rows cannot
— "what was this checkpoint for?" — and gives the relationship graph something to hang the
models beneath.

Two properties make this detector different from every other one here.

It **never claims its subtree**. A project contains models, datasets and runs; claiming
would suppress the very things worth finding. On the development machine
``F:\project\NLP-Project\thorn-nlp`` sits above two HuggingFace repos, eleven fine-tuned
checkpoints and a legal corpus, and a subtree-claiming project detector would replace all
fourteen rows with one.

It **requires corroboration**. A single ``requirements.txt`` is a Python project, not an AI
one, and almost every directory on a developer's disk has a ``README.md``. Two independent
signals are demanded before anything is claimed, which is what keeps this off ordinary
source trees.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from ai_asset_manager.backend.detectors.base import BaseDetector, DetectionResult
from ai_asset_manager.backend.models.enums import AssetKind, Framework
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Priority sits above the generic dataset band so that a project is recognised as a
#: project rather than as whatever pile of files happens to be at its root, but below the
#: specific layouts: a directory that is genuinely a COCO dataset is a dataset even when a
#: training script sits beside it.
PRIORITY_PROJECT = 55

#: Scripts whose names state what the codebase does. Matched exactly: ``train.py`` is a
#: training entrypoint, ``pretrain_utils.py`` is a module.
ENTRYPOINT_SCRIPTS: frozenset[str] = frozenset(
    {
        "train.py", "training.py", "finetune.py", "fine_tune.py", "pretrain.py",
        "infer.py", "inference.py", "predict.py", "evaluate.py", "eval.py",
        "export.py", "benchmark.py", "distill.py", "quantize.py", "serve.py",
        "train.sh", "train.ipynb", "main.py",
    }
)

#: Directory names that a training codebase keeps and a web application does not.
PROJECT_DIRS: frozenset[str] = frozenset(
    {
        "configs", "config", "weights", "checkpoints", "notebooks", "experiments",
        "runs", "datasets", "dataset", "recipes", "scripts", "training", "evals",
    }
)

#: Dependency declarations worth reading. Small files, and the only reliable statement of
#: which ecosystem a codebase belongs to.
REQUIREMENT_FILES: tuple[str, ...] = (
    "requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml",
    "Pipfile", "setup.py", "poetry.lock", "uv.lock", "conda.yaml",
)


class FrameworkMarker(NamedTuple):
    """A framework, and the dependency names that identify it."""

    framework: Framework
    label: str
    markers: tuple[str, ...]


#: Ordered most specific first: ``ultralytics`` implies ``torch``, so whichever matches
#: first should be the more informative answer.
FRAMEWORK_MARKERS: tuple[FrameworkMarker, ...] = (
    FrameworkMarker(Framework.ULTRALYTICS, "Ultralytics", ("ultralytics", "yolov5", "yolov8")),
    FrameworkMarker(Framework.PYTORCH, "MMDetection", ("mmdet", "mmcv", "mmengine", "mmseg")),
    FrameworkMarker(Framework.PYTORCH, "Detectron2", ("detectron2",)),
    FrameworkMarker(Framework.DIFFUSERS, "Diffusers", ("diffusers", "comfyui", "invokeai")),
    FrameworkMarker(Framework.PEFT, "PEFT", ("peft", "unsloth", "axolotl", "trl", "deepspeed")),
    FrameworkMarker(
        Framework.SENTENCE_TRANSFORMERS, "Sentence Transformers", ("sentence-transformers",)
    ),
    FrameworkMarker(
        Framework.TRANSFORMERS, "Transformers",
        ("transformers", "accelerate", "datasets", "langchain", "llama-index",
         "llama_index", "haystack", "vllm", "sglang"),
    ),
    FrameworkMarker(
        Framework.PYTORCH, "PyTorch", ("torch", "pytorch-lightning", "lightning", "timm"),
    ),
    FrameworkMarker(Framework.TENSORFLOW, "TensorFlow", ("tensorflow", "tf-nightly")),
    FrameworkMarker(Framework.KERAS, "Keras", ("keras",)),
    FrameworkMarker(Framework.PADDLE, "PaddlePaddle", ("paddlepaddle", "paddleocr")),
    FrameworkMarker(Framework.ONNXRUNTIME, "ONNX Runtime", ("onnxruntime",)),
)

#: Cap on how much of a dependency file is read. A ``pyproject.toml`` is kilobytes; a
#: ``poetry.lock`` can be megabytes and its first pages already name the ecosystem.
MAX_REQUIREMENTS_BYTES = 256 * 1024


class AIProjectDetector(BaseDetector):
    """Detects a codebase that trains, serves or evaluates models."""

    name = "ai_project"
    priority = PRIORITY_PROJECT

    #: Independent signals required before a directory is called an AI project. One is
    #: never enough: ``requirements.txt`` alone describes most of a developer's disk.
    MIN_SIGNALS = 2

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one project when enough independent AI signals agree."""
        if not self._is_project_root(ctx):
            return []

        entrypoints = sorted(self._entrypoints(ctx))
        project_dirs = sorted(ctx.lower_child_dir_names & PROJECT_DIRS)
        notebooks = ctx.count_extension(".ipynb")
        framework, framework_label, dependency = self._framework(ctx)

        signals = sum(
            (
                bool(entrypoints),
                len(project_dirs) >= 2,
                notebooks > 0,
                framework is not Framework.UNKNOWN,
            )
        )
        if signals < self.MIN_SIGNALS:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.PROJECT,
                subkind=framework_label.lower().replace(" ", "_") if framework_label else None,
                framework=framework,
                confidence=min(0.95, 0.5 + 0.15 * signals),
                # A project holds assets; it is not made of them. Claiming the subtree
                # would hide every model and dataset it was built to produce, and counting
                # their bytes as the project's would report the same gigabytes twice.
                claims_subtree=False,
                excludes_nested=True,
                evidence={
                    "entrypoints": entrypoints[:10],
                    "project_dirs": project_dirs,
                    "notebooks": notebooks,
                    "framework": framework_label,
                    "declared_in": dependency,
                    "signals": signals,
                },
            )
        ]

    @staticmethod
    def _entrypoints(ctx: DirectoryContext) -> set[str]:
        """Return the entrypoint scripts present directly in this directory."""
        return {name for name in ctx.file_names if name.lower() in ENTRYPOINT_SCRIPTS}

    def _is_project_root(self, ctx: DirectoryContext) -> bool:
        """Report whether this directory could be the top of a codebase.

        A project root is where the dependency declaration or the version control
        directory lives. Requiring one of those stops every ``src/`` and ``services/``
        subdirectory of a large repository from being catalogued as its own project.
        """
        if ctx.has_any(*REQUIREMENT_FILES):
            return True
        return ctx.has_any_dir(".git", ".hg", ".svn")

    def _framework(self, ctx: DirectoryContext) -> tuple[Framework, str, str | None]:
        """Identify the ecosystem from the project's declared dependencies.

        Returns:
            The framework, a display label, and the file the evidence came from.
        """
        for filename in REQUIREMENT_FILES:
            text = self._read(ctx, filename)
            if text is None:
                continue
            lowered = text.lower()
            for entry in FRAMEWORK_MARKERS:
                if any(marker in lowered for marker in entry.markers):
                    return entry.framework, entry.label, filename
        return Framework.UNKNOWN, "", None

    def _read(self, ctx: DirectoryContext, filename: str) -> str | None:
        """Read a dependency file, bounded."""
        entry = ctx.entry(filename)
        if entry is None:
            return None
        try:
            with open(entry.path, encoding="utf-8", errors="replace") as handle:
                return handle.read(MAX_REQUIREMENTS_BYTES)
        except OSError as exc:
            logger.debug("Cannot read %s: %s", entry.path, exc)
            return None


def project_root_of(path: str, projects: dict[str, str]) -> str | None:
    """Return the innermost catalogued project containing ``path``, if any.

    Args:
        path: Any asset root.
        projects: Mapping of normalised project root to project identifier.

    Returns:
        The identifier of the deepest project that contains the path. Deepest rather than
        first, because a monorepo may hold a project inside a project and the nearer one
        is the meaningful owner.
    """
    best: tuple[int, str] | None = None
    lowered = path.lower()
    for root, identifier in projects.items():
        root_lower = root.lower()
        if lowered == root_lower:
            continue
        prefix = root_lower.rstrip("\\/") + os.sep
        if lowered.startswith(prefix) and (best is None or len(root_lower) > best[0]):
            best = (len(root_lower), identifier)
    return best[1] if best else None
