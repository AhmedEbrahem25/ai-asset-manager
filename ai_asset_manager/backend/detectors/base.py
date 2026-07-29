"""Detector protocol and the result it produces.

A detector answers one question: *is there an asset here, and what kind?* It classifies
from structure — which files sit beside which — and leaves metadata extraction to the
parsers. That split is why a detector is typically twenty lines.

Two properties of the result carry the weight of the design:

``root_path`` may differ from the directory examined. A HuggingFace cache repository is
detected at ``models--org--name/`` but its content lives in ``snapshots/<sha>/``, and the
asset's files must be gathered from the latter.

Detection returns a *list*. Most directories yield one asset, but a folder holding twenty
unrelated ``.gguf`` files holds twenty models, and an Ollama store holds one per manifest.
A single-result protocol would force those cases to be special-cased downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ai_asset_manager.backend.models.enums import AssetFormat, AssetKind, Framework
from ai_asset_manager.backend.scanner.context import DirectoryContext

#: Priority bands. Higher wins, and the first band whose detector matches claims the
#: directory. Specific layouts must outrank generic ones: a Diffusers pipeline contains
#: safetensors files, so if the bare-weights detector ran first every pipeline would be
#: catalogued as a loose pile of tensors.
PRIORITY_STORE = 100  # Ollama and similar multi-model stores
PRIORITY_CACHE = 90  # HuggingFace cache repositories
PRIORITY_PIPELINE = 80  # Diffusers, Sentence-Transformers: multi-component layouts
PRIORITY_ADAPTER = 75  # PEFT/LoRA, which sit beside ordinary weights
PRIORITY_REPO = 70  # a standard HuggingFace model directory
PRIORITY_DATASET_SPECIFIC = 60  # COCO, KITTI, Cityscapes and other named layouts
PRIORITY_DATASET_GENERIC = 40  # classification trees, loose media collections
PRIORITY_FRAMEWORK = 35  # TensorFlow SavedModel, OpenVINO, TensorRT
PRIORITY_LOOSE_WEIGHTS = 20  # individual weight files with no surrounding structure


@dataclass(slots=True)
class DetectionResult:
    """One asset identified within a directory."""

    kind: AssetKind
    name: str
    #: Directory (or file) that *is* the asset. May be a subdirectory of the directory
    #: that was examined — see the module docstring.
    root_path: str
    detector: str
    subkind: str | None = None
    is_single_file: bool = False
    format: AssetFormat = AssetFormat.UNKNOWN
    framework: Framework = Framework.UNKNOWN
    #: 0..1. Used to pick a winner when two detectors in the same band both match.
    confidence: float = 1.0
    #: What the detector saw, kept for display and for debugging misclassification.
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Directory the asset's files should be gathered from, when it differs from
    #: ``root_path``. Set by the HuggingFace cache detector to point at the snapshot.
    content_root: str | None = None
    #: Restrict the asset's files to exactly these paths rather than the whole subtree.
    #: Used where several assets share one directory, as in an Ollama blob store.
    explicit_files: list[str] | None = None
    #: Whether directories beneath ``root_path`` are part of this asset and must not be
    #: examined separately. False for stores, which contain many independent assets.
    claims_subtree: bool = True
    #: Whether files belonging to *other* detected assets should be left out of this one.
    #:
    #: Set for containers -- a project, a labelling workspace. A project's subtree holds
    #: the models it produced, and counting those as part of the project both doubles them
    #: in the storage total and lets their contents decide what the project is classified
    #: as: on the development machine a project containing twenty-two Weights & Biases runs
    #: was itself filed as an experiment log. A container's own weight is what is left when
    #: the assets inside it are taken out.
    excludes_nested: bool = False

    @property
    def file_root(self) -> str:
        """Return the directory the asset's files should be collected from."""
        return self.content_root or self.root_path


@runtime_checkable
class AssetDetector(Protocol):
    """Classifies a directory as zero or more assets."""

    name: str
    priority: int

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Return the assets found in this directory, or an empty list."""
        ...


class BaseDetector:
    """Convenience base for detectors.

    Subclasses implement :meth:`detect` and inherit sane defaults for the rest.
    """

    name: str = "base"
    priority: int = PRIORITY_LOOSE_WEIGHTS

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Return the assets found in this directory. Overridden by subclasses."""
        raise NotImplementedError

    def _result(
        self,
        ctx: DirectoryContext,
        *,
        kind: AssetKind,
        name: str | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """Build a result rooted at the examined directory."""
        return DetectionResult(
            kind=kind,
            name=name or ctx.name,
            root_path=ctx.path,
            detector=self.name,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<{type(self).__name__} {self.name!r} priority={self.priority}>"


def pick_best(results: list[DetectionResult]) -> DetectionResult | None:
    """Return the most confident result from a set of competing detections."""
    return max(results, key=lambda item: item.confidence) if results else None
