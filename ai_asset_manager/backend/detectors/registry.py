"""Detector registry and the claim-resolution walk.

Detection is a single parent-before-child traversal of the walked tree. At each directory
every detector is offered the chance to match; the highest-priority band that produces a
result wins, and ties inside a band are broken by confidence.

The subtlety is *claiming*. When a directory is recognised as an asset, the directories
beneath it are part of that asset and must not be examined again — otherwise a Diffusers
pipeline would be catalogued once as a pipeline and then five more times as its component
subdirectories. Some detectors opt out of that (``claims_subtree=False``) because they
describe containers of independent assets rather than assets themselves: an Ollama store
holds many models, and a folder of loose GGUF files holds one asset per file with the
folder itself holding nothing.
"""

from __future__ import annotations

from ai_asset_manager.backend.detectors.base import AssetDetector, DetectionResult
from ai_asset_manager.backend.detectors.datasets import (
    Bdd100kDetector,
    CityscapesDetector,
    CocoDetector,
    HfDatasetCacheDetector,
    ImageClassificationDetector,
    KittiDetector,
    MediaCollectionDetector,
    MotDetector,
    NuScenesDetector,
    PascalVocDetector,
    YoloDetector,
)
from ai_asset_manager.backend.detectors.models import (
    DiffusersDetector,
    HfCacheDetector,
    HfRepoDetector,
    LooseWeightsDetector,
    MlxDetector,
    OllamaStoreDetector,
    OpenVinoDetector,
    PeftAdapterDetector,
    SentenceTransformerDetector,
    TensorFlowSavedModelDetector,
)
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import DirectoryTree
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)


def default_detectors() -> list[AssetDetector]:
    """Return one instance of every built-in detector."""
    return [
        # Multi-model stores and caches.
        OllamaStoreDetector(),
        HfCacheDetector(),
        # Structured model layouts.
        DiffusersDetector(),
        SentenceTransformerDetector(),
        PeftAdapterDetector(),
        HfRepoDetector(),
        # Framework-specific model layouts.
        TensorFlowSavedModelDetector(),
        OpenVinoDetector(),
        MlxDetector(),
        # Named dataset layouts.
        CocoDetector(),
        YoloDetector(),
        PascalVocDetector(),
        CityscapesDetector(),
        KittiDetector(),
        NuScenesDetector(),
        MotDetector(),
        Bdd100kDetector(),
        HfDatasetCacheDetector(),
        # Generic fallbacks.
        ImageClassificationDetector(),
        MediaCollectionDetector(),
        LooseWeightsDetector(),
    ]


class DetectorRegistry:
    """Applies detectors across a walked tree and resolves competing claims."""

    def __init__(self, detectors: list[AssetDetector] | None = None) -> None:
        """Initialise the registry.

        Args:
            detectors: Detectors to use; defaults to :func:`default_detectors`. Sorted by
                descending priority so the traversal can stop at the first band that
                matches.
        """
        chosen = detectors if detectors is not None else default_detectors()
        self.detectors: list[AssetDetector] = sorted(
            chosen, key=lambda item: item.priority, reverse=True
        )

    def register(self, detector: AssetDetector) -> None:
        """Add a detector, keeping the list ordered by priority."""
        self.detectors.append(detector)
        self.detectors.sort(key=lambda item: item.priority, reverse=True)

    def detect_one(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Run detectors against a single directory.

        Detectors are grouped into priority bands; the first band that yields anything
        wins outright. Within a band, all results are kept when they describe distinct
        assets (a folder of loose weights) but reduced to the most confident when they
        describe the same directory.

        A detector that raises is logged and skipped, so one bad heuristic cannot abort
        the scan of a whole drive.
        """
        results_by_priority: dict[int, list[DetectionResult]] = {}

        for detector in self.detectors:
            try:
                found = detector.detect(ctx)
            except Exception as exc:
                logger.warning("Detector %s failed on %s: %s", detector.name, ctx.path, exc)
                continue

            if found:
                results_by_priority.setdefault(detector.priority, []).extend(found)

        if not results_by_priority:
            return []

        best_priority = max(results_by_priority)
        winners = results_by_priority[best_priority]

        # Several detectors claiming the same path is a genuine tie; several results at
        # different paths are distinct assets and all survive.
        by_path: dict[str, DetectionResult] = {}
        for result in winners:
            existing = by_path.get(result.root_path)
            if existing is None or result.confidence > existing.confidence:
                by_path[result.root_path] = result
        return list(by_path.values())

    def detect_tree(self, tree: DirectoryTree) -> list[DetectionResult]:
        """Detect every asset in a walked tree.

        Walks parent-before-child, so that a directory recognised as an asset can
        suppress its descendants before they are examined.

        Args:
            tree: A tree produced by the walker.

        Returns:
            Every asset found, in traversal order.
        """
        found: list[DetectionResult] = []
        claimed: set[str] = set()

        for node in tree.iter_descending():
            if node.path in claimed:
                continue

            ctx = DirectoryContext(tree, node)
            results = self.detect_one(ctx)
            if not results:
                continue

            found.extend(results)

            for result in results:
                if not result.claims_subtree:
                    continue
                # Everything below the examined directory is part of this asset. The
                # directory itself is not re-added: it has already been visited, and the
                # traversal never returns to it.
                for descendant in tree.subtree_paths(node.path):
                    if descendant != node.path:
                        claimed.add(descendant)

        logger.info("Detected %d asset(s) across %d directories", len(found), len(tree.nodes))
        return found
