"""Parser registry.

Parsers are independent and order-insensitive: every applicable parser runs, all of their
assertions are collected, and precedence is settled once by
:mod:`ai_asset_manager.backend.metadata.merge`. Adding a format means appending one class
here, not editing a dispatch chain.
"""

from __future__ import annotations

from ai_asset_manager.backend.parsers.base import FactSet, MetadataParser
from ai_asset_manager.backend.parsers.dataset_meta import (
    CocoMetadataParser,
    ImageFolderMetadataParser,
    VocMetadataParser,
    YoloDatasetMetadataParser,
)
from ai_asset_manager.backend.parsers.gguf import GgufParser
from ai_asset_manager.backend.parsers.hf_cache import HfCacheParser
from ai_asset_manager.backend.parsers.hf_config import (
    AdapterConfigParser,
    HfConfigParser,
    ModelIndexParser,
    SentenceTransformerParser,
    TokenizerConfigParser,
)
from ai_asset_manager.backend.parsers.model_card import ModelCardParser
from ai_asset_manager.backend.parsers.onnx import OnnxParser
from ai_asset_manager.backend.parsers.safetensors import SafetensorsParser
from ai_asset_manager.backend.parsers.torch_checkpoint import TorchCheckpointParser
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)


def default_parsers() -> list[MetadataParser]:
    """Return one instance of every built-in parser.

    Instances are stateless, so a fresh list per scan costs nothing and avoids sharing
    mutable state across threads.
    """
    return [
        HfCacheParser(),
        HfConfigParser(),
        AdapterConfigParser(),
        ModelIndexParser(),
        TokenizerConfigParser(),
        SentenceTransformerParser(),
        SafetensorsParser(),
        GgufParser(),
        TorchCheckpointParser(),
        OnnxParser(),
        ModelCardParser(),
        CocoMetadataParser(),
        YoloDatasetMetadataParser(),
        VocMetadataParser(),
        ImageFolderMetadataParser(),
    ]


class ParserRegistry:
    """Runs every applicable parser over a directory and merges their assertions."""

    def __init__(self, parsers: list[MetadataParser] | None = None) -> None:
        """Initialise the registry.

        Args:
            parsers: Parsers to run; defaults to :func:`default_parsers`.
        """
        self.parsers: list[MetadataParser] = parsers if parsers is not None else default_parsers()

    def register(self, parser: MetadataParser) -> None:
        """Add a parser."""
        self.parsers.append(parser)

    def collect(
        self, ctx: DirectoryContext, *, extra: list[MetadataParser] | None = None
    ) -> FactSet:
        """Run all applicable parsers and return their combined assertions.

        A parser that raises is logged and skipped. That is deliberate: the catalogue is
        expected to contain broken and half-downloaded assets, and one parser choking on
        a corrupt header must not stop the other ten from describing the asset.

        Args:
            ctx: Directory to parse.
            extra: Additional parsers for this call only, used for context-specific
                parsers such as the Ollama one, which is bound to a resolved model.

        Returns:
            The merged :class:`FactSet`.
        """
        combined = FactSet()

        for parser in [*self.parsers, *(extra or [])]:
            try:
                if not parser.supports(ctx):
                    continue
                combined.merge_from(parser.parse(ctx))
            except Exception as exc:
                logger.warning(
                    "Parser %s failed on %s: %s", getattr(parser, "name", parser), ctx.path, exc
                )
                combined.warn(f"parser {getattr(parser, 'name', '?')} failed: {exc}")

        return combined
