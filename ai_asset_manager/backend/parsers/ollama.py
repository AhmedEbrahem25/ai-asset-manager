"""Ollama model store reader.

Ollama uses an OCI-style layout rather than a directory per model::

    manifests/registry.ollama.ai/library/deepseek-r1/8b   <- a JSON manifest, one per tag
    blobs/sha256-<digest>                                 <- content, addressed by hash

The manifest names its layers by media type, so the weight blob, the licence, the prompt
template and the parameter block can each be located without guessing. This matters
because the blob store is flat and opaque: a directory listing shows nothing but hashes.

The path from ``manifests/`` down to the tag file yields the model reference a user
actually recognises — ``deepseek-r1:8b``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ai_asset_manager.backend.models.enums import FactSource, Framework
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import DirectoryTree
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Layer media types Ollama writes.
MEDIA_TYPE_MODEL = "application/vnd.ollama.image.model"
MEDIA_TYPE_LICENSE = "application/vnd.ollama.image.license"
MEDIA_TYPE_TEMPLATE = "application/vnd.ollama.image.template"
MEDIA_TYPE_PARAMS = "application/vnd.ollama.image.params"
MEDIA_TYPE_SYSTEM = "application/vnd.ollama.image.system"
MEDIA_TYPE_ADAPTER = "application/vnd.ollama.image.adapter"
MEDIA_TYPE_PROJECTOR = "application/vnd.ollama.image.projector"

#: Cap on auxiliary blobs read as text. Licences and templates are kilobytes.
MAX_TEXT_BLOB_BYTES = 256 * 1024


@dataclass(slots=True)
class OllamaLayer:
    """One layer referenced by an Ollama manifest."""

    media_type: str
    digest: str
    size: int

    @property
    def blob_name(self) -> str:
        """Return the on-disk blob filename for this layer's digest."""
        return self.digest.replace(":", "-")


@dataclass(slots=True)
class OllamaModel:
    """One model resolved from an Ollama manifest."""

    #: The reference a user types, e.g. ``deepseek-r1:8b``.
    reference: str
    name: str
    tag: str
    registry: str
    namespace: str
    manifest_path: str
    layers: list[OllamaLayer] = field(default_factory=list)
    #: Absolute path of the weight blob, when it exists on disk.
    model_blob_path: str | None = None
    model_blob_size: int = 0
    license_text: str | None = None
    parameters: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Report whether the manifest parsed and named a weight layer."""
        return self.error is None and self.model_blob_size > 0

    @property
    def total_size(self) -> int:
        """Return the summed size of every layer."""
        return sum(layer.size for layer in self.layers)

    def layer(self, media_type: str) -> OllamaLayer | None:
        """Return the first layer of a given media type."""
        for item in self.layers:
            if item.media_type == media_type:
                return item
        return None


def find_ollama_root(path: str) -> str | None:
    """Return the Ollama models root that contains ``path``, if any.

    A store is identified by holding both ``manifests`` and ``blobs``; either alone is
    ambiguous enough to belong to something else.
    """
    current = os.path.abspath(path)
    for _ in range(12):
        if os.path.isdir(os.path.join(current, "manifests")) and os.path.isdir(
            os.path.join(current, "blobs")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def is_ollama_store(ctx: DirectoryContext) -> bool:
    """Report whether a directory is the root of an Ollama model store."""
    return ctx.has_dir("manifests") and ctx.has_dir("blobs")


def parse_manifest(
    manifest_path: str,
    blobs_dir: str,
    *,
    manifests_root: str,
) -> OllamaModel:
    """Parse a single Ollama manifest file.

    Args:
        manifest_path: Path to the tag file, e.g. ``.../library/deepseek-r1/8b``.
        blobs_dir: Path to the store's ``blobs`` directory.
        manifests_root: Path to the store's ``manifests`` directory, used to recover the
            registry/namespace/name components from the manifest's location.

    Returns:
        An :class:`OllamaModel`; ``error`` is set when the manifest is unusable.
    """
    relative = os.path.relpath(manifest_path, manifests_root).replace("\\", "/")
    parts = relative.split("/")

    tag = parts[-1] if parts else "latest"
    name = parts[-2] if len(parts) >= 2 else "unknown"
    namespace = parts[-3] if len(parts) >= 3 else ""
    registry = parts[0] if len(parts) >= 4 else ""

    # The `library` namespace is Ollama's default and is not part of how a user refers
    # to the model: `library/llama3:8b` is just `llama3:8b`.
    display = name if namespace in ("library", "") else f"{namespace}/{name}"

    model = OllamaModel(
        reference=f"{display}:{tag}",
        name=display,
        tag=tag,
        registry=registry,
        namespace=namespace,
        manifest_path=manifest_path,
    )

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        model.error = f"unreadable manifest: {exc}"
        return model

    if not isinstance(manifest, dict):
        model.error = "manifest is not a JSON object"
        return model

    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list):
        model.error = "manifest has no layer list"
        return model

    for raw in raw_layers:
        if not isinstance(raw, dict):
            continue
        media_type = raw.get("mediaType")
        digest = raw.get("digest")
        size = raw.get("size")
        if isinstance(media_type, str) and isinstance(digest, str) and isinstance(size, int):
            model.layers.append(OllamaLayer(media_type=media_type, digest=digest, size=size))

    weight_layer = model.layer(MEDIA_TYPE_MODEL)
    if weight_layer is None:
        model.error = "manifest names no model layer"
        return model

    blob_path = os.path.join(blobs_dir, weight_layer.blob_name)
    model.model_blob_size = weight_layer.size
    if os.path.exists(blob_path):
        model.model_blob_path = blob_path
    else:
        # The manifest survives `ollama rm` failures and partial pulls; a missing blob is
        # a real condition the health checker reports rather than a parse failure.
        model.error = "model blob is missing from the store"

    model.license_text = _read_text_layer(model, MEDIA_TYPE_LICENSE, blobs_dir)
    model.parameters = _read_json_layer(model, MEDIA_TYPE_PARAMS, blobs_dir)
    return model


def _read_text_layer(model: OllamaModel, media_type: str, blobs_dir: str) -> str | None:
    """Read a small text layer from the blob store."""
    layer = model.layer(media_type)
    if layer is None or layer.size > MAX_TEXT_BLOB_BYTES:
        return None
    try:
        with open(os.path.join(blobs_dir, layer.blob_name), encoding="utf-8",
                  errors="replace") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _read_json_layer(model: OllamaModel, media_type: str, blobs_dir: str) -> dict[str, object]:
    """Read a small JSON layer from the blob store."""
    text = _read_text_layer(model, media_type, blobs_dir)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def discover_ollama_models(ctx: DirectoryContext, tree: DirectoryTree) -> list[OllamaModel]:
    """Enumerate every model in an Ollama store.

    Args:
        ctx: Context positioned on the store root (the directory holding ``manifests``).
        tree: The walked tree, used to enumerate manifest files without new I/O.

    Returns:
        One :class:`OllamaModel` per manifest found, valid or not.
    """
    manifests_root = os.path.join(ctx.path, "manifests")
    blobs_dir = os.path.join(ctx.path, "blobs")

    models: list[OllamaModel] = []
    for entry in tree.iter_subtree_files(manifests_root):
        # Every regular file beneath `manifests/` is a tag manifest; the tag is the
        # filename and carries no extension.
        if entry.size == 0 or entry.size > 1024 * 1024:
            continue
        model = parse_manifest(entry.path, blobs_dir, manifests_root=manifests_root)
        if model.layers or model.error:
            models.append(model)

    logger.debug("Found %d Ollama model(s) under %s", len(models), ctx.path)
    return models


class OllamaParser(BaseParser):
    """Extracts facts for one Ollama model.

    Unlike the other parsers this one is driven by a pre-resolved :class:`OllamaModel`
    rather than by a directory, because a store holds many models in one flat blob
    directory and the detector has already split them apart.
    """

    name = "ollama"

    def __init__(self, model: OllamaModel | None = None) -> None:
        """Bind the parser to a resolved model."""
        self.model = model

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a model has been bound to this parser."""
        return self.model is not None

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Emit facts describing the bound Ollama model."""
        facts = self._new_facts()
        model = self.model
        if model is None:
            return facts

        facts.add("name", model.reference, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("display_name", model.reference, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("repo_id", model.name, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("revision", model.tag, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("framework", Framework.OLLAMA.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("format", "gguf", source=FactSource.EXPLICIT_CONFIG, confidence=0.8,
                  origin=self.name)

        if model.namespace and model.namespace != "library":
            facts.add("author", model.namespace, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        if model.license_text:
            facts.add("license", _summarise_license(model.license_text),
                      source=FactSource.SIDECAR_DOC, origin=self.name)

        context_length = model.parameters.get("num_ctx")
        if isinstance(context_length, int):
            facts.add("context_length", context_length, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        self._read_weight_blob(facts, model)

        if model.error:
            facts.warn(f"{model.reference}: {model.error}")

        return facts

    def _read_weight_blob(self, facts: FactSet, model: OllamaModel) -> None:
        """Parse the weight blob's GGUF header.

        The blob is named for its digest and carries no extension, so the extension-driven
        GGUF parser never sees it. The manifest's media type is what tells us the file is
        GGUF at all — without this the store's models would be catalogued with a size and
        nothing else.
        """
        if not model.model_blob_path:
            return

        # Imported here rather than at module scope: the GGUF parser imports the config
        # parser, and a top-level import would tangle the parser modules together.
        from ai_asset_manager.backend.parsers.gguf import read_gguf_header

        info = read_gguf_header(model.model_blob_path)
        if not info.is_valid:
            facts.warn(f"{model.reference}: weight blob is not readable GGUF ({info.error})")
            return

        facts.add("architecture", info.architecture, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("quantization", info.quantization or info.dominant_tensor_type,
                  source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add("precision", info.precision.value, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("tensor_count", info.tensor_count, source=FactSource.BINARY_HEADER,
                  origin=self.name)

        if info.param_count_is_exact and info.param_count > 0:
            facts.add("param_count", info.param_count, source=FactSource.BINARY_HEADER,
                      origin=self.name)
            facts.add("param_count_is_exact", True, source=FactSource.BINARY_HEADER,
                      origin=self.name)

        for suffix, key in (
            ("block_count", "num_layers"),
            ("embedding_length", "hidden_size"),
            ("context_length", "context_length"),
        ):
            facts.add(key, info.arch_value(suffix), source=FactSource.BINARY_HEADER,
                      confidence=0.9, origin=self.name)

        from ai_asset_manager.backend.parsers.hf_config import classify_architecture

        model_type = classify_architecture(info.architecture, info.model_name)
        if model_type is not None:
            facts.add("model_type", model_type.value, source=FactSource.BINARY_HEADER,
                      confidence=0.8, origin=self.name)


#: Recognisable licence names, longest first so "Apache License 2.0" is not matched as
#: a bare "Apache".
_LICENSE_MARKERS: tuple[tuple[str, str], ...] = (
    ("apache license", "Apache-2.0"),
    ("mit license", "MIT"),
    ("bsd 3-clause", "BSD-3-Clause"),
    ("bsd 2-clause", "BSD-2-Clause"),
    ("gnu general public", "GPL"),
    ("llama 3.1 community", "Llama-3.1"),
    ("llama 3.2 community", "Llama-3.2"),
    ("llama 3 community", "Llama-3"),
    ("llama 2 community", "Llama-2"),
    ("gemma terms of use", "Gemma"),
    ("tongyi qianwen", "Tongyi-Qianwen"),
    ("qwen license", "Qwen"),
    ("deepseek license", "DeepSeek"),
    ("creative commons", "CC"),
)


def _summarise_license(text: str) -> str:
    """Reduce full licence text to a short identifier.

    Ollama embeds the entire licence document; storing tens of kilobytes of legal prose
    in a column that the UI renders as a chip would be useless.
    """
    lowered = text[:4000].lower()
    for marker, label in _LICENSE_MARKERS:
        if marker in lowered:
            return label
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:64] if first_line else "custom"
