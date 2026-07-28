"""Model detectors.

Each classifies by structure — which files sit beside which — never by reading weights.
Ordered by the priority bands in :mod:`ai_asset_manager.backend.detectors.base`, so a
specific layout always claims a directory before a generic one gets to see it.
"""

from __future__ import annotations

import os
import re

from ai_asset_manager.backend.detectors.base import (
    PRIORITY_ADAPTER,
    PRIORITY_CACHE,
    PRIORITY_FRAMEWORK,
    PRIORITY_LOOSE_WEIGHTS,
    PRIORITY_PIPELINE,
    PRIORITY_REPO,
    PRIORITY_STORE,
    BaseDetector,
    DetectionResult,
)
from ai_asset_manager.backend.models.enums import AssetFormat, AssetKind, Framework, ModelType
from ai_asset_manager.backend.parsers.hf_cache import find_snapshot, is_cache_repo_dir
from ai_asset_manager.backend.parsers.ollama import discover_ollama_models, is_ollama_store
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import FileEntry

#: Weight extensions mapped to the format they imply.
WEIGHT_FORMATS: tuple[tuple[str, AssetFormat], ...] = (
    (".safetensors", AssetFormat.SAFETENSORS),
    (".gguf", AssetFormat.GGUF),
    (".ggml", AssetFormat.GGML),
    (".onnx", AssetFormat.ONNX),
    (".engine", AssetFormat.TENSORRT),
    (".plan", AssetFormat.TENSORRT),
    (".pt", AssetFormat.PYTORCH),
    (".pth", AssetFormat.PYTORCH),
    (".ckpt", AssetFormat.PYTORCH),
    (".bin", AssetFormat.PYTORCH),
    (".keras", AssetFormat.KERAS),
    (".h5", AssetFormat.KERAS),
    (".tflite", AssetFormat.TFLITE),
    (".mlmodel", AssetFormat.COREML),
    (".npz", AssetFormat.NUMPY),
    (".pdparams", AssetFormat.PADDLE),
)

#: Filename fragments that identify a well-known vision or audio model, used only when
#: nothing more structural is available.
VISION_NAME_MARKERS: tuple[tuple[tuple[str, ...], ModelType, Framework], ...] = (
    (("yolov", "yolo11", "yolo12", "yolo_", "yolov8", "yolo-world"),
     ModelType.OBJECT_DETECTION, Framework.ULTRALYTICS),
    (("sam_vit", "sam2", "segment-anything", "mobile_sam", "sam_h", "sam_b", "sam_l"),
     ModelType.SEGMENTATION, Framework.PYTORCH),
    (("groundingdino", "grounding_dino", "groundingdino_swint"),
     ModelType.OBJECT_DETECTION, Framework.PYTORCH),
    (("rtdetr", "rt-detr", "detr_"), ModelType.OBJECT_DETECTION, Framework.PYTORCH),
    (("clip-vit", "clip_vit", "open_clip", "openclip"), ModelType.MULTIMODAL, Framework.PYTORCH),
    (("blip", "blip2"), ModelType.VISION_LANGUAGE, Framework.PYTORCH),
    (("whisper",), ModelType.SPEECH_RECOGNITION, Framework.PYTORCH),
    (("vae-ft", "vae_ft", "sd-vae"), ModelType.IMAGE_GENERATION, Framework.DIFFUSERS),
    (("controlnet",), ModelType.IMAGE_GENERATION, Framework.DIFFUSERS),
    (("esrgan", "realesrgan", "gfpgan", "codeformer"), ModelType.VISION, Framework.PYTORCH),
)

#: Filenames that mark a directory as a LoRA/PEFT adapter even without a config.
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)

#: ``model-00001-of-00005.safetensors`` and the equivalent GGUF spelling.
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.(safetensors|bin|gguf)$", re.IGNORECASE)


def format_for(filename: str) -> AssetFormat:
    """Return the format implied by a weight file's extension."""
    lowered = filename.lower()
    for suffix, asset_format in WEIGHT_FORMATS:
        if lowered.endswith(suffix):
            return asset_format
    return AssetFormat.UNKNOWN


def classify_by_name(filename: str) -> tuple[ModelType, Framework] | None:
    """Infer a model type from well-known filename conventions.

    The weakest signal available, used only where a file sits alone with no config
    beside it — which is exactly how YOLO and SAM weights are usually kept.
    """
    lowered = filename.lower()
    for markers, model_type, framework in VISION_NAME_MARKERS:
        if any(marker in lowered for marker in markers):
            return model_type, framework
    return None


class OllamaStoreDetector(BaseDetector):
    """Detects an Ollama model store and emits one asset per manifest.

    The store keeps every model's weights in one flat, hash-named blob directory, so the
    directory structure cannot be used to separate them. The manifests are the only thing
    that can, which is why this detector reads them rather than deferring to a parser.
    """

    name = "ollama_store"
    priority = PRIORITY_STORE

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one model per manifest found in the store."""
        if not is_ollama_store(ctx):
            return []

        results: list[DetectionResult] = []
        for model in discover_ollama_models(ctx, ctx.tree):
            files = [model.model_blob_path] if model.model_blob_path else []
            results.append(
                DetectionResult(
                    kind=AssetKind.MODEL,
                    name=model.reference,
                    # The manifest is the only path unique to this model; the blob
                    # directory is shared by every model in the store.
                    root_path=model.manifest_path,
                    detector=self.name,
                    subkind=ModelType.LLM.value,
                    is_single_file=True,
                    format=AssetFormat.GGUF,
                    framework=Framework.OLLAMA,
                    confidence=1.0,
                    evidence={
                        "ollama_reference": model.reference,
                        "layers": len(model.layers),
                        "blob": os.path.basename(model.model_blob_path)
                        if model.model_blob_path
                        else None,
                    },
                    explicit_files=files,
                    # Every model in the store is independent, so the store directory
                    # must stay open for the rest of them.
                    claims_subtree=False,
                )
            )
        return results


class HfCacheDetector(BaseDetector):
    """Detects a HuggingFace cache repository and points at its snapshot."""

    name = "hf_cache"
    priority = PRIORITY_CACHE

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one asset rooted at the cache directory, with content in the snapshot."""
        if not is_cache_repo_dir(ctx.name):
            return []

        snapshot_path, revision = find_snapshot(ctx)
        kind = AssetKind.DATASET if ctx.name.startswith("datasets--") else AssetKind.MODEL

        return [
            DetectionResult(
                kind=kind,
                name=ctx.name,
                root_path=ctx.path,
                detector=self.name,
                confidence=1.0,
                evidence={"revision": revision, "snapshot": snapshot_path},
                # Files are gathered from the snapshot, but the asset is identified by
                # the cache directory, which is what survives a revision change.
                content_root=snapshot_path,
            )
        ]


class DiffusersDetector(BaseDetector):
    """Detects a Diffusers pipeline by its ``model_index.json`` manifest."""

    name = "diffusers"
    priority = PRIORITY_PIPELINE

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one image-generation model for the whole pipeline directory."""
        if not isinstance(ctx.read_json("model_index.json"), dict):
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.MODEL,
                subkind=ModelType.IMAGE_GENERATION.value,
                format=AssetFormat.SAFETENSORS
                if ctx.glob_subtree("*.safetensors", limit=1)
                else AssetFormat.PYTORCH,
                framework=Framework.DIFFUSERS,
                evidence={"marker": "model_index.json", "components": sorted(ctx.child_dir_names)},
            )
        ]


class SentenceTransformerDetector(BaseDetector):
    """Detects a Sentence-Transformers model by its ``modules.json`` manifest."""

    name = "sentence_transformers"
    priority = PRIORITY_PIPELINE

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one embedding model."""
        if not ctx.has_any("modules.json", "config_sentence_transformers.json"):
            return []
        # `modules.json` alone is ambiguous; requiring a config or weights beside it
        # avoids claiming an unrelated directory that happens to use the name.
        if not ctx.has("config.json") and not ctx.glob_subtree("*.safetensors", limit=1):
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.MODEL,
                subkind=ModelType.EMBEDDING.value,
                format=AssetFormat.SAFETENSORS
                if ctx.glob_subtree("*.safetensors", limit=1)
                else AssetFormat.PYTORCH,
                framework=Framework.SENTENCE_TRANSFORMERS,
                evidence={"marker": "modules.json"},
            )
        ]


class PeftAdapterDetector(BaseDetector):
    """Detects a PEFT/LoRA adapter.

    Ranked above the plain-repository detector because an adapter directory looks like a
    small model repository, and misfiling a 30 MB adapter as an 8 B model is exactly the
    confusion this catalogue exists to remove.
    """

    name = "peft_adapter"
    priority = PRIORITY_ADAPTER

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one adapter asset."""
        has_config = isinstance(ctx.read_json("adapter_config.json"), dict)
        has_weights = ctx.has_any(*ADAPTER_WEIGHT_NAMES)
        if not has_config and not has_weights:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.ADAPTER,
                subkind=ModelType.LORA.value,
                format=AssetFormat.SAFETENSORS
                if ctx.glob("*.safetensors")
                else AssetFormat.PYTORCH,
                framework=Framework.PEFT,
                confidence=1.0 if has_config else 0.8,
                evidence={
                    "adapter_config": has_config,
                    "adapter_weights": has_weights,
                },
            )
        ]


class HfRepoDetector(BaseDetector):
    """Detects a standard HuggingFace model directory: a ``config.json`` beside weights."""

    name = "hf_repo"
    priority = PRIORITY_REPO

    #: Weight patterns that, with a config, make a directory a model repository.
    WEIGHT_PATTERNS = (
        "*.safetensors",
        "pytorch_model*.bin",
        "model*.bin",
        "tf_model*.h5",
        "flax_model*.msgpack",
        "*.onnx",
        "model.pdparams",
    )

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one model asset when a config sits beside recognisable weights."""
        config = ctx.read_json("config.json")
        if not isinstance(config, dict):
            return []

        weights = [entry for pattern in self.WEIGHT_PATTERNS for entry in ctx.glob(pattern)]
        if not weights:
            # A config with no weights beside it is either a component of a larger
            # pipeline (a `text_encoder/` subdirectory) or an incomplete download. The
            # parent's detector claims the former; the latter is caught by health checks
            # only if it is catalogued, so it is claimed here at low confidence.
            if not ctx.has_any("tokenizer_config.json", "preprocessor_config.json"):
                return []
            return [
                self._result(
                    ctx,
                    kind=AssetKind.MODEL,
                    framework=Framework.TRANSFORMERS,
                    confidence=0.4,
                    evidence={"config_without_weights": True},
                )
            ]

        return [
            self._result(
                ctx,
                kind=AssetKind.MODEL,
                format=format_for(weights[0].name),
                framework=Framework.TRANSFORMERS,
                evidence={
                    "weight_files": len(weights),
                    "sharded": any(SHARD_RE.search(entry.name) for entry in weights),
                },
            )
        ]


class TensorFlowSavedModelDetector(BaseDetector):
    """Detects a TensorFlow SavedModel directory."""

    name = "tensorflow_savedmodel"
    priority = PRIORITY_FRAMEWORK

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one model when the SavedModel protobuf and variables are present."""
        if not ctx.has_any("saved_model.pb", "saved_model.pbtxt"):
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.MODEL,
                format=AssetFormat.TENSORFLOW,
                framework=Framework.TENSORFLOW,
                evidence={
                    "marker": "saved_model.pb",
                    "has_variables": ctx.has_dir("variables"),
                },
            )
        ]


class OpenVinoDetector(BaseDetector):
    """Detects an OpenVINO IR model: paired ``.xml`` topology and ``.bin`` weights."""

    name = "openvino"
    priority = PRIORITY_FRAMEWORK

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one model per matched ``.xml``/``.bin`` pair."""
        results: list[DetectionResult] = []
        bin_stems = {entry.stem.lower() for entry in ctx.glob("*.bin")}

        for xml_entry in ctx.glob("*.xml"):
            if xml_entry.stem.lower() not in bin_stems:
                continue
            # `.xml` beside `.bin` also describes Pascal VOC annotations and countless
            # other things; requiring the stems to match makes the pairing meaningful.
            results.append(
                DetectionResult(
                    kind=AssetKind.MODEL,
                    name=xml_entry.stem,
                    root_path=ctx.path if len(bin_stems) == 1 else xml_entry.path,
                    detector=self.name,
                    is_single_file=len(bin_stems) > 1,
                    format=AssetFormat.OPENVINO,
                    framework=Framework.OPENVINO,
                    confidence=0.8,
                    evidence={"topology": xml_entry.name},
                )
            )
        return results


class MlxDetector(BaseDetector):
    """Detects an MLX model by Apple's converter marker files."""

    name = "mlx"
    priority = PRIORITY_FRAMEWORK

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one model when MLX conversion markers are present."""
        config = ctx.read_json("config.json")
        is_mlx = ctx.has("mlx_config.json") or (
            isinstance(config, dict) and "quantization" in config and ctx.glob("*.npz")
        )
        if not is_mlx:
            return []

        return [
            self._result(
                ctx,
                kind=AssetKind.MODEL,
                format=AssetFormat.MLX,
                framework=Framework.MLX,
                confidence=0.7,
                evidence={"marker": "mlx"},
            )
        ]


class LooseWeightsDetector(BaseDetector):
    r"""Detects standalone weight files with no surrounding repository structure.

    The catch-all, and the reason a ``D:\\Models\\GGUF`` folder holding twenty unrelated
    quantisations becomes twenty catalogued models rather than one directory-shaped blob.
    Multi-part shards are grouped back into a single asset.
    """

    name = "loose_weights"
    priority = PRIORITY_LOOSE_WEIGHTS

    #: Extensions treated as self-contained models when found on their own.
    STANDALONE_EXTENSIONS = (
        "*.gguf", "*.ggml", "*.onnx", "*.engine", "*.plan",
        "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.keras", "*.h5", "*.tflite",
    )

    #: Files below this size are companions — tokenisers, projections, configs — rather
    #: than models in their own right.
    MIN_SIZE_BYTES = 1024 * 1024

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one asset per standalone weight file."""
        candidates = [
            entry
            for pattern in self.STANDALONE_EXTENSIONS
            for entry in ctx.glob(pattern)
            if entry.size >= self.MIN_SIZE_BYTES
        ]
        if not candidates:
            return []

        results: list[DetectionResult] = []
        for group_name, entries in self._group_shards(candidates).items():
            primary = entries[0]
            classified = classify_by_name(primary.name)
            model_type, framework = classified or (ModelType.UNKNOWN, Framework.UNKNOWN)

            results.append(
                DetectionResult(
                    kind=AssetKind.MODEL,
                    name=group_name,
                    root_path=primary.path,
                    detector=self.name,
                    subkind=model_type.value if model_type is not ModelType.UNKNOWN else None,
                    is_single_file=True,
                    format=format_for(primary.name),
                    framework=framework,
                    confidence=0.9 if classified else 0.6,
                    evidence={
                        "standalone": True,
                        "shards": len(entries),
                        "name_matched": classified is not None,
                    },
                    explicit_files=[entry.path for entry in entries],
                    claims_subtree=False,
                )
            )
        return results

    def _group_shards(self, entries: list[FileEntry]) -> dict[str, list[FileEntry]]:
        """Group multi-part shards under one name.

        ``model-00001-of-00003.gguf`` and its siblings are one model, not three.
        """
        groups: dict[str, list[FileEntry]] = {}
        for entry in sorted(entries, key=lambda item: item.name):
            match = SHARD_RE.search(entry.name)
            key = entry.name[: match.start()] if match else entry.stem
            groups.setdefault(key, []).append(entry)
        return groups
