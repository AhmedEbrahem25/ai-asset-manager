"""Knowledge every model shares, whatever it is for.

Parameter counts, quantisation, tokenizers, shard completeness. Task-specific model
knowledge lives in the plugin for that domain; this is the part that is true of a YOLO
checkpoint and a 70B language model alike.

Like everything else in the taxonomy, all of it comes from what the scanner already
recorded. The shard check in particular is worth noting: a sharded model names its own
expected total in every filename — ``model-00001-of-00004.safetensors`` — so counting the
recorded names is enough to prove a download finished, with no file opened and no
directory walked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import AssetFormat, Framework, ModelType, Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import is_model
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import AssetProfile, Finding

#: Filenames that mean a tokenizer is present. Any one of them is enough — repositories
#: ship different subsets depending on how the tokenizer was trained.
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
    "vocab.txt", "spiece.model", "merges.txt", "sentencepiece.bpe.model",
    "preprocessor_config.json",
)

#: Extensions that hold weights, mapped to the format name reported.
WEIGHT_EXTENSIONS = {
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".bin": "pytorch",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".engine": "tensorrt",
    ".plan": "tensorrt",
    ".tflite": "tflite",
    ".mlmodel": "coreml",
    ".npz": "numpy",
    ".ckpt": "checkpoint",
    ".h5": "keras",
    ".keras": "keras",
    ".pdparams": "paddle",
}

#: A shard filename states how many shards there are supposed to be.
_SHARD_PATTERN = re.compile(r"-(\d{5})-of-(\d{5})\.(safetensors|bin|gguf)$")

#: Model types that are useless without a tokenizer.
_NEEDS_TOKENIZER = frozenset(
    {ModelType.LLM, ModelType.VISION_LANGUAGE, ModelType.OCR, ModelType.EMBEDDING,
     ModelType.RERANKER, ModelType.MULTIMODAL}
)

#: Frameworks whose repositories are laid out as a directory of separate files, so that a
#: missing config or tokenizer really is missing. A GGUF or Ollama model carries both
#: inside the weight file, and judging one by the same rules would report every quantised
#: model as broken.
_NEEDS_CONFIG = frozenset(
    {Framework.TRANSFORMERS, Framework.SENTENCE_TRANSFORMERS, Framework.PEFT}
)

#: Formats that only exist because the scanner successfully parsed weights.
_WEIGHT_FORMATS = frozenset(
    value for value in AssetFormat if value is not AssetFormat.UNKNOWN
)


def register(registry: TaxonomyRegistry) -> None:
    """Register model statistics and health rules."""
    registry.add_statistic(_model_statistics, name="model")
    registry.add_health_rule(_weights_present, name="model.no_weights")
    registry.add_health_rule(_shards_complete, name="model.missing_shards")
    registry.add_health_rule(_tokenizer_present, name="model.no_tokenizer")
    registry.add_health_rule(_config_present, name="model.no_config")


def _model_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return what defines a model: size, precision, context and provenance."""
    if not is_model(profile) or profile.model is None:
        return {}

    details = profile.model
    stats: dict[str, Any] = {}

    for key, value in (
        ("parameters", details.param_count),
        ("quantization", details.quantization),
        ("context_length", details.context_length),
        ("hidden_size", details.hidden_size),
        ("layers", details.num_layers),
        ("vocab_size", details.vocab_size),
        ("tensors", details.tensor_count),
        ("repo_id", details.repo_id),
        ("revision", details.revision),
        ("author", details.author),
        ("license", details.license),
        ("base_model", details.base_model),
        ("pipeline_tag", details.pipeline_tag),
    ):
        if value:
            stats[key] = value

    if details.param_count:
        stats["parameters_exact"] = details.param_count_is_exact
    if details.precision and details.precision != "unknown":
        stats["precision"] = details.precision

    if profile.files.loaded:
        stats["tokenizer"] = profile.files.has_name(*TOKENIZER_FILES)

        formats = {
            name
            for extension, name in WEIGHT_EXTENSIONS.items()
            if profile.files.count(extension)
        }
        if formats:
            stats["weight_formats"] = sorted(formats)

        expected, present = _shard_counts(profile)
        if expected:
            stats["shards"] = f"{present}/{expected}"

    return stats


def _shard_counts(profile: AssetProfile) -> tuple[int, int]:
    """Return ``(expected, present)`` shard counts from the recorded filenames."""
    expected = 0
    present = 0
    for name in profile.files.names:
        match = _SHARD_PATTERN.search(name)
        if match:
            present += 1
            expected = max(expected, int(match.group(2)))
    return expected, present


def _weights_present(profile: AssetProfile) -> Sequence[Finding]:
    """Report a model directory with no weights in it.

    Filenames are the weaker of the two signals and are checked second. Ollama stores its
    weights in a digest-named blob with no extension at all, so a rule reading only
    extensions would call every Ollama model metadata-only. The format the scanner recorded
    is the stronger evidence: it is set by parsing the weights, so a known format means
    there were weights there to parse.
    """
    if not is_model(profile) or not profile.files.loaded:
        return ()

    if profile.format in _WEIGHT_FORMATS:
        return ()
    if any(profile.files.count(extension) for extension in WEIGHT_EXTENSIONS):
        return ()

    return (
        Finding(
            code="model.no_weights",
            severity=Severity.ERROR,
            message="No weight files — configuration and tokenizer only",
            fix_hint="A metadata-only checkout. Fetch the weights to make it usable.",
        ),
    )


def _shards_complete(profile: AssetProfile) -> Sequence[Finding]:
    """Report a sharded model that is missing shards.

    The most valuable check in the set, because a partially downloaded sharded model looks
    entirely healthy in a file browser: the directory is there, most of the weights are
    there, and the failure only shows up when something tries to load it.
    """
    if not is_model(profile) or not profile.files.loaded:
        return ()

    expected, present = _shard_counts(profile)
    if not expected or present >= expected:
        return ()

    return (
        Finding(
            code="model.missing_shards",
            severity=Severity.ERROR,
            message=f"Only {present} of {expected} weight shards present",
            fix_hint="Resume the download; the model cannot load until every shard exists.",
        ),
    )


def _tokenizer_present(profile: AssetProfile) -> Sequence[Finding]:
    """Report a text model with no tokenizer beside it."""
    if not is_model(profile) or not profile.files.loaded or profile.model is None:
        return ()

    # Only directory-shaped repositories are judged. GGUF embeds its tokenizer in the file
    # header and Ollama keeps it in the manifest, so neither needs one beside it.
    if profile.framework not in _NEEDS_CONFIG:
        return ()
    if profile.model.model_type not in _NEEDS_TOKENIZER:
        return ()
    if profile.files.has_name(*TOKENIZER_FILES):
        return ()

    return (
        Finding(
            code="model.no_tokenizer",
            severity=Severity.WARNING,
            message="No tokenizer files",
            fix_hint="Fetch the tokenizer from the source repository, or point at the base model.",
        ),
    )


def _config_present(profile: AssetProfile) -> Sequence[Finding]:
    """Report a framework repository missing the config that describes it."""
    if not is_model(profile) or not profile.files.loaded:
        return ()
    if profile.framework not in _NEEDS_CONFIG:
        return ()
    if profile.files.has_name("config.json", "adapter_config.json", "model_index.json"):
        return ()

    return (
        Finding(
            code="model.no_config",
            severity=Severity.WARNING,
            message="No config.json",
            fix_hint="Transformers cannot instantiate the architecture without it.",
        ),
    )
