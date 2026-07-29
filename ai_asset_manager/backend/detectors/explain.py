"""Why a detector concluded what it did.

Every detector already records what it saw. What it did not do was say so in a form a
person can read: ``{"marker": "model_index.json", "components": ["unet", "vae"]}`` is
evidence, but it is not an explanation, and "why is this an OCR model?" was answerable only
by reading the detector's source.

This module turns evidence into a sentence and a checklist. It does it in one place rather
than in forty detectors for the same reason fact precedence lives in one place: a detector
should stay twenty lines, and the phrasing should be consistent whichever one fired.

The rendering is deliberately dumb. It reads keys the detectors already write, and where a
detector has something specific to say it writes an ``evidence["signals"]`` list and this
module prints it verbatim. Nothing here re-derives a classification — an explanation that
disagreed with the verdict would be worse than none.
"""

from __future__ import annotations

from typing import Any

from ai_asset_manager.backend.detectors.base import DetectionResult
from ai_asset_manager.backend.metadata.records import AssetRecord

#: Evidence keys rendered as a plain bullet when true, mapped to how they read.
_FLAG_LABELS: dict[str, str] = {
    "adapter_config": "adapter_config.json",
    "adapter_weights": "adapter weight file",
    "has_variables": "variables/ directory",
    "sharded": "multi-part shards",
    "standalone": "self-contained weight file",
    "config_without_weights": "config present, weights missing",
    "security_dataset": "security data layout",
    "archive": "archive listed without extraction",
    "graph": "inference graph",
    "params": "parameter file",
    "is_hf_cache": "HuggingFace cache layout",
}

#: Evidence keys rendered as ``label: value``, mapped to how the label reads.
_VALUE_LABELS: dict[str, str] = {
    "marker": "marker file",
    "revision": "revision",
    "weight_files": "weight file(s)",
    "shards": "shard(s)",
    "components": "pipeline components",
    "layers": "manifest layer(s)",
    "ollama_reference": "Ollama reference",
    "topology": "topology file",
    "stage": "pipeline stage",
    "archive_format": "container format",
    "members_listed": "member(s) listed",
    "known_dataset": "public dataset",
    "evidence_score": "evidence score",
    "class_names": "classes",
    "split_dirs": "splits",
    "annotation_files": "annotation file(s)",
}

#: Keys that are bookkeeping rather than evidence, and are never rendered.
_SUPPRESSED: frozenset[str] = frozenset(
    {
        "provenance", "explanation", "identity", "signals", "unchanged",
        "snapshot", "extracted", "listing_truncated", "metadata_read",
    }
)

#: Cap on rendered bullets. An explanation is a summary; a detector that recorded thirty
#: observations should show the strongest ones and stop.
MAX_SIGNALS = 8


def explain(record: AssetRecord, detection: DetectionResult) -> dict[str, Any]:
    """Return a human-readable account of why an asset was classified as it was.

    Args:
        record: The finished record, consulted for what the parsers added.
        detection: The detector's own verdict and observations.

    Returns:
        A mapping with ``summary``, ``confidence``, ``detector`` and ``signals`` — the
        last a list of short phrases, strongest first. Stored on the asset so the
        explanation survives into the catalogue and does not need the scanner to reproduce.
    """
    signals = _signals(record, detection)
    return {
        "summary": _summary(record),
        # Stored unrounded. Rounding here and again at display time made one panel show
        # the same verdict as both 82% and 83%: `round(0.825, 2)` is 0.83, while the
        # asset's own confidence rounds straight to 82. One rounding, at the edge.
        "confidence": detection.confidence,
        "detector": detection.detector,
        "signals": signals[:MAX_SIGNALS],
    }


def _summary(record: AssetRecord) -> str:
    """Return the one-line verdict: what this is, in words."""
    subject = record.subkind or record.kind.value
    label = subject.replace("_", " ").strip()

    identity = record.evidence.get("identity")
    if isinstance(identity, dict):
        product = identity.get("product") or identity.get("vendor")
        if product:
            return f"{label} shipped by {product}".strip()

    return label or record.kind.value


def _signals(record: AssetRecord, detection: DetectionResult) -> list[str]:
    """Return the evidence, as phrases, strongest first.

    A detector's own ``signals`` list comes first and unchanged: where a detector went to
    the trouble of phrasing its reasoning, that phrasing is better than anything this
    module could reconstruct from the keys around it.
    """
    rendered: list[str] = []

    explicit = detection.evidence.get("signals")
    if isinstance(explicit, (list, tuple)):
        rendered.extend(str(item) for item in explicit if item)

    for key, value in detection.evidence.items():
        if key in _SUPPRESSED or value in (None, False, "", [], {}):
            continue
        phrase = _render(key, value)
        if phrase and phrase not in rendered:
            rendered.append(phrase)

    rendered.extend(_parser_signals(record))
    return rendered


def _render(key: str, value: Any) -> str | None:
    """Turn one evidence entry into a phrase, or ``None`` when it does not read as one."""
    if key in _FLAG_LABELS:
        return _FLAG_LABELS[key] if value is True else None

    label = _VALUE_LABELS.get(key)
    if label is None:
        return None

    if isinstance(value, (list, tuple)):
        shown = ", ".join(str(item) for item in list(value)[:5])
        return f"{label}: {shown}" if shown else None
    return f"{label}: {value}"


def _parser_signals(record: AssetRecord) -> list[str]:
    """Return the corroborating facts the parsers established.

    Distinct from the detector's evidence and worth showing beside it: the detector said
    "a config sits beside weights", and the parser said "the config declares
    Qwen2VLForConditionalGeneration". Together they are an explanation; separately they are
    two half-answers.
    """
    found: list[str] = []

    model = record.model
    if model is not None:
        if model.architecture:
            found.append(f"architecture {model.architecture}")
        if model.pipeline_tag:
            found.append(f"declared task {model.pipeline_tag}")
        if model.param_count:
            precision = "exact" if model.param_count_is_exact else "estimated"
            found.append(f"{model.param_count:,} parameters ({precision})")
        if model.quantization:
            found.append(f"quantised {model.quantization}")

    dataset = record.dataset
    if dataset is not None:
        if dataset.num_classes:
            found.append(f"{dataset.num_classes} class(es)")
        if dataset.splits:
            found.append(f"splits: {', '.join(sorted(dataset.splits))}")
        if dataset.num_images:
            found.append(f"{dataset.num_images:,} image(s)")

    return found


def explanation_of(evidence: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Return an asset's stored explanation as ``(heading, signals)``.

    Kept here rather than in the CLI so the API, the exporters and the terminal all phrase
    it the same way, and returned unformatted so each of them can decorate it in its own
    idiom.

    Examples:
        >>> explanation_of({"explanation": {"summary": "ocr", "confidence": 0.97,
        ...                                 "signals": ["marker file: config.json"]}})
        ('OCR — confidence 97%', ['marker file: config.json'])
    """
    payload = evidence.get("explanation")
    if not isinstance(payload, dict):
        return None

    summary = str(payload.get("summary") or "").strip()
    confidence = payload.get("confidence")
    heading = _titleise(summary) if summary else "Classified"
    if isinstance(confidence, (int, float)):
        heading = f"{heading} — confidence {round(float(confidence) * 100)}%"

    raw = payload.get("signals")
    signals = (
        [str(item) for item in raw if item] if isinstance(raw, (list, tuple)) else []
    )
    return heading, signals


def render_explanation(evidence: dict[str, Any]) -> list[str]:
    """Return an asset's stored explanation as plain display lines.

    Examples:
        >>> render_explanation({"explanation": {"summary": "ocr", "confidence": 0.97,
        ...                                     "signals": ["marker file: config.json"]}})
        ['OCR — confidence 97%', '  ✓ marker file: config.json']
    """
    found = explanation_of(evidence)
    if found is None:
        return []
    heading, signals = found
    return [heading, *(f"  ✓ {signal}" for signal in signals)]


def _titleise(text: str) -> str:
    """Capitalise a summary for display, keeping known acronyms upper-cased."""
    from ai_asset_manager.backend.identity.vendors import CASING

    words = [
        CASING.get(word.lower(), word[:1].upper() + word[1:])
        for word in text.split()
        if word
    ]
    return " ".join(words)
