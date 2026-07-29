"""Model and dataset card reader.

HuggingFace cards carry a YAML front-matter block delimited by ``---`` lines, holding the
licence, tags, base model and pipeline tag. It is the only place much of that appears —
``config.json`` never states a licence — so it is worth parsing, but it is author-written
prose rather than machine-generated, so its facts rank below explicit configuration.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from ai_asset_manager.backend.models.enums import FactSource, ModelType
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Front matter: leading ``---``, the block, then a closing ``---`` on its own line.
FRONT_MATTER_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)

#: Cap on the front-matter block. Cards occasionally embed large result tables; parsing
#: megabytes of YAML per asset would dominate scan time.
MAX_FRONT_MATTER_BYTES = 256 * 1024

#: Card filenames, in the order they are tried.
CARD_FILENAMES = ("README.md", "readme.md", "MODEL_CARD.md", "model_card.md", "DATASET_CARD.md")

#: ``pipeline_tag`` values mapped to a model type.
PIPELINE_TAG_TYPES: dict[str, ModelType] = {
    "text-generation": ModelType.LLM,
    "text2text-generation": ModelType.LLM,
    "conversational": ModelType.LLM,
    "fill-mask": ModelType.EMBEDDING,
    "feature-extraction": ModelType.EMBEDDING,
    "sentence-similarity": ModelType.EMBEDDING,
    "text-ranking": ModelType.RERANKER,
    "image-text-to-text": ModelType.VISION_LANGUAGE,
    "visual-question-answering": ModelType.VISION_LANGUAGE,
    "image-to-text": ModelType.VISION_LANGUAGE,
    "text-to-image": ModelType.IMAGE_GENERATION,
    "image-to-image": ModelType.IMAGE_GENERATION,
    "text-to-video": ModelType.IMAGE_GENERATION,
    "unconditional-image-generation": ModelType.IMAGE_GENERATION,
    "object-detection": ModelType.OBJECT_DETECTION,
    "zero-shot-object-detection": ModelType.OBJECT_DETECTION,
    "image-segmentation": ModelType.SEGMENTATION,
    "mask-generation": ModelType.SEGMENTATION,
    "depth-estimation": ModelType.VISION,
    "image-classification": ModelType.CLASSIFICATION,
    "zero-shot-image-classification": ModelType.MULTIMODAL,
    "video-classification": ModelType.VISION,
    "text-classification": ModelType.CLASSIFICATION,
    "token-classification": ModelType.CLASSIFICATION,
    "automatic-speech-recognition": ModelType.SPEECH_RECOGNITION,
    "audio-classification": ModelType.AUDIO,
    "text-to-speech": ModelType.TEXT_TO_SPEECH,
    "text-to-audio": ModelType.TEXT_TO_SPEECH,
    "keypoint-detection": ModelType.POSE,
}


def parse_front_matter(text: str) -> dict[str, Any] | None:
    """Extract and parse a card's YAML front matter.

    Args:
        text: Full card text.

    Returns:
        The parsed mapping, or ``None`` when there is no front matter or it is malformed.
        ``yaml.safe_load`` is used rather than ``load``: card YAML is untrusted input and
        the full loader can instantiate arbitrary Python objects.
    """
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return None

    block = match.group(1)
    if len(block) > MAX_FRONT_MATTER_BYTES:
        logger.debug("Front matter of %d bytes is too large to parse", len(block))
        return None

    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        logger.debug("Malformed card front matter: %s", exc)
        return None

    return parsed if isinstance(parsed, dict) else None


def extract_description(text: str, *, max_length: int = 600) -> str | None:
    """Pull a short description from a card's prose.

    Takes the first substantial paragraph after the front matter, skipping headings,
    badge images and HTML, which is what the top of a typical card is made of.
    """
    body = FRONT_MATTER_RE.sub("", text, count=1)
    for raw_paragraph in body.split("\n\n"):
        paragraph = " ".join(raw_paragraph.split())
        if len(paragraph) < 40:
            continue
        if paragraph.startswith(("#", ">", "|", "-", "*", "<", "!", "[")):
            continue
        if paragraph.count("](") > 2:  # a row of badges
            continue
        return paragraph[:max_length].rstrip() + ("…" if len(paragraph) > max_length else "")
    return None


def _as_str_list(value: Any) -> list[str]:
    """Coerce a scalar or list from YAML into a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


class ModelCardParser(BaseParser):
    """Reads licence, tags and provenance from a model or dataset card."""

    name = "model_card"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether any recognised card file is present."""
        return ctx.has_any(*CARD_FILENAMES)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Parse the card's front matter and opening prose."""
        facts = self._new_facts()

        text: str | None = None
        for filename in CARD_FILENAMES:
            if ctx.has(filename):
                text = ctx.read_text(filename)
                if text:
                    break
        if not text:
            return facts

        description = extract_description(text)
        if description:
            facts.add("description", description, source=FactSource.SIDECAR_DOC,
                      origin=self.name)

        front_matter = parse_front_matter(text)
        if front_matter is None:
            return facts

        self._parse_scalars(facts, front_matter)
        self._parse_lists(facts, front_matter)
        self._parse_dataset_fields(facts, front_matter)
        return facts

    def _parse_scalars(self, facts: FactSet, front_matter: dict[str, Any]) -> None:
        """Read single-valued front-matter fields."""
        for key, target in (
            ("license", "license"),
            ("license_name", "license"),
            ("library_name", "library_name"),
            ("pipeline_tag", "pipeline_tag"),
            ("model_name", "name"),
            ("pretty_name", "name"),
        ):
            value = front_matter.get(key)
            if isinstance(value, str) and value.strip():
                facts.add(target, value.strip(), source=FactSource.SIDECAR_DOC, origin=self.name)

        pipeline_tag = front_matter.get("pipeline_tag")
        if isinstance(pipeline_tag, str):
            model_type = PIPELINE_TAG_TYPES.get(pipeline_tag.strip().lower())
            if model_type is not None:
                facts.add("model_type", model_type.value, source=FactSource.SIDECAR_DOC,
                          confidence=0.9, origin=self.name)

    def _parse_lists(self, facts: FactSet, front_matter: dict[str, Any]) -> None:
        """Read list-valued front-matter fields."""
        tags = _as_str_list(front_matter.get("tags"))
        if tags:
            facts.add("card_tags", tags, source=FactSource.SIDECAR_DOC, origin=self.name)

        base_models = _as_str_list(front_matter.get("base_model"))
        if base_models:
            facts.add("base_model", base_models[0], source=FactSource.SIDECAR_DOC,
                      origin=self.name)

        languages = _as_str_list(front_matter.get("language"))
        if languages:
            facts.add("languages", languages, source=FactSource.SIDECAR_DOC, origin=self.name)

        datasets = _as_str_list(front_matter.get("datasets"))
        if datasets:
            facts.add("training_datasets", datasets, source=FactSource.SIDECAR_DOC,
                      origin=self.name)

    def _parse_dataset_fields(self, facts: FactSet, front_matter: dict[str, Any]) -> None:
        """Read the ``dataset_info`` block that dataset cards carry.

        Gives split names and row counts without opening a single Parquet file.
        """
        info = front_matter.get("dataset_info")
        if isinstance(info, list):
            info = info[0] if info and isinstance(info[0], dict) else None
        if not isinstance(info, dict):
            return

        splits = info.get("splits")
        if isinstance(splits, list):
            counts: dict[str, int] = {}
            for split in splits:
                if not isinstance(split, dict):
                    continue
                split_name = split.get("name")
                num_examples = split.get("num_examples")
                if isinstance(split_name, str) and isinstance(num_examples, int):
                    counts[split_name] = num_examples
            if counts:
                facts.add("splits", counts, source=FactSource.SIDECAR_DOC, origin=self.name)

        features = info.get("features")
        if isinstance(features, list):
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                class_label = feature.get("class_label")
                if isinstance(class_label, dict):
                    names = class_label.get("names")
                    if isinstance(names, dict):
                        facts.add("class_names", [str(v) for v in names.values()],
                                  source=FactSource.SIDECAR_DOC, origin=self.name)
                    elif isinstance(names, list):
                        facts.add("class_names", [str(v) for v in names],
                                  source=FactSource.SIDECAR_DOC, origin=self.name)
