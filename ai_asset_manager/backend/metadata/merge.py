"""Fact resolution: the single place precedence is decided.

Parsers assert; they never decide. Every competing claim about an asset is settled here by
one rule, applied uniformly:

    explicit config > binary header > sidecar doc > filename > directory name

with a parser's own confidence breaking ties within a rank. Keeping this in one function
is what allows forty-odd parsers to stay short and independent, and makes "why does this
model say it is a LoRA?" answerable by reading a single file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_asset_manager.backend.detectors.base import DetectionResult
from ai_asset_manager.backend.detectors.explain import explain
from ai_asset_manager.backend.identity import identify
from ai_asset_manager.backend.metadata.records import (
    AssetRecord,
    DatasetFacts,
    FileRecord,
    ModelFacts,
)
from ai_asset_manager.backend.models.enums import (
    AssetFormat,
    AssetKind,
    DatasetFormat,
    Framework,
    Modality,
    ModelType,
    Precision,
)
from ai_asset_manager.backend.parsers.base import FactSet
from ai_asset_manager.backend.scanner.types import FileEntry
from ai_asset_manager.backend.utils.hashing import fingerprint_entries
from ai_asset_manager.backend.utils.paths import classify_extension, get_drive, safe_relpath


def _coerce_enum[EnumT](value: Any, enum_cls: type[EnumT], default: EnumT) -> EnumT:
    """Convert a parser's string into an enum member, falling back to ``default``.

    Parsers emit plain strings so they never need to import the enum module; unknown
    values degrade to the default rather than raising, because an unrecognised format
    should still be catalogued.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)  # type: ignore[call-arg]
        except ValueError:
            return default
    return default


def _as_int(value: Any) -> int | None:
    """Coerce a value to a positive integer, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def _as_str_list(value: Any) -> list[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def build_model_facts(facts: FactSet, *, fallback_subkind: str | None = None) -> ModelFacts:
    """Resolve model metadata from a fact set.

    Args:
        facts: Everything the parsers asserted.
        fallback_subkind: The detector's own classification, used when no parser produced
            a model type. A standalone ``yolov8n.pt`` has no config to read, so the
            detector's structural verdict is the only signal available.
    """
    model_type = _coerce_enum(facts.get("model_type"), ModelType, ModelType.UNKNOWN)
    if model_type is ModelType.UNKNOWN and fallback_subkind:
        model_type = _coerce_enum(fallback_subkind, ModelType, ModelType.UNKNOWN)

    resolved = ModelFacts(
        model_type=model_type,
        architecture=facts.get("architecture"),
        param_count=_as_int(facts.get("param_count")),
        param_count_is_exact=bool(facts.get("param_count_is_exact", False)),
        quantization=facts.get("quantization"),
        precision=_coerce_enum(facts.get("precision"), Precision, Precision.UNKNOWN),
        context_length=_as_int(facts.get("context_length")),
        hidden_size=_as_int(facts.get("hidden_size")),
        num_layers=_as_int(facts.get("num_layers")),
        vocab_size=_as_int(facts.get("vocab_size")),
        tensor_count=_as_int(facts.get("tensor_count")),
        repo_id=facts.get("repo_id"),
        revision=facts.get("revision"),
        author=facts.get("author"),
        license=facts.get("license"),
        description=facts.get("description"),
        base_model=facts.get("base_model"),
        pipeline_tag=facts.get("pipeline_tag"),
        library_name=facts.get("library_name"),
        card_tags=_as_str_list(facts.get("card_tags")),
    )
    resolved.extra = _collect_extra(facts, _MODEL_EXTRA_KEYS)
    return resolved


def build_dataset_facts(
    facts: FactSet,
    *,
    fallback_subkind: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> DatasetFacts:
    """Resolve dataset metadata from a fact set.

    Args:
        facts: Everything the parsers asserted.
        fallback_subkind: The detector's structural verdict — for datasets this is
            usually the *only* source of the format, since no dataset ships a file
            declaring "I am COCO".
        evidence: The detector's observations, mined for counts and class lists it
            already computed while matching.
    """
    class_names = _as_str_list(facts.get("class_names"))
    splits = facts.get("splits")
    modalities = _as_str_list(facts.get("modalities"))

    evidence = evidence or {}
    if not class_names:
        class_names = _as_str_list(evidence.get("class_names"))
    if not modalities:
        modalities = _as_str_list(evidence.get("modalities"))

    dataset_format = _coerce_enum(
        facts.get("dataset_format"), DatasetFormat, DatasetFormat.UNKNOWN
    )
    if dataset_format is DatasetFormat.UNKNOWN and fallback_subkind:
        dataset_format = _coerce_enum(fallback_subkind, DatasetFormat, DatasetFormat.UNKNOWN)
    if dataset_format is DatasetFormat.UNKNOWN and facts.get("is_hf_cache"):
        dataset_format = DatasetFormat.HF_DATASET

    resolved = DatasetFacts(
        dataset_format=dataset_format,
        task=facts.get("task"),
        num_images=_as_int(facts.get("num_images")) or 0,
        num_videos=_as_int(facts.get("num_videos")) or 0,
        num_audio_files=_as_int(facts.get("num_audio_files")) or 0,
        num_text_files=_as_int(facts.get("num_text_files")) or 0,
        num_annotations=_as_int(facts.get("num_annotations")) or 0,
        # An explicit class count wins; otherwise the label list length stands in.
        num_classes=_as_int(facts.get("num_classes")) or (len(class_names) or None),
        class_names=class_names,
        splits=splits if isinstance(splits, dict) else {},
        modalities=modalities,
        has_bounding_boxes=bool(facts.get("has_bounding_boxes", False)),
        has_masks=bool(facts.get("has_masks", False)),
        has_keypoints=bool(facts.get("has_keypoints", False)),
        has_lidar=bool(facts.get("has_lidar", False)),
        has_radar=bool(facts.get("has_radar", False)),
        has_depth=bool(facts.get("has_depth", False)),
        has_thermal=bool(facts.get("has_thermal", False)),
        repo_id=facts.get("repo_id"),
        license=facts.get("license"),
        description=facts.get("description"),
    )
    resolved.extra = _collect_extra(facts, _DATASET_EXTRA_KEYS)
    # A handful of the detector's own observations belong with the dataset rather than in
    # the asset's evidence blob: taxonomy plugins receive dataset facts and never see
    # evidence, so "which public dataset is this?" has to arrive here to be usable.
    for key in _DATASET_EVIDENCE_KEYS:
        value = evidence.get(key)
        if value is not None and key not in resolved.extra:
            resolved.extra[key] = value
    return resolved


#: Fields kept in ``extra`` rather than promoted to columns: useful for display and
#: search but too format-specific or too rare to justify schema real estate.
_MODEL_EXTRA_KEYS = frozenset(
    {
        "adapter_type", "lora_rank", "lora_alpha", "target_modules", "gguf_version",
        "shard_count", "onnx_ir_version", "onnx_opset", "producer", "producer_version",
        "source_framework", "torch_version", "torch_legacy_format", "transformers_version",
        "diffusers_version", "pipeline_class", "components", "tokenizer_class",
        "is_instruct", "has_tokenizer", "has_generation_config", "embedding_dimension",
        "sentence_transformers_version", "languages", "training_datasets", "cache_layout",
        "is_hf_cache", "num_classes", "class_names", "task",
    }
)

_DATASET_EXTRA_KEYS = frozenset(
    {"cache_layout", "is_hf_cache", "annotation_files", "split_dirs", "card_tags", "languages"}
)

#: Detector observations copied from the detection's evidence onto the dataset itself.
_DATASET_EVIDENCE_KEYS = ("known_dataset", "security_dataset", "evidence_score")


def _collect_extra(facts: FactSet, keys: frozenset[str]) -> dict[str, Any]:
    """Gather supplementary fields into a JSON-serialisable mapping."""
    extra: dict[str, Any] = {}
    for key in keys:
        value = facts.get(key)
        if value is not None and value != [] and value != {}:
            extra[key] = value
    return extra


def _resolve_format(facts: FactSet, detection: DetectionResult) -> AssetFormat:
    """Resolve the storage format, preferring what a parser actually read."""
    parsed = _coerce_enum(facts.get("format"), AssetFormat, AssetFormat.UNKNOWN)
    if parsed is not AssetFormat.UNKNOWN:
        return parsed
    return detection.format


def _resolve_framework(facts: FactSet, detection: DetectionResult) -> Framework:
    """Resolve the framework, preferring what a parser actually read."""
    parsed = _coerce_enum(facts.get("framework"), Framework, Framework.UNKNOWN)
    if parsed is not Framework.UNKNOWN:
        return parsed
    return detection.framework


def build_file_records(files: list[FileEntry], root_path: str) -> list[FileRecord]:
    """Convert walked entries into persistable file records."""
    return [
        FileRecord(
            relpath=safe_relpath(entry.path, root_path),
            abspath=entry.path,
            size=entry.size,
            mtime=entry.mtime,
            inode=entry.inode,
            device=entry.device,
            nlink=entry.nlink,
            is_symlink=entry.is_symlink,
            is_payload=entry.is_payload,
            extension=entry.extension or None,
        )
        for entry in files
    ]


def compute_physical_size(files: list[FileEntry]) -> int:
    """Sum file sizes counting each physical extent once.

    Hardlinked and symlinked copies share storage. Summing apparent sizes would inflate
    every total in the catalogue and, worse, present linked copies as reclaimable space —
    sending the user to delete files that free nothing.

    Symlinks are excluded outright (their target is counted where it lives) and hardlinked
    files are counted once per ``(device, inode)``.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    for entry in files:
        if entry.is_symlink:
            continue
        identity = entry.identity
        if identity is not None and entry.nlink > 1:
            if identity in seen:
                continue
            seen.add(identity)
        total += entry.size
    return total


def build_asset_record(
    detection: DetectionResult,
    facts: FactSet,
    files: list[FileEntry],
) -> AssetRecord:
    """Assemble the final record for one asset.

    Args:
        detection: What the detector concluded, including the asset root.
        facts: Everything the parsers asserted about it.
        files: Every file belonging to the asset.

    Returns:
        A fully resolved :class:`AssetRecord`, ready to persist.
    """
    root_path = detection.root_path
    kind = _coerce_enum(facts.get("kind"), AssetKind, detection.kind)

    name = facts.get("name") or detection.name
    display_name = facts.get("display_name")

    size_bytes = sum(entry.size for entry in files if not entry.is_symlink)
    newest = max((entry.mtime for entry in files), default=None)
    oldest_created = min((entry.ctime for entry in files), default=None)
    latest_access = max((entry.atime for entry in files), default=None)

    record = AssetRecord(
        kind=kind,
        name=str(name),
        display_name=str(display_name) if display_name else None,
        root_path=root_path,
        subkind=detection.subkind,
        drive=get_drive(root_path),
        is_single_file=detection.is_single_file,
        size_bytes=size_bytes,
        physical_size_bytes=compute_physical_size(files),
        file_count=len(files),
        format=_resolve_format(facts, detection),
        framework=_resolve_framework(facts, detection),
        created_at=_to_datetime(oldest_created),
        modified_at=_to_datetime(newest),
        accessed_at=_to_datetime(latest_access),
        fingerprint=fingerprint_entries(
            (safe_relpath(entry.path, root_path), entry.size, entry.mtime) for entry in files
        ),
        detector=detection.detector,
        detection_confidence=detection.confidence,
        evidence=dict(detection.evidence),
        files=build_file_records(files, root_path),
        warnings=list(facts.warnings),
    )

    if kind in _STRUCTURAL_KINDS:
        # A project, a training run and a labelling task are none of them weights, so
        # attaching model details would invent an architecture and a parameter count for
        # something that has neither. Their subkind, evidence and file list say everything
        # that is true about them.
        record.subkind = record.subkind or detection.subkind
    elif kind is AssetKind.DATASET:
        record.dataset = build_dataset_facts(
            facts, fallback_subkind=detection.subkind, evidence=detection.evidence
        )
        _fill_dataset_counts(record.dataset, record.files)
    else:
        record.model = build_model_facts(facts, fallback_subkind=detection.subkind)
        # An adapter is an adapter wherever it is stored. A HuggingFace cache directory
        # is named `models--...` whatever it holds, so without this a cached LoRA would
        # be filed as a model while an identical local one is filed as an adapter.
        if record.model.model_type is ModelType.LORA:
            record.kind = AssetKind.ADAPTER
        record.subkind = record.subkind or record.model.model_type.value

    record.evidence["provenance"] = facts.provenance()
    _apply_identity(record, detection)
    record.evidence["explanation"] = explain(record, detection)
    return record


def _apply_identity(record: AssetRecord, detection: DetectionResult) -> None:
    r"""Attach vendor, product, source and — where it helps — a better name.

    Runs after everything else so it can see what the parsers found. A model that declared
    a repository id has already told us who made it and needs no help; one called ``model``
    inside ``Google\\Chrome\\User Data\\screen_ai`` has told us nothing, and the path has
    told us everything.

    The derived name goes to ``display_name`` rather than ``name``. ``name`` is what was
    read off the disk and stays that way — a user searching for the folder they created
    must still find it — while ``display_name`` is what the inventory shows.
    """
    identity = identify(
        record.root_path,
        name=record.display_name or record.name,
        is_single_file=record.is_single_file,
    )
    if identity.is_empty:
        return

    record.evidence["identity"] = identity.as_dict()
    if identity.display_name and not record.display_name:
        record.display_name = identity.display_name


#: Kinds that describe a *container of work* rather than a weight file or a corpus, and so
#: carry neither model nor dataset details.
#:
#: An archive is here for a sharper reason than the rest: its contents were listed, not
#: read. Attaching model details to a ``.zip`` that appears to hold weights would invent an
#: architecture and a parameter count from a filename, and the whole point of the archive
#: reader is that it does not open what it has not unpacked.
_STRUCTURAL_KINDS = frozenset(
    {
        AssetKind.PROJECT,
        AssetKind.EXPERIMENT,
        AssetKind.ANNOTATION_PROJECT,
        AssetKind.ARCHIVE,
    }
)

#: Extensions that hold annotations rather than media.
_ANNOTATION_EXTENSIONS = frozenset({".json", ".xml", ".txt", ".csv", ".yaml", ".yml"})


def _fill_dataset_counts(dataset: DatasetFacts, files: list[FileRecord]) -> None:
    """Count a dataset's contents from the files already walked.

    Costs nothing extra: the census runs over metadata gathered during the single walk,
    so a dataset of two million images is counted without a second traversal.

    Counts already supplied by a parser — split sizes from a dataset card, say — are not
    overwritten, since a declared row count is more meaningful than a file count.
    """
    images = videos = audio = text = annotations = 0
    for record in files:
        extension = (record.extension or "").lower()
        klass = classify_extension(record.relpath)
        if klass == "image":
            images += 1
        elif klass == "video":
            videos += 1
        elif klass == "audio":
            audio += 1
        elif extension in _ANNOTATION_EXTENSIONS:
            text += 1
            annotations += 1

    dataset.num_images = dataset.num_images or images
    dataset.num_videos = dataset.num_videos or videos
    dataset.num_audio_files = dataset.num_audio_files or audio
    dataset.num_text_files = dataset.num_text_files or text
    dataset.num_annotations = dataset.num_annotations or annotations

    if not dataset.modalities:
        inferred: list[str] = []
        if images:
            inferred.append(Modality.RGB.value)
        if videos:
            inferred.append(Modality.VIDEO.value)
        if audio:
            inferred.append(Modality.AUDIO.value)
        if text and not (images or videos or audio):
            inferred.append(Modality.TEXT.value)
        dataset.modalities = inferred


def _to_datetime(timestamp: float | None) -> datetime | None:
    """Convert a POSIX timestamp to an aware UTC datetime.

    Out-of-range values appear on files restored from archives with corrupt headers and
    would otherwise raise on some platforms.
    """
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
