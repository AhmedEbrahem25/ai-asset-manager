"""Turning catalogue rows into something plugins can reason about.

The taxonomy is deliberately ignorant of SQLAlchemy: a plugin receives an
:class:`~ai_asset_manager.backend.taxonomy.AssetProfile` and cannot reach a session, a
path or a file handle through it. This module is the one place that translation happens.

The interesting part is :func:`load_file_summaries`. Dataset intelligence — is there a
README, a validation split, a calibration folder, all four weight shards — is a set of
questions about an asset's file list, and the scanner already wrote that list down. Asking
the database is both instant and *more* honest than asking the disk, because it reports the
library as catalogued rather than as it happens to be mid-download.

Loading it is optional. A plain listing never needs it, so the engine only pays for it
when something actually asks about contents.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import (
    Asset,
    AssetFile,
    AssetLink,
    DatasetDetails,
    ModelDetails,
)
from ai_asset_manager.backend.taxonomy.types import (
    AssetProfile,
    DatasetFacts,
    FileSummary,
    ModelFacts,
)

#: How many asset ids to put in one ``IN`` clause. SQLite caps bound parameters at 999 by
#: default, and a library with thousands of assets would blow past that in one statement.
_ID_CHUNK = 500


def build_profile(
    asset: Asset,
    model: ModelDetails | None,
    dataset: DatasetDetails | None,
    files: FileSummary | None = None,
) -> AssetProfile:
    """Assemble the profile for one catalogued asset."""
    return AssetProfile(
        asset_id=asset.id,
        kind=asset.kind,
        name=asset.display_name or asset.name,
        path=asset.root_path,
        subkind=asset.subkind,
        framework=asset.framework,
        format=asset.format,
        drive=asset.drive,
        is_single_file=asset.is_single_file,
        size_bytes=asset.size_bytes,
        physical_size_bytes=asset.physical_size_bytes,
        file_count=asset.file_count,
        modified_at=asset.modified_at,
        created_at=asset.created_at,
        is_missing=asset.is_missing,
        detector=asset.detector,
        evidence=dict(asset.evidence or {}),
        tags=tuple(tag.name for tag in asset.tags),
        model=_model_facts(model),
        dataset=_dataset_facts(dataset),
        files=files or FileSummary(total=asset.file_count, total_bytes=asset.size_bytes),
    )


def load_file_summaries(
    session: Session, asset_ids: Sequence[int]
) -> dict[int, FileSummary]:
    """Return a file summary per asset, in as few queries as the id count allows.

    One query per chunk of ids rather than one per asset: a library of 2,000 assets would
    otherwise mean 2,000 round trips to answer a question the database can answer in four.
    """
    if not asset_ids:
        return {}

    accumulators: dict[int, _Accumulator] = {
        asset_id: _Accumulator() for asset_id in asset_ids
    }

    for chunk in _chunked(asset_ids, _ID_CHUNK):
        rows = session.execute(
            select(
                AssetFile.asset_id,
                AssetFile.relpath,
                AssetFile.extension,
                AssetFile.size_bytes,
            ).where(AssetFile.asset_id.in_(chunk))
        ).all()

        for asset_id, relpath, extension, size_bytes in rows:
            accumulator = accumulators.get(asset_id)
            if accumulator is not None:
                accumulator.add(relpath, extension, size_bytes or 0)

    return {
        asset_id: accumulator.finish() for asset_id, accumulator in accumulators.items()
    }


#: How an edge reads in a detail view, from the point of view of each end. A checkpoint
#: says "Produced by: train-3"; the run says "Produced: best.pt". Same row, two sentences.
_RELATION_LABELS: dict[str, tuple[str, str]] = {
    "belongs_to": ("Part of", "Contains"),
    "produced_by": ("Produced by", "Produced"),
    "adapts": ("Adapts", "Adapted by"),
    "derived_from": ("Derived from", "Source of"),
    "trained_on": ("Trained on", "Used to train"),
}

#: Cap on edges reported per asset. A project with sixty runs beneath it would otherwise
#: bury its own detail block; the count is what matters at that point, not the list.
MAX_LINKS_PER_ASSET = 8


def load_links(
    session: Session, asset_ids: Sequence[int]
) -> dict[int, tuple[tuple[str, str], ...]]:
    """Return each asset's relationships, described from that asset's own side.

    Both directions are loaded, because an edge is interesting to both ends: the run wants
    to list what it produced and the checkpoint wants to name the run that produced it, and
    the graph stores that once.
    """
    if not asset_ids:
        return {}

    wanted = set(asset_ids)
    names: dict[int, str] = {
        row[0]: row[1]
        for row in session.execute(
            select(Asset.id, Asset.name).where(Asset.id.in_(wanted))
        ).all()
    }
    collected: dict[int, list[tuple[str, str]]] = {}

    for chunk in _chunked(asset_ids, _ID_CHUNK):
        rows = session.execute(
            select(AssetLink.source_id, AssetLink.target_id, AssetLink.relation).where(
                AssetLink.source_id.in_(chunk) | AssetLink.target_id.in_(chunk)
            )
        ).all()

        for source_id, target_id, relation in rows:
            forward, reverse = _RELATION_LABELS.get(relation, (relation, relation))
            if source_id in wanted and target_id in names:
                collected.setdefault(source_id, []).append((forward, names[target_id]))
            if target_id in wanted and source_id in names:
                collected.setdefault(target_id, []).append((reverse, names[source_id]))

    return {
        asset_id: tuple(sorted(set(edges))[:MAX_LINKS_PER_ASSET])
        for asset_id, edges in collected.items()
    }


def _model_facts(model: ModelDetails | None) -> ModelFacts | None:
    """Copy model columns into the taxonomy's read-only view of them."""
    if model is None:
        return None
    return ModelFacts(
        model_type=model.model_type,
        architecture=model.architecture,
        param_count=model.param_count,
        param_count_is_exact=model.param_count_is_exact,
        quantization=model.quantization,
        precision=model.precision,
        context_length=model.context_length,
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        vocab_size=model.vocab_size,
        tensor_count=model.tensor_count,
        repo_id=model.repo_id,
        revision=model.revision,
        author=model.author,
        license=model.license,
        description=model.description,
        base_model=model.base_model,
        pipeline_tag=model.pipeline_tag,
        library_name=model.library_name,
        card_tags=tuple(model.card_tags or ()),
        extra=dict(model.extra or {}),
    )


def _dataset_facts(dataset: DatasetDetails | None) -> DatasetFacts | None:
    """Copy dataset columns into the taxonomy's read-only view of them."""
    if dataset is None:
        return None
    return DatasetFacts(
        dataset_format=dataset.dataset_format,
        task=dataset.task,
        num_images=dataset.num_images,
        num_videos=dataset.num_videos,
        num_audio_files=dataset.num_audio_files,
        num_text_files=dataset.num_text_files,
        num_annotations=dataset.num_annotations,
        num_classes=dataset.num_classes,
        class_names=tuple(dataset.class_names or ()),
        splits=dict(dataset.splits or {}),
        modalities=tuple(dataset.modalities or ()),
        has_bounding_boxes=dataset.has_bounding_boxes,
        has_masks=dataset.has_masks,
        has_keypoints=dataset.has_keypoints,
        has_lidar=dataset.has_lidar,
        has_radar=dataset.has_radar,
        has_depth=dataset.has_depth,
        has_thermal=dataset.has_thermal,
        repo_id=dataset.repo_id,
        license=dataset.license,
        description=dataset.description,
        extra=dict(dataset.extra or {}),
    )


class _Accumulator:
    """Folds one asset's file rows into a :class:`FileSummary`."""

    __slots__ = ("_by_extension", "_bytes_by_extension", "_directories", "_names",
                 "_relpaths", "_top_level", "_total", "_total_bytes")

    def __init__(self) -> None:
        self._total = 0
        self._total_bytes = 0
        self._names: set[str] = set()
        self._by_extension: dict[str, int] = {}
        self._bytes_by_extension: dict[str, int] = {}
        self._directories: set[str] = set()
        self._top_level: set[str] = set()
        self._relpaths: list[str] = []

    def add(self, relpath: str, extension: str | None, size_bytes: int) -> None:
        """Fold in one file row."""
        # Windows paths are stored with backslashes; normalising here means every plugin
        # can pattern-match on '/' without each one having to remember to.
        normalised = relpath.replace("\\", "/").lower()
        self._total += 1
        self._total_bytes += size_bytes
        self._relpaths.append(normalised)

        segments = normalised.split("/")
        self._names.add(segments[-1])
        if len(segments) > 1:
            self._directories.update(segments[:-1])
            self._top_level.add(segments[0])

        suffix = (extension or "").lower()
        if not suffix:
            _, dot, tail = segments[-1].rpartition(".")
            suffix = f".{tail}" if dot and tail else ""
        elif not suffix.startswith("."):
            suffix = f".{suffix}"

        if suffix:
            self._by_extension[suffix] = self._by_extension.get(suffix, 0) + 1
            self._bytes_by_extension[suffix] = (
                self._bytes_by_extension.get(suffix, 0) + size_bytes
            )

    def finish(self) -> FileSummary:
        """Return the completed summary."""
        return FileSummary(
            total=self._total,
            total_bytes=self._total_bytes,
            loaded=True,
            names=frozenset(self._names),
            by_extension=self._by_extension,
            bytes_by_extension=self._bytes_by_extension,
            directories=frozenset(self._directories),
            top_level=frozenset(self._top_level),
            relpaths=tuple(self._relpaths),
        )


def _chunked(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    """Yield fixed-size slices of a sequence."""
    for start in range(0, len(values), size):
        yield values[start : start + size]
