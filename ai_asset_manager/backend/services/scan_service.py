"""Scan orchestration and persistence.

The one place a scan touches the database. The pipeline runs without a session, and its
records are written here in a single transaction per root, which keeps SQLite's writer
lock held for as short a time as possible while the API keeps reading.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ai_asset_manager.backend.database.base import utcnow
from ai_asset_manager.backend.metadata.records import AssetRecord
from ai_asset_manager.backend.models import (
    Asset,
    AssetFile,
    DatasetDetails,
    HealthFinding,
    ModelDetails,
    ScanRoot,
    ScanRun,
)
from ai_asset_manager.backend.models.enums import ScanStatus, Severity
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.scanner.progress import ScanCancelled, ScanContext, ScanPhase
from ai_asset_manager.backend.scanner.walker import WalkError
from ai_asset_manager.backend.utils.paths import normalize_path
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Errors retained on a scan run for display. The rest go to the log only.
MAX_STORED_ERRORS = 50


class ScanService:
    """Runs scans and persists their results."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        pipeline: ScanPipeline | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            session: Open database session; the caller owns its lifecycle.
            settings: Configuration; defaults to the singleton.
            pipeline: Scan pipeline; constructed from settings when omitted.
        """
        self.session = session
        self.settings = settings or get_settings()
        self.pipeline = pipeline or ScanPipeline(settings=self.settings)

    # -- roots --------------------------------------------------------------

    def add_root(self, path: str, *, label: str | None = None) -> ScanRoot:
        """Register a folder for scanning, or return the existing registration."""
        normalized = normalize_path(path)
        existing = self.session.scalar(select(ScanRoot).where(ScanRoot.path == normalized))
        if existing is not None:
            return existing

        root = ScanRoot(path=normalized, label=label or os.path.basename(normalized) or normalized)
        self.session.add(root)
        self.session.flush()
        logger.info("Registered scan root %s", normalized)
        return root

    def remove_root(self, path: str) -> bool:
        """Unregister a scan root. Catalogued assets are left in place."""
        normalized = normalize_path(path)
        root = self.session.scalar(select(ScanRoot).where(ScanRoot.path == normalized))
        if root is None:
            return False
        self.session.delete(root)
        return True

    def list_roots(self, *, enabled_only: bool = False) -> Sequence[ScanRoot]:
        """Return registered scan roots."""
        statement = select(ScanRoot).order_by(ScanRoot.path)
        if enabled_only:
            statement = statement.where(ScanRoot.enabled.is_(True))
        return self.session.scalars(statement).all()

    # -- scanning -----------------------------------------------------------

    def scan(
        self,
        roots: Sequence[str] | None = None,
        *,
        context: ScanContext | None = None,
        incremental: bool = True,
        prune_missing: bool = True,
    ) -> ScanRun:
        """Scan the given roots, or every enabled registered root.

        Args:
            roots: Paths to scan. When omitted, the enabled registered roots are used.
            context: Progress and cancellation context.
            incremental: Skip parsing assets whose fingerprint is unchanged.
            prune_missing: Flag catalogued assets under the scanned roots that no longer
                exist on disk. They are marked, not deleted, so that a temporarily
                unplugged drive does not destroy its own catalogue.

        Returns:
            The completed :class:`ScanRun`.
        """
        targets = [normalize_path(path) for path in roots] if roots else [
            root.path for root in self.list_roots(enabled_only=True)
        ]

        run = ScanRun(roots=list(targets), status=ScanStatus.RUNNING)
        self.session.add(run)
        self.session.commit()

        if not targets:
            run.status = ScanStatus.COMPLETED
            run.message = "No scan roots configured"
            run.finished_at = utcnow()
            run.duration_seconds = 0.0
            self.session.commit()
            return run

        started = time.monotonic()
        seen_paths: set[str] = set()
        errors: list[dict[str, str]] = []

        try:
            for target in targets:
                logger.info("Scanning %s", target)
                try:
                    records = self.pipeline.scan_root(
                        target,
                        context=context,
                        fingerprint_lookup=self._fingerprint_lookup if incremental else None,
                    )
                except WalkError as exc:
                    logger.warning("Cannot scan %s: %s", target, exc)
                    errors.append({"root": target, "error": str(exc)})
                    continue

                if context is not None:
                    context.set_phase(
                        ScanPhase.PERSISTING, total=len(records), message=f"Saving {target}"
                    )

                self._persist_records(records, run, context=context)
                seen_paths.update(record.root_path for record in records)
                self._touch_root(target, len(records))
                self.session.commit()

            if prune_missing:
                run.assets_missing = self._mark_missing(targets, seen_paths)

            run.status = ScanStatus.COMPLETED

        except ScanCancelled:
            run.status = ScanStatus.CANCELLED
            run.message = "Cancelled by user"
            logger.info("Scan %d cancelled", run.id)
        except Exception as exc:
            run.status = ScanStatus.FAILED
            run.message = str(exc)
            logger.exception("Scan %d failed", run.id)
        finally:
            run.finished_at = utcnow()
            run.duration_seconds = round(time.monotonic() - started, 3)
            run.error_count = len(errors)
            run.errors = errors[:MAX_STORED_ERRORS]
            if context is not None:
                snapshot = context.snapshot()
                run.files_seen = snapshot.files_seen
                run.bytes_seen = snapshot.bytes_seen
                run.directories_seen = snapshot.directories_seen
            self.session.commit()

        logger.info(
            "Scan %d %s in %.1fs: %d found, %d new, %d updated, %d unchanged, %d missing",
            run.id, run.status, run.duration_seconds or 0.0, run.assets_found,
            run.assets_created, run.assets_updated, run.assets_unchanged, run.assets_missing,
        )
        return run

    # -- persistence --------------------------------------------------------

    def _fingerprint_lookup(self, root_path: str) -> str | None:
        """Return the stored fingerprint for an asset, if it is already catalogued."""
        return self.session.scalar(
            select(Asset.fingerprint).where(Asset.root_path == root_path)
        )

    def _persist_records(
        self,
        records: list[AssetRecord],
        run: ScanRun,
        *,
        context: ScanContext | None = None,
    ) -> None:
        """Upsert a batch of records."""
        for index, record in enumerate(records):
            if context is not None:
                context.update(completed=index)

            existing = self.session.scalar(
                select(Asset).where(Asset.root_path == record.root_path)
            )

            if record.evidence.get("unchanged") and existing is not None:
                existing.last_scanned = utcnow()
                existing.is_missing = False
                run.assets_unchanged += 1
                run.assets_found += 1
                continue

            if existing is None:
                # Every NOT NULL column is populated here, before the row is added.
                # `_apply_record` issues queries, and any of them can trigger an autoflush
                # of a half-built entity — which fails on `assets.name` rather than
                # waiting politely for the rest of the fields to arrive.
                asset = Asset(
                    root_path=record.root_path,
                    name=record.name,
                    kind=record.kind.value,
                    first_seen=utcnow(),
                )
                self.session.add(asset)
                run.assets_created += 1
            else:
                asset = existing
                run.assets_updated += 1

            self._apply_record(asset, record)
            run.assets_found += 1

    def _apply_record(self, asset: Asset, record: AssetRecord) -> None:
        """Copy a record onto an ORM entity, replacing its children."""
        asset.kind = record.kind.value
        asset.subkind = record.subkind
        asset.name = record.name
        asset.display_name = record.display_name
        asset.drive = record.drive
        asset.is_single_file = record.is_single_file
        asset.size_bytes = record.size_bytes
        asset.physical_size_bytes = record.physical_size_bytes
        asset.file_count = record.file_count
        asset.format = record.format.value
        asset.framework = record.framework.value
        asset.created_at = record.created_at
        asset.modified_at = record.modified_at
        asset.accessed_at = record.accessed_at
        asset.fingerprint = record.fingerprint
        asset.detector = record.detector
        asset.detection_confidence = record.detection_confidence
        asset.evidence = record.evidence
        asset.last_scanned = utcnow()
        asset.is_missing = False

        self.session.flush()  # assign asset.id for the child rows below

        self._replace_files(asset, record)
        self._replace_details(asset, record)
        self._replace_parse_warnings(asset, record)

    def _replace_files(self, asset: Asset, record: AssetRecord) -> None:
        """Replace an asset's file rows.

        Delete-then-insert rather than a diff: an asset's file list is small relative to
        the catalogue, and reconciling it row by row would cost more than rewriting it.
        Hashes are carried across by path so that lazily computed digests survive a
        rescan and do not have to be recomputed.
        """
        previous_hashes: dict[str, tuple[str | None, str | None]] = {
            row.relpath: (row.quick_sig, row.sha256)
            for row in self.session.scalars(
                select(AssetFile).where(AssetFile.asset_id == asset.id)
            )
        }
        self.session.execute(delete(AssetFile).where(AssetFile.asset_id == asset.id))

        for file_record in record.files:
            quick_sig, sha256 = previous_hashes.get(file_record.relpath, (None, None))
            self.session.add(
                AssetFile(
                    asset_id=asset.id,
                    relpath=file_record.relpath,
                    extension=file_record.extension,
                    size_bytes=file_record.size,
                    modified_at=_epoch_to_datetime(file_record.mtime),
                    quick_sig=quick_sig,
                    sha256=sha256,
                    inode=file_record.inode,
                    device=file_record.device,
                    nlink=file_record.nlink,
                    is_symlink=file_record.is_symlink,
                    is_payload=file_record.is_payload,
                )
            )

    def _replace_details(self, asset: Asset, record: AssetRecord) -> None:
        """Replace an asset's model or dataset detail row."""
        self.session.execute(delete(ModelDetails).where(ModelDetails.asset_id == asset.id))
        self.session.execute(delete(DatasetDetails).where(DatasetDetails.asset_id == asset.id))

        if record.model is not None:
            model_facts = record.model
            self.session.add(
                ModelDetails(
                    asset_id=asset.id,
                    model_type=model_facts.model_type.value,
                    architecture=model_facts.architecture,
                    param_count=model_facts.param_count,
                    param_count_is_exact=model_facts.param_count_is_exact,
                    quantization=model_facts.quantization,
                    precision=model_facts.precision.value,
                    context_length=model_facts.context_length,
                    hidden_size=model_facts.hidden_size,
                    num_layers=model_facts.num_layers,
                    vocab_size=model_facts.vocab_size,
                    tensor_count=model_facts.tensor_count,
                    repo_id=model_facts.repo_id,
                    revision=model_facts.revision,
                    author=model_facts.author,
                    license=model_facts.license,
                    description=model_facts.description,
                    base_model=model_facts.base_model,
                    pipeline_tag=model_facts.pipeline_tag,
                    library_name=model_facts.library_name,
                    card_tags=model_facts.card_tags,
                    extra=model_facts.extra,
                )
            )
        elif record.dataset is not None:
            dataset_facts = record.dataset
            self.session.add(
                DatasetDetails(
                    asset_id=asset.id,
                    dataset_format=dataset_facts.dataset_format.value,
                    task=dataset_facts.task,
                    num_images=dataset_facts.num_images,
                    num_videos=dataset_facts.num_videos,
                    num_audio_files=dataset_facts.num_audio_files,
                    num_text_files=dataset_facts.num_text_files,
                    num_annotations=dataset_facts.num_annotations,
                    num_classes=dataset_facts.num_classes,
                    class_names=dataset_facts.class_names,
                    splits=dataset_facts.splits,
                    modalities=dataset_facts.modalities,
                    has_bounding_boxes=dataset_facts.has_bounding_boxes,
                    has_masks=dataset_facts.has_masks,
                    has_keypoints=dataset_facts.has_keypoints,
                    has_lidar=dataset_facts.has_lidar,
                    has_radar=dataset_facts.has_radar,
                    has_depth=dataset_facts.has_depth,
                    has_thermal=dataset_facts.has_thermal,
                    repo_id=dataset_facts.repo_id,
                    license=dataset_facts.license,
                    description=dataset_facts.description,
                    extra=dataset_facts.extra,
                )
            )

    def _replace_parse_warnings(self, asset: Asset, record: AssetRecord) -> None:
        """Record parser warnings as health findings.

        Only warnings raised by parsing are written here; the rule-based health checker
        owns its own codes and replaces its own findings independently.
        """
        self.session.execute(
            delete(HealthFinding).where(
                HealthFinding.asset_id == asset.id,
                HealthFinding.code == "parser.warning",
            )
        )
        for message in record.warnings:
            self.session.add(
                HealthFinding(
                    asset_id=asset.id,
                    code="parser.warning",
                    severity=Severity.WARNING.value,
                    message=message,
                    fix_hint="Re-download the asset if the file is truncated or corrupt.",
                )
            )

    def _touch_root(self, path: str, asset_count: int) -> None:
        """Update a registered root's last-scanned bookkeeping."""
        root = self.session.scalar(select(ScanRoot).where(ScanRoot.path == path))
        if root is not None:
            root.last_scanned = utcnow()
            root.last_asset_count = asset_count

    def _mark_missing(self, roots: Sequence[str], seen: set[str]) -> int:
        """Flag catalogued assets under the scanned roots that were not found.

        Marked rather than deleted: an asset that vanished because its drive was
        unplugged should reappear intact on the next scan, with its tags still attached.
        """
        missing = 0
        for asset in self.session.scalars(select(Asset).where(Asset.is_missing.is_(False))):
            if asset.root_path in seen:
                continue
            if not any(_is_within(asset.root_path, root) for root in roots):
                continue
            if os.path.exists(asset.root_path):
                continue
            asset.is_missing = True
            missing += 1

        if missing:
            logger.info("Flagged %d asset(s) as missing", missing)
        return missing


def _is_within(path: str, root: str) -> bool:
    r"""Report whether ``path`` lies at or beneath ``root``.

    Compared case-insensitively on Windows, where ``D:\\Models`` and ``d:\\models`` are
    the same directory.
    """
    normalized_path = os.path.normcase(os.path.normpath(path))
    normalized_root = os.path.normcase(os.path.normpath(root))
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root + os.sep
    )


def _epoch_to_datetime(timestamp: float) -> datetime | None:
    """Convert a POSIX timestamp to an aware UTC datetime, tolerating bad values."""
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
