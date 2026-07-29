"""Tests for persistence, incremental rescanning and querying."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.metadata.merge import compute_physical_size
from ai_asset_manager.backend.models import Asset, AssetFile, HealthFinding, Tag
from ai_asset_manager.backend.models.enums import ScanStatus
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.scanner.types import FileEntry
from ai_asset_manager.backend.services.asset_service import AssetFilter, AssetService
from ai_asset_manager.backend.services.scan_service import ScanService
from tests import factories as F


@pytest.fixture
def service(session: Session, settings) -> ScanService:
    """Return a scan service bound to the in-memory database."""
    return ScanService(session, settings=settings, pipeline=ScanPipeline(settings=settings))


def entry(path: str, size: int, *, inode: int = 0, nlink: int = 1, symlink: bool = False):
    """Build a synthetic walked entry."""
    return FileEntry(
        name=os.path.basename(path), path=path, size=size, mtime=1.0, ctime=1.0, atime=1.0,
        inode=inode, device=1 if inode else 0, nlink=nlink, is_symlink=symlink,
    )


class TestPhysicalSize:
    def test_counts_plain_files_in_full(self) -> None:
        files = [entry("/a", 100), entry("/b", 200)]

        assert compute_physical_size(files) == 300

    def test_counts_a_hardlinked_extent_once(self) -> None:
        # Two names for one physical file. Counting both would inflate every total and,
        # worse, present the second as reclaimable space that frees nothing.
        files = [entry("/a", 100, inode=42, nlink=2), entry("/b", 100, inode=42, nlink=2)]

        assert compute_physical_size(files) == 100

    def test_distinct_inodes_are_both_counted(self) -> None:
        files = [entry("/a", 100, inode=1, nlink=2), entry("/b", 100, inode=2, nlink=2)]

        assert compute_physical_size(files) == 200

    def test_symlinks_are_excluded(self) -> None:
        files = [entry("/real", 100), entry("/link", 100, symlink=True)]

        assert compute_physical_size(files) == 100


class TestScanPersistence:
    def test_first_scan_creates_assets(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")
        F.make_coco_dataset(tmp_path, "coco")

        run = service.scan([str(tmp_path)])

        assert run.status == ScanStatus.COMPLETED
        assert run.assets_created == 2
        assert run.assets_found == 2
        assert service.session.query(Asset).count() == 2

    def test_details_rows_are_written(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")

        service.scan([str(tmp_path)])
        asset = service.session.query(Asset).one()

        assert asset.model_details is not None
        assert asset.model_details.architecture == "Qwen2ForCausalLM"
        assert asset.dataset_details is None
        assert len(asset.files) == asset.file_count

    def test_rescan_skips_unchanged_assets(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])

        second = service.scan([str(tmp_path)])

        assert second.assets_unchanged == 1
        assert second.assets_created == 0
        assert second.assets_updated == 0

    def test_rescan_reparses_a_modified_asset(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        directory = F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])

        (directory / "extra.json").write_text('{"added": true}', encoding="utf-8")
        second = service.scan([str(tmp_path)])

        assert second.assets_unchanged == 0
        assert second.assets_updated == 1

    def test_full_scan_ignores_fingerprints(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])

        second = service.scan([str(tmp_path)], incremental=False)

        assert second.assets_unchanged == 0
        assert second.assets_updated == 1

    def test_asset_identity_survives_a_rescan(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])
        original_id = service.session.query(Asset).one().id

        service.scan([str(tmp_path)], incremental=False)

        assert service.session.query(Asset).one().id == original_id

    def test_user_tags_survive_a_rescan(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])

        asset = service.session.query(Asset).one()
        favourite = service.session.query(Tag).filter_by(name="Favorite").one()
        asset.tags.append(favourite)
        service.session.commit()

        service.scan([str(tmp_path)], incremental=False)

        assert [tag.name for tag in service.session.query(Asset).one().tags] == ["Favorite"]

    def test_computed_hashes_survive_a_rescan(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.scan([str(tmp_path)])

        # Hashing is lazy and expensive; a rescan must not throw the results away.
        record = service.session.query(AssetFile).filter_by(relpath="config.json").one()
        record.sha256 = "deadbeef" * 8
        service.session.commit()

        service.scan([str(tmp_path)], incremental=False)
        refreshed = service.session.query(AssetFile).filter_by(relpath="config.json").one()

        assert refreshed.sha256 == "deadbeef" * 8

    def test_deleted_assets_are_marked_not_removed(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_hf_model(tmp_path, "keep")
        removable = F.make_hf_model(tmp_path, "remove")
        service.scan([str(tmp_path)])

        shutil.rmtree(removable)
        run = service.scan([str(tmp_path)])

        assert run.assets_missing == 1
        # Marked, not deleted: an unplugged drive must not destroy its own catalogue.
        assert service.session.query(Asset).count() == 2
        assert service.session.query(Asset).filter_by(is_missing=True).one().name == "remove"

    def test_parser_warnings_become_health_findings(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_truncated_model(tmp_path, "truncated")

        service.scan([str(tmp_path)])

        findings = service.session.query(HealthFinding).all()
        assert findings
        assert any("truncated" in finding.message for finding in findings)

    def test_warnings_are_not_duplicated_across_rescans(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_truncated_model(tmp_path, "truncated")
        service.scan([str(tmp_path)])
        first = service.session.query(HealthFinding).count()

        service.scan([str(tmp_path)], incremental=False)

        assert service.session.query(HealthFinding).count() == first

    def test_unreachable_root_is_recorded_not_fatal(
        self, service: ScanService, tmp_path: Path
    ) -> None:
        F.make_hf_model(tmp_path, "model-a")

        run = service.scan([str(tmp_path), str(tmp_path / "does-not-exist")])

        assert run.status == ScanStatus.COMPLETED
        assert run.assets_found == 1
        assert run.error_count == 1

    def test_scan_with_no_roots_completes_cleanly(self, service: ScanService) -> None:
        run = service.scan([])

        assert run.status == ScanStatus.COMPLETED
        assert run.assets_found == 0


class TestScanRoots:
    def test_add_is_idempotent(self, service: ScanService, tmp_path: Path) -> None:
        service.add_root(str(tmp_path))
        service.add_root(str(tmp_path))

        assert len(service.list_roots()) == 1

    def test_remove(self, service: ScanService, tmp_path: Path) -> None:
        service.add_root(str(tmp_path))

        assert service.remove_root(str(tmp_path))
        assert not service.remove_root(str(tmp_path))

    def test_scan_uses_registered_roots(self, service: ScanService, tmp_path: Path) -> None:
        F.make_hf_model(tmp_path, "model-a")
        service.add_root(str(tmp_path))

        run = service.scan()

        assert run.assets_found == 1
        assert service.list_roots()[0].last_asset_count == 1


class TestAssetQueries:
    @pytest.fixture
    def populated(self, service: ScanService, tmp_path: Path) -> AssetService:
        """Return a query service over a small catalogued tree."""
        F.make_hf_model(tmp_path, "text-model")
        F.make_peft_adapter(tmp_path, "an-adapter")
        F.make_coco_dataset(tmp_path, "coco")
        service.scan([str(tmp_path)])
        return AssetService(service.session)

    def test_lists_everything_by_default(self, populated: AssetService) -> None:
        assert populated.list().total == 3

    def test_filters_by_kind(self, populated: AssetService) -> None:
        page = populated.list(AssetFilter(kinds=["dataset"]))

        assert page.total == 1
        assert page.items[0].kind == "dataset"

    def test_filters_by_model_type(self, populated: AssetService) -> None:
        page = populated.list(AssetFilter(model_types=["lora"]))

        assert page.total == 1
        assert page.items[0].name == "an-adapter"

    def test_free_text_matches_name(self, populated: AssetService) -> None:
        assert populated.list(AssetFilter(text="coco")).total == 1

    def test_free_text_matches_architecture(self, populated: AssetService) -> None:
        assert populated.list(AssetFilter(text="Qwen2ForCausalLM")).total == 1

    def test_size_filter(self, populated: AssetService) -> None:
        assert populated.list(AssetFilter(min_size=10**9)).total == 0

    def test_paging_is_stable(self, populated: AssetService) -> None:
        first = populated.list(limit=2, offset=0)
        second = populated.list(limit=2, offset=2)

        assert first.has_more
        assert not second.has_more
        assert {item.id for item in first.items}.isdisjoint({item.id for item in second.items})

    def test_counts_and_totals(self, populated: AssetService) -> None:
        counts = populated.counts_by_kind()

        assert counts["dataset"] == 1
        assert counts["adapter"] == 1
        assert populated.total_size() > 0
        assert populated.file_count() > 0

    def test_missing_assets_are_hidden_by_default(
        self, populated: AssetService, session: Session
    ) -> None:
        asset = session.query(Asset).first()
        asset.is_missing = True
        session.commit()

        assert populated.list().total == 2
        assert populated.list(AssetFilter(include_missing=True)).total == 3
