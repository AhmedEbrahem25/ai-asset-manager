"""Tests for the Inventory Engine."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.inventory import (
    InventoryCategory,
    InventoryEngine,
    InventorySection,
    classify_dataset,
    classify_model,
    export_report,
    known_aliases,
    resolve_alias,
)
from ai_asset_manager.backend.inventory.categories import (
    DATASET_FORMAT_CATEGORIES,
    MODEL_TYPE_CATEGORIES,
    classify_asset,
)
from ai_asset_manager.backend.models.enums import DatasetFormat, ModelType
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.services.scan_service import ScanService
from tests import factories as F


@pytest.fixture
def catalogued(session: Session, settings, tmp_path: Path) -> tuple[Session, Path]:
    """Scan a varied tree and return the session plus the scanned directory."""
    assets = tmp_path / "library"
    assets.mkdir()

    F.make_hf_model(assets, "text-model", architecture="Qwen2ForCausalLM")
    F.make_hf_model(assets, "encoder", architecture="BertForMaskedLM", model_type="bert")
    F.make_peft_adapter(assets, "my-lora")
    F.make_diffusers_pipeline(assets, "sd-pipeline")
    F.make_coco_dataset(assets, "coco-mini")
    F.make_yolo_dataset(assets, "yolo-mini")
    F.make_imagefolder_dataset(assets, "flowers")
    F.write_yolo_checkpoint(assets / "weights" / "yolov8n.pt", storage_bytes=2 * 1024 * 1024)

    service = ScanService(session, settings=settings, pipeline=ScanPipeline(settings=settings))
    service.scan([str(assets)])
    return session, assets


class TestModelClassification:
    @pytest.mark.parametrize("model_type", list(ModelType))
    def test_every_model_type_maps_to_a_category(self, model_type: ModelType) -> None:
        category = classify_model(model_type.value)

        assert isinstance(category, InventoryCategory)
        assert category is MODEL_TYPE_CATEGORIES[model_type]

    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("llm", InventoryCategory.LLM),
            ("ocr", InventoryCategory.OCR),
            ("vision_language", InventoryCategory.VISION_LANGUAGE),
            ("multimodal", InventoryCategory.VISION_LANGUAGE),
            ("speech_recognition", InventoryCategory.SPEECH),
            ("text_to_speech", InventoryCategory.TEXT_TO_SPEECH),
            ("image_generation", InventoryCategory.DIFFUSION),
            ("lora", InventoryCategory.ADAPTER),
            ("embedding", InventoryCategory.EMBEDDING),
        ],
    )
    def test_representative_mappings(self, model_type: str, expected: InventoryCategory) -> None:
        assert classify_model(model_type) is expected

    def test_tracker_beats_the_detector_mapping(self) -> None:
        # Trackers are detectors plus association, so the stored model type says
        # "object detection"; only the name distinguishes them.
        assert classify_model("object_detection", name="bytetrack_x_mot17") is (
            InventoryCategory.TRACKING
        )

    def test_unknown_type_falls_back_to_the_name(self) -> None:
        assert classify_model(None, name="surya_ocr_rec") is InventoryCategory.OCR
        assert classify_model(None, name="whisper-large-v3") is InventoryCategory.SPEECH
        assert classify_model(None, name="yolov8n") is InventoryCategory.OBJECT_DETECTION

    def test_torchvision_backbones_are_classification(self) -> None:
        # These ship as bare .pth files with no config; the family name is all there is.
        for name in ("resnet18-f37072fd", "mobilenet_v3_small", "efficientnet_b0"):
            assert classify_model(None, name=name) is InventoryCategory.CLASSIFICATION

    def test_completely_unidentifiable_model(self) -> None:
        assert classify_model(None, name="checkpoint_final") is InventoryCategory.OTHER_MODEL


class TestDatasetClassification:
    @pytest.mark.parametrize("dataset_format", list(DatasetFormat))
    def test_every_dataset_format_maps_to_a_category(
        self, dataset_format: DatasetFormat
    ) -> None:
        category = classify_dataset(dataset_format.value)

        assert isinstance(category, InventoryCategory)
        assert category in DATASET_FORMAT_CATEGORIES.values() or category in (
            InventoryCategory.OTHER_DATASET,
            InventoryCategory.NLP_DATASET,
        )

    @pytest.mark.parametrize(
        ("dataset_format", "expected"),
        [
            ("coco", InventoryCategory.DETECTION_DATASET),
            ("yolo", InventoryCategory.DETECTION_DATASET),
            ("kitti", InventoryCategory.DETECTION_DATASET),
            ("cityscapes", InventoryCategory.SEGMENTATION_DATASET),
            ("mot", InventoryCategory.TRACKING_DATASET),
            ("imagenet", InventoryCategory.IMAGE_DATASET),
            ("video", InventoryCategory.VIDEO_DATASET),
            ("audio", InventoryCategory.AUDIO_DATASET),
            ("nlp", InventoryCategory.NLP_DATASET),
        ],
    )
    def test_representative_mappings(
        self, dataset_format: str, expected: InventoryCategory
    ) -> None:
        assert classify_dataset(dataset_format) is expected

    def test_ocr_task_overrides_the_layout(self) -> None:
        # An OCR corpus is usually COCO-shaped or a plain image folder; the task is what
        # makes it an OCR dataset, not the directory structure.
        assert classify_dataset("coco", name="TextOCR-GT") is InventoryCategory.OCR_DATASET
        assert classify_dataset("image_classification", task="ocr") is (
            InventoryCategory.OCR_DATASET
        )

    def test_generic_containers_classify_by_contents(self) -> None:
        assert classify_dataset("hf_dataset", num_audio_files=5000) is (
            InventoryCategory.AUDIO_DATASET
        )
        assert classify_dataset("custom", num_videos=800) is InventoryCategory.VIDEO_DATASET
        assert classify_dataset("custom", num_images=9000) is InventoryCategory.IMAGE_DATASET
        assert classify_dataset("hf_dataset") is InventoryCategory.NLP_DATASET

    def test_adapters_are_categorised_by_kind(self) -> None:
        assert classify_asset("adapter", name="my-lora") is InventoryCategory.ADAPTER


class TestAliases:
    @pytest.mark.parametrize("alias", ["llm", "ocr", "datasets", "models", "vision", "speech",
                                       "embeddings", "adapters", "detection", "tracking"])
    def test_documented_aliases_resolve(self, alias: str) -> None:
        resolved = resolve_alias(alias)

        assert resolved is not None
        assert all(isinstance(item, InventoryCategory) for item in resolved)

    def test_bare_category_values_resolve(self) -> None:
        assert resolve_alias("object_detection") == (InventoryCategory.OBJECT_DETECTION,)
        assert resolve_alias("object-detection") == (InventoryCategory.OBJECT_DETECTION,)

    def test_unknown_alias_returns_none(self) -> None:
        # Returning None rather than an empty tuple lets the CLI distinguish "no such
        # category" from "that category is empty".
        assert resolve_alias("banana") is None

    def test_every_advertised_alias_is_resolvable(self) -> None:
        for alias in known_aliases():
            assert resolve_alias(alias) is not None

    def test_vision_alias_spans_the_vision_family(self) -> None:
        resolved = resolve_alias("vision")

        assert resolved is not None
        assert InventoryCategory.OBJECT_DETECTION in resolved
        assert InventoryCategory.SEGMENTATION in resolved
        assert InventoryCategory.LLM not in resolved


class TestEngine:
    def test_builds_a_report_of_everything(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()

        assert not report.is_empty
        assert report.summary.total_assets == len(report.items)
        assert report.summary.total_bytes > 0
        assert report.summary.by_category

    def test_categorises_the_scanned_library(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()
        found = {item.category for item in report.items}

        assert InventoryCategory.LLM in found
        assert InventoryCategory.ADAPTER in found
        assert InventoryCategory.DIFFUSION in found
        assert InventoryCategory.DETECTION_DATASET in found

    def test_filters_to_one_category(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build([InventoryCategory.LLM])

        assert report.items
        assert all(item.category is InventoryCategory.LLM for item in report.items)

    def test_sections_are_assigned(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()

        for item in report.items:
            if item.category is InventoryCategory.DETECTION_DATASET:
                assert item.section is InventorySection.DATASETS
            if item.category is InventoryCategory.LLM:
                assert item.section is InventorySection.MODELS

    def test_grouping_partitions_without_loss(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(group_by="category")

        assert report.groups
        assert sum(group.count for group in report.groups) == len(report.items)

    @pytest.mark.parametrize("group_by", ["category", "framework", "drive", "format", "section"])
    def test_every_grouping_field_works(self, catalogued, group_by: str) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(group_by=group_by)

        assert report.groups
        assert sum(group.count for group in report.groups) == len(report.items)

    def test_sort_by_size_descending(self, catalogued) -> None:
        session, _ = catalogued

        items = InventoryEngine(session).build(sort="size").items
        sizes = [item.size_bytes for item in items]

        assert sizes == sorted(sizes, reverse=True)

    def test_sort_by_name_is_alphabetical(self, catalogued) -> None:
        session, _ = catalogued

        items = InventoryEngine(session).build(sort="name").items
        names = [item.name.lower() for item in items]

        assert names == sorted(names)

    def test_limit_truncates_items_but_not_the_summary(self, catalogued) -> None:
        session, _ = catalogued
        engine = InventoryEngine(session)

        full = engine.build()
        limited = engine.build(limit=2)

        assert len(limited.items) == 2
        # A truncated table must never misreport the size of the library.
        assert limited.summary.total_assets == full.summary.total_assets
        assert limited.summary.total_bytes == full.summary.total_bytes

    def test_drive_filter(self, catalogued) -> None:
        session, _ = catalogued
        engine = InventoryEngine(session)

        drive = engine.build().items[0].drive
        assert drive is not None

        assert engine.build(drives=[drive]).items
        assert not engine.build(drives=["ZZ:"]).items

    def test_locate_finds_assets_by_name(self, catalogued) -> None:
        session, _ = catalogued

        matches = InventoryEngine(session).locate("coco")

        assert matches
        assert any("coco" in item.name.lower() for item in matches)

    def test_locate_with_no_match(self, catalogued) -> None:
        session, _ = catalogued

        assert InventoryEngine(session).locate("nothing-like-this") == []

    def test_empty_catalogue_yields_an_empty_report(self, session: Session) -> None:
        report = InventoryEngine(session).build()

        assert report.is_empty
        assert report.summary.total_assets == 0


class TestReadOnly:
    def test_inventory_never_touches_the_filesystem(self, catalogued) -> None:
        """The engine must answer from the database alone.

        Proven by deleting every scanned file first: a report that still comes back
        complete cannot have read anything from disk.
        """
        session, assets = catalogued
        before = InventoryEngine(session).build()
        assert before.summary.total_assets > 0

        shutil.rmtree(assets)
        assert not assets.exists()

        after = InventoryEngine(session).build()

        assert after.summary.total_assets == before.summary.total_assets
        assert after.summary.total_bytes == before.summary.total_bytes
        assert [item.path for item in after.items] == [item.path for item in before.items]

    def test_building_a_report_leaves_the_catalogue_unchanged(self, catalogued) -> None:
        session, _ = catalogued
        from ai_asset_manager.backend.models import Asset

        before = {
            (asset.id, asset.name, asset.size_bytes)
            for asset in session.query(Asset).all()
        }

        InventoryEngine(session).build(group_by="category", sort="name")

        after = {
            (asset.id, asset.name, asset.size_bytes)
            for asset in session.query(Asset).all()
        }
        assert before == after


class TestExport:
    def test_csv_has_a_row_per_asset(self, catalogued, tmp_path: Path) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        rendered = export_report(report, "csv", tmp_path / "out.csv")
        lines = [line for line in rendered.splitlines() if line.strip()]

        assert len(lines) == len(report.items) + 1  # header
        assert "size_bytes" in lines[0]
        assert (tmp_path / "out.csv").exists()

    def test_csv_carries_both_raw_and_human_sizes(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        rendered = export_report(report, "csv")

        # Raw bytes so a spreadsheet can total them, human sizes so a person can skim.
        assert "size_bytes" in rendered
        assert "size_human" in rendered

    def test_json_round_trips(self, catalogued, tmp_path: Path) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        export_report(report, "json", tmp_path / "out.json")
        loaded = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

        assert loaded["summary"]["total_assets"] == report.summary.total_assets
        assert len(loaded["items"]) == len(report.items)
        assert "generated_at" in loaded

    def test_markdown_contains_a_table_and_summary(self, catalogued, tmp_path: Path) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        rendered = export_report(report, "markdown", tmp_path / "out.md")

        assert "# AI Asset Inventory" in rendered
        assert "## Summary" in rendered
        assert "| Name | Category |" in rendered

    def test_markdown_escapes_pipes_in_names(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()
        report.items[0].name = "weird|name"

        rendered = export_report(report, "markdown")

        assert r"weird\|name" in rendered

    def test_grouped_markdown_has_a_section_per_group(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build(group_by="category")

        rendered = export_report(report, "markdown")

        for group in report.groups:
            assert f"## {group.label}" in rendered

    def test_unknown_format_is_rejected(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        with pytest.raises(ValueError, match="Unknown export format"):
            export_report(report, "pdf")
