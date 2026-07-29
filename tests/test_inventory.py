"""Tests for the Inventory Engine.

The engine's own job is narrow: fetch rows, hand them to the taxonomy, shape the answers.
What it classifies things *as* is tested in :mod:`tests.test_taxonomy`; what is tested here
is that the shaping is correct and that the read-only guarantee holds.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.inventory import (
    InventoryEngine,
    build_tree,
    export_report,
    flatten,
    known_aliases,
    resolve_alias,
    section_of,
)
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


class TestAliases:
    @pytest.mark.parametrize("alias", ["llm", "ocr", "datasets", "models", "vision", "speech",
                                       "embeddings", "adapters", "detection", "tracking",
                                       "experiments", "medical", "all"])
    def test_documented_aliases_resolve(self, alias: str) -> None:
        resolved = resolve_alias(alias)

        assert resolved
        assert all(isinstance(item, str) for item in resolved)

    def test_bare_category_ids_resolve(self) -> None:
        assert resolve_alias("object_detection") == ("object_detection",)
        assert resolve_alias("object-detection") == ("object_detection",)

    def test_unknown_alias_returns_none(self) -> None:
        # Returning None rather than an empty tuple lets the CLI distinguish "no such
        # category" from "that category is empty".
        assert resolve_alias("banana") is None

    def test_every_advertised_alias_is_resolvable(self) -> None:
        for alias in known_aliases():
            assert resolve_alias(alias) is not None, alias

    def test_vision_alias_spans_the_vision_family(self) -> None:
        resolved = resolve_alias("vision")

        assert resolved is not None
        assert "object_detection" in resolved
        assert "segmentation" in resolved
        assert "llm" not in resolved


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

        assert "llm" in found
        assert "adapter" in found
        assert "diffusion" in found
        assert "detection_dataset" in found

    def test_every_item_gets_a_task_and_a_domain(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()

        # "What is it for?" is the question this feature exists to answer; an item that
        # cannot answer it is a gap in the taxonomy, not an acceptable outcome.
        assert all(item.task for item in report.items)
        assert all(item.domain for item in report.items)
        assert all(item.task_label for item in report.items)

    def test_filters_to_one_category(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(["llm"])

        assert report.items
        assert all(item.category == "llm" for item in report.items)

    def test_filters_by_task_and_domain(self, catalogued) -> None:
        session, _ = catalogued
        engine = InventoryEngine(session)

        assert engine.build(tasks=["object_detection"]).items
        assert engine.build(domains=["vision"]).items
        assert not engine.build(tasks=["variant_calling"]).items

    def test_sections_are_assigned(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()

        for item in report.items:
            assert item.section == section_of(item.category)
            if item.category == "detection_dataset":
                assert item.section == "datasets"
                assert item.is_dataset
            if item.category == "llm":
                assert item.section == "models"
                assert item.is_model

    def test_grouping_partitions_without_loss(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(group_by="category")

        assert report.groups
        assert sum(group.count for group in report.groups) == len(report.items)

    @pytest.mark.parametrize(
        "group_by",
        ["category", "framework", "drive", "format", "section", "task", "domain",
         "family", "health"],
    )
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

    def test_sort_by_health_puts_the_worst_first(self, catalogued) -> None:
        session, _ = catalogued

        items = InventoryEngine(session).build(sort="health").items
        scores = [item.health_score for item in items if item.health_score is not None]

        # Worst first regardless of direction: a health listing exists to surface
        # problems, and burying them under the healthy assets defeats it.
        assert scores == sorted(scores)

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


class TestIntelligence:
    def test_datasets_carry_content_statistics(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(["detection_dataset"])

        assert report.items
        for item in report.items:
            assert item.stat("storage_format")
            assert item.stat("has_readme") is not None

    def test_models_carry_weight_statistics(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build(["llm"])

        assert report.items
        assert all(item.stat("weight_formats") for item in report.items)

    def test_health_is_scored_for_every_asset(self, catalogued) -> None:
        session, _ = catalogued

        report = InventoryEngine(session).build()

        for item in report.items:
            assert item.health is not None
            assert item.health.evaluated
            assert 0 <= item.health.score <= 100
        assert report.summary.average_health is not None

    def test_only_unhealthy_narrows_to_assets_needing_attention(self, catalogued) -> None:
        session, _ = catalogued
        engine = InventoryEngine(session)

        everything = engine.build()
        flagged = engine.build(only_unhealthy=True)

        assert len(flagged.items) <= len(everything.items)
        assert all(item.is_incomplete for item in flagged.items)

    def test_a_broken_download_is_reported(self, session: Session, settings,
                                           tmp_path: Path) -> None:
        assets = tmp_path / "lib"
        assets.mkdir()
        F.make_incomplete_download(assets, "half-model")

        ScanService(
            session, settings=settings, pipeline=ScanPipeline(settings=settings)
        ).scan([str(assets)])

        report = InventoryEngine(session).build(only_unhealthy=True)

        assert report.items
        assert any(
            item.health is not None
            and any("incomplete" in finding.code or "shard" in finding.code
                    or "weights" in finding.code
                    for finding in item.health.findings)
            for item in report.items
        )

    def test_classification_is_the_same_in_every_view(self, catalogued) -> None:
        """An asset must not change category depending on the command you ran.

        The file list is loaded for every build, not only detailed ones, precisely so
        that a listing and a tree cannot disagree.
        """
        session, _ = catalogued
        engine = InventoryEngine(session)

        plain = {item.asset_id: item.category for item in engine.build().items}
        limited = {item.asset_id: item.category for item in engine.build(limit=3).items}
        grouped = {
            item.asset_id: item.category
            for group in engine.build(group_by="task").groups
            for item in group.items
        }

        assert plain == grouped
        assert all(plain[key] == value for key, value in limited.items())


class TestTree:
    def test_tree_holds_every_asset_exactly_once(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        root = build_tree(report)

        assert root.count == len(report.items)
        assert len(flatten(root)) == len(report.items)
        assert root.total_bytes == report.summary.total_bytes

    def test_tree_nests_sections_then_categories(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        root = build_tree(report)

        assert {child.level for child in root.children} == {"section"}
        for section in root.children:
            assert all(child.level in ("category", "asset") for child in section.children)

    def test_custom_nesting(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        root = build_tree(report, levels=("domain", "task"))

        assert root.children
        assert {child.level for child in root.children} == {"domain"}
        assert len(flatten(root)) == len(report.items)

    def test_single_asset_family_rungs_are_collapsed(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        root = build_tree(report)

        # A family branch holding one asset says nothing the asset's own line does not.
        for section in root.children:
            for category in section.children:
                for node in category.children:
                    assert node.is_leaf or node.count > 1

    def test_empty_report_yields_a_bare_root(self, session: Session) -> None:
        root = build_tree(InventoryEngine(session).build())

        assert root.children == []
        assert root.count == 0


class TestReadOnly:
    def test_inventory_never_touches_the_filesystem(self, catalogued) -> None:
        """The engine must answer from the database alone.

        Proven by deleting every scanned file first: a report that still comes back
        complete, still classified and still health-scored cannot have read anything from
        disk.
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
        assert [item.category for item in after.items] == [
            item.category for item in before.items
        ]
        assert [item.health_score for item in after.items] == [
            item.health_score for item in before.items
        ]

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
        rows = list(csv.DictReader(io.StringIO(rendered)))

        assert len(rows) == len(report.items)
        assert (tmp_path / "out.csv").exists()

    def test_csv_carries_the_intelligence_columns(self, catalogued) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        rows = list(csv.DictReader(io.StringIO(export_report(report, "csv"))))

        # Raw bytes so a spreadsheet can total them, human sizes so a person can skim.
        assert {"size_bytes", "size_human", "task", "domain", "family",
                "health_score", "health_status"} <= set(rows[0])
        assert any(row["task"] for row in rows)
        assert all(row["health_score"] for row in rows)

    def test_json_round_trips(self, catalogued, tmp_path: Path) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        export_report(report, "json", tmp_path / "out.json")
        loaded = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))

        assert loaded["summary"]["total_assets"] == report.summary.total_assets
        assert len(loaded["items"]) == len(report.items)
        assert "generated_at" in loaded
        assert loaded["items"][0]["health"]["score"] >= 0
        assert "statistics" in loaded["items"][0]

    def test_markdown_contains_a_table_and_summary(self, catalogued, tmp_path: Path) -> None:
        session, _ = catalogued
        report = InventoryEngine(session).build()

        rendered = export_report(report, "markdown", tmp_path / "out.md")

        assert "# AI Asset Inventory" in rendered
        assert "## Summary" in rendered
        assert "| Name | Category | Task |" in rendered

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
