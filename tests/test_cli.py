"""CLI smoke tests.

Each invocation goes through the real Typer app against a real temporary database, so
argument wiring, the service calls and the rendering are all exercised together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_asset_manager.cli import app
from tests import factories as F


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer test runner with a fixed terminal width.

    Rich adapts its tables to the terminal, so without a pinned width these assertions
    would pass or fail depending on the window the suite happened to run in.
    """
    return CliRunner(env={"COLUMNS": "200", "TERM": "dumb"})


@pytest.fixture
def catalogue(tmp_path: Path, runner: CliRunner) -> tuple[Path, Path]:
    """Return ``(assets_dir, database_path)`` for an already-scanned tree."""
    assets = tmp_path / "assets"
    assets.mkdir()
    F.make_hf_model(assets, "text-model")
    F.make_coco_dataset(assets, "coco")
    F.make_peft_adapter(assets, "an-adapter")

    database = tmp_path / "catalog.db"
    result = runner.invoke(app, ["--database", str(database), "scan", str(assets), "-q"])
    assert result.exit_code == 0, result.output
    return assets, database


class TestScan:
    def test_reports_what_it_found(self, tmp_path: Path, runner: CliRunner) -> None:
        assets = tmp_path / "assets"
        assets.mkdir()
        F.make_hf_model(assets, "model-a")

        result = runner.invoke(
            app, ["--database", str(tmp_path / "db.sqlite"), "scan", str(assets), "-q"]
        )

        assert result.exit_code == 0
        assert "1 assets" in result.output
        assert "1 new" in result.output

    def test_rescan_reports_unchanged(self, catalogue, runner: CliRunner) -> None:
        assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "scan", str(assets), "-q"])

        assert result.exit_code == 0
        assert "3 unchanged" in result.output

    def test_refuses_to_run_with_no_roots(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "scan"])

        assert result.exit_code == 1

    def test_add_registers_roots(self, tmp_path: Path, runner: CliRunner) -> None:
        assets = tmp_path / "assets"
        assets.mkdir()
        F.make_hf_model(assets, "model-a")
        database = tmp_path / "db.sqlite"

        runner.invoke(app, ["--database", str(database), "scan", str(assets), "--add", "-q"])
        result = runner.invoke(app, ["--database", str(database), "roots", "list"])

        assert result.exit_code == 0
        assert "assets" in result.output


class TestList:
    def test_lists_assets(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "list"])

        assert result.exit_code == 0
        assert "text-model" in result.output
        assert "coco" in result.output

    def test_filters_by_kind(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "list", "--kind", "dataset"])

        assert result.exit_code == 0
        assert "coco" in result.output
        assert "text-model" not in result.output

    def test_search_matches(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "list", "-s", "adapter"])

        assert result.exit_code == 0
        assert "an-adapter" in result.output

    def test_min_size_filter_parses_human_sizes(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "list", "--min-size", "100GB"]
        )

        assert result.exit_code == 0
        assert "No matching assets" in result.output

    def test_empty_catalogue_says_so(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "empty.db"), "list"])

        assert result.exit_code == 0
        assert "No matching assets" in result.output


class TestShow:
    def test_shows_details(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "show", "1"])

        assert result.exit_code == 0
        assert "Detected by" in result.output

    def test_lists_files_on_request(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "show", "1", "--files"])

        assert result.exit_code == 0

    def test_unknown_id_exits_nonzero(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "show", "9999"])

        assert result.exit_code == 1


class TestStats:
    def test_summarises_catalogue(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "stats"])

        assert result.exit_code == 0
        assert "Catalogue" in result.output
        assert "Total size" in result.output

    def test_empty_catalogue_says_so(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "empty.db"), "stats"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()


class TestRoots:
    def test_add_list_remove(self, tmp_path: Path, runner: CliRunner) -> None:
        database = tmp_path / "db.sqlite"
        target = tmp_path / "models"
        target.mkdir()

        added = runner.invoke(app, ["--database", str(database), "roots", "add", str(target)])
        listed = runner.invoke(app, ["--database", str(database), "roots", "list"])
        removed = runner.invoke(
            app, ["--database", str(database), "roots", "remove", str(target)]
        )

        assert added.exit_code == 0
        assert "models" in listed.output
        assert removed.exit_code == 0

    def test_removing_unknown_root_exits_nonzero(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app, ["--database", str(tmp_path / "db.sqlite"), "roots", "remove", str(tmp_path)]
        )

        assert result.exit_code == 1

    def test_empty_root_list(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "roots", "list"])

        assert result.exit_code == 0
        assert "No scan roots" in result.output


class TestInventory:
    def test_lists_everything_with_a_summary(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory"])

        assert result.exit_code == 0
        assert "AI Asset Inventory" in result.output
        assert "Total Assets" in result.output
        assert "text-model" in result.output

    @pytest.mark.parametrize("category", ["llm", "datasets", "models", "adapters", "all"])
    def test_category_filters_run(self, catalogue, runner: CliRunner, category: str) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", category])

        assert result.exit_code == 0

    def test_llm_excludes_datasets(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "llm"])

        assert result.exit_code == 0
        assert "coco" not in result.output

    def test_unknown_category_reports_the_valid_ones(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "banana"])

        assert result.exit_code == 2
        assert "Unknown category" in result.output

    def test_grouping(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--group-by", "category"]
        )

        assert result.exit_code == 0
        assert "asset(s)" in result.output

    def test_details_mode(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--details"]
        )

        assert result.exit_code == 0
        assert "Root folder" in result.output
        assert "Architecture" in result.output

    def test_limit_reports_the_true_total(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--limit", "1"]
        )

        assert result.exit_code == 0
        assert "Showing 1 of" in result.output

    @pytest.mark.parametrize(("fmt", "suffix"), [("csv", "csv"), ("json", "json"),
                                                 ("markdown", "md")])
    def test_exports(
        self, catalogue, runner: CliRunner, tmp_path: Path, fmt: str, suffix: str
    ) -> None:
        _assets, database = catalogue
        destination = tmp_path / f"inventory.{suffix}"

        result = runner.invoke(
            app,
            ["--database", str(database), "inventory", "--export", fmt,
             "--output", str(destination)],
        )

        assert result.exit_code == 0
        assert destination.exists()
        assert destination.read_text(encoding="utf-8").strip()

    def test_unknown_export_format_exits_nonzero(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--export", "pdf"]
        )

        assert result.exit_code == 2

    def test_empty_catalogue_is_explained(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(
            app, ["--database", str(tmp_path / "empty.db"), "inventory"]
        )

        assert result.exit_code == 0
        assert "No assets found" in result.output


class TestWhere:
    def test_finds_an_asset(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "where", "coco"])

        assert result.exit_code == 0
        assert "coco" in result.output

    def test_no_match_says_so(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "where", "zzzznope"])

        assert result.exit_code == 0
        assert "Nothing matching" in result.output


def test_version(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "version"])

    assert result.exit_code == 0
    assert "AI Asset Manager" in result.output
