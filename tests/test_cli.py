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
    """Return a Typer test runner."""
    return CliRunner()


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


def test_version(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "version"])

    assert result.exit_code == 0
    assert "AI Asset Manager" in result.output
