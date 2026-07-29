"""CLI smoke tests.

Each invocation goes through the real Typer app against a real temporary database, so
argument wiring, the service calls and the rendering are all exercised together.
"""

from __future__ import annotations

import sys
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


class TestFirstImpression:
    """What the tool says to someone who has just installed it."""

    def test_bare_invocation_offers_folders_to_scan(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        # A bare `aam` should not print a wall of twelve commands. It should say whether
        # there is anything to look at yet, and if not, how to make some.
        result = runner.invoke(app, ["--database", str(tmp_path / "none.db")])

        assert result.exit_code == 0
        assert "Nothing catalogued yet" in result.output
        assert "aam scan" in result.output

    def test_bare_invocation_with_a_catalogue_says_what_is_in_it(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database)])

        assert result.exit_code == 0
        assert "asset(s) catalogued" in result.output
        assert "aam inventory" in result.output

    def test_asking_for_the_version_does_not_create_a_database(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        """A freshly copied binary being asked its version should leave nothing behind."""
        database = tmp_path / "should-not-exist.db"

        result = runner.invoke(app, ["--database", str(database), "--version"])

        assert result.exit_code == 0
        assert "AI Asset Manager" in result.output
        assert not database.exists()

    def test_guide_lists_worked_examples(self, tmp_path: Path, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "guide"])

        assert result.exit_code == 0
        assert "aam scan --auto" in result.output
        assert "aam inventory missing" in result.output

    def test_asking_for_the_guide_does_not_create_a_database(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        """The welcome screen sends new users to `guide`; it must stay a pure answer.

        Creating the schema here made the first command a new user runs print a line of
        INFO logging about twelve tables before the worked examples.
        """
        database = tmp_path / "should-not-exist.db"

        result = runner.invoke(app, ["--database", str(database), "guide"])

        assert result.exit_code == 0
        assert "aam scan --auto" in result.output
        assert not database.exists()

    def test_help_offers_no_completion_flags_on_windows(self, runner: CliRunner) -> None:
        """Typer's completion flags raise on Windows, so they must not be advertised there.

        shellingham cannot detect the shell on 'nt': both --install-completion and
        --show-completion die with "Shell detection not implemented for 'nt'".
        """
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        if sys.platform == "win32":
            assert "--install-completion" not in result.output
            assert "--show-completion" not in result.output
        else:
            assert "--install-completion" in result.output

    def test_help_groups_commands_into_panels(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Getting started" in result.output
        assert "Your library" in result.output
        # Start comes before about, or the first thing a new user reads is "version".
        assert result.output.index("Getting started") < result.output.index("About")

    def test_a_mistyped_category_suggests_the_right_one(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "detecton"])

        assert result.exit_code == 2
        assert "Did you mean" in result.output
        assert "detection" in result.output

    def test_scan_with_no_roots_points_at_auto(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "scan"])

        assert result.exit_code != 0
        assert "--auto" in result.output


class TestDiscoverAndStatus:
    @pytest.fixture(autouse=True)
    def isolated_state(self, tmp_path: Path, monkeypatch):
        """Keep discovery's memory inside the test's own directory."""
        monkeypatch.setattr(
            "ai_asset_manager.backend.state.state_path", lambda: tmp_path / "state.json"
        )

    def test_discover_offers_and_adds(self, tmp_path: Path, runner: CliRunner,
                                      monkeypatch) -> None:
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        result = runner.invoke(
            app,
            ["--database", str(tmp_path / "db.sqlite"), "discover", "--yes", "--no-scan",
             "--no-sweep"],
        )

        assert result.exit_code == 0
        assert "Found AI assets" in result.output
        assert "Added" in result.output

    def test_discover_scans_nothing_when_declined(
        self, tmp_path: Path, runner: CliRunner, monkeypatch
    ) -> None:
        # Nothing is scanned before approval. That is the whole contract of the prompt.
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        result = runner.invoke(
            app,
            ["--database", str(tmp_path / "db.sqlite"), "discover", "--no-sweep"],
            input="N\n",
        )

        assert result.exit_code == 0
        assert "Nothing added" in result.output

    def test_discover_remembers_it_ran(self, tmp_path: Path, runner: CliRunner,
                                       monkeypatch) -> None:
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))
        database = str(tmp_path / "db.sqlite")

        runner.invoke(app, ["--database", database, "discover", "--no-sweep"], input="N\n")
        again = runner.invoke(app, ["--database", database, "discover", "--no-sweep"])

        assert "Nothing new" in again.output
        assert "previously declined" in again.output

    def test_status_reports_an_empty_installation(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "status"])

        assert result.exit_code == 0
        assert "Last scan" in result.output
        assert "never" in result.output
        assert "No managed folders" in result.output

    def test_status_reports_a_catalogue(self, catalogue, runner: CliRunner) -> None:
        assets, database = catalogue
        # The shared fixture scans without registering, which is a legitimate way to use
        # the tool; managed folders only appear once something is remembered.
        runner.invoke(app, ["--database", str(database), "roots", "add", str(assets)])

        result = runner.invoke(app, ["--database", str(database), "status"])

        assert result.exit_code == 0
        assert "Managed folders" in result.output
        assert "Watcher" in result.output
        assert "not running" in result.output
        assert "Taxonomy" in result.output

    def test_status_says_when_nothing_is_remembered(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "status"])

        assert result.exit_code == 0
        assert "No managed folders" in result.output

    def test_watch_status_with_nothing_running(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app, ["--database", str(tmp_path / "db.sqlite"), "watch", "--status"]
        )

        assert result.exit_code == 0
        assert "Not running" in result.output

    def test_watch_stop_with_nothing_running(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app, ["--database", str(tmp_path / "db.sqlite"), "watch", "--stop"]
        )

        assert result.exit_code == 0
        assert "No watcher is running" in result.output

    def test_watch_refuses_with_nothing_to_watch(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "watch"])

        assert result.exit_code == 1
        assert "discover" in result.output

    def test_a_stale_watcher_pid_is_not_reported_as_running(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        """A watcher that was killed leaves its pid behind; status must not believe it."""
        from ai_asset_manager.backend.state import AppState, save_state

        state = AppState()
        state.watcher_pid = 4_000_000_000
        save_state(state)

        result = runner.invoke(
            app, ["--database", str(tmp_path / "db.sqlite"), "watch", "--status"]
        )

        assert "Not running" in result.output

    def test_scan_rejects_contradictory_flags(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            app,
            ["--database", str(tmp_path / "db.sqlite"), "scan", str(tmp_path),
             "--full", "--incremental"],
        )

        assert result.exit_code == 2
        assert "opposite" in result.output


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

    def test_reports_the_same_health_as_the_inventory(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        """`show` must explain a low score, not leave the user to guess at it.

        Only parser warnings are persisted as findings, so reading that table alone made
        `show` silent about an asset the health view had just scored 65/100.
        """
        assets = tmp_path / "assets"
        assets.mkdir()
        F.make_incomplete_download(assets, "half-downloaded")
        database = tmp_path / "catalog.db"
        runner.invoke(app, ["--database", str(database), "scan", str(assets), "-q"])

        health = runner.invoke(app, ["--database", str(database), "inventory", "health"])
        shown = runner.invoke(app, ["--database", str(database), "show", "1"])

        assert shown.exit_code == 0
        assert "Health" in shown.output
        # The finding the health view reports is the finding `show` explains.
        assert "unfinished download" in health.output
        assert "unfinished download" in shown.output


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

    def test_a_filter_that_matches_nothing_blames_the_filter(
        self, catalogue, runner: CliRunner
    ) -> None:
        """The advice has to match the reason, or it sends the user in a circle.

        `--drive Z:` used to report "no assets found for 'all'" and suggest running
        `aam inventory all` -- which is what had just been run.
        """
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--drive", "Z:"]
        )

        assert result.exit_code == 0
        assert "--drive Z:" in result.output
        assert "Run 'aam scan" not in result.output

    def test_an_empty_category_points_at_the_full_list(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "speech"])

        assert result.exit_code == 0
        assert "aam inventory all" in result.output

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

    def test_details_answers_what_each_asset_is_for(
        self, catalogue, runner: CliRunner
    ) -> None:
        """The success criterion for the whole feature.

        After one command a user should know what each asset does, where it is and
        whether it is complete.
        """
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--details"]
        )

        assert result.exit_code == 0
        for expected in ("Task", "Domain", "Health", "Path", "Identified by"):
            assert expected in result.output

    def test_tree_view(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "--tree"])

        assert result.exit_code == 0
        assert "AI Library" in result.output
        assert "Models" in result.output
        assert "text-model" in result.output

    def test_tree_nesting_can_be_chosen(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--tree-by", "domain,task"]
        )

        assert result.exit_code == 0
        assert "AI Library" in result.output

    def test_health_view(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "health"])

        assert result.exit_code == 0
        assert "Score" in result.output
        assert "/100" in result.output

    def test_missing_view_lists_only_problems(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(app, ["--database", str(database), "inventory", "missing"])

        assert result.exit_code == 0
        # Either it found something to fix, or it said there was nothing to fix. Silence
        # would be the failure.
        assert "Findings" in result.output or "Nothing needs attention" in result.output

    def test_missing_on_a_clean_catalogue_says_so(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        database = tmp_path / "empty.db"

        result = runner.invoke(app, ["--database", str(database), "inventory", "missing"])

        assert result.exit_code == 0
        assert "Nothing needs attention" in result.output

    def test_task_filter(self, catalogue, runner: CliRunner) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--task", "object_detection"]
        )

        assert result.exit_code == 0

    def test_storage_breakdown_includes_task_distribution(
        self, catalogue, runner: CliRunner
    ) -> None:
        _assets, database = catalogue

        result = runner.invoke(
            app, ["--database", str(database), "inventory", "--storage"]
        )

        assert result.exit_code == 0
        assert "Storage by drive" in result.output
        assert "Assets by task" in result.output

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
        assert "catalogue is empty" in result.output.lower()
        assert "aam scan" in result.output


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


class TestLegacyConsoleEncoding:
    """A cp1252 console must not be able to kill a command mid-render.

    The development machine's console is cp1252, and `aam inventory` died part-way through
    a table with `UnicodeEncodeError` — after printing half of it. Two entirely ordinary
    inputs reach it: a model whose name is not Latin-1, and this CLI's own tick marks.
    """

    def test_streams_are_reconfigured_to_utf8(self) -> None:
        import io

        from ai_asset_manager.cli import _force_utf8_streams

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        original, sys.stdout = sys.stdout, stream
        try:
            _force_utf8_streams()
            assert stream.encoding == "utf-8"
            assert stream.errors == "replace"
        finally:
            sys.stdout = original

    def test_a_stream_without_reconfigure_is_left_alone(self) -> None:
        """Under pytest's capture the streams are not real files. That is not an error."""
        from ai_asset_manager.cli import _force_utf8_streams

        original, sys.stdout = sys.stdout, object()
        try:
            _force_utf8_streams()  # must not raise
        finally:
            sys.stdout = original

    def test_a_non_latin1_name_renders(self, tmp_path: Path, runner: CliRunner) -> None:
        assets = tmp_path / "assets"
        F.make_hf_model(assets, "Qwen2.5-日本語-0.5B")
        database = tmp_path / "db.sqlite"

        scan = runner.invoke(app, ["--database", str(database), "scan", str(assets)])
        assert scan.exit_code == 0

        result = runner.invoke(app, ["--database", str(database), "list"])
        assert result.exit_code == 0
        assert result.exception is None


def test_version(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["--database", str(tmp_path / "db.sqlite"), "version"])

    assert result.exit_code == 0
    assert "AI Asset Manager" in result.output
