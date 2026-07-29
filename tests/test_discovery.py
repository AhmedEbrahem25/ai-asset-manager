r"""Tests for auto discovery.

Two things must hold, and they pull in opposite directions. Discovery has to find enough to
spare the user from remembering where anything is, and it has to stay away from everything
else on the disk — a discovery pass that wanders into ``C:\\Windows`` is not slow, it is
unusable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.scanner.locations import (
    SOURCES,
    SWEEP_EXCLUDED,
    discover_sources,
    group_locations,
    known_locations,
    likely_asset_folders,
    sweep_drives,
)
from ai_asset_manager.backend.services.discovery_service import DiscoveryService
from ai_asset_manager.backend.state import AppState, load_state, save_state


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch):
    """Point the state file at a temporary directory for every test in this module."""
    monkeypatch.setattr(
        "ai_asset_manager.backend.state.state_path", lambda: tmp_path / "state.json"
    )
    return tmp_path / "state.json"


class TestSourceCatalogue:
    def test_the_tools_named_in_the_brief_are_covered(self) -> None:
        labels = " ".join(source.label.lower() for source in SOURCES)
        keys = {source.key for source in SOURCES}

        for expected in ("huggingface", "ollama", "torch", "whisper", "comfyui",
                         "automatic1111", "invokeai", "lmstudio", "koboldcpp",
                         "llama_cpp", "vllm", "onnx", "tfhub", "keras", "ngc",
                         "ultralytics", "mmdetection", "detectron2", "paddleocr",
                         "easyocr", "tesseract", "opencv", "fiftyone", "cvat",
                         "label_studio", "roboflow", "wandb", "mlflow", "tensorboard"):
            assert expected in keys, f"{expected} missing from the discovery catalogue"
        assert "yolo" in labels

    def test_every_source_offers_somewhere_to_look(self) -> None:
        for source in SOURCES:
            assert source.env_vars or source.defaults, source.key

    def test_only_existing_directories_are_offered(self) -> None:
        # A suggestion that turns out to be an empty path is worse than no suggestion,
        # because the user cannot tell which of the two it is.
        for location in known_locations():
            assert location.path.is_dir()
            assert location.label


class TestEnvironmentOverrides:
    def test_an_override_is_used(self, tmp_path: Path, monkeypatch) -> None:
        moved = tmp_path / "hf-on-another-drive"
        moved.mkdir()
        monkeypatch.setenv("HF_HOME", str(moved))

        found = {location.path: location for location in discover_sources()}
        assert moved.resolve() in found
        assert found[moved.resolve()].origin == "HF_HOME"

    def test_an_override_suppresses_the_default(self, tmp_path: Path, monkeypatch) -> None:
        """Someone who moved their cache did it because the default no longer holds it.

        Offering the old path too would send a scan at a folder that is gone or stale.
        """
        moved = tmp_path / "moved-cache"
        moved.mkdir()
        fake_home = tmp_path / "home"
        (fake_home / ".cache" / "huggingface").mkdir(parents=True)

        monkeypatch.setenv("USERPROFILE", str(fake_home))
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HF_HOME", str(moved))

        paths = {location.path for location in discover_sources()}
        assert moved.resolve() in paths
        assert (fake_home / ".cache" / "huggingface").resolve() not in paths

    def test_a_pointless_override_is_ignored(self, tmp_path: Path, monkeypatch) -> None:
        # An override naming a directory that does not exist is worse than no override.
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "nowhere"))

        paths = {location.path for location in discover_sources()}
        assert (tmp_path / "nowhere").resolve() not in paths

    def test_xdg_cache_home_relocates_every_cache_at_once(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "xdg"
        (cache / "huggingface").mkdir(parents=True)
        (cache / "torch").mkdir(parents=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))

        paths = {location.path for location in discover_sources()}
        assert (cache / "huggingface").resolve() in paths
        assert (cache / "torch").resolve() in paths

    def test_a_marker_is_required_where_one_is_declared(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Tesseract's folder only counts if it holds language data; an empty install
        # directory is not a model library.
        empty = tmp_path / "tessdata"
        empty.mkdir()
        monkeypatch.setenv("TESSDATA_PREFIX", str(empty))
        assert empty.resolve() not in {item.path for item in discover_sources()}

        (empty / "eng.traineddata").write_bytes(b"x")
        assert empty.resolve() in {item.path for item in discover_sources()}


class TestSweep:
    def test_likely_folder_names_are_found(self, tmp_path: Path) -> None:
        for name in ("Models", "Datasets", "checkpoints"):
            (tmp_path / name).mkdir()
        (tmp_path / "Holiday Photos").mkdir()

        found = {location.path for location in sweep_drives([tmp_path])}

        assert (tmp_path / "Models").resolve() in found
        assert (tmp_path / "Datasets").resolve() in found
        assert (tmp_path / "Holiday Photos").resolve() not in found

    def test_it_reaches_two_levels_down(self, tmp_path: Path) -> None:
        # Caches live under the profile, but a curated library sits at D:\Work\Models.
        (tmp_path / "Work" / "Models").mkdir(parents=True)

        found = {location.path for location in sweep_drives([tmp_path])}
        assert (tmp_path / "Work" / "Models").resolve() in found

    def test_an_interesting_parent_is_offered_instead_of_its_children(
        self, tmp_path: Path
    ) -> None:
        """A folder that is itself a match is offered whole rather than picked apart.

        The scanner recurses, so offering the parent covers everything below it; offering
        both would mean walking the same tree twice.
        """
        (tmp_path / "AI" / "Models").mkdir(parents=True)
        (tmp_path / "AI" / "datasets").mkdir(parents=True)

        found = {location.path for location in sweep_drives([tmp_path])}

        assert found == {(tmp_path / "AI").resolve()}

    def test_it_does_not_descend_into_a_match(self, tmp_path: Path) -> None:
        # The scanner recurses; offering both the parent and the child would scan twice.
        (tmp_path / "Models" / "checkpoints").mkdir(parents=True)

        found = {location.path for location in sweep_drives([tmp_path])}
        assert (tmp_path / "Models" / "checkpoints").resolve() not in found

    def test_system_directories_are_never_entered(self, tmp_path: Path) -> None:
        """The one rule that makes a whole-disk sweep affordable."""
        for system in ("Windows", "Program Files", "$Recycle.Bin", "node_modules"):
            (tmp_path / system / "Models").mkdir(parents=True)

        found = {location.path for location in sweep_drives([tmp_path])}

        assert found == set()

    def test_the_exclusion_list_covers_the_obvious_offenders(self) -> None:
        for name in ("windows", "program files", "program files (x86)", "system32",
                     "$recycle.bin", "node_modules", "temp", "system volume information"):
            assert name in SWEEP_EXCLUDED

    def test_results_are_capped(self, tmp_path: Path) -> None:
        # A machine with a hundred candidates has a naming problem, not a discovery one.
        for index in range(80):
            (tmp_path / f"models{index}" / "models").mkdir(parents=True)

        assert len(sweep_drives([tmp_path])) <= 60


class TestGrouping:
    def test_locations_are_grouped_for_display(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        (tmp_path / "hf").mkdir()

        grouped = group_locations(likely_asset_folders([tmp_path]))

        assert "Model caches" in grouped
        assert all(isinstance(items, list) for items in grouped.values())


class TestDiscoveryService:
    def test_candidates_exclude_what_is_already_managed(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        service = DiscoveryService(session)
        service.scans.add_root(str(cache))
        session.commit()

        report = service.discover(sweep=False)

        assert cache.resolve() not in {item.path for item in report.candidates}
        assert cache.resolve() in {item.path for item in report.already_managed}

    def test_a_folder_inside_a_managed_root_is_not_offered_again(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        parent = tmp_path / "everything"
        cache = parent / "hf"
        cache.mkdir(parents=True)
        monkeypatch.setenv("HF_HOME", str(cache))

        service = DiscoveryService(session)
        service.scans.add_root(str(parent))
        session.commit()

        report = service.discover(sweep=False)
        assert cache.resolve() not in {item.path for item in report.candidates}

    def test_declining_is_remembered(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        service = DiscoveryService(session)
        first = service.discover(sweep=False)
        service.decline(first.candidates)

        second = service.discover(sweep=False)
        assert not second.candidates
        assert second.previously_declined

    def test_include_declined_offers_them_again(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        # Asking again is the whole reason someone runs discovery a second time.
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        service = DiscoveryService(session)
        service.decline(service.discover(sweep=False).candidates)

        assert service.discover(sweep=False, include_declined=True).candidates

    def test_accepting_registers_scan_roots(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        service = DiscoveryService(session)
        added = service.accept(service.discover(sweep=False).candidates)

        assert added
        assert {root.path for root in service.scans.list_roots()} == set(added)

    def test_discovery_runs_once(self, session: Session) -> None:
        # Being asked the same question at every launch is worse than never being asked.
        assert DiscoveryService.should_run_on_startup() is True

        DiscoveryService(session).complete()

        assert DiscoveryService.should_run_on_startup() is False

    def test_saying_no_still_counts_as_having_run(self, session: Session) -> None:
        service = DiscoveryService(session)
        report = service.discover(sweep=False)
        service.complete(declined=report.candidates)

        assert load_state().discovery_completed is True

    def test_it_never_scans_anything_itself(
        self, session: Session, tmp_path: Path, monkeypatch
    ) -> None:
        """Discovery decides where to look. The user decides whether to look there."""
        cache = tmp_path / "hf"
        cache.mkdir()
        monkeypatch.setenv("HF_HOME", str(cache))

        from ai_asset_manager.backend.models import Asset, ScanRun

        DiscoveryService(session).discover()

        assert session.query(Asset).count() == 0
        assert session.query(ScanRun).count() == 0


class TestStateIsolation:
    def test_declined_paths_survive_a_reload(self, tmp_path: Path) -> None:
        state = AppState()
        state.declined_paths = ["C:/one", "C:/two"]
        save_state(state)

        assert load_state().declined_paths == ["C:/one", "C:/two"]
