"""Tests for live indexing.

The debouncer and the indexer are tested directly rather than through a real watchdog
observer. Filesystem event delivery is the operating system's job and differs between
platforms; what this project has to get right is what it *does* with the events, and that
is deterministic.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.services.scan_service import ScanService
from ai_asset_manager.backend.state import (
    AppState,
    load_state,
    process_is_alive,
    save_state,
)
from ai_asset_manager.backend.utils.paths import normalize_path
from ai_asset_manager.backend.watch.debounce import EventDebouncer
from ai_asset_manager.backend.watch.handler import ChangeHandler
from ai_asset_manager.backend.watch.indexer import LiveIndexer
from tests import factories as F


class TestDebouncer:
    def test_a_burst_becomes_one_batch(self) -> None:
        """A thousand files copied must produce one update, not a thousand."""
        batches: list[set[str]] = []
        debouncer = EventDebouncer(batches.append, quiet_seconds=0.2, max_wait_seconds=5)

        with debouncer:
            for index in range(1000):
                debouncer.add(f"/library/model/shard-{index}.safetensors")
            _wait_for(lambda: len(batches) == 1)

        assert len(batches) == 1
        assert len(batches[0]) == 1000

    def test_duplicate_paths_are_coalesced(self) -> None:
        batches: list[set[str]] = []
        debouncer = EventDebouncer(batches.append, quiet_seconds=0.2)

        with debouncer:
            for _ in range(50):
                debouncer.add("/library/model/config.json")
            _wait_for(lambda: batches)

        assert batches[0] == {"/library/model/config.json"}

    def test_the_quiet_period_is_honoured(self) -> None:
        batches: list[set[str]] = []
        debouncer = EventDebouncer(batches.append, quiet_seconds=0.6)

        with debouncer:
            debouncer.add("/library/a")
            time.sleep(0.2)
            assert not batches, "emitted before the quiet period elapsed"
            _wait_for(lambda: batches, timeout=3.0)

        assert batches

    def test_a_continuous_stream_is_flushed_by_the_maximum_wait(self) -> None:
        """A long download never goes quiet, and must not hold every change until it ends.

        Without a ceiling on the wait, a 40 GB model writing for minutes would keep the
        inventory stale for the whole download.
        """
        batches: list[set[str]] = []
        debouncer = EventDebouncer(batches.append, quiet_seconds=5.0, max_wait_seconds=0.5)

        with debouncer:
            deadline = time.monotonic() + 1.5
            index = 0
            while time.monotonic() < deadline:
                debouncer.add(f"/library/big/part-{index}")
                index += 1
                time.sleep(0.05)
            _wait_for(lambda: batches, timeout=3.0)

        assert batches, "the maximum wait never fired"

    def test_a_failing_handler_does_not_kill_the_worker(self) -> None:
        # A watcher that silently stops acting on events looks exactly like one that is
        # working, which is the worst possible failure mode.
        seen: list[set[str]] = []

        def handler(batch: set[str]) -> None:
            if not seen:
                seen.append(batch)
                raise RuntimeError("indexing blew up")
            seen.append(batch)

        debouncer = EventDebouncer(handler, quiet_seconds=0.15)
        with debouncer:
            debouncer.add("/one")
            _wait_for(lambda: len(seen) == 1)
            debouncer.add("/two")
            _wait_for(lambda: len(seen) == 2)

        assert len(seen) == 2
        assert debouncer.stats.errors == 1

    def test_stopping_flushes_what_is_pending(self) -> None:
        # Changes seen a second before Ctrl-C are still changes.
        batches: list[set[str]] = []
        debouncer = EventDebouncer(batches.append, quiet_seconds=60.0)
        debouncer.start()
        debouncer.add("/library/late-change")
        debouncer.stop(flush=True)

        assert batches == [{"/library/late-change"}]


class TestChangeHandler:
    def _handler(self) -> tuple[ChangeHandler, list[str]]:
        captured: list[str] = []
        debouncer = EventDebouncer(lambda batch: captured.extend(batch))
        return ChangeHandler(debouncer, excluded={".git", "__pycache__"}), captured

    def test_excluded_directories_are_ignored(self) -> None:
        handler, _ = self._handler()

        assert handler._is_ignored("/library/model/.git/objects/ab/cdef")
        assert handler._is_ignored("/library/model/__pycache__/x.pyc")
        assert not handler._is_ignored("/library/model/config.json")

    def test_exclusion_is_by_segment_not_substring(self) -> None:
        # A folder legitimately called "cached-models" must not be pruned because a
        # substring of it appears in the exclusion list.
        handler, _ = self._handler()
        handler._excluded = frozenset({"cache"})

        assert not handler._is_ignored("/library/cached-models/model.safetensors")
        assert handler._is_ignored("/library/cache/model.safetensors")

    def test_editor_scratch_files_are_ignored(self) -> None:
        handler, _ = self._handler()

        assert handler._is_ignored("/library/notes.md.swp")
        assert handler._is_ignored("/library/catalog.db-wal")


@pytest.fixture
def live(session: Session, settings, tmp_path: Path):
    """Return a scanned library plus an indexer bound to it."""
    library = tmp_path / "library"
    library.mkdir()
    F.make_hf_model(library, "first-model", architecture="Qwen2ForCausalLM")

    scans = ScanService(session, settings=settings, pipeline=ScanPipeline(settings=settings))
    scans.add_root(str(library))
    session.commit()
    scans.scan([str(library)])

    @contextmanager
    def factory():
        """Hand the indexer the test's own session, uncommitted state and all."""
        yield session

    return library, session, LiveIndexer(factory, settings=settings)


class TestLiveIndexer:
    def test_a_change_inside_a_known_asset_targets_that_asset(self, live) -> None:
        library, session, indexer = live
        asset_root = normalize_path(str(library / "first-model"))

        targets = indexer.resolve_targets(
            session,
            {str(library / "first-model" / "config.json")},
            [normalize_path(str(library))],
        )

        # The narrowest subtree that certainly contains the change.
        assert targets == [asset_root]

    def test_an_unknown_path_falls_back_to_the_managed_root(self, live) -> None:
        """A new asset has no known boundary, so the root is the only safe answer.

        A new HuggingFace repo is three directories below the cache root and its manifest
        and weights live in different subtrees; guessing a narrower target would miss half
        of it.
        """
        library, session, indexer = live
        root = normalize_path(str(library))

        targets = indexer.resolve_targets(
            session, {str(library / "brand-new-model" / "config.json")}, [root]
        )

        assert targets == [root]

    def test_a_root_scan_absorbs_the_asset_scans_beneath_it(self, live) -> None:
        library, session, indexer = live
        root = normalize_path(str(library))

        targets = indexer.resolve_targets(
            session,
            {
                str(library / "first-model" / "config.json"),
                str(library / "brand-new-model" / "config.json"),
            },
            [root],
        )

        # One walk of the root does everything the per-asset scan would have done.
        assert targets == [root]

    def test_paths_outside_every_root_are_dropped(self, live) -> None:
        library, session, indexer = live

        targets = indexer.resolve_targets(
            session, {"C:/somewhere/else/model.safetensors"}, [normalize_path(str(library))]
        )

        assert targets == []

    def test_a_sibling_root_is_not_mistaken_for_a_parent(self, live, tmp_path: Path) -> None:
        # "D:\Models2" must never be treated as living inside "D:\Models".
        library, session, indexer = live
        sibling = normalize_path(f"{library}2")

        targets = indexer.resolve_targets(
            session, {f"{sibling}/thing.bin"}, [normalize_path(str(library))]
        )

        assert targets == []

    def test_a_new_asset_is_catalogued(self, live) -> None:
        library, session, indexer = live
        F.make_hf_model(library, "second-model", architecture="Qwen2ForCausalLM")

        result = indexer.handle({str(library / "second-model" / "config.json")})

        assert result.assets_created == 1
        assert "second-model" in " ".join(
            asset.name for asset in session.query(__import__(
                "ai_asset_manager.backend.models", fromlist=["Asset"]
            ).Asset).all()
        )

    def test_a_deleted_asset_is_marked_missing_not_removed(self, live) -> None:
        """Marking rather than deleting is what makes an unplugged drive survivable."""
        import shutil

        from ai_asset_manager.backend.models import Asset

        library, session, indexer = live
        shutil.rmtree(library / "first-model")

        result = indexer.handle({str(library / "first-model")})

        assert result.assets_missing == 1
        asset = session.query(Asset).filter_by(name="first-model").one()
        assert asset.is_missing is True

    def test_an_unchanged_asset_costs_nothing(self, live) -> None:
        library, _session, indexer = live

        result = indexer.handle({str(library / "first-model" / "config.json")})

        assert result.assets_created == 0
        assert result.assets_updated == 0
        assert result.assets_unchanged == 1

    def test_an_empty_batch_does_nothing(self, live) -> None:
        _library, _session, indexer = live

        assert indexer.handle(set()).targets == []


class TestAppState:
    def test_a_missing_file_reads_as_defaults(self, tmp_path: Path) -> None:
        state = load_state(tmp_path / "nothing.json")

        assert state.discovery_completed is False
        assert state.watcher_pid is None

    def test_a_corrupt_file_reads_as_defaults(self, tmp_path: Path) -> None:
        # Losing this costs one repeated prompt; failing the command would cost the user
        # the command.
        path = tmp_path / "state.json"
        path.write_text("{ this is not json", encoding="utf-8")

        assert load_state(path).discovery_completed is False

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        # A state file written by a newer version must not stop an older one running.
        path = tmp_path / "state.json"
        path.write_text('{"discovery_completed": true, "from_the_future": 42}', "utf-8")

        assert load_state(path).discovery_completed is True

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = AppState()
        state.mark_discovery_done(declined=["C:/nope"])
        save_state(state, path)

        loaded = load_state(path)
        assert loaded.discovery_completed is True
        assert loaded.declined_paths == ["C:/nope"]

    def test_the_current_process_is_alive_and_a_made_up_one_is_not(self) -> None:
        import os

        assert process_is_alive(os.getpid()) is True
        assert process_is_alive(None) is False
        assert process_is_alive(0) is False
        # A pid this high is not in use on any machine that has just booted a test suite.
        assert process_is_alive(4_000_000_000) is False


def _wait_for(condition, *, timeout: float = 5.0) -> None:
    """Block until a condition holds, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"condition never became true within {timeout}s")


def test_the_debouncer_is_safe_across_threads() -> None:
    """Events arrive on watchdog's thread and drain on ours; the counters must survive."""
    batches: list[set[str]] = []
    debouncer = EventDebouncer(batches.append, quiet_seconds=0.4, max_wait_seconds=10)

    def produce(start: int) -> None:
        for index in range(start, start + 200):
            debouncer.add(f"/library/file-{index}")

    with debouncer:
        threads = [threading.Thread(target=produce, args=(base,)) for base in (0, 200, 400)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _wait_for(lambda: batches)

    assert sum(len(batch) for batch in batches) == 600
