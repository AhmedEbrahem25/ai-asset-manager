"""The live filesystem watcher.

Runs for as long as it is asked to, keeping the catalogue in step with the disk. The parts
it wires together are each dull on their own, which is the point: watchdog produces events,
:class:`~ai_asset_manager.backend.watch.debounce.EventDebouncer` turns storms of them into
occasional batches, and
:class:`~ai_asset_manager.backend.watch.indexer.LiveIndexer` turns a batch into the
smallest scan that will make the catalogue right.

Two things this deliberately does not do.

It does not run as a service or fork itself. A watcher that survives the terminal it was
started from is a thing users then have to find and kill; ``aam watch`` occupies its
terminal and stops with Ctrl-C, and ``aam watch --stop`` exists for the other terminal.

It does not filter events by extension. What matters is which *directory* changed, and a
``.tmp`` file appearing in a model folder is evidence that the folder is being written to
even though the file itself will never be catalogued.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ai_asset_manager.backend.services.scan_service import ScanService
from ai_asset_manager.backend.state import AppState, load_state, save_state
from ai_asset_manager.backend.watch.debounce import EventDebouncer
from ai_asset_manager.backend.watch.indexer import IndexResult, LiveIndexer
from ai_asset_manager.config import Settings, get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Returns a context-managed session. ``session_scope`` satisfies it; so does a
#: test's factory, which is what lets the watcher be driven without a real database.
SessionFactory = Callable[[], AbstractContextManager[Session]]

#: How often the run loop checks whether a stop was requested from another process.
STOP_POLL_SECONDS = 1.0

#: Longest the watcher waits for a batch in flight when shutting down. A scan of a large
#: root can be mid-flight, and killing it would leave the run row marked running forever.
SHUTDOWN_GRACE_SECONDS = 30.0


@dataclass(slots=True)
class WatchStatus:
    """A snapshot of what the watcher is doing."""

    running: bool
    roots: list[str]
    pending_paths: int
    batches_processed: int
    paths_processed: int
    assets_created: int
    assets_updated: int
    assets_missing: int
    errors: int
    started_at: float | None
    last_batch_at: float | None
    last_result: IndexResult | None


class WatchService:
    """Watches managed roots and keeps the catalogue synchronised."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        settings: Settings | None = None,
        indexer: LiveIndexer | None = None,
    ) -> None:
        """Create a watcher bound to a session factory."""
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self._indexer = indexer or LiveIndexer(session_factory, settings=self._settings)

        self._debouncer = EventDebouncer(
            self._on_batch,
            quiet_seconds=self._settings.watch_debounce_seconds,
            max_wait_seconds=self._settings.watch_max_wait_seconds,
        )
        self._observer: object | None = None
        self._roots: list[str] = []
        self._started_at: float | None = None
        self._stop = threading.Event()
        self._on_result: Callable[[IndexResult], None] | None = None

    # -- introspection ------------------------------------------------------

    def roots(self) -> list[str]:
        """Return the roots this watcher covers."""
        if self._roots:
            return list(self._roots)
        with self._session_factory() as session:
            return [root.path for root in ScanService(session).list_roots(enabled_only=True)]

    def status(self) -> WatchStatus:
        """Return what the watcher is currently doing."""
        stats = self._indexer.stats
        return WatchStatus(
            running=self._observer is not None,
            roots=list(self._roots),
            pending_paths=self._debouncer.pending,
            batches_processed=stats.batches,
            paths_processed=stats.paths_seen,
            assets_created=stats.assets_created,
            assets_updated=stats.assets_updated,
            assets_missing=stats.assets_missing,
            errors=stats.errors,
            started_at=self._started_at,
            last_batch_at=stats.last_run_at,
            last_result=stats.last_result,
        )

    # -- lifecycle ----------------------------------------------------------

    def run(
        self,
        roots: Sequence[str] | None = None,
        *,
        on_result: Callable[[IndexResult], None] | None = None,
        on_ready: Callable[[list[str]], None] | None = None,
    ) -> None:
        """Watch until stopped. Blocks the calling thread.

        Args:
            roots: Directories to watch; the managed roots when omitted.
            on_result: Called after each batch, for progress display.
            on_ready: Called once watching has started, with the watched roots.

        Raises:
            RuntimeError: If no roots are available to watch.
        """
        self._on_result = on_result
        self._roots = list(roots) if roots else self.roots()

        if not self._roots:
            raise RuntimeError(
                "No folders to watch. Add one with 'aam scan --add <path>' or "
                "'aam discover'."
            )

        observer = self._start_observer(self._roots)
        self._observer = observer
        self._started_at = time.time()
        self._debouncer.start()
        self._publish_state(running=True)

        if on_ready is not None:
            on_ready(list(self._roots))

        logger.info("Watching %d root(s)", len(self._roots))

        try:
            self._loop()
        except KeyboardInterrupt:
            logger.info("Watcher interrupted")
        finally:
            self.shutdown()

    def request_stop(self) -> None:
        """Ask the run loop to finish, from any thread."""
        self._stop.set()

    def shutdown(self) -> None:
        """Stop watching and process whatever is still pending."""
        observer, self._observer = self._observer, None
        if observer is not None:
            try:
                observer.stop()  # type: ignore[attr-defined]
                observer.join(timeout=SHUTDOWN_GRACE_SECONDS)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Observer did not stop cleanly", exc_info=True)

        # Flush on the way out. Changes seen a second before Ctrl-C are still changes, and
        # discarding them would leave the catalogue quietly wrong until the next scan.
        self._debouncer.stop(flush=True, timeout=SHUTDOWN_GRACE_SECONDS)
        self._publish_state(running=False)
        self._started_at = None
        logger.info("Watcher stopped")

    # -- internals ----------------------------------------------------------

    def _loop(self) -> None:
        """Wait until stopped, checking for an out-of-process stop request."""
        while not self._stop.wait(STOP_POLL_SECONDS):
            state = load_state()
            if state.watcher_stop_requested:
                logger.info("Stop requested by another process")
                # Clear the request before exiting, or the next watcher started would read
                # a stale flag and shut down immediately.
                state.watcher_stop_requested = False
                save_state(state)
                return

    def _start_observer(self, roots: Sequence[str]) -> object:
        """Create and start a watchdog observer over the given roots."""
        from watchdog.observers import Observer

        from ai_asset_manager.backend.watch.handler import ChangeHandler

        observer = Observer()
        handler = ChangeHandler(self._debouncer, excluded=self._settings.excluded_dirs)

        for root in roots:
            try:
                observer.schedule(handler, root, recursive=True)
            except OSError as exc:
                # A root on an unplugged drive should cost that root, not the watcher.
                logger.warning("Cannot watch %s: %s", root, exc)

        observer.start()
        return observer

    def _on_batch(self, paths: set[str]) -> None:
        """Index one debounced batch."""
        result = self._indexer.handle(paths)
        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception:
                logger.debug("Progress callback failed", exc_info=True)

    def _publish_state(self, *, running: bool) -> None:
        """Record the watcher's liveness so other processes can report it."""
        import os

        state: AppState = load_state()
        if running:
            state.watcher_pid = os.getpid()
            state.watcher_started_at = _iso_now()
            state.watcher_roots = list(self._roots)
            state.watcher_stop_requested = False
        else:
            state.watcher_pid = None
            state.watcher_started_at = None
            state.watcher_roots = []
        save_state(state)


def _iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def request_watcher_stop() -> bool:
    """Ask a running watcher in another process to stop.

    Returns whether one appeared to be running. Sets a flag rather than sending a signal:
    signals are unreliable across the console/service boundary on Windows, and a watcher
    mid-scan should finish its batch and shut down cleanly rather than be interrupted.
    """
    from ai_asset_manager.backend.state import process_is_alive

    state = load_state()
    if not process_is_alive(state.watcher_pid):
        # Tidy up after a watcher that was killed rather than stopped.
        if state.watcher_pid is not None:
            state.watcher_pid = None
            state.watcher_started_at = None
            state.watcher_roots = []
            save_state(state)
        return False

    state.watcher_stop_requested = True
    save_state(state)
    return True
