"""Turning a storm of filesystem events into a handful of updates.

Copying a model into a watched folder produces thousands of events in a few seconds. Acting
on each one would mean thousands of rescans of the same directory, which is both slower and
less correct than doing the work once at the end.

Two timers, not one:

*Quiet period.* Nothing is processed until the events stop arriving for a moment. This is
the usual debounce and it handles the common case — a copy finishes, a beat passes, one
update runs.

*Maximum wait.* A long download never goes quiet: a 40 GB model writes continuously for
minutes, so a pure quiet-period debounce would hold every change until it finished and show
the user a stale inventory the whole time. After the maximum wait the batch is flushed
regardless, and whatever is still arriving lands in the next one.

Everything here is thread-safe by construction: events arrive on the watchdog observer's
thread and are drained on this module's own worker.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: How often the worker wakes to decide whether a batch is ready. Small enough that the
#: quiet period is honoured closely, large enough to cost nothing while idle.
TICK_SECONDS = 0.25


@dataclass(slots=True)
class DebounceStats:
    """Counters describing what the debouncer has absorbed."""

    events_received: int = 0
    batches_emitted: int = 0
    paths_emitted: int = 0
    #: Events that arrived while a batch was being processed.
    events_during_processing: int = 0
    last_batch_at: float | None = None
    last_batch_size: int = 0
    errors: int = 0
    #: Set while a batch is being handled, so status can say "busy" honestly.
    processing: bool = False
    pending: int = 0
    _extra: dict[str, int] = field(default_factory=dict)


class EventDebouncer:
    """Collects paths and emits them in coalesced batches."""

    def __init__(
        self,
        on_batch: Callable[[set[str]], None],
        *,
        quiet_seconds: float = 2.0,
        max_wait_seconds: float = 30.0,
        name: str = "aam-debounce",
    ) -> None:
        """Create a debouncer.

        Args:
            on_batch: Called on the worker thread with the coalesced paths.
            quiet_seconds: How long events must stop before a batch is emitted.
            max_wait_seconds: Emit anyway once the oldest pending event is this old.
            name: Worker thread name, so a stack dump is readable.
        """
        self._on_batch = on_batch
        self._quiet = max(0.05, quiet_seconds)
        # Not clamped to the quiet period. A maximum wait *shorter* than the quiet period
        # is a legitimate thing to ask for - it means "flush on a fixed cadence whatever
        # is happening" - and quietly raising it to match would ignore the request.
        self._max_wait = max(TICK_SECONDS, max_wait_seconds)
        self._name = name

        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._first_event: float | None = None
        self._last_event: float | None = None

        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.stats = DebounceStats()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread."""
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._worker.start()

    def stop(self, *, flush: bool = True, timeout: float = 30.0) -> None:
        """Stop the worker, optionally processing whatever is still pending."""
        self._stop.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=timeout)
        if flush:
            self.flush()

    def __enter__(self) -> EventDebouncer:
        """Start the worker on entry."""
        self.start()
        return self

    def __exit__(self, *exception: object) -> None:
        """Stop the worker on exit."""
        self.stop()

    # -- input --------------------------------------------------------------

    def add(self, path: str) -> None:
        """Record a changed path."""
        self.add_many((path,))

    def add_many(self, paths: Iterable[str]) -> None:
        """Record several changed paths at once."""
        now = time.monotonic()
        with self._lock:
            before = len(self._pending)
            self._pending.update(paths)
            added = len(self._pending) - before
            self.stats.events_received += max(added, 1) if added else 1
            if self.stats.processing:
                self.stats.events_during_processing += 1
            if self._first_event is None:
                self._first_event = now
            self._last_event = now
            self.stats.pending = len(self._pending)

    def flush(self) -> int:
        """Process everything pending immediately. Returns how many paths were emitted."""
        batch = self._drain()
        if batch:
            self._emit(batch)
        return len(batch)

    @property
    def pending(self) -> int:
        """Return how many distinct paths are waiting."""
        with self._lock:
            return len(self._pending)

    # -- internals ----------------------------------------------------------

    def _drain(self) -> set[str]:
        """Take everything pending and reset the timers."""
        with self._lock:
            batch = self._pending
            self._pending = set()
            self._first_event = None
            self._last_event = None
            self.stats.pending = 0
        return batch

    def _ready(self, now: float) -> bool:
        """Report whether a batch should be emitted."""
        with self._lock:
            if not self._pending or self._last_event is None or self._first_event is None:
                return False
            quiet = now - self._last_event >= self._quiet
            overdue = now - self._first_event >= self._max_wait
            return quiet or overdue

    def _emit(self, batch: set[str]) -> None:
        """Hand a batch to the callback, absorbing whatever it throws.

        A failing handler must not kill the worker: the watcher would go on collecting
        events forever and never act on them, which looks exactly like it is running fine.
        """
        with self._lock:
            self.stats.processing = True
        try:
            self._on_batch(batch)
            with self._lock:
                self.stats.batches_emitted += 1
                self.stats.paths_emitted += len(batch)
                self.stats.last_batch_at = time.time()
                self.stats.last_batch_size = len(batch)
        except Exception:
            with self._lock:
                self.stats.errors += 1
            logger.exception("Failed to process a batch of %d path(s)", len(batch))
        finally:
            with self._lock:
                self.stats.processing = False

    def _run(self) -> None:
        """Wake periodically and emit a batch when one is ready."""
        logger.debug(
            "Debouncer started: quiet=%.1fs max_wait=%.1fs", self._quiet, self._max_wait
        )
        while not self._stop.is_set():
            self._stop.wait(TICK_SECONDS)
            if self._ready(time.monotonic()):
                batch = self._drain()
                if batch:
                    self._emit(batch)
        logger.debug("Debouncer stopped")
