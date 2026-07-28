"""Progress reporting and cooperative cancellation for long-running scans."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class ScanPhase(StrEnum):
    """Phases a scan moves through, reported to the UI in order."""

    STARTING = "starting"
    WALKING = "walking"
    DETECTING = "detecting"
    PARSING = "parsing"
    PERSISTING = "persisting"
    HASHING = "hashing"
    FINALISING = "finalising"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ScanProgress:
    """A snapshot of scan state.

    Emitted to observers rather than mutated in place by them, so a consumer that is slow
    (an SSE client on a laggy connection) cannot stall the scan.
    """

    phase: ScanPhase = ScanPhase.STARTING
    current_path: str = ""
    directories_seen: int = 0
    files_seen: int = 0
    bytes_seen: int = 0
    assets_found: int = 0
    assets_created: int = 0
    assets_updated: int = 0
    assets_unchanged: int = 0
    errors: int = 0
    #: Total units for the active phase, when known. Zero means indeterminate.
    total: int = 0
    completed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    message: str = ""

    @property
    def elapsed_seconds(self) -> float:
        """Return seconds since the scan began."""
        return time.monotonic() - self.started_at

    @property
    def fraction(self) -> float | None:
        """Return completion in the range 0..1, or ``None`` when indeterminate."""
        if self.total <= 0:
            return None
        return min(1.0, self.completed / self.total)

    def snapshot(self) -> ScanProgress:
        """Return an independent copy safe to hand to another thread."""
        return ScanProgress(
            phase=self.phase,
            current_path=self.current_path,
            directories_seen=self.directories_seen,
            files_seen=self.files_seen,
            bytes_seen=self.bytes_seen,
            assets_found=self.assets_found,
            assets_created=self.assets_created,
            assets_updated=self.assets_updated,
            assets_unchanged=self.assets_unchanged,
            errors=self.errors,
            total=self.total,
            completed=self.completed,
            started_at=self.started_at,
            message=self.message,
        )


#: Signature of a progress observer.
ProgressCallback = Callable[[ScanProgress], None]


class ScanContext:
    """Shared, thread-safe state for one scan.

    Every worker thread updates counters through this object, so all mutation is guarded
    by a single lock and observers receive consistent snapshots.
    """

    def __init__(
        self,
        *,
        on_progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        emit_interval: float = 0.1,
    ) -> None:
        """Initialise the context.

        Args:
            on_progress: Observer invoked with progress snapshots.
            cancel_event: Externally owned cancellation flag; created if omitted.
            emit_interval: Minimum seconds between callbacks. Throttling matters — a
                walk discovering thousands of files a second would otherwise spend more
                time repainting a progress bar than reading the disk.
        """
        self._lock = threading.Lock()
        self._progress = ScanProgress()
        self._on_progress = on_progress
        self._cancel_event = cancel_event or threading.Event()
        self._emit_interval = emit_interval
        self._last_emit = 0.0

    @property
    def cancel_event(self) -> threading.Event:
        """Return the cancellation flag, for handing to hashing and walking helpers."""
        return self._cancel_event

    @property
    def cancelled(self) -> bool:
        """Report whether cancellation has been requested."""
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Request cancellation. Workers stop at their next checkpoint."""
        self._cancel_event.set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`ScanCancelled` if cancellation has been requested."""
        if self._cancel_event.is_set():
            raise ScanCancelled

    def snapshot(self) -> ScanProgress:
        """Return a consistent copy of current progress."""
        with self._lock:
            return self._progress.snapshot()

    def set_phase(self, phase: ScanPhase, *, total: int = 0, message: str = "") -> None:
        """Move to a new phase, resetting per-phase counters."""
        with self._lock:
            self._progress.phase = phase
            self._progress.total = total
            self._progress.completed = 0
            if message:
                self._progress.message = message
        self._emit(force=True)

    def update(self, **fields: object) -> None:
        """Set progress fields absolutely.

        Unknown field names are ignored rather than raising, so an instrumentation call
        added in one place cannot crash a scan.
        """
        with self._lock:
            for key, value in fields.items():
                if hasattr(self._progress, key):
                    setattr(self._progress, key, value)
        self._emit()

    def increment(self, **fields: int) -> None:
        """Add to numeric progress counters."""
        with self._lock:
            for key, delta in fields.items():
                current = getattr(self._progress, key, None)
                if isinstance(current, int):
                    setattr(self._progress, key, current + delta)
        self._emit()

    def _emit(self, *, force: bool = False) -> None:
        """Invoke the observer, throttled unless ``force`` is set."""
        if self._on_progress is None:
            return
        now = time.monotonic()
        if not force and now - self._last_emit < self._emit_interval:
            return
        self._last_emit = now
        snapshot = self.snapshot()
        try:
            self._on_progress(snapshot)
        except Exception:
            pass


class ScanCancelled(RuntimeError):
    """Raised inside worker threads once cancellation has been requested."""
