"""The watchdog event handler.

Its only job is to decide what is worth telling the debouncer about, and to do that
cheaply: this runs on watchdog's observer thread, and anything slow here backs up the OS
event queue and starts losing events.

So there is no database access, no path resolution and no filesystem call. Just a string
check against the same prune list the walker uses, which is what keeps ``.git`` churn and
``__pycache__`` writes from waking the indexer.
"""

from __future__ import annotations

from collections.abc import Iterable

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from ai_asset_manager.backend.watch.debounce import EventDebouncer
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Filename suffixes that are pure noise: editors, downloads in progress and databases
#: writing their journals. The *directory* still gets reported, because a partial file
#: appearing is evidence the folder is being written to - it is the file itself that is
#: not worth naming.
_NOISE_SUFFIXES = (".swp", ".swx", "~", ".db-journal", ".db-wal", ".db-shm")


class ChangeHandler(FileSystemEventHandler):
    """Feeds interesting filesystem events to a debouncer."""

    def __init__(self, debouncer: EventDebouncer, *, excluded: Iterable[str] = ()) -> None:
        """Create a handler.

        Args:
            debouncer: Where accepted paths are sent.
            excluded: Directory names to ignore, matched anywhere in the path. The
                walker's prune list, so that what is watched matches what is scanned.
        """
        super().__init__()
        self._debouncer = debouncer
        self._excluded = frozenset(name.lower() for name in excluded)

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Record any event that is not obviously noise."""
        paths = [event.src_path]
        # A move is two facts: something left one place and arrived at another. Both
        # directories need re-examining, and reporting only the destination is how a moved
        # model ends up catalogued twice.
        destination = getattr(event, "dest_path", None)
        if destination:
            paths.append(destination)

        accepted = [
            self._as_text(path)
            for path in paths
            if path and not self._is_ignored(self._as_text(path))
        ]
        if accepted:
            self._debouncer.add_many(accepted)

    @staticmethod
    def _as_text(path: str | bytes) -> str:
        """Return a path as text; watchdog reports bytes for some backends."""
        return path.decode("utf-8", "replace") if isinstance(path, bytes) else path

    def _is_ignored(self, path: str) -> bool:
        """Report whether a path is not worth waking the indexer for."""
        lowered = path.replace("\\", "/").lower()

        if lowered.endswith(_NOISE_SUFFIXES):
            return True

        # Segment-wise rather than substring: a folder legitimately called "cached-models"
        # must not be pruned because "cache" appears inside its name.
        return any(segment in self._excluded for segment in lowered.split("/"))
