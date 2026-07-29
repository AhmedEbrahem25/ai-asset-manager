"""Live filesystem monitoring.

Keeps the catalogue in step with the disk without the user having to remember to rescan.
Events are debounced into batches, and each batch is turned into the narrowest scan that
will make the catalogue correct.
"""

from __future__ import annotations

from ai_asset_manager.backend.watch.debounce import DebounceStats, EventDebouncer
from ai_asset_manager.backend.watch.indexer import IndexerStats, IndexResult, LiveIndexer
from ai_asset_manager.backend.watch.service import (
    WatchService,
    WatchStatus,
    request_watcher_stop,
)

__all__ = [
    "DebounceStats",
    "EventDebouncer",
    "IndexResult",
    "IndexerStats",
    "LiveIndexer",
    "WatchService",
    "WatchStatus",
    "request_watcher_stop",
]
