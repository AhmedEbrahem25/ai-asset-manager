"""Small facts about this installation that outlive a single command.

Has discovery run? When did the last automatic scan happen? Is a watcher alive, and if so
which process is it? None of this belongs in the catalogue: it describes the installation
rather than the assets, it must be readable before a database exists, and losing it should
cost a user nothing more than one repeated prompt.

So it is a JSON file, written atomically. Two processes racing on it — a watcher and a CLI
command — is expected, which is why every write is a whole-file replace onto a temporary
file followed by a rename, and why a corrupt or unreadable file is treated as an empty one
rather than as an error. Nothing here is important enough to fail a command over.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_asset_manager.config import get_settings
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Filename inside the data directory.
STATE_FILENAME = "state.json"


def _now() -> str:
    """Return the current time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class AppState:
    """What this installation remembers between commands."""

    #: Set once the user has been shown discovery results and answered, either way.
    #: Answering "no" still counts: being asked the same question at every launch is
    #: worse than never being asked at all.
    discovery_completed: bool = False
    discovery_completed_at: str | None = None
    #: Locations the user declined, so a later discovery can stop offering them.
    declined_paths: list[str] = field(default_factory=list)

    #: When an automatic incremental scan last ran, used to rate-limit them.
    last_auto_scan_at: str | None = None

    #: The live watcher, when one is running.
    watcher_pid: int | None = None
    watcher_started_at: str | None = None
    watcher_roots: list[str] = field(default_factory=list)
    #: Set by `aam watch --stop`; the watcher polls it and shuts down cleanly.
    watcher_stop_requested: bool = False

    def mark_discovery_done(self, *, declined: list[str] | None = None) -> None:
        """Record that discovery has run."""
        self.discovery_completed = True
        self.discovery_completed_at = _now()
        if declined:
            self.declined_paths = sorted(set(self.declined_paths) | set(declined))

    def mark_auto_scan(self) -> None:
        """Record that an automatic scan has just run."""
        self.last_auto_scan_at = _now()

    def seconds_since_auto_scan(self) -> float | None:
        """Return how long ago the last automatic scan was, in seconds."""
        return _age_of(self.last_auto_scan_at)

    @property
    def watcher_age_seconds(self) -> float | None:
        """Return how long the watcher has been running, in seconds."""
        return _age_of(self.watcher_started_at)


def _age_of(timestamp: str | None) -> float | None:
    """Return the age of an ISO-8601 timestamp in seconds, or ``None``."""
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (datetime.now(UTC) - moment).total_seconds()


def state_path() -> Path:
    """Return where the state file lives."""
    return get_settings().data_dir / STATE_FILENAME


def load_state(path: Path | None = None) -> AppState:
    """Read the state file, returning defaults when it is absent or unreadable."""
    target = path or state_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppState()

    if not isinstance(raw, dict):
        return AppState()

    # Unknown keys are dropped rather than raising: a state file written by a newer
    # version must not stop an older one from running.
    known = {entry.name for entry in fields(AppState)}
    return AppState(**{key: value for key, value in raw.items() if key in known})


def save_state(state: AppState, path: Path | None = None) -> None:
    """Write the state file atomically.

    A half-written state file read by a concurrent command would parse as corrupt and be
    silently discarded, losing the "discovery already ran" flag and re-prompting the user.
    Writing to a temporary file and renaming makes the update indivisible.
    """
    target = path or state_path()

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=".state-", suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(asdict(state), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, target)
    except OSError as exc:
        # Losing this costs one repeated prompt. Failing the user's command over it would
        # cost them the command.
        logger.debug("Could not write %s: %s", target, exc)


def update_state(path: Path | None = None, **changes: Any) -> AppState:
    """Load, modify and save the state in one call."""
    state = load_state(path)
    for key, value in changes.items():
        setattr(state, key, value)
    save_state(state, path)
    return state


def process_is_alive(pid: int | None) -> bool:
    """Report whether a process id is currently running.

    A watcher that was killed leaves its pid in the state file, and reporting it as alive
    would make ``aam status`` lie about the thing it exists to report.
    """
    if not pid or pid <= 0:
        return False

    if os.name == "nt":
        import ctypes

        # Ask the OS directly. OpenProcess with QUERY_LIMITED_INFORMATION succeeds for a
        # live process the caller may not otherwise touch.
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Running, and owned by someone else.
        return True
    return True
