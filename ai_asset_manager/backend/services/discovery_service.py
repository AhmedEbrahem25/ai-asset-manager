"""Deciding what to offer the user, and remembering what they said.

Kept apart from the prompting deliberately. The CLI asks the question and the future
dashboard will ask it differently, but which locations are worth offering — and which the
user has already turned down, and which are already managed — is one answer that both need
and neither should reinvent.

Nothing here scans anything. Discovery decides *where* to look; it is the user who decides
whether to look there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ai_asset_manager.backend.scanner.locations import KnownLocation, discover
from ai_asset_manager.backend.services.scan_service import ScanService
from ai_asset_manager.backend.state import AppState, load_state, save_state
from ai_asset_manager.backend.utils.paths import normalize_path
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DiscoveryReport:
    """What a discovery pass found, split by what should happen next."""

    #: Worth offering: exists, holds assets, not already managed, not previously declined.
    candidates: list[KnownLocation] = field(default_factory=list)
    #: Already registered as scan roots, or inside one.
    already_managed: list[KnownLocation] = field(default_factory=list)
    #: Offered before and turned down.
    previously_declined: list[KnownLocation] = field(default_factory=list)

    @property
    def has_candidates(self) -> bool:
        """Report whether there is anything to ask the user about."""
        return bool(self.candidates)

    @property
    def total_found(self) -> int:
        """Return how many locations exist on this machine in total."""
        return (
            len(self.candidates) + len(self.already_managed) + len(self.previously_declined)
        )


class DiscoveryService:
    """Finds asset locations and reconciles them against what is already managed."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a database session."""
        self.session = session
        self.scans = ScanService(session)

    def discover(
        self, *, include_declined: bool = False, sweep: bool = True
    ) -> DiscoveryReport:
        """Return the locations on this machine, sorted into what to do with them.

        Args:
            include_declined: Offer locations the user previously turned down. What
                ``aam discover`` passes when run explicitly, since asking again is the
                whole reason someone would run it a second time.
            sweep: Also look for likely folders near each drive root.
        """
        state = load_state()
        declined = {normalize_path(path) for path in state.declined_paths}
        managed = [root.path for root in self.scans.list_roots()]

        report = DiscoveryReport()

        for location in discover(sweep=sweep):
            path = normalize_path(str(location.path))

            if _is_managed(path, managed):
                report.already_managed.append(location)
            elif path in declined and not include_declined:
                report.previously_declined.append(location)
            else:
                report.candidates.append(location)

        logger.debug(
            "Discovery: %d candidate(s), %d already managed, %d previously declined",
            len(report.candidates), len(report.already_managed),
            len(report.previously_declined),
        )
        return report

    def accept(self, locations: list[KnownLocation]) -> list[str]:
        """Register the chosen locations as scan roots. Returns the paths added."""
        added: list[str] = []
        for location in locations:
            root = self.scans.add_root(str(location.path), label=location.label)
            added.append(root.path)
        self.session.commit()
        logger.info("Added %d scan root(s) from discovery", len(added))
        return added

    def decline(self, locations: list[KnownLocation]) -> None:
        """Remember that these locations were turned down."""
        state = load_state()
        state.declined_paths = sorted(
            set(state.declined_paths)
            | {normalize_path(str(location.path)) for location in locations}
        )
        save_state(state)

    def complete(self, *, declined: list[KnownLocation] | None = None) -> AppState:
        """Record that discovery has run, so it is not repeated at every launch."""
        state = load_state()
        state.mark_discovery_done(
            declined=[normalize_path(str(item.path)) for item in declined or []]
        )
        save_state(state)
        return state

    @staticmethod
    def should_run_on_startup() -> bool:
        """Report whether this is a first run that should offer discovery."""
        return not load_state().discovery_completed


def _is_managed(path: str, roots: list[str]) -> bool:
    """Report whether a path is a managed root or sits inside one."""
    import os

    normalised = path.replace("\\", "/")
    if os.name == "nt":
        normalised = normalised.lower()

    for root in roots:
        prepared = root.replace("\\", "/")
        if os.name == "nt":
            prepared = prepared.lower()
        if normalised == prepared or normalised.startswith(prepared.rstrip("/") + "/"):
            return True
    return False
