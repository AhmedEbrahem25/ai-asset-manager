"""Parser protocol and the fact container parsers produce.

Every parser answers the same question — "what can you tell me about this directory?" —
and returns its answers as attributed facts rather than as a finished record. Merging
those answers is one function in :mod:`ai_asset_manager.backend.metadata.merge`, which is
what keeps forty-odd parsers from each re-implementing precedence rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ai_asset_manager.backend.models.enums import FACT_SOURCE_PRIORITY, FactSource
from ai_asset_manager.backend.scanner.context import DirectoryContext


@dataclass(slots=True, frozen=True)
class Fact:
    """One attributed metadata value."""

    value: Any
    source: FactSource
    #: Parser's own certainty, 0..1. Breaks ties between facts of equal source rank.
    confidence: float = 1.0
    #: Name of the parser that produced this, for provenance display and debugging.
    origin: str = ""

    @property
    def weight(self) -> tuple[int, float]:
        """Return the sort key used to resolve competing facts."""
        return (FACT_SOURCE_PRIORITY[self.source], self.confidence)


@dataclass(slots=True)
class FactSet:
    """Facts gathered about one asset, keyed by field name.

    Multiple parsers may assert the same field; all assertions are retained so the merge
    step can pick a winner and so provenance stays inspectable.
    """

    facts: dict[str, list[Fact]] = field(default_factory=dict)
    #: Non-fatal problems noticed while parsing, promoted to health findings later.
    warnings: list[str] = field(default_factory=list)

    def add(
        self,
        key: str,
        value: Any,
        *,
        source: FactSource,
        confidence: float = 1.0,
        origin: str = "",
    ) -> None:
        """Record a fact.

        ``None`` and empty strings are dropped: a parser that found nothing should not
        outrank a lower-priority parser that found something.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        self.facts.setdefault(key, []).append(
            Fact(value=value, source=source, confidence=confidence, origin=origin)
        )

    def add_many(
        self,
        values: dict[str, Any],
        *,
        source: FactSource,
        confidence: float = 1.0,
        origin: str = "",
    ) -> None:
        """Record several facts sharing one source and confidence."""
        for key, value in values.items():
            self.add(key, value, source=source, confidence=confidence, origin=origin)

    def warn(self, message: str) -> None:
        """Record a non-fatal parsing problem."""
        self.warnings.append(message)

    def best(self, key: str) -> Fact | None:
        """Return the highest-priority fact for a field."""
        candidates = self.facts.get(key)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.weight)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the winning value for a field."""
        fact = self.best(key)
        return fact.value if fact is not None else default

    def keys(self) -> set[str]:
        """Return every field name asserted by any parser."""
        return set(self.facts)

    def merge_from(self, other: FactSet) -> None:
        """Absorb another fact set, keeping every assertion from both."""
        for key, items in other.facts.items():
            self.facts.setdefault(key, []).extend(items)
        self.warnings.extend(other.warnings)

    def provenance(self) -> dict[str, str]:
        """Return the winning source for each field, for display and debugging."""
        result: dict[str, str] = {}
        for key in self.facts:
            fact = self.best(key)
            if fact is not None:
                result[key] = fact.origin or fact.source.value
        return result

    def __len__(self) -> int:
        """Return the number of distinct fields asserted."""
        return len(self.facts)

    def __contains__(self, key: str) -> bool:
        """Report whether any parser asserted a field."""
        return key in self.facts


@runtime_checkable
class MetadataParser(Protocol):
    """Extracts facts from a directory.

    Implementations must be side-effect free and must never raise for malformed input:
    the catalogue is expected to contain broken assets, and a parser that throws on a
    truncated file would prevent that asset being catalogued at all.
    """

    name: str

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether this parser has anything to say about the directory."""
        ...

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract facts. Must return an empty set rather than raise on bad input."""
        ...


class BaseParser:
    """Convenience base implementing the boilerplate half of :class:`MetadataParser`."""

    name: str = "base"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether this parser applies. Overridden by subclasses."""
        raise NotImplementedError

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract facts. Overridden by subclasses."""
        raise NotImplementedError

    def _new_facts(self) -> FactSet:
        """Return an empty fact set."""
        return FactSet()

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"<{type(self).__name__} {self.name!r}>"
