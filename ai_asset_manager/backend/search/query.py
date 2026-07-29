"""The search query language.

One input box handles both free text and structured filters::

    llama                          qwen quant:Q4_K_M
    coco type:dataset              yolov12 tag:production
    type:lora framework:peft       size:>10GB drive:F year:2025
    segmentation license:apache-2.0 -archived

Bare words become a full-text match; ``key:value`` becomes a structured filter; a leading
``-`` negates. Parsing is deliberately forgiving — an unrecognised key is treated as free
text rather than rejected, because a search box that answers "syntax error" to a search
for ``ratio:1.5`` is worse than one that just looks for those characters.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from ai_asset_manager.backend.services.asset_service import AssetFilter
from ai_asset_manager.backend.utils.humanize import parse_size

#: Filter keys, mapped to the :class:`AssetFilter` list field they populate. Aliases are
#: generous on purpose: users type `fmt`, `format`, `type` and `kind` interchangeably.
LIST_FIELD_ALIASES: dict[str, str] = {
    "kind": "kinds",
    "type": "model_types",
    "modeltype": "model_types",
    "model_type": "model_types",
    "dataset": "dataset_formats",
    "datasetformat": "dataset_formats",
    "dataset_format": "dataset_formats",
    "format": "formats",
    "fmt": "formats",
    "framework": "frameworks",
    "fw": "frameworks",
    "lib": "frameworks",
    "library": "frameworks",
    "drive": "drives",
    "disk": "drives",
    "tag": "tags",
    "author": "authors",
    "org": "authors",
    "license": "licenses",
    "lic": "licenses",
    "quant": "quantizations",
    "quantization": "quantizations",
    "q": "quantizations",
}

#: Keys that take a size, with an optional comparison prefix.
SIZE_KEYS = frozenset({"size", "bytes"})
MIN_SIZE_KEYS = frozenset({"minsize", "min_size", "larger", "bigger"})
MAX_SIZE_KEYS = frozenset({"maxsize", "max_size", "smaller"})

#: Values of ``kind:`` that are really asset kinds rather than model types. Without this,
#: ``type:dataset`` would filter on a model type that does not exist and return nothing.
KIND_VALUES = frozenset({"model", "dataset", "adapter", "checkpoint", "paper", "unknown"})

#: ``size:>10GB``, ``size:<=500MB``, ``size:1TB``.
_COMPARISON_RE = re.compile(r"^(?P<op>>=|<=|>|<|=)?\s*(?P<value>.+)$")

#: A bare ``key:value`` token. Values may be quoted to contain spaces.
_TERM_RE = re.compile(r"^(?P<negate>-)?(?P<key>[a-zA-Z_][a-zA-Z0-9_]*):(?P<value>.*)$")


@dataclass(slots=True)
class ParsedQuery:
    """The result of parsing a query string."""

    #: Free-text terms, joined for the full-text match.
    text: str = ""
    filters: AssetFilter = field(default_factory=AssetFilter)
    #: Filter keys the user negated, e.g. ``-tag:archived``.
    excluded_tags: list[str] = field(default_factory=list)
    #: Keys that were not recognised, kept so the UI can hint at a typo.
    unknown_keys: list[str] = field(default_factory=list)
    #: True when nothing at all was supplied.
    is_empty: bool = False

    @property
    def has_text(self) -> bool:
        """Report whether a free-text term was given."""
        return bool(self.text.strip())


def parse_query(raw: str) -> ParsedQuery:
    """Parse a search string into free text plus structured filters.

    Args:
        raw: The user's input.

    Returns:
        A :class:`ParsedQuery`. Never raises: malformed input degrades to free text.

    Examples:
        >>> parse_query("llama type:llm size:>1GB").filters.model_types
        ['llm']
        >>> parse_query("llama type:llm").text
        'llama'
        >>> parse_query("type:dataset").filters.kinds
        ['dataset']
    """
    parsed = ParsedQuery()
    if not raw or not raw.strip():
        parsed.is_empty = True
        return parsed

    try:
        # posix=False keeps Windows backslashes intact: a user pasting D:\Models should
        # not have their path silently mangled into DModels by escape processing.
        tokens = shlex.split(raw, posix=False)
    except ValueError:
        tokens = raw.split()

    text_terms: list[str] = []

    for token in tokens:
        cleaned = token.strip().strip('"')
        if not cleaned:
            continue

        match = _TERM_RE.match(cleaned)
        if match is None:
            text_terms.append(cleaned)
            continue

        key = match.group("key").lower()
        value = match.group("value").strip().strip('"')
        negated = bool(match.group("negate"))

        if not value:
            continue

        if not _apply_term(parsed, key, value, negated=negated):
            # Not a recognised filter; treat the whole token as text so a search for
            # something like "ratio:1.5" still finds it.
            parsed.unknown_keys.append(key)
            text_terms.append(cleaned)

    parsed.text = " ".join(text_terms)
    parsed.filters.text = parsed.text or None
    parsed.is_empty = not parsed.text and _filters_are_empty(parsed.filters)
    return parsed


def _apply_term(parsed: ParsedQuery, key: str, value: str, *, negated: bool) -> bool:
    """Apply one ``key:value`` term. Returns whether the key was recognised."""
    filters = parsed.filters

    if key in LIST_FIELD_ALIASES:
        target = LIST_FIELD_ALIASES[key]

        # `type:dataset` means the asset kind, not a model type. Routing it by value
        # rather than by key is what makes the obvious query work.
        if target == "model_types" and value.lower() in KIND_VALUES:
            target = "kinds"

        if negated:
            if target == "tags":
                parsed.excluded_tags.append(value)
                return True
            # Negation is only meaningful for tags today; ignore it elsewhere rather
            # than silently inverting a filter the user did not ask to invert.
            return True

        current: list[str] = getattr(filters, target)
        current.append(_normalise_value(target, value))
        return True

    if key in SIZE_KEYS:
        return _apply_size(filters, value)

    if key in MIN_SIZE_KEYS:
        return _set_size(filters, "min_size", value)

    if key in MAX_SIZE_KEYS:
        return _set_size(filters, "max_size", value)

    if key == "year":
        if value.isdigit() and 1970 <= int(value) <= 2999:
            filters.year = int(value)
            return True
        return False

    if key in ("health", "status"):
        filters.health_status = value.lower()
        return True

    if key in ("missing", "include_missing"):
        filters.include_missing = value.lower() in ("1", "true", "yes", "on")
        return True

    return False


def _normalise_value(target: str, value: str) -> str:
    """Canonicalise a filter value to match how it is stored."""
    if target == "drives":
        # Users type `drive:F`, `drive:f:` or `drive:F:`; storage holds `F:`.
        upper = value.upper()
        return upper if upper.endswith(":") else f"{upper}:"
    if target in ("kinds", "model_types", "dataset_formats", "formats", "frameworks"):
        return value.lower().replace("-", "_")
    return value


def _apply_size(filters: AssetFilter, value: str) -> bool:
    """Apply a ``size:`` term, honouring a comparison prefix."""
    match = _COMPARISON_RE.match(value)
    if match is None:
        return False

    operator = match.group("op") or ">="
    try:
        size = parse_size(match.group("value"))
    except ValueError:
        return False

    if operator in (">", ">="):
        filters.min_size = size
    elif operator in ("<", "<="):
        filters.max_size = size
    else:
        # An exact size is expressed as a narrow band; storage sizes are exact integers
        # so a 1% tolerance keeps `size:=7GB` useful rather than pedantic.
        filters.min_size = int(size * 0.99)
        filters.max_size = int(size * 1.01)
    return True


def _set_size(filters: AssetFilter, attribute: str, value: str) -> bool:
    """Set an explicit min/max size bound."""
    try:
        setattr(filters, attribute, parse_size(value))
    except ValueError:
        return False
    return True


def _filters_are_empty(filters: AssetFilter) -> bool:
    """Report whether a filter set imposes no restriction at all."""
    for name in (
        "kinds", "model_types", "dataset_formats", "formats", "frameworks",
        "drives", "tags", "authors", "licenses", "quantizations",
    ):
        if getattr(filters, name):
            return False
    return all(
        getattr(filters, name) is None
        for name in ("min_size", "max_size", "year", "health_status")
    )


def build_fts_match(text: str) -> str:
    """Turn free text into an FTS5 MATCH expression.

    Each term becomes a prefix query so that typing ``llam`` finds ``llama`` — the
    behaviour users expect from a search box that filters as they type.

    FTS5 operators are stripped rather than escaped. A user typing ``qwen-2.5`` means a
    literal hyphen, not the NOT operator, and letting it through would silently return
    the wrong rows instead of an error.

    Examples:
        >>> build_fts_match("llama")
        '"llama"*'
        >>> build_fts_match("qwen 2.5")
        '"qwen"* AND "2.5"*'
        >>> build_fts_match("")
        ''
    """
    terms: list[str] = []
    for raw_term in text.split():
        # Keep only characters that cannot be read as FTS syntax, then re-quote.
        cleaned = re.sub(r'[^\w.\-/+]', " ", raw_term, flags=re.UNICODE).strip()
        for part in cleaned.split():
            escaped = part.replace('"', "")
            if escaped:
                terms.append(f'"{escaped}"*')
    return " AND ".join(terms)


def describe(parsed: ParsedQuery) -> dict[str, Any]:
    """Summarise a parsed query for display and debugging."""
    active: dict[str, Any] = {}
    for name in (
        "kinds", "model_types", "dataset_formats", "formats", "frameworks",
        "drives", "tags", "authors", "licenses", "quantizations",
        "min_size", "max_size", "year", "health_status",
    ):
        value = getattr(parsed.filters, name)
        if value:
            active[name] = value
    return {"text": parsed.text, "filters": active, "unknown_keys": parsed.unknown_keys}
