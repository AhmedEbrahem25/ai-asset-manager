"""Category lookups over the active taxonomy.

Once the home of a hard-coded list of categories; now a thin set of lookups over the
plugin registry. Kept as a module because the engine, the renderer, the exporters and the
CLI all ask the same handful of questions about a category id, and routing them through
one place means the registry is fetched once and the call sites stay readable.

Category, section, task and domain ids are plain strings throughout. That is what lets a
plugin introduce one without an enum to extend, a migration to write or a core file to
edit — and it is why an id from a plugin that is no longer installed still renders under a
readable label rather than crashing the report.
"""

from __future__ import annotations

from ai_asset_manager.backend.taxonomy import default_registry
from ai_asset_manager.backend.taxonomy.types import Category, Section

#: Section a category falls back to when its plugin is not installed.
FALLBACK_SECTION = "other"


def category_info(category_id: str) -> Category:
    """Return the full descriptor for a category id."""
    return default_registry().category(category_id)


def label_of(category_id: str) -> str:
    """Return a category's display label.

    Examples:
        >>> label_of("llm")
        'LLM'
        >>> label_of("ocr")
        'OCR Model'
    """
    return default_registry().label_of(category_id)


def section_of(category_id: str) -> str:
    """Return the id of the section a category belongs to."""
    return default_registry().section_of(category_id)


def order_of(category_id: str) -> int:
    """Return a category's display sort position."""
    return default_registry().order_of(category_id)


def section_info(section_id: str) -> Section:
    """Return the full descriptor for a section id."""
    return default_registry().section(section_id)


def section_label(section_id: str) -> str:
    """Return a section's display label."""
    return default_registry().section(section_id).label


def section_order(section_id: str) -> int:
    """Return a section's display sort position."""
    return default_registry().section(section_id).order


def task_label(task_id: str | None) -> str:
    """Return a task's display label, or an empty string when there is no task."""
    if not task_id:
        return ""
    return default_registry().task(task_id).label


def domain_label(domain_id: str | None) -> str:
    """Return a domain's display label, or an empty string when there is no domain."""
    if not domain_id:
        return ""
    return default_registry().domain(domain_id).label


def categories_in_section(section_id: str) -> list[str]:
    """Return every category in a section, in display order."""
    return [category.id for category in default_registry().categories(section=section_id)]


def resolve_alias(value: str) -> tuple[str, ...] | None:
    """Resolve a user-typed selector to category ids.

    Accepts an alias, a section name, a domain name, a bare category id or ``all``.
    Returns ``None`` for anything unrecognised so the caller can list the valid options.

    Examples:
        >>> resolve_alias("llm")
        ('llm',)
        >>> resolve_alias("nonsense") is None
        True
    """
    return default_registry().resolve_alias(value)


def known_aliases() -> list[str]:
    """Return every accepted selector, for help text and error messages."""
    return default_registry().known_aliases()


def all_categories() -> list[Category]:
    """Return every registered category in display order."""
    return default_registry().categories()


def all_sections() -> list[Section]:
    """Return every registered section in display order."""
    return default_registry().sections()
