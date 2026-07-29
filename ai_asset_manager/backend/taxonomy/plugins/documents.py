"""Research papers and documentation kept alongside the assets they describe."""

from __future__ import annotations

import re

from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: An arXiv identifier, in either the modern or the legacy form. The single most reliable
#: sign that a PDF on a machine learning practitioner's disk is a paper rather than a
#: manual or an invoice.
_ARXIV = re.compile(r"\b(\d{4}\.\d{4,5}(v\d+)?|arxiv[-_.]?\d{4})", re.IGNORECASE)


def register(registry: TaxonomyRegistry) -> None:
    """Register the documents shelf and its classifier."""
    registry.add_task(Task(id="reading", label="Reading", domain="general", order=30))

    registry.add_category(
        Category(id="paper", label="Research Paper", section="documents", order=300,
                 domain="general", aliases=("papers", "research"))
    )
    registry.add_category(
        Category(id="documentation", label="Documentation", section="documents", order=310,
                 domain="general", aliases=("docs",))
    )

    registry.add_classifier(_document, name="documents", priority=870)


def _document(profile: AssetProfile) -> Classification | None:
    """Claim papers and documentation."""
    if profile.kind == AssetKind.PAPER:
        return Classification(
            category="paper", task="reading", domain="general", modalities=("document",),
            confidence=CONFIDENCE_CERTAIN, evidence="catalogued as a paper",
        )

    if not profile.files.loaded:
        return None

    pdfs = profile.files.count(".pdf")
    if not pdfs:
        return None

    # A directory that is nothing but PDFs is a reading pile. One PDF inside a model
    # repository is that model's paper, and the repository should stay a model.
    if pdfs < max(1, profile.file_count // 2):
        return None

    arxiv = any(_ARXIV.search(name) for name in profile.files.names)
    return Classification(
        category="paper" if arxiv else "documentation", task="reading", domain="general",
        modalities=("document",), confidence=CONFIDENCE_STRONG,
        evidence=f"{pdfs} PDF(s)" + (" with arXiv identifiers" if arxiv else ""),
    )
