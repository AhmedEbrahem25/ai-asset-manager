"""Structural vocabulary and the rules that apply to every asset.

Registers the sections, domains and modalities the other plugins refer to, the statistics
every asset has, the health rules that are not specific to any domain, and the fallback
classifier of last resort.

Nothing here is privileged. A plugin may register a section or domain of its own, and
none of these ids is referenced by name from the core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import INCOMPLETE_SUFFIXES
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Domain,
    Finding,
    Modality,
    Section,
)

#: Sections, in the order a report reads.
SECTIONS = (
    Section(id="models", label="Models", order=10),
    Section(id="datasets", label="Datasets", order=20),
    Section(id="experiments", label="Experiments & Logs", order=30),
    Section(id="documents", label="Documents", order=40),
    Section(id="other", label="Other", order=90),
)

#: Fields of AI work. Categories and tasks name one of these; a domain that no plugin
#: declares still works, so this list constrains nothing.
DOMAINS = (
    Domain(id="vision", label="Computer Vision", order=10),
    Domain(id="nlp", label="Natural Language Processing", order=20),
    Domain(id="speech", label="Speech", order=30),
    Domain(id="audio", label="Audio", order=40),
    Domain(id="multimodal", label="Multimodal", order=50),
    Domain(id="document_ai", label="Document AI", order=60),
    Domain(id="generative", label="Generative", order=70),
    Domain(id="autonomous_driving", label="Autonomous Driving", order=80),
    Domain(id="robotics", label="Robotics", order=90),
    Domain(id="three_d", label="3D Vision", order=100),
    Domain(id="medical", label="Medical Imaging", order=110),
    Domain(id="remote_sensing", label="Remote Sensing", order=120),
    Domain(id="timeseries", label="Time Series", order=130),
    Domain(id="tabular", label="Tabular", order=140),
    Domain(id="reinforcement_learning", label="Reinforcement Learning", order=150),
    Domain(id="scientific", label="Scientific Computing", order=160),
    Domain(id="synthetic", label="Synthetic Data", order=170),
    Domain(id="mlops", label="MLOps", order=180),
    Domain(id="general", label="General", order=900),
)

#: Signals an asset may carry.
MODALITIES = (
    Modality(id="rgb", label="RGB", order=10),
    Modality(id="depth", label="Depth", order=20),
    Modality(id="thermal", label="Thermal", order=30),
    Modality(id="infrared", label="Infrared", order=40),
    Modality(id="lidar", label="LiDAR", order=50),
    Modality(id="radar", label="Radar", order=60),
    Modality(id="point_cloud", label="Point Cloud", order=70),
    Modality(id="video", label="Video", order=80),
    Modality(id="audio", label="Audio", order=90),
    Modality(id="text", label="Text", order=100),
    Modality(id="document", label="Document", order=110),
    Modality(id="tabular", label="Tabular", order=120),
    Modality(id="sensor_fusion", label="Sensor Fusion", order=130),
    Modality(id="multimodal", label="Multimodal", order=140),
)


def register(registry: TaxonomyRegistry) -> None:
    """Populate the registry with the structural vocabulary and universal rules."""
    for section in SECTIONS:
        registry.add_section(section)
    for domain in DOMAINS:
        registry.add_domain(domain)
    for modality in MODALITIES:
        registry.add_modality(modality)

    registry.add_category(
        Category(
            id="unclassified",
            label="Unclassified",
            section="other",
            order=999,
            aliases=("unknown", "other"),
            description="Catalogued, but no plugin recognised what it is for.",
        )
    )

    registry.add_classifier(_fallback, name="fallback", priority=-1000)
    registry.add_statistic(_storage_statistics, name="storage")
    registry.add_health_rule(_missing_from_disk, name="asset.missing")
    registry.add_health_rule(_empty_asset, name="asset.empty")
    registry.add_health_rule(_interrupted_download, name="asset.incomplete_download")


def _fallback(profile: AssetProfile) -> Classification:
    """Classify anything no other plugin claimed.

    Registered at the bottom of the priority order so it only ever sees assets every other
    rule declined. It always answers, which is what guarantees the inventory lists
    everything the scanner found rather than quietly dropping the unfamiliar.
    """
    return Classification(
        category="unclassified",
        domain="general",
        confidence=CONFIDENCE_WEAK,
        evidence=f"catalogued as {profile.kind}",
    )


def _storage_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return the statistics every asset has, whatever it is."""
    stats: dict[str, Any] = {
        "files": profile.file_count,
        "size_bytes": profile.size_bytes,
    }

    if profile.physical_size_bytes and profile.physical_size_bytes != profile.size_bytes:
        # Diverges in linked caches, where several assets share one physical copy. Worth
        # showing, because it is the difference between what a folder claims to weigh and
        # what deleting it would actually free.
        stats["on_disk_bytes"] = profile.physical_size_bytes

    if profile.files.loaded and profile.files.by_extension:
        ranked = sorted(
            profile.files.by_extension.items(), key=lambda pair: (-pair[1], pair[0])
        )
        stats["top_extensions"] = dict(ranked[:5])

    return stats


def _missing_from_disk(profile: AssetProfile) -> Sequence[Finding]:
    """Report an asset the last scan could not find."""
    if not profile.is_missing:
        return ()
    return (
        Finding(
            code="asset.missing",
            severity=Severity.ERROR,
            message="No longer present at its recorded location",
            fix_hint="Rescan to confirm, or remove it from the catalogue.",
        ),
    )


def _empty_asset(profile: AssetProfile) -> Sequence[Finding]:
    """Report an asset with no files or no bytes."""
    findings: list[Finding] = []

    if profile.file_count == 0:
        findings.append(
            Finding(
                code="asset.empty",
                severity=Severity.ERROR,
                message="Contains no files",
                fix_hint="The download probably never started. Fetch it again.",
            )
        )
    elif profile.size_bytes == 0:
        findings.append(
            Finding(
                code="asset.zero_bytes",
                severity=Severity.ERROR,
                message=f"All {profile.file_count} file(s) are zero bytes",
                fix_hint="Placeholder files only. Fetch the real content.",
            )
        )

    return findings


def _interrupted_download(profile: AssetProfile) -> Sequence[Finding]:
    """Report the residue an interrupted download leaves behind.

    A partial file means the asset on disk is not the asset the catalogue describes — the
    weights are short, and loading it will fail in a way that looks like a corrupt model
    rather than an unfinished transfer.
    """
    if not profile.files.loaded:
        return ()

    partial = [
        name for name in profile.files.names if name.endswith(INCOMPLETE_SUFFIXES)
    ]
    if not partial:
        return ()

    return (
        Finding(
            code="asset.incomplete_download",
            severity=Severity.ERROR,
            message=f"{len(partial)} unfinished download file(s), e.g. {sorted(partial)[0]}",
            fix_hint="Resume or restart the download, then rescan.",
        ),
    )
