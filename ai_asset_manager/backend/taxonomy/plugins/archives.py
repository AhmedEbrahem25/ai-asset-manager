"""Archives.

An archive is filed under the section its contents belong to — a packed model appears
among models, a packed dataset among datasets — because that is where someone taking stock
would look for it. What it is *not* is merged with the unpacked thing: its label says
"Archive" throughout, and its statistics report members listed rather than files owned,
because nothing was unpacked and the catalogue must not imply otherwise.

The classification itself was done by
:mod:`ai_asset_manager.backend.archives.classify` at scan time, from the table of contents.
This plugin only decides where the verdict is shelved and how it reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.archives.reader import NO_READER_PREFIX, is_missing_reader
from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Finding,
    Task,
)

#: The subkinds the archive classifier emits, mapped to the shelf and task each belongs to.
SUBKIND_CATEGORIES: Mapping[str, tuple[str, str | None]] = {
    "model_archive": ("model_archive", "packaging"),
    "adapter_archive": ("model_archive", "packaging"),
    "yolo_dataset_archive": ("dataset_archive", "packaging"),
    "coco_dataset_archive": ("dataset_archive", "packaging"),
    "hf_dataset_archive": ("dataset_archive", "packaging"),
    "tracking_dataset_archive": ("dataset_archive", "packaging"),
    "image_dataset_archive": ("dataset_archive", "packaging"),
    "video_dataset_archive": ("dataset_archive", "packaging"),
    "audio_dataset_archive": ("dataset_archive", "packaging"),
    "tabular_dataset_archive": ("dataset_archive", "packaging"),
    # Media with no split, no labels and no class list. Catalogued and sized, but filed
    # away from the datasets shelf, because a course download is not a corpus.
    "media_archive": ("media_archive", "packaging"),
    "network_dataset_archive": ("security_archive", "packaging"),
    "host_log_dataset_archive": ("security_archive", "packaging"),
    "malware_dataset_archive": ("security_archive", "packaging"),
    "training_archive": ("training_archive", "packaging"),
    "code_archive": ("code_archive", "packaging"),
    "archive": ("unknown_archive", None),
}

CATEGORIES = (
    Category(id="model_archive", label="Model Archive", section="models", order=195,
             domain="general", aliases=("model-archives", "packed-models"),
             description="A packed model, catalogued from its table of contents."),
    Category(id="dataset_archive", label="Dataset Archive", section="datasets", order=295,
             domain="general", aliases=("dataset-archives", "packed-datasets"),
             description="A packed dataset, catalogued from its table of contents."),
    Category(id="security_archive", label="Security Dataset Archive", section="datasets",
             order=296, domain="security", aliases=("security-archives",),
             description="A packed capture, log or malware corpus."),
    Category(id="training_archive", label="Training Run Archive", section="experiments",
             order=395, domain="mlops", aliases=("run-archives", "packed-runs"),
             description="A packed training run: events, checkpoints and run metadata."),
    Category(id="code_archive", label="Code Archive", section="other", order=880,
             domain="general", aliases=("code-archives",),
             description="A packed codebase."),
    Category(id="media_archive", label="Media Archive", section="other", order=885,
             domain="general", aliases=("media-archives",),
             description="Packed images, video or audio with nothing marking it a dataset."),
    Category(id="unknown_archive", label="Archive", section="other", order=890,
             domain="general", aliases=("archives", "zips"),
             description="An archive whose contents no rule recognised."),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the archive shelves, their classifier and their rules."""
    registry.add_task(Task(id="packaging", label="Packaged / Archived", order=900))
    for category in CATEGORIES:
        registry.add_category(category)

    registry.add_alias(
        "archives",
        (
            "model_archive", "dataset_archive", "security_archive",
            "training_archive", "code_archive", "media_archive", "unknown_archive",
        ),
    )

    # Structural band: the scanner already read the table of contents, and no rule
    # reasoning from names should be allowed to overrule it.
    registry.add_classifier(_archive, name="archive", priority=950)
    registry.add_statistic(_archive_statistics, name="archive")
    registry.add_health_rule(_unreadable_archive, name="archive.unreadable")


def _is_archive(profile: AssetProfile) -> bool:
    """Report whether an asset was catalogued as an archive."""
    return profile.kind == "archive"


def _archive(profile: AssetProfile) -> Classification | None:
    """Shelve an archive according to the verdict the scanner recorded."""
    if not _is_archive(profile):
        return None

    category, task = SUBKIND_CATEGORIES.get(
        profile.subkind or "archive", ("unknown_archive", None)
    )
    signals = _signals(profile)

    return Classification(
        category=category,
        task=task,
        domain="security" if category == "security_archive" else "general",
        confidence=CONFIDENCE_CERTAIN if signals else CONFIDENCE_WEAK,
        evidence="; ".join(signals[:3]) if signals else "archive contents not listed",
    )


def _archive_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return what the listing said, phrased so it cannot be mistaken for an unpack."""
    if not _is_archive(profile):
        return {}

    stats: dict[str, Any] = {"extracted": False}
    evidence = _evidence(profile)

    for source, key in (
        ("archive_format", "archive_format"),
        ("members_listed", "members_listed"),
        ("archive_label", "archive_kind"),
        ("images_listed", "images_listed"),
        ("shards_listed", "shards_listed"),
        ("tables_listed", "tables_listed"),
    ):
        value = evidence.get(source)
        if value not in (None, 0, ""):
            stats[key] = value

    if evidence.get("listing_truncated"):
        stats["listing_truncated"] = True
    if evidence.get("metadata_read"):
        stats["metadata_read_in_memory"] = evidence["metadata_read"]

    return stats


def _unreadable_archive(profile: AssetProfile) -> Sequence[Finding]:
    """Report an archive whose table of contents could not be read.

    Worth saying out loud rather than leaving as a silent low-confidence row: an archive
    nobody can list is encrypted, damaged, in a format whose optional reader is not
    installed, or in a format this build cannot read at all — and which of those it is
    changes what the user should do about it, so the hint distinguishes them.
    """
    if not _is_archive(profile):
        return ()

    error = _evidence(profile).get("listing_error")
    if not error:
        return ()

    return (
        Finding(
            code="archive.not_listed",
            severity=Severity.WARNING,
            message=f"Contents could not be listed ({error})",
            fix_hint=_listing_hint(str(error)),
        ),
    )


def _listing_hint(error: str) -> str:
    """Return advice matching why the listing failed.

    Three distinct causes wear the same symptom, and only one of them is the user's to
    fix. Telling the owner of a perfectly good msys2 package that it may be damaged is
    worse than saying nothing.
    """
    if error.lower().startswith(NO_READER_PREFIX):
        return (
            "This container format has no reader in this build; the archive is "
            "catalogued from its name and size."
        )
    if is_missing_reader(error):
        return "Install the optional readers: pip install 'ai-asset-manager[archives]'."
    return "The archive may be encrypted or damaged; try opening it manually."


def _evidence(profile: AssetProfile) -> Mapping[str, Any]:
    """Return the archive observations the scanner recorded.

    Archives carry neither model nor dataset details — nothing was unpacked to fill them —
    so the detector's own evidence is the only record there is.
    """
    return profile.evidence or {}


def _signals(profile: AssetProfile) -> list[str]:
    """Return the evidence the archive classifier matched on."""
    raw = _evidence(profile).get("signals")
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item]
    return []
