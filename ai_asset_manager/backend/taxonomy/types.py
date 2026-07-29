"""The vocabulary of the taxonomy engine.

Everything here is *machinery*. Not one dataset format, model family or task name is
mentioned in this module — those all arrive from plugins. That is the whole point: a new
AI domain should cost one new file under :mod:`ai_asset_manager.backend.taxonomy.plugins`
and no edit anywhere else.

The two halves are:

* **Descriptors** — :class:`Section`, :class:`Domain`, :class:`Task`, :class:`Modality`
  and :class:`Category`. Value objects with string ids, registered at import time.
* **Evidence and verdicts** — :class:`AssetProfile` is everything the catalogue recorded
  about one asset; :class:`Classification`, :class:`Finding` and the statistic mapping are
  what plugins derive from it.

An :class:`AssetProfile` is assembled entirely from database rows. Plugins are handed one
and can do nothing else: they have no session, no path handling and no I/O. A plugin
therefore *cannot* rescan, which is how the read-only guarantee survives contributions
from code this project has never seen.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_asset_manager.backend.models.enums import Severity

#: Confidence returned by a classifier that matched on strong, unambiguous evidence — a
#: declared architecture, a format that only one kind of asset uses.
CONFIDENCE_CERTAIN = 1.0

#: Matched on a good but defeasible signal, such as a declared task or pipeline tag.
CONFIDENCE_STRONG = 0.8

#: Matched on a name or a family marker. Right most of the time, and the only thing
#: available for bare weight files that ship without configuration.
CONFIDENCE_WEAK = 0.5


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Section:
    """A top-level division of the inventory, such as models or datasets."""

    id: str
    label: str
    order: int = 100


@dataclass(frozen=True, slots=True)
class Domain:
    """A field of AI work: computer vision, speech, medical imaging, robotics.

    Orthogonal to :class:`Category`. A detection *model* and a detection *dataset* land in
    different categories and different sections but share the vision domain, which is what
    makes "show me everything vision" answerable.
    """

    id: str
    label: str
    order: int = 100


@dataclass(frozen=True, slots=True)
class Task:
    """What an asset is *for*: object detection, OCR, speech recognition.

    The single most useful field in the whole inventory, and the one the catalogue is
    least likely to have been told directly — most of the time it is inferred.
    """

    id: str
    label: str
    domain: str | None = None
    order: int = 100


@dataclass(frozen=True, slots=True)
class Modality:
    """A kind of signal an asset carries: RGB, LiDAR, audio, text."""

    id: str
    label: str
    order: int = 100


@dataclass(frozen=True, slots=True)
class Category:
    """A shelf in the inventory.

    Categories are what a person names when taking stock — "OCR models", "detection
    datasets". Several :class:`Task` values may share one, and a category may exist for
    assets whose task is unknown.
    """

    id: str
    label: str
    section: str
    order: int = 500
    domain: str | None = None
    #: Names the user may type on the command line to select this category.
    aliases: tuple[str, ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """What the catalogue recorded about a model, verbatim."""

    model_type: str | None = None
    architecture: str | None = None
    param_count: int | None = None
    param_count_is_exact: bool = False
    quantization: str | None = None
    precision: str | None = None
    context_length: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    tensor_count: int | None = None
    repo_id: str | None = None
    revision: str | None = None
    author: str | None = None
    license: str | None = None
    description: str | None = None
    base_model: str | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    card_tags: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetFacts:
    """What the catalogue recorded about a dataset, verbatim."""

    dataset_format: str | None = None
    task: str | None = None
    num_images: int = 0
    num_videos: int = 0
    num_audio_files: int = 0
    num_text_files: int = 0
    num_annotations: int = 0
    num_classes: int | None = None
    class_names: tuple[str, ...] = ()
    splits: Mapping[str, int] = field(default_factory=dict)
    modalities: tuple[str, ...] = ()
    has_bounding_boxes: bool = False
    has_masks: bool = False
    has_keypoints: bool = False
    has_lidar: bool = False
    has_radar: bool = False
    has_depth: bool = False
    has_thermal: bool = False
    repo_id: str | None = None
    license: str | None = None
    description: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FileSummary:
    """The asset's file list, as recorded by the scanner and reduced to what plugins ask.

    This is the reason dataset health can be judged without touching a disk. The scanner
    wrote down every file's relative path, extension and size; "is there a README?",
    "is there a validation split?" and "how many label files?" are all questions about
    that recorded list, not about the filesystem.

    ``loaded`` is false when the caller asked for a cheap listing and the file rows were
    never fetched. Rules must check it rather than mistake "not looked at" for "not there"
    — a health report that invents findings from absent data is worse than none.
    """

    total: int = 0
    total_bytes: int = 0
    loaded: bool = False
    #: Lower-cased basenames of every file.
    names: frozenset[str] = frozenset()
    #: Lower-cased extensions including the dot, mapped to how many files carry them.
    by_extension: Mapping[str, int] = field(default_factory=dict)
    #: Total bytes per extension.
    bytes_by_extension: Mapping[str, int] = field(default_factory=dict)
    #: Lower-cased names of every directory appearing anywhere in a relative path.
    directories: frozenset[str] = frozenset()
    #: Lower-cased first path segment of each relative path, i.e. the asset's own layout.
    top_level: frozenset[str] = frozenset()
    #: Lower-cased relative paths, forward-slashed.
    relpaths: tuple[str, ...] = ()

    def count(self, *extensions: str) -> int:
        """Return how many files carry any of these extensions.

        Examples:
            >>> summary = FileSummary(by_extension={".jpg": 12, ".png": 3})
            >>> summary.count(".jpg", ".png")
            15
        """
        return sum(self.by_extension.get(extension, 0) for extension in extensions)

    def bytes_in(self, *extensions: str) -> int:
        """Return the total size of files carrying any of these extensions."""
        return sum(self.bytes_by_extension.get(extension, 0) for extension in extensions)

    def has_name(self, *names: str) -> bool:
        """Report whether any of these exact basenames is present."""
        return any(name.lower() in self.names for name in names)

    def has_stem(self, *stems: str) -> bool:
        """Report whether any basename starts with one of these stems.

        Catches the family of a file rather than one spelling of it: ``has_stem("readme")``
        finds ``README.md``, ``readme.txt`` and ``README``.
        """
        lowered = tuple(stem.lower() for stem in stems)
        return any(name.startswith(lowered) for name in self.names)

    def has_dir(self, *names: str) -> bool:
        """Report whether a directory with any of these names appears in the tree."""
        return any(name.lower() in self.directories for name in names)

    def matching(self, pattern: str) -> int:
        """Return how many relative paths match a regular expression."""
        compiled = re.compile(pattern)
        return sum(1 for relpath in self.relpaths if compiled.search(relpath))


@dataclass(slots=True)
class AssetProfile:
    """Everything the catalogue knows about one asset, in the form plugins consume.

    Deliberately inert: no session, no filesystem access, no lazy loading. A plugin
    handed one of these can only read.
    """

    asset_id: int
    kind: str
    name: str
    path: str
    subkind: str | None = None
    framework: str = "unknown"
    format: str = "unknown"
    drive: str | None = None
    is_single_file: bool = False
    size_bytes: int = 0
    physical_size_bytes: int = 0
    file_count: int = 0
    modified_at: datetime | None = None
    created_at: datetime | None = None
    is_missing: bool = False
    detector: str | None = None
    tags: tuple[str, ...] = ()
    model: ModelFacts | None = None
    dataset: DatasetFacts | None = None
    files: FileSummary = field(default_factory=FileSummary)

    #: Cached lower-cased text of every naming signal. Built on first use because most
    #: classifiers ask for it and rebuilding it per rule would dominate the run.
    _haystack: str | None = field(default=None, repr=False, compare=False)

    @property
    def folder(self) -> str:
        """Return the name of the directory containing this asset, lower-cased."""
        trimmed = self.path.replace("\\", "/").rstrip("/")
        parent = trimmed.rsplit("/", 1)[0] if "/" in trimmed else ""
        return parent.rsplit("/", 1)[-1].lower()

    @property
    def haystack(self) -> str:
        """Return lower-cased text of every name-like signal, for marker matching.

        Includes the asset name, architecture, repository id, declared task, card tags and
        class names — the places a family or task name actually shows up.
        """
        if self._haystack is None:
            parts: list[str] = [self.name, self.subkind or ""]
            if self.is_single_file:
                # A lone weight file is often named by the folder that holds it and not by
                # itself: ``.cache/whisper/large-v3-turbo.pt`` says "whisper" nowhere in
                # the filename, and ``kraken_cache/arabic_historical.mlmodel`` says nothing
                # about OCR. The containing folder is the naming authority in exactly this
                # case, and only this one -- for a directory asset the folder *is* the
                # name, and for anything deeper the path is full of words that mean
                # nothing about the asset.
                parts.append(self.folder)
            if self.model is not None:
                parts += [
                    self.model.architecture or "",
                    self.model.repo_id or "",
                    self.model.base_model or "",
                    self.model.pipeline_tag or "",
                    self.model.library_name or "",
                    " ".join(self.model.card_tags),
                ]
            if self.dataset is not None:
                parts += [
                    self.dataset.dataset_format or "",
                    self.dataset.task or "",
                    self.dataset.repo_id or "",
                    " ".join(self.dataset.class_names[:64]),
                ]
            self._haystack = " ".join(part for part in parts if part).lower()
        return self._haystack

    def matches(self, markers: Iterable[str]) -> str | None:
        """Return the first marker occurring in :attr:`haystack`, or ``None``.

        Returning the marker rather than a boolean lets a classifier report *why* it
        matched, which is what makes a surprising classification debuggable.
        """
        haystack = self.haystack
        for marker in markers:
            if marker in haystack:
                return marker
        return None

    @property
    def is_dataset_like(self) -> bool:
        """Report whether this asset was catalogued as a dataset."""
        return self.dataset is not None or self.kind == "dataset"

    @property
    def is_model_like(self) -> bool:
        """Report whether this asset was catalogued as a model, adapter or checkpoint."""
        return self.model is not None or self.kind in ("model", "adapter", "checkpoint")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Classification:
    """What a plugin concluded about an asset."""

    category: str
    task: str | None = None
    domain: str | None = None
    #: Model or dataset family, e.g. the thing a user would name: "YOLO", "Qwen", "COCO".
    family: str | None = None
    modalities: tuple[str, ...] = ()
    confidence: float = CONFIDENCE_STRONG
    #: Short phrase naming the evidence, surfaced by ``--details`` and ``--explain``.
    evidence: str = ""
    #: Which classifier produced this, filled in by the registry.
    source: str = ""


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong, or possibly wrong, with an asset."""

    code: str
    severity: Severity
    message: str
    fix_hint: str = ""

    @property
    def penalty(self) -> int:
        """Return the health-score cost of this finding."""
        return _SEVERITY_PENALTY[self.severity]


#: What each severity costs the 0-100 health score. An error is something that stops the
#: asset being usable; a warning is something a user would want to fix before relying on
#: it; info is a note. Four warnings still leave a dataset above 50 — a missing README and
#: a missing licence should not read as badly as a missing annotation set.
_SEVERITY_PENALTY: dict[Severity, int] = {
    Severity.ERROR: 25,
    Severity.WARNING: 10,
    Severity.INFO: 3,
}


@dataclass(frozen=True, slots=True)
class HealthReport:
    """An asset's health verdict."""

    score: int = 100
    findings: tuple[Finding, ...] = ()
    #: False when no rule could run because the file list was not loaded.
    evaluated: bool = True

    @property
    def status(self) -> str:
        """Return ``ok``, ``warning``, ``error`` or ``unknown``."""
        if not self.evaluated:
            return "unknown"
        if any(finding.severity is Severity.ERROR for finding in self.findings):
            return "error"
        if any(finding.severity is Severity.WARNING for finding in self.findings):
            return "warning"
        return "ok"

    @property
    def is_healthy(self) -> bool:
        """Report whether nothing above informational was found."""
        return self.status == "ok"

    def messages(self) -> list[str]:
        """Return every finding's message, worst first."""
        ordered = sorted(self.findings, key=lambda finding: -finding.severity.rank)
        return [finding.message for finding in ordered]


# ---------------------------------------------------------------------------
# plugin contracts
# ---------------------------------------------------------------------------


#: A classification rule: return a verdict, or ``None`` to let the next rule try.
ClassifierFunction = Callable[["AssetProfile"], "Classification | None"]

#: A health rule: return findings, or an empty sequence when there is nothing to report.
HealthRuleFunction = Callable[["AssetProfile"], Sequence["Finding"]]

#: A statistics provider: return values to merge into the asset's report entry.
StatisticFunction = Callable[["AssetProfile"], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class Classifier:
    """A registered classification rule.

    Name and priority live here rather than on the function because a plugin's
    ``register`` then reads as a manifest — every rule it contributes and the order they
    run in, visible in one block instead of scattered across the module.
    """

    name: str
    priority: int
    run: ClassifierFunction

    def __call__(self, profile: AssetProfile) -> Classification | None:
        """Apply the rule."""
        return self.run(profile)


@dataclass(frozen=True, slots=True)
class HealthRule:
    """A registered health rule."""

    name: str
    run: HealthRuleFunction

    def __call__(self, profile: AssetProfile) -> Sequence[Finding]:
        """Apply the rule."""
        return self.run(profile)


@dataclass(frozen=True, slots=True)
class StatisticProvider:
    """A registered statistics provider."""

    name: str
    run: StatisticFunction

    def __call__(self, profile: AssetProfile) -> Mapping[str, Any]:
        """Compute the statistics."""
        return self.run(profile)
