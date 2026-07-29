r"""Deriving a usable name from a path.

The derivation, in order:

1. **Vendor and product.** The path is searched against :data:`~.vendors.VENDORS`, most
   specific entry first. The *deepest* match wins, so ``Google\\Chrome`` beats a bare
   ``Google`` seen higher up.
2. **Component.** The segments below the product, with the noise words stripped, matched
   against :data:`~.vendors.COMPONENTS`. Failing a table match, the deepest informative
   segment is used as-is — an unrecognised component name still beats "model".
3. **Task.** From the component's table entry, or from the same tokens read again.
4. **Name.** Assembled only when the existing name is generic. A model already called
   ``resnet18-f37072fd`` keeps its name; the vendor and source are recorded regardless,
   because "where did this come from?" is worth answering for every asset. A bare vendor
   is refused as a name — "Google Model" says nothing "model" did not — but a bare
   product or a bare component is accepted, since both answer the question the user is
   actually asking of an unnamed row.

Nothing here reads a file or stats a path. The input is a string.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ai_asset_manager.backend.identity.vendors import (
    CASING,
    COMPONENTS,
    GENERIC_NAMES,
    NOISE_SEGMENTS,
    VENDORS,
)

#: Source recorded when no vendor matched but the asset was clearly placed by hand — a
#: model in a project folder or a downloads directory.
SOURCE_MANUAL = "manual"

#: Source recorded when nothing at all could be determined.
SOURCE_UNKNOWN = "unknown"

#: Segments that are a version, a hash or a serial number, and so never a component.
_UNINFORMATIVE_SEGMENT = re.compile(
    r"""^(?:
        v?\d+(?:[._-]\d+)*        # 1.2.3, v2, 20250218
        |[0-9a-f]{8,}             # a hex digest or a git sha
        |\d{4}[-_]?\d{2}[-_]?\d{2}  # a date
        |[{(]?[0-9a-f]{8}-[0-9a-f-]{27}[)}]?  # a GUID
        |[a-z]{2}([-_][a-z]{2,4})?  # a locale: en, en-us, zh-hans
    )$""",
    re.VERBOSE | re.IGNORECASE,
)

#: Suffixes stripped from a filename before it is judged generic. ``model_quantized`` and
#: ``model.fp16`` say nothing that ``model`` did not.
_VARIANT_SUFFIXES: tuple[str, ...] = (
    "_quantized", "_quantised", "_optimized", "_optimised", "_dynamic", "_static",
    "_int8", "_uint8", "_fp16", "_fp32", "_float16", "_float32", "_q4", "_q8",
    "_final", "_export", "_exported", "_converted", "_opt", "_sim", "_simplified",
)


#: Words that already end a name properly, so that "Zoom Speech Recognition" is not
#: turned into "Zoom Speech Recognition Model".
_NAME_ENDINGS: tuple[str, ...] = (
    "model", "encoder", "decoder", "recognition", "detection", "segmentation",
    "estimation", "classification", "generation", "removal", "suppression",
    "cancellation", "analysis", "completion", "prediction", "reader", "filter",
    "reply", "safety", "ranking", "tracking", "landmarks", "identification",
    "embedding", "reranker", "synthesis", "enhancement", "translation",
    "summarisation", "understanding",
)


@dataclass(frozen=True, slots=True)
class AssetIdentity:
    """Who an asset belongs to and what it is for."""

    vendor: str | None = None
    product: str | None = None
    component: str | None = None
    task: str | None = None
    #: Short identifier for the software that put the asset here: ``"chrome"``,
    #: ``"huggingface"``, ``"manual"``, ``"unknown"``.
    source: str = SOURCE_UNKNOWN
    #: A better name, or ``None`` when the asset's own name was already meaningful.
    display_name: str | None = None
    #: What the derivation matched on, for the same reason detectors record evidence.
    signals: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Report whether nothing at all was determined."""
        return not (self.vendor or self.product or self.component or self.display_name)

    def as_dict(self) -> dict[str, object]:
        """Return the identity as a JSON-serialisable mapping, omitting empty fields."""
        payload: dict[str, object] = {"source": self.source}
        for key, value in (
            ("vendor", self.vendor),
            ("product", self.product),
            ("component", self.component),
            ("task", self.task),
            ("display_name", self.display_name),
        ):
            if value:
                payload[key] = value
        if self.signals:
            payload["signals"] = list(self.signals)
        return payload


def is_generic_name(name: str) -> bool:
    """Report whether a name says nothing about what the asset is.

    Examples:
        >>> is_generic_name("model")
        True
        >>> is_generic_name("model_quantized")
        True
        >>> is_generic_name("resnet18-f37072fd")
        False
    """
    stem = os.path.splitext(name.strip())[0].lower()
    for suffix in _VARIANT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    stem = stem.strip(" _-.")
    if not stem:
        return True
    if stem in GENERIC_NAMES:
        return True
    # A pure version, digest or serial number names nothing either.
    return bool(_UNINFORMATIVE_SEGMENT.match(stem))


def identify(path: str, *, name: str = "", is_single_file: bool = False) -> AssetIdentity:
    """Derive vendor, product, component, task and a usable name from a path.

    Args:
        path: Absolute path to the asset — a directory, or the file itself.
        name: The name the detector produced, used to decide whether a better one is
            needed. An empty string is treated as generic.
        is_single_file: Whether ``path`` names a file rather than a directory. Decides
            where the search for a component starts.

    Returns:
        An identity. Never raises, and never returns ``display_name`` when the existing
        name was already informative.
    """
    segments = _segments(path)
    if not segments:
        return AssetIdentity()

    # A filename's extension is not a vendor. Without this, every `model.onnx` on the
    # machine was attributed to the runtime that reads ONNX files.
    searchable = segments[:-1] if is_single_file and len(segments) > 1 else segments

    signals: list[str] = []
    vendor, product, source, vendor_index = _match_vendor(searchable, path)
    if vendor:
        signals.append(f"path names {product or vendor}")

    # From the matched segment inclusive: the segment that identified the product often
    # also names the component, as a VS Code extension folder does.
    below = segments[vendor_index:] if vendor_index >= 0 else segments[-4:]
    if is_single_file and below:
        below = [*below[:-1], os.path.splitext(below[-1])[0]]

    component, task, matched = _match_component(below, name)
    if matched:
        signals.append(f"component {matched!r}")

    if component is None:
        # The asset's own name is excluded: a component that merely restates it adds
        # nothing, and reads as though something was determined when nothing was.
        component = _fallback_component(
            below, skip={vendor, product, name, os.path.splitext(name)[0]}
        )
        if component:
            signals.append(f"nearest informative folder {component!r}")

    if source == SOURCE_UNKNOWN and _looks_hand_placed(segments):
        source = SOURCE_MANUAL

    display_name = None
    if is_generic_name(name):
        display_name = _compose(vendor, product, component, task, name=name)
        if display_name:
            signals.append(f"renamed from {name or '(unnamed)'}")

    return AssetIdentity(
        vendor=vendor,
        product=product,
        component=component,
        task=task,
        source=source,
        display_name=display_name,
        signals=tuple(signals),
    )


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def _segments(path: str) -> list[str]:
    """Split a path into its components, dropping the drive and empty parts."""
    normalised = path.replace("\\", "/")
    _, _, tail = normalised.partition(":")
    return [part for part in (tail or normalised).split("/") if part]


def _match_vendor(
    segments: list[str], path: str
) -> tuple[str | None, str | None, str, int]:
    r"""Return the deepest vendor match, and the index of the segment that matched.

    Deepest rather than first: an installer under ``Program Files\\Google\\Chrome`` matches
    both ``google`` and ``google\\chrome``, and the product is what identifies the model.
    """
    lowered = [segment.lower() for segment in segments]
    joined = path.replace("/", "\\").lower()

    # An extension folder names the extension's ecosystem, not the application holding it.
    # `Roaming\Cursor\...\ms-vscode.js-debug\...` is a model Cursor installed, and reading
    # the deepest match would file it under VS Code because a VS Code extension id happens
    # to sit below. Application directories are consulted first; extension ids only decide
    # it when nothing else does.
    # Each candidate carries its position in the real path, so a filtered pass still
    # reports where in the path the match was — two segments can share a name, and
    # searching for the string again would find the wrong one.
    applications = [
        (position, segment)
        for position, segment in enumerate(lowered)
        if not _is_extension_id(segment)
    ]
    everything = list(enumerate(lowered))

    for candidates in (applications, everything):
        best: tuple[int, str, str | None, str] | None = None
        for markers, vendor, product, source in VENDORS:
            index = _deepest_marker(candidates, joined, markers)
            if index is None:
                continue
            # A tie at the same depth goes to the entry declared first, which is the more
            # specific one: the tables are ordered that way deliberately.
            if best is None or index > best[0]:
                best = (index, vendor, product, source)

        if best is not None:
            return best[1], best[2], best[3], best[0]

    return None, None, SOURCE_UNKNOWN, -1


#: A publisher-qualified extension identifier: ``ms-vscode.js-debug``,
#: ``devsense.intelli-php-vscode-1.2``. Leading dots are excluded so that ``.vscode`` and
#: ``.antigravity`` — which *are* application directories — still match normally.
_EXTENSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]*\.[a-z0-9][a-z0-9.-]*$", re.IGNORECASE)


def _is_extension_id(segment: str) -> bool:
    """Report whether a path segment names an editor extension rather than an application."""
    return bool(_EXTENSION_ID.match(segment))


def _deepest_marker(
    candidates: list[tuple[int, str]], joined: str, markers: tuple[str, ...]
) -> int | None:
    """Return the path position of the deepest segment matching any marker, or ``None``.

    Args:
        candidates: ``(position in the full path, lower-cased segment)`` pairs.
        joined: The whole path, backslash-separated and lower-cased, for two-part markers.
        markers: The markers of one vendor entry.
    """
    found: int | None = None
    for position, segment in candidates:
        for marker in markers:
            if "\\" in marker or "/" in marker:
                # A two-part marker such as `google\chrome` has to be matched against the
                # joined path; when it hits, the product segment is the later of the two.
                parts = marker.replace("/", "\\").strip("\\").split("\\")
                if marker.replace("/", "\\") in joined and segment in parts:
                    found = position
                continue
            if _segment_matches(segment, marker):
                found = position
    return found


def _segment_matches(segment: str, marker: str) -> bool:
    """Report whether a path segment names a marker.

    Substring matching alone was too eager — ``intel`` matched ``intellij`` and ``jan``
    matched anything with those three letters in it. A marker matches when it *is* the
    segment, when it is one of the segment's words, when it survives the segment's
    punctuation being removed, or when it is long enough that a substring hit cannot be a
    coincidence.
    """
    if segment == marker:
        return True

    words = set(re.split(r"[^a-z0-9]+", segment))
    words.discard("")
    if marker in words:
        return True

    if len(marker) >= 6:
        if marker in segment:
            return True
        # `devsense.intelli-php-vscode-1.2` names `intelliphp`; the punctuation is noise.
        squashed = re.sub(r"[^a-z0-9]+", "", segment)
        if re.sub(r"[^a-z0-9]+", "", marker) in squashed:
            return True

    return False


def _match_component(
    segments: list[str], name: str
) -> tuple[str | None, str | None, str | None]:
    """Return the component and task the path names.

    Searched deepest first, because the segment nearest the weights is the one that
    describes them. The asset's own name is searched last: it is usually the generic word
    that started this, but occasionally it carries the only useful token.
    """
    candidates = [segment.lower() for segment in reversed(segments)]
    if name:
        candidates.append(name.lower())

    # Each segment is searched both as written and with its punctuation removed, so
    # `intelli-php` and `intelliphp` are the same component.
    haystacks = [
        (candidate, re.sub(r"[^a-z0-9]+", "", candidate)) for candidate in candidates
    ]

    for raw, squashed in haystacks:
        for markers, component, task in COMPONENTS:
            for marker in markers:
                if marker in raw or re.sub(r"[^a-z0-9]+", "", marker) in squashed:
                    return component, task, marker
    return None, None, None


def _fallback_component(segments: list[str], *, skip: set[str | None]) -> str | None:
    """Return the deepest segment that says anything, prettified.

    The rule that rescues components nobody has tabulated. ``.../Zoom/bin/aomhost/
    frames_processor/model.onnx`` is not in any table, but "Frames Processor" is a great
    deal more use than "model".
    """
    forbidden = {value.lower() for value in skip if value}
    for segment in reversed(segments):
        lowered = segment.lower()
        if lowered in NOISE_SEGMENTS or _UNINFORMATIVE_SEGMENT.match(lowered):
            continue
        if is_generic_name(lowered) or lowered in forbidden:
            continue
        if len(lowered) < 3:
            continue
        return prettify(segment)
    return None


def _looks_hand_placed(segments: list[str]) -> bool:
    """Report whether the path is somewhere a person put things, not an application."""
    markers = {
        "downloads", "download", "desktop", "documents", "projects", "project",
        "repos", "work", "workspace", "dev", "src", "code", "onedrive", "dropbox",
    }
    return any(segment.lower() in markers for segment in segments)


def _compose(
    vendor: str | None,
    product: str | None,
    component: str | None,
    task: str | None,
    *,
    name: str = "",
) -> str | None:
    """Assemble a display name from whatever was determined.

    Returns ``None`` when there is nothing to say: a name built from no evidence would be
    worse than the generic one, because it would look informative.
    """
    parts: list[str] = []

    owner = product or vendor
    if owner:
        parts.append(owner)
    if component and not _already_said(parts, component):
        parts.append(component)
    if task and not _overlaps(parts, task):
        parts.append(task)

    if not parts:
        return None

    if len(parts) == 1 and not product:
        # A bare *vendor* is not an identity: "Google Model" names nothing that "model" did
        # not. A bare *product* is — "Cursor Model" at least says which application put it
        # there, which is the question a user staring at fifty identical rows is asking.
        if not component:
            return None
        # A bare *component* is also an identity, and a better one than the word it
        # replaces: `catboost_info\test` is "CatBoost Info Test". Nothing here claims the
        # asset is a model — with no vendor and no product there is no evidence for that —
        # so the "Model" suffix below is skipped and the discriminating word is kept.
        return _collapse_repeats(_with_qualifier(parts[0], name))

    if not parts[-1].lower().endswith(_NAME_ENDINGS):
        parts.append("Model")

    return _collapse_repeats(" ".join(parts))


def _with_qualifier(component: str, name: str) -> str:
    """Append the asset's own name to a bare component when it still distinguishes it.

    ``catboost_info`` holds both ``learn`` and ``test``. Calling them both "CatBoost Info"
    would reintroduce the identical-rows problem this module exists to solve, one level
    up. A generic *word* is kept as a qualifier for that reason; a digest or a version is
    not, because it distinguishes without informing.
    """
    stem = os.path.splitext(name.strip())[0].strip(" _-.")
    if not stem or _UNINFORMATIVE_SEGMENT.match(stem.lower()):
        return component
    if _already_said([component], stem):
        return component
    return f"{component} {prettify(stem)}"


def _collapse_repeats(phrase: str) -> str:
    """Drop a word that immediately repeats the one before it.

    ``VS Code`` plus ``Code Completion`` is ``VS Code Code Completion``, which reads as a
    mistake even though both halves are right.

    Examples:
        >>> _collapse_repeats("VS Code Code Completion")
        'VS Code Completion'
    """
    words = phrase.split()
    kept = [
        word
        for index, word in enumerate(words)
        if index == 0 or word.lower() != words[index - 1].lower()
    ]
    return " ".join(kept)


def _already_said(parts: list[str], candidate: str) -> bool:
    """Report whether a phrase repeats everything already assembled."""
    said = " ".join(parts).lower()
    words = candidate.lower().split()
    return all(word in said for word in words)


#: Characters of a shared prefix that make two words the same idea. Long enough that
#: "reranker" and "reranking" match while "prediction" and "precision" do not.
_STEM_LENGTH = 5


def _overlaps(parts: list[str], candidate: str) -> bool:
    """Report whether a phrase says anything already assembled.

    The test the *task* is held to, and a looser one than :func:`_already_said` on purpose.
    A component usually names the task already in different words — "Text Prediction" is a
    text-generation model, "Reranker" does reranking — and appending the task anyway
    produces "VS Code Reranker Reranking". One shared idea is enough to stop.

    Compared on stems rather than whole words, because the component and the task are
    routinely the same word in different parts of speech.
    """
    said = " ".join(parts).lower().split()
    for word in candidate.lower().split():
        for existing in said:
            if word == existing:
                return True
            if (
                len(word) >= _STEM_LENGTH
                and len(existing) >= _STEM_LENGTH
                and word[:_STEM_LENGTH] == existing[:_STEM_LENGTH]
            ):
                return True
    return False


def prettify(segment: str) -> str:
    """Turn a path segment into a readable phrase.

    Examples:
        >>> prettify("text_recognition")
        'Text Recognition'
        >>> prettify("gocr-mobile-und")
        'Gocr Mobile Und'
        >>> prettify("screen_ai")
        'Screen AI'
    """
    words = [word for word in re.split(r"[\s_\-.]+", segment.strip()) if word]
    rendered: list[str] = []
    for word in words:
        lowered = word.lower()
        if lowered in CASING:
            rendered.append(CASING[lowered])
        elif word.isupper() and len(word) <= 5:
            rendered.append(word)
        else:
            rendered.append(word[:1].upper() + word[1:])
    return " ".join(rendered)


__all__ = ["AssetIdentity", "identify", "is_generic_name", "prettify"]
