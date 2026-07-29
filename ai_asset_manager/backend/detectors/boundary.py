r"""Where an asset starts and stops.

Detection walks parents before children and a claim suppresses everything beneath it, so
whichever directory matches *first* becomes the asset. That ordering is what makes the
generic detectors dangerous: a rule that fires on "two ``.jsonl`` files somewhere below"
fires at the drive root long before it reaches the corpus that actually contains them, and
the whole disk becomes one dataset.

An asset is the *smallest* directory that represents one logical AI object.
``F:\Models\Qwen2.5-7B`` is an asset; ``F:\Models`` is a shelf that holds assets;
``F:\`` is a disk. Nothing structural distinguishes them — all three are directories with
files below — so the distinction has to be drawn deliberately, and this module is where.

The guard applies only to *generic* detectors (see
:data:`~ai_asset_manager.backend.detectors.base.PRIORITY_DATASET_GENERIC`). A directory
holding ``annotations/instances_train.json`` is a COCO dataset whatever it is called and
however deep it sits, so structural evidence overrules everything here. What is guarded is
the class of rule that infers a dataset from a *pile of files*, which is exactly the class
of rule that cannot tell a corpus from a container.
"""

from __future__ import annotations

import os

from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.utils.paths import PAYLOAD_EXTENSIONS

#: Directory names that hold assets rather than being one. Two groups, and the second is
#: the less obvious: user folders (``Downloads``, ``Desktop``) are containers because
#: anything at all may land in them, and *library* folders (``Models``, ``datasets``,
#: ``checkpoints``) are containers because that is what the name means. Someone with
#: ``D:\Models\Qwen2.5-7B`` wants the Qwen row, not a single row called "Models" covering
#: 400 GB.
CONTAINER_NAMES: frozenset[str] = frozenset(
    {
        # User profile folders
        "downloads", "download", "desktop", "documents", "my documents", "pictures",
        "videos", "music", "onedrive", "dropbox", "google drive", "my drive",
        "public", "users", "home", "shared", "temp", "tmp", "new folder",
        # Windows profile roots. Every installed application writes under these, so any
        # rule that reads a whole subtree finds a little of everything here — and a claim
        # on one of them suppresses every real asset beneath it. `AppData\Roaming` was
        # reported as a 14 GB host-log corpus on the strength of four scattered log files,
        # and took thirty-two models down with it.
        "appdata", "roaming", "local", "locallow", "programs", "packages",
        # Development containers
        "projects", "project", "repos", "repositories", "repo", "git", "source",
        "src", "code", "work", "workspace", "dev", "sandbox", "playground",
        # AI library shelves: the thing above the asset, never the asset
        "models", "model", "datasets", "dataset", "data", "checkpoints", "checkpoint",
        "weights", "ai", "ml", "llm", "llms", "cache", "caches", ".cache", "hub",
        "training", "experiments", "runs", "output", "outputs", "results",
    }
)

#: A flat pile of images with no labels, no manifest and no split structure is a dataset
#: only when it is big enough that nothing else plausibly explains it. Twenty PNGs are far
#: more likely to be a report's figures — which is exactly what ``work/figs`` turned out to
#: be on the development machine — than a training set.
MIN_UNSTRUCTURED_IMAGES = 200


#: Files that a dataset root carries and a container does not. Their presence is a
#: deliberate act by whoever built the dataset, which is what makes them trustworthy.
DATASET_MANIFESTS: tuple[str, ...] = (
    "data.yaml", "data.yml", "dataset.yaml", "dataset_info.json", "dataset_infos.json",
    "dataset_dict.json", "classes.txt", "labels.txt", "metadata.jsonl", "manifest.json",
    "manifest.jsonl", "annotations.json", "state.json", "_stats.json", "stats.json",
)

#: Directory names that mean "an application writes its state here". Never a dataset,
#: however dataset-shaped the contents look.
#:
#: This is the class of false positive a whole-machine scan turns up by the dozen. Chat
#: transcripts, IDE telemetry and agent session records are all line-delimited JSON, which
#: is exactly what an NLP corpus is; on the development machine fifty of them were
#: catalogued as datasets. Nothing about the *files* distinguishes them. The directory's
#: name does.
APPLICATION_STATE_NAMES: frozenset[str] = frozenset(
    {
        "log", "logs", "telemetry", "crashes", "crashreports", "crashdumps",
        "diagnostics", "dumps", "minidumps", "sessions", "settingslogs", "history",
        "subagents", "workspacestorage", "globalstorage", "localstorage",
        "sessionstorage", "localstate", "eventlog", "journal", "traces",
        # Agent and editor state directories. Their transcripts are line-delimited JSON
        # sitting in per-project folders, which is a corpus down to the byte.
        ".claude", ".codex", ".cursor", ".aider", ".continue", ".gemini",
        ".antigravity", ".copilot", ".ollama-ui",
    }
)

#: Substrings of the same thing, for names that carry an identifier or a prefix:
#: ``mcp-logs-ide``, ``emptyWindowChatSessions``.
APPLICATION_STATE_TOKENS: tuple[str, ...] = (
    "mcp-logs", "crashpad", "sentry", "chatsessions", "chat-sessions",
)


def is_drive_root(path: str) -> bool:
    r"""Report whether a path is the root of a drive or filesystem.

    ``F:\`` and ``/`` are the two cases; both satisfy ``dirname(p) == p``.
    """
    normalised = os.path.normpath(path)
    return os.path.dirname(normalised) == normalised


def is_container_name(name: str) -> bool:
    """Report whether a directory's own name marks it as holding assets, not being one."""
    return name.strip().lower() in CONTAINER_NAMES


def is_application_state(path: str) -> bool:
    r"""Report whether a path lies anywhere inside an application's own state directory.

    Every segment is checked, not just the last one, because the marker is usually an
    ancestor: the transcripts live in ``.claude\projects\<project name>`` and the session
    records in ``.codex\sessions\2026\05\21``, where the *leaf* is named after a project
    or a date and says nothing. Guarding on the leaf alone caught none of them.
    """
    for segment in path.replace("\\", "/").split("/"):
        lowered = segment.strip().lower()
        if not lowered:
            continue
        if lowered in APPLICATION_STATE_NAMES:
            return True
        if any(token in lowered for token in APPLICATION_STATE_TOKENS):
            return True
    return False


def holds_structured_children(ctx: DirectoryContext) -> bool:
    """Report whether this directory's children look like assets in their own right.

    A leaf class folder inside a dataset holds images and nothing else. A directory whose
    children each have their own subdirectories is a level too high — it is holding
    *things that have structure*, which is what a container does.
    """
    branching = sum(1 for child in ctx.children() if not child.is_leaf)
    return branching >= 2


def has_direct_payload(ctx: DirectoryContext) -> bool:
    """Report whether this directory itself holds weights or data files.

    A container's substance is in its children; an asset's substance is in itself. This is
    the one signal that can rescue a directory whose name looks like a shelf.
    """
    return any(entry.extension in PAYLOAD_EXTENSIONS for entry in ctx.files)


def looks_like_dataset_root(ctx: DirectoryContext) -> bool:
    """Report positive evidence that this directory *is* one dataset, not a shelf of them.

    Structurally a dataset with ``train/`` and ``val/`` splits is indistinguishable from a
    folder holding two unrelated datasets: both are directories whose children branch. The
    split names are the difference. Nobody names two unrelated projects ``train`` and
    ``test``, so those names — or a manifest the dataset's author wrote by hand — settle
    the question that shape alone cannot.
    """
    from ai_asset_manager.backend.parsers.dataset_meta import canonical_split

    splits = sum(1 for name in ctx.child_dir_names if canonical_split(name))
    return splits >= 2 or ctx.has_any(*DATASET_MANIFESTS)


def may_claim_generic(ctx: DirectoryContext) -> tuple[bool, str]:
    """Decide whether a generic detector may claim this whole directory as one asset.

    Returns:
        ``(allowed, reason)``. The reason is empty when allowed and names the rule that
        refused otherwise, so a surprising non-detection can be explained rather than
        merely observed.
    """
    if is_drive_root(ctx.path):
        return False, "drive root"

    # Checked before the dataset-root exemption: an application that writes a `state.json`
    # beside its logs must not thereby become a dataset.
    if is_application_state(ctx.path):
        return False, "inside an application state directory"

    if looks_like_dataset_root(ctx):
        return True, ""

    if is_container_name(ctx.name) and not has_direct_payload(ctx):
        return False, f"container directory ({ctx.name})"

    if holds_structured_children(ctx) and not has_direct_payload(ctx):
        return False, "children have their own structure"

    return True, ""
