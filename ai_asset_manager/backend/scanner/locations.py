r"""Where AI assets end up, and how to find them without scanning the whole disk.

The first question a new user has is "what do I point it at?", and the honest answer is
that they mostly do not know: the whole reason a catalogue is useful is that downloads
scatter themselves across caches nobody chose. So the tool works it out.

Three rules make that safe:

*Overrides beat defaults, always.* Someone who set ``HF_HOME`` did it because the cache
outgrew its drive, so the default path is not merely redundant — it is wrong, and offering
it would send a scan at a folder that no longer holds anything.

*Nothing is offered that does not exist.* A suggestion that turns out to be an empty path
is worse than no suggestion, because the user cannot tell which of the two it is.

*The wide sweep never enters a system directory.* Looking two levels down from each drive
root finds ``D:\Models`` and ``E:\AI\datasets``; it must never wander into ``C:\Windows``,
``Program Files`` or a browser cache. The exclusion list below is what keeps a discovery
pass measured in milliseconds instead of minutes.

Reusable beyond the CLI: the dashboard's "add a folder" screen wants the same list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Cap on how many locations a sweep will offer. A machine with a hundred candidate
#: folders has a naming problem the user should resolve, not a discovery problem.
MAX_DISCOVERED = 60

#: How far below a drive root to look for asset folders. One level finds ``D:\Models``;
#: two finds ``D:\AI\Models``, which is where people actually put things. Three would
#: start costing real time on a full disk for almost no extra yield.
SWEEP_DEPTH = 2

#: Directory names the wide sweep must never enter, lower-cased. System trees, package
#: managers and browser caches are all large, all irrelevant, and all capable of turning a
#: discovery pass into a disk-thrashing crawl.
SWEEP_EXCLUDED = frozenset({
    # Windows system and vendor trees
    "windows", "winnt", "program files", "program files (x86)", "programdata",
    "system volume information", "$recycle.bin", "recovery", "perflogs",
    "msocache", "$windows.~bt", "$windows.~ws", "config.msi", "system32",
    "appdata",  # reached explicitly by name below, never swept
    # Unix system trees
    "proc", "sys", "dev", "boot", "bin", "sbin", "lib", "lib64", "usr", "etc",
    "var", "run", "tmp", "private", "system", "library", "applications",
    # Package managers, build output and caches
    "node_modules", "__pycache__", ".git", ".svn", ".hg", "venv", ".venv",
    "site-packages", "dist-packages", "target", "build", "dist", ".gradle",
    ".nuget", ".cargo", ".rustup", ".npm", ".yarn", ".pnpm-store", "vendor",
    # Browser and application caches
    "cache2", "code cache", "gpucache", "service worker", "crashpad",
    "webcache", "inetcache", "temporary internet files", "temp", "tmpfiles",
    # Cloud sync placeholders, which are often not on disk at all
    "onedrivetemp",
})

#: Environment variables naming a directory that overrides a default, per tool. Order
#: matters: the first one set and pointing somewhere real wins.
#: Two are deliberately absent as *sources* and used only to expand paths below --
#: ``HOME``/``USERPROFILE`` and ``XDG_CACHE_HOME`` describe the user, not a tool.
_GENERIC_ENV = ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_CACHE_HOME")


@dataclass(frozen=True, slots=True)
class DiscoverySource:
    """A tool, and the places it is known to keep downloads."""

    key: str
    label: str
    #: Grouping shown to the user: what kind of thing this holds.
    group: str
    #: Environment variables that override the defaults, most specific first.
    env_vars: tuple[str, ...] = ()
    #: Default locations, as templates expanded against this machine's directories.
    defaults: tuple[str, ...] = ()
    #: When set, the path only counts if one of these entries exists inside it. Stops a
    #: bare application folder from being offered as a model library.
    markers: tuple[str, ...] = ()


#: Everything this tool knows how to find. Adding a row is the whole cost of supporting a
#: new ecosystem, which is the same bargain the taxonomy plugins offer.
SOURCES: tuple[DiscoverySource, ...] = (
    # -- model hubs and runtimes -------------------------------------------
    DiscoverySource(
        "huggingface", "HuggingFace", "Model caches",
        env_vars=("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"),
        defaults=("{cache}/huggingface", "{home}/.huggingface"),
    ),
    DiscoverySource(
        "ollama", "Ollama", "Model caches",
        env_vars=("OLLAMA_MODELS",),
        defaults=("{home}/.ollama", "{localappdata}/Ollama"),
    ),
    DiscoverySource(
        "torch", "PyTorch Hub", "Model caches",
        env_vars=("TORCH_HOME", "TORCH_HUB"),
        defaults=("{cache}/torch", "{home}/.torch"),
    ),
    DiscoverySource(
        "llama_cpp", "llama.cpp", "Model caches",
        env_vars=("LLAMA_CACHE",),
        defaults=("{home}/.cache/llama.cpp", "{home}/llama.cpp/models"),
    ),
    DiscoverySource(
        "lmstudio", "LM Studio", "Model caches",
        defaults=("{home}/.lmstudio/models", "{cache}/lm-studio/models",
                  "{home}/.cache/lm-studio/models"),
    ),
    DiscoverySource(
        "koboldcpp", "KoboldCpp", "Model caches",
        defaults=("{home}/koboldcpp/models", "{home}/.koboldcpp"),
    ),
    DiscoverySource(
        "vllm", "vLLM", "Model caches",
        env_vars=("VLLM_CACHE_ROOT",),
        defaults=("{cache}/vllm",),
    ),
    DiscoverySource(
        "modelscope", "ModelScope", "Model caches",
        env_vars=("MODELSCOPE_CACHE",),
        defaults=("{cache}/modelscope",),
    ),
    DiscoverySource(
        "kaggle", "Kaggle", "Model caches",
        env_vars=("KAGGLEHUB_CACHE",),
        defaults=("{cache}/kagglehub",),
    ),
    DiscoverySource(
        "ngc", "NVIDIA NGC", "Model caches",
        env_vars=("NGC_HOME",),
        defaults=("{home}/.ngc", "{cache}/ngc"),
    ),

    # -- image generation ---------------------------------------------------
    DiscoverySource(
        "comfyui", "ComfyUI", "Image generation",
        env_vars=("COMFYUI_PATH", "COMFYUI_MODEL_PATH"),
        defaults=("{home}/ComfyUI/models", "{documents}/ComfyUI/models"),
    ),
    DiscoverySource(
        "automatic1111", "Automatic1111", "Image generation",
        defaults=("{home}/stable-diffusion-webui/models",
                  "{documents}/stable-diffusion-webui/models"),
    ),
    DiscoverySource(
        "invokeai", "InvokeAI", "Image generation",
        env_vars=("INVOKEAI_ROOT",),
        defaults=("{home}/invokeai/models", "{home}/.invokeai/models"),
    ),
    DiscoverySource(
        "textgen", "text-generation-webui", "Image generation",
        defaults=("{home}/text-generation-webui/models",),
    ),

    # -- vision and OCR -----------------------------------------------------
    DiscoverySource(
        "ultralytics", "Ultralytics / YOLO", "Vision models",
        env_vars=("YOLO_CONFIG_DIR",),
        defaults=("{home}/.config/Ultralytics", "{appdata}/Ultralytics",
                  "{home}/.ultralytics"),
    ),
    DiscoverySource(
        "mmdetection", "MMDetection", "Vision models",
        env_vars=("MMENGINE_HOME",),
        defaults=("{cache}/mim", "{home}/.mim", "{home}/.cache/openmim"),
    ),
    DiscoverySource(
        "detectron2", "Detectron2", "Vision models",
        env_vars=("FVCORE_CACHE", "DETECTRON2_DATASETS"),
        defaults=("{cache}/iopath_cache", "{home}/.torch/iopath_cache"),
    ),
    DiscoverySource(
        "opencv", "OpenCV model zoo", "Vision models",
        defaults=("{cache}/opencv", "{home}/.opencv"),
    ),
    DiscoverySource(
        "insightface", "InsightFace", "Vision models",
        defaults=("{home}/.insightface",),
    ),
    DiscoverySource(
        "rembg", "rembg", "Vision models",
        defaults=("{home}/.u2net",),
    ),
    DiscoverySource(
        "paddleocr", "PaddleOCR", "OCR",
        defaults=("{home}/.paddleocr", "{home}/.paddlex"),
    ),
    DiscoverySource(
        "easyocr", "EasyOCR", "OCR",
        env_vars=("EASYOCR_MODULE_PATH",),
        defaults=("{home}/.EasyOCR",),
    ),
    DiscoverySource(
        "tesseract", "Tesseract", "OCR",
        env_vars=("TESSDATA_PREFIX",),
        defaults=("{home}/tessdata", "C:/Program Files/Tesseract-OCR/tessdata"),
        markers=("eng.traineddata",),
    ),
    DiscoverySource(
        "surya", "Surya", "OCR",
        defaults=("{cache}/datalab", "{home}/.cache/datalab"),
    ),

    # -- speech -------------------------------------------------------------
    DiscoverySource(
        "whisper", "Whisper", "Speech",
        env_vars=("WHISPER_CACHE",),
        defaults=("{cache}/whisper", "{home}/.cache/whisper"),
    ),
    DiscoverySource(
        "piper", "Piper", "Speech",
        defaults=("{home}/.local/share/piper", "{appdata}/piper"),
    ),

    # -- classic frameworks -------------------------------------------------
    DiscoverySource(
        "tfhub", "TensorFlow Hub", "Model caches",
        env_vars=("TFHUB_CACHE_DIR",),
        defaults=("{cache}/tfhub_modules",),
    ),
    DiscoverySource(
        "keras", "Keras", "Model caches",
        env_vars=("KERAS_HOME",),
        defaults=("{home}/.keras",),
    ),
    DiscoverySource(
        "onnx", "ONNX", "Model caches",
        defaults=("{cache}/onnx", "{home}/.onnx"),
    ),

    # -- datasets and annotation -------------------------------------------
    DiscoverySource(
        "fiftyone", "FiftyOne", "Datasets",
        env_vars=("FIFTYONE_DATABASE_DIR", "FIFTYONE_DEFAULT_DATASET_DIR"),
        defaults=("{home}/fiftyone",),
    ),
    DiscoverySource(
        "cvat", "CVAT exports", "Datasets",
        defaults=("{documents}/cvat", "{home}/cvat-exports", "{downloads}/cvat"),
    ),
    DiscoverySource(
        "label_studio", "Label Studio", "Datasets",
        env_vars=("LABEL_STUDIO_DATA_DIR", "LABEL_STUDIO_BASE_DATA_DIR"),
        defaults=("{home}/.local/share/label-studio", "{localappdata}/label-studio"),
    ),
    DiscoverySource(
        "roboflow", "Roboflow", "Datasets",
        defaults=("{home}/.cache/roboflow", "{downloads}/roboflow"),
    ),

    # -- experiments and logs -----------------------------------------------
    DiscoverySource(
        "wandb", "Weights & Biases", "Experiments",
        env_vars=("WANDB_DIR", "WANDB_CACHE_DIR"),
        defaults=("{home}/.cache/wandb", "{home}/wandb"),
    ),
    DiscoverySource(
        "mlflow", "MLflow", "Experiments",
        defaults=("{home}/mlruns", "{documents}/mlruns"),
    ),
    DiscoverySource(
        "tensorboard", "TensorBoard logs", "Experiments",
        defaults=("{home}/runs", "{home}/tb_logs", "{documents}/runs"),
    ),
)

#: Folder names on a drive root that are worth offering. Deliberately short: a longer list
#: turns a sweep into a source of false positives, and the user can always name a folder
#: explicitly.
SWEEP_NAMES = frozenset({
    "models", "model", "ai", "ml", "llm", "llms", "datasets", "dataset", "data",
    "checkpoints", "weights", "ai-models", "aimodels", "ai_models", "training",
    "huggingface", "gguf", "loras", "embeddings", "comfyui", "stable-diffusion",
})


@dataclass(frozen=True, slots=True)
class KnownLocation:
    """A directory that probably holds AI assets."""

    path: Path
    label: str
    #: What put it there, shown so the user can recognise it.
    source: str
    #: Heading to list this under.
    group: str = "Other"
    #: How it was found: an environment variable name, or a short phrase.
    origin: str = "default location"

    def __str__(self) -> str:
        """Return the path as text."""
        return str(self.path)


def _directories() -> dict[str, Path]:
    """Return the machine's base directories, honouring the environment.

    ``HOME``/``USERPROFILE`` and ``XDG_CACHE_HOME`` are read here rather than treated as
    tool overrides, because they relocate everything at once rather than one tool.
    """
    home = Path(
        os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home()
    ).expanduser()

    cache = os.environ.get("XDG_CACHE_HOME")
    local = os.environ.get("LOCALAPPDATA")
    roaming = os.environ.get("APPDATA")

    return {
        "home": home,
        "cache": Path(cache).expanduser() if cache else home / ".cache",
        "localappdata": Path(local) if local else home / "AppData" / "Local",
        "appdata": Path(roaming) if roaming else home / "AppData" / "Roaming",
        "documents": home / "Documents",
        "downloads": home / "Downloads",
    }


def _expand(template: str, directories: dict[str, Path]) -> Path | None:
    """Expand a path template against this machine's base directories."""
    try:
        return Path(template.format(**directories)).expanduser()
    except (KeyError, ValueError, OSError):
        return None


def _override_for(source: DiscoverySource) -> tuple[Path, str] | None:
    """Return the overriding directory for a tool, and which variable set it."""
    for name in source.env_vars:
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        # HF_HOME names a parent that contains `hub/`; the others name the cache itself.
        # Offering the parent is still correct, since the scanner recurses.
        if candidate.is_dir():
            return candidate, name
    return None


def _has_marker(path: Path, markers: tuple[str, ...]) -> bool:
    """Report whether a directory holds one of the entries that make it interesting."""
    if not markers:
        return True
    return any((path / marker).exists() for marker in markers)


def discover_sources() -> list[KnownLocation]:
    """Return every known tool location that exists on this machine.

    Environment overrides win outright: when ``HF_HOME`` points somewhere, the default
    ``~/.cache/huggingface`` is not also offered, because the user moved it for a reason
    and the old path is either gone or stale.
    """
    directories = _directories()
    found: dict[Path, KnownLocation] = {}

    for source in SOURCES:
        override = _override_for(source)
        candidates: list[tuple[Path, str]] = []

        if override is not None:
            candidates.append(override)
        else:
            for template in source.defaults:
                path = _expand(template, directories)
                if path is not None:
                    candidates.append((path, "default location"))

        for path, origin in candidates:
            try:
                if not path.is_dir() or not _has_marker(path, source.markers):
                    continue
                resolved = path.resolve()
            except OSError:
                continue

            if resolved not in found:
                found[resolved] = KnownLocation(
                    path=resolved, label=source.label, source=source.key,
                    group=source.group, origin=origin,
                )

    return sorted(found.values(), key=lambda item: (item.group, item.label.lower()))


def sweep_drives(roots: list[Path] | None = None) -> list[KnownLocation]:
    r"""Return likely asset folders found near the roots of each drive.

    This is the half that matters most on Windows. Caches live under the profile, but the
    library a person actually curates sits on whichever drive had room, in a folder called
    ``D:\Models`` or ``E:\AI\datasets``. Two levels down from each drive root finds those
    without ever touching a system tree.
    """
    directories = _directories()
    candidates = roots if roots is not None else _drive_roots()
    found: list[KnownLocation] = []
    seen: set[Path] = set()

    for root in candidates:
        for path in _sweep(root, depth=SWEEP_DEPTH):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or resolved == directories["home"]:
                continue
            seen.add(resolved)
            found.append(
                KnownLocation(
                    path=resolved, label=path.name, source="sweep",
                    group="Folders on your drives", origin=f"found on {root}",
                )
            )
            if len(found) >= MAX_DISCOVERED:
                return found

    return found


def _sweep(root: Path, *, depth: int) -> list[Path]:
    """Return interestingly named directories within ``depth`` levels of ``root``."""
    if depth <= 0:
        return []

    matches: list[Path] = []
    try:
        entries = sorted(root.iterdir())
    except (OSError, PermissionError):
        return []

    for entry in entries:
        try:
            if not entry.is_dir() or entry.is_symlink():
                continue
        except OSError:
            continue

        name = entry.name.lower()
        if name in SWEEP_EXCLUDED or name.startswith("$"):
            continue

        if name in SWEEP_NAMES:
            matches.append(entry)
            # Do not descend into something already offered; the scanner will recurse.
            continue

        matches.extend(_sweep(entry, depth=depth - 1))

    return matches


def discover(*, sweep: bool = True) -> list[KnownLocation]:
    """Return everything worth offering the user, tool locations first."""
    locations = discover_sources()
    if sweep:
        known = {location.path for location in locations}
        locations.extend(
            candidate for candidate in sweep_drives() if candidate.path not in known
        )
    return locations[:MAX_DISCOVERED]


def group_locations(locations: list[KnownLocation]) -> dict[str, list[KnownLocation]]:
    """Bucket locations by their display group, preserving order within each."""
    grouped: dict[str, list[KnownLocation]] = {}
    for location in locations:
        grouped.setdefault(location.group, []).append(location)
    return grouped


def _drive_roots() -> list[Path]:
    """Return the filesystem roots worth glancing at.

    On Windows that is every drive letter that responds; elsewhere it is the user's home,
    since Unix systems do not scatter a library across mount points the same way.
    """
    if os.name != "nt":
        return [_directories()["home"]]

    roots: list[Path] = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\")
        try:
            if root.is_dir():
                roots.append(root)
        except OSError:
            continue
    return roots


# -- backwards-compatible names --------------------------------------------


def known_locations() -> list[KnownLocation]:
    """Return every known tool location on this machine."""
    return discover_sources()


def likely_asset_folders(roots: list[Path] | None = None) -> list[KnownLocation]:
    """Return tool locations plus likely folders on the given (or all) drive roots."""
    locations = discover_sources()
    known = {location.path for location in locations}
    locations.extend(
        candidate for candidate in sweep_drives(roots) if candidate.path not in known
    )
    return locations[:MAX_DISCOVERED]
