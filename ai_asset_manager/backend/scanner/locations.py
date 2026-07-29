"""Well-known places AI assets end up.

The first question a new user of this tool has is "what do I point it at?", and the honest
answer is that they mostly do not know: the whole reason the catalogue is useful is that
downloads scatter themselves across caches nobody chose. So the tool works it out.

Every location here is checked for existence before being offered — a suggestion that
turns out to be an empty path is worse than no suggestion. Environment overrides are
honoured first, because a user who has moved their HuggingFace cache to another drive has
usually done it precisely because it got too big to ignore.

Reusable beyond the CLI: the dashboard's "add a folder" screen wants the same list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Sensible cap on how deep to look for the nested layouts below.
_MAX_CANDIDATES = 40


@dataclass(frozen=True, slots=True)
class KnownLocation:
    """A directory that probably holds AI assets."""

    path: Path
    label: str
    #: What put it there, shown so the user can recognise it.
    source: str

    def __str__(self) -> str:
        """Return the path as text."""
        return str(self.path)


def _home() -> Path:
    """Return the user's home directory."""
    return Path.home()


def _from_env(*names: str) -> Path | None:
    """Return the first environment variable that names an existing directory."""
    for name in names:
        value = os.environ.get(name)
        if value:
            candidate = Path(value).expanduser()
            if candidate.is_dir():
                return candidate
    return None


def _tool_locations() -> list[tuple[Path | None, str, str]]:
    """Return ``(path, label, source)`` for every tool this module knows about."""
    home = _home()

    return [
        (_from_env("HF_HOME", "HUGGINGFACE_HUB_CACHE") or home / ".cache" / "huggingface",
         "HuggingFace cache", "transformers, diffusers, datasets"),
        (_from_env("OLLAMA_MODELS") or home / ".ollama",
         "Ollama", "ollama pull"),
        (_from_env("TORCH_HOME") or home / ".cache" / "torch",
         "PyTorch hub cache", "torchvision, torch.hub"),
        (home / ".cache" / "kagglehub", "Kaggle cache", "kagglehub"),
        (home / ".cache" / "lm-studio" / "models", "LM Studio", "LM Studio"),
        (home / ".lmstudio" / "models", "LM Studio", "LM Studio"),
        (home / ".cache" / "modelscope", "ModelScope cache", "modelscope"),
        (home / ".paddleocr", "PaddleOCR", "paddleocr"),
        (home / ".EasyOCR", "EasyOCR", "easyocr"),
        (home / ".insightface", "InsightFace", "insightface"),
        (home / ".u2net", "rembg", "rembg"),
        (home / "fiftyone", "FiftyOne", "fiftyone"),
        (home / ".cache" / "whisper", "Whisper", "openai-whisper"),
        (home / "ComfyUI" / "models", "ComfyUI", "ComfyUI"),
        (home / "stable-diffusion-webui" / "models", "Automatic1111", "stable-diffusion-webui"),
        (home / "text-generation-webui" / "models", "text-generation-webui",
         "text-generation-webui"),
    ]


def known_locations() -> list[KnownLocation]:
    """Return every well-known asset directory that exists on this machine.

    Deduplicated by resolved path: several tools share the HuggingFace cache, and offering
    it three times under three names would be worse than offering it once.
    """
    found: dict[Path, KnownLocation] = {}

    for path, label, source in _tool_locations():
        if path is None or not path.is_dir():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in found:
            found[resolved] = KnownLocation(path=resolved, label=label, source=source)

    return sorted(found.values(), key=lambda location: location.label.lower())


def likely_asset_folders(roots: list[Path] | None = None) -> list[KnownLocation]:
    r"""Return known locations plus obvious model and dataset folders on each drive.

    The second half matters more than the first on Windows: caches live on ``C:``, but the
    library a person actually curates tends to sit on whichever drive had room, in a folder
    called something like ``D:\\Models``. Those are found by looking one level down from
    each drive root rather than by walking anything.
    """
    locations = known_locations()
    seen = {location.path for location in locations}

    for root in roots if roots is not None else _drive_roots():
        for name in ("Models", "models", "AI", "ai", "Datasets", "datasets", "LLMs",
                     "llms", "Checkpoints", "checkpoints", "AI-Models", "ML"):
            candidate = root / name
            if not candidate.is_dir():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            locations.append(
                KnownLocation(path=resolved, label=candidate.name, source=f"on {root}")
            )
            if len(locations) >= _MAX_CANDIDATES:
                return locations

    return locations


def _drive_roots() -> list[Path]:
    """Return the filesystem roots worth glancing at.

    On Windows that is every fixed drive letter; elsewhere it is the user's home, since
    Unix systems do not scatter a library across mount points the same way.
    """
    if os.name != "nt":
        return [_home()]

    roots: list[Path] = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\")
        try:
            if root.is_dir():
                roots.append(root)
        except OSError:
            continue
    return roots
