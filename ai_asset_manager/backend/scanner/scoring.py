r"""How AI-ish a directory looks, from one ``scandir`` and nothing else.

Deep discovery has to decide where to go next before it has been there. The only budget
that allows is the listing it already has: names of files, names of subdirectories, and
nothing read, opened or hashed. Every signal below is computed from that listing, which is
why a whole-drive pass costs seconds rather than minutes.

The score is not a probability and does not need to be. It is used for two decisions —
*is this worth descending into* and *is this worth showing the user* — and both are
threshold comparisons where being roughly right is enough. What matters far more is that
the negative signals are honest: a directory of ISOs must score low enough that the sweep
turns around, because the time saved there is what pays for looking everywhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: Weight formats. Their presence is the single strongest signal available: nothing but a
#: model ships a ``.safetensors``.
#:
#: ``.bin`` is deliberately absent even though the detectors accept it. A detector sees
#: ``pytorch_model.bin`` beside ``config.json`` and is right; a scorer sees only an
#: extension that Electron apps, webcam drivers and game installers all use for their own
#: blobs. Including it sent the deep sweep to Riot Games and iVCam and offered them as AI
#: libraries. Where ``.bin`` really is a model, the manifest beside it scores anyway.
WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors", ".gguf", ".ggml", ".onnx", ".pt", ".pth", ".ckpt",
    ".engine", ".plan", ".tflite", ".mlmodel", ".pdparams", ".pdmodel", ".h5", ".keras",
    ".npz", ".msgpack",
)

#: Files a model or dataset directory carries and an ordinary folder does not.
MANIFEST_FILES: frozenset[str] = frozenset(
    {
        "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
        "adapter_config.json", "model_index.json", "modules.json", "preprocessor_config.json",
        "special_tokens_map.json", "model.safetensors.index.json",
        "data.yaml", "data.yml", "dataset.yaml", "dataset_info.json", "dataset_infos.json",
        "dataset_dict.json", "classes.txt", "labels.txt", "metadata.jsonl",
        "meta.yaml", "args.yaml", "hparams.yaml", "results.csv", "wandb-metadata.json",
    }
)

#: Directory names that state what the tree beneath them is for.
SIGNAL_DIRS: frozenset[str] = frozenset(
    {
        "snapshots", "blobs", "refs", "checkpoints", "weights", "annotations", "labels",
        "images", "wandb", "mlruns", "runs", "lightning_logs", "tensorboard", "configs",
        "notebooks", "adapters", "loras", "embeddings", "tokenizer",
    }
)

#: Split names. Two or more of them together is a dataset's signature.
SPLIT_DIRS: frozenset[str] = frozenset(
    {"train", "val", "valid", "validation", "test", "testing", "training", "eval", "dev"}
)

#: Development files that suggest a codebase which might train something.
CODE_FILES: frozenset[str] = frozenset(
    {
        "requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml",
        "train.py", "infer.py", "predict.py", "evaluate.py", "export.py", "finetune.py",
    }
)

#: Suffixes whose presence in bulk means this tree is something else entirely. Installers,
#: disc images and media libraries are the three things most likely to be large, deep and
#: completely irrelevant.
DISTRACTOR_SUFFIXES: tuple[str, ...] = (
    ".exe", ".msi", ".iso", ".vhd", ".vhdx", ".vmdk", ".dll", ".sys", ".cab",
    ".mp4", ".mkv", ".avi", ".srt", ".pdf", ".epub", ".docx", ".pptx", ".zip", ".rar",
)

#: Directory names that hold assets and are therefore worth entering even when the
#: directory itself shows nothing — the listing of ``D:\Models`` is just folder names.
CONTAINER_HINTS: frozenset[str] = frozenset(
    {
        "models", "model", "datasets", "dataset", "data", "ai", "ml", "llm", "llms",
        "checkpoints", "weights", "huggingface", "gguf", "loras", "embeddings",
        "training", "experiments", "runs", "cache", ".cache", "hub", "comfyui",
        "stable-diffusion", "projects", "project", "research", "work",
    }
)

#: Score at or above which a directory is worth descending into.
DESCEND_THRESHOLD = 0.30

#: Score at or above which a directory is worth offering to the user.
REPORT_THRESHOLD = 0.80


@dataclass(frozen=True, slots=True)
class DirectoryScore:
    """How promising a directory looks, and why."""

    score: float
    evidence: tuple[str, ...] = field(default=())

    @property
    def worth_descending(self) -> bool:
        """Report whether the sweep should look inside this directory."""
        return self.score >= DESCEND_THRESHOLD

    @property
    def worth_reporting(self) -> bool:
        """Report whether this directory should be offered to the user."""
        return self.score >= REPORT_THRESHOLD


def score_directory(
    name: str, filenames: Sequence[str], dirnames: Sequence[str]
) -> DirectoryScore:
    """Score one directory from its listing.

    Args:
        name: The directory's own name.
        filenames: Names of the files directly inside it.
        dirnames: Names of the subdirectories directly inside it.

    Returns:
        A score with the evidence that produced it, so a surprising result can be
        explained rather than merely disbelieved.
    """
    lower_files = [item.lower() for item in filenames]
    lower_dirs = {item.lower() for item in dirnames}
    evidence: list[str] = []
    score = 0.0

    weights = sum(1 for item in lower_files if item.endswith(WEIGHT_SUFFIXES))
    if weights:
        score += 1.0
        evidence.append(f"{weights} weight file(s)")

    manifests = sorted({item for item in lower_files if item in MANIFEST_FILES})
    if manifests:
        # Two manifests are much better than one: `config.json` alone appears in every
        # JavaScript project ever written, while `config.json` beside `tokenizer.json` is
        # a model and nothing else.
        score += 0.5 if len(manifests) == 1 else 0.9
        evidence.append(", ".join(manifests[:3]))

    if any(item.startswith("events.out.tfevents") for item in lower_files):
        score += 1.0
        evidence.append("tensorboard events")

    if any("-idx" in item and "ubyte" in item for item in lower_files):
        score += 1.0
        evidence.append("idx binary dataset")

    # A cache *root* shows nothing of itself — its listing is just folder names — but the
    # shape of those names gives it away, and it is the directory worth pointing a scan at
    # rather than the individual repositories beneath it.
    cache_entries = sum(
        1 for item in lower_dirs if item.startswith(("models--", "datasets--"))
    )
    if cache_entries:
        score += 1.0
        evidence.append(f"{cache_entries} cached repo(s)")

    if {"manifests", "blobs"} <= lower_dirs:
        score += 1.0
        evidence.append("ollama store")

    signal_dirs = sorted(lower_dirs & SIGNAL_DIRS)
    if signal_dirs:
        score += min(0.6, 0.3 * len(signal_dirs))
        evidence.append("/".join(signal_dirs[:3]))

    splits = sorted(lower_dirs & SPLIT_DIRS)
    if len(splits) >= 2:
        score += 0.6
        evidence.append("+".join(splits[:3]))

    notebooks = sum(1 for item in lower_files if item.endswith(".ipynb"))
    if notebooks:
        score += 0.3
        evidence.append(f"{notebooks} notebook(s)")

    code = sorted({item for item in lower_files if item in CODE_FILES})
    if code:
        score += 0.25 if len(code) == 1 else 0.5
        evidence.append(", ".join(code[:2]))

    lowered_name = name.lower()
    if lowered_name in CONTAINER_HINTS:
        score += 0.4
        evidence.append(f"named {name!r}")
    if lowered_name.startswith(("models--", "datasets--")):
        score += 1.0
        evidence.append("huggingface cache entry")

    distractors = sum(1 for item in lower_files if item.endswith(DISTRACTOR_SUFFIXES))
    if distractors and distractors >= max(3, len(lower_files) // 2):
        # Not a fixed penalty: a directory that is *mostly* installers or video is a
        # different kind of thing, and pretending otherwise is what makes a deep sweep
        # spend its budget in a course library.
        score -= 0.8
        evidence.append(f"{distractors} unrelated file(s)")

    return DirectoryScore(score=max(0.0, round(score, 3)), evidence=tuple(evidence))


def best_evidence(scores: Iterable[DirectoryScore], limit: int = 3) -> tuple[str, ...]:
    """Merge the evidence from several scores, keeping the first mentions."""
    seen: dict[str, None] = {}
    for item in scores:
        for reason in item.evidence:
            seen.setdefault(reason, None)
    return tuple(seen)[:limit]
