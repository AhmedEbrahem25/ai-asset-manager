"""Deciding what an archive holds from its table of contents.

The listing is the only evidence. That turns out to be enough far more often than it
sounds: an archive containing ``config.json``, ``tokenizer.json`` and ``model.safetensors``
is a model however it was compressed, and one containing ``images/train/``,
``labels/train/`` and ``data.yaml`` is a YOLO dataset whatever it is called.

Rules are ordered specific-first and the first to match wins, mirroring the detector
registry. Each returns the evidence it matched on, because "why is this a training
archive?" has to be answerable from the catalogue rather than by re-reading the zip.

An archive that no rule recognises is still catalogued. It is a large opaque object taking
up real space, and the inventory's job is to say so.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ai_asset_manager.backend.archives.reader import ArchiveListing
from ai_asset_manager.backend.models.enums import AssetFormat, AssetKind, Framework

#: Members whose presence marks the archive as holding weights.
_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".gguf",
    ".ggml",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".pb",
    ".h5",
    ".tflite",
    ".engine",
    ".pdparams",
)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
_CAPTURE_EXTENSIONS = (".pcap", ".pcapng", ".cap", ".erf", ".snoop")
_FLOW_EXTENSIONS = (".netflow", ".binetflow", ".flow", ".argus", ".ipfix")
_TABLE_EXTENSIONS = (".csv", ".tsv", ".parquet", ".arrow", ".feather", ".jsonl")

#: TensorBoard writes one of these per run and nothing else does.
_TFEVENTS = re.compile(r"events\.out\.tfevents")

#: A file named after its own SHA-256 or MD5 digest, which is how malware corpora and
#: sample repositories are laid out. Nothing benign names files this way in bulk.
_DIGEST_NAME = re.compile(r"^[0-9a-f]{32}(?:[0-9a-f]{32})?(?:\.[a-z0-9]{1,8})?$")


@dataclass(slots=True)
class ArchiveVerdict:
    """What an archive was judged to hold."""

    #: How the asset is filed. Always :attr:`~AssetKind.ARCHIVE`: the archive is the
    #: object on disk, and the taxonomy decides which section its subkind belongs in.
    kind: AssetKind = AssetKind.ARCHIVE
    #: Machine-readable classification, e.g. ``"model_archive"``.
    subkind: str = "archive"
    #: Human-readable classification, e.g. ``"Model Archive"``.
    label: str = "Archive"
    confidence: float = 0.5
    #: What the rule matched on, in the order it looked. Surfaced verbatim.
    signals: list[str] = field(default_factory=list)
    format: AssetFormat = AssetFormat.UNKNOWN
    framework: Framework = Framework.UNKNOWN
    #: Extra observations worth persisting, merged into the asset's evidence.
    extra: dict[str, object] = field(default_factory=dict)


#: A rule takes a listing and returns a verdict, or ``None`` to let the next rule try.
ArchiveRule = Callable[[ArchiveListing], ArchiveVerdict | None]


def classify_listing(listing: ArchiveListing) -> ArchiveVerdict:
    """Return what an archive holds.

    Always answers. When nothing matched, the verdict says so at low confidence rather
    than omitting the archive, so a 12 GB unrecognised zip still appears in the inventory
    where a user looking for space can see it.
    """
    verdict = next(
        (found for found in (rule(listing) for rule in _RULES) if found is not None),
        None,
    ) or _unrecognised(listing)

    # Annotated after the fallback rather than inside the loop. The archives that most
    # need `listing_error` recorded are exactly the ones no rule could classify, and
    # setting it only on the matched path meant it was never recorded at all — so the
    # health rule that reports an unreadable archive could not fire.
    if listing.truncated:
        verdict.extra["listing_truncated"] = True
    if listing.error:
        verdict.extra["listing_error"] = listing.error
    return verdict


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def _model_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed model repository.

    Weights alone are not enough — a training archive is full of ``.pt`` files too. What
    separates a model is a *declaration*: a ``config.json`` or ``model_index.json`` that
    says how to load the weights beside it.
    """
    weights = listing.count(*_WEIGHT_EXTENSIONS)
    if not weights:
        return None

    # Checked before the general case: an adapter ships an `adapter_config.json` and
    # nothing else that declares it, so the gate below would decline it and a 30 MB LoRA
    # would be filed as an unrecognised archive.
    if listing.has_name("adapter_config.json") or listing.has_name(
        "adapter_model.safetensors", "adapter_model.bin", "pytorch_lora_weights.safetensors"
    ):
        return ArchiveVerdict(
            subkind="adapter_archive",
            label="Adapter Archive",
            confidence=0.95,
            signals=_signals(
                listing,
                (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "pytorch_lora_weights.safetensors",
                ),
            ),
            format=_weight_format(listing),
            framework=Framework.PEFT,
        )

    declared = listing.has_name(
        "config.json", "model_index.json", "modules.json", "config.pbtxt"
    )
    companions = listing.has_name(
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "vocab.txt",
        "vocab.json",
        "special_tokens_map.json",
    )
    if not declared and not companions:
        return None

    signals: list[str] = []
    if declared:
        signals.append(_first_present(listing, ("config.json", "model_index.json", "modules.json")))
    if companions:
        signals.append(
            _first_present(listing, ("tokenizer.json", "tokenizer_config.json", "vocab.txt"))
        )
    signals.append(f"{weights} weight file(s)")

    return ArchiveVerdict(
        subkind="model_archive",
        label="Model Archive",
        confidence=0.95 if declared and companions else 0.85,
        signals=[signal for signal in signals if signal],
        format=_weight_format(listing),
        framework=(
            Framework.DIFFUSERS
            if listing.has_name("model_index.json")
            else Framework.TRANSFORMERS
            if declared
            else Framework.UNKNOWN
        ),
    )


def _yolo_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed YOLO dataset: paired image and label trees, plus a manifest."""
    has_manifest = listing.has_name("data.yaml", "data.yml", "dataset.yaml")
    has_images = listing.has_dir("images") or listing.matching("/images/", "images/")
    has_labels = listing.has_dir("labels") or listing.matching("/labels/", "labels/")

    if not (has_images and has_labels):
        return None
    if not has_manifest and not listing.count(*_IMAGE_EXTENSIONS):
        return None

    signals = ["images/ tree", "labels/ tree"]
    if has_manifest:
        signals.append(_first_present(listing, ("data.yaml", "data.yml", "dataset.yaml")))

    images = listing.count(*_IMAGE_EXTENSIONS)
    if images:
        signals.append(f"{images} image(s)")

    return ArchiveVerdict(
        subkind="yolo_dataset_archive",
        label="YOLO Dataset Archive",
        confidence=0.95 if has_manifest else 0.8,
        signals=signals,
        framework=Framework.ULTRALYTICS if has_manifest else Framework.UNKNOWN,
        extra={"images_listed": images},
    )


def _coco_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed COCO dataset by its annotation file naming."""
    annotations = listing.matching(
        "annotations/instances_", "annotations/captions_", "annotations/person_keypoints_"
    )
    if not annotations:
        return None

    images = listing.count(*_IMAGE_EXTENSIONS)
    return ArchiveVerdict(
        subkind="coco_dataset_archive",
        label="COCO Dataset Archive",
        confidence=0.95,
        signals=[f"{annotations} COCO annotation file(s)", f"{images} image(s)"],
        extra={"images_listed": images},
    )


def _tracking_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed MOT-style tracking dataset.

    Worth a rule of its own because the generic media rule gets there first and is wrong in
    a way that matters: MOT17 is a tracking benchmark with ground-truth trajectories, and
    calling it "an image dataset" loses the only thing that distinguishes it. The
    ``seqinfo.ini`` beside a ``gt/`` or ``det/`` folder is the MOT layout exactly.
    """
    seqinfo = listing.matching("seqinfo.ini")
    ground_truth = listing.matching("/gt/gt.txt", "gt/gt.txt")
    detections = listing.matching("/det/det.txt", "det/det.txt")

    if not seqinfo or not (ground_truth or detections):
        return None

    signals = [f"{seqinfo} seqinfo.ini sequence descriptor(s)"]
    if ground_truth:
        signals.append(f"{ground_truth} ground-truth track file(s)")
    if detections:
        signals.append(f"{detections} detection file(s)")

    images = listing.count(*_IMAGE_EXTENSIONS)
    return ArchiveVerdict(
        subkind="tracking_dataset_archive",
        label="Tracking Dataset Archive",
        confidence=0.95,
        signals=signals,
        extra={"images_listed": images},
    )


def _hf_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed HuggingFace dataset: a manifest beside arrow or parquet shards."""
    declared = listing.has_name(
        "dataset_info.json", "dataset_infos.json", "dataset_dict.json", "state.json"
    )
    shards = listing.count(".arrow", ".parquet")
    if not declared and shards < 2:
        return None

    signals: list[str] = []
    if declared:
        signals.append(
            _first_present(listing, ("dataset_info.json", "dataset_infos.json", "state.json"))
        )
    if shards:
        signals.append(f"{shards} arrow/parquet shard(s)")

    return ArchiveVerdict(
        subkind="hf_dataset_archive",
        label="HuggingFace Dataset Archive",
        confidence=0.95 if declared else 0.75,
        signals=signals,
        extra={"shards_listed": shards},
    )


def _training_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed training run: event files, run metadata, epoch checkpoints.

    Ranked below the model rules because a run archive contains weights too. What makes it
    a run rather than a model is the *history* beside them — the event stream, the
    per-epoch files, the ``wandb`` directory.
    """
    events = sum(1 for entry in listing.entries if _TFEVENTS.search(entry.name))
    wandb = listing.has_dir("wandb") or listing.matching("wandb/")
    run_meta = listing.has_name(
        "wandb-metadata.json", "meta.yaml", "mlmodel", "results.csv", "hparams.yaml", "args.yaml"
    )
    epochs = listing.matching("epoch", "checkpoint", "ckpt", "last.pt", "best.pt")

    score = sum((bool(events), bool(wandb), bool(run_meta), epochs >= 2))
    if score < 2:
        return None

    signals: list[str] = []
    if events:
        signals.append(f"{events} TensorBoard event file(s)")
    if wandb:
        signals.append("wandb/ directory")
    if run_meta:
        signals.append(
            _first_present(listing, ("wandb-metadata.json", "results.csv", "hparams.yaml"))
        )
    if epochs >= 2:
        signals.append(f"{epochs} checkpoint-shaped member(s)")

    return ArchiveVerdict(
        kind=AssetKind.ARCHIVE,
        subkind="training_archive",
        label="Training Archive",
        confidence=0.85,
        signals=signals,
    )


def _malware_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed malware or sample corpus.

    Deliberately conservative and never inferred from one signal. A ``samples/`` folder is
    a folder; a ``samples/`` folder full of digest-named files beside a hash manifest is a
    malware corpus, and the difference matters because the label is one a user will act on.
    """
    marker_dirs = [
        name
        for name in ("malware", "samples", "benign", "malicious", "goodware", "binaries", "apks")
        if listing.has_dir(name)
    ]
    digest_named = sum(
        1 for entry in listing.entries if _DIGEST_NAME.match(entry.basename)
    )
    hash_manifest = listing.has_name(
        "sha256sums.txt", "md5sums.txt", "hashes.txt", "sha256.txt", "labels.csv"
    )
    family_words = listing.matching(
        "trojan", "ransomware", "backdoor", "botnet", "worm", "spyware", "virusshare",
        "malimg", "ember", "malmem",
    )

    score = sum((len(marker_dirs) >= 1, digest_named >= 5, hash_manifest, family_words >= 3))
    if score < 2:
        return None

    signals: list[str] = []
    if marker_dirs:
        signals.append(f"{'/, '.join(marker_dirs)}/ director(y/ies)")
    if digest_named >= 5:
        signals.append(f"{digest_named} digest-named sample(s)")
    if hash_manifest:
        signals.append(_first_present(listing, ("sha256sums.txt", "hashes.txt", "labels.csv")))
    if family_words >= 3:
        signals.append(f"{family_words} member(s) naming a malware family")

    return ArchiveVerdict(
        subkind="malware_dataset_archive",
        label="Malware Dataset Archive",
        confidence=0.85,
        signals=signals,
    )


def _capture_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed network capture or flow corpus."""
    captures = listing.count(*_CAPTURE_EXTENSIONS)
    flows = listing.count(*_FLOW_EXTENSIONS)
    zeek = listing.matching("conn.log", "dns.log", "http.log", "ssl.log", "notice.log")
    suricata = listing.has_name("eve.json", "fast.log", "suricata.yaml")

    if captures + flows + zeek == 0 and not suricata:
        return None

    signals: list[str] = []
    if captures:
        signals.append(f"{captures} packet capture(s)")
    if flows:
        signals.append(f"{flows} flow record file(s)")
    if zeek:
        signals.append(f"{zeek} Zeek log(s)")
    if suricata:
        signals.append("Suricata output")

    return ArchiveVerdict(
        subkind="network_dataset_archive",
        label="Network Capture Archive",
        confidence=0.9 if captures or zeek or suricata else 0.75,
        signals=signals,
    )


def _windows_log_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed Windows event log or Sysmon collection."""
    evtx = listing.count(".evtx", ".etl")
    sysmon = listing.matching("sysmon", "security.evtx", "microsoft-windows-")
    if evtx < 2 and not (evtx and sysmon):
        return None

    signals = [f"{evtx} Windows event log(s)"]
    if sysmon:
        signals.append(f"{sysmon} Sysmon/Windows channel member(s)")

    return ArchiveVerdict(
        subkind="host_log_dataset_archive",
        label="Host Log Archive",
        confidence=0.85,
        signals=signals,
    )


#: Names inside an archive that mean somebody *assembled* the media rather than merely
#: collecting it: a split layout, labels beside the samples, a class list.
_ASSEMBLY_MARKERS: tuple[str, ...] = (
    "train/", "val/", "valid/", "test/", "eval/", "training/", "validation/", "testing/",
    "labels/", "annotations/", "masks/", "ground_truth/", "gt/",
    "classes.txt", "labels.txt", "metadata.csv", "metadata.jsonl", "annotations.json",
)


def _media_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed collection of media.

    Bulk is not evidence on its own — the same rule the boundary guard applies on disk. A
    thousand lecture videos and a thousand training clips are the same shape, and this
    machine has several gigabytes of the former. So an archive full of media is only a
    *dataset* when something inside says it was assembled: a split layout, labels beside the
    samples, or a class list. Without that it is still catalogued, still sized, and still
    described — just not as a dataset.
    """
    images = listing.count(*_IMAGE_EXTENSIONS)
    videos = listing.count(*_VIDEO_EXTENSIONS)
    audio = listing.count(*_AUDIO_EXTENSIONS)
    total = images + videos + audio
    if total < 25:
        return None

    dominant, count = max(
        (("image", images), ("video", videos), ("audio", audio)),
        key=lambda item: item[1],
    )
    assembled = listing.matching(*_ASSEMBLY_MARKERS)

    signals = [f"{count} {dominant} file(s)"]
    if assembled:
        signals.append(f"{assembled} member(s) under a split or label tree")
    if listing.truncated:
        signals.append(f"listing capped at {len(listing.entries)} entries")

    extra: dict[str, object] = {
        "images_listed": images,
        "videos_listed": videos,
        "audio_listed": audio,
    }

    if not assembled:
        return ArchiveVerdict(
            subkind="media_archive",
            label=f"{dominant.title()} Archive",
            confidence=0.6,
            signals=[*signals, "no split, label or class list: not catalogued as a dataset"],
            extra=extra,
        )

    return ArchiveVerdict(
        subkind=f"{dominant}_dataset_archive",
        label=f"{dominant.title()} Dataset Archive",
        confidence=0.75,
        signals=signals,
        extra=extra,
    )


def _tabular_dataset_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed table collection.

    Three tables and up, because one CSV in a zip is a spreadsheet somebody mailed and not
    a dataset — the same rule the cybersecurity detector applies on disk.
    """
    tables = listing.count(*_TABLE_EXTENSIONS)
    if tables < 3:
        return None

    return ArchiveVerdict(
        subkind="tabular_dataset_archive",
        label="Tabular Dataset Archive",
        confidence=0.65,
        signals=[f"{tables} table file(s)"],
        extra={"tables_listed": tables},
    )


def _code_archive(listing: ArchiveListing) -> ArchiveVerdict | None:
    """Recognise a packed codebase, so it is not mistaken for data."""
    sources = listing.count(".py", ".ipynb", ".js", ".ts", ".java", ".cpp", ".go", ".rs")
    manifest = listing.has_name(
        "requirements.txt", "pyproject.toml", "setup.py", "package.json", "cargo.toml"
    )
    if sources < 5 or not manifest:
        return None

    return ArchiveVerdict(
        subkind="code_archive",
        label="Code Archive",
        confidence=0.7,
        signals=[
            f"{sources} source file(s)",
            _first_present(listing, ("pyproject.toml", "requirements.txt", "package.json")),
        ],
    )


def _unrecognised(listing: ArchiveListing) -> ArchiveVerdict:
    """Describe an archive nothing claimed."""
    if not listing.is_listed:
        reason = listing.error or "no table of contents recovered"
        return ArchiveVerdict(
            subkind="archive",
            label="Archive",
            confidence=0.3,
            signals=[f"contents not listed ({reason})"],
        )

    ranked = sorted(
        listing.extension_counts.items(), key=lambda pair: (-pair[1], pair[0])
    )[:3]
    signals = [f"{len(listing.file_entries)} member(s)"]
    if ranked:
        signals.append(
            "mostly " + ", ".join(f"{ext} ({count})" for ext, count in ranked)
        )

    return ArchiveVerdict(
        subkind="archive",
        label="Archive",
        confidence=0.4,
        signals=signals,
    )


#: Ordered specific-first. Model rules precede training rules because a run archive holds
#: weights, and capture rules precede the generic table rule because an intrusion dataset
#: is a pile of CSVs to anything that only counts extensions.
_RULES: tuple[ArchiveRule, ...] = (
    _model_archive,
    _yolo_dataset_archive,
    _coco_dataset_archive,
    _tracking_dataset_archive,
    _hf_dataset_archive,
    _capture_archive,
    _windows_log_archive,
    _malware_archive,
    _training_archive,
    _media_dataset_archive,
    _code_archive,
    _tabular_dataset_archive,
)


def _weight_format(listing: ArchiveListing) -> AssetFormat:
    """Return the storage format of the weights inside an archive."""
    for extension, asset_format in (
        (".safetensors", AssetFormat.SAFETENSORS),
        (".gguf", AssetFormat.GGUF),
        (".onnx", AssetFormat.ONNX),
        (".tflite", AssetFormat.TFLITE),
        (".pdparams", AssetFormat.PADDLE),
        (".h5", AssetFormat.KERAS),
        (".pb", AssetFormat.TENSORFLOW),
        (".pt", AssetFormat.PYTORCH),
        (".pth", AssetFormat.PYTORCH),
        (".ckpt", AssetFormat.PYTORCH),
        (".bin", AssetFormat.PYTORCH),
    ):
        if listing.count(extension):
            return asset_format
    return AssetFormat.UNKNOWN


def _first_present(listing: ArchiveListing, names: tuple[str, ...]) -> str:
    """Return the first of these basenames that the archive contains."""
    present = listing.basenames
    for name in names:
        if name.lower() in present:
            return name
    return ""


def _signals(listing: ArchiveListing, names: tuple[str, ...]) -> list[str]:
    """Return every one of these basenames the archive contains."""
    present = listing.basenames
    return [name for name in names if name.lower() in present]
