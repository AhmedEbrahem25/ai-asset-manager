"""Helpers shared by the built-in plugins.

Underscore-prefixed so the loader skips it: this module registers nothing.

Nothing here is required to write a plugin — a plugin only has to expose ``register``.
These are the conveniences the built-ins found themselves repeating.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ai_asset_manager.backend.taxonomy.types import AssetProfile

#: A family table maps a display name to the markers that identify it. Ordered, because
#: markers overlap: "yolo-world" must be tested before "yolo", and "qwen2-vl" before
#: "qwen". Longest-first ordering within a table is the caller's responsibility.
FamilyTable = Sequence[tuple[str, Sequence[str]]]

#: Extensions that hold pictures. ``.webp`` and ``.avif`` are included because modern
#: scraped datasets are full of them.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp",
                    ".avif", ".ppm", ".pgm", ".jfif")

#: Extensions that hold video.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v",
                    ".wmv", ".flv", ".ts")

#: Extensions that hold audio.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
                    ".aiff", ".au")

#: Extensions used for tabular and record-oriented data.
TABULAR_EXTENSIONS = (".csv", ".tsv", ".parquet", ".arrow", ".feather", ".jsonl",
                      ".ndjson", ".orc", ".avro")

#: Extensions used for annotations and labels.
ANNOTATION_EXTENSIONS = (".json", ".xml", ".txt", ".csv", ".yaml", ".yml", ".pkl")

#: Extensions used for 3D point clouds and range data.
POINT_CLOUD_EXTENSIONS = (".pcd", ".ply", ".bin", ".las", ".laz", ".e57", ".obj",
                          ".off", ".xyz")

#: Directory and file names that mark a dataset split, mapped to the canonical name the
#: inventory reports. Datasets spell these a dozen ways; collapsing them is what makes
#: "is the validation split present?" answerable across formats.
SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "training": "train",
    "trainval": "train",
    "train2017": "train",
    "train2014": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "val2017": "val",
    "val2014": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "test2017": "test",
    "eval": "test",
    "unlabeled": "unlabeled",
    "unlabelled": "unlabeled",
    "raw": "unlabeled",
}

#: Basenames that count as a readme, in any casing or extension.
README_STEMS = ("readme", "read_me", "dataset_card", "datasetcard", "model_card",
                "modelcard", "index.md")

#: Basenames that count as a licence.
LICENCE_STEMS = ("license", "licence", "copying", "notice")

#: Suffixes left behind by an interrupted download. Their presence means the asset on disk
#: is not the asset the user thinks they have.
INCOMPLETE_SUFFIXES = (".incomplete", ".part", ".partial", ".download", ".tmp",
                       ".crdownload", ".aria2")


def family_of(profile: AssetProfile, table: FamilyTable) -> str | None:
    """Return the family whose markers appear in the asset's names.

    The asset's own name is searched first, and only then everything else the catalogue
    recorded. Order matters because distills carry two family names: ``deepseek-r1:8b`` is
    a Qwen3 model underneath, so its architecture says Qwen while its name says DeepSeek.
    The name is what the user typed when they pulled it and what they will look for, so
    the name wins.

    Examples:
        >>> table = (("YOLO", ("yolov8", "yolo")), ("SAM", ("sam2", "segment-anything")))
        >>> profile = AssetProfile(asset_id=1, kind="model", name="yolov8n", path="/x")
        >>> family_of(profile, table)
        'YOLO'
    """
    for scope in (profile.name.lower(), profile.haystack):
        for label, markers in table:
            if any(marker in scope for marker in markers):
                return label
    return None


def first_marker(profile: AssetProfile, markers: Iterable[str]) -> str | None:
    """Return the first marker present in the asset's names, or ``None``."""
    return profile.matches(markers)


def is_dataset(profile: AssetProfile) -> bool:
    """Report whether an asset should be judged by dataset rules."""
    return profile.is_dataset_like


def is_model(profile: AssetProfile) -> bool:
    """Report whether an asset should be judged by model rules."""
    return profile.is_model_like and not profile.is_dataset_like


def declared_format(profile: AssetProfile) -> str:
    """Return the dataset layout the scanner recorded, or an empty string."""
    return (profile.dataset.dataset_format if profile.dataset else None) or ""


def splits_present(profile: AssetProfile) -> set[str]:
    """Return the canonical split names this dataset appears to have.

    Two sources, both already in the catalogue: the split counts the scanner parsed from
    a manifest, and the directory names it recorded. A dataset laid out as
    ``images/train/`` has no manifest to parse, so directories are the only evidence.
    """
    found: set[str] = set()

    if profile.dataset is not None:
        for name in profile.dataset.splits:
            canonical = SPLIT_ALIASES.get(name.strip().lower())
            if canonical:
                found.add(canonical)

    for directory in profile.files.directories:
        canonical = SPLIT_ALIASES.get(directory)
        if canonical:
            found.add(canonical)

    return found


def has_readme(profile: AssetProfile) -> bool:
    """Report whether the asset ships documentation."""
    return profile.files.has_stem(*README_STEMS)


def has_licence(profile: AssetProfile) -> bool:
    """Report whether the asset ships a licence file."""
    return profile.files.has_stem(*LICENCE_STEMS)


def image_count(profile: AssetProfile) -> int:
    """Return the number of images, preferring the scanner's count over the file list.

    The scanner's number is authoritative when present; the file tally is the fallback for
    assets catalogued before dataset details existed, or by a detector that did not count.
    """
    if profile.dataset is not None and profile.dataset.num_images:
        return profile.dataset.num_images
    return profile.files.count(*IMAGE_EXTENSIONS)


def video_count(profile: AssetProfile) -> int:
    """Return the number of videos."""
    if profile.dataset is not None and profile.dataset.num_videos:
        return profile.dataset.num_videos
    return profile.files.count(*VIDEO_EXTENSIONS)


def audio_count(profile: AssetProfile) -> int:
    """Return the number of audio files."""
    if profile.dataset is not None and profile.dataset.num_audio_files:
        return profile.dataset.num_audio_files
    return profile.files.count(*AUDIO_EXTENSIONS)
