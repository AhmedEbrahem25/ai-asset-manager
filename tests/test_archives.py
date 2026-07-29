"""Archive listing, classification and detection.

The load-bearing assertion in this file is :func:`test_nothing_is_ever_extracted`: the whole
feature is worthless if it writes a temporary directory somewhere, and the guarantee is
easier to break than to notice.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from ai_asset_manager.backend.archives import (
    archive_format,
    classify_listing,
    inspect_archive,
    is_archive_name,
)
from ai_asset_manager.backend.archives.reader import MAX_ENTRIES, MAX_METADATA_BYTES
from ai_asset_manager.backend.detectors.archives import (
    MAX_ARCHIVES_PER_DIR,
    MIN_ARCHIVE_BYTES,
    ArchiveDetector,
)
from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.taxonomy.plugins.archives import _listing_hint

PADDING = b"\0" * 4096


def _padding() -> dict[str, bytes]:
    """Return a member that pushes an archive past the detector's minimum size.

    Incompressible, because the size that matters is the archive's on disk. The extension
    is deliberately one no classification rule looks at, so padding never changes a verdict.
    """
    return {"filler.dat": os.urandom(MIN_ARCHIVE_BYTES)}


def _zip(path: Path, members: dict[str, bytes], *, pad: bool = False) -> Path:
    """Write a zip with the given members and return its path."""
    if pad:
        members = {**members, **_padding()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _tar(
    path: Path, members: dict[str, bytes], *, mode: str = "w:gz", pad: bool = False
) -> Path:
    """Write a tar with the given members and return its path."""
    if pad:
        members = {**members, **_padding()}
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def test_recognises_archive_names():
    assert is_archive_name("dataset.tar.gz")
    assert is_archive_name("MODEL.ZIP")
    assert not is_archive_name("model.safetensors")
    assert archive_format("corpus.tgz") == "tar.gz"
    assert archive_format("corpus.7z") == "7z"
    assert archive_format("weights.bin") is None


def test_lists_zip_members_without_extracting(tmp_path: Path):
    path = _zip(tmp_path / "bundle.zip", {"a/b/config.json": b"{}", "a/weights.bin": PADDING})
    listing = inspect_archive(str(path), path.stat().st_size)

    assert listing.error is None
    assert listing.has_name("config.json")
    assert listing.count(".bin") == 1
    assert "a" in listing.top_level
    assert listing.has_dir("b")


def test_reads_only_allow_listed_metadata(tmp_path: Path):
    path = _zip(
        tmp_path / "m.zip",
        {
            "config.json": json.dumps({"architectures": ["BertModel"]}).encode(),
            "secret.txt": b"not metadata",
            "model.safetensors": PADDING,
        },
    )
    listing = inspect_archive(str(path), path.stat().st_size)

    assert listing.json_metadata("config.json") == {"architectures": ["BertModel"]}
    assert "secret.txt" not in listing.metadata
    assert "model.safetensors" not in listing.metadata


def test_oversized_metadata_is_left_in_the_archive(tmp_path: Path):
    path = _zip(tmp_path / "m.zip", {"config.json": b"x" * (MAX_METADATA_BYTES + 1)})
    listing = inspect_archive(str(path), path.stat().st_size)

    assert listing.has_name("config.json")
    assert listing.metadata == {}


def test_lists_compressed_tar(tmp_path: Path):
    path = _tar(
        tmp_path / "ds.tar.gz",
        {"images/train/a.jpg": PADDING, "labels/train/a.txt": b"0 1 1 1 1", "data.yaml": b"nc: 1"},
    )
    listing = inspect_archive(str(path), path.stat().st_size)

    assert listing.error is None
    assert listing.has_name("data.yaml")
    assert listing.metadata["data.yaml"] == b"nc: 1"


def test_entry_cap_is_reported_rather_than_exceeded(tmp_path: Path):
    members = {f"img/{index}.jpg": b"x" for index in range(MAX_ENTRIES + 50)}
    path = _zip(tmp_path / "many.zip", members)
    listing = inspect_archive(str(path), path.stat().st_size)

    assert len(listing.entries) == MAX_ENTRIES
    assert listing.truncated


def test_corrupt_archive_is_reported_not_raised(tmp_path: Path):
    path = tmp_path / "broken.zip"
    path.write_bytes(b"PK\x03\x04 this is not really a zip")

    listing = inspect_archive(str(path), path.stat().st_size)
    assert listing.error is not None
    assert listing.entries == []


def test_unreadable_format_degrades_to_name_and_size(tmp_path: Path):
    path = tmp_path / "CICIDS2017.7z"
    path.write_bytes(b"7z\xbc\xaf\x27\x1c" + PADDING)

    listing = inspect_archive(str(path), path.stat().st_size)
    verdict = classify_listing(listing)

    # Whatever the reason, the archive is still catalogued and says why it is opaque.
    assert verdict.subkind == "archive"
    assert verdict.signals
    # Recorded, not merely mentioned: the health rule reads this. It was set only on the
    # classified path, which is every path except the one that needs it.
    assert verdict.extra["listing_error"]


def test_an_msys2_package_is_not_reported_as_damaged(tmp_path: Path):
    """A `.tar.zst` has no reader here; that is our limitation, not the archive's fault.

    Fourteen perfectly good msys2 packages on the development machine were each told they
    "may be encrypted or damaged", because the hint tested the error prose for the phrase
    "not installed" and a missing *format* reader does not say that.
    """
    path = tmp_path / "mingw-w64-x86_64-gcc-15.2.0-8-any.pkg.tar.zst"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + PADDING)

    listing = inspect_archive(str(path), path.stat().st_size)
    assert listing.missing_reader is True

    hint = _listing_hint(str(listing.error))
    assert "damaged" not in hint
    # Nor is it fixed by the optional extra: that ships 7z and rar readers, not zstd.
    assert "pip install" not in hint


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("py7zr not installed", "pip install"),
        ("rarfile not installed", "pip install"),
        ("no reader for tar.zst", "no reader in this build"),
        ("BadZipFile: File is not a zip file", "damaged"),
        ("RuntimeError: File is encrypted, password required", "damaged"),
    ],
)
def test_the_listing_hint_matches_the_cause(error: str, expected: str):
    assert expected in _listing_hint(error)


def test_a_truncated_listing_says_so(tmp_path: Path):
    members = {f"img/{index}.jpg": b"x" for index in range(MAX_ENTRIES + 50)}
    path = _zip(tmp_path / "many.zip", members)

    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))
    assert verdict.extra["listing_truncated"] is True


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (
            {
                "config.json": b"{}",
                "tokenizer.json": b"{}",
                "model.safetensors": PADDING,
            },
            "model_archive",
        ),
        (
            {
                "adapter_config.json": b"{}",
                "adapter_model.safetensors": PADDING,
            },
            "adapter_archive",
        ),
        (
            {
                "data.yaml": b"nc: 2",
                "images/train/a.jpg": b"x",
                "labels/train/a.txt": b"0 0 0 0 0",
            },
            "yolo_dataset_archive",
        ),
        (
            {
                "dataset_info.json": b"{}",
                "train/data-00000.arrow": PADDING,
                "train/data-00001.arrow": PADDING,
            },
            "hf_dataset_archive",
        ),
        (
            {
                "run/events.out.tfevents.1700000000.host": PADDING,
                "run/wandb/wandb-metadata.json": b"{}",
                "run/checkpoint_epoch1.pt": PADDING,
            },
            "training_archive",
        ),
        (
            {
                "samples/" + "a" * 32: b"x",
                "samples/" + "b" * 32: b"x",
                "samples/" + "c" * 32: b"x",
                "samples/" + "d" * 32: b"x",
                "samples/" + "e" * 32: b"x",
                "sha256sums.txt": b"",
            },
            "malware_dataset_archive",
        ),
        (
            {"captures/day1.pcap": PADDING, "captures/day2.pcapng": PADDING},
            "network_dataset_archive",
        ),
        (
            {"annotations/instances_train2017.json": b"{}", "train2017/1.jpg": b"x"},
            "coco_dataset_archive",
        ),
    ],
)
def test_classifies_from_the_table_of_contents(tmp_path: Path, members, expected):
    path = _zip(tmp_path / "a.zip", members)
    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))

    assert verdict.subkind == expected
    assert verdict.signals, "a verdict must say what it matched on"


def test_a_run_archive_is_not_read_as_a_model(tmp_path: Path):
    """Weights alone do not make a model; a run archive is full of them."""
    path = _zip(
        tmp_path / "run.zip",
        {
            "events.out.tfevents.1700000000.host": PADDING,
            "epoch0.pt": PADDING,
            "epoch1.pt": PADDING,
            "results.csv": b"epoch,loss\n",
        },
    )
    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))
    assert verdict.subkind == "training_archive"


def test_bulk_media_alone_is_not_a_dataset(tmp_path: Path):
    """A course download and a training set are the same shape; only one was assembled."""
    path = _zip(
        tmp_path / "course.zip",
        {f"Week {index // 10}/lecture{index}.mp4": b"x" for index in range(40)},
    )
    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))

    assert verdict.subkind == "media_archive"
    assert any("not catalogued as a dataset" in signal for signal in verdict.signals)


def test_media_under_a_split_layout_is_a_dataset(tmp_path: Path):
    members = {f"train/{index}.jpg": b"x" for index in range(30)}
    members.update({f"labels/{index}.txt": b"0" for index in range(30)})
    path = _zip(tmp_path / "corpus.zip", members)

    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))

    assert verdict.subkind == "image_dataset_archive"


def test_one_csv_in_a_zip_is_not_a_dataset(tmp_path: Path):
    path = _zip(tmp_path / "sheet.zip", {"quarterly.csv": b"a,b\n1,2\n"})
    verdict = classify_listing(inspect_archive(str(path), path.stat().st_size))
    assert verdict.subkind == "archive"


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_detects_an_archive_as_an_asset(tmp_path: Path, context_for):
    _zip(tmp_path / "qwen.zip", {"config.json": b"{}", "model.safetensors": PADDING}, pad=True)

    results = ArchiveDetector().detect(context_for(tmp_path))

    assert len(results) == 1
    result = results[0]
    assert result.kind is AssetKind.ARCHIVE
    assert result.subkind == "model_archive"
    assert result.is_single_file
    assert result.evidence["extracted"] is False
    assert result.evidence["signals"]


def test_small_archives_are_ignored(tmp_path: Path, context_for):
    _zip(tmp_path / "tiny.zip", {"config.json": b"{}"})
    assert ArchiveDetector().detect(context_for(tmp_path)) == []


def test_archive_named_for_its_only_root_folder(tmp_path: Path, context_for):
    _zip(
        tmp_path / "download.zip",
        {"UNSW-NB15/a.pcap": PADDING, "UNSW-NB15/b.pcap": PADDING},
        pad=True,
    )
    results = ArchiveDetector().detect(context_for(tmp_path))
    assert results[0].name == "UNSW-NB15"


def test_per_directory_cap_is_spent_on_the_largest(tmp_path: Path, context_for):
    """The cap is a budget, so it goes to the archives that account for the space."""
    for index in range(MAX_ARCHIVES_PER_DIR + 5):
        # Larger index, larger archive: the last few written must be the ones listed.
        payload = os.urandom(MIN_ARCHIVE_BYTES + index * 4096)
        _zip(tmp_path / f"a{index:02d}.zip", {"config.json": b"{}", "big.dat": payload})

    results = ArchiveDetector().detect(context_for(tmp_path))

    assert len(results) == MAX_ARCHIVES_PER_DIR
    listed = {Path(result.root_path).stem for result in results}
    assert "a29" in listed
    assert "a00" not in listed


def test_split_volumes_are_not_opened_individually(tmp_path: Path, context_for):
    _zip(tmp_path / "set.zip", {"config.json": b"{}", "model.safetensors": PADDING}, pad=True)
    (tmp_path / "set.z01").write_bytes(os.urandom(MIN_ARCHIVE_BYTES))

    results = ArchiveDetector().detect(context_for(tmp_path))

    assert [Path(result.root_path).name for result in results] == ["set.zip"]


def test_archives_and_loose_weights_coexist(tmp_path: Path, detectors, context_for):
    """Both live in one priority band, so a folder with each yields two assets."""
    _zip(tmp_path / "coco.zip", {"annotations/instances_train2017.json": b"{}"}, pad=True)
    (tmp_path / "qwen.gguf").write_bytes(b"GGUF" + b"\0" * (2 * 1024 * 1024))

    found = detectors.detect_one(context_for(tmp_path))
    detectors_used = {result.detector for result in found}

    assert detectors_used == {"archive", "loose_weights"}


def test_nothing_is_ever_extracted(tmp_path: Path, context_for):
    """The guarantee the whole feature rests on: no file is created anywhere."""
    archive_dir = tmp_path / "library"
    archive_dir.mkdir()
    _tar(
        archive_dir / "big.tar.gz",
        {
            "config.json": b"{}",
            "model.safetensors": PADDING,
            "images/a.jpg": PADDING,
            "images/b.jpg": PADDING,
        },
        pad=True,
    )

    before = {
        os.path.join(root, name)
        for root, _dirs, files in os.walk(tmp_path)
        for name in files
    }
    temp_before = set(os.listdir(os.environ.get("TEMP", ".")))

    results = ArchiveDetector().detect(context_for(archive_dir))
    assert results

    after = {
        os.path.join(root, name)
        for root, _dirs, files in os.walk(tmp_path)
        for name in files
    }
    assert after == before
    assert set(os.listdir(os.environ.get("TEMP", "."))) - temp_before == set()
