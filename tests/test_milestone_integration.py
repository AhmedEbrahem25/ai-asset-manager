"""End to end: a directory holding one of everything this milestone added.

Detection, persistence, taxonomy classification and the duplicate-installation report all
run against the same tree, because each of the four has been broken by a change in one of
the others at least once.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ai_asset_manager.backend.duplicate import find_duplicate_installations
from ai_asset_manager.backend.inventory import InventoryEngine
from ai_asset_manager.backend.models import Asset
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.services.scan_service import ScanService

UNSW_HEADER = (
    b"srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sload,dload,"
    b"ct_srv_src,attack_cat,label\n"
)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """Build a tree holding a packed model, a packed dataset, a corpus and two app models."""
    filler = os.urandom(1_200_000)

    packed = tmp_path / "library"
    packed.mkdir()
    with zipfile.ZipFile(packed / "qwen-packed.zip", "w") as archive:
        archive.writestr("config.json", json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
        archive.writestr("tokenizer.json", "{}")
        archive.writestr("model.safetensors", b"\0" * 1024)
        archive.writestr("filler.dat", filler)

    with zipfile.ZipFile(packed / "coco8.zip", "w") as archive:
        archive.writestr("coco8/data.yaml", "nc: 1")
        for index in range(30):
            archive.writestr(f"coco8/images/train/{index}.jpg", b"x")
            archive.writestr(f"coco8/labels/train/{index}.txt", b"0 0 0 0 0")
        archive.writestr("filler.dat", filler)

    corpus = tmp_path / "datasets" / "UNSW-NB15"
    corpus.mkdir(parents=True)
    for name in ("train.csv", "test.csv"):
        (corpus / name).write_bytes(UNSW_HEADER + b"0," * 5000)

    captures = tmp_path / "datasets" / "captures"
    captures.mkdir(parents=True)
    for index in range(5):
        (captures / f"day{index}.pcap").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\0" * 20000)

    # The same model, shipped by two applications, named `model` by both.
    weights = b"TFL3" + b"\xab" * (3 * 1024 * 1024)
    for vendor in ("Google/Chrome", "Microsoft/Edge"):
        target = tmp_path / "AppData" / "Local" / vendor / "User Data" / "screen_ai"
        target.mkdir(parents=True)
        (target / "model.tflite").write_bytes(weights)

    return tmp_path


@pytest.fixture
def scanned(session: Session, settings, library: Path) -> Session:
    """Scan the library into the catalogue."""
    service = ScanService(session, settings=settings, pipeline=ScanPipeline(settings=settings))
    service.scan([str(library)])
    return session


def _by_name(session: Session) -> dict[str, Asset]:
    """Return catalogued assets keyed by the name the inventory would show."""
    return {
        (asset.display_name or asset.name): asset
        for asset in session.query(Asset).all()
    }


def test_every_asset_is_found_and_nothing_swallows_the_tree(scanned: Session):
    """The failure this guards is silent: one claim too high hides everything below it."""
    found = _by_name(scanned)

    assert set(found) == {
        "qwen-packed",
        "coco8",
        "UNSW-NB15",
        "captures",
        "Chrome ScreenAI OCR Model",
        "Edge ScreenAI OCR Model",
    }


def test_archives_are_catalogued_as_archives(scanned: Session):
    found = _by_name(scanned)

    assert found["qwen-packed"].kind == "archive"
    assert found["qwen-packed"].subkind == "model_archive"
    assert found["coco8"].subkind == "yolo_dataset_archive"

    # Nothing was unpacked, so no model or dataset details were invented for either.
    assert found["qwen-packed"].model_details is None
    assert found["coco8"].dataset_details is None
    assert found["qwen-packed"].evidence["extracted"] is False


def test_security_datasets_are_catalogued_with_their_public_name(scanned: Session):
    found = _by_name(scanned)

    assert found["UNSW-NB15"].kind == "dataset"
    assert found["UNSW-NB15"].subkind == "intrusion_detection"
    assert found["UNSW-NB15"].dataset_details.extra["known_dataset"] == "UNSW-NB15"
    assert found["captures"].subkind == "network_capture"


def test_generic_names_are_replaced_and_the_disk_name_kept(scanned: Session):
    found = _by_name(scanned)
    chrome = found["Chrome ScreenAI OCR Model"]

    assert chrome.name == "model", "the name on disk is what the user will search for"
    assert chrome.evidence["identity"] == {
        "source": "chrome",
        "vendor": "Google",
        "product": "Chrome",
        "component": "ScreenAI",
        "task": "OCR",
        "display_name": "Chrome ScreenAI OCR Model",
        "signals": chrome.evidence["identity"]["signals"],
    }


def test_every_asset_explains_itself(scanned: Session):
    for asset in scanned.query(Asset).all():
        explanation = asset.evidence.get("explanation")
        assert explanation, f"{asset.name} has no explanation"
        assert explanation["signals"], f"{asset.name} explains nothing"
        assert 0.0 < explanation["confidence"] <= 1.0


def test_the_taxonomy_shelves_everything(scanned: Session):
    report = InventoryEngine(scanned).build()
    categories = {item.category for item in report.items}

    assert "model_archive" in categories
    assert "dataset_archive" in categories
    assert "intrusion_dataset" in categories
    assert "network_dataset" in categories
    assert "unclassified" not in categories


def test_the_doubled_install_is_reported(scanned: Session):
    groups = find_duplicate_installations(scanned, across_applications_only=True)

    assert len(groups) == 1
    group = groups[0]
    assert group.install_count == 2
    assert group.sources == ["Chrome", "Edge"]
    assert group.reclaimable_bytes == group.unit_size_bytes
