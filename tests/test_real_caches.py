"""Opt-in tests against the machine's real HuggingFace, Ollama and torch caches.

Skipped automatically when a cache is absent, so the suite stays green on a clean
machine and in CI. Run them with::

    pytest -m real_cache

Synthetic fixtures cannot prove that the parsers survive real files: real caches carry
multiple snapshots per repo, lock directories, zero-byte marker files, models saved by
half a dozen library versions, and names in scripts the console cannot encode. This is
where that gets checked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_asset_manager.backend.parsers.gguf import read_gguf_header
from ai_asset_manager.backend.parsers.hf_cache import is_cache_repo_dir
from ai_asset_manager.backend.parsers.safetensors import read_safetensors_header
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.config import DEFAULT_EXCLUDED_DIRS

pytestmark = pytest.mark.real_cache

HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface"))
OLLAMA_STORE = Path(os.path.expanduser("~/.ollama/models"))
TORCH_HUB = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))

needs_hf = pytest.mark.skipif(not HF_CACHE.exists(), reason="no HuggingFace cache")
needs_ollama = pytest.mark.skipif(not OLLAMA_STORE.exists(), reason="no Ollama store")
needs_torch = pytest.mark.skipif(not TORCH_HUB.exists(), reason="no torch hub cache")


@needs_hf
class TestHuggingFaceCache:
    def test_every_repo_is_catalogued_exactly_once(self, pipeline: ScanPipeline) -> None:
        expected = {
            name
            for name in os.listdir(HF_CACHE / "hub")
            if is_cache_repo_dir(name)
        } if (HF_CACHE / "hub").exists() else set()

        records = pipeline.scan_root(HF_CACHE)
        cache_records = [r for r in records if r.detector == "hf_cache"]
        roots = [record.root_path for record in cache_records]

        assert len(roots) == len(set(roots)), "a repository was catalogued twice"
        if expected:
            assert len(cache_records) == len(expected)

    def test_repo_ids_are_readable(self, pipeline: ScanPipeline) -> None:
        records = pipeline.scan_root(HF_CACHE)

        for record in records:
            details = record.model or record.dataset
            if record.detector != "hf_cache" or details is None:
                continue
            assert details.repo_id, f"{record.root_path} produced no repo id"
            assert "--" not in details.repo_id, "the flattened name leaked into the repo id"

    def test_no_repo_is_mistaken_for_an_adapter(self, pipeline: ScanPipeline) -> None:
        # `.no_exist` holds zero-byte markers named after files the repo does *not* have,
        # `adapter_config.json` among them. Failing to prune it turns every cached model
        # into a LoRA.
        records = pipeline.scan_root(HF_CACHE)
        adapters = [r for r in records if r.model and r.model.model_type.value == "lora"]

        for record in adapters:
            config = Path(record.root_path)
            snapshots = list(config.glob("snapshots/*/adapter_config.json"))
            assert snapshots, f"{record.name} claims to be an adapter without a real config"

    def test_safetensors_headers_all_parse(self) -> None:
        checked = 0
        for path in HF_CACHE.rglob("*.safetensors"):
            # `.no_exist` holds zero-byte placeholders named after files the repo does
            # not have — including `model.safetensors`. The walker prunes that directory,
            # so this test must respect the same exclusion rather than assert against
            # files the scanner never sees.
            if any(part in DEFAULT_EXCLUDED_DIRS for part in (p.lower() for p in path.parts)):
                continue

            info = read_safetensors_header(path)
            assert info.is_valid, f"{path}: {info.error}"
            assert info.param_count > 0
            checked += 1
            if checked >= 25:
                break

        if checked == 0:
            pytest.skip("no safetensors files in the cache")


@needs_ollama
class TestOllamaStore:
    def test_models_resolve_to_references_and_blobs(self, pipeline: ScanPipeline) -> None:
        records = pipeline.scan_root(OLLAMA_STORE)

        if not records:
            pytest.skip("Ollama store holds no models")

        for record in records:
            assert ":" in record.name, "an Ollama model should be named repo:tag"
            assert record.size_bytes > 0

    def test_weight_blobs_parse_as_gguf(self, pipeline: ScanPipeline) -> None:
        records = pipeline.scan_root(OLLAMA_STORE)

        if not records:
            pytest.skip("Ollama store holds no models")

        # The blob is named for its digest and has no extension; only the manifest's
        # media type reveals it is GGUF at all.
        for record in records:
            assert record.model is not None
            assert record.model.architecture, f"{record.name} yielded no architecture"
            assert record.model.param_count, f"{record.name} yielded no parameter count"


@needs_torch
class TestTorchHub:
    def test_checkpoints_are_inspected_without_unpickling(self) -> None:
        from ai_asset_manager.backend.parsers.torch_checkpoint import inspect_torch_checkpoint

        checked = 0
        for path in list(TORCH_HUB.glob("*.pt")) + list(TORCH_HUB.glob("*.pth")):
            info = inspect_torch_checkpoint(path)
            assert info.is_valid, f"{path}: {info.error}"
            # Either a modern ZIP archive or a recognised legacy pickle; both are fine,
            # and neither involves deserialising anything.
            assert info.is_zip or info.is_legacy_pickle
            checked += 1

        if checked == 0:
            pytest.skip("no torch checkpoints present")


@needs_hf
def test_full_scan_of_real_cache_is_stable(pipeline: ScanPipeline) -> None:
    """Two consecutive scans of the same tree must agree."""
    first = pipeline.scan_root(HF_CACHE)
    second = pipeline.scan_root(HF_CACHE)

    assert {r.root_path for r in first} == {r.root_path for r in second}
    assert {r.fingerprint for r in first} == {r.fingerprint for r in second}


@needs_ollama
def test_gguf_reader_handles_a_multi_gigabyte_file() -> None:
    """Header parsing must not depend on file size."""
    blobs = OLLAMA_STORE / "blobs"
    if not blobs.exists():
        pytest.skip("no blob store")

    largest = max(
        (p for p in blobs.iterdir() if p.is_file()), key=lambda p: p.stat().st_size, default=None
    )
    if largest is None or largest.stat().st_size < 100 * 1024 * 1024:
        pytest.skip("no large blob to test against")

    info = read_gguf_header(largest)

    assert info.is_valid, info.error
    assert info.param_count_is_exact
    assert info.param_count > 10**8
