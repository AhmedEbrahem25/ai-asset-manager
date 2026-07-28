"""Tests for the walker, detectors and the end-to-end pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_asset_manager.backend.models.enums import (
    AssetFormat,
    AssetKind,
    DatasetFormat,
    Framework,
    ModelType,
)
from ai_asset_manager.backend.parsers.hf_cache import decode_cache_dir_name, is_cache_repo_dir
from ai_asset_manager.backend.scanner.walker import WalkError, walk_tree
from tests import factories as F


def by_name(records):
    """Index records by name for readable assertions."""
    return {record.name: record for record in records}


class TestWalker:
    def test_collects_files_and_sizes(self, tmp_path: Path, settings) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.bin").write_bytes(b"x" * 100)
        (tmp_path / "two.bin").write_bytes(b"y" * 50)

        tree = walk_tree(tmp_path, settings=settings)

        assert tree.total_files == 2
        assert tree.total_bytes == 150
        assert len(tree.nodes) == 2

    def test_prunes_excluded_directories(self, tmp_path: Path, settings) -> None:
        for excluded in (".git", "node_modules", "__pycache__"):
            (tmp_path / excluded).mkdir()
            (tmp_path / excluded / "junk.bin").write_bytes(b"x")
        (tmp_path / "keep.bin").write_bytes(b"y")

        tree = walk_tree(tmp_path, settings=settings)

        assert tree.total_files == 1

    def test_prunes_huggingface_bookkeeping_dirs(self, tmp_path: Path, settings) -> None:
        # `.locks` mirrors every repo name and `.no_exist` holds markers named after
        # files the repo does not have, including adapter_config.json.
        F.make_hf_cache_repo(tmp_path, with_lock_dir=True, with_no_exist=True)

        tree = walk_tree(tmp_path, settings=settings)
        paths = "|".join(tree.nodes)

        assert ".locks" not in paths
        assert ".no_exist" not in paths

    def test_respects_depth_limit(self, tmp_path: Path, settings) -> None:
        deep = tmp_path
        for index in range(10):
            deep = deep / f"level{index}"
        deep.mkdir(parents=True)
        (deep / "deep.bin").write_bytes(b"x")

        settings.max_depth = 3
        tree = walk_tree(tmp_path, settings=settings)

        assert tree.total_files == 0
        assert all(node.depth <= 3 for node in tree.nodes.values())

    def test_child_references_never_dangle(self, tmp_path: Path, settings) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        settings.max_depth = 1

        tree = walk_tree(tmp_path, settings=settings)

        for node in tree.nodes.values():
            for child in node.child_dirs:
                assert child in tree.nodes

    def test_missing_root_raises(self, tmp_path: Path, settings) -> None:
        with pytest.raises(WalkError):
            walk_tree(tmp_path / "absent", settings=settings)

    def test_file_as_root_raises(self, tmp_path: Path, settings) -> None:
        target = tmp_path / "f.txt"
        target.write_text("x")

        with pytest.raises(WalkError):
            walk_tree(target, settings=settings)

    def test_subtree_files_are_gathered(self, tmp_path: Path, settings) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "one.bin").write_bytes(b"x")
        (tmp_path / "a" / "b" / "two.bin").write_bytes(b"y")

        tree = walk_tree(tmp_path, settings=settings)

        assert len(tree.iter_subtree_files(str(tmp_path / "a"))) == 2


class TestCacheNameDecoding:
    @pytest.mark.parametrize(
        ("directory", "expected_repo", "expected_kind"),
        [
            ("models--Qwen--Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", AssetKind.MODEL),
            ("models--meta-llama--Meta-Llama-3-8B-Instruct",
             "meta-llama/Meta-Llama-3-8B-Instruct", AssetKind.MODEL),
            ("datasets--openai--gsm8k", "openai/gsm8k", AssetKind.DATASET),
            ("models--gpt2", "gpt2", AssetKind.MODEL),
        ],
    )
    def test_decodes_repo_ids(
        self, directory: str, expected_repo: str, expected_kind: AssetKind
    ) -> None:
        decoded = decode_cache_dir_name(directory)

        assert decoded is not None
        kind, repo_id, _author, _name = decoded
        assert repo_id == expected_repo
        assert kind is expected_kind

    def test_repo_name_containing_double_dash_is_preserved(self) -> None:
        # Only the first separator splits owner from name; repository names legitimately
        # contain "--" and splitting on all of them would mangle them.
        decoded = decode_cache_dir_name("models--org--my--weird--model")

        assert decoded is not None
        assert decoded[1] == "org/my--weird--model"

    @pytest.mark.parametrize("name", ["not-a-cache", "model--x", "randomdir", ""])
    def test_rejects_non_cache_names(self, name: str) -> None:
        assert decode_cache_dir_name(name) is None
        assert not is_cache_repo_dir(name)


class TestModelDetection:
    def test_hf_model_directory(self, tmp_path: Path, pipeline) -> None:
        F.make_hf_model(tmp_path, "my-model")

        records = pipeline.scan_root(tmp_path)

        assert len(records) == 1
        record = records[0]
        assert record.kind is AssetKind.MODEL
        assert record.format is AssetFormat.SAFETENSORS
        assert record.framework is Framework.TRANSFORMERS
        assert record.model.architecture == "Qwen2ForCausalLM"
        assert record.model.model_type is ModelType.LLM
        assert record.model.license == "apache-2.0"
        assert record.model.param_count_is_exact

    def test_peft_adapter_is_not_filed_as_a_model(self, tmp_path: Path, pipeline) -> None:
        F.make_peft_adapter(tmp_path, "my-lora", base_model="meta-llama/Llama-3-8B")

        records = pipeline.scan_root(tmp_path)

        assert len(records) == 1
        assert records[0].kind is AssetKind.ADAPTER
        assert records[0].model.model_type is ModelType.LORA
        assert records[0].model.base_model == "meta-llama/Llama-3-8B"

    def test_diffusers_pipeline_is_one_asset_not_many(self, tmp_path: Path, pipeline) -> None:
        F.make_diffusers_pipeline(tmp_path, "sd")

        records = pipeline.scan_root(tmp_path)

        # The pipeline holds three component subdirectories, each with its own config and
        # weights. Claiming the subtree is what stops them being catalogued separately.
        assert len(records) == 1
        assert records[0].framework is Framework.DIFFUSERS
        assert records[0].model.model_type is ModelType.IMAGE_GENERATION

    def test_hf_cache_repo_resolves_snapshot(self, tmp_path: Path, pipeline) -> None:
        F.make_hf_cache_repo(tmp_path / "hub", "Qwen/Qwen2.5-0.5B-Instruct", revision="c" * 40)

        records = pipeline.scan_root(tmp_path)

        assert len(records) == 1
        record = records[0]
        assert record.model.repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert record.model.author == "Qwen"
        assert record.model.revision == "c" * 40
        assert record.file_count > 0

    def test_cached_dataset_is_a_dataset(self, tmp_path: Path, pipeline) -> None:
        F.make_hf_cache_repo(
            tmp_path / "hub", "openai/gsm8k", kind="datasets", revision="d" * 40
        )

        records = pipeline.scan_root(tmp_path)

        assert records[0].kind is AssetKind.DATASET
        assert records[0].dataset.dataset_format is DatasetFormat.HF_DATASET

    def test_stale_snapshots_are_reported_as_reclaimable(self, tmp_path: Path, pipeline) -> None:
        hub = tmp_path / "hub"
        repo = F.make_hf_cache_repo(hub, "org/model", revision="a" * 40)
        # A second pull at a different revision; the hub never removes the first.
        stale = repo / "snapshots" / ("b" * 40)
        stale.mkdir(parents=True)
        F.write_safetensors(stale / "model.safetensors")

        records = pipeline.scan_root(tmp_path)

        assert any("superseded revision" in warning for warning in records[0].warnings)

    def test_loose_weights_become_one_asset_each(self, tmp_path: Path, pipeline) -> None:
        folder = tmp_path / "gguf"
        F.write_gguf(folder / "mistral.Q4_K_M.gguf", architecture="llama",
                     name="Mistral", pad_to_bytes=2 * 1024 * 1024)
        F.write_gguf(folder / "phi3.Q8_0.gguf", architecture="phi3", name="Phi",
                     file_type=7, pad_to_bytes=2 * 1024 * 1024)

        records = pipeline.scan_root(tmp_path)

        assert len(records) == 2
        found = by_name(records)
        assert found["Mistral"].model.quantization == "Q4_K_M"
        assert found["Phi"].model.quantization == "Q8_0"
        # Each asset must see only its own header, not the union of both.
        assert found["Mistral"].model.architecture == "llama"
        assert found["Phi"].model.architecture == "phi3"

    def test_single_file_asset_counts_its_bytes_once(self, tmp_path: Path, pipeline) -> None:
        folder = tmp_path / "gguf"
        F.write_gguf(folder / "a.gguf", name="Solo", pad_to_bytes=2 * 1024 * 1024)

        record = pipeline.scan_root(tmp_path)[0]

        assert record.file_count == 1
        assert record.size_bytes == os.path.getsize(folder / "a.gguf")

    def test_small_companion_files_are_not_catalogued(self, tmp_path: Path, pipeline) -> None:
        folder = tmp_path / "weights"
        folder.mkdir()
        (folder / "tiny.pt").write_bytes(b"\x00" * 1024)

        assert pipeline.scan_root(tmp_path) == []

    def test_ollama_store_emits_one_asset_per_manifest(self, tmp_path: Path, pipeline) -> None:
        store = tmp_path / "ollama"
        F.make_ollama_store(store, reference="llama3:8b", blob_bytes=4096)
        F.make_ollama_store(store, reference="qwen3:4b", blob_bytes=8192)

        records = pipeline.scan_root(tmp_path)
        found = by_name(records)

        assert set(found) == {"llama3:8b", "qwen3:4b"}
        assert found["llama3:8b"].framework is Framework.OLLAMA
        assert found["llama3:8b"].model.license == "MIT"


class TestDatasetDetection:
    def test_coco(self, tmp_path: Path, pipeline) -> None:
        F.make_coco_dataset(tmp_path, "coco-mini", images=25)

        record = pipeline.scan_root(tmp_path)[0]

        assert record.kind is AssetKind.DATASET
        assert record.dataset.dataset_format is DatasetFormat.COCO
        assert record.dataset.class_names == ["person", "car"]
        assert record.dataset.num_annotations == 50
        assert record.dataset.splits == {"train": 25, "val": 25}
        assert record.dataset.has_bounding_boxes

    def test_yolo(self, tmp_path: Path, pipeline) -> None:
        F.make_yolo_dataset(tmp_path, "yolo-mini", images=25)

        record = pipeline.scan_root(tmp_path)[0]

        assert record.dataset.dataset_format is DatasetFormat.YOLO
        assert record.dataset.num_classes == 3
        assert record.dataset.class_names == ["person", "car", "dog"]
        assert record.dataset.splits == {"train": 25, "val": 25}

    def test_pascal_voc(self, tmp_path: Path, pipeline) -> None:
        F.make_voc_dataset(tmp_path, "voc-mini", images=25)

        record = pipeline.scan_root(tmp_path)[0]

        assert record.dataset.dataset_format is DatasetFormat.PASCAL_VOC
        assert "person" in record.dataset.class_names
        assert record.dataset.num_annotations == 25

    def test_imagefolder_classification(self, tmp_path: Path, pipeline) -> None:
        F.make_imagefolder_dataset(
            tmp_path, "flowers", classes=("rose", "tulip", "daisy"), per_class=10
        )

        record = pipeline.scan_root(tmp_path)[0]

        assert record.dataset.dataset_format is DatasetFormat.IMAGE_CLASSIFICATION
        assert record.dataset.num_classes == 3
        assert record.dataset.num_images == 60

    def test_a_few_stray_images_are_not_a_dataset(self, tmp_path: Path, pipeline) -> None:
        folder = tmp_path / "screenshots"
        folder.mkdir()
        for index in range(5):
            (folder / f"shot{index}.png").write_bytes(b"\x89PNG" + b"\x00" * 32)

        assert pipeline.scan_root(tmp_path) == []


class TestMixedTree:
    def test_everything_is_found_exactly_once(self, tmp_path: Path, pipeline) -> None:
        F.make_hf_model(tmp_path, "text-model")
        F.make_peft_adapter(tmp_path, "adapter")
        F.make_diffusers_pipeline(tmp_path, "sd")
        F.make_hf_cache_repo(tmp_path / "hub", "Qwen/Qwen2.5-0.5B-Instruct")
        F.make_coco_dataset(tmp_path, "coco")
        F.make_yolo_dataset(tmp_path, "yolo")
        F.make_ollama_store(tmp_path / "ollama")

        records = pipeline.scan_root(tmp_path)
        kinds = [record.kind for record in records]

        assert len(records) == 7
        assert kinds.count(AssetKind.DATASET) == 2
        assert kinds.count(AssetKind.ADAPTER) == 1
        assert kinds.count(AssetKind.MODEL) == 4
        assert len({record.root_path for record in records}) == 7

    def test_incomplete_download_is_flagged(self, tmp_path: Path, pipeline) -> None:
        F.make_truncated_model(tmp_path, "truncated")

        record = pipeline.scan_root(tmp_path)[0]

        assert any("truncated" in warning for warning in record.warnings)
