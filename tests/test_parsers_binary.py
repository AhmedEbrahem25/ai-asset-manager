"""Tests for the raw-bytes format readers.

Every case writes a genuinely valid (or genuinely broken) file and reads it back, so the
offset arithmetic is actually exercised rather than mocked away.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ai_asset_manager.backend.models.enums import Precision
from ai_asset_manager.backend.parsers.gguf import GGUF_MAGIC, read_gguf_header
from ai_asset_manager.backend.parsers.onnx import read_onnx_header
from ai_asset_manager.backend.parsers.safetensors import read_safetensors_header
from ai_asset_manager.backend.parsers.torch_checkpoint import inspect_torch_checkpoint
from tests import factories as F


class TestSafetensors:
    def test_reads_exact_param_count_from_shapes(self, tmp_path: Path) -> None:
        tensors = {
            "embed.weight": ([151936, 896], "BF16"),
            "layer.0.weight": ([896, 896], "BF16"),
        }
        path = F.write_safetensors(tmp_path / "model.safetensors", tensors)

        info = read_safetensors_header(path)

        assert info.is_valid
        assert info.tensor_count == 2
        assert info.param_count == 151936 * 896 + 896 * 896
        assert info.param_count == F.expected_param_count(tensors)

    def test_reports_dominant_precision(self, tmp_path: Path) -> None:
        path = F.write_safetensors(
            tmp_path / "m.safetensors",
            {"a": ([4, 4], "BF16"), "b": ([4, 4], "BF16"), "norm": ([4], "F32")},
        )

        # F32 norms alongside BF16 weights make a BF16 model, not a mixed one.
        assert read_safetensors_header(path).precision is Precision.BF16

    def test_preserves_metadata_block(self, tmp_path: Path) -> None:
        path = F.write_safetensors(
            tmp_path / "m.safetensors", metadata={"format": "pt", "author": "someone"}
        )

        info = read_safetensors_header(path)

        assert info.metadata["format"] == "pt"
        assert "__metadata__" not in info.dtypes

    def test_scalar_tensor_counts_as_one_parameter(self, tmp_path: Path) -> None:
        path = F.write_safetensors(tmp_path / "m.safetensors", {"scale": ([], "F32")})

        assert read_safetensors_header(path).param_count == 1

    def test_detects_truncated_payload(self, tmp_path: Path) -> None:
        path = F.write_safetensors(
            tmp_path / "m.safetensors", {"w": ([64, 64], "F32")}, truncate_payload=True
        )

        info = read_safetensors_header(path)

        assert info.is_valid, "a truncated file still has a readable header"
        assert info.is_truncated
        assert info.max_data_offset > info.available_data_bytes

    def test_intact_file_is_not_truncated(self, tmp_path: Path) -> None:
        path = F.write_safetensors(tmp_path / "m.safetensors")

        assert not read_safetensors_header(path).is_truncated

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            (b"", "empty file"),
            (b"\x05\x00\x00", "shorter than the length prefix"),
            (struct.pack("<Q", 0), "zero-length header"),
            (struct.pack("<Q", 10) + b"not json..", "malformed JSON"),
            (struct.pack("<Q", 2**40), "implausible header length"),
            (struct.pack("<Q", 4) + b"[1]", "header is not an object"),
        ],
    )
    def test_malformed_input_reports_error_without_raising(
        self, tmp_path: Path, payload: bytes, reason: str
    ) -> None:
        path = tmp_path / "broken.safetensors"
        path.write_bytes(payload)

        info = read_safetensors_header(path)

        assert not info.is_valid, reason
        assert info.error

    def test_missing_file_reports_error(self, tmp_path: Path) -> None:
        info = read_safetensors_header(tmp_path / "absent.safetensors")

        assert not info.is_valid


class TestGguf:
    def test_reads_metadata_and_exact_param_count(self, tmp_path: Path) -> None:
        tensors = {"token_embd.weight": [4096, 151936], "blk.0.attn_q.weight": [4096, 4096]}
        path = F.write_gguf(
            tmp_path / "model.gguf",
            architecture="qwen3",
            name="Test Qwen",
            file_type=15,
            tensors=tensors,
            block_count=36,
            embedding_length=4096,
            context_length=131072,
        )

        info = read_gguf_header(path)

        assert info.is_valid
        assert info.version == 3
        assert info.architecture == "qwen3"
        assert info.model_name == "Test Qwen"
        assert info.quantization == "Q4_K_M"
        assert info.param_count_is_exact
        assert info.param_count == 4096 * 151936 + 4096 * 4096
        assert info.arch_value("block_count") == 36
        assert info.arch_value("context_length") == 131072

    def test_records_array_length_without_retaining_contents(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf")

        info = read_gguf_header(path)

        # The vocabulary is read to keep the stream aligned but only its size is kept;
        # holding 150k token strings per model would be pointless.
        assert info.metadata["tokenizer.ggml.tokens.__len__"] == 3
        assert info.metadata["tokenizer.ggml.tokens"] == "<array[3]>"

    def test_quantization_maps_to_int4_precision(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf", file_type=15)

        assert read_gguf_header(path).precision is Precision.INT4

    def test_f16_file_type_maps_to_fp16(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf", file_type=1)

        assert read_gguf_header(path).precision is Precision.FP16

    def test_bad_magic_is_rejected(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf", bad_magic=True)

        info = read_gguf_header(path)

        assert not info.is_valid
        assert "magic" in (info.error or "")

    def test_unsupported_version_is_rejected(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf", version=99)

        assert not read_gguf_header(path).is_valid

    def test_implausible_tensor_count_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(
            GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 2**40) + struct.pack("<Q", 1)
        )

        info = read_gguf_header(path)

        assert not info.is_valid
        assert "implausible" in (info.error or "")

    def test_truncated_stream_does_not_raise(self, tmp_path: Path) -> None:
        full = F.write_gguf(tmp_path / "m.gguf").read_bytes()
        path = tmp_path / "cut.gguf"
        path.write_bytes(full[:40])

        assert not read_gguf_header(path).is_valid

    def test_budget_stops_runaway_reads(self, tmp_path: Path) -> None:
        path = F.write_gguf(tmp_path / "m.gguf")

        info = read_gguf_header(path, budget_bytes=16)

        assert not info.is_valid


class TestTorchCheckpoint:
    def test_reads_zip_checkpoint_without_unpickling(self, tmp_path: Path) -> None:
        path = F.write_torch_checkpoint(tmp_path / "model.pt", storages=5, storage_bytes=4096)

        info = inspect_torch_checkpoint(path)

        assert info.is_valid
        assert info.is_zip
        assert info.storage_count == 5
        assert info.storage_bytes == 5 * 4096
        assert info.torch_version == "3"
        assert info.precision is Precision.FP32
        assert info.estimated_params == 5 * 4096 // 4

    def test_identifies_ultralytics_by_byte_markers(self, tmp_path: Path) -> None:
        path = F.write_yolo_checkpoint(tmp_path / "yolov8n.pt", class_names=("person", "car"))

        info = inspect_torch_checkpoint(path)

        assert info.is_valid
        assert "ultralytics" in info.detected_markers
        assert info.model_type is not None
        assert "person" in info.class_names

    def test_legacy_pickle_is_recognised_not_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pth"
        path.write_bytes(b"\x80\x02}q\x00X\x05\x00\x00\x00stateq\x01.")

        info = inspect_torch_checkpoint(path)

        assert info.is_valid
        assert info.is_legacy_pickle
        assert not info.is_zip
        assert info.estimated_params is None

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pt"
        path.write_bytes(b"")

        info = inspect_torch_checkpoint(path)

        assert not info.is_valid
        assert "empty" in (info.error or "")

    def test_corrupt_archive_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 200)

        assert not inspect_torch_checkpoint(path).is_valid


class TestOnnx:
    def _build(
        self, *, producer: str = "pytorch", ir_version: int = 8, opset: int = 17
    ) -> bytes:
        """Build a minimal ``ModelProto`` by hand.

        Tags are ``(field_number << 3) | wire_type``: 0x08 is field 1 varint,
        0x12 is field 2 length-delimited, 0x42 is field 8 length-delimited. Inside the
        embedded ``OperatorSetIdProto``, 0x10 is field 2 (version) as a varint.
        """
        producer_bytes = producer.encode()
        opset_message = b"\x10" + bytes([opset])
        return (
            b"\x08" + bytes([ir_version])
            + b"\x12" + bytes([len(producer_bytes)]) + producer_bytes
            + b"\x42" + bytes([len(opset_message)]) + opset_message
        )

    def test_reads_producer_and_ir_version(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        path.write_bytes(self._build(producer="pytorch", ir_version=8))

        info = read_onnx_header(path)

        assert info.is_valid
        assert info.ir_version == 8
        assert info.producer_name == "pytorch"
        assert info.default_opset == 17

    def test_garbage_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "m.onnx"
        path.write_bytes(b"\xff" * 64)

        assert not read_onnx_header(path).is_valid

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        assert not read_onnx_header(tmp_path / "absent.onnx").is_valid
