"""GGUF header reader.

Layout: the magic ``GGUF``, a ``u32`` version, tensor and key-value counts, a typed
key-value block, then a tensor-info block. Weights follow and are never read.

Both blocks are parsed. The key-value block supplies architecture, name and quantisation;
the tensor-info block supplies dimensions, from which an exact parameter count follows —
the same guarantee the safetensors reader gives, for the format most local LLMs ship in.

Reads are bounded by a byte budget so that a corrupt length field cannot make the parser
allocate its way through memory.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO

from ai_asset_manager.backend.models.enums import FactSource, Precision
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

GGUF_MAGIC = b"GGUF"
DEFAULT_BUDGET_BYTES = 16 * 1024 * 1024

#: Sanity caps. Real models sit orders of magnitude below these; exceeding one means the
#: file is corrupt or is not GGUF, and the parser stops rather than trusting the length.
MAX_TENSORS = 1_000_000
MAX_KV_PAIRS = 100_000
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 4_000_000
MAX_DIMENSIONS = 8


class _GgufType:
    """GGUF metadata value type tags."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    _GgufType.UINT8: ("<B", 1),
    _GgufType.INT8: ("<b", 1),
    _GgufType.UINT16: ("<H", 2),
    _GgufType.INT16: ("<h", 2),
    _GgufType.UINT32: ("<I", 4),
    _GgufType.INT32: ("<i", 4),
    _GgufType.FLOAT32: ("<f", 4),
    _GgufType.BOOL: ("<?", 1),
    _GgufType.UINT64: ("<Q", 8),
    _GgufType.INT64: ("<q", 8),
    _GgufType.FLOAT64: ("<d", 8),
}

#: ``general.file_type`` values, the ggml "ftype" enum. This is the authoritative
#: quantisation label; filename conventions are only a fallback.
FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_F16",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    33: "Q4_0_4_4",
    34: "Q4_0_4_8",
    35: "Q4_0_8_8",
    36: "TQ1_0",
    37: "TQ2_0",
}

#: Per-tensor ``ggml_type`` values, used to describe the dominant tensor encoding.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}


class GgufError(ValueError):
    """Raised internally when a GGUF stream is malformed. Never escapes the parser."""


class _Reader:
    """Budgeted little-endian reader over a binary stream."""

    __slots__ = ("_budget", "_consumed", "_stream", "_u32_lengths")

    def __init__(self, stream: BinaryIO, budget: int, *, u32_lengths: bool) -> None:
        """Initialise the reader.

        Args:
            stream: Open binary file.
            budget: Maximum bytes this reader may consume.
            u32_lengths: GGUF v1 encoded counts and string lengths as ``u32``; v2 and
                later use ``u64``.
        """
        self._stream = stream
        self._budget = budget
        self._consumed = 0
        self._u32_lengths = u32_lengths

    def read(self, count: int) -> bytes:
        """Read exactly ``count`` bytes, honouring the budget."""
        if count < 0:
            raise GgufError(f"negative read length {count}")
        if self._consumed + count > self._budget:
            raise GgufError("header exceeds the read budget")
        data = self._stream.read(count)
        if len(data) != count:
            raise GgufError("unexpected end of file")
        self._consumed += count
        return data

    def scalar(self, fmt: str, size: int) -> object:
        """Read one packed scalar."""
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        """Read a ``u32``."""
        return int(struct.unpack("<I", self.read(4))[0])

    def u64(self) -> int:
        """Read a ``u64``."""
        return int(struct.unpack("<Q", self.read(8))[0])

    def length(self) -> int:
        """Read a count or length, honouring the version's width."""
        return self.u32() if self._u32_lengths else self.u64()

    def string(self) -> str:
        """Read a length-prefixed UTF-8 string."""
        size = self.length()
        if size > MAX_STRING_BYTES:
            raise GgufError(f"string length {size} is implausible")
        return self.read(size).decode("utf-8", errors="replace")

    def value(self, value_type: int) -> object:
        """Read one typed metadata value, recursing into arrays."""
        if value_type in _SCALAR_FORMATS:
            fmt, size = _SCALAR_FORMATS[value_type]
            return self.scalar(fmt, size)
        if value_type == _GgufType.STRING:
            return self.string()
        if value_type == _GgufType.ARRAY:
            return self._array()
        raise GgufError(f"unknown value type {value_type}")

    def _array(self) -> list[object]:
        """Read a typed array."""
        item_type = self.u32()
        count = self.length()
        if count > MAX_ARRAY_ITEMS:
            raise GgufError(f"array length {count} is implausible")

        # Token vocabularies are arrays of ~150k strings. They are read (to keep the
        # stream aligned for the tensor block that follows) but only the length is kept.
        items: list[object] = []
        for _ in range(count):
            items.append(self.value(item_type))
        return items


@dataclass(slots=True)
class GgufInfo:
    """Result of inspecting one GGUF file."""

    path: str
    file_size: int
    version: int = 0
    tensor_count: int = 0
    kv_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    param_count: int = 0
    param_count_is_exact: bool = False
    tensor_types: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Report whether the header parsed cleanly."""
        return self.error is None

    @property
    def architecture(self) -> str | None:
        """Return the declared architecture, e.g. ``"llama"``."""
        value = self.metadata.get("general.architecture")
        return str(value) if isinstance(value, str) else None

    @property
    def model_name(self) -> str | None:
        """Return the declared model name."""
        value = self.metadata.get("general.name")
        return str(value) if isinstance(value, str) else None

    @property
    def quantization(self) -> str | None:
        """Return the quantisation label from ``general.file_type``."""
        value = self.metadata.get("general.file_type")
        if isinstance(value, int):
            return FILE_TYPE_NAMES.get(value)
        return None

    @property
    def dominant_tensor_type(self) -> str | None:
        """Return the most common per-tensor encoding."""
        if not self.tensor_types:
            return None
        return max(self.tensor_types.items(), key=lambda item: item[1])[0]

    @property
    def precision(self) -> Precision:
        """Return a coarse precision class derived from the quantisation label."""
        label = (self.quantization or self.dominant_tensor_type or "").upper()
        if not label:
            return Precision.UNKNOWN
        if label.startswith("F32"):
            return Precision.FP32
        if label.startswith(("F16", "BF16")):
            return Precision.BF16 if label.startswith("BF16") else Precision.FP16
        if label.startswith(("Q8", "I8")):
            return Precision.INT8
        if label.startswith(("Q2", "Q3", "Q4", "Q5", "Q6", "IQ1", "IQ2", "IQ3", "IQ4", "TQ")):
            return Precision.INT4
        return Precision.UNKNOWN

    def arch_value(self, suffix: str) -> int | None:
        """Return an architecture-scoped integer metadata value.

        GGUF namespaces these by architecture (``llama.block_count``), so the key cannot
        be known until the architecture has been read.
        """
        if not self.architecture:
            return None
        value = self.metadata.get(f"{self.architecture}.{suffix}")
        return value if isinstance(value, int) else None


def read_gguf_header(
    path: str | os.PathLike[str],
    *,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    read_tensor_info: bool = True,
) -> GgufInfo:
    """Parse a GGUF file's header.

    Never raises for malformed input; problems are reported on the returned object.

    Args:
        path: File to inspect.
        budget_bytes: Maximum bytes to read before giving up.
        read_tensor_info: Parse the tensor-info block to derive an exact parameter
            count. Costs one pass over a few hundred short records.

    Returns:
        A :class:`GgufInfo`, with ``error`` set when parsing failed.
    """
    path_str = os.fspath(path)
    try:
        file_size = os.path.getsize(path_str)
    except OSError as exc:
        return GgufInfo(path=path_str, file_size=0, error=str(exc))

    info = GgufInfo(path=path_str, file_size=file_size)

    try:
        with open(path_str, "rb") as handle:
            magic = handle.read(4)
            if magic != GGUF_MAGIC:
                info.error = f"bad magic {magic!r}; expected {GGUF_MAGIC!r}"
                return info

            (version,) = struct.unpack("<I", handle.read(4))
            info.version = version
            if version < 1 or version > 10:
                info.error = f"unsupported GGUF version {version}"
                return info

            reader = _Reader(handle, budget_bytes, u32_lengths=version == 1)
            tensor_count = reader.length()
            kv_count = reader.length()

            if tensor_count > MAX_TENSORS:
                info.error = f"tensor count {tensor_count} is implausible"
                return info
            if kv_count > MAX_KV_PAIRS:
                info.error = f"metadata count {kv_count} is implausible"
                return info

            info.tensor_count = tensor_count
            info.kv_count = kv_count

            for _ in range(kv_count):
                key = reader.string()
                value_type = reader.u32()
                value = reader.value(value_type)
                # Vocabularies and merge tables are huge and uninteresting; keep their
                # size rather than their contents.
                if isinstance(value, list):
                    info.metadata[key] = f"<array[{len(value)}]>"
                    info.metadata[f"{key}.__len__"] = len(value)
                else:
                    info.metadata[key] = value

            if read_tensor_info:
                _read_tensor_info(reader, info)

    except GgufError as exc:
        info.error = str(exc)
    except (OSError, struct.error) as exc:
        info.error = f"read failed: {exc}"

    return info


def _read_tensor_info(reader: _Reader, info: GgufInfo) -> None:
    """Parse the tensor-info block, accumulating an exact parameter count.

    A partial failure is tolerated: whatever was counted before the error is discarded
    rather than reported as exact, because a truncated tensor block would otherwise
    produce a confidently wrong parameter count.
    """
    total = 0
    try:
        for _ in range(info.tensor_count):
            reader.string()  # tensor name
            n_dims = reader.u32()
            if n_dims > MAX_DIMENSIONS:
                raise GgufError(f"tensor has {n_dims} dimensions")
            dims = [reader.length() for _ in range(n_dims)]
            ggml_type = reader.u32()
            reader.u64()  # data offset

            type_name = GGML_TYPE_NAMES.get(ggml_type, f"type_{ggml_type}")
            info.tensor_types[type_name] = info.tensor_types.get(type_name, 0) + 1
            total += math.prod(dims) if dims else 0
    except GgufError as exc:
        logger.debug("Tensor block in %s ended early: %s", info.path, exc)
        return

    info.param_count = total
    info.param_count_is_exact = True


class GgufParser(BaseParser):
    """Extracts facts from GGUF weights."""

    name = "gguf"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether the directory contains GGUF weights."""
        return bool(ctx.glob("*.gguf"))

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Read the first (or only) GGUF shard in the directory.

        Multi-part GGUF files are named ``model-00001-of-00003.gguf``; the first part
        carries the full metadata block, so the others need not be opened.
        """
        facts = self._new_facts()
        shards = sorted(ctx.glob("*.gguf"), key=lambda entry: entry.name)
        if not shards:
            return facts

        primary = shards[0]
        info = read_gguf_header(primary.path)

        if not info.is_valid:
            facts.warn(f"{primary.name}: {info.error}")
            return facts

        facts.add("format", "gguf", source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add("gguf_version", info.version, source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add("tensor_count", info.tensor_count, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("architecture", info.architecture, source=FactSource.BINARY_HEADER,
                  origin=self.name)

        # GGUF states only a bare family name ("llama", "qwen3"), never a task, so the
        # model type has to be inferred from it. Imported here rather than at module
        # scope: the config parser owns the architecture patterns and importing it
        # eagerly would make two parser modules import each other.
        from ai_asset_manager.backend.parsers.hf_config import classify_architecture

        model_type = classify_architecture(info.architecture, info.model_name)
        if model_type is not None:
            facts.add("model_type", model_type.value, source=FactSource.BINARY_HEADER,
                      confidence=0.8, origin=self.name)
        facts.add("name", info.model_name, source=FactSource.BINARY_HEADER, confidence=0.8,
                  origin=self.name)
        facts.add("precision", info.precision.value, source=FactSource.BINARY_HEADER,
                  origin=self.name)

        quantization = info.quantization or info.dominant_tensor_type
        facts.add("quantization", quantization, source=FactSource.BINARY_HEADER, origin=self.name)

        if info.param_count_is_exact and info.param_count > 0:
            facts.add("param_count", info.param_count, source=FactSource.BINARY_HEADER,
                      origin=self.name)
            facts.add("param_count_is_exact", True, source=FactSource.BINARY_HEADER,
                      origin=self.name)

        for suffix, key in (
            ("block_count", "num_layers"),
            ("embedding_length", "hidden_size"),
            ("context_length", "context_length"),
        ):
            facts.add(key, info.arch_value(suffix), source=FactSource.BINARY_HEADER,
                      origin=self.name)

        vocab_size = info.metadata.get("tokenizer.ggml.tokens.__len__")
        if isinstance(vocab_size, int):
            facts.add("vocab_size", vocab_size, source=FactSource.BINARY_HEADER, origin=self.name)

        for key, target in (
            ("general.license", "license"),
            ("general.description", "description"),
            ("general.author", "author"),
            ("general.basename", "base_model"),
            ("general.organization", "author"),
        ):
            value = info.metadata.get(key)
            if isinstance(value, str):
                facts.add(target, value, source=FactSource.BINARY_HEADER, confidence=0.9,
                          origin=self.name)

        if len(shards) > 1:
            facts.add("shard_count", len(shards), source=FactSource.BINARY_HEADER,
                      origin=self.name)

        return facts
