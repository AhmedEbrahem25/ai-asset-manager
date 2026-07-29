"""Safetensors header reader.

Layout: an 8-byte little-endian header length, then that many bytes of JSON. The JSON maps
tensor names to ``{"dtype", "shape", "data_offsets"}`` and may carry a ``__metadata__``
object. Weights follow; they are never read here.

Summing ``prod(shape)`` over the tensor table yields an exact parameter count without
loading a byte of weight data, and without depending on ``torch`` or ``safetensors``.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Any

from ai_asset_manager.backend.models.enums import FactSource, Precision
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

HEADER_LENGTH_BYTES = 8

#: Refuse absurd header lengths. A real header is kilobytes to low megabytes; a larger
#: value means the file is not safetensors, or is corrupt, and blindly trusting it would
#: mean allocating that many bytes.
DEFAULT_MAX_HEADER_BYTES = 100 * 1024 * 1024

#: Mapping from safetensors dtype names to bytes-per-element and precision class.
DTYPE_INFO: dict[str, tuple[int, Precision]] = {
    "F64": (8, Precision.FP32),
    "F32": (4, Precision.FP32),
    "F16": (2, Precision.FP16),
    "BF16": (2, Precision.BF16),
    "F8_E4M3": (1, Precision.FP8),
    "F8_E5M2": (1, Precision.FP8),
    "I64": (8, Precision.INT8),
    "I32": (4, Precision.INT8),
    "I16": (2, Precision.INT8),
    "I8": (1, Precision.INT8),
    "U8": (1, Precision.INT8),
    "BOOL": (1, Precision.INT8),
}

#: Precision reported for the model as a whole, in descending priority. A checkpoint
#: holding BF16 weights alongside F32 norms is a BF16 model, not a mixed one.
_PRECISION_PRIORITY = (
    Precision.FP8,
    Precision.INT4,
    Precision.INT8,
    Precision.BF16,
    Precision.FP16,
    Precision.FP32,
)


@dataclass(slots=True)
class SafetensorsInfo:
    """Result of inspecting one safetensors file."""

    path: str
    file_size: int
    header_size: int
    tensor_count: int = 0
    param_count: int = 0
    dtypes: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    #: Highest byte offset any tensor claims, relative to the start of the data section.
    max_data_offset: int = 0
    #: Bytes the data section actually has available.
    available_data_bytes: int = 0
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Report whether the header parsed cleanly."""
        return self.error is None

    @property
    def is_truncated(self) -> bool:
        """Report whether tensors claim more data than the file contains.

        The signature of an interrupted download: the header is complete and parses, but
        the weights behind it are missing. Nothing short of reading the header catches
        this, and a size check alone would not.
        """
        return self.is_valid and self.max_data_offset > self.available_data_bytes

    @property
    def precision(self) -> Precision:
        """Return the dominant precision across the file's tensors."""
        if not self.dtypes:
            return Precision.UNKNOWN
        present = {
            DTYPE_INFO[name][1] for name in self.dtypes if name in DTYPE_INFO
        }
        for candidate in _PRECISION_PRIORITY:
            if candidate in present:
                return candidate
        return Precision.UNKNOWN


def read_safetensors_header(
    path: str | os.PathLike[str],
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> SafetensorsInfo:
    """Parse the header of a safetensors file.

    Never raises for malformed input; failures are reported on the returned object so a
    corrupt file still ends up in the catalogue, flagged.

    Args:
        path: File to inspect.
        max_header_bytes: Reject headers claiming to be larger than this.

    Returns:
        A :class:`SafetensorsInfo`, with ``error`` set when parsing failed.
    """
    path_str = os.fspath(path)
    try:
        file_size = os.path.getsize(path_str)
    except OSError as exc:
        return SafetensorsInfo(path=path_str, file_size=0, header_size=0, error=str(exc))

    info = SafetensorsInfo(path=path_str, file_size=file_size, header_size=0)

    if file_size < HEADER_LENGTH_BYTES:
        info.error = "file shorter than the 8-byte header length prefix"
        return info

    try:
        with open(path_str, "rb") as handle:
            (header_size,) = struct.unpack("<Q", handle.read(HEADER_LENGTH_BYTES))
            info.header_size = header_size

            if header_size == 0:
                info.error = "header length is zero"
                return info
            if header_size > max_header_bytes:
                info.error = f"header length {header_size} exceeds the {max_header_bytes} limit"
                return info
            if HEADER_LENGTH_BYTES + header_size > file_size:
                info.error = "header extends past the end of the file"
                return info

            raw = handle.read(header_size)
    except OSError as exc:
        info.error = str(exc)
        return info

    if len(raw) != header_size:
        info.error = "truncated header"
        return info

    try:
        header: Any = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        info.error = f"malformed header JSON: {exc}"
        return info

    if not isinstance(header, dict):
        info.error = "header is not a JSON object"
        return info

    raw_metadata = header.pop("__metadata__", None)
    if isinstance(raw_metadata, dict):
        info.metadata = {str(k): str(v) for k, v in raw_metadata.items()}

    info.available_data_bytes = file_size - HEADER_LENGTH_BYTES - header_size

    for tensor_name, spec in header.items():
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        dtype = spec.get("dtype")

        if isinstance(dtype, str):
            info.dtypes[dtype] = info.dtypes.get(dtype, 0) + 1

        if isinstance(shape, list) and all(isinstance(dim, int) and dim >= 0 for dim in shape):
            # A scalar tensor has an empty shape; prod(()) is 1, which is correct.
            info.param_count += math.prod(shape)
        else:
            logger.debug("Tensor %r in %s has an unusable shape", tensor_name, path_str)

        offsets = spec.get("data_offsets")
        if (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) for value in offsets)
        ):
            info.max_data_offset = max(info.max_data_offset, offsets[1])

        info.tensor_count += 1

    return info


@dataclass(slots=True)
class ShardIndex:
    """Parsed ``*.index.json`` describing a sharded checkpoint."""

    total_size: int = 0
    shard_files: set[str] = field(default_factory=set)
    tensor_count: int = 0


def read_shard_index(ctx: DirectoryContext, name: str) -> ShardIndex | None:
    """Parse a sharded-checkpoint index file.

    Args:
        ctx: Directory holding the index.
        name: Index filename, e.g. ``model.safetensors.index.json``.

    Returns:
        The parsed index, or ``None`` if absent or malformed.
    """
    data = ctx.read_json(name)
    if not isinstance(data, dict):
        return None

    index = ShardIndex()
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        total = metadata.get("total_size")
        if isinstance(total, int):
            index.total_size = total

    weight_map = data.get("weight_map")
    if isinstance(weight_map, dict):
        index.tensor_count = len(weight_map)
        index.shard_files = {str(value) for value in weight_map.values()}

    return index


class SafetensorsParser(BaseParser):
    """Extracts tensor-level facts from safetensors weights."""

    name = "safetensors"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether the directory contains safetensors weights."""
        return bool(ctx.glob("*.safetensors"))

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Read the header of one shard and, when present, the shard index.

        Only a single shard is opened. For a sharded checkpoint the index file already
        carries the authoritative total, and opening every shard of a 140 GB model to
        re-derive it would cost far more than it is worth.
        """
        facts = self._new_facts()
        shards = sorted(ctx.glob("*.safetensors"), key=lambda entry: entry.name)
        if not shards:
            return facts

        index = self._find_index(ctx)
        primary = shards[0]
        info = read_safetensors_header(primary.path)

        if not info.is_valid:
            facts.warn(f"{primary.name}: {info.error}")
            return facts

        facts.add("format", "safetensors", source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add(
            "precision",
            info.precision.value,
            source=FactSource.BINARY_HEADER,
            origin=self.name,
        )

        if info.is_truncated:
            facts.warn(
                f"{primary.name}: tensors reference {info.max_data_offset} bytes but only "
                f"{info.available_data_bytes} are present (truncated download)"
            )

        if index is not None and len(shards) > 1:
            self._add_sharded_facts(facts, info, index, shard_count=len(shards))
        else:
            facts.add(
                "param_count",
                info.param_count,
                source=FactSource.BINARY_HEADER,
                origin=self.name,
            )
            facts.add(
                "param_count_is_exact", True, source=FactSource.BINARY_HEADER, origin=self.name
            )
            facts.add(
                "tensor_count", info.tensor_count, source=FactSource.BINARY_HEADER, origin=self.name
            )

        # Training-time provenance that trainers write into __metadata__.
        for key, target in (
            ("format", "producer_format"),
            ("modelspec.architecture", "architecture"),
            ("ss_base_model_version", "base_model"),
            ("ss_network_module", "adapter_module"),
        ):
            if key in info.metadata:
                facts.add(
                    target,
                    info.metadata[key],
                    source=FactSource.BINARY_HEADER,
                    confidence=0.7,
                    origin=self.name,
                )

        return facts

    def _find_index(self, ctx: DirectoryContext) -> ShardIndex | None:
        """Locate and parse a shard index, whichever naming variant is used."""
        for candidate in (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ):
            if ctx.has(candidate):
                return read_shard_index(ctx, candidate)
        return None

    def _add_sharded_facts(
        self,
        facts: FactSet,
        info: SafetensorsInfo,
        index: ShardIndex,
        *,
        shard_count: int,
    ) -> None:
        """Derive whole-model figures for a sharded checkpoint.

        The index gives an exact tensor count and total byte size but no shape data, so
        the parameter count is estimated from total bytes and the observed element width.
        It is flagged inexact, and a lower confidence keeps it from overriding a value a
        config file states outright.
        """
        facts.add("tensor_count", index.tensor_count, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("shard_count", shard_count, source=FactSource.BINARY_HEADER, origin=self.name)

        bytes_per_element = self._dominant_element_size(info)
        if index.total_size > 0 and bytes_per_element > 0:
            facts.add(
                "param_count",
                index.total_size // bytes_per_element,
                source=FactSource.BINARY_HEADER,
                confidence=0.8,
                origin=self.name,
            )
            facts.add(
                "param_count_is_exact", False, source=FactSource.BINARY_HEADER, origin=self.name
            )

    def _dominant_element_size(self, info: SafetensorsInfo) -> int:
        """Return the element width of the most common dtype in the inspected shard."""
        if not info.dtypes:
            return 0
        dominant = max(info.dtypes.items(), key=lambda item: item[1])[0]
        return DTYPE_INFO.get(dominant, (0, Precision.UNKNOWN))[0]
