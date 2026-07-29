"""ONNX metadata reader.

An ``.onnx`` file is a serialised protobuf ``ModelProto``. Only the small scalar fields at
the top of the message are needed — producer, IR version, opset, doc string — and those
are decoded with a minimal varint reader rather than by taking a dependency on ``onnx``,
which pulls in protobuf and numpy for what amounts to reading a header.

The ``graph`` field is skipped. It holds every initializer in the model and is the bulk of
the file; walking it to recover a parameter count would mean decoding the whole tensor
table, and the file size already tells the user what they need to know.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ai_asset_manager.backend.models.enums import FactSource, Framework
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Only the head of the file is read; the scalar fields precede the graph.
HEADER_READ_BYTES = 256 * 1024

#: Protobuf wire types.
_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH = 2
_WIRE_32BIT = 5

#: ``ModelProto`` field numbers.
_F_IR_VERSION = 1
_F_PRODUCER_NAME = 2
_F_PRODUCER_VERSION = 3
_F_DOMAIN = 4
_F_MODEL_VERSION = 5
_F_DOC_STRING = 6
_F_GRAPH = 7
_F_OPSET_IMPORT = 8

#: ``OperatorSetIdProto`` field numbers.
_F_OPSET_DOMAIN = 1
_F_OPSET_VERSION = 2

#: Guard against a corrupt length prefix claiming a huge field.
MAX_FIELD_BYTES = 32 * 1024 * 1024

#: ONNX IR version to the ONNX release that introduced it, for display.
IR_VERSION_NAMES: dict[int, str] = {
    3: "1.1", 4: "1.2", 5: "1.3", 6: "1.4", 7: "1.6",
    8: "1.10", 9: "1.13", 10: "1.15", 11: "1.16",
}


class _ProtoReader:
    """Minimal protobuf field reader over an in-memory buffer."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        """Wrap a buffer."""
        self._data = data
        self._pos = 0

    @property
    def exhausted(self) -> bool:
        """Report whether the buffer has been fully consumed."""
        return self._pos >= len(self._data)

    def varint(self) -> int:
        """Read a base-128 varint."""
        result = 0
        shift = 0
        while True:
            if self._pos >= len(self._data):
                raise ValueError("truncated varint")
            if shift > 63:
                raise ValueError("varint is too long")
            byte = self._data[self._pos]
            self._pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    def tag(self) -> tuple[int, int]:
        """Read a field tag, returning ``(field_number, wire_type)``."""
        key = self.varint()
        return key >> 3, key & 0x07

    def length_delimited(self) -> bytes:
        """Read a length-prefixed byte field."""
        size = self.varint()
        if size < 0 or size > MAX_FIELD_BYTES:
            raise ValueError(f"implausible field length {size}")
        if self._pos + size > len(self._data):
            raise ValueError("field extends past the buffer")
        chunk = self._data[self._pos : self._pos + size]
        self._pos += size
        return chunk

    def skip(self, wire_type: int) -> None:
        """Skip a field of the given wire type."""
        if wire_type == _WIRE_VARINT:
            self.varint()
        elif wire_type == _WIRE_LENGTH:
            self.length_delimited()
        elif wire_type == _WIRE_64BIT:
            self._pos += 8
        elif wire_type == _WIRE_32BIT:
            self._pos += 4
        else:
            raise ValueError(f"unknown wire type {wire_type}")


@dataclass(slots=True)
class OnnxInfo:
    """Result of inspecting an ONNX file."""

    path: str
    file_size: int
    ir_version: int | None = None
    producer_name: str | None = None
    producer_version: str | None = None
    domain: str | None = None
    model_version: int | None = None
    doc_string: str | None = None
    opsets: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Report whether anything identifiable was decoded."""
        return self.error is None and (
            self.ir_version is not None or self.producer_name is not None
        )

    @property
    def default_opset(self) -> int | None:
        """Return the opset version of the default (empty-domain) operator set."""
        return self.opsets.get("") or self.opsets.get("ai.onnx")

    @property
    def onnx_release(self) -> str | None:
        """Return the ONNX release matching this IR version, if known."""
        return IR_VERSION_NAMES.get(self.ir_version) if self.ir_version else None


def _decode_string(raw: bytes) -> str | None:
    """Decode a protobuf string field leniently."""
    text = raw.decode("utf-8", errors="replace").strip()
    return text or None


def read_onnx_header(path: str | os.PathLike[str]) -> OnnxInfo:
    """Read the scalar metadata fields of an ONNX model.

    Never raises for malformed input.

    Args:
        path: File to inspect.

    Returns:
        An :class:`OnnxInfo`, with ``error`` set when decoding failed.
    """
    path_str = os.fspath(path)
    try:
        file_size = os.path.getsize(path_str)
        with open(path_str, "rb") as handle:
            blob = handle.read(HEADER_READ_BYTES)
    except OSError as exc:
        return OnnxInfo(path=path_str, file_size=0, error=str(exc))

    info = OnnxInfo(path=path_str, file_size=file_size)
    reader = _ProtoReader(blob)

    try:
        while not reader.exhausted:
            field_number, wire_type = reader.tag()

            if field_number == _F_GRAPH:
                # The graph is the rest of the file. Everything of interest precedes it.
                break
            if field_number == _F_IR_VERSION and wire_type == _WIRE_VARINT:
                info.ir_version = reader.varint()
            elif field_number == _F_MODEL_VERSION and wire_type == _WIRE_VARINT:
                info.model_version = reader.varint()
            elif field_number == _F_PRODUCER_NAME and wire_type == _WIRE_LENGTH:
                info.producer_name = _decode_string(reader.length_delimited())
            elif field_number == _F_PRODUCER_VERSION and wire_type == _WIRE_LENGTH:
                info.producer_version = _decode_string(reader.length_delimited())
            elif field_number == _F_DOMAIN and wire_type == _WIRE_LENGTH:
                info.domain = _decode_string(reader.length_delimited())
            elif field_number == _F_DOC_STRING and wire_type == _WIRE_LENGTH:
                doc = _decode_string(reader.length_delimited())
                info.doc_string = doc[:600] if doc else None
            elif field_number == _F_OPSET_IMPORT and wire_type == _WIRE_LENGTH:
                _parse_opset(reader.length_delimited(), info)
            else:
                reader.skip(wire_type)
    except (ValueError, IndexError) as exc:
        # A partially decoded header is still worth keeping: the fields already read are
        # valid, and the model is catalogued rather than dropped.
        if not info.is_valid:
            info.error = f"not a decodable ONNX model: {exc}"
        else:
            logger.debug("ONNX header of %s ended early: %s", path_str, exc)

    return info


def _parse_opset(raw: bytes, info: OnnxInfo) -> None:
    """Parse one embedded ``OperatorSetIdProto``."""
    reader = _ProtoReader(raw)
    domain = ""
    version: int | None = None
    try:
        while not reader.exhausted:
            field_number, wire_type = reader.tag()
            if field_number == _F_OPSET_DOMAIN and wire_type == _WIRE_LENGTH:
                domain = reader.length_delimited().decode("utf-8", errors="replace")
            elif field_number == _F_OPSET_VERSION and wire_type == _WIRE_VARINT:
                version = reader.varint()
            else:
                reader.skip(wire_type)
    except (ValueError, IndexError):
        return
    if version is not None:
        info.opsets[domain] = version


class OnnxParser(BaseParser):
    """Extracts producer and opset facts from ONNX models."""

    name = "onnx"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether the directory contains ONNX models."""
        return bool(ctx.glob("*.onnx"))

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Read the header of the largest ONNX file present."""
        facts = self._new_facts()
        candidates = ctx.glob("*.onnx")
        if not candidates:
            return facts

        primary = max(candidates, key=lambda entry: entry.size)
        info = read_onnx_header(primary.path)

        if not info.is_valid:
            facts.warn(f"{primary.name}: {info.error or 'no decodable ONNX metadata'}")
            return facts

        facts.add("format", "onnx", source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add("framework", Framework.ONNXRUNTIME.value, source=FactSource.BINARY_HEADER,
                  confidence=0.7, origin=self.name)
        facts.add("onnx_ir_version", info.ir_version, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("onnx_opset", info.default_opset, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("producer", info.producer_name, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("producer_version", info.producer_version, source=FactSource.BINARY_HEADER,
                  origin=self.name)
        facts.add("description", info.doc_string, source=FactSource.BINARY_HEADER,
                  confidence=0.4, origin=self.name)

        # Some exporters name the source framework, which is better provenance than the
        # generic "onnxruntime" default.
        producer = (info.producer_name or "").lower()
        if "pytorch" in producer:
            facts.add("source_framework", "pytorch", source=FactSource.BINARY_HEADER,
                      origin=self.name)
        elif "tf2onnx" in producer or "tensorflow" in producer:
            facts.add("source_framework", "tensorflow", source=FactSource.BINARY_HEADER,
                      origin=self.name)
        elif "ultralytics" in producer:
            facts.add("framework", Framework.ULTRALYTICS.value, source=FactSource.BINARY_HEADER,
                      confidence=0.9, origin=self.name)

        return facts
