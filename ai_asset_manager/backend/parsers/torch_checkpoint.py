"""PyTorch checkpoint inspector.

**Checkpoints are never unpickled.** ``torch.load`` executes arbitrary code embedded in
the file, which is unacceptable in a tool whose entire purpose is reading untrusted files
found on a disk. A malicious ``.pt`` would get code execution simply by being scanned.

Since PyTorch 1.6 a checkpoint is a ZIP archive::

    archive/data.pkl        the pickled object graph
    archive/data/0, 1, ...  raw tensor storages
    archive/version

The ZIP *central directory* is read — a table of names, sizes and offsets — which tells us
how many storages there are and how many bytes of weights they hold. Where the pickle
itself must be consulted, it is scanned as raw bytes for known literals and never
deserialised.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field

from ai_asset_manager.backend.models.enums import FactSource, Framework, ModelType, Precision
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Legacy (pre-1.6) checkpoints are bare pickle streams, which open with the PROTO opcode.
LEGACY_PICKLE_MAGIC = b"\x80"

#: Bytes of ``data.pkl`` scanned for identifying literals. The class names and metadata we
#: look for sit near the front; reading the whole pickle of a large model would be wasteful.
PICKLE_SCAN_BYTES = 512 * 1024

#: Storage class names mapped to element width and precision.
STORAGE_DTYPES: tuple[tuple[bytes, int, Precision], ...] = (
    (b"BFloat16Storage", 2, Precision.BF16),
    (b"HalfStorage", 2, Precision.FP16),
    (b"FloatStorage", 4, Precision.FP32),
    (b"DoubleStorage", 8, Precision.FP32),
    (b"CharStorage", 1, Precision.INT8),
    (b"ByteStorage", 1, Precision.INT8),
    (b"LongStorage", 8, Precision.INT8),
    (b"IntStorage", 4, Precision.INT8),
)

#: Literals that identify an Ultralytics checkpoint. Matched against raw pickle bytes,
#: which is safe; unpickling to ask the same question is not.
ULTRALYTICS_MARKERS = (b"ultralytics", b"models.yolo", b"DetectionModel", b"SegmentationModel")

#: Ultralytics task markers mapped to a model type.
YOLO_TASK_MARKERS: tuple[tuple[bytes, ModelType], ...] = (
    (b"SegmentationModel", ModelType.SEGMENTATION),
    (b"PoseModel", ModelType.POSE),
    (b"ClassificationModel", ModelType.CLASSIFICATION),
    (b"OBBModel", ModelType.OBJECT_DETECTION),
    (b"DetectionModel", ModelType.OBJECT_DETECTION),
)

#: ``yolov8n``, ``yolo11s-seg``, ``yolov12x`` and friends.
YOLO_NAME_RE = re.compile(r"\byolo\s*v?(\d+)([nstmlx])?\b", re.IGNORECASE)

_STORAGE_PATH_RE = re.compile(r"(^|/)data/\d+$")


@dataclass(slots=True)
class TorchCheckpointInfo:
    """Result of inspecting a PyTorch checkpoint."""

    path: str
    file_size: int
    is_zip: bool = False
    is_legacy_pickle: bool = False
    entry_count: int = 0
    storage_count: int = 0
    #: Total uncompressed bytes across tensor storage entries.
    storage_bytes: int = 0
    torch_version: str | None = None
    precision: Precision = Precision.UNKNOWN
    element_size: int = 0
    detected_markers: list[str] = field(default_factory=list)
    model_type: ModelType | None = None
    class_names: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Report whether the archive could be read."""
        return self.error is None

    @property
    def estimated_params(self) -> int | None:
        """Estimate parameter count from storage bytes and element width.

        Always an estimate: buffers, optimiser state and non-parameter tensors are
        counted too, and the element width is whatever dominates the pickle. Callers must
        record it as inexact.
        """
        if self.storage_bytes <= 0 or self.element_size <= 0:
            return None
        return self.storage_bytes // self.element_size


def inspect_torch_checkpoint(path: str | os.PathLike[str]) -> TorchCheckpointInfo:
    """Inspect a ``.pt``/``.pth``/``.ckpt`` file without deserialising it.

    Args:
        path: File to inspect.

    Returns:
        A :class:`TorchCheckpointInfo`; ``error`` is set when the archive is unreadable,
        which is itself the signal that a checkpoint is corrupt.
    """
    path_str = os.fspath(path)
    try:
        file_size = os.path.getsize(path_str)
    except OSError as exc:
        return TorchCheckpointInfo(path=path_str, file_size=0, error=str(exc))

    info = TorchCheckpointInfo(path=path_str, file_size=file_size)

    if file_size == 0:
        info.error = "file is empty"
        return info

    if not zipfile.is_zipfile(path_str):
        try:
            with open(path_str, "rb") as handle:
                head = handle.read(2)
        except OSError as exc:
            info.error = str(exc)
            return info
        if head.startswith(LEGACY_PICKLE_MAGIC):
            # Pre-1.6 format. Recognised and sized, but deliberately not inspected
            # further: there is no container to read, only a pickle stream.
            info.is_legacy_pickle = True
            return info
        info.error = "not a ZIP archive or a recognisable pickle stream"
        return info

    info.is_zip = True
    try:
        with zipfile.ZipFile(path_str) as archive:
            entries = archive.infolist()
            info.entry_count = len(entries)

            pickle_entry: zipfile.ZipInfo | None = None
            for entry in entries:
                normalised = entry.filename.replace("\\", "/")
                if _STORAGE_PATH_RE.search(normalised):
                    info.storage_count += 1
                    info.storage_bytes += entry.file_size
                elif normalised.endswith("data.pkl"):
                    pickle_entry = entry
                elif normalised.endswith("version"):
                    info.torch_version = _read_version(archive, entry)

            if pickle_entry is not None:
                _scan_pickle(archive, pickle_entry, info)
            elif info.storage_count == 0:
                info.error = "ZIP archive contains no recognisable tensor storages"

    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        info.error = f"unreadable archive: {exc}"

    return info


def _read_version(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> str | None:
    """Read the small ``version`` member from a checkpoint archive."""
    if entry.file_size > 64:
        return None
    try:
        return archive.read(entry).decode("ascii", errors="replace").strip() or None
    except (OSError, zipfile.BadZipFile):
        return None


def _scan_pickle(
    archive: zipfile.ZipFile, entry: zipfile.ZipInfo, info: TorchCheckpointInfo
) -> None:
    """Scan ``data.pkl`` for identifying byte literals.

    Reads and pattern-matches. Nothing is deserialised, so a hostile pickle is inert.
    """
    try:
        with archive.open(entry) as handle:
            blob = handle.read(PICKLE_SCAN_BYTES)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        logger.debug("Cannot read pickle member of %s: %s", info.path, exc)
        return

    for marker, element_size, precision in STORAGE_DTYPES:
        if marker in blob:
            info.element_size = element_size
            info.precision = precision
            break

    for marker in ULTRALYTICS_MARKERS:
        if marker in blob:
            info.detected_markers.append(marker.decode("ascii"))

    if info.detected_markers:
        for marker, model_type in YOLO_TASK_MARKERS:
            if marker in blob:
                info.model_type = model_type
                break
        info.class_names = _extract_class_names(blob)


def _extract_class_names(blob: bytes) -> list[str]:
    """Recover Ultralytics class names from raw pickle bytes.

    Ultralytics stores ``names`` as an index-to-label mapping. Rather than unpickling,
    the short ASCII strings that follow the ``names`` key are read out directly. Best
    effort by design: a partial class list is useful, and a wrong one is only cosmetic.
    """
    marker = blob.find(b"names")
    if marker == -1:
        return []

    window = blob[marker : marker + 16384]
    candidates = re.findall(rb"[\x20-\x7e]{2,40}", window)

    names: list[str] = []
    for candidate in candidates:
        text = candidate.decode("ascii", errors="ignore").strip()
        # Class labels are words; pickle machinery and module paths are not.
        if not text or len(text) < 2 or text in names:
            continue
        if any(char in text for char in "{}[]()<>/\\") or "." in text:
            continue
        if text in ("names", "q", "X", "u", "e"):
            continue
        names.append(text)
        if len(names) >= 100:
            break
    return names


class TorchCheckpointParser(BaseParser):
    """Extracts facts from PyTorch checkpoints, without unpickling them."""

    name = "torch_checkpoint"

    #: Extensions treated as PyTorch checkpoints.
    EXTENSIONS = ("*.pt", "*.pth", "*.ckpt", "*.bin")

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether the directory holds PyTorch weights."""
        return any(ctx.glob(pattern) for pattern in self.EXTENSIONS)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Inspect the largest checkpoint in the directory.

        The largest file is chosen because a directory frequently holds a large set of
        weights beside small auxiliary tensors, and the big one is the model.
        """
        facts = self._new_facts()

        candidates = [entry for pattern in self.EXTENSIONS for entry in ctx.glob(pattern)]
        if not candidates:
            return facts

        primary = max(candidates, key=lambda entry: entry.size)
        info = inspect_torch_checkpoint(primary.path)

        if not info.is_valid:
            facts.warn(f"{primary.name}: {info.error}")
            return facts

        facts.add("format", "pytorch", source=FactSource.BINARY_HEADER, origin=self.name)
        facts.add("framework", Framework.PYTORCH.value, source=FactSource.BINARY_HEADER,
                  confidence=0.5, origin=self.name)

        if info.is_legacy_pickle:
            facts.add("torch_legacy_format", True, source=FactSource.BINARY_HEADER,
                      origin=self.name)
            return facts

        if info.precision is not Precision.UNKNOWN:
            facts.add("precision", info.precision.value, source=FactSource.BINARY_HEADER,
                      confidence=0.7, origin=self.name)

        estimated = info.estimated_params
        if estimated:
            # Lower confidence than the exact counts safetensors and GGUF provide, so a
            # config-stated figure wins if one exists.
            facts.add("param_count", estimated, source=FactSource.BINARY_HEADER,
                      confidence=0.4, origin=self.name)
            facts.add("param_count_is_exact", False, source=FactSource.BINARY_HEADER,
                      origin=self.name)

        facts.add("tensor_count", info.storage_count or None, source=FactSource.BINARY_HEADER,
                  confidence=0.6, origin=self.name)
        facts.add("torch_version", info.torch_version, source=FactSource.BINARY_HEADER,
                  origin=self.name)

        if info.detected_markers:
            facts.add("framework", Framework.ULTRALYTICS.value, source=FactSource.BINARY_HEADER,
                      confidence=0.95, origin=self.name)
            facts.add("architecture", self._yolo_architecture(primary.name),
                      source=FactSource.FILENAME, origin=self.name)
            if info.model_type is not None:
                facts.add("model_type", info.model_type.value, source=FactSource.BINARY_HEADER,
                          confidence=0.9, origin=self.name)
            if info.class_names:
                facts.add("class_names", info.class_names, source=FactSource.BINARY_HEADER,
                          confidence=0.5, origin=self.name)
                facts.add("num_classes", len(info.class_names), source=FactSource.BINARY_HEADER,
                          confidence=0.5, origin=self.name)

        return facts

    def _yolo_architecture(self, filename: str) -> str | None:
        """Derive a YOLO architecture label such as ``"YOLOv8"`` from a filename."""
        match = YOLO_NAME_RE.search(filename)
        if not match:
            return "YOLO"
        return f"YOLOv{match.group(1)}"
