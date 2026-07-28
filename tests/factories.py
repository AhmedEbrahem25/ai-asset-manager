"""Synthetic asset factories.

Builds byte-valid assets on disk so parsers are exercised against real structures rather
than mocks. A hand-written safetensors file here has a genuine 8-byte length prefix and
genuine JSON header, so if the parser's offset arithmetic is wrong the test fails — which
a mocked ``open`` would never catch.

Everything is small: headers are real, weight payloads are zero-filled.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Binary formats
# --------------------------------------------------------------------------


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[list[int], str]] | None = None,
    *,
    metadata: dict[str, str] | None = None,
    truncate_payload: bool = False,
) -> Path:
    """Write a valid safetensors file.

    Args:
        path: Destination.
        tensors: Map of tensor name to ``(shape, dtype)``. Defaults to two small tensors.
        metadata: Optional ``__metadata__`` contents.
        truncate_payload: Omit the weight payload while leaving the header intact,
            reproducing an interrupted download.

    Returns:
        The path written.
    """
    tensors = tensors or {
        "model.embed_tokens.weight": ([128, 64], "F32"),
        "model.layers.0.mlp.weight": ([64, 64], "F32"),
    }
    element_sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "BOOL": 1}

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for name, (shape, dtype) in tensors.items():
        nbytes = math.prod(shape) * element_sizes.get(dtype, 4)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes

    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        if not truncate_payload:
            handle.write(b"\x00" * offset)
    return path


def expected_param_count(tensors: dict[str, tuple[list[int], str]]) -> int:
    """Return the parameter count a parser should report for these tensors."""
    return sum(math.prod(shape) for shape, _ in tensors.values())


# GGUF metadata value type tags.
_GGUF_UINT32 = 4
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _gguf_string(text: str) -> bytes:
    """Encode a GGUF length-prefixed string."""
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(
    path: Path,
    *,
    architecture: str = "llama",
    name: str = "Test Model",
    file_type: int = 15,
    tensors: dict[str, list[int]] | None = None,
    block_count: int = 4,
    embedding_length: int = 64,
    context_length: int = 2048,
    version: int = 3,
    bad_magic: bool = False,
    pad_to_bytes: int = 0,
) -> Path:
    """Write a valid GGUF file with a real key-value block and tensor table.

    Args:
        path: Destination.
        architecture: ``general.architecture`` value.
        name: ``general.name`` value.
        file_type: ``general.file_type`` enum value; 15 is ``Q4_K_M``.
        tensors: Map of tensor name to shape. Defaults to two small tensors.
        block_count: ``<arch>.block_count`` value.
        embedding_length: ``<arch>.embedding_length`` value.
        context_length: ``<arch>.context_length`` value.
        version: GGUF version to declare.
        bad_magic: Write a corrupt magic number, for error-path tests.
        pad_to_bytes: Pad the file to at least this size. Needed when a test must clear
            the loose-weights detector's minimum-size threshold, which exists so that
            small companion files are not catalogued as models.

    Returns:
        The path written.
    """
    tensors = tensors or {"token_embd.weight": [64, 128], "blk.0.attn_q.weight": [64, 64]}

    kv_pairs: list[bytes] = [
        _gguf_string("general.architecture") + struct.pack("<I", _GGUF_STRING)
        + _gguf_string(architecture),
        _gguf_string("general.name") + struct.pack("<I", _GGUF_STRING) + _gguf_string(name),
        _gguf_string("general.file_type") + struct.pack("<I", _GGUF_UINT32)
        + struct.pack("<I", file_type),
        _gguf_string(f"{architecture}.block_count") + struct.pack("<I", _GGUF_UINT32)
        + struct.pack("<I", block_count),
        _gguf_string(f"{architecture}.embedding_length") + struct.pack("<I", _GGUF_UINT32)
        + struct.pack("<I", embedding_length),
        _gguf_string(f"{architecture}.context_length") + struct.pack("<I", _GGUF_UINT32)
        + struct.pack("<I", context_length),
        # An array value, so the reader's recursive array handling is exercised.
        _gguf_string("tokenizer.ggml.tokens") + struct.pack("<I", _GGUF_ARRAY)
        + struct.pack("<I", _GGUF_STRING) + struct.pack("<Q", 3)
        + _gguf_string("<s>") + _gguf_string("</s>") + _gguf_string("<pad>"),
    ]

    tensor_block = b""
    offset = 0
    for tensor_name, shape in tensors.items():
        tensor_block += _gguf_string(tensor_name)
        tensor_block += struct.pack("<I", len(shape))
        for dim in shape:
            tensor_block += struct.pack("<Q", dim)
        tensor_block += struct.pack("<I", 12)  # ggml type Q4_K
        tensor_block += struct.pack("<Q", offset)
        offset += math.prod(shape)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"XXXX" if bad_magic else b"GGUF")
        handle.write(struct.pack("<I", version))
        handle.write(struct.pack("<Q", len(tensors)))
        handle.write(struct.pack("<Q", len(kv_pairs)))
        for pair in kv_pairs:
            handle.write(pair)
        handle.write(tensor_block)
        handle.write(b"\x00" * 256)
        if pad_to_bytes:
            remaining = pad_to_bytes - handle.tell()
            if remaining > 0:
                handle.write(b"\x00" * remaining)
    return path


def write_torch_checkpoint(path: Path, *, storages: int = 3, storage_bytes: int = 4096) -> Path:
    """Write a ZIP-format PyTorch checkpoint.

    The pickle member contains only the literals the parser scans for; it is never
    deserialised, so it does not need to be a valid pickle stream.
    """
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02}q\x00FloatStorageq\x01.")
        archive.writestr("archive/version", "3\n")
        for index in range(storages):
            archive.writestr(f"archive/data/{index}", b"\x00" * storage_bytes)
    return path


def write_yolo_checkpoint(
    path: Path,
    *,
    class_names: tuple[str, ...] = ("person", "car"),
    storage_bytes: int = 8192,
) -> Path:
    """Write a ZIP checkpoint carrying Ultralytics marker literals.

    Args:
        path: Destination.
        class_names: Labels embedded after the ``names`` marker.
        storage_bytes: Size of the tensor storage member. Pass at least 1 MiB when the
            file must clear the loose-weights detector's minimum-size threshold.
    """
    import zipfile

    # Framed the way a real pickle frames them: each string carries a SHORT_BINUNICODE
    # opcode and a length byte, and a MEMOIZE opcode follows. Separating the names with
    # spaces instead would be unrealistic and would let a class name containing a space
    # ("traffic light") pass a test that real data would fail.
    payload = (
        b"\x80\x04\x95\x00\x00FloatStorage\x94"
        b"ultralytics.nn.tasks\x94DetectionModel\x94names\x94"
    )
    for label in class_names:
        encoded = label.encode("utf-8")
        payload += b"\x8c" + bytes([len(encoded)]) + encoded + b"\x94"

    path.parent.mkdir(parents=True, exist_ok=True)
    # Stored, not deflated: a compressed run of zero bytes would leave the file far
    # below any size threshold the test is trying to clear.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", "3\n")
        archive.writestr("archive/data/0", b"\x00" * storage_bytes)
    return path


# --------------------------------------------------------------------------
# Model directories
# --------------------------------------------------------------------------


def make_hf_model(
    root: Path,
    name: str = "test-model",
    *,
    architecture: str = "Qwen2ForCausalLM",
    model_type: str = "qwen2",
    with_tokenizer: bool = True,
    with_card: bool = True,
    license_name: str = "apache-2.0",
    tensors: dict[str, tuple[list[int], str]] | None = None,
) -> Path:
    """Create a standard HuggingFace model directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": model_type,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "max_position_embeddings": 2048,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(directory / "model.safetensors", tensors)

    if with_tokenizer:
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 2048}),
            encoding="utf-8",
        )
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")

    if with_card:
        (directory / "README.md").write_text(
            f"---\nlicense: {license_name}\ntags:\n  - text-generation\n"
            f"pipeline_tag: text-generation\n---\n\n"
            "This is a synthetic model used to exercise the metadata parsers end to end.\n",
            encoding="utf-8",
        )
    return directory


def make_hf_cache_repo(
    cache_root: Path,
    repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    revision: str = "a" * 40,
    kind: str = "models",
    with_lock_dir: bool = True,
    with_no_exist: bool = True,
) -> Path:
    """Create a HuggingFace hub cache repository.

    Args:
        cache_root: The ``hub`` directory.
        repo_id: Repository id to encode into the directory name.
        revision: Commit sha for the snapshot directory.
        kind: ``models`` or ``datasets``.
        with_lock_dir: Also create the ``.locks`` mirror the hub writes, which the walker
            must ignore or every repo is catalogued twice.
        with_no_exist: Also create ``.no_exist`` markers, which name files the repo does
            *not* have — including ``adapter_config.json``.

    Returns:
        The cache repository directory.
    """
    encoded = f"{kind}--" + repo_id.replace("/", "--")
    repo_dir = cache_root / encoded
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)

    (repo_dir / "blobs").mkdir(exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")

    if kind == "models":
        (snapshot / "config.json").write_text(
            json.dumps(
                {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
                 "hidden_size": 64, "torch_dtype": "bfloat16"}
            ),
            encoding="utf-8",
        )
        (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        write_safetensors(snapshot / "model.safetensors")
    else:
        (snapshot / "dataset_info.json").write_text(json.dumps({"splits": {}}), encoding="utf-8")
        (snapshot / "train.parquet").write_bytes(b"PAR1" + b"\x00" * 1024)

    if with_no_exist:
        no_exist = repo_dir / ".no_exist" / revision
        no_exist.mkdir(parents=True, exist_ok=True)
        (no_exist / "adapter_config.json").touch()
        (no_exist / "added_tokens.json").touch()

    if with_lock_dir:
        locks = cache_root / ".locks" / encoded
        locks.mkdir(parents=True, exist_ok=True)
        (locks / f"{revision}.lock").touch()

    return repo_dir


def make_peft_adapter(
    root: Path, name: str = "my-lora", *, base_model: str = "meta-llama/Llama-3-8B"
) -> Path:
    """Create a PEFT/LoRA adapter directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": base_model,
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(directory / "adapter_model.safetensors", {"lora_A": ([16, 64], "F32")})
    return directory


def make_diffusers_pipeline(root: Path, name: str = "sd-pipeline") -> Path:
    """Create a Diffusers pipeline directory with component subdirectories."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.31.0",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        ),
        encoding="utf-8",
    )
    for component in ("unet", "vae", "text_encoder"):
        sub = directory / component
        sub.mkdir(exist_ok=True)
        (sub / "config.json").write_text(json.dumps({"_class_name": component}), encoding="utf-8")
        write_safetensors(sub / "diffusion_pytorch_model.safetensors")
    return directory


def make_ollama_store(root: Path, *, reference: str = "llama3:8b", blob_bytes: int = 2048) -> Path:
    """Create an Ollama store with one model manifest and its blobs."""
    name, _, tag = reference.partition(":")
    manifests = root / "manifests" / "registry.ollama.ai" / "library" / name
    manifests.mkdir(parents=True, exist_ok=True)
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)

    model_digest = "sha256:" + "1" * 64
    license_digest = "sha256:" + "2" * 64

    (blobs / model_digest.replace(":", "-")).write_bytes(b"\x00" * blob_bytes)
    (blobs / license_digest.replace(":", "-")).write_text("MIT License\n", encoding="utf-8")

    (manifests / (tag or "latest")).write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {"digest": "sha256:" + "0" * 64, "size": 487},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": model_digest,
                        "size": blob_bytes,
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.license",
                        "digest": license_digest,
                        "size": 12,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------
# Dataset directories
# --------------------------------------------------------------------------


def _write_images(directory: Path, count: int, *, prefix: str = "img") -> None:
    """Write placeholder image files."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{prefix}_{index:05d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)


def make_coco_dataset(root: Path, name: str = "coco-mini", *, images: int = 25) -> Path:
    """Create a COCO dataset with schema-valid annotation files."""
    directory = root / name
    annotations = directory / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        _write_images(directory / f"{split}2017", images, prefix=split)
        (annotations / f"instances_{split}2017.json").write_text(
            json.dumps(
                {
                    "info": {"description": "synthetic"},
                    "images": [
                        {"id": i, "file_name": f"{split}_{i:05d}.jpg"} for i in range(images)
                    ],
                    "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 1, 1]}
                                    for i in range(images)],
                    "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
                }
            ),
            encoding="utf-8",
        )
    return directory


def make_yolo_dataset(root: Path, name: str = "yolo-mini", *, images: int = 25) -> Path:
    """Create a YOLO dataset with a manifest and parallel image/label trees."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 3\n"
        "names: ['person', 'car', 'dog']\n",
        encoding="utf-8",
    )
    for split in ("train", "val"):
        _write_images(directory / "images" / split, images, prefix=split)
        labels = directory / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(images):
            (labels / f"{split}_{index:05d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return directory


def make_voc_dataset(root: Path, name: str = "voc-mini", *, images: int = 25) -> Path:
    """Create a Pascal VOC dataset."""
    directory = root / name
    _write_images(directory / "JPEGImages", images)
    (directory / "Annotations").mkdir(parents=True, exist_ok=True)
    (directory / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)

    for index in range(images):
        (directory / "Annotations" / f"img_{index:05d}.xml").write_text(
            "<annotation><object><name>person</name></object></annotation>", encoding="utf-8"
        )
    (directory / "ImageSets" / "Main" / "train.txt").write_text(
        "\n".join(f"img_{i:05d}" for i in range(images)), encoding="utf-8"
    )
    return directory


def make_imagefolder_dataset(
    root: Path,
    name: str = "flowers",
    *,
    classes: tuple[str, ...] = ("rose", "tulip", "daisy"),
    per_class: int = 10,
) -> Path:
    """Create a directory-per-class image dataset."""
    directory = root / name
    for split in ("train", "val"):
        for class_name in classes:
            _write_images(directory / split / class_name, per_class, prefix=class_name)
    return directory


# --------------------------------------------------------------------------
# Damaged assets, for the health checker
# --------------------------------------------------------------------------


def make_incomplete_download(root: Path, name: str = "half-downloaded") -> Path:
    """Create a model directory bearing the marks of an interrupted download."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )
    write_safetensors(directory / "model.safetensors")
    (directory / "model-00002-of-00002.safetensors.incomplete").write_bytes(b"\x00" * 512)
    return directory


def make_truncated_model(root: Path, name: str = "truncated") -> Path:
    """Create a model whose safetensors header outlives its payload."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )
    write_safetensors(directory / "model.safetensors", truncate_payload=True)
    return directory


def make_duplicate_pair(root: Path, *, size: int = 64 * 1024) -> tuple[Path, Path]:
    """Create two byte-identical model directories."""
    payload = os.urandom(size)
    created: list[Path] = []
    for name in ("copy-a", "copy-b"):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
            encoding="utf-8",
        )
        (directory / "model.bin").write_bytes(payload)
        created.append(directory)
    return created[0], created[1]
