# AI Asset Manager

A local-first "AI Asset OS": discover, catalogue, search, deduplicate and health-check
every AI model, dataset, adapter and checkpoint scattered across your drives.

Point it at your folders. It figures out what everything is.

```console
$ aam scan D:\Models E:\LLMs F:\Datasets
$ aam search "llama quantization:Q4_K_M"
$ aam duplicates --min-size 1GB
$ aam desktop
```

## Why

If you work with AI locally you eventually end up with hundreds of gigabytes of models and
datasets spread across drives, half of it in caches that are unreadable by design:

```
models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/
```

You forget what you downloaded, you download it twice, some of it is half-finished, and
nothing tells you which of it is broken. This catalogues the lot.

## What it recognises

**Model formats** — Safetensors, GGUF, PyTorch (`.pt`/`.pth`), ONNX, TensorRT, TensorFlow
SavedModel, Keras, OpenVINO, MLX, CoreML, TFLite.

**Ecosystems** — HuggingFace repos and cache, Diffusers, PEFT/LoRA adapters, Sentence
Transformers, Ultralytics/YOLO, Ollama, timm, llama.cpp.

**Model kinds** — LLMs, vision-language models, embedding and reranking models, object
detection, segmentation, pose, OCR, speech recognition, TTS, image generation.

**Dataset layouts**, identified by structure rather than by name — COCO, YOLO, Pascal VOC,
ImageNet, KITTI, Waymo, nuScenes, BDD100K, Cityscapes, MOT, CrowdHuman, Open Images,
ADE20K, LVIS, HuggingFace datasets, plus generic classification, segmentation, tracking,
video, audio and NLP corpora.

## Design notes

**No heavyweight ML dependencies.** Every format is parsed from raw bytes — there is no
`torch`, `transformers` or `onnx` in the dependency tree. Installation is small, and the
catalogue can describe models this machine has no way to run.

**`.pt` files are never unpickled.** Loading a PyTorch checkpoint executes arbitrary code,
which is unacceptable in a tool whose entire job is reading untrusted files off your
disks. Checkpoints are inspected via their ZIP central directory instead.

**Hashing is lazy and three-tiered.** Full SHA256 over a large model library costs hours of
I/O, so it is the last resort rather than the first step:

1. Files sharing no byte size with any other file are never read at all.
2. Size collisions get a cheap signature over their head and tail.
3. Only signature collisions are hashed in full.

**Shared storage is counted once.** HuggingFace and Ollama link their caches — by symlink,
by hardlink, or (on Windows without developer mode) not at all. Reporting linked copies as
reclaimable space would send you deleting files that free nothing, so physical identity is
resolved before anything is called a duplicate.

**Rescans are incremental.** Each asset carries a fingerprint over its files' sizes and
modification times; unchanged assets skip detection and parsing entirely.

## Status

Under active development. See `docs/` for architecture notes and the roadmap.

## Licence

MIT
