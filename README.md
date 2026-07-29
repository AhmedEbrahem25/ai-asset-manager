# AI Asset Manager

A local-first "AI Asset OS": discover, catalogue, search, deduplicate and health-check
every AI model, dataset, adapter and checkpoint scattered across your drives.

Point it at your folders. It figures out what everything is.

```console
$ aam scan D:\Models E:\LLMs F:\Datasets
$ aam inventory              # what do I have, and what is it for?
$ aam inventory ocr          # which OCR models are installed?
$ aam inventory missing      # what is incomplete or broken?
$ aam inventory --tree       # the shape of the whole library
$ aam where qwen             # I know I downloaded it - where did it go?
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

## Inventory

`aam inventory` is the catalogue's read side: what you own, what each thing is *for*, and
whether any of it is broken. It reads the database and nothing else — no folder is walked
and no file is opened, so it answers instantly however large the library is, and it is
strictly read-only. Nothing in this feature can move, rename or delete anything.

```console
$ aam inventory
┌────── AI Asset Inventory ───────┐
│ OCR Model          4   14.6 GiB │
│ LLM                3    6.9 GiB │
│ Vision-Language    1    2.3 GiB │
│ Embedding          2    1.0 GiB │
│ ...                             │
│ Total Assets      16            │
│ Total Storage          25.6 GiB │
│ Health            98/100        │
│ Need attention     2            │
└─────────────────────────────────┘
```

Every asset gets a **category** (which shelf it belongs on), a **task** (what it does), a
**domain**, a **family**, statistics drawn from its recorded file list, and a **health
score** with specific findings.

| Command | Answers |
|---|---|
| `aam inventory [category]` | What do I have? Any alias, section, or domain — `llm`, `ocr`, `vision`, `datasets`, `medical`, `experiments`. |
| `aam inventory --details` | Everything known about each asset, as readable records. |
| `aam inventory --tree` | How is the library shaped? Section → category → family. |
| `aam inventory health` | Score and findings for every asset, plus what to do about them. |
| `aam inventory missing` | Only what needs attention. |
| `aam inventory --group-by task\|domain\|family\|drive\|…` | Cut it a different way. |
| `aam inventory --export csv\|json\|markdown` | Take it elsewhere. |
| `aam where <name>` | Where did I put it? |

Health findings are derived from the file list the scanner recorded, which is why they cost
nothing to compute. A sharded model states its expected shard count in every filename, so
`model-00002-of-00004.safetensors` with three siblings missing is provably an unfinished
download — the kind of failure that looks perfectly healthy in a file browser and only
surfaces when something tries to load it.

## Extending the taxonomy

Nothing the inventory knows is hard-coded in its core. Categories, tasks, domains,
modalities, classifiers, health rules and statistics all come from plugins under
`ai_asset_manager/backend/taxonomy/plugins/`. Supporting a new AI domain means adding one
file there — the scanner, the inventory engine, the database schema and the CLI are all
untouched:

```python
def register(registry: TaxonomyRegistry) -> None:
    registry.add_domain(Domain(id="bioinformatics", label="Bioinformatics"))
    registry.add_category(Category(
        id="genomics_dataset", label="Genomics Dataset", section="datasets",
        order=400, domain="bioinformatics", aliases=("genomics",),
    ))

    @registry.classifier(priority=600, name="genomics")
    def _genomics(profile: AssetProfile) -> Classification | None:
        if profile.files.count(".fastq", ".bam", ".vcf"):
            return Classification(category="genomics_dataset", task="variant_calling",
                                  domain="bioinformatics", evidence="sequencing files")
        return None
```

`aam inventory genomics` works immediately, and so does `aam inventory datasets` — section
and domain selectors are resolved live rather than from a stored list. Distributions
outside this repository can register the same way through the `ai_asset_manager.taxonomy`
entry-point group.

A plugin receives an `AssetProfile` and nothing else: no session, no paths, no file
handles. It therefore *cannot* make the inventory slow or unsafe, which is what lets the
read-only guarantee survive plugins this project has never seen.

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

**The inventory reads the catalogue, never the disk.** Every question it answers — what a
dataset contains, whether a model's shards are all present, which splits exist — is a
question about the file list the scanner already wrote down. The test that proves it
deletes every scanned file and asserts the report still comes back complete, classified and
health-scored.

## Installing

**As a standalone executable** — no Python needed on the target machine:

```console
$ python scripts/build_exe.py --clean --zip
dist\aam\aam.exe  (36.8 MiB)
dist\aam-0.1.0-windows-x64.zip  (20.4 MiB)
```

Unzip anywhere and run `aam.exe`. One-directory by default because a one-file build
unpacks itself to a temporary folder on every run, and a second of start-up is a poor
trade for a tool whose selling point is answering instantly. `--onefile` is there if you
would rather hand someone a single file.

The build is not finished until the binary has been run: `scripts/build_exe.py` executes
it and asserts all 17 taxonomy plugins loaded. That check earns its place — plugins are
imported by name from a package directory listing, and a frozen bundle has no directory,
so getting it wrong produces a binary that runs, prints tables, and quietly files every
asset as "unclassified". `aam version --plugins` reports the same thing at any time.

**From source:**

```console
$ uv sync --extra dev
$ uv run aam inventory
```

`make help` lists the development tasks (`make check`, `make exe`, `make dist`).

## Status

Under active development. See `docs/` for architecture notes and the roadmap.

## Licence

MIT
