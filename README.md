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
ADE20K, LVIS, HuggingFace datasets, MNIST-family IDX archives, plus generic classification,
segmentation, tracking, video, audio, tabular and NLP corpora.

**Security datasets**, which have no config file and no `images/` tree and so are invisible
to every other rule here — packet captures (PCAP/PCAPNG), NetFlow and Argus flow records,
Zeek and Suricata output, Windows event logs and Sysmon, malware corpora, threat-intel and
IOC collections, CTF material, and labelled intrusion tables recognised from their column
names. Public corpora are named where they are recognised: UNSW-NB15, CICIDS2017,
CSE-CIC-IDS2018, NSL-KDD, KDD99, CTU-13, DARPA, MAWI, TON_IoT, Bot-IoT, IoT-23, USTC-TFC,
MalMem, EMBER, VirusShare, Malimg and around forty more.

**Archives** — `.zip`, `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`, `.tar.xz`, and `.7z`/`.rar`
with the optional readers installed. Catalogued from the table of contents and *never
unpacked*: a packed model, a YOLO or COCO or HuggingFace dataset, a tracking benchmark, a
training run, a capture corpus or a malware sample set are each recognised from their member
names. A 5.6 GB zip is listed in about a tenth of a second, because that means reading the
central directory and nothing else.

**The work around them** — AI projects (the codebase that trained the thing), training runs
from TensorBoard, Weights & Biases, MLflow, Ultralytics and Lightning, checkpoints, and
labelling projects from CVAT, Label Studio, Roboflow and Supervisely.

## Archives are listed, never extracted

Nothing is ever unpacked to disk. No temporary directory is created, and disk usage during
archive inspection is zero.

Three levels of access, and the boundaries between them are the design:

1. **The listing** — name, size, member names and sizes. For a zip that is the central
   directory at the end of the file, which is proportional to the number of members rather
   than to their size. For a tar it is the member headers.
2. **Named metadata, in memory** — a `config.json`, `data.yaml` or `dataset_info.json`
   *inside* the archive is read into a bytes object and parsed, provided it is on the
   allow-list and small enough to be configuration. It never touches the filesystem.
3. **Never** — images, video, weights, checkpoints, parquet, arrow, ONNX, GGUF,
   safetensors. Not at any size, for any reason. The gate is an allow-list of names, not a
   deny-list of extensions, so an unfamiliar format is refused by default.

Cost is bounded twice over, by member count and by bytes decompressed, because an archive is
an untrusted input and "how long does listing this take" must have an answer that does not
depend on what is inside it. Pointed at a 44 MB zip bomb, the scanner reports that the
contents could not be listed and moves on.

## One CSV is not a dataset

Security data is a pile of CSVs, or a pile of logs, or a pile of JSON — and so is half of a
normal machine. So the security detector never concludes anything from one observation.
Each independent signal carries a weight; a directory becomes a dataset when the total
clears a threshold *and* either several signals agree or one of them is both strong and
plural. A folder with a single `conn.log` is declined. The same folder with nine other Zeek
logs beside it is not.

Two further guards: a shelf holding CICIDS2017 and UNSW-NB15 defers to its contents rather
than reporting itself as one dataset, and a directory that is an application writing its own
state is never a corpus however corpus-shaped it looks.

Malware corpora get a stricter rule again, because the ordinary words are all taken. A
`samples/` folder, a `DLLs/` folder, ten executables and a few hash-named cache entries are
what normal software looks like — on the development machine those signals found "malware
corpora" in Python, in Ghidra, in Zoom, in a video editor and in a 14 GB NLP project, and
because a claim suppresses everything beneath it, the last of those hid twenty-two training
runs and eleven checkpoints. So at least one *unambiguous* observation is now required — a
directory that is about malware, a corpus where most files are named after their own digest,
or a recognised public dataset — and there must be something to analyse. Executables and
hash manifests can support that conclusion; they can no longer reach it.

## Where an asset starts and stops

The hardest part of discovery is not recognising a model, it is knowing *which directory is
the model*. An asset is the smallest directory representing one logical thing:
`F:\Models\Qwen2.5-7B` is an asset, `F:\Models` is a shelf that holds assets, and `F:\` is a
disk.

Nothing structural separates those three — all are directories with files below — so the
distinction is drawn deliberately. Generic rules (the ones that infer a dataset from a pile
of files) must pass a boundary test before they may claim a directory; specific ones do not,
because `annotations/instances_train.json` is proof wherever it sits.

That guard exists because the alternative is not a stray extra row. Detection runs parents
before children and a claim suppresses everything below it, so a rule that fires too high
*deletes* every correct answer underneath it. Pointed at this repository's development
machine, the old generic rule found two `.jsonl` files five levels down and filed a
372-directory project as one NLP dataset — hiding two HuggingFace models, three PaddleOCR
models, a Kraken model, eleven checkpoints and twenty-two W&B runs behind a single wrong row.

Bulk is never evidence on its own. A thousand screenshots and a thousand training images are
the same shape, so media only counts as a dataset when something says it was *assembled*: a
split layout, a manifest, or labels beside it.

## Staying up to date

You should not have to remember where anything is, or that a scan is due.

**`aam discover`** finds it for you. It knows where thirty-odd tools keep their downloads —
HuggingFace, Ollama, ComfyUI, Automatic1111, InvokeAI, LM Studio, llama.cpp, vLLM, PaddleOCR,
EasyOCR, Tesseract, Ultralytics, MMDetection, Detectron2, Whisper, TensorFlow Hub, Keras,
NGC, FiftyOne, CVAT, Label Studio, Roboflow, W&B, MLflow, TensorBoard and the rest — honours
the environment variables people set when a cache outgrows its drive, and glances two levels
below each drive root for folders named like a model library. **Nothing is scanned until you
approve it.**

```console
$ aam discover
Found AI assets

Model caches
  1  HuggingFace     C:\Users\pc\.cache\huggingface
  2  Ollama          C:\Users\pc\.ollama
  3  PyTorch Hub     C:\Users\pc\.cache\torch
Speech
  4  Whisper         C:\Users\pc\.cache\whisper

Add these to your AI library? [Y]es / [N]o / [E]dit:
```

It runs once. Declining is remembered, so you are not asked again; `aam discover --all`
re-offers what you turned down.

**`aam discover --deep`** searches by *content* rather than by name, which is what finds the
libraries no folder name announces. It scores every directory from its own listing — weight
files, a `config.json` beside a `tokenizer.json`, a folder of `models--*` entries, event
files — descends only into the promising ones, and stops at a time budget:

```console
$ aam discover --deep
Searching your drives by content (up to 60s)...

Found by deep search
  1  hub             C:\Users\pc\.cache\huggingface\hub          <- 12 cached repo(s)
  2  hub             F:\project\NLP-Project\...\data\hf_cache\hub  <- 2 cached repo(s)
  3  kraken_cache    F:\project\NLP-Project\...\data\kraken_cache  <- 1 weight file(s)
```

The second row is the point: a HuggingFace cache four levels inside a project, where nothing
in the path says "model". On the development machine a full pass over both drives takes
0.2 s, because directories full of ISOs and installers score below the descend threshold and
cost one listing each.

**`aam watch`** keeps the catalogue in step as files change. Events are debounced, so
copying a thousand files produces one update rather than a thousand, and each update
rescans only the subtree that actually changed:

```console
$ aam watch
Watching 1 folder(s):
  C:\Users\pc\.cache\huggingface
03:43:19  1 new across 1 location(s) in 0.0s
03:43:42  1 gone across 1 location(s) in 0.0s
```

Ctrl-C stops it, or `aam watch --stop` from another terminal. A deleted asset is *marked*
missing rather than deleted, so unplugging a drive does not destroy its catalogue.

**Between the two**, commands that read the catalogue run a quick incremental catch-up
first — skipped when a watcher is already running, rate-limited so four commands in a row
cost one scan, and silent unless something changed. `aam status` reports the lot:

| Command | Answers |
|---|---|
| `aam status` | What is managed, when it was last scanned, whether a watcher is live, database size, plugin count. |
| `aam watch --status` | Is a watcher running, and over what. |
| `aam scan --full` | Re-parse everything, ignoring fingerprints. |

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
| `aam inventory projects\|experiments` | Which codebases and training runs produced all this? |
| `aam inventory security\|archives` | Security corpora; packed models and datasets. |
| `aam duplicates` | Which models are installed more than once, and by what? |
| `aam where <name>` | Where did I put it? |

## Why does it think that?

Every asset records the evidence its detector matched on, so a classification can be checked
rather than trusted. `aam show <id>` prints it:

```
Why  Speech Recognition — confidence 95%
  ✓ encoder.onnx
  ✓ decoder.onnx
  ✓ tokens.txt
  ✓ declared task automatic-speech-recognition
```

The same block records where the asset came from. Applications ship the models they embed
under names like `model.tflite`, because inside the application there is only one — and
catalogued verbatim, fifty of them produce fifty identical rows. The path is not silent
about it, though: `AppData\Local\Google\Chrome\User Data\screen_ai\...\model.tflite` names
the vendor, the product, the component and the task in that order. So it is read, and the
row reads **Chrome ScreenAI OCR Model** with `Vendor: Google`, `Product: Chrome`,
`Source: chrome` beside it.

The name on disk is never changed — a folder you created is still findable by the name you
gave it. The derived name is a display name, and it is only ever offered when the existing
one says nothing.

## Duplicate installations

Applications do not repackage the models they embed, so Chrome, Edge, VS Code and Cursor all
carry byte-identical copies of the same optimisation model. `aam duplicates` finds them from
the catalogue alone — no file is opened and nothing is hashed, so it is instant after a scan:

```console
$ aam duplicates --across-apps
Model                        Installs   One copy   Reclaimable   Found in
Chrome Optimization Guide    6 copies     8.1 MiB      40.5 MiB   Chrome, Edge, Cursor, VS Code, …
```

Where an earlier duplicate pass left content digests behind the grouping uses them and the
row is marked ✓; otherwise it groups on the shape of the asset, which is enough to recognise
one build shipped six times. The reclaim figure is an upper bound: an application that finds
its bundled model gone will usually fetch it again, so this is space that can be *recovered*
rather than space being wasted. Nothing here deletes anything.

Assets are also linked to each other. Containment is derived after every scan — a checkpoint
inside a run was produced by it, a model inside a project belongs to it — and an adapter is
matched to the base model it declares. `aam inventory --details` reads the edges back:

```
best_dapt
  Category   Checkpoint
  Size       507.4 MiB
  Part of    clause_detector
```

That line is the difference between a row you can act on and one you cannot: "a 507 MiB
checkpoint called best_dapt" is not something you can decide about, and "a checkpoint from
the clause-detector project" is.

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
