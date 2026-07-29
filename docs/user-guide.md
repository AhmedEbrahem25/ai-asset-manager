# AI Asset Manager — User Guide

**Version 0.1.0 · Windows x64 · No Python required**

AI Asset Manager answers three questions about the machine it runs on:

- **What AI models and datasets are on here?**
- **What is each of them for?**
- **Which of them are broken, and which are the same thing twice?**

It works by reading files, never by running them. It parses safetensors headers, GGUF
key-value blocks, HuggingFace configs, Ollama manifests, PEFT adapter configs, COCO and
YOLO annotation schemas and a few dozen other formats straight from their bytes. That is
why it can describe a 70-billion-parameter model on a laptop that could never load one,
and why it needs neither PyTorch nor a GPU nor a network connection.

> **It never writes to your library.** Scanning opens files read-only. Nothing is moved,
> renamed, deleted or modified, ever. The only thing the tool writes is its own catalogue,
> in its own directory.

---

## Table of contents

1. [Installation](#1-installation)
2. [First-time setup](#2-first-time-setup)
3. [Command reference](#3-command-reference)
4. [Configuration](#4-configuration)
5. [Examples](#5-examples)
6. [Common workflows](#6-common-workflows)
7. [Troubleshooting](#7-troubleshooting)
8. [Known limitations](#8-known-limitations)

---

## 1. Installation

### What you need

Nothing. The download contains its own Python interpreter, its own SQLite, and the
Microsoft C runtime. It runs on a machine that has never had Python or a Visual C++
redistributable installed.

- Windows 10 or 11, 64-bit
- About 45 MB of disk space once unpacked (140 files), plus the catalogue, which is well
  under a megabyte for most libraries
- No administrator rights

### Install

1. Unzip `aam-0.1.0-windows-x64.zip` anywhere you can write — `C:\Tools\aam` is a good
   choice. Avoid `C:\Program Files`, which needs elevation.

2. You will get a folder like this:

   ```
   aam\
     aam.exe          <- the program
     _internal\       <- interpreter and libraries; keep it beside aam.exe
   ```

   **Keep `aam.exe` and `_internal` together.** Moving the .exe out of its folder breaks
   it. If you want a single file you can drop anywhere, see
   [one-file builds](#one-file-builds) below.

3. Check that it runs:

   ```powershell
   C:\Tools\aam\aam.exe version
   ```

   ```
   AI Asset Manager 0.1.0
   Database: C:\Users\you\AppData\Local\AIAssetManager\catalog.db
   Data directory: C:\Users\you\AppData\Local\AIAssetManager
   Taxonomy: 52 categories, 92 tasks from 20 plugins
   Detectors: 36
   ```

   The last two lines are worth a glance: they say the classification rules and detectors
   loaded. A build that lost them would still run and still print tables, but would file
   everything as "unclassified".

### Put it on your PATH (recommended)

So you can type `aam` instead of the full path. In PowerShell, for your user only:

```powershell
[Environment]::SetEnvironmentVariable(
    'Path',
    [Environment]::GetEnvironmentVariable('Path', 'User') + ';C:\Tools\aam',
    'User')
```

Open a new terminal, then:

```powershell
aam version
```

The rest of this guide assumes `aam` is on your PATH.

### Windows SmartScreen

The executable is not code-signed, so the first launch may show
*"Windows protected your PC"*. Choose **More info → Run anyway**. If you would rather
verify before trusting it, right-click `aam.exe` → **Properties → Details**: it carries a
version resource naming the product and version.

### One-file builds

A single self-contained `aam.exe` with no `_internal` folder can be built from source with
`python scripts/build_exe.py --onefile`. It is easier to hand to someone, but starts about
a second slower on every run, because it unpacks itself to a temporary folder each time.
The one-directory build in the zip is the one to use day to day.

### Uninstalling

Delete the folder you unzipped, remove the PATH entry if you added one, and delete
`%LOCALAPPDATA%\AIAssetManager` to remove the catalogue. Your models and datasets are
untouched — the tool never had write access to them.

---

## 2. First-time setup

### Step 1 — run it bare

```powershell
aam
```

With no catalogue yet, it tells you what it is and offers concrete folders it can see:

```
AI Asset Manager 0.1.0
Nothing catalogued yet. Point it at a folder and it works out what is in there -
models, datasets, adapters, which are duplicates and which are broken.

Found these on this machine:
F:\Datasets            Datasets
F:\Downloads\data      data
F:\Downloads\datasets  datasets

  aam scan --auto   catalogue all of them

Scanning only reads. Nothing is moved, renamed or deleted, ever.
```

### Step 2 — catalogue something

You have three ways in. Pick whichever matches how tidy your machine is.

**If your models live in the usual caches** (HuggingFace, Ollama, torch hub, and about
thirty others):

```powershell
aam scan --auto
```

**If you want to be asked first**, and to catch folders on your other drives:

```powershell
aam discover
```

This lists what it found, grouped, numbered, and asks whether to add them. Answer `Y` for
all, `N` for none, or `E` to pick by number. Nothing is scanned until you say so.

**If you know exactly where things are:**

```powershell
aam scan D:\Models E:\Datasets --add
```

`--add` remembers those folders, so a later bare `aam scan` rescans them.

### Step 3 — look at what you have

```powershell
aam inventory
```

```
┌───────── AI Asset Inventory ─────────┐
│ LLM                     9    4.2 MiB │
│ Object Detection        1    2.0 MiB │
│ Diffusion               1  144.9 KiB │
│ Embedding               1   48.6 KiB │
│ Speech                  1   48.3 KiB │
│ Adapter / LoRA          2    8.5 KiB │
│ Detection Dataset       3   20.3 KiB │
│ Image Dataset           1    4.0 KiB │
│ NLP Dataset             1    1.0 KiB │
│                                      │
│ Total Assets           20            │
│ Total Storage                6.5 MiB │
│ Files                 371            │
│ Health             94/100            │
│ Need attention          5            │
└──────────────────────────────────────┘

Name                        Category           Task                Framework          Size  Location
Mistral 7B Instruct         LLM                Chat                unknown         3.0 MiB  ...q4_k_m.gguf
yolov8n                     Object Detection   Object Detection    ultralytics     2.0 MiB  ...yolov8n.pt
stable-diffusion-v1-5       Diffusion          Image Generation    diffusers     144.9 KiB  ...-v1-5
...
```

### Step 4 — keep it current (optional)

Two ways, and you do not need either to start:

- **Automatic catch-up.** Commands that read the catalogue quietly run a quick incremental
  rescan first, at most once every 15 minutes. This is on by default.
- **A live watcher.** `aam watch` follows filesystem events and updates the catalogue as
  files land. It runs in the foreground until Ctrl-C.

### How long a scan takes

Rescans are incremental: an asset whose files have not changed is skipped without being
re-parsed, so a second scan of an unchanged library costs one directory walk. The demo
library in this guide (375 files) scans in about 0.4 seconds and rescans in 0.1. A real
multi-terabyte library takes minutes on the first pass and seconds thereafter.

---

## 3. Command reference

Twelve commands, grouped the way `aam --help` groups them.

### Global options

These go **before** the command name.

| Option | Meaning |
|---|---|
| `-v`, `--verbose` | Debug logging. |
| `--database <path>` | Use a specific SQLite file instead of the default catalogue. |
| `--version` | Print the version and exit. |
| `--help` | Help for any command: `aam inventory --help`. |

Shell tab-completion is offered on Linux and macOS but **not on Windows**, where the shell
cannot be detected; the flags are hidden rather than left to fail.

```powershell
aam --database D:\work.db inventory     # a separate catalogue, for one project
```

---

### Getting started

#### `aam discover`

Find where AI assets are stored on this machine and offer to catalogue them.

Looks in the caches thirty-odd tools use, honours the environment variables people set
when a cache outgrows its drive, and glances a couple of levels below each drive root for
folders named like a model library. System directories are never entered.

| Option | Default | Meaning |
|---|---|---|
| `-y`, `--yes` | off | Add everything found without asking. |
| `--scan` / `--no-scan` | `--scan` | Scan the added folders straight away. |
| `--all` | off | Also re-offer locations you turned down before. |
| `--sweep` / `--no-sweep` | `--sweep` | Also look for asset folders on your drives. |
| `--deep` | off | Search by *content* rather than by folder name. |
| `--max-seconds <n>` | 60 | Time budget for `--deep`. |
| `--depth <n>` | 8 | How far below each root `--deep` may descend. |
| `--in <path>` | — | Restrict `--deep` to these folders. Repeatable. |

`--deep` is what finds a model library buried inside a project, where nothing in the path
says "model". It scores each directory from its own listing and follows only promising
ones, on a time budget.

```powershell
aam discover                                  # ask before adding
aam discover --yes                            # add everything found
aam discover --deep --in D:\Projects          # search one tree by content
aam discover --deep --max-seconds 120         # give it longer
```

#### `aam scan [PATHS...]`

Scan folders and update the catalogue. With no paths, scans the registered roots.

| Option | Meaning |
|---|---|
| `--auto` | Also scan the model and dataset caches found on this machine. |
| `--full` | Re-parse every asset, ignoring fingerprints. |
| `--incremental` | Only process what changed. This is already the default. |
| `--add` | Also register these paths as permanent scan roots. |
| `-q`, `--quiet` | Suppress the progress bar. |

Rescans are incremental by default. `--incremental` is accepted for symmetry and changes
nothing; `--full` is the flag that does.

```powershell
aam scan --auto                    # every cache the tool can find
aam scan D:\Models --add           # a folder, remembered for next time
aam scan                           # everything remembered
aam scan --full                    # re-parse everything from scratch
```

Ctrl-C during a scan cancels it cleanly and records the run as cancelled, rather than
leaving the catalogue mid-write.

**Exit codes:** `0` success · `1` no scan roots configured, or `--auto` found nothing ·
`2` contradictory flags (`--full` with `--incremental`).

#### `aam roots`

Manage the folders that get scanned.

```powershell
aam roots add D:\Models --label "Main models"   # register a folder
aam roots list                                  # show all registered roots
aam roots remove D:\Models                      # unregister (catalogued assets are kept)
```

`aam roots list` shows path, label, whether it is enabled, how many assets it held at the
last scan, and when that was.

Removing a root exits `1` if the path was not registered.

---

### Your library

#### `aam inventory [CATEGORY]`

The main view. Lists everything in your library, by category.

Reads **only** the catalogue — no folders are walked and no files are opened — so it
answers instantly however large the library is.

**Category** defaults to `all`. It accepts any of ~183 names and shorthands, including:

| Group | Names |
|---|---|
| Models | `llm`, `vlm`, `ocr`, `object-detection`, `segmentation`, `tracking`, `classification`, `vision`, `diffusion`, `embedding`, `reranker`, `speech`, `tts`, `audio`, `adapter`, `models` |
| Datasets | `datasets`, `detection-datasets`, `image-datasets`, `video-datasets`, `nlp-datasets`, `audio-datasets`, `ocr-datasets`, `medical-datasets`, `tabular-datasets`, `annotations` |
| Work products | `projects`, `experiments`, `runs`, `checkpoints`, `evaluations`, `papers`, `docs` |
| Security | `security`, `malware`, `pcap`, `intrusion-datasets`, `ctf`, `threat-intel` |
| Packed | `archives`, `packed-models`, `packed-datasets` |
| Special views | `all`, `health`, `missing` |

Two of those replace the category rather than filter it: **`health`** scores every asset
and lists what is wrong with it, and **`missing`** shows only what needs attention.

If you mistype, it suggests the closest match and prints the full list by section.

**Choosing what to show**

| Option | Meaning |
|---|---|
| `--drive <d>` | Restrict to one drive, e.g. `F:`. |
| `--framework <f>` | Restrict to one framework, e.g. `transformers`, `gguf`, `ultralytics`. |
| `--task <t>` | Restrict to one task, e.g. `object_detection`. |
| `--domain <d>` | Restrict to one domain (see below). |
| `-n`, `--limit <n>` | Show only the first N assets. |
| `--include-missing` | Include assets no longer on disk. |

Domain ids are short: `vision`, `nlp`, `speech`, `audio`, `multimodal`, `document_ai`,
`generative`, `autonomous_driving`, `robotics`, `three_d`, `medical`, `remote_sensing`,
`timeseries`, `tabular`, `reinforcement_learning`, `scientific`, `synthetic`, `security`,
`mlops`, `general`.

> **Watch the field names.** `inventory` and `list` take different sort fields —
> `inventory --sort date` is `list --sort modified`. An unrecognised `--sort` or
> `--group-by` value is **ignored silently** and the default is used, so if a sort appears
> to do nothing, check it against the tables here.

**Choosing how to show it**

| Option | Meaning |
|---|---|
| `-g`, `--group-by <field>` | Group by `category`, `section`, `task`, `domain`, `family`, `framework`, `drive`, `architecture`, `dataset_type`, `format`, `health`. |
| `--sort <field>` | Sort by `name`, `size`, `date`, `category`, `framework`, `files`, `health`, `task`. Default `size`. |
| `--asc` | Sort ascending instead of descending. |
| `-d`, `--details` | Show everything known about each asset, as blocks rather than rows. |
| `-t`, `--tree` | Show the library as a tree. |
| `--tree-by <levels>` | Comma-separated tree nesting, e.g. `section,category,family`. |

**Taking it elsewhere**

| Option | Meaning |
|---|---|
| `--export <fmt>` | Export instead of printing: `csv`, `json`, `markdown`. |
| `-o`, `--output <file>` | File to write the export to. A sensible name is chosen if omitted. |
| `--storage` | Also show the storage breakdown by drive, framework, task and family. |

An unknown export format exits `2` and names the valid ones.

#### `aam list`

A flatter listing than `inventory`, closer to `ls`. Useful when you want asset **ids** for
`aam show`.

| Option | Meaning |
|---|---|
| `--kind <k>` | `model`, `dataset`, `adapter`, `checkpoint`. |
| `--model-type <t>` | `llm`, `vision_language`, `embedding`, … |
| `--framework <f>` | `transformers`, `gguf`, `ultralytics`, … |
| `--drive <d>` | Restrict to one drive. |
| `--tag <t>` | Restrict to assets carrying a tag. |
| `--min-size <s>` | Minimum size, e.g. `1GB`, `500MB`, `256KB`. |
| `-s`, `--search <text>` | Free-text match. |
| `--sort <f>` | `size`, `name`, `modified`, `created`, `files`, `kind`, `scanned`. Default `size`. |
| `-n`, `--limit <n>` | Rows to show. Default 30. |
| `--paths` | Show full paths instead of the detail columns. |

```powershell
aam list --kind adapter
aam list --min-size 1GB --sort size
aam list --search qwen --paths
```

#### `aam show <ID>`

Everything known about one asset: size, format, framework, architecture, parameter count,
quantization, context length, licence, base model, where it came from, **why it was
classified the way it was**, and its health.

| Option | Meaning |
|---|---|
| `--files` | Also list the asset's files, largest first (up to 200). |

```powershell
aam show 3
```

```
┌──────────────────────────────────────────────────────────────┐
│ llama-3-70b-partial                                          │
│ D:\Models\llama-3-70b-partial                                │
│                                                              │
│ Kind          model / llm                                    │
│ Size          48.8 KiB across 3 file(s)                      │
│ Format        safetensors                                    │
│ Framework     transformers                                   │
│ Detected by   hf_repo (100% confidence)                      │
│                                                              │
│ Type          llm                                            │
│ Architecture  LlamaForCausalLM                               │
│ Parameters    12.3K                                          │
│                                                              │
│ Why  LLM — confidence 100%                                   │
│   ✓ weight file(s): 1                                        │
│   ✓ architecture LlamaForCausalLM                            │
│   ✓ 12,288 parameters (exact)                                │
└──────────────────────────────────────────────────────────────┘

Health  65/100
  x 1 unfinished download file(s), e.g. model-00002-of-00002.safetensors.incomplete
    Resume or restart the download, then rescan.
  ! No tokenizer files
    Fetch the tokenizer from the source repository, or point at the base model.
```

The **Why** block is the answer to "why does it think this is an OCR model?". The
**Health** block is the same rule set `aam inventory health` runs, with the fix hint under
each finding.

Exits `1` if there is no asset with that id.

#### `aam where <NAME>`

"I know I downloaded this — where did it go?" Matches part of a name or path.

| Option | Meaning |
|---|---|
| `-n`, `--limit <n>` | Results to show. Default 20. |

```powershell
aam where qwen
```

```
Name                        Category            Size  Location
Qwen2.5-0.5B-Instruct       LLM             48.6 KiB  D:\Models\Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-0.5B-Instruct  LLM             48.3 KiB  ...\models--Qwen--Qwen2.5-0.5B-Instruct
qwen-chat-lora              Adapter / LoRA   4.2 KiB  D:\Adapters\qwen-chat-lora
```

No match is not an error — it exits `0` and says nothing matched.

#### `aam stats`

A one-screen summary: counts by kind, total and on-disk size, storage by drive and by
framework, and the five largest assets.

#### `aam duplicates`

Models installed more than once, and what a single copy weighs.

| Option | Default | Meaning |
|---|---|---|
| `--across-apps` | off | Only show models that several *different applications* each ship a copy of. |
| `--min-size <s>` | `256KB` | Ignore models whose single copy is below this. |
| `--limit <n>` | 20 | How many groups to show. |

Answers from the catalogue alone — no file is opened and nothing is hashed, so it is
instant after a scan.

> The reclaim figure is an **upper bound**. An application that finds its bundled model
> gone will usually fetch it again, so this is space that *can* be recovered rather than
> space that is being wasted. Nothing here deletes anything.

Rows marked `✓` were verified by content hash. Rows without it were grouped on size and
file layout, which is enough to recognise the same build shipped by six applications.

#### `aam status`

What is managed, when it was last scanned, and whether a watcher is live.

Shows asset and file counts, the last scan's outcome and duration, watcher state and pid,
taxonomy size, the catalogue file and its size, whether discovery has run, and a table of
every managed folder.

Unlike most commands, `status` does **not** trigger a catch-up scan first — silently
rescanning before reporting "last scan: just now" would be a lie about itself.

#### `aam watch [PATHS...]`

Keep the catalogue in step with the disk as files change. Runs in the foreground until
Ctrl-C.

| Option | Meaning |
|---|---|
| `--stop` | Ask a running watcher to shut down. |
| `--status` | Report whether a watcher is running, and over what. |
| `--scan` / `--no-scan` | Scan once before watching starts. Default `--scan`. |

Filesystem events are collected and debounced, so copying a thousand files produces one
update rather than a thousand. Each update rescans only the subtree that changed.

```powershell
aam watch                # in one terminal
aam watch --status       # in another: "Running · pid 30104 · up 20s"
aam watch --stop         # in another: stops it cleanly
```

Starting a second watcher while one runs exits `1` rather than racing it.

---

### About

#### `aam version`

Version, catalogue location, data directory, taxonomy size and detector count.

| Option | Meaning |
|---|---|
| `--plugins` | Also list the loaded taxonomy plugins and what each provides. |

This command deliberately does **not** create a catalogue, so a freshly downloaded binary
asked its version leaves nothing behind.

#### `aam guide`

Worked examples of the things people actually want to do, grouped by intent. Like
`version`, it creates nothing.

---

## 4. Configuration

Most people never need this. The defaults are chosen to be right.

### Where things live

| What | Where |
|---|---|
| Catalogue | `%LOCALAPPDATA%\AIAssetManager\catalog.db` |
| Data directory | `%LOCALAPPDATA%\AIAssetManager` |

`aam version` prints both. Deleting the data directory resets the catalogue and nothing
else — your assets are not touched.

### How settings are resolved

In decreasing precedence:

1. Command-line flags (`--database`, `--verbose`)
2. Environment variables prefixed `AAM_`
3. A `.env` file in the current directory
4. Built-in defaults

### Environment variables

Every setting is `AAM_` plus the setting name in upper case.

**Storage**

| Variable | Default | Meaning |
|---|---|---|
| `AAM_DATA_DIR` | `%LOCALAPPDATA%\AIAssetManager` | Where the catalogue and state live. |
| `AAM_DATABASE_URL` | — | Full SQLAlchemy URL, overriding the SQLite default. |

**Scanning**

| Variable | Default | Meaning |
|---|---|---|
| `AAM_SCAN_WORKERS` | `8` | Parallel parse workers (1–64). |
| `AAM_HASH_WORKERS` | `4` | Parallel hash workers (1–32). Kept low: parallel reads thrash spinning disks. |
| `AAM_MAX_DEPTH` | `40` | Directory depth guard. |
| `AAM_FOLLOW_SYMLINKS` | `false` | Descending into symlinked directories risks cycles. |
| `AAM_EXCLUDED_DIRS` | see below | Comma-separated directory names never descended into. Replaces the default list. |
| `AAM_EXCLUDED_FILE_GLOBS` | `*.tmp,*.swp,~$*,…` | Comma-separated globs never treated as evidence of an asset. |

**Behaviour**

| Variable | Default | Meaning |
|---|---|---|
| `AAM_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `AAM_LOG_FILE` | — | Also write a rotating log file. |
| `AAM_AUTO_SCAN` | `true` | Quick catch-up rescan before commands that read the catalogue. |
| `AAM_AUTO_SCAN_INTERVAL_SECONDS` | `900` | Do not repeat that catch-up more often than this. |
| `AAM_AUTO_DISCOVER` | `true` | Offer to catalogue discovered locations on first use. |
| `AAM_WATCH_DEBOUNCE_SECONDS` | `2.0` | How long events must stop before a batch is processed. |
| `AAM_WATCH_MAX_WAIT_SECONDS` | `30.0` | Process a batch anyway once its oldest event is this old. |

**Quieting the catch-up scan.** The default `INFO` level means the automatic rescan prints
a few log lines before each command's output. To keep the output clean:

```powershell
$env:AAM_LOG_LEVEL = 'WARNING'
```

Or put it in a `.env` file beside wherever you run `aam` from:

```ini
AAM_LOG_LEVEL=WARNING
AAM_SCAN_WORKERS=16
```

### What is skipped by default

The scanner refuses to descend into version-control directories, `node_modules`,
`__pycache__`, virtualenvs and `site-packages`, Windows system trees (`Windows`,
`Program Files`, `ProgramData`, `$Recycle.Bin`, …), browser and Electron caches, and
package-manager caches (`.npm`, `.cargo`, uv and pip caches).

This is a default for **walks that reach them**, not a prohibition. Naming a path
explicitly still scans it:

```powershell
aam scan "C:\Program Files\SomeApp"    # works exactly as asked
```

`AAM_EXCLUDED_DIRS` **replaces** that list rather than adding to it, so set it only when
you want a different policy entirely:

```powershell
$env:AAM_EXCLUDED_DIRS = 'node_modules,.git,__pycache__'
```

> **This is easy to get wrong.** The defaults are load-bearing. Replacing them with a
> short list re-admits HuggingFace's `.locks` directory, which mirrors every repo name —
> so every cached model gets catalogued twice — and `.no_exist`, whose zero-byte
> `adapter_config.json` markers make cached models look like LoRA adapters. If you only
> want to add an exclusion, list the defaults you still want alongside it.

### Separate catalogues

`--database` gives you an isolated catalogue, which is useful for a one-off audit of a
drive without disturbing your main one:

```powershell
aam --database D:\audit.db scan E:\ --quiet
aam --database D:\audit.db inventory --storage
```

---

## 5. Examples

### Find out what is on a new machine

```powershell
aam discover --yes
aam inventory
```

### Which OCR models do I have?

```powershell
aam inventory ocr
```

### Just the language models, biggest first

```powershell
aam inventory llm --sort size
```

### The shape of the whole library

```powershell
aam inventory --tree
```

```
AI Library  20 asset(s) · 6.5 MiB
├── Models  15 · 6.4 MiB
│   ├── LLM  9 · 4.2 MiB
│   │   ├── Mistral 7B Instruct  3.0 MiB
│   │   ├── Llama  5 · 1.1 MiB
│   │   │   ├── copy-a  512.1 KiB · 90/100
│   │   │   ├── copy-b  512.1 KiB · 90/100
│   │   │   └── llama-3-70b-partial  48.8 KiB · 65/100
│   │   └── Qwen  2 · 96.9 KiB
│   ├── Object Detection  1 · 2.0 MiB
│   │   └── yolov8n  2.0 MiB
│   └── Adapter / LoRA  2 · 8.5 KiB
└── Datasets  5 · 25.3 KiB
    ├── Detection Dataset  3 · 20.3 KiB
    └── Image Dataset  1 · 4.0 KiB
```

### Everything about my datasets

```powershell
aam inventory datasets
```

Datasets get their own columns — samples, classes and splits — because they have no
architecture or parameter count:

```
Name            Task                  Format          Samples  Classes  Splits                Size  Health
coco-mini       Object Detection      coco / imagef…       60        2  train=30, val=30  10.7 KiB  91/100
traffic-signs   Object Detection      yolo / imagef…       80        3  val=40, train=40   6.9 KiB  91/100
flowers         Image Classification  image classif…       60        3  train=30, val=30   4.0 KiB  91/100
```

### Grouped by model family

```powershell
aam inventory --group-by family
```

### Where has my disk space gone?

```powershell
aam inventory --storage
```

Adds breakdowns by drive and framework, plus asset counts by task and family.

### Take it to a spreadsheet

```powershell
aam inventory --export csv -o library.csv
```

36 columns, one row per asset: name, category, task, domain, family, framework,
architecture, format, parameters, quantization, precision, context length, size, file
count, images, classes, splits, health score and findings, drive, path, licence, tags.

### Paste it into notes

```powershell
aam inventory --export markdown -o library.md
```

### Machine-readable, with the query that produced it

```powershell
aam inventory adapters --export json -o adapters.json
```

The JSON carries a `query` block recording exactly which filters were applied, so a saved
export explains itself.

### One drive only, biggest first

```powershell
aam inventory --drive F: --sort size
```

### Everything in one domain

```powershell
aam inventory --domain vision
aam inventory --domain medical
```

---

## 6. Common workflows

### "I am out of disk space"

```powershell
aam inventory --storage          # where has it gone?
aam duplicates                   # what is installed twice?
aam duplicates --across-apps     # what do six apps each ship a copy of?
aam inventory --sort size -n 20  # the twenty biggest things
```

`duplicates` tells you what a single copy weighs and what could be recovered. It deletes
nothing; you decide.

### "Something is broken and I do not know what"

```powershell
aam inventory missing     # only what needs attention
aam inventory health      # score and findings for everything
aam show <id>             # why this one scores what it does
```

`inventory missing` ends with a **What to do** table that aggregates the fixes across your
whole library, so you can see that four models are missing tokenizers before deciding
whether that matters.

Health findings are marked by severity: `x` error (stops the asset being usable), `!`
warning (fix before relying on it), `-` info (a note).

### "Where did I put that model?"

```powershell
aam where qwen
aam where .gguf
aam list --search llama --paths
```

### "I want to audit a drive without touching my main catalogue"

```powershell
aam --database D:\audit.db scan E:\ --quiet
aam --database D:\audit.db inventory --storage
aam --database D:\audit.db inventory --export csv -o audit.csv
```

Delete `D:\audit.db` when done.

### "Keep the catalogue current while I download things"

In one terminal:

```powershell
aam watch
```

```
Catching up before watching...
┌────────────── Scan completed ───────────────┐
│ 21 assets  (0 new, 0 updated, 21 unchanged) │
└─────────────────────────────────────────────┘

Watching 1 folder(s):
  D:\Models

Ctrl-C to stop, or 'aam watch --stop' from elsewhere.
19:10:38 1 updated across 1 location(s) in 0.1s
```

From another terminal, `aam watch --status` to check on it and `aam watch --stop` to end
it.

### "What did I train, and from what?"

```powershell
aam inventory experiments     # training runs, W&B, MLflow, TensorBoard
aam inventory projects        # the codebases that produced them
aam inventory checkpoints     # the checkpoints they wrote
aam inventory --details       # including what relates to what
```

### "I have a folder of downloaded archives"

```powershell
aam inventory archives
```

Archives are catalogued **from their table of contents, without being extracted**. A
3 MB `.zip` holding a COCO dataset is identified as a COCO Dataset Archive with its image
and member counts, and nothing is unpacked.

### Regular housekeeping

```powershell
aam status                 # is everything still being tracked?
aam scan                   # rescan the remembered roots (incremental, cheap)
aam inventory missing      # did anything break?
```

---

## 7. Troubleshooting

### "Windows protected your PC"

SmartScreen, because the binary is unsigned. **More info → Run anyway**. See
[installation](#windows-smartscreen).

### `aam` is not recognised as a command

It is not on your PATH, or you have not opened a new terminal since adding it. Use the
full path to check: `C:\Tools\aam\aam.exe version`.

### "Failed to load Python DLL" / the exe will not start

`aam.exe` has been separated from its `_internal` folder. They must stay together. Unzip
again and run it in place, or use a shortcut rather than moving the file.

### Log lines appear before my output

That is the automatic catch-up scan at the default `INFO` level. Silence it:

```powershell
$env:AAM_LOG_LEVEL = 'WARNING'
```

Or turn the catch-up off entirely with `AAM_AUTO_SCAN=false` and rescan by hand.

### A scan finds nothing

- Check the folder is actually registered: `aam roots list`
- Check what it thinks it scanned: `aam status`
- Assets under a default-excluded directory (`node_modules`, `site-packages`, `Windows`,
  `Program Files`, …) are skipped unless you name the path explicitly.
- Very small files are deliberately ignored: loose weight files below a size floor, and
  archives below 1 MiB, are treated as companions and attachments rather than assets.
- Run with `-v` to see what the walker did: `aam -v scan D:\Models`

### A scan of a whole drive is slow

The default exclusions already skip the trees that cost the most. Beyond that:

```powershell
aam scan D:\Models          # scan the folder you care about, not the drive root
$env:AAM_SCAN_WORKERS = '16'
```

Ctrl-C cancels cleanly at any point; the catalogue keeps what it had already committed.

### Something is classified wrongly

`aam show <id>` prints a **Why** block listing the signals the classifier used. That tells
you whether it was, say, an `adapter_config.json` that made a full model look like a LoRA.

### `--sort` or `--group-by` seems to do nothing

An unrecognised value is ignored silently rather than rejected, and the default is used.
The two commands also differ: `inventory` sorts by `date`, `list` sorts by `modified`.
Check the value against the tables in the [command reference](#aam-inventory-category).

### The tree or table is cut off

Rich sizes tables to the terminal. Widen the window, or use an export:

```powershell
aam inventory --export csv -o library.csv
```

### Non-Latin characters show as `?`

The Windows console's legacy code page cannot represent them. The tool forces UTF-8 on its
own output and replaces anything unrepresentable rather than crashing, but the console
still has to be able to draw it. Windows Terminal handles this; `conhost` often does not.

### The catalogue looks stale

```powershell
aam scan --full     # re-parse everything, ignoring fingerprints
```

### Starting a watcher says one is already running

```powershell
aam watch --status    # confirm
aam watch --stop      # stop it
```

If the process died without cleaning up, `aam watch --status` reports it as not running
and the next `aam watch` will start normally.

### Start over completely

Delete `%LOCALAPPDATA%\AIAssetManager`. Your models and datasets are untouched.

---

## 8. Known limitations

**Platform**

- This build is **Windows x64 only**. The source is cross-platform; the packaged binary is
  not.
- Not code-signed, so SmartScreen warns on first run.

**Coverage**

- **`.rar` archives are not readable in this build.** `.zip`, `.tar` and its compressed
  forms, and `.7z` all work — `.7z` support is bundled. `.rar` additionally needs an
  external `unrar` binary and is not included; such an archive is catalogued by name and
  size, without its contents.
- **Archives below 1 MiB are ignored**, and at most **25 archives per directory** are
  opened. Both are deliberate: opening files is the only detection step that costs I/O, so
  it is bounded rather than proportional to somebody's backup folder.
- Loose weight files below a size floor are treated as companion files rather than models.

**Duplicates**

- `aam duplicates` groups on size and file layout unless content digests already exist in
  the catalogue. **No CLI command currently computes those digests**, so in practice rows
  will not carry the `✓` verified mark. The grouping is still reliable for recognising the
  same build shipped by several applications, but it is a heuristic, not a hash.
- The reclaimable figure is an upper bound, not waste. Applications usually re-download a
  bundled model they find missing.

**Analysis depth**

- Everything is read from file headers and metadata. The tool **never loads or runs a
  model**, so it cannot tell you whether weights are numerically valid — only whether the
  files are structurally complete.
- Parameter counts from GGUF and safetensors headers are exact; counts inferred from
  config alone are marked with `~` or "(estimated)".
- No network access. There is no HuggingFace enrichment: everything shown was read from
  your disk.

**Scope**

- Read-only by design. There is no move, rename, delete, dedupe-apply or cleanup command,
  and none is planned for this version.
- Single-user, single-machine, local SQLite. No server, no sharing, no sync.
- One watcher at a time.
- This build is the **CLI only**. The HTTP API and desktop shell in the source tree are
  deliberately excluded from the executable.

**Behaviour worth knowing**

- The automatic catch-up scan runs at most once every 15 minutes, and is skipped entirely
  while a watcher is live or if the previous scan failed or was cancelled.
- `status`, `version`, `guide`, `scan`, `watch`, `discover` and `roots` never trigger it.
- Health scores are advisory. "No licence file" costs a dataset points but may be entirely
  correct for your situation.
- An unrecognised `--sort` or `--group-by` value is ignored silently rather than rejected,
  and `inventory` and `list` use different names for the same field (`date` vs
  `modified`).

---

## Getting help

```powershell
aam --help              # every command, grouped
aam <command> --help    # options for one command
aam guide               # worked examples
aam version --plugins   # what classification rules are loaded
```
