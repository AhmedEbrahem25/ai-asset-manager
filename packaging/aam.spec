# PyInstaller build definition for the `aam` command-line tool.
#
# Run it through `python scripts/build_exe.py`, which cleans up, sets the environment
# variables read below, and smoke-tests the result. `pyinstaller packaging/aam.spec`
# works too if you want the raw build.
#
# The one thing this file has to get right that a default build would not:
#
# Taxonomy plugins are imported *by name at runtime*, discovered from the contents of
# their package directory. Nothing imports them statically, so PyInstaller's analysis has
# no way to see them and would ship a binary whose inventory classified everything as
# "unclassified" while passing every test on the developer's machine. They are enumerated
# from the source tree below and declared as hidden imports.
#
# The enumeration is deliberately a directory listing rather than a written-out list. A
# hard-coded list here would be the same mistake the taxonomy exists to avoid: adding a
# plugin would silently work in development and silently fail in the shipped binary.

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is injected by PyInstaller
sys.path.insert(0, str(PROJECT_ROOT))

PLUGIN_PACKAGE = "ai_asset_manager.backend.taxonomy.plugins"
PLUGIN_DIR = PROJECT_ROOT / PLUGIN_PACKAGE.replace(".", "/")

#: Every taxonomy plugin in the source tree, including the underscore-prefixed helpers the
#: loader skips but the plugins themselves import.
plugin_modules = sorted(
    f"{PLUGIN_PACKAGE}.{path.stem}"
    for path in PLUGIN_DIR.glob("*.py")
    if path.stem != "__init__"
)

if not plugin_modules:
    raise SystemExit(f"No taxonomy plugins found under {PLUGIN_DIR}; refusing to build.")

#: One-file builds unpack the whole bundle to a temporary directory on every run, which
#: costs a second or more of start-up. For a tool whose selling point is answering
#: instantly, one-directory is the default and one-file is opt-in.
ONEFILE = os.environ.get("AAM_BUILD_ONEFILE") == "1"

#: Packages that are declared dependencies of the project but unreachable from the CLI.
#: PyInstaller would not bundle them anyway; listing them makes that explicit and keeps an
#: accidental import from quietly adding 30 MB to the download.
EXCLUDES = [
    "fastapi",
    "uvicorn",
    "starlette",
    "webview",
    "tkinter",
    "unittest",
    "pydoc_data",
    "pytest",
    "_pytest",
    "mypy",
    "ruff",
    "IPython",
    "numpy",
    "matplotlib",
]

analysis = Analysis(  # noqa: F821
    [str(PROJECT_ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=plugin_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)  # noqa: F821

if ONEFILE:
    exe = EXE(  # noqa: F821
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="aam",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        version=os.environ.get("AAM_VERSION_FILE") or None,
    )
else:
    exe = EXE(  # noqa: F821
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="aam",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        version=os.environ.get("AAM_VERSION_FILE") or None,
    )
    COLLECT(  # noqa: F821
        exe,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=False,
        name="aam",
    )
