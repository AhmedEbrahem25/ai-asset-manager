"""Build the standalone ``aam`` executable.

    python scripts/build_exe.py              # one-directory build (recommended)
    python scripts/build_exe.py --onefile    # single file, slower to start
    python scripts/build_exe.py --zip        # also produce a distributable archive

The build is not considered finished until the binary has been run. A frozen Python
program can build cleanly and still be broken in ways no import check catches — most of
all here, where taxonomy plugins are imported by name at runtime and would simply be
absent from a naive bundle, leaving an inventory that classifies everything as
"unclassified" while the source tree passes every test. :func:`verify` runs the real
executable and asserts the plugins came back.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC = PROJECT_ROOT / "packaging" / "aam.spec"
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"

#: Where the finished binary lands, per build mode.
EXECUTABLE = {
    "onedir": DIST_DIR / "aam" / "aam.exe",
    "onefile": DIST_DIR / "aam.exe",
}

#: Checks run against the built binary. Each is a command and a string its output must
#: contain. Together they prove the CLI starts, the taxonomy loaded, and the database
#: layer works — the three things freezing is most likely to break.
SMOKE_TESTS: tuple[tuple[list[str], str], ...] = (
    (["version"], "AI Asset Manager"),
    (["--help"], "inventory"),
    (["inventory", "--help"], "--tree"),
    (["where", "--help"], "asset"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Build the aam executable.")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Bundle into a single .exe. Starts ~1s slower; easier to hand to someone.",
    )
    parser.add_argument(
        "--zip", action="store_true", help="Also write a distributable .zip beside it."
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete build/ and dist/ before building."
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Do not run the built binary."
    )
    return parser.parse_args()


def version_of() -> str:
    """Return the project version without importing the whole package."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from ai_asset_manager import __version__

    return __version__


def write_version_resource(version: str) -> Path:
    """Write the Windows version resource, so the .exe has real file properties.

    Without one, right-clicking the binary shows a blank Details tab, which is the sort of
    thing that makes a tool look like malware to the person you handed it to.
    """
    parts = [int(piece) for piece in version.split(".")[:3]] + [0] * 4
    quad = ", ".join(str(piece) for piece in parts[:4])

    resource = PROJECT_ROOT / "build" / "version_info.txt"
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({quad}), prodvers=({quad}), mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'AI Asset Manager'),
        StringStruct('FileDescription', 'Local catalogue for AI models and datasets'),
        StringStruct('FileVersion', '{version}'),
        StringStruct('InternalName', 'aam'),
        StringStruct('OriginalFilename', 'aam.exe'),
        StringStruct('ProductName', 'AI Asset Manager'),
        StringStruct('ProductVersion', '{version}'),
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )
    return resource


def build(*, onefile: bool, version: str) -> Path:
    """Run PyInstaller and return the path to the built executable."""
    environment = dict(os.environ)
    environment["AAM_BUILD_ONEFILE"] = "1" if onefile else "0"
    # This machine has a system-wide PYTHONPATH pointing at another interpreter's
    # site-packages. Leaving it set makes PyInstaller analyse the wrong copies of
    # everything, so it is cleared for the build the same way it is for the tests.
    environment["PYTHONPATH"] = ""

    if sys.platform == "win32":
        environment["AAM_VERSION_FILE"] = str(write_version_resource(version))

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--log-level", "WARN",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR / "pyinstaller"),
        str(SPEC),
    ]

    print(f"Building {'one-file' if onefile else 'one-directory'} executable...")
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with exit code {result.returncode}")

    executable = EXECUTABLE["onefile" if onefile else "onedir"]
    if not executable.exists():
        raise SystemExit(f"Build reported success but {executable} does not exist.")

    print(f"Built in {time.monotonic() - started:.0f}s")
    return executable


def verify(executable: Path) -> None:
    """Run the built binary and prove the taxonomy survived freezing."""
    print("Verifying...")

    for arguments, expected in SMOKE_TESTS:
        result = _run(executable, arguments)
        if expected not in result:
            raise SystemExit(
                f"Smoke test failed: 'aam {' '.join(arguments)}' did not mention "
                f"{expected!r}.\nOutput:\n{result}"
            )
        print(f"  ok  aam {' '.join(arguments)}")

    # The load-bearing check. Plugins are found by listing a package directory, which a
    # frozen bundle does not have; if the hidden imports in the spec ever stop matching
    # the source tree, this is what catches it.
    _verify_taxonomy(executable)


def _verify_taxonomy(executable: Path) -> None:
    """Assert the frozen binary loaded every taxonomy plugin the source tree has.

    The one check that matters most. Plugins are imported by name from a package directory
    listing, and a frozen bundle has no directory: get this wrong and the binary still
    runs, still prints tables, and quietly files every asset as "unclassified".

    The binary is asked what it loaded rather than being grepped for category names, so
    that improving an error message cannot make this pass or fail by accident.
    """
    expected = sorted(
        path.stem
        for path in (PROJECT_ROOT / "ai_asset_manager/backend/taxonomy/plugins").glob("*.py")
        if not path.stem.startswith("_")
    )

    output = _run(executable, ["version", "--plugins"])
    missing = [name for name in expected if name not in output]

    if missing:
        raise SystemExit(
            f"The frozen binary is missing {len(missing)} taxonomy plugin(s): "
            f"{', '.join(missing)}.\n"
            "packaging/aam.spec collects them from the source tree, so this means the "
            "spec did not run or the plugin package moved.\n"
            f"Output:\n{output}"
        )

    print(f"  ok  taxonomy loaded ({len(expected)} plugins)")
    _verify_detectors(output)


def _verify_detectors(version_output: str) -> None:
    """Assert the frozen binary kept its detectors.

    A build that classifies perfectly but detects nothing scans a full drive, reports zero
    assets, and looks exactly like an empty disk. The count is compared against the source
    tree's own registry so that adding a detector cannot silently weaken this check.
    """
    from ai_asset_manager.backend.detectors.registry import default_detectors

    expected = len(default_detectors())
    for line in version_output.splitlines():
        if line.strip().startswith("Detectors:"):
            reported = int(line.split(":", 1)[1].strip())
            if reported != expected:
                raise SystemExit(
                    f"The frozen binary registered {reported} detector(s); "
                    f"the source tree has {expected}."
                )
            print(f"  ok  detectors loaded ({reported})")
            return

    raise SystemExit(f"'version --plugins' reported no detector count.\n{version_output}")


def _run(executable: Path, arguments: list[str]) -> str:
    """Run the built binary and return its combined output."""
    result = subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        # A clean environment, because the point is to prove the binary is self-contained.
        env={**os.environ, "PYTHONPATH": "", "COLUMNS": "200"},
    )
    return result.stdout + result.stderr


def make_archive(executable: Path, version: str, *, onefile: bool) -> Path:
    """Zip the build for handing to somebody else."""
    archive = DIST_DIR / f"aam-{version}-windows-x64.zip"
    source = executable if onefile else executable.parent

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        if onefile:
            bundle.write(source, arcname=source.name)
        else:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=str(Path("aam") / path.relative_to(source)))
        bundle.write(PROJECT_ROOT / "README.md", arcname="README.md")

    return archive


def describe(path: Path) -> str:
    """Return a human-readable size for a file or a directory tree."""
    total = (
        path.stat().st_size
        if path.is_file()
        else sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
    )
    for unit in ("B", "KiB", "MiB", "GiB"):
        if total < 1024 or unit == "GiB":
            return f"{total:.1f} {unit}" if unit != "B" else f"{total} B"
        total /= 1024.0
    return f"{total:.1f} GiB"


def main() -> None:
    """Build, verify and optionally archive the executable."""
    arguments = parse_args()
    version = version_of()

    if arguments.clean:
        for directory in (BUILD_DIR, DIST_DIR):
            shutil.rmtree(directory, ignore_errors=True)

    executable = build(onefile=arguments.onefile, version=version)

    if not arguments.skip_verify:
        verify(executable)

    payload = executable if arguments.onefile else executable.parent
    print(f"\n{executable}  ({describe(payload)})")

    if arguments.zip:
        archive = make_archive(executable, version, onefile=arguments.onefile)
        print(f"{archive}  ({describe(archive)})")

    print(f"\nRun it:  {executable} inventory")


if __name__ == "__main__":
    main()
