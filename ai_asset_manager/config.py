"""Application settings.

Settings are resolved from, in decreasing precedence: explicit constructor arguments,
environment variables prefixed ``AAM_``, a ``.env`` file, then the defaults below.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Directory entries never descended into. Matching is case-insensitive on the raw
#: directory name. Keeping this as a frozenset makes the walker's hot-path check O(1).
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        # HuggingFace hub bookkeeping. `.locks` mirrors every repo name, so leaving it in
        # catalogues each cached repo twice. `.no_exist` holds zero-byte markers named
        # after files the repo does *not* have — including `adapter_config.json`, which
        # would otherwise make every cached model look like a LoRA adapter.
        ".locks",
        ".no_exist",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "dist-packages",
        # Operating-system and vendor trees. Excluding `windows` alone is not enough:
        # pointing a scan at `C:\` walks `Program Files`, `ProgramData` and every installed
        # application, and on the development machine that had not finished after twenty
        # minutes. Discovery has always refused to enter these; the scanner now agrees.
        #
        # This is a default for *walks that reach them*, not a prohibition. Naming a path
        # explicitly still scans it, because the walker only checks the names of
        # directories it discovers, never the root it was given -- so
        # `aam scan "C:\Program Files\SomeApp"` works exactly as asked.
        "$recycle.bin",
        "system volume information",
        "windows",
        "$windows.~ws",
        "$windows.~bt",
        "$winreagent",
        "program files",
        "program files (x86)",
        "programdata",
        "perflogs",
        "msocache",
        "config.msi",
        "recovery",
        "appdata\\local\\temp",
        # Browser, Electron and package-manager caches. Universally large, universally
        # worthless to a catalogue, and the reason a whole-drive walk spends its time in
        # AppData: over a million files on the development machine, none of them an asset.
        # Discovery has always refused to enter these.
        "cache2",
        "code cache",
        "gpucache",
        "service worker",
        "crashpad_reports",
        "webcache",
        "inetcache",
        "temporary internet files",
        ".npm",
        ".nuget",
        ".cargo",
        ".rustup",
        ".pnpm-store",
        ".yarn",
        # Python package caches. The same thing as `site-packages`, one step earlier:
        # unpacked wheels, including their test fixtures. `pyarrow`'s bundled sample
        # Parquet files were being catalogued as HuggingFace datasets.
        "appdata\\local\\uv\\cache",
        "appdata\\local\\pip\\cache",
        ".cache\\uv",
        ".cache\\pip",
        ".trash",
        ".trash-1000",
        "lost+found",
        ".dropbox.cache",
        ".terraform",
        "target",
        "build",
        ".gradle",
        ".tox",
        ".nox",
    }
)

#: Files that are never treated as evidence of an asset.
DEFAULT_EXCLUDED_FILE_GLOBS: tuple[str, ...] = (
    "*.tmp",
    "*.swp",
    "~$*",
    "desktop.ini",
    "Thumbs.db",
    ".DS_Store",
)


def _default_data_dir() -> Path:
    """Return the per-user data directory, honouring platform conventions."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return Path(base) / "AIAssetManager"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "ai-asset-manager"
    return Path.home() / ".local" / "share" / "ai-asset-manager"


class Settings(BaseSettings):
    """Runtime configuration for every entry point (CLI, API and desktop shell)."""

    model_config = SettingsConfigDict(
        env_prefix="AAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- storage ------------------------------------------------------------
    data_dir: Path = Field(default_factory=_default_data_dir)
    database_url: str | None = Field(
        default=None,
        description="Full SQLAlchemy URL. When unset, a SQLite file under data_dir is used.",
    )
    echo_sql: bool = False

    # -- scanning -----------------------------------------------------------
    scan_workers: int = Field(default=8, ge=1, le=64)
    hash_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Kept low by default: parallel reads thrash spinning disks.",
    )
    follow_symlinks: bool = Field(
        default=False,
        description="Descending into symlinked directories risks cycles; off by default.",
    )
    max_depth: int = Field(default=40, ge=1, description="Guards against pathological trees.")
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS
    excluded_file_globs: tuple[str, ...] = DEFAULT_EXCLUDED_FILE_GLOBS

    # -- hashing ------------------------------------------------------------
    quick_hash_chunk_bytes: int = Field(default=4 * 1024 * 1024, ge=64 * 1024)
    quick_hash_min_file_bytes: int = Field(
        default=1024 * 1024,
        ge=0,
        description="Files smaller than this are hashed in full; chunking buys nothing.",
    )
    full_hash_max_bytes: int = Field(
        default=0,
        ge=0,
        description="Refuse full SHA256 above this size (0 disables the limit).",
    )

    # -- parsing ------------------------------------------------------------
    max_header_read_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=4096,
        description="Upper bound on bytes read from any single binary header.",
    )
    max_safetensors_header_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)

    # -- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8420, ge=1, le=65535)
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    # -- behaviour ----------------------------------------------------------
    log_level: str = "INFO"
    log_file: Path | None = None
    online_mode: bool = Field(
        default=False,
        description="Reserved for HuggingFace enrichment; the app never requires network.",
    )
    # -- live indexing ------------------------------------------------------
    #: How long filesystem events must stop before a batch is processed.
    watch_debounce_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    #: Process a batch anyway once its oldest event is this old. Without a ceiling, a
    #: multi-gigabyte download that writes continuously for minutes would hold every
    #: change back until it finished, and the inventory would be stale the whole time.
    watch_max_wait_seconds: float = Field(default=30.0, ge=1.0, le=600.0)

    #: Run a quick incremental scan before commands that read the catalogue, so the
    #: inventory reflects the disk without the user remembering to rescan.
    auto_scan: bool = Field(default=True)
    #: Do not repeat that scan more often than this. A user running several commands in a
    #: row should pay for it once, not once per command.
    auto_scan_interval_seconds: float = Field(default=900.0, ge=0.0)
    #: Offer to catalogue discovered locations on first use.
    auto_discover: bool = Field(default=True)

    @field_validator("excluded_dirs", mode="before")
    @classmethod
    def _normalise_excluded_dirs(cls, value: object) -> object:
        """Lower-case directory exclusions so walker comparisons stay case-insensitive."""
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set, frozenset)):
            return frozenset(str(item).lower() for item in value)
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        """Accept lower-case log levels from the environment."""
        return value.upper() if isinstance(value, str) else value

    @property
    def db_path(self) -> Path:
        """Return the default on-disk SQLite location."""
        return self.data_dir / "catalog.db"

    @property
    def resolved_database_url(self) -> str:
        """Return the effective SQLAlchemy URL, defaulting to SQLite under ``data_dir``."""
        if self.database_url:
            return self.database_url
        return f"sqlite+pysqlite:///{self.db_path.as_posix()}"

    @property
    def is_sqlite(self) -> bool:
        """Return whether the configured backend is SQLite."""
        return self.resolved_database_url.startswith("sqlite")

    def ensure_data_dir(self) -> Path:
        """Create the data directory if needed and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that every injection site observes the same object. Tests clear the cache
    via ``get_settings.cache_clear()``.
    """
    return Settings()
