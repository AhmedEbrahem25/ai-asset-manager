"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ai_asset_manager.backend.database.engine import create_db_engine, reset_engine
from ai_asset_manager.backend.database.schema import init_database
from ai_asset_manager.backend.detectors.registry import DetectorRegistry
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.pipeline import ScanPipeline
from ai_asset_manager.backend.scanner.types import DirectoryTree
from ai_asset_manager.backend.scanner.walker import walk_tree
from ai_asset_manager.config import Settings, get_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings isolated to the test's temporary directory."""
    return Settings(data_dir=tmp_path / "data", scan_workers=2, hash_workers=2)


@pytest.fixture(autouse=True)
def _reset_global_state() -> Iterator[None]:
    """Clear cached settings and engines between tests.

    Both are process-wide singletons; without this a test that points them at its own
    tmp_path would leak that into every test that ran afterwards.
    """
    get_settings.cache_clear()
    reset_engine()
    yield
    get_settings.cache_clear()
    reset_engine()


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Return an initialised in-memory database engine."""
    db_engine = create_db_engine(url="sqlite+pysqlite:///:memory:")
    init_database(db_engine)
    yield db_engine
    db_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Return an open session on the in-memory database."""
    with Session(engine) as db_session:
        yield db_session


@pytest.fixture
def walk(settings: Settings):
    """Return a helper that walks a directory with test settings."""

    def _walk(path: Path) -> DirectoryTree:
        return walk_tree(path, settings=settings)

    return _walk


@pytest.fixture
def context_for(walk):
    """Return a helper building a :class:`DirectoryContext` for a directory."""

    def _context(path: Path, subdirectory: Path | None = None) -> DirectoryContext:
        tree = walk(path)
        target = str(subdirectory or path)
        node = tree.nodes[target]
        return DirectoryContext(tree, node)

    return _context


@pytest.fixture
def pipeline(settings: Settings) -> ScanPipeline:
    """Return a pipeline configured for tests."""
    return ScanPipeline(settings=settings)


@pytest.fixture
def detectors() -> DetectorRegistry:
    """Return the default detector registry."""
    return DetectorRegistry()
