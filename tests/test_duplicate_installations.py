"""The same model, shipped by several applications."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ai_asset_manager.backend.duplicate import (
    find_duplicate_installations,
    total_reclaimable,
)
from ai_asset_manager.backend.models import Asset, AssetFile

MIB = 1024 * 1024


def _install(
    session: Session,
    path: str,
    *,
    size: int = 8 * MIB,
    files: tuple[tuple[str, int], ...] = (("model.tflite", 8 * MIB),),
    digests: tuple[str, ...] = (),
    identity: dict[str, str] | None = None,
) -> Asset:
    """Persist one catalogued copy of a model."""
    asset = Asset(
        kind="model",
        name="model",
        root_path=path,
        size_bytes=size,
        physical_size_bytes=size,
        file_count=len(files),
        format="tflite",
        evidence={"identity": identity} if identity else {},
    )
    session.add(asset)
    session.flush()

    for index, (relpath, file_size) in enumerate(files):
        session.add(
            AssetFile(
                asset_id=asset.id,
                relpath=relpath,
                size_bytes=file_size,
                extension=".tflite",
                sha256=digests[index] if index < len(digests) else None,
            )
        )
    session.flush()
    return asset


def test_groups_the_same_model_installed_by_several_applications(session: Session):
    for product, source, path in (
        ("Chrome", "chrome", r"C:\Users\p\AppData\Local\Google\Chrome\og\model.tflite"),
        ("Edge", "edge", r"C:\Users\p\AppData\Local\Microsoft\Edge\og\model.tflite"),
        ("Cursor", "cursor", r"C:\Users\p\AppData\Local\Programs\cursor\og\model.tflite"),
    ):
        _install(session, path, identity={"product": product, "source": source})

    groups = find_duplicate_installations(session)

    assert len(groups) == 1
    group = groups[0]
    assert group.install_count == 3
    assert group.unit_size_bytes == 8 * MIB
    assert group.reclaimable_bytes == 16 * MIB
    assert group.sources == ["Chrome", "Cursor", "Edge"]
    assert group.spans_applications
    assert not group.verified_by_hash


def test_hashes_are_preferred_and_reported_when_present(session: Session):
    for path in (r"C:\a\model.tflite", r"C:\b\model.tflite"):
        _install(session, path, digests=("f" * 64,))

    groups = find_duplicate_installations(session)

    assert len(groups) == 1
    assert groups[0].verified_by_hash


def test_same_size_but_different_content_is_not_grouped(session: Session):
    _install(session, r"C:\a\model.tflite", digests=("a" * 64,))
    _install(session, r"C:\b\model.tflite", digests=("b" * 64,))

    assert find_duplicate_installations(session) == []


def test_different_file_layouts_are_not_grouped(session: Session):
    _install(session, r"C:\a\m", files=(("model.tflite", 8 * MIB),))
    _install(
        session,
        r"C:\b\m",
        files=(("encoder.tflite", 4 * MIB), ("decoder.tflite", 4 * MIB)),
    )

    assert find_duplicate_installations(session) == []


def test_small_models_are_below_the_reporting_floor(session: Session):
    for path in (r"C:\a\tiny.tflite", r"C:\b\tiny.tflite"):
        _install(session, path, size=1024, files=(("model.tflite", 1024),))

    assert find_duplicate_installations(session) == []


def test_one_application_keeping_two_copies_can_be_filtered_out(session: Session):
    for path in (r"C:\chrome\v1\model.tflite", r"C:\chrome\v2\model.tflite"):
        _install(session, path, identity={"product": "Chrome", "source": "chrome"})

    assert len(find_duplicate_installations(session)) == 1
    assert find_duplicate_installations(session, across_applications_only=True) == []


def test_missing_assets_are_excluded(session: Session):
    _install(session, r"C:\a\model.tflite")
    gone = _install(session, r"C:\b\model.tflite")
    gone.is_missing = True
    session.flush()

    assert find_duplicate_installations(session) == []


def test_identity_is_derived_when_the_catalogue_predates_it(session: Session):
    """A catalogue written before the identity layer still reports which app holds a copy."""
    _install(session, r"C:\Users\p\AppData\Local\Google\Chrome\og\model.tflite")
    _install(session, r"C:\Users\p\AppData\Roaming\Zoom\bin\model.tflite")

    groups = find_duplicate_installations(session)

    assert groups[0].sources == ["Chrome", "Zoom"]


def test_total_is_the_sum_of_every_group(session: Session):
    for path in (r"C:\a\model.tflite", r"C:\b\model.tflite", r"C:\c\model.tflite"):
        _install(session, path)
    for path in (r"C:\d\other", r"C:\e\other"):
        _install(session, path, size=4 * MIB, files=(("w.tflite", 4 * MIB),))

    groups = find_duplicate_installations(session)

    assert len(groups) == 2
    assert total_reclaimable(groups) == 16 * MIB + 4 * MIB
