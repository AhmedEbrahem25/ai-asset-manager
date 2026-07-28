"""Remote sensing and aerial imagery.

Satellite and drone datasets are ordinary detection or segmentation datasets in layout,
so this plugin keeps them on those shelves and contributes the domain, the family and the
sensor modalities — the parts that make ``aam inventory remote_sensing`` mean something.
"""

from __future__ import annotations

from ai_asset_manager.backend.taxonomy.plugins._shared import family_of, is_dataset
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_STRONG,
    AssetProfile,
    Classification,
    Task,
)

#: Public earth-observation datasets.
GEO_FAMILIES = (
    ("xView", ("xview",)),
    ("DOTA", ("dota-v", "dota_v", "dotav")),
    ("SpaceNet", ("spacenet",)),
    ("EuroSAT", ("eurosat",)),
    ("BigEarthNet", ("bigearthnet",)),
    ("Sentinel", ("sentinel-1", "sentinel-2", "sentinel2")),
    ("Landsat", ("landsat",)),
    ("FAIR1M", ("fair1m",)),
    ("iSAID", ("isaid",)),
    ("LoveDA", ("loveda",)),
    ("Inria Aerial", ("inria-aerial", "inria_aerial")),
    ("VisDrone", ("visdrone",)),
    ("UAVDT", ("uavdt",)),
)

#: Words that mark earth observation even without a known dataset name.
GEO_MARKERS = ("satellite", "aerial", "remote-sensing", "remote_sensing", "orthophoto",
               "multispectral", "hyperspectral", "geotiff", "earth-observation",
               "land-cover", "landcover", "building-footprint")

TASKS = (
    Task(id="land_cover_classification", label="Land Cover Classification",
         domain="remote_sensing", order=10),
    Task(id="building_extraction", label="Building Extraction",
         domain="remote_sensing", order=20),
    Task(id="change_detection", label="Change Detection", domain="remote_sensing", order=30),
    Task(id="aerial_detection", label="Aerial Object Detection",
         domain="remote_sensing", order=40),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register remote-sensing tasks and the classifier that assigns the domain."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_alias("remote-sensing", ("detection_dataset", "segmentation_dataset",
                                          "image_dataset"))

    registry.add_classifier(_geospatial_dataset, name="geospatial.dataset", priority=615)


def _geospatial_dataset(profile: AssetProfile) -> Classification | None:
    """Claim earth-observation datasets, keeping their vision shelf."""
    if not is_dataset(profile):
        return None

    family = family_of(profile, GEO_FAMILIES)
    marker = profile.matches(GEO_MARKERS)
    if family is None and marker is None:
        return None

    details = profile.dataset
    haystack = profile.haystack

    if details is not None and details.has_masks:
        category = "segmentation_dataset"
        task = "building_extraction" if "building" in haystack else "land_cover_classification"
    elif details is not None and details.has_bounding_boxes:
        category, task = "detection_dataset", "aerial_detection"
    elif "change" in haystack:
        category, task = "segmentation_dataset", "change_detection"
    else:
        category, task = "image_dataset", "land_cover_classification"

    modalities = ["rgb"]
    if "multispectral" in haystack or "hyperspectral" in haystack or family in (
        "Sentinel", "Landsat", "BigEarthNet", "EuroSAT"
    ):
        modalities.append("infrared")

    return Classification(
        category=category, task=task, domain="remote_sensing", family=family,
        modalities=tuple(modalities), confidence=CONFIDENCE_STRONG,
        evidence=f"{family} dataset" if family else f"name contains {marker!r}",
    )
