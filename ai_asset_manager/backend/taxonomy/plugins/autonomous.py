"""Autonomous driving datasets.

A worked example of a plugin that adds *domain knowledge* without adding a shelf. KITTI,
Waymo and nuScenes are detection datasets and stay on the detection shelf; what this
plugin contributes is that they are driving data, that they carry LiDAR and radar as well
as camera frames, and that a KITTI directory without ``calib/`` is broken in a way no
generic rule would notice.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_asset_manager.backend.models.enums import DatasetFormat, Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    declared_format,
    family_of,
    is_dataset,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    AssetProfile,
    Classification,
    Finding,
    Task,
)

#: Driving datasets, and the sensor rigs they are distributed with.
DRIVING_FAMILIES = (
    ("KITTI", ("kitti",)),
    ("Waymo", ("waymo",)),
    ("nuScenes", ("nuscenes", "nuimages")),
    ("BDD100K", ("bdd100k", "bdd")),
    ("Argoverse", ("argoverse",)),
    ("Lyft L5", ("lyft-level5", "lyft_level5")),
    ("ApolloScape", ("apolloscape",)),
    ("A2D2", ("a2d2",)),
    ("ONCE", ("once-3d",)),
    ("Cityscapes", ("cityscapes",)),
    ("Mapillary", ("mapillary",)),
)

#: Layouts that are driving datasets by definition.
DRIVING_FORMATS = frozenset(
    {DatasetFormat.KITTI, DatasetFormat.WAYMO, DatasetFormat.NUSCENES, DatasetFormat.BDD100K}
)

#: Layouts that are mask supervision, whatever domain the data comes from. Named locally
#: rather than imported from the vision plugin so that neither has to be installed for the
#: other to work.
_SEGMENTATION_LAYOUTS = frozenset(
    {DatasetFormat.CITYSCAPES, DatasetFormat.ADE20K, DatasetFormat.SEGMENTATION}
)

#: Families that ship LiDAR sweeps. Their absence in such a dataset is a real finding, not
#: a stylistic one — half the published baselines on them are LiDAR-only.
_LIDAR_FAMILIES = frozenset({"KITTI", "Waymo", "nuScenes", "Argoverse", "Lyft L5", "ONCE"})

#: Directory names that hold point clouds, across the rigs above.
_LIDAR_DIRS = ("velodyne", "lidar", "lidar_top", "points", "pointcloud", "velodyne_points",
               "lidar_points", "samples")

#: Directory or file names that hold sensor extrinsics and intrinsics.
_CALIBRATION_MARKERS = ("calib", "calibration", "calibs", "calib_cam_to_cam.txt",
                        "calibrated_sensor.json")

TASKS = (
    Task(id="3d_object_detection", label="3D Object Detection",
         domain="autonomous_driving", order=10),
    Task(id="bev_perception", label="BEV Perception", domain="autonomous_driving", order=20),
    Task(id="lane_detection", label="Lane Detection", domain="autonomous_driving", order=30),
    Task(id="trajectory_prediction", label="Trajectory Prediction",
         domain="autonomous_driving", order=40),
    Task(id="sensor_fusion", label="Sensor Fusion", domain="autonomous_driving", order=50),
    Task(id="drivable_area", label="Drivable Area Segmentation",
         domain="autonomous_driving", order=60),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register driving tasks, the domain override and the sensor health rules."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_alias("driving", ("detection_dataset", "segmentation_dataset",
                                   "tracking_dataset"))

    registry.add_classifier(_driving_dataset, name="autonomous.dataset", priority=610)
    registry.add_health_rule(_lidar_present, name="driving.no_lidar")
    registry.add_health_rule(_calibration_present, name="driving.no_calibration")


def _driving_dataset(profile: AssetProfile) -> Classification | None:
    """Claim driving datasets, keeping the detection shelf but the driving domain."""
    if not is_dataset(profile):
        return None

    family = family_of(profile, DRIVING_FAMILIES)
    layout = declared_format(profile)
    if family is None and layout not in DRIVING_FORMATS:
        return None

    details = profile.dataset
    modalities: list[str] = ["rgb"]
    if details is not None:
        if details.has_lidar or family in _LIDAR_FAMILIES:
            modalities.append("lidar")
        if details.has_radar:
            modalities.append("radar")
        if details.has_depth:
            modalities.append("depth")
    if len(modalities) > 2:
        modalities.append("sensor_fusion")

    # A driving dataset distributed as masks stays on the segmentation shelf. Cityscapes
    # is the case that matters: it is driving data, but nobody trains a box detector on
    # it, and claiming it for detection would put it next to COCO.
    masks = (details is not None and details.has_masks) or layout in _SEGMENTATION_LAYOUTS

    if masks:
        category, task = "segmentation_dataset", "drivable_area"
    elif "lidar" in modalities:
        category, task = "detection_dataset", "3d_object_detection"
    else:
        category, task = "detection_dataset", "object_detection"

    return Classification(
        category=category,
        task=task, domain="autonomous_driving", family=family,
        modalities=tuple(modalities), confidence=CONFIDENCE_CERTAIN,
        evidence=f"{family} driving dataset" if family else f"{layout} layout",
    )


def _lidar_present(profile: AssetProfile) -> Sequence[Finding]:
    """Report a driving dataset distributed with LiDAR that has none on disk.

    Usually means a partial download: the camera archive is a fraction of the size of the
    point-cloud archive, so the images arrive and the sweeps do not.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()

    family = family_of(profile, DRIVING_FAMILIES)
    if family not in _LIDAR_FAMILIES:
        return ()

    details = profile.dataset
    if details is not None and details.has_lidar:
        return ()
    if profile.files.has_dir(*_LIDAR_DIRS) or profile.files.count(".pcd", ".ply", ".las"):
        return ()

    return (
        Finding(
            code="driving.no_lidar",
            severity=Severity.WARNING,
            message=f"{family} normally ships LiDAR sweeps; none found",
            fix_hint="Fetch the point-cloud archive, or ignore if you only need camera data.",
        ),
    )


def _calibration_present(profile: AssetProfile) -> Sequence[Finding]:
    """Report a multi-sensor dataset with no calibration.

    Without extrinsics there is no way to project a point cloud into an image, which makes
    every fusion task on the dataset impossible.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()

    family = family_of(profile, DRIVING_FAMILIES)
    if family not in _LIDAR_FAMILIES:
        return ()

    if profile.files.has_dir(*_CALIBRATION_MARKERS) or profile.files.has_stem("calib"):
        return ()

    return (
        Finding(
            code="driving.no_calibration",
            severity=Severity.WARNING,
            message="No sensor calibration files",
            fix_hint="Camera-LiDAR projection needs the calibration archive.",
        ),
    )
