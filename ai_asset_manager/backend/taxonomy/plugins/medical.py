"""Medical imaging.

A domain the scanner knows nothing about. It has no DICOM detector and no NIfTI parser,
so a folder of scans arrives in the catalogue as a generic dataset — and yet this plugin
can still recognise it, name its task and give it its own shelf, because the file
extensions the scanner recorded are enough.

That is the point of the plugin system: knowledge can be added where it is cheap, without
touching the scanner, the schema or the CLI.
"""

from __future__ import annotations

from ai_asset_manager.backend.taxonomy.plugins._shared import family_of, is_dataset
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Extensions that only medical imaging uses.
MEDICAL_EXTENSIONS = (".dcm", ".dicom", ".nii", ".mha", ".mhd", ".nrrd", ".dcm30")

#: Public medical datasets, by name.
MEDICAL_FAMILIES = (
    ("BraTS", ("brats",)),
    ("LUNA16", ("luna16", "luna-16")),
    ("CheXpert", ("chexpert",)),
    ("MIMIC-CXR", ("mimic-cxr", "mimic_cxr")),
    ("ISIC", ("isic",)),
    ("Medical Segmentation Decathlon", ("msd-", "medical-decathlon", "medicaldecathlon")),
    ("ACDC", ("acdc",)),
    ("KiTS", ("kits19", "kits21", "kits23")),
    ("PadChest", ("padchest",)),
    ("NIH ChestX-ray", ("chestx-ray", "chestxray")),
    ("RSNA", ("rsna-",)),
    ("TotalSegmentator", ("totalsegmentator",)),
)

#: Words that mark clinical imagery even in a folder of ordinary PNGs.
MEDICAL_MARKERS = ("dicom", "radiolog", "histopath", "pathology", "mammograph",
                   "ct-scan", "ct_scan", "mri", "x-ray", "xray", "ultrasound",
                   "endoscop", "retinopathy", "fundus", "tumour", "tumor", "lesion",
                   "clinical", "biomedical")

TASKS = (
    Task(id="medical_segmentation", label="Medical Segmentation", domain="medical", order=10),
    Task(id="lesion_detection", label="Lesion Detection", domain="medical", order=20),
    Task(id="diagnosis", label="Diagnosis", domain="medical", order=30),
    Task(id="radiology_report", label="Radiology Report Generation", domain="medical", order=40),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the medical imaging shelf and its classifier."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="medical_dataset", label="Medical Dataset", section="datasets", order=280,
                 domain="medical", aliases=("medical-datasets", "medical-imaging"),
                 description="Clinical and biomedical imagery.")
    )

    registry.add_classifier(_medical_dataset, name="medical.dataset", priority=620)


def _medical_dataset(profile: AssetProfile) -> Classification | None:
    """Claim clinical imagery, by file format first and by name second."""
    if not is_dataset(profile):
        return None

    scans = profile.files.count(*MEDICAL_EXTENSIONS)
    compressed = sum(1 for relpath in profile.files.relpaths if relpath.endswith(".nii.gz"))
    family = family_of(profile, MEDICAL_FAMILIES)
    marker = profile.matches(MEDICAL_MARKERS)

    if not scans and not compressed and family is None and marker is None:
        return None

    details = profile.dataset
    if details is not None and details.has_masks:
        task = "medical_segmentation"
    elif details is not None and details.has_bounding_boxes:
        task = "lesion_detection"
    elif "report" in profile.haystack:
        task = "radiology_report"
    else:
        task = "diagnosis"

    volumetric = bool(scans or compressed)
    return Classification(
        category="medical_dataset", task=task, domain="medical", family=family,
        modalities=("depth", "rgb") if volumetric else ("rgb",),
        confidence=CONFIDENCE_CERTAIN if volumetric else CONFIDENCE_STRONG,
        evidence=f"{scans + compressed:,} DICOM/NIfTI file(s)" if volumetric
        else (f"{family} dataset" if family else f"name contains {marker!r}"),
    )
