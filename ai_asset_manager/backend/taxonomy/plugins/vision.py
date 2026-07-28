"""Computer vision: detection, segmentation, tracking, classification and their datasets.

The largest built-in plugin, because vision has the most distinct tasks and the least
self-description. A detector and a classifier are both "a model with convolutions"; what
separates them is the head, and bare ``.pt`` checkpoints carry no configuration saying so.
Family names therefore do a lot of work here, and each rule reports the marker it matched
so a surprising answer can be traced.
"""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import DatasetFormat, ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    declared_format,
    family_of,
    image_count,
    is_dataset,
    is_model,
    video_count,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Model families, longest marker first within each entry so that ``yolo-world`` is not
#: swallowed by ``yolo``.
DETECTION_FAMILIES = (
    ("YOLO-World", ("yolo-world", "yoloworld")),
    ("YOLO", ("yolov", "yolo11", "yolo12", "yolox", "yolo")),
    ("RT-DETR", ("rt-detr", "rtdetr")),
    ("Grounding DINO", ("grounding-dino", "groundingdino", "grounding_dino")),
    ("DETR", ("detr",)),
    ("DINO", ("dinov2", "dino")),
    ("Faster R-CNN", ("faster-rcnn", "fasterrcnn", "faster_rcnn")),
    ("Mask R-CNN", ("mask-rcnn", "maskrcnn", "mask_rcnn")),
    ("RetinaNet", ("retinanet",)),
    ("SSD", ("ssd300", "ssdlite", "ssd")),
    ("EfficientDet", ("efficientdet",)),
    ("CenterNet", ("centernet",)),
    ("DEIM", ("deim",)),
)

SEGMENTATION_FAMILIES = (
    ("SAM", ("sam2", "sam_", "segment-anything", "segment_anything", "mobile-sam", "sam-vit")),
    ("SegFormer", ("segformer",)),
    ("Mask2Former", ("mask2former",)),
    ("MaskFormer", ("maskformer",)),
    ("OneFormer", ("oneformer",)),
    ("DeepLab", ("deeplab",)),
    ("U-Net", ("unet", "u-net")),
    ("YOLO", ("yolov8-seg", "yolo-seg", "yolov")),
)

CLASSIFICATION_FAMILIES = (
    ("ConvNeXt", ("convnext",)),
    ("EfficientNet", ("efficientnet",)),
    ("MobileNet", ("mobilenet",)),
    ("ResNeXt", ("resnext",)),
    ("ResNet", ("resnet",)),
    ("RegNet", ("regnet",)),
    ("DenseNet", ("densenet",)),
    ("ShuffleNet", ("shufflenet",)),
    ("SqueezeNet", ("squeezenet",)),
    ("Inception", ("inception", "googlenet")),
    ("VGG", ("vgg",)),
    ("AlexNet", ("alexnet",)),
    ("MNASNet", ("mnasnet",)),
    ("Swin", ("swin",)),
    ("DeiT", ("deit",)),
    ("ViT", ("vit-", "vit_", "vision-transformer")),
    ("DINOv2", ("dinov2",)),
)

TRACKING_FAMILIES = (
    ("ByteTrack", ("bytetrack",)),
    ("BoT-SORT", ("botsort", "bot-sort")),
    ("DeepSORT", ("deepsort", "deep-sort")),
    ("OC-SORT", ("ocsort", "oc-sort")),
    ("StrongSORT", ("strongsort",)),
    ("FairMOT", ("fairmot",)),
    ("TrackFormer", ("trackformer",)),
    ("JDE", ("jde",)),
)

DEPTH_FAMILIES = (
    ("Depth Anything", ("depth-anything", "depth_anything", "depthanything")),
    ("MiDaS", ("midas",)),
    ("ZoeDepth", ("zoedepth",)),
    ("DPT", ("dpt-", "dpt_")),
    ("Metric3D", ("metric3d",)),
)

POSE_FAMILIES = (
    ("YOLO-Pose", ("yolov8-pose", "yolo-pose", "yolo11-pose")),
    ("OpenPose", ("openpose",)),
    ("HRNet", ("hrnet",)),
    ("MediaPipe", ("mediapipe", "blazepose")),
    ("ViTPose", ("vitpose",)),
    ("RTMPose", ("rtmpose",)),
)

#: Dataset layouts whose whole purpose is bounding boxes.
DETECTION_FORMATS = frozenset(
    {
        DatasetFormat.COCO,
        DatasetFormat.YOLO,
        DatasetFormat.PASCAL_VOC,
        DatasetFormat.KITTI,
        DatasetFormat.WAYMO,
        DatasetFormat.NUSCENES,
        DatasetFormat.BDD100K,
        DatasetFormat.CROWDHUMAN,
        DatasetFormat.OPEN_IMAGES,
        DatasetFormat.LVIS,
    }
)

#: Dataset layouts whose whole purpose is pixel masks.
SEGMENTATION_FORMATS = frozenset(
    {DatasetFormat.CITYSCAPES, DatasetFormat.ADE20K, DatasetFormat.SEGMENTATION}
)

#: Well-known dataset families, so a folder called ``coco2017`` is recognisably COCO.
DATASET_FAMILIES = (
    ("COCO", ("coco",)),
    ("LVIS", ("lvis",)),
    ("Open Images", ("open-images", "openimages", "oid")),
    ("Objects365", ("objects365", "obj365")),
    ("Pascal VOC", ("pascal", "voc2007", "voc2012", "voc")),
    ("ImageNet", ("imagenet", "ilsvrc")),
    ("Cityscapes", ("cityscapes",)),
    ("ADE20K", ("ade20k", "ade_20k")),
    ("KITTI", ("kitti",)),
    ("Waymo", ("waymo",)),
    ("nuScenes", ("nuscenes",)),
    ("BDD100K", ("bdd100k", "bdd")),
    ("CrowdHuman", ("crowdhuman",)),
    ("MOT", ("mot17", "mot20", "mot16", "motchallenge")),
    ("VisDrone", ("visdrone",)),
    ("DOTA", ("dota",)),
    ("CIFAR", ("cifar",)),
    ("MNIST", ("mnist",)),
)

#: Transformer heads that only ever sit on text. A model carrying one is a language model
#: whatever its declared type says, so the vision rules decline it and let the NLP plugin
#: take it.
_TEXT_HEADS = ("forsequenceclassification", "fortokenclassification", "formultiplechoice",
               "forquestionanswering", "formaskedlm")

TASKS = (
    Task(id="object_detection", label="Object Detection", domain="vision", order=10),
    Task(id="instance_segmentation", label="Instance Segmentation", domain="vision", order=20),
    Task(id="semantic_segmentation", label="Semantic Segmentation", domain="vision", order=30),
    Task(id="panoptic_segmentation", label="Panoptic Segmentation", domain="vision", order=40),
    Task(id="image_classification", label="Image Classification", domain="vision", order=50),
    Task(id="pose_estimation", label="Pose Estimation", domain="vision", order=60),
    Task(id="depth_estimation", label="Depth Estimation", domain="vision", order=70),
    Task(id="stereo_vision", label="Stereo Vision", domain="vision", order=80),
    Task(id="optical_flow", label="Optical Flow", domain="vision", order=90),
    Task(id="tracking", label="Object Tracking", domain="vision", order=100),
    Task(id="multi_object_tracking", label="Multi-Object Tracking", domain="vision", order=110),
    Task(id="re_identification", label="Re-Identification", domain="vision", order=120),
    Task(id="face_recognition", label="Face Recognition", domain="vision", order=130),
    Task(id="anomaly_detection", label="Anomaly Detection", domain="vision", order=140),
    Task(id="super_resolution", label="Super Resolution", domain="vision", order=150),
    Task(id="image_restoration", label="Image Restoration", domain="vision", order=160),
    Task(id="image_retrieval", label="Image Retrieval", domain="vision", order=170),
    Task(id="video_classification", label="Video Classification", domain="vision", order=180),
    Task(id="feature_extraction", label="Feature Extraction", domain="vision", order=190),
)

CATEGORIES = (
    Category(id="object_detection", label="Object Detection", section="models", order=40,
             domain="vision", aliases=("detection", "detectors")),
    Category(id="segmentation", label="Segmentation", section="models", order=50,
             domain="vision", aliases=("segmentation-models", "seg")),
    Category(id="tracking", label="Tracking", section="models", order=60, domain="vision",
             aliases=("trackers", "mot")),
    Category(id="classification", label="Classification", section="models", order=70,
             domain="vision", aliases=("classifiers", "backbones")),
    Category(id="vision", label="Vision Model", section="models", order=80, domain="vision",
             aliases=("vision-models",)),
    Category(id="detection_dataset", label="Detection Dataset", section="datasets", order=200,
             domain="vision", aliases=("detection-datasets",)),
    Category(id="segmentation_dataset", label="Segmentation Dataset", section="datasets",
             order=210, domain="vision", aliases=("segmentation-datasets",)),
    Category(id="tracking_dataset", label="Tracking Dataset", section="datasets", order=220,
             domain="vision", aliases=("tracking-datasets",)),
    Category(id="image_dataset", label="Image Dataset", section="datasets", order=230,
             domain="vision", aliases=("image-datasets", "images")),
    Category(id="video_dataset", label="Video Dataset", section="datasets", order=240,
             domain="vision", aliases=("video-datasets", "videos")),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register vision categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)
    for category in CATEGORIES:
        registry.add_category(category)

    # "vision" as a selector means every model a computer-vision engineer would call one,
    # which is wider than the "vision" category and narrower than the vision domain — the
    # domain would drag the datasets in too.
    registry.add_alias(
        "vision",
        (
            "vision",
            "object_detection",
            "segmentation",
            "classification",
            "tracking",
            "vision_language",
        ),
    )
    registry.add_alias("vision-datasets", (
        "detection_dataset", "segmentation_dataset", "tracking_dataset",
        "image_dataset", "video_dataset",
    ))

    registry.add_classifier(_tracking_model, name="vision.tracker", priority=660)
    registry.add_classifier(_detection_model, name="vision.detector", priority=650)
    registry.add_classifier(_segmentation_model, name="vision.segmenter", priority=640)
    registry.add_classifier(_geometry_model, name="vision.geometry", priority=630)
    registry.add_classifier(_classification_model, name="vision.classifier", priority=620)
    registry.add_classifier(_generic_vision_model, name="vision.generic", priority=610)
    registry.add_classifier(_detection_dataset, name="vision.detection-dataset", priority=600)
    registry.add_classifier(_segmentation_dataset, name="vision.segmentation-dataset", priority=590)
    registry.add_classifier(_tracking_dataset, name="vision.tracking-dataset", priority=580)
    registry.add_classifier(_video_dataset, name="vision.video-dataset", priority=570)
    registry.add_classifier(_image_dataset, name="vision.image-dataset", priority=560)


# -- models -----------------------------------------------------------------


def _tracking_model(profile: AssetProfile) -> Classification | None:
    """Claim multi-object trackers.

    Ranked above detection because a tracker *is* a detector plus association, and the
    detection rule would otherwise take it. Trackers have no distinguishing architecture,
    so the family name is the only evidence there is.
    """
    if not is_model(profile):
        return None
    family = family_of(profile, TRACKING_FAMILIES)
    if family is None:
        return None
    return Classification(
        category="tracking",
        task="multi_object_tracking",
        domain="vision",
        family=family,
        modalities=("rgb",),
        confidence=CONFIDENCE_WEAK,
        evidence=f"{family} tracker",
    )


def _detection_model(profile: AssetProfile) -> Classification | None:
    """Claim object detectors."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, DETECTION_FAMILIES)

    if declared == ModelType.OBJECT_DETECTION:
        return Classification(
            category="object_detection", task="object_detection", domain="vision",
            family=family, modalities=("rgb",), confidence=CONFIDENCE_CERTAIN,
            evidence="declared an object-detection model",
        )

    if family is not None:
        return Classification(
            category="object_detection", task="object_detection", domain="vision",
            family=family, modalities=("rgb",), confidence=CONFIDENCE_WEAK,
            evidence=f"{family} detector",
        )

    return None


def _segmentation_model(profile: AssetProfile) -> Classification | None:
    """Claim segmentation models, distinguishing the three segmentation tasks by name."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, SEGMENTATION_FAMILIES)

    if declared != ModelType.SEGMENTATION and family is None:
        return None

    haystack = profile.haystack
    if "panoptic" in haystack:
        task = "panoptic_segmentation"
    elif "instance" in haystack or family in ("Mask R-CNN", "SAM"):
        task = "instance_segmentation"
    else:
        task = "semantic_segmentation"

    return Classification(
        category="segmentation", task=task, domain="vision", family=family,
        modalities=("rgb",),
        confidence=CONFIDENCE_CERTAIN if declared == ModelType.SEGMENTATION
        else CONFIDENCE_WEAK,
        evidence=f"{family} segmentation model" if family else "declared a segmentation model",
    )


def _geometry_model(profile: AssetProfile) -> Classification | None:
    """Claim pose, depth and flow models.

    Grouped because they share a shape — an image in, a dense or structured prediction out
    — and because none of them is common enough to deserve a shelf of its own. The task
    field keeps them distinguishable.
    """
    if not is_model(profile):
        return None

    depth = family_of(profile, DEPTH_FAMILIES)
    if depth is not None or profile.matches(("depth-estimation", "monodepth")):
        return Classification(
            category="vision", task="depth_estimation", domain="vision", family=depth,
            modalities=("rgb", "depth"), confidence=CONFIDENCE_WEAK,
            evidence=f"{depth} depth model" if depth else "depth estimation model",
        )

    pose = family_of(profile, POSE_FAMILIES)
    declared = profile.model.model_type if profile.model else None
    if pose is not None or declared == ModelType.POSE:
        return Classification(
            category="vision", task="pose_estimation", domain="vision", family=pose,
            modalities=("rgb",),
            confidence=CONFIDENCE_CERTAIN if declared == ModelType.POSE else CONFIDENCE_WEAK,
            evidence=f"{pose} pose model" if pose else "declared a pose model",
        )

    if profile.matches(("optical-flow", "raft", "flownet")):
        return Classification(
            category="vision", task="optical_flow", domain="vision", modalities=("rgb",),
            confidence=CONFIDENCE_WEAK, evidence="optical flow model",
        )

    return None


def _classification_model(profile: AssetProfile) -> Classification | None:
    """Claim image classifiers and the backbones that ship as bare checkpoints.

    Torchvision and timm distribute ``resnet18.pth`` with nothing beside it — no config,
    no card, no tokenizer. The family name is the entire evidence base, and without this
    rule every backbone on a machine files itself as "other".
    """
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, CLASSIFICATION_FAMILIES)
    architecture = (profile.model.architecture or "").lower() if profile.model else ""

    # "Classification" is not a vision word. A sentiment model and a ResNet are both
    # classifiers, and the catalogue records both as ModelType.CLASSIFICATION, so the
    # declaration alone cannot be trusted — the head gives it away instead.
    if architecture.endswith(_TEXT_HEADS):
        return None

    if declared == ModelType.CLASSIFICATION:
        return Classification(
            category="classification", task="image_classification", domain="vision",
            family=family, modalities=("rgb",), confidence=CONFIDENCE_CERTAIN,
            evidence="declared an image classifier",
        )

    if family is not None:
        return Classification(
            category="classification", task="image_classification", domain="vision",
            family=family, modalities=("rgb",), confidence=CONFIDENCE_WEAK,
            evidence=f"{family} backbone",
        )

    return None


def _generic_vision_model(profile: AssetProfile) -> Classification | None:
    """Claim models the scanner called vision without saying which vision task."""
    if not is_model(profile):
        return None
    if (profile.model.model_type if profile.model else None) != ModelType.VISION:
        return None
    return Classification(
        category="vision", domain="vision", modalities=("rgb",),
        confidence=CONFIDENCE_STRONG, evidence="declared a vision model",
    )


# -- datasets ---------------------------------------------------------------


def _detection_dataset(profile: AssetProfile) -> Classification | None:
    """Claim datasets laid out for bounding-box supervision."""
    if not is_dataset(profile):
        return None

    details = profile.dataset
    layout = declared_format(profile)

    if layout not in DETECTION_FORMATS and not (details and details.has_bounding_boxes):
        return None

    return Classification(
        category="detection_dataset", task="object_detection", domain="vision",
        family=family_of(profile, DATASET_FAMILIES), modalities=("rgb",),
        confidence=CONFIDENCE_CERTAIN,
        evidence=f"{layout} layout" if layout else "bounding-box annotations",
    )


def _segmentation_dataset(profile: AssetProfile) -> Classification | None:
    """Claim datasets laid out for mask supervision."""
    if not is_dataset(profile):
        return None

    details = profile.dataset
    layout = declared_format(profile)

    if layout not in SEGMENTATION_FORMATS and not (details and details.has_masks):
        return None

    haystack = profile.haystack
    task = "panoptic_segmentation" if "panoptic" in haystack else (
        "instance_segmentation" if "instance" in haystack else "semantic_segmentation"
    )

    return Classification(
        category="segmentation_dataset", task=task, domain="vision",
        family=family_of(profile, DATASET_FAMILIES), modalities=("rgb",),
        confidence=CONFIDENCE_CERTAIN,
        evidence=f"{layout} layout" if layout else "mask annotations",
    )


def _tracking_dataset(profile: AssetProfile) -> Classification | None:
    """Claim datasets of annotated sequences."""
    if not is_dataset(profile):
        return None
    if declared_format(profile) not in (DatasetFormat.MOT, DatasetFormat.TRACKING):
        return None
    return Classification(
        category="tracking_dataset", task="multi_object_tracking", domain="vision",
        family=family_of(profile, DATASET_FAMILIES), modalities=("video", "rgb"),
        confidence=CONFIDENCE_CERTAIN, evidence="sequence layout with track ids",
    )


def _video_dataset(profile: AssetProfile) -> Classification | None:
    """Claim datasets that are mostly video."""
    if not is_dataset(profile):
        return None

    videos = video_count(profile)
    if declared_format(profile) != DatasetFormat.VIDEO and videos <= image_count(profile):
        return None
    if not videos and declared_format(profile) != DatasetFormat.VIDEO:
        return None

    return Classification(
        category="video_dataset", task="video_classification", domain="vision",
        family=family_of(profile, DATASET_FAMILIES), modalities=("video",),
        confidence=CONFIDENCE_STRONG, evidence=f"{videos:,} video file(s)",
    )


def _image_dataset(profile: AssetProfile) -> Classification | None:
    """Claim image collections, labelled by folder or not labelled at all."""
    if not is_dataset(profile):
        return None

    layout = declared_format(profile)
    images = image_count(profile)

    if layout not in (DatasetFormat.IMAGENET, DatasetFormat.IMAGE_CLASSIFICATION) \
            and images < 1:
        return None

    labelled = layout in (DatasetFormat.IMAGENET, DatasetFormat.IMAGE_CLASSIFICATION) or bool(
        profile.dataset and profile.dataset.num_classes
    )

    return Classification(
        category="image_dataset",
        task="image_classification" if labelled else "feature_extraction",
        domain="vision", family=family_of(profile, DATASET_FAMILIES), modalities=("rgb",),
        confidence=CONFIDENCE_STRONG if labelled else CONFIDENCE_WEAK,
        evidence=f"{layout} layout" if layout else f"{images:,} image(s)",
    )
