"""Cybersecurity datasets and the models trained on them.

The shelf the scanner's security detector fills. Its classifiers read the ``dataset_format``
the detector recorded, which is a structural verdict reached from packet captures, Zeek
logs and intrusion-table columns — so this plugin is mostly a labelling layer over evidence
already gathered, and deliberately does not re-derive it from names.

Names still matter for the case the detector cannot reach: a *model* trained on network
traffic looks like any other classifier on disk. There the name is the only signal, and it
is treated as the weak one it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    declared_format,
    is_dataset,
    is_model,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Domain,
    Finding,
    Task,
)

#: The dataset formats the security detector records, mapped to the shelf each belongs on.
FORMAT_CATEGORIES: Mapping[str, tuple[str, str]] = {
    "network_capture": ("network_dataset", "traffic_analysis"),
    "network_flow": ("network_dataset", "traffic_analysis"),
    "intrusion_detection": ("intrusion_dataset", "intrusion_detection"),
    "host_log": ("host_log_dataset", "log_analysis"),
    "malware": ("malware_dataset", "malware_classification"),
    "threat_intel": ("threat_intel_dataset", "threat_intelligence"),
    "ctf": ("ctf_dataset", "security_research"),
}

#: Words that mark a *model* as a security model. Only ever consulted when the asset is a
#: model, and only as a last resort — a folder called ``intrusion-detector`` holding
#: weights is the one case where nothing structural is available.
SECURITY_MODEL_MARKERS = (
    "intrusion", "ids-model", "nids", "hids", "malware", "ransomware", "phishing",
    "anomaly-detection", "network-anomaly", "botnet", "exploit", "vulnerability",
    "cve-", "spam-detect", "fraud-detect", "attack-classifier", "threat-detect",
)

TASKS = (
    Task(id="traffic_analysis", label="Network Traffic Analysis", domain="security", order=10),
    Task(id="intrusion_detection", label="Intrusion Detection", domain="security", order=20),
    Task(id="log_analysis", label="Log & Host Telemetry Analysis", domain="security", order=30),
    Task(id="malware_classification", label="Malware Classification",
         domain="security", order=40),
    Task(id="threat_intelligence", label="Threat Intelligence", domain="security", order=50),
    Task(id="security_research", label="Security Research", domain="security", order=60),
)

CATEGORIES = (
    Category(id="network_dataset", label="Network Traffic Dataset", section="datasets",
             order=280, domain="security",
             aliases=("network-datasets", "pcap", "traffic", "netflow"),
             description="Packet captures or flow records collected from a network."),
    Category(id="intrusion_dataset", label="Intrusion Detection Dataset", section="datasets",
             order=281, domain="security",
             aliases=("intrusion-datasets", "ids", "ids-datasets", "nids"),
             description="Labelled attack and benign traffic, usually as flow tables."),
    Category(id="host_log_dataset", label="Host Log Dataset", section="datasets",
             order=282, domain="security",
             aliases=("host-logs", "event-logs", "evtx", "sysmon"),
             description="Windows event logs, Sysmon output, authentication and audit logs."),
    Category(id="malware_dataset", label="Malware Dataset", section="datasets",
             order=283, domain="security",
             aliases=("malware", "malware-datasets", "samples", "malware-corpus"),
             description="Malware samples, their extracted features, or images of them."),
    Category(id="threat_intel_dataset", label="Threat Intelligence Dataset",
             section="datasets", order=284, domain="security",
             aliases=("threat-intel", "ioc", "iocs", "indicators"),
             description="Indicators of compromise and threat feeds."),
    Category(id="ctf_dataset", label="CTF / Security Research", section="datasets",
             order=285, domain="security",
             aliases=("ctf", "challenges", "writeups"),
             description="Capture-the-flag challenges and the artefacts around them."),
    Category(id="security_model", label="Security Model", section="models",
             order=175, domain="security",
             aliases=("security-models", "ids-model", "malware-model"),
             description="A model that classifies traffic, logs, files or indicators."),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the security domain, its shelves and its rules."""
    registry.add_domain(Domain(id="security", label="Cybersecurity", order=175))
    for task in TASKS:
        registry.add_task(task)
    for category in CATEGORIES:
        registry.add_category(category)

    registry.add_alias(
        "security",
        (
            "network_dataset", "intrusion_dataset", "host_log_dataset",
            "malware_dataset", "threat_intel_dataset", "ctf_dataset", "security_model",
        ),
    )
    registry.add_alias("cyber", ("network_dataset", "intrusion_dataset", "malware_dataset"))

    # Above the ordinary domain rules: an intrusion dataset is a folder of CSVs, and the
    # tabular rule would take it if it saw it first.
    registry.add_classifier(_security_dataset, name="security.dataset", priority=700)
    registry.add_classifier(_security_model, name="security.model", priority=250)
    registry.add_statistic(_security_statistics, name="security")
    registry.add_health_rule(_unlabelled_corpus, name="security.unlabelled")


def _security_dataset(profile: AssetProfile) -> Classification | None:
    """Claim a dataset the scanner recognised as security data."""
    if not is_dataset(profile):
        return None

    mapped = FORMAT_CATEGORIES.get(declared_format(profile))
    if mapped is None:
        return None

    category, task = mapped
    known = _known_dataset_name(profile)
    return Classification(
        category=category,
        task=task,
        domain="security",
        family=known,
        modalities=("text",),
        confidence=CONFIDENCE_CERTAIN if known else CONFIDENCE_STRONG,
        evidence=(
            f"recognised as {known}"
            if known
            else f"scanner recorded format {declared_format(profile)}"
        ),
    )


def _security_model(profile: AssetProfile) -> Classification | None:
    """Claim a model whose name says it works on security data.

    Weak by construction, and it says so. There is nothing in a set of weights that reveals
    it was trained on NetFlow rather than on iris measurements, so a name is all there is —
    and a name is worth reporting at low confidence rather than pretending to certainty.
    """
    if not is_model(profile):
        return None

    marker = profile.matches(SECURITY_MODEL_MARKERS)
    if marker is None:
        return None

    return Classification(
        category="security_model",
        task="intrusion_detection" if "intrusion" in marker or "ids" in marker else None,
        domain="security",
        confidence=CONFIDENCE_WEAK,
        evidence=f"name contains {marker!r}",
    )


def _security_statistics(profile: AssetProfile) -> Mapping[str, Any]:
    """Return what a security corpus contains, counted from the recorded file list."""
    if not is_dataset(profile) or declared_format(profile) not in FORMAT_CATEGORIES:
        return {}
    if not profile.files.loaded:
        return {}

    stats: dict[str, Any] = {}
    for key, extensions in (
        ("captures", (".pcap", ".pcapng", ".cap", ".erf")),
        ("flow_files", (".binetflow", ".netflow", ".argus", ".ipfix")),
        ("event_logs", (".evtx", ".etl")),
        ("tables", (".csv", ".tsv", ".parquet")),
        ("logs", (".log", ".jsonl")),
    ):
        count = profile.files.count(*extensions)
        if count:
            stats[key] = count

    known = _known_dataset_name(profile)
    if known:
        stats["public_dataset"] = known

    return stats


def _unlabelled_corpus(profile: AssetProfile) -> Sequence[Finding]:
    """Report a capture corpus with no labels beside it.

    Raw traffic with no ground truth trains nothing supervised. It is a perfectly ordinary
    thing to keep — for baselining, for replay — so this is a note about what it is missing
    rather than a claim that it is broken.
    """
    if not is_dataset(profile) or not profile.files.loaded:
        return ()
    if declared_format(profile) not in {"network_capture", "network_flow"}:
        return ()

    labelled = profile.files.has_stem("label", "ground_truth", "groundtruth", "annotation")
    if labelled or profile.files.count(".csv", ".tsv"):
        return ()

    return (
        Finding(
            code="security.no_labels",
            severity=Severity.INFO,
            message="Raw captures with no label file beside them",
            fix_hint="Fine for replay or baselining; supervised training needs ground truth.",
        ),
    )


def _known_dataset_name(profile: AssetProfile) -> str | None:
    """Return the public dataset the scanner recognised, if it recognised one."""
    if profile.dataset is None:
        return None
    value = profile.dataset.extra.get("known_dataset")
    return str(value) if value else None
