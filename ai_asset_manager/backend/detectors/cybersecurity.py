r"""Cybersecurity dataset detector.

Security data was the largest blind spot a whole-machine scan turned up. The reason is
structural rather than accidental: a security corpus has no ``config.json``, no
``data.yaml`` and no ``images/`` tree. It is a pile of CSVs, or a pile of logs, or a pile
of packet captures — and to every generic dataset rule in this package, a pile of CSVs is
either a tabular dataset or nothing at all.

What identifies it is *which* pile, and the discipline this module applies is that no
single observation is ever enough:

    One CSV is a spreadsheet. One log is a log. One JSON file is a JSON file.

So detection is by accumulated evidence. Each :class:`_Signal` is one independent
observation with a weight; a directory becomes a dataset when the total clears
:data:`MIN_SCORE` *and* at least :data:`MIN_SIGNALS` distinct observations agree. A folder
containing a single ``conn.log`` scores 2 and is declined; the same folder with nine other
Zeek logs beside it scores 5 and is not.

Two further guards keep it from over-reaching:

*Shelves defer to their contents.* A ``cybersecurity-datasets/`` folder holding CICIDS2017
and UNSW-NB15 is two datasets, not one. When the directory has no security files of its
own and two or more children each qualify on their own evidence, this detector declines and
lets them match individually.

*Application state is never a dataset.* The same rule the boundary guard applies: a
firewall's own log directory is a program writing to disk, not a corpus somebody collected.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from ai_asset_manager.backend.detectors.base import (
    PRIORITY_DATASET_SPECIFIC,
    BaseDetector,
    DetectionResult,
)
from ai_asset_manager.backend.detectors.boundary import (
    is_application_state,
    is_container_name,
    is_drive_root,
    looks_like_dataset_root,
)
from ai_asset_manager.backend.models.enums import AssetKind, DatasetFormat
from ai_asset_manager.backend.scanner.context import DirectoryContext
from ai_asset_manager.backend.scanner.types import FileEntry
from ai_asset_manager.logging_conf import get_logger

logger = get_logger(__name__)

#: Evidence needed before a directory is called a security dataset. Tuned so that a single
#: strong-and-plural observation (three packet captures, five Zeek logs) clears it, while
#: any single file of any kind does not.
MIN_SCORE = 4.0

#: Distinct observations needed alongside the score. A known dataset name plus one data
#: file is two; a lone ``.pcap`` is one, and stays uncatalogued.
MIN_SIGNALS = 2

#: The weight above which one observation stands on its own.
#:
#: Only ever awarded to evidence that is already plural or already conjunctive — three
#: packet captures, two Zeek logs, a table whose header carries UNSW-NB15's columns, a
#: sample corpus agreeing on two independent counts. So "one file is never enough" survives
#: intact: no single file earns this, by construction. What it rescues is the corpus that
#: *is* one unmistakable thing and has nothing else to say about itself, which is what a
#: folder of nine Zeek logs looks like.
STRONG_SIGNAL_WEIGHT = 4.0

#: Bytes read from the front of a CSV to look at its column names. One read, one buffer,
#: never the body: an intrusion dataset's CSV is routinely a gigabyte and its header is
#: three hundred bytes.
HEADER_BYTES = 8192

#: How many CSVs are sniffed per directory. The header of the first few settles the
#: question; opening two thousand of them would not.
MAX_HEADER_SNIFFS = 3


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

#: Packet captures. The strongest single marker in the whole module: nothing else on a
#: normal machine writes a ``.pcap``.
#:
#: ``.dmp`` is deliberately absent. It was here, and on the development machine it matched
#: VMware's ``vmware-vmx.dmp`` and Windows' own crash dumps — 190 GB of memory dumps
#: reported as three packet-capture corpora. A ``.dmp`` is a memory dump far more often
#: than it is anything else, and an extension that ambiguous cannot carry a strong signal.
CAPTURE_EXTENSIONS: tuple[str, ...] = (".pcap", ".pcapng", ".cap", ".erf", ".snoop")

#: Aggregated flow records. The payload is gone; what is left is who talked to whom.
FLOW_EXTENSIONS: tuple[str, ...] = (
    ".binetflow", ".netflow", ".flow", ".argus", ".ipfix", ".nfcapd", ".sflow",
)

#: Windows host telemetry.
EVENT_LOG_EXTENSIONS: tuple[str, ...] = (".evtx", ".etl")

#: Zeek (formerly Bro) writes one log per protocol, with these exact names. Two or more of
#: them together is a Zeek run and cannot be anything else.
ZEEK_LOG_NAMES: frozenset[str] = frozenset(
    {
        "conn.log", "dns.log", "http.log", "ssl.log", "x509.log", "files.log",
        "weird.log", "notice.log", "smtp.log", "ssh.log", "ftp.log", "dhcp.log",
        "ntp.log", "smb_files.log", "smb_mapping.log", "kerberos.log", "rdp.log",
        "software.log", "tunnel.log", "capture_loss.log", "stats.log", "known_hosts.log",
        "known_services.log", "signatures.log", "intel.log", "traceroute.log",
    }
)

#: Suricata's output and configuration.
SURICATA_NAMES: frozenset[str] = frozenset(
    {"eve.json", "fast.log", "suricata.yaml", "stats.log", "http.log", "tls.log"}
)

#: Snort's.
SNORT_NAMES: frozenset[str] = frozenset({"alert.fast", "alert.full", "snort.conf", "unified2"})

#: Sysmon and Windows security channels. Matched against a whole *basename*, not as a
#: substring: ``lastlog`` and ``wtmp`` are also the names of man pages, and matching them
#: loosely turned ``msys64/usr/share/man`` into a host-telemetry corpus.
HOST_LOG_NAMES: frozenset[str] = frozenset(
    {
        "security.evtx", "system.evtx", "application.evtx",
        "microsoft-windows-sysmon%4operational.evtx",
        "microsoft-windows-powershell%4operational.evtx",
        "windows powershell.evtx", "sysmon.evtx", "sysmonconfig.xml",
        "auth.log", "authpriv.log", "secure.log", "audit.log", "sudo.log",
        "wtmp", "btmp", "utmp", "lastlog",
    }
)

#: Threat-intelligence and indicator formats. Matched as whole words within a filename
#: rather than as substrings — ``ioc`` and ``otx`` are three letters that turn up inside
#: perfectly ordinary filenames, and as substrings they found 458 "indicator files" in a
#: flashcard application's cache.
THREAT_INTEL_TOKENS: frozenset[str] = frozenset(
    {
        "ioc", "iocs", "indicator", "indicators", "misp", "stix", "taxii", "openioc",
        "yara", "threatfox", "alienvault", "otx", "malwarebazaar", "urlhaus", "cti",
    }
)

#: Directory names that mean malware and cannot mean anything else.
#:
#: What is *not* here matters more than what is. ``samples``, ``binaries``, ``packed``,
#: ``pe``, ``elf``, ``dlls``, ``apks``, ``benign`` and ``goodware`` were all here once, and
#: between them they claimed Python's ``DLLs`` directory, Ghidra's and Cutter's processor
#: modules, dnSpy, three IDE plugin folders and an NLP project. They are ordinary words in
#: ordinary software. Only names that are *about* malware survive.
MALWARE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "malware", "malwares", "malicious", "malware_samples", "malwaresamples",
        "trojan", "trojans", "ransomware", "backdoor", "spyware", "worm", "adware",
        "rootkit", "botnet", "virusshare", "virussign", "malwarebazaar", "thezoo",
        "malimg", "malmem", "vxheaven", "vx-underground", "theZoo",
    }
)

#: A ``benign``/``malicious`` pair is meaningful where neither name is on its own: nobody
#: labels two folders that way except to separate a corpus into classes.
MALWARE_CLASS_PAIRS: tuple[tuple[str, str], ...] = (
    ("benign", "malicious"),
    ("benign", "malware"),
    ("goodware", "malware"),
    ("clean", "infected"),
)

#: Executable payload extensions. Supporting evidence only, and never a reason on its own:
#: every application on a Windows machine ships hundreds of these.
SAMPLE_EXTENSIONS: tuple[str, ...] = (".exe", ".dll", ".apk", ".elf", ".vir", ".sys")

#: Digest-named files needed before the naming pattern counts, and the share of the
#: directory they must make up. A cache that happens to hash a few filenames is not a
#: sample corpus; a directory where most files are named after their own digest is.
MIN_DIGEST_NAMED = 20
MIN_DIGEST_SHARE = 0.4

#: Capture-the-flag material.
CTF_FRAGMENTS: tuple[str, ...] = (
    "ctf", "capture-the-flag", "picoctf", "hackthebox", "tryhackme", "htb-",
    "writeup", "writeups", "pwnable", "reversing", "forensics-challenge",
)

#: A file named after its own digest. How sample corpora are laid out, and nothing else.
_DIGEST_NAME = re.compile(r"^[0-9a-f]{32}(?:[0-9a-f]{32})?(?:\.[a-z0-9]{1,8})?$")

#: Column names that only appear in an intrusion-detection table. Grouped by the corpus
#: family that uses them, so a match names the family rather than merely asserting one.
IDS_COLUMN_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("UNSW-NB15", ("attack_cat", "sttl", "dttl", "sload", "dload", "ct_srv_src")),
    ("CIC flow", ("flow duration", "fwd packet length max", "bwd packet length max",
                  "flow bytes/s", "flow iat mean", "total fwd packets")),
    ("KDD", ("protocol_type", "src_bytes", "dst_bytes", "num_failed_logins",
             "srv_serror_rate", "dst_host_srv_count")),
    ("CTU/Argus", ("srcaddr", "dstaddr", "sport", "dport", "totbytes", "srcbytes")),
    ("Zeek/TSV", ("ts", "uid", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p")),
    ("generic flow", ("src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label")),
    ("generic flow", ("srcip", "dstip", "sport", "dsport", "proto", "attack")),
)

#: Public datasets recognised by name, mapped to the format they are. Matched against the
#: directory name and its ancestors, normalised so ``CSE-CIC-IDS2018``, ``cse_cic_ids2018``
#: and ``csecicids2018`` are the same string.
#:
#: This table is a convenience, never a requirement: the signal rules above recognise a
#: corpus that is not on it, and a name on it alone is not enough to detect anything.
KNOWN_DATASETS: tuple[tuple[str, str, DatasetFormat], ...] = (
    ("unswnb15", "UNSW-NB15", DatasetFormat.INTRUSION_DETECTION),
    ("unswnb", "UNSW-NB15", DatasetFormat.INTRUSION_DETECTION),
    ("cicids2017", "CICIDS2017", DatasetFormat.INTRUSION_DETECTION),
    ("csecicids2018", "CSE-CIC-IDS2018", DatasetFormat.INTRUSION_DETECTION),
    ("cicids2018", "CSE-CIC-IDS2018", DatasetFormat.INTRUSION_DETECTION),
    ("cicddos2019", "CIC-DDoS2019", DatasetFormat.INTRUSION_DETECTION),
    ("ciciot2023", "CIC-IoT2023", DatasetFormat.INTRUSION_DETECTION),
    ("cicandmal", "CIC-AndMal", DatasetFormat.MALWARE),
    ("nslkdd", "NSL-KDD", DatasetFormat.INTRUSION_DETECTION),
    ("kddcup99", "KDD Cup 99", DatasetFormat.INTRUSION_DETECTION),
    ("kddcup", "KDD Cup 99", DatasetFormat.INTRUSION_DETECTION),
    ("kdd99", "KDD Cup 99", DatasetFormat.INTRUSION_DETECTION),
    ("ctu13", "CTU-13", DatasetFormat.NETWORK_FLOW),
    ("ctumalware", "CTU Malware Capture", DatasetFormat.NETWORK_CAPTURE),
    ("darpa", "DARPA", DatasetFormat.INTRUSION_DETECTION),
    ("mawi", "MAWI", DatasetFormat.NETWORK_CAPTURE),
    ("caida", "CAIDA", DatasetFormat.NETWORK_CAPTURE),
    ("toniot", "TON_IoT", DatasetFormat.INTRUSION_DETECTION),
    ("tonio", "TON_IoT", DatasetFormat.INTRUSION_DETECTION),
    ("botiot", "Bot-IoT", DatasetFormat.INTRUSION_DETECTION),
    ("iot23", "IoT-23", DatasetFormat.NETWORK_FLOW),
    ("ustctfc", "USTC-TFC2016", DatasetFormat.NETWORK_CAPTURE),
    ("iscx", "ISCX", DatasetFormat.INTRUSION_DETECTION),
    ("malmem", "CIC-MalMem2022", DatasetFormat.MALWARE),
    ("ember", "EMBER", DatasetFormat.MALWARE),
    ("virusshare", "VirusShare", DatasetFormat.MALWARE),
    ("virustotal", "VirusTotal", DatasetFormat.MALWARE),
    ("malimg", "Malimg", DatasetFormat.MALWARE),
    ("bodmas", "BODMAS", DatasetFormat.MALWARE),
    ("sorel20m", "SOREL-20M", DatasetFormat.MALWARE),
    ("drebin", "Drebin", DatasetFormat.MALWARE),
    ("theZoo", "theZoo", DatasetFormat.MALWARE),
    ("adfald", "ADFA-LD", DatasetFormat.HOST_LOG),
    ("adfa", "ADFA", DatasetFormat.HOST_LOG),
    ("ottrf", "OpTC", DatasetFormat.HOST_LOG),
    ("dapt2020", "DAPT-2020", DatasetFormat.INTRUSION_DETECTION),
    ("lanl", "LANL Auth", DatasetFormat.HOST_LOG),
    ("mordor", "Mordor/Security Datasets", DatasetFormat.HOST_LOG),
    ("evtxattacksamples", "EVTX-ATTACK-SAMPLES", DatasetFormat.HOST_LOG),
    ("attackdatasets", "Security Datasets", DatasetFormat.HOST_LOG),
    ("unibs", "UNIBS", DatasetFormat.NETWORK_CAPTURE),
    ("isot", "ISOT", DatasetFormat.NETWORK_FLOW),
    ("hikari", "HIKARI-2021", DatasetFormat.INTRUSION_DETECTION),
    ("edgeiiot", "Edge-IIoTset", DatasetFormat.INTRUSION_DETECTION),
    ("wustl", "WUSTL-IIoT", DatasetFormat.INTRUSION_DETECTION),
    ("swat", "SWaT", DatasetFormat.INTRUSION_DETECTION),
    ("gaspipeline", "Gas Pipeline (MSU)", DatasetFormat.INTRUSION_DETECTION),
    ("phishtank", "PhishTank", DatasetFormat.THREAT_INTEL),
    ("phiusiil", "PhiUSIIL", DatasetFormat.THREAT_INTEL),
)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Signal:
    """One independent observation supporting a security-dataset verdict."""

    #: Weight towards :data:`MIN_SCORE`. Strong markers score 3-4 on their own; weak ones
    #: contribute only in company.
    weight: float
    #: How the observation reads in a report: "18 packet capture file(s)".
    description: str
    #: The format this observation argues for, when it argues for one.
    dataset_format: DatasetFormat | None = None


@dataclass(slots=True)
class _Evidence:
    """Everything observed about one directory."""

    signals: list[_Signal] = field(default_factory=list)
    #: Public dataset recognised by name, when one was.
    known_dataset: str | None = None
    #: True when the *directory* is named for the corpus, rather than one file inside it.
    #:
    #: The distinction decides what the asset is called. A folder called ``CICIDS2017`` is
    #: that dataset. A folder called ``Datasets`` holding ``UNSW-NB15_1.csv`` beside 71 GB
    #: of LANL logs, 23 GB of Suricata output and the CSIC HTTP corpus *contains*
    #: UNSW-NB15 among others, and naming the whole 90 GB after it would be a lie about
    #: 89 GB of it.
    named_by_directory: bool = False

    @property
    def score(self) -> float:
        """Return the summed weight of every observation."""
        return sum(signal.weight for signal in self.signals)

    @property
    def qualifies(self) -> bool:
        """Report whether the evidence is enough to call this a security dataset.

        Two ways through, and both require the score. Either several observations agree,
        or one of them is strong enough to stand alone — see :data:`STRONG_SIGNAL_WEIGHT`
        for why that is not a hole in the one-file rule.
        """
        if self.score < MIN_SCORE:
            return False
        if len(self.signals) >= MIN_SIGNALS:
            return True
        return any(signal.weight >= STRONG_SIGNAL_WEIGHT for signal in self.signals)

    def resolve_format(self) -> DatasetFormat:
        """Return the format the heaviest observation argues for.

        Ties break towards whichever came first, which is the order the rules run in —
        captures before flows before tables, most specific first.
        """
        best: tuple[float, DatasetFormat] | None = None
        for signal in self.signals:
            if signal.dataset_format is None:
                continue
            if best is None or signal.weight > best[0]:
                best = (signal.weight, signal.dataset_format)
        return best[1] if best else DatasetFormat.CUSTOM

    def descriptions(self) -> list[str]:
        """Return every observation, heaviest first."""
        ordered = sorted(self.signals, key=lambda signal: -signal.weight)
        return [signal.description for signal in ordered]


class CyberSecurityDatasetDetector(BaseDetector):
    """Detects network, host-log, malware and threat-intelligence datasets."""

    name = "cybersecurity_dataset"
    priority = PRIORITY_DATASET_SPECIFIC

    def detect(self, ctx: DirectoryContext) -> list[DetectionResult]:
        """Emit one dataset when the accumulated evidence clears the thresholds."""
        if is_drive_root(ctx.path) or is_application_state(ctx.path):
            return []

        evidence = gather_evidence(ctx)
        if not evidence.qualifies:
            return []

        if is_security_container(ctx) and not evidence.known_dataset:
            # A folder called `datasets`, `downloads` or `data` is named for what it holds.
            # It may still *be* one when it holds captures directly or when the path names a
            # corpus — `UNSW-NB15/csv/training` is a split, whatever `training` means
            # elsewhere — but a stray CSV in Downloads must not turn the folder into a
            # dataset and swallow everything else in it.
            logger.debug(
                "Not claiming %s: a container, with no security files of its own", ctx.path
            )
            return []

        if self._defers_to_children(ctx):
            logger.debug(
                "Deferring %s to its children: the evidence is theirs, not its", ctx.path
            )
            return []

        dataset_format = evidence.resolve_format()
        name = (
            evidence.known_dataset
            if evidence.known_dataset and evidence.named_by_directory
            else ctx.name
        )

        return [
            self._result(
                ctx,
                kind=AssetKind.DATASET,
                name=name,
                subkind=dataset_format.value,
                confidence=min(0.99, 0.55 + 0.05 * evidence.score),
                evidence={
                    "security_dataset": True,
                    "dataset_format": dataset_format.value,
                    "known_dataset": evidence.known_dataset,
                    "evidence_score": round(evidence.score, 1),
                    "signals": evidence.descriptions(),
                },
            )
        ]

    def _defers_to_children(self, ctx: DirectoryContext) -> bool:
        """Report whether a child of this directory is the real dataset.

        The security detector sits in the specific band and so is exempt from the boundary
        guard, which means nothing else stops it claiming a directory far above the corpus
        — and because a claim suppresses everything beneath it, claiming too high does not
        add a wrong row, it *deletes* every right one. Pointed at a folder holding a
        library, a captures corpus and an intrusion dataset, the subtree evidence made the
        folder itself look like one packet-capture dataset and hid all three.

        So the rule is that the *smallest* qualifying directory wins. A directory with
        security files of its own is anchored and keeps its claim; one whose evidence lives
        entirely in a subdirectory hands the claim down.

        Only asked of a directory that already qualified, so the extra subtree walks are
        paid for by the handful that reach this point rather than by every directory on the
        drive.
        """
        if _is_anchored(ctx):
            return False
        if looks_like_dataset_root(ctx):
            # `train/` and `test/` beside each other are one dataset's splits, and a
            # manifest at the root is its author saying so. Without this, every split
            # corpus would be reported as two datasets called "train" and "test".
            return False

        return any(gather_evidence(child).qualifies for child in ctx.children())


def gather_evidence(ctx: DirectoryContext) -> _Evidence:
    """Collect every security-dataset observation about a directory.

    Exposed rather than private because the shelf test asks it about children, and because
    it is the natural seam for testing the rules without constructing a detection.
    """
    evidence = _Evidence()

    # Built once and handed to every rule. Six rules each calling `_nearby_files` meant six
    # passes over the same files at every directory on the drive.
    nearby = _nearby_files(ctx)

    # The name runs first because the malware rule consults it: a path naming EMBER or
    # VirusShare is one of the three unambiguous things that rule will accept.
    _name_signals(ctx, evidence)
    _network_signals(ctx, evidence, nearby)
    _host_log_signals(ctx, evidence, nearby)
    _threat_intel_signals(ctx, evidence, nearby)
    _malware_signals(ctx, evidence, nearby)
    _ctf_signals(ctx, evidence, nearby)
    _table_signals(ctx, evidence, nearby)

    return evidence


def _network_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe packet captures, flow records and IDS output."""
    captures = ctx.count_extension(*CAPTURE_EXTENSIONS)
    if captures:
        evidence.signals.append(
            _Signal(
                weight=4.0 if captures >= 3 else 3.0,
                description=f"{captures} packet capture file(s)",
                dataset_format=DatasetFormat.NETWORK_CAPTURE,
            )
        )

    flows = ctx.count_extension(*FLOW_EXTENSIONS)
    if flows:
        evidence.signals.append(
            _Signal(
                weight=4.0 if flows >= 3 else 3.0,
                description=f"{flows} flow record file(s)",
                dataset_format=DatasetFormat.NETWORK_FLOW,
            )
        )

    names = {entry.name.lower() for entry in nearby}

    zeek = sorted(names & ZEEK_LOG_NAMES)
    if len(zeek) >= 2:
        evidence.signals.append(
            _Signal(
                weight=4.0,
                description=f"Zeek logs ({', '.join(zeek[:4])})",
                dataset_format=DatasetFormat.NETWORK_CAPTURE,
            )
        )
    elif zeek:
        evidence.signals.append(
            _Signal(weight=1.5, description=f"Zeek log ({zeek[0]})")
        )

    suricata = sorted(names & SURICATA_NAMES)
    if "eve.json" in suricata or "suricata.yaml" in suricata:
        evidence.signals.append(
            _Signal(
                weight=3.0,
                description=f"Suricata output ({', '.join(suricata[:3])})",
                dataset_format=DatasetFormat.NETWORK_CAPTURE,
            )
        )

    snort = sorted(names & SNORT_NAMES)
    if snort:
        evidence.signals.append(
            _Signal(
                weight=2.5,
                description=f"Snort output ({', '.join(snort[:3])})",
                dataset_format=DatasetFormat.NETWORK_CAPTURE,
            )
        )


def _host_log_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe Windows event logs, Sysmon output and authentication logs."""
    event_logs = ctx.count_extension(*EVENT_LOG_EXTENSIONS)
    if event_logs:
        evidence.signals.append(
            _Signal(
                weight=4.0 if event_logs >= 3 else 3.0,
                description=f"{event_logs} Windows event log file(s)",
                dataset_format=DatasetFormat.HOST_LOG,
            )
        )

    # Counted only where they are gathered: in this directory, or one level below it. Four
    # authentication logs scattered across a 14 GB application tree are four applications
    # each writing its own log, not somebody's corpus — and reading the whole subtree could
    # not tell the difference.
    matched = sum(1 for entry in nearby if entry.name.lower() in HOST_LOG_NAMES)
    if matched >= 3:
        evidence.signals.append(
            _Signal(
                weight=2.5,
                description=f"{matched} host telemetry file(s) (Sysmon/auth/audit)",
                dataset_format=DatasetFormat.HOST_LOG,
            )
        )


def _malware_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe a sample corpus.

    The rule this module got most wrong, and the correction is worth stating plainly.

    Counting ordinary things — a ``samples/`` folder, ten executables, a few hash-named
    cache entries — found "malware corpora" in Python's ``DLLs`` directory, in Ghidra, in
    Zoom, in a video editor and, worst of all, in a 14 GB NLP project, where the claim
    suppressed twenty-two training runs and eleven checkpoints beneath it. A wrong claim
    high in a tree does not add a wrong row; it deletes every right one.

    So evidence is now split. At least one **unambiguous** observation is required — a
    directory that is *about* malware, a corpus where most files are named after their own
    digest, or a recognised public malware dataset — and at least one more of anything.
    Executables and hash manifests can support that conclusion; they can no longer reach it.
    """
    # The directory itself or one of its immediate children — not anywhere in the subtree.
    # `F:\New Apps` is a folder of applications, one of which keeps YARA rules under
    # `Yara Roles/rules/malware`; counting that marker three levels down reported 25 GB of
    # installed software as a malware corpus. A corpus's own root is next to its samples.
    marker_dirs = sorted(
        name
        for name in {ctx.name.lower(), *ctx.lower_child_dir_names}
        if name in MALWARE_DIR_NAMES
    )

    # Where the samples would be. Inside the malware-named directory when there is one —
    # `theZoo/malware/Binaries/` puts them two levels down and a fixed depth would miss
    # them — and otherwise near this directory. Never the whole subtree: that scan, run at
    # every directory on the drive, was the detector's largest cost.
    scope = _sample_scope(ctx, marker_dirs, nearby)
    scope_names = [entry.name.lower() for entry in scope]
    digest_named = sum(1 for name in scope_names if _DIGEST_NAME.match(name))
    digest_share = digest_named / len(scope_names) if scope_names else 0.0
    digest_corpus = digest_named >= MIN_DIGEST_NAMED and digest_share >= MIN_DIGEST_SHARE
    nearby_names = scope_names

    known_malware = (
        evidence.known_dataset is not None
        and evidence.signals
        and any(
            signal.dataset_format is DatasetFormat.MALWARE for signal in evidence.signals
        )
    )

    strong = [
        description
        for description, holds in (
            (f"{'/, '.join(marker_dirs[:3])}/ named for malware", bool(marker_dirs)),
            (
                f"{digest_named} of {len(nearby_names)} files named after their digest",
                digest_corpus,
            ),
            (f"recognised corpus {evidence.known_dataset}", bool(known_malware)),
        )
        if holds
    ]
    if not strong:
        return

    directories = _subtree_dir_names(ctx)
    class_split = next(
        (
            f"{left}/ beside {right}/"
            for left, right in MALWARE_CLASS_PAIRS
            if left in directories and right in directories
        ),
        None,
    )
    samples = ctx.count_extension(*SAMPLE_EXTENSIONS)
    manifest = any(
        name in {"sha256sums.txt", "md5sums.txt", "hashes.txt", "sha256.txt"}
        for name in nearby_names
    )

    supporting = [
        description
        for description, holds in (
            (str(class_split), class_split is not None),
            (f"{samples} executable sample(s)", samples >= 50),
            ("hash manifest", manifest),
        )
        if holds
    ]

    agreeing = len(strong) + len(supporting)
    if agreeing < 2:
        return

    if not (digest_named >= 5 or samples >= 10 or class_split):
        # A corpus with no samples in it is not a corpus. SecLists keeps a `Malware/`
        # folder of malware-themed *passwords* beside a `hashes.txt` of password hashes,
        # and on the evidence above that reads as a sample collection. It is a wordlist,
        # and the thing that says so is the absence of anything to analyse.
        logger.debug("Malware layout at %s has no samples in it; declining", ctx.path)
        return

    evidence.signals.append(
        _Signal(
            weight=2.5 + 1.0 * agreeing,
            description="malware corpus layout: " + "; ".join([*strong, *supporting]),
            dataset_format=DatasetFormat.MALWARE,
        )
    )


def _threat_intel_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe indicator feeds and threat-intelligence exports."""
    matched = sum(
        1
        for entry in nearby
        if THREAT_INTEL_TOKENS & set(re.split(r"[^a-z0-9]+", entry.name.lower()))
    )
    if matched < 2:
        return

    evidence.signals.append(
        _Signal(
            weight=3.0 if matched >= 5 else 2.0,
            description=f"{matched} indicator/threat-intel file(s)",
            dataset_format=DatasetFormat.THREAT_INTEL,
        )
    )


def _ctf_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe capture-the-flag challenge material."""
    # The path test is a string comparison and the file count is a scan, so the cheap one
    # decides first. This runs at every directory on the drive.
    if not any(
        fragment in ctx.path.replace("\\", "/").lower() for fragment in CTF_FRAGMENTS
    ):
        return

    flags = sum(
        1
        for entry in nearby
        if entry.name.lower() in {"flag.txt", "flag", "flag.png", "solve.py", "exploit.py"}
    )
    if not flags:
        return

    evidence.signals.append(
        _Signal(
            weight=3.0,
            description=f"CTF layout ({flags} challenge artefact(s))",
            dataset_format=DatasetFormat.CTF,
        )
    )


def _table_signals(
    ctx: DirectoryContext, evidence: _Evidence, nearby: list[FileEntry]
) -> None:
    """Observe intrusion-detection tables by their column names.

    The only rule here that opens a file, and it reads :data:`HEADER_BYTES` from the front
    of at most :data:`MAX_HEADER_SNIFFS` of them. That is what separates ``UNSW_NB15.csv``
    — 700 MB of labelled flows — from an exported spreadsheet, and no amount of looking at
    the directory structure can do it.

    Like every other rule here it looks only where the evidence would be gathered: a
    corpus keeps its tables at its root or one level below, and reading the whole subtree
    at every directory on the drive was the single largest cost in the detector.
    """
    tables = [
        entry
        for entry in nearby
        if entry.extension in {".csv", ".tsv", ".txt"} and entry.size > 4096
    ]
    if not tables:
        return

    # Largest first, but any table whose *name* points at a public corpus is sniffed
    # ahead of them. A folder holding 71 GB of authentication logs beside a 165 MB
    # `UNSW-NB15_1.csv` would otherwise spend all three reads on the logs.
    tables.sort(
        key=lambda entry: (_match_known_name(entry.name)[0] is None, -entry.size)
    )
    for entry in tables[:MAX_HEADER_SNIFFS]:
        family = _sniff_ids_columns(entry.path)
        if family is None:
            continue
        evidence.signals.append(
            _Signal(
                weight=4.0,
                description=f"{entry.name} carries {family} intrusion-detection columns",
                dataset_format=DatasetFormat.INTRUSION_DETECTION,
            )
        )
        if len(tables) > 1:
            evidence.signals.append(
                _Signal(
                    weight=1.0,
                    description=f"{len(tables)} table file(s) in the same layout",
                )
            )
        return


def _name_signals(ctx: DirectoryContext, evidence: _Evidence) -> None:
    """Observe a recognised public dataset name in the path or in a filename.

    Never sufficient on its own — a folder called ``CICIDS2017`` with nothing in it is an
    empty folder — but it names the corpus, which is what the inventory wants to show, and
    it supplies the second observation a single strong marker needs.

    Filenames are searched as well as directories, and that is not a refinement. The
    development machine keeps its security corpora in one flat folder called ``Datasets``:
    ``UNSW-NB15_1.csv`` beside ``eve.json`` beside 71 GB of LANL authentication logs.
    Nothing in the *path* names a corpus, and the raw UNSW partitions ship with no header
    row, so column sniffing cannot reach them either. The filename is the only thing that
    knows, and looking only at directories missed the whole directory.

    Only the directory's *immediate* files are read, which keeps this proportional to the
    directory rather than to its subtree.
    """
    label, dataset_format, where = _match_known_dataset(ctx.path)
    from_directory = label is not None

    if label is None:
        for entry in ctx.files:
            label, dataset_format, where = _match_known_name(entry.name)
            if label is not None:
                break

    if label is None:
        return

    evidence.known_dataset = label
    evidence.named_by_directory = from_directory
    evidence.signals.append(
        _Signal(
            weight=2.5,
            description=f"recognised public dataset {label} (from {where})",
            dataset_format=dataset_format,
        )
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalise_token(text: str) -> str:
    """Reduce a path segment to letters and digits, lower-cased.

    ``CSE-CIC-IDS2018``, ``cse_cic_ids2018`` and ``CSE CIC IDS 2018`` all become the same
    string, which is what makes one table entry match every spelling in the wild.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


#: The dataset table, normalised once at import. Rebuilding it per candidate string was
#: measurable at whole-machine scale: this is consulted for every directory reached.
_NORMALISED_DATASETS: tuple[tuple[str, str, DatasetFormat], ...] = tuple(
    (_normalise_token(needle), label, dataset_format)
    for needle, label, dataset_format in KNOWN_DATASETS
)


@lru_cache(maxsize=1 << 16)
def _match_known_name(name: str) -> tuple[str | None, DatasetFormat | None, str]:
    """Return the public dataset a single directory or file name refers to, if any.

    Two kinds of test, and the split is what keeps this both correct and affordable.

    *Long entries* are matched as substrings of the name with its punctuation removed.
    Nothing but the corpus contains ``unswnb15``, and the squashing is what lets one table
    entry match ``UNSW-NB15``, ``unsw_nb15`` and ``UNSW NB 15``.

    *Short entries* have to sit on a word boundary — a whole word of the name, or a whole
    word followed by digits, which is how ``DARPA1998`` and ``KDD99`` are spelled. Matched
    as substrings they were a menace: ``isot`` inside ``PKCS7_type_is_other.3ossl.gz`` and
    ``ember`` inside ``remember.c`` reported an OpenSSL manual page and a Dart icon as
    security datasets.

    The name is normalised and tokenised **once**, not once per table entry. Doing it per
    entry meant fifty regular-expression passes over every filename on the drive, and it
    was the single largest cost in the whole detector.
    """
    squashed = _normalise_token(name)
    if not squashed:
        return None, None, ""

    tokens: list[str] | None = None
    for needle, label, dataset_format in _NORMALISED_DATASETS:
        if len(needle) >= 6:
            if needle in squashed:
                return label, dataset_format, name
            continue

        if tokens is None:
            tokens = [part for part in re.split(r"[^a-z0-9]+", name.lower()) if part]
        for token in tokens:
            if token == needle or (
                token.startswith(needle) and token[len(needle) :].isdigit()
            ):
                return label, dataset_format, name

    return None, None, ""


def _match_known_dataset(
    path: str,
) -> tuple[str | None, DatasetFormat | None, str]:
    """Return the public dataset a path names, if any.

    The leaf is checked first and then its ancestors, so ``UNSW-NB15/csv/training`` is
    still recognised as UNSW-NB15 when the detector is looking at the ``training`` folder.
    """
    segments = [part for part in path.replace("\\", "/").split("/") if part]
    for segment in reversed(segments):
        label, dataset_format, where = _match_known_name(segment)
        if label is not None:
            return label, dataset_format, where
    return None, None, ""


def _sample_scope(
    ctx: DirectoryContext, marker_dirs: list[str], nearby: list[FileEntry]
) -> list[FileEntry]:
    """Return the files a malware corpus's samples would be among.

    A marker directory bounds the search to the corpus itself, wherever inside it the
    binaries sit. Without a marker there is nothing to bound it with, so the search stays
    near this directory rather than expanding to its whole subtree — which is the
    difference between a scan proportional to a corpus and one proportional to a drive.
    """
    if not marker_dirs:
        return nearby

    if ctx.name.lower() in marker_dirs:
        return ctx.subtree_files

    collected: list[FileEntry] = list(ctx.files)
    for child in ctx.children():
        if child.name.lower() in marker_dirs:
            collected.extend(child.subtree_files)
        else:
            collected.extend(child.files)
    return collected


def _nearby_files(ctx: DirectoryContext) -> list[FileEntry]:
    """Return the directory's own files and those of its immediate children.

    The scope for evidence that only means something when it is *concentrated*. A corpus
    keeps its logs together; a profile directory has one of everything spread across a
    hundred applications, and a subtree count cannot tell those apart.
    """
    nearby = list(ctx.files)
    for child in ctx.children():
        nearby.extend(child.files)
    return nearby


def _subtree_dir_names(ctx: DirectoryContext) -> set[str]:
    """Return every directory name appearing beneath this directory, lower-cased."""
    root = ctx.path
    names: set[str] = set()
    for candidate in ctx.tree.subtree_paths(root):
        if candidate != root:
            names.add(os.path.basename(candidate).lower())
    return names


#: Files an author writes at the root of a corpus to say "this is the root". A hash
#: manifest is the malware-corpus equivalent of a ``data.yaml``.
_ROOT_MANIFESTS: frozenset[str] = frozenset(
    {
        "sha256sums.txt", "md5sums.txt", "hashes.txt", "sha256.txt",
        "labels.csv", "index.csv", "metadata.csv",
    }
)


def _is_anchored(ctx: DirectoryContext) -> bool:
    """Report whether the corpus starts at this directory rather than below it.

    Two ways to be anchored, and both mean "the evidence is mine, not my children's":
    security data sitting directly in the directory, or a manifest its author put at the
    root. Without the second, a sample corpus whose ``sha256sums.txt`` sits beside a
    ``malware/`` folder would hand its claim down to that folder and leave the manifest
    outside the asset.
    """
    security_extensions = set(CAPTURE_EXTENSIONS + FLOW_EXTENSIONS + EVENT_LOG_EXTENSIONS)
    anchoring_names = ZEEK_LOG_NAMES | SURICATA_NAMES | _ROOT_MANIFESTS
    return any(
        entry.extension in security_extensions
        or entry.name.lower() in anchoring_names
        # A file named for a public corpus is the corpus, wherever it is kept. The
        # development machine keeps `UNSW-NB15_1.csv` in a folder called `Datasets`, and
        # without this the container guard refused the only real security data on it.
        or _match_known_name(entry.name)[0] is not None
        for entry in ctx.files
    )


def _sniff_ids_columns(path: str) -> str | None:
    """Return the intrusion-dataset family a table's header identifies, or ``None``.

    Reads the first :data:`HEADER_BYTES` and nothing else. A file that cannot be read is
    reported as no match, which is the same answer as a file whose columns mean nothing:
    either way this observation contributes nothing and the others decide.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(HEADER_BYTES)
    except OSError as exc:
        logger.debug("Cannot read header of %s: %s", path, exc)
        return None

    line = head.split(b"\n", 1)[0].decode("utf-8", errors="replace").lower()
    if not line:
        return None

    columns = {
        column.strip().strip('"').strip("'")
        for column in re.split(r"[,;\t|]", line)
    }
    columns.discard("")
    if len(columns) < 4:
        return None

    for family, expected in IDS_COLUMN_SETS:
        if sum(1 for column in expected if column in columns) >= 3:
            return family
    return None


def is_security_container(ctx: DirectoryContext) -> bool:
    r"""Report whether a directory is a shelf that holds datasets rather than being one.

    The security equivalent of ``D:\Models``: a folder called ``datasets``, ``downloads`` or
    ``data`` is named for what it contains, and only its own files can rescue it. Uses the
    same name list the boundary guard applies to the generic detectors, because the
    judgement is the same one.
    """
    return is_container_name(ctx.name) and not _is_anchored(ctx)
