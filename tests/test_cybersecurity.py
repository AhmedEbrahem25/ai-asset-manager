"""The cybersecurity dataset detector.

Half of these tests are about what the detector must *not* claim. That is deliberate: a
security corpus is structurally a pile of CSVs, logs or JSON, and so is half of a normal
machine. The value of the detector is entirely in where it draws the line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_asset_manager.backend.detectors.cybersecurity import (
    CyberSecurityDatasetDetector,
    _match_known_name,
    gather_evidence,
)
from ai_asset_manager.backend.models.enums import AssetKind, DatasetFormat

#: A CSV header carrying UNSW-NB15's distinctive columns.
UNSW_HEADER = (
    b"srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,sload,dload,"
    b"ct_srv_src,attack_cat,label\n"
)

#: CICFlowMeter's, which every CIC dataset since 2017 uses.
CIC_HEADER = (
    b"Flow ID,Source IP,Source Port,Destination IP,Flow Duration,Total Fwd Packets,"
    b"Fwd Packet Length Max,Bwd Packet Length Max,Flow Bytes/s,Flow IAT Mean,Label\n"
)


def _rows(header: bytes, count: int = 400) -> bytes:
    """Return a plausible CSV body, big enough to clear the size floor."""
    return header + b",".join([b"0"] * 16) + b"\n" * count + b"0," * 4096


def _detect(path: Path, context_for):
    """Run the detector against one directory."""
    return CyberSecurityDatasetDetector().detect(context_for(path))


# ---------------------------------------------------------------------------
# what it must claim
# ---------------------------------------------------------------------------


def test_detects_a_packet_capture_corpus(tmp_path: Path, context_for):
    root = tmp_path / "traffic"
    root.mkdir()
    for day in range(4):
        (root / f"day{day}.pcap").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\0" * 8192)
    (root / "labels.csv").write_bytes(b"file,attack\nday0.pcap,dos\n")

    results = _detect(root, context_for)

    assert len(results) == 1
    assert results[0].kind is AssetKind.DATASET
    assert results[0].subkind == DatasetFormat.NETWORK_CAPTURE.value
    assert results[0].evidence["signals"]


def test_detects_a_zeek_run(tmp_path: Path, context_for):
    root = tmp_path / "zeek-logs"
    root.mkdir()
    for name in ("conn.log", "dns.log", "http.log", "ssl.log", "weird.log"):
        (root / name).write_bytes(b"#separator \\x09\nts\tuid\n")

    results = _detect(root, context_for)

    assert results
    assert results[0].subkind == DatasetFormat.NETWORK_CAPTURE.value
    assert any("Zeek" in signal for signal in results[0].evidence["signals"])


def test_detects_an_intrusion_table_by_its_columns(tmp_path: Path, context_for):
    root = tmp_path / "flows"
    root.mkdir()
    (root / "training.csv").write_bytes(_rows(UNSW_HEADER))
    (root / "testing.csv").write_bytes(_rows(UNSW_HEADER))

    results = _detect(root, context_for)

    assert results
    assert results[0].subkind == DatasetFormat.INTRUSION_DETECTION.value


def test_recognises_a_public_dataset_by_name(tmp_path: Path, context_for):
    root = tmp_path / "CSE-CIC-IDS2018"
    root.mkdir()
    (root / "Thursday-01-03-2018.csv").write_bytes(_rows(CIC_HEADER))

    results = _detect(root, context_for)

    assert results
    assert results[0].name == "CSE-CIC-IDS2018"
    assert results[0].evidence["known_dataset"] == "CSE-CIC-IDS2018"


def test_recognises_a_dataset_named_by_an_ancestor(tmp_path: Path, context_for):
    root = tmp_path / "UNSW-NB15"
    inner = root / "csv" / "training"
    inner.mkdir(parents=True)
    (inner / "part1.csv").write_bytes(_rows(UNSW_HEADER))

    results = _detect(inner, context_for)

    assert results
    assert results[0].evidence["known_dataset"] == "UNSW-NB15"


def test_recognises_a_corpus_named_only_by_a_filename(tmp_path: Path, context_for):
    """The layout the development machine actually uses, and the one that was missed.

    One flat folder called ``Datasets``: a headerless ``UNSW-NB15_1.csv`` beside Suricata
    output and 71 GB of authentication logs. Nothing in the path names a corpus, the raw
    UNSW partitions ship with no header row so column sniffing cannot reach them, and
    ``Datasets`` is a container name — so every guard in the module refused it.
    """
    root = tmp_path / "Datasets"
    root.mkdir()
    (root / "UNSW-NB15_1.csv").write_bytes(b"59.166.0.0,1390,149.171.126.6,53,udp\n" * 500)
    (root / "eve.json").write_bytes(b'{"event_type":"alert"}\n' * 500)
    (root / "auth.txt").write_bytes(b"1,ANONYMOUS LOGON@C586,C1250,NTLM,Network,LogOn\n" * 999)

    results = _detect(root, context_for)

    assert results
    result = results[0]
    # Named for the folder, not for the corpus: it holds UNSW-NB15 among others, and
    # calling the whole thing UNSW-NB15 would misdescribe most of it.
    assert result.name == "Datasets"
    assert result.evidence["known_dataset"] == "UNSW-NB15"
    assert any("UNSW-NB15_1.csv" in signal for signal in result.evidence["signals"])


def test_a_folder_named_for_a_corpus_takes_its_name(tmp_path: Path, context_for):
    root = tmp_path / "CICIDS2017"
    root.mkdir()
    (root / "eve.json").write_bytes(b'{"event_type":"alert"}\n' * 500)
    (root / "day1.pcap").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\0" * 8192)

    results = _detect(root, context_for)

    assert results
    assert results[0].name == "CICIDS2017"


def test_detects_windows_event_logs(tmp_path: Path, context_for):
    root = tmp_path / "EVTX-ATTACK-SAMPLES"
    root.mkdir()
    for index in range(4):
        (root / f"sysmon_{index}.evtx").write_bytes(b"ElfFile\0" + b"\0" * 4096)

    results = _detect(root, context_for)

    assert results
    assert results[0].subkind == DatasetFormat.HOST_LOG.value


def test_detects_a_malware_corpus(tmp_path: Path, context_for):
    root = tmp_path / "corpus"
    samples = root / "malware" / "samples"
    samples.mkdir(parents=True)
    for index in range(8):
        (samples / f"{index:032x}").write_bytes(b"MZ" + b"\0" * 512)
    (root / "sha256sums.txt").write_bytes(b"")

    results = _detect(root, context_for)

    assert results
    assert results[0].subkind == DatasetFormat.MALWARE.value


# ---------------------------------------------------------------------------
# what it must not claim
# ---------------------------------------------------------------------------


def test_one_csv_is_not_a_dataset(tmp_path: Path, context_for):
    root = tmp_path / "reports"
    root.mkdir()
    (root / "expenses.csv").write_bytes(b"date,amount\n2026-01-01,12\n" + b"0," * 4096)

    assert _detect(root, context_for) == []


def test_one_log_is_not_a_dataset(tmp_path: Path, context_for):
    root = tmp_path / "server"
    root.mkdir()
    (root / "conn.log") .write_bytes(b"ts\tuid\n")

    assert _detect(root, context_for) == []


def test_one_json_is_not_a_dataset(tmp_path: Path, context_for):
    root = tmp_path / "config"
    root.mkdir()
    (root / "eve.json").write_bytes(b"{}")

    # Suricata's `eve.json` is a strong marker but a single observation, and the rule is
    # that one file is never enough.
    evidence = gather_evidence(context_for(root))
    assert len(evidence.signals) < 2
    assert _detect(root, context_for) == []


def test_a_bare_dataset_name_claims_nothing(tmp_path: Path, context_for):
    root = tmp_path / "CICIDS2017"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"downloaded this one day")

    assert _detect(root, context_for) == []


def test_a_samples_folder_alone_is_not_malware(tmp_path: Path, context_for):
    root = tmp_path / "project"
    samples = root / "samples"
    samples.mkdir(parents=True)
    for index in range(20):
        (samples / f"example_{index}.txt").write_bytes(b"hello")

    assert _detect(root, context_for) == []


class TestFalsePositivesFoundOnTheDevelopmentMachine:
    """Every one of these was reported as a security dataset by the first version.

    Kept as a class because they belong together: each is an ordinary directory on an
    ordinary Windows machine, and between them they cost a 14 GB NLP project, twenty-two
    training runs and 190 GB of crash dumps their correct classification.
    """

    def test_a_python_install_is_not_a_malware_corpus(self, tmp_path: Path, context_for):
        """Python ships a directory literally called `DLLs`."""
        root = tmp_path / "Python313"
        dlls = root / "DLLs"
        dlls.mkdir(parents=True)
        for index in range(60):
            (dlls / f"_module{index}.pyd").write_bytes(b"MZ")
            (dlls / f"lib{index}.dll").write_bytes(b"MZ")

        assert _detect(root, context_for) == []

    def test_a_disassembler_is_not_a_malware_corpus(self, tmp_path: Path, context_for):
        """Ghidra and Cutter ship `pe/` and `elf/` processor modules."""
        root = tmp_path / "ghidra_11.4.1_PUBLIC"
        for family in ("pe", "elf"):
            target = root / "Processors" / family
            target.mkdir(parents=True)
            for index in range(30):
                (target / f"loader{index}.dll").write_bytes(b"MZ")

        assert _detect(root, context_for) == []

    def test_an_ai_project_is_not_a_malware_corpus(self, tmp_path: Path, context_for):
        """The worst of them: a claim here suppressed every asset in the project."""
        root = tmp_path / "thorn-nlp"
        site = root / ".venv" / "Lib" / "site-packages"
        site.mkdir(parents=True)
        for index in range(120):
            (site / f"_ext{index}.dll").write_bytes(b"MZ")
        cache = root / "cache"
        cache.mkdir()
        for index in range(9):
            (cache / f"{index:032x}").write_bytes(b"x")

        assert _detect(root, context_for) == []

    def test_a_hash_named_cache_is_not_a_malware_corpus(self, tmp_path: Path, context_for):
        """Browsers and editors name cache entries after digests; most of their files."""
        root = tmp_path / "Brave-Browser"
        cache = root / "Cache"
        cache.mkdir(parents=True)
        for index in range(23):
            (cache / f"{index:032x}").write_bytes(b"x")
        for index in range(200):
            (cache / f"asset{index}.dat").write_bytes(b"x")

        assert _detect(root, context_for) == []

    def test_crash_dumps_are_not_packet_captures(self, tmp_path: Path, context_for):
        """`.dmp` is a memory dump far more often than it is anything else."""
        root = tmp_path / "windows10"
        root.mkdir()
        for index in range(5):
            (root / f"vmware-vmx-{index}.dmp").write_bytes(b"\0" * 8192)

        assert _detect(root, context_for) == []

    def test_man_pages_are_not_host_telemetry(self, tmp_path: Path, context_for):
        """`wtmp`, `btmp` and `lastlog` are also the names of manual pages."""
        root = tmp_path / "man"
        root.mkdir()
        for name in ("wtmp.5.gz", "btmp.5.gz", "lastlog.8.gz", "sudo.log.5.gz"):
            (root / name).write_bytes(b"\x1f\x8b")

        assert _detect(root, context_for) == []

    @pytest.mark.parametrize(
        "name",
        ["PKCS7_type_is_other.3ossl.gz", "remember.c", "card_membership.png",
         "is_otherwise.py", "december.log"],
    )
    def test_a_short_corpus_name_does_not_match_by_accident(self, name):
        """`isot` matched an OpenSSL manual page; `ember` matched a Dart icon."""
        assert _match_known_name(name)[0] is None

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("UNSW-NB15_1.csv", "UNSW-NB15"), ("DARPA1998.csv", "DARPA"),
         ("KDD99.txt", "KDD Cup 99"), ("NSL-KDD.csv", "NSL-KDD"),
         ("ember_features.jsonl", "EMBER")],
    )
    def test_a_corpus_name_is_still_recognised(self, name, expected):
        assert _match_known_name(name)[0] == expected

    def test_a_marker_deep_in_a_subtree_does_not_claim_the_root(
        self, tmp_path: Path, context_for
    ):
        r"""`F:\New Apps` held YARA rules under `Yara Roles/rules/malware`, three levels down.

        Counting that marker reported 25 GB of installed software as a malware corpus. A
        corpus's own root sits next to its samples.
        """
        root = tmp_path / "New Apps"
        rules = root / "Yara Roles" / "rules" / "malware"
        rules.mkdir(parents=True)
        for index in range(40):
            (rules / f"rule{index}.yar").write_bytes(b"rule x {}")
        tools = root / "SomeTool"
        tools.mkdir()
        for index in range(200):
            (tools / f"lib{index}.dll").write_bytes(b"MZ")

        assert _detect(root, context_for) == []

    def test_a_windows_profile_root_is_never_a_corpus(self, tmp_path: Path, context_for):
        r"""`AppData\Roaming` was reported as a 14 GB host-log dataset.

        Four applications each writing their own log, spread across a tree holding
        everything installed on the machine — and the claim suppressed thirty-two models
        beneath it. Evidence that only means something concentrated is now counted only
        where it is concentrated.
        """
        root = tmp_path / "Roaming"
        for index, name in enumerate(("auth.log", "audit.log", "secure.log", "sudo.log")):
            app = root / f"SomeApp{index}" / "logs" / "nested"
            app.mkdir(parents=True)
            (app / name).write_bytes(b"log line\n" * 100)

        assert _detect(root, context_for) == []

    def test_a_wordlist_collection_is_not_a_malware_corpus(
        self, tmp_path: Path, context_for
    ):
        """SecLists keeps malware-themed *passwords* beside a list of password hashes."""
        root = tmp_path / "Passwords"
        malware = root / "Malware"
        malware.mkdir(parents=True)
        (malware / "miner-passwords.txt").write_bytes(b"hunter2\n" * 500)
        (malware / "conficker.txt").write_bytes(b"letmein\n" * 500)
        (root / "hashes.txt").write_bytes(b"5f4dcc3b5aa765d61d8327deb882cf99\n" * 100)

        assert _detect(root, context_for) == []

    def test_a_real_sample_corpus_still_qualifies(self, tmp_path: Path, context_for):
        """The tightening must not cost the true positive it was protecting."""
        root = tmp_path / "theZoo"
        malware = root / "malware" / "Binaries"
        malware.mkdir(parents=True)
        for index in range(40):
            (malware / f"{index:032x}.exe").write_bytes(b"MZ")
        (root / "sha256sums.txt").write_bytes(b"")

        results = _detect(root, context_for)

        assert results
        assert results[0].subkind == DatasetFormat.MALWARE.value


def test_application_state_is_never_a_dataset(tmp_path: Path, context_for):
    root = tmp_path / ".claude" / "logs"
    root.mkdir(parents=True)
    for name in ("conn.log", "dns.log", "http.log", "ssl.log"):
        (root / name).write_bytes(b"ts\tuid\n")

    assert _detect(root, context_for) == []


def test_a_shelf_defers_to_the_datasets_it_holds(tmp_path: Path, context_for):
    shelf = tmp_path / "cybersecurity-datasets"
    for name, header in (("UNSW-NB15", UNSW_HEADER), ("CICIDS2017", CIC_HEADER)):
        child = shelf / name
        child.mkdir(parents=True)
        (child / "train.csv").write_bytes(_rows(header))
        (child / "test.csv").write_bytes(_rows(header))

    assert _detect(shelf, context_for) == []

    for name in ("UNSW-NB15", "CICIDS2017"):
        found = _detect(shelf / name, context_for)
        assert found, f"{name} should be detected on its own"
        assert found[0].evidence["known_dataset"]


def test_the_smallest_qualifying_directory_wins(tmp_path: Path, context_for):
    """A claim suppresses everything below it, so claiming too high deletes right answers.

    The regression this guards: a folder holding a model library, a capture corpus and an
    intrusion dataset looked, from its subtree alone, like one packet-capture dataset — and
    claiming it hid all three.
    """
    root = tmp_path / "work"
    captures = root / "captures"
    captures.mkdir(parents=True)
    for index in range(5):
        (captures / f"day{index}.pcap").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\0" * 8192)

    flows = root / "UNSW-NB15"
    flows.mkdir()
    (flows / "train.csv").write_bytes(_rows(UNSW_HEADER))

    assert _detect(root, context_for) == [], "the parent must not swallow both"
    assert _detect(captures, context_for)
    assert _detect(flows, context_for)


def test_a_container_named_folder_needs_data_of_its_own(tmp_path: Path, context_for):
    root = tmp_path / "Downloads"
    root.mkdir()
    (root / "export.csv").write_bytes(_rows(UNSW_HEADER))

    assert _detect(root, context_for) == []


def test_a_container_named_split_inside_a_known_dataset_still_counts(
    tmp_path: Path, context_for
):
    """`training` is a container name everywhere except inside a corpus that names itself."""
    split = tmp_path / "NSL-KDD" / "training"
    split.mkdir(parents=True)
    (split / "part1.csv").write_bytes(_rows(UNSW_HEADER))

    results = _detect(split, context_for)

    assert results
    assert results[0].name == "NSL-KDD"


def test_a_root_manifest_anchors_the_corpus(tmp_path: Path, context_for):
    """A hash manifest at the root is its author saying where the dataset starts."""
    root = tmp_path / "corpus"
    samples = root / "malware"
    samples.mkdir(parents=True)
    for index in range(8):
        (samples / f"{index:032x}").write_bytes(b"MZ" + b"\0" * 512)
    (root / "sha256sums.txt").write_bytes(b"")

    results = _detect(root, context_for)

    assert results, "the manifest belongs to the asset, so the asset starts here"
    assert results[0].subkind == DatasetFormat.MALWARE.value


def test_a_single_dataset_with_subfolders_is_not_treated_as_a_shelf(
    tmp_path: Path, context_for
):
    root = tmp_path / "TON_IoT"
    for split in ("train", "test"):
        (root / split).mkdir(parents=True)
        (root / split / "flows.csv").write_bytes(_rows(UNSW_HEADER))

    results = _detect(root, context_for)

    # Both children qualify on their own evidence, which is exactly the case the shelf
    # rule must not mistake for two datasets. `train` and `test` beside each other are one
    # corpus's splits, and that is what tells them apart from two unrelated downloads.
    assert len(results) == 1
    assert results[0].name == "TON_IoT"
