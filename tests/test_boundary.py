r"""Tests for asset boundary detection.

Every case here is one that actually happened on the development machine. The generic
detectors are the only ones that can mistake a container for an asset — a COCO annotation
file is proof wherever it sits — so this is where their limits are pinned down.

The failure being guarded against has a particular shape: detection runs parents before
children and a claim suppresses everything below it, so a rule that fires too high does not
merely add a wrong row, it *removes* every right one underneath.
"""

from __future__ import annotations

from pathlib import Path

from ai_asset_manager.backend.detectors.boundary import (
    is_container_name,
    is_drive_root,
    looks_like_dataset_root,
    may_claim_generic,
)
from ai_asset_manager.backend.models.enums import AssetKind
from tests import factories as F


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _images(directory: Path, count: int, prefix: str = "img") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{prefix}{index}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 64)


class TestDriveRoots:
    def test_a_drive_root_is_recognised(self) -> None:
        assert is_drive_root("F:\\") or is_drive_root("/")

    def test_an_ordinary_directory_is_not(self) -> None:
        assert not is_drive_root("F:\\Models")
        assert not is_drive_root("/home/user/models")

    def test_no_ancestor_is_claimed_for_records_buried_beneath_it(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """The headline failure: two stray records making a whole disk one dataset.

        The rule that fired counted the *subtree*, so it matched at every level above the
        corpus and, running parents first, claimed the topmost. Only the corpus itself may
        be claimed now, whatever is above it.
        """
        corpus = tmp_path / "drive" / "projects" / "thesis" / "corpus"
        _write(corpus / "a.jsonl", '{"t": 1}\n' * 40)
        _write(corpus / "b.jsonl", '{"t": 2}\n' * 40)
        _write(corpus / "_stats.json", '{"records": 80}')
        (tmp_path / "drive" / "unrelated").mkdir(parents=True)

        claimed = [item.root_path for item in detectors.detect_tree(walk(tmp_path))]

        assert claimed == [str(corpus)]


class TestContainerNames:
    def test_user_folders_are_containers(self) -> None:
        for name in ("Downloads", "Desktop", "Documents", "Pictures", "Projects"):
            assert is_container_name(name), name

    def test_library_shelves_are_containers(self) -> None:
        # The less obvious half: "Models" is where assets live, not an asset.
        for name in ("Models", "datasets", "checkpoints", "weights", "AI", "runs"):
            assert is_container_name(name), name

    def test_an_asset_name_is_not(self) -> None:
        for name in ("Qwen2.5-7B", "coco8", "hymenoptera_data", "MNIST"):
            assert not is_container_name(name), name

    def test_a_container_holding_weights_may_still_be_claimed(
        self, tmp_path: Path, context_for
    ) -> None:
        """A folder called "models" that *is* a model is still a model.

        The name is evidence, not a verdict; payload files sitting directly in it outrank
        what it happens to be called.
        """
        directory = tmp_path / "models"
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"0" * 128)

        allowed, _ = may_claim_generic(context_for(tmp_path, directory))
        assert allowed is True


class TestDatasetRoots:
    def test_split_directories_identify_a_dataset_root(
        self, tmp_path: Path, context_for
    ) -> None:
        for split in ("train", "val"):
            _images(tmp_path / "corpus" / split, 3)

        assert looks_like_dataset_root(context_for(tmp_path, tmp_path / "corpus"))

    def test_a_manifest_identifies_a_dataset_root(self, tmp_path: Path, context_for) -> None:
        _write(tmp_path / "corpus" / "data.yaml", "nc: 2\n")

        assert looks_like_dataset_root(context_for(tmp_path, tmp_path / "corpus"))

    def test_a_dataset_root_beats_the_container_name(
        self, tmp_path: Path, context_for
    ) -> None:
        """``datasets/train`` + ``datasets/val`` is one dataset, not a shelf."""
        for split in ("train", "val"):
            _images(tmp_path / "datasets" / split, 3)

        allowed, _ = may_claim_generic(context_for(tmp_path, tmp_path / "datasets"))
        assert allowed is True

    def test_a_folder_of_datasets_is_not_a_dataset(
        self, tmp_path: Path, context_for
    ) -> None:
        F.make_coco_dataset(tmp_path / "library", "coco-a", images=3)
        F.make_coco_dataset(tmp_path / "library", "coco-b", images=3)

        allowed, reason = may_claim_generic(context_for(tmp_path, tmp_path / "library"))
        assert allowed is False
        assert "structure" in reason or "container" in reason


class TestTheRegressionsThisFixed:
    def test_a_project_is_not_claimed_as_a_corpus(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """The exact failure found on disk: a whole project filed as one NLP dataset.

        Two ``.jsonl`` files five levels down were enough, because the rule counted the
        subtree. The project root must claim nothing, and the corpus must be found where it
        actually is.
        """
        project = tmp_path / "NLP-Project"
        _write(project / "requirements.txt", "transformers\ntorch\n")
        _write(project / "train.py", "print('train')\n")
        corpus = project / "data" / "civil_law" / "knowledge_base"
        _write(corpus / "articles.jsonl", '{"a": 1}\n' * 40)
        _write(corpus / "clauses.jsonl", '{"b": 2}\n' * 40)
        _write(corpus / "_stats.json", '{"records": 80}')

        found = {r.root_path: r for r in detectors.detect_tree(walk(tmp_path))}

        assert str(corpus) in found, "the corpus itself was not found"
        assert found[str(corpus)].kind is AssetKind.DATASET
        assert found[str(project)].kind is AssetKind.PROJECT

    def test_a_folder_of_figures_is_not_a_dataset(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        # Twenty chart PNGs in a report's `figs` directory used to clear the threshold.
        _images(tmp_path / "work" / "figs", 20, prefix="fig_")

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_a_video_library_is_not_a_dataset(self, tmp_path: Path, detectors, walk) -> None:
        """Bulk is not evidence: a course download and a video dataset are one shape."""
        course = tmp_path / "Android Hacking Course"
        course.mkdir(parents=True)
        for index in range(30):
            (course / f"lesson{index}.mp4").write_bytes(b"0" * 2048)

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_a_screenshot_folder_is_not_a_dataset(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        _images(tmp_path / "Screenshots", 400, prefix="Screenshot ")

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_labelled_media_still_is_a_dataset(self, tmp_path: Path, detectors, walk) -> None:
        """The other side of that trade: labels beside the media make it a dataset again."""
        dataset = tmp_path / "traffic"
        _images(dataset, 400, prefix="frame")
        _write(dataset / "labels" / "frame0.txt", "0 0.5 0.5 0.2 0.2\n")

        found = detectors.detect_tree(walk(tmp_path))
        assert [item.root_path for item in found] == [str(dataset)]

    def test_a_session_log_store_is_not_a_corpus(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """Chat transcripts are line-delimited JSON, which is what a corpus is too.

        The rule that let these through accepted "there is a ``.jsonl`` here" as evidence
        that a ``.jsonl`` pile was a dataset. Fifty of them were catalogued on the
        development machine before it was caught.
        """
        store = tmp_path / ".claude" / "projects" / "some-project"
        for name in ("39d3e5c3.jsonl", "40813648.jsonl", "52bef5ad.jsonl"):
            _write(store / name, '{"role": "user"}\n' * 40_000)

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_the_marker_may_be_an_ancestor(self, tmp_path: Path, detectors, walk) -> None:
        """The leaf is named after a project or a date; the ancestor is the giveaway."""
        for parent in (
            tmp_path / ".codex" / "sessions" / "2026" / "05" / "21",
            tmp_path / "AppData" / "Roaming" / "Code" / "User" / "globalStorage" / "chats",
        ):
            for name in ("a.jsonl", "b.jsonl"):
                _write(parent / name, '{"m": 1}\n' * 40_000)

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_an_application_state_directory_is_never_a_dataset(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        # IDE telemetry is CSV, and a `state.json` beside it must not rescue it either.
        logs = tmp_path / "IdeaIC2024" / "log"
        _write(logs / "state.json", "{}")
        for name in ("events.csv", "metrics.csv"):
            _write(logs / name, "a,b,c\n" + "1,2,3\n" * 40_000)

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_resource_folders_are_not_a_class_taxonomy(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """Android's `data/res` is a folder-per-qualifier tree full of icons."""
        for qualifier in ("drawable-hdpi", "drawable-mdpi", "color-night", "anim-watch"):
            _images(tmp_path / "res" / qualifier, 30, prefix="ic_")

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_a_class_layout_still_counts_with_a_class_list(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """The other side of that trade: a declared class list is enough on its own."""
        dataset = tmp_path / "flowers"
        for name in ("rose", "tulip", "daisy"):
            _images(dataset / name, 30, prefix=name)
        _write(dataset / "classes.txt", "rose\ntulip\ndaisy\n")

        found = detectors.detect_tree(walk(tmp_path))
        assert [item.root_path for item in found] == [str(dataset)]

    def test_a_folder_per_thing_layout_is_not_a_class_taxonomy(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        # 110 add-in folders holding one icon each looked exactly like ImageNet.
        for index in range(12):
            _images(tmp_path / "addins" / f"FA{index:09d}", 1, prefix="icon")

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_two_classes_under_splits_are_one_dataset(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """The canonical two-class tutorial set became four datasets, one per class dir."""
        dataset = F.make_imagefolder_dataset(
            tmp_path, "hymenoptera_data", classes=("ants", "bees"), per_class=20
        )

        found = detectors.detect_tree(walk(tmp_path))

        assert [item.root_path for item in found] == [str(dataset)]

    def test_samples_alone_is_not_nuscenes(self, tmp_path: Path, detectors, walk) -> None:
        # A folder of security logs with a `samples/` child was filed as driving data.
        logs = tmp_path / "Datasets"
        _write(logs / "samples" / "auth.txt", "user login\n")
        _write(logs / "eve.json", "{}\n")

        assert not any(
            item.subkind == "nuscenes" for item in detectors.detect_tree(walk(tmp_path))
        )

    def test_mot_is_rooted_at_the_dataset_not_an_ancestor(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """A subtree search let whatever container held the sequences claim them."""
        split = tmp_path / "Downloads" / "MOT17" / "train"
        for sequence in ("MOT17-02", "MOT17-04"):
            _write(split / sequence / "seqinfo.ini", "[Sequence]\nname=x\n")

        found = {r.root_path: r for r in detectors.detect_tree(walk(tmp_path))}
        mot = [path for path, item in found.items() if item.subkind == "mot"]

        assert mot == [str(split)]
