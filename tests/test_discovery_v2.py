r"""Tests for deep discovery, the new asset kinds and the relationship graph.

Deep discovery is a search with a budget, so both halves are tested: that it reaches
libraries the name-based sweep cannot see, and that it refuses to go where it must not.
"""

from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy.orm import Session

from ai_asset_manager.backend.models import Asset, AssetLink
from ai_asset_manager.backend.models.enums import AssetKind
from ai_asset_manager.backend.scanner.locations import deep_sweep
from ai_asset_manager.backend.scanner.scoring import (
    DESCEND_THRESHOLD,
    REPORT_THRESHOLD,
    score_directory,
)
from ai_asset_manager.backend.services.linking_service import LinkingService
from tests import factories as F


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestScoring:
    def test_weights_are_the_strongest_signal(self) -> None:
        rating = score_directory("anything", ["model.safetensors"], [])
        assert rating.worth_reporting

    def test_a_lone_config_is_not_enough_to_report(self) -> None:
        # Every JavaScript project on earth has a config.json.
        rating = score_directory("thing", ["config.json"], [])
        assert not rating.worth_reporting

    def test_a_config_beside_a_tokenizer_is(self) -> None:
        rating = score_directory("thing", ["config.json", "tokenizer.json"], [])
        assert rating.worth_reporting

    def test_a_cache_root_is_recognised_by_the_shape_of_its_children(self) -> None:
        """A cache root's own listing is nothing but folder names."""
        rating = score_directory(
            "hub", ["version.txt"], ["models--Qwen--Qwen2.5-0.5B", "models--google--byt5"]
        )
        assert rating.worth_reporting

    def test_an_ollama_store_is_recognised(self) -> None:
        assert score_directory("models", [], ["manifests", "blobs"]).worth_reporting

    def test_a_folder_of_installers_scores_nothing(self) -> None:
        rating = score_directory(
            "Setup", ["a.exe", "b.exe", "c.msi", "d.iso", "readme.txt"], []
        )
        assert rating.score == 0.0
        assert not rating.worth_descending

    def test_bin_files_alone_do_not_look_like_a_model(self) -> None:
        """Electron apps, webcam drivers and installers all ship ``.bin`` blobs.

        Including that extension sent the sweep to a game launcher and a virtual camera
        and offered both as AI libraries.
        """
        rating = score_directory("resources", ["app.bin", "data.bin"], [])
        assert not rating.worth_reporting

    def test_a_library_shelf_is_worth_entering_but_not_reporting(self) -> None:
        rating = score_directory("models", [], ["qwen", "llama"])
        assert rating.worth_descending
        assert not rating.worth_reporting

    def test_the_thresholds_are_ordered(self) -> None:
        assert DESCEND_THRESHOLD < REPORT_THRESHOLD

    def test_evidence_explains_the_score(self) -> None:
        rating = score_directory("m", ["model.safetensors", "config.json"], [])
        assert rating.evidence


class TestDeepSweep:
    def test_it_finds_a_library_nested_inside_a_project(self, tmp_path: Path) -> None:
        """The case the name-based sweep cannot reach: nothing in the path says "model"."""
        cache = tmp_path / "NLP-Project" / "app" / "data" / "hf_cache" / "hub"
        F.make_hf_cache_repo(cache, "Qwen/Qwen1.5-4B-Chat")

        found = {item.path for item in deep_sweep([tmp_path], budget_seconds=20)}

        assert cache.resolve() in found

    def test_it_never_enters_a_system_directory(self, tmp_path: Path) -> None:
        for system in ("Windows", "Program Files", "$Recycle.Bin", "node_modules", ".venv"):
            target = tmp_path / system / "models"
            target.mkdir(parents=True)
            (target / "model.safetensors").write_bytes(b"0" * 128)

        assert deep_sweep([tmp_path], budget_seconds=20) == []

    def test_it_stops_at_its_budget(self, tmp_path: Path) -> None:
        # A wide tree that would take far longer than the budget to exhaust.
        for outer in range(40):
            for inner in range(40):
                (tmp_path / f"d{outer}" / f"e{inner}").mkdir(parents=True)

        started = time.monotonic()
        deep_sweep([tmp_path], budget_seconds=1.0)
        elapsed = time.monotonic() - started

        assert elapsed < 15.0, "the budget did not stop the search"

    def test_a_reported_directory_is_not_picked_apart(self, tmp_path: Path) -> None:
        """Offering both a library and its contents means scanning the same tree twice."""
        hub = tmp_path / "cache" / "hub"
        F.make_hf_cache_repo(hub, "Qwen/Qwen2.5-0.5B-Instruct")

        found = {item.path for item in deep_sweep([tmp_path], budget_seconds=20)}

        assert hub.resolve() in found
        assert not any(str(hub) in str(path) and path != hub.resolve() for path in found)

    def test_an_unreadable_directory_does_not_hide_its_target(self, tmp_path: Path) -> None:
        r"""Windows keeps junctions like ``C:\Documents and Settings`` that deny access.

        They resolve onto a real directory, so marking that identity as visited before the
        read succeeds lets the junction hide the folder it points at. On the development
        machine this hid the entire user profile and every cache beneath it.
        """
        real = tmp_path / "Users"
        cache = real / "pc" / ".cache" / "hub"
        F.make_hf_cache_repo(cache, "Qwen/Qwen2.5-0.5B-Instruct")

        found = {item.path for item in deep_sweep([tmp_path], budget_seconds=20)}

        assert cache.resolve() in found

    def test_results_carry_the_evidence_that_earned_them(self, tmp_path: Path) -> None:
        target = tmp_path / "somewhere" / "weights"
        target.mkdir(parents=True)
        (target / "model.safetensors").write_bytes(b"0" * 128)

        found = deep_sweep([tmp_path], budget_seconds=20)

        assert found and all(item.origin for item in found)


class TestProjectDetection:
    def test_a_training_codebase_is_a_project(self, tmp_path: Path, detectors, walk) -> None:
        _write(tmp_path / "app" / "requirements.txt", "ultralytics\ntorch\n")
        _write(tmp_path / "app" / "train.py", "pass\n")

        found = detectors.detect_tree(walk(tmp_path))

        assert [item.kind for item in found] == [AssetKind.PROJECT]

    def test_one_signal_is_not_enough(self, tmp_path: Path, detectors, walk) -> None:
        # A requirements.txt describes most of a developer's disk.
        _write(tmp_path / "app" / "requirements.txt", "requests\nflask\n")

        assert detectors.detect_tree(walk(tmp_path)) == []

    def test_a_subdirectory_is_not_its_own_project(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """Without a root marker, every ``src/`` in a repository becomes a project."""
        _write(tmp_path / "app" / "pyproject.toml", "[project]\ndependencies=['torch']\n")
        _write(tmp_path / "app" / "train.py", "pass\n")
        _write(tmp_path / "app" / "src" / "train.py", "pass\n")
        _write(tmp_path / "app" / "src" / "predict.py", "pass\n")

        found = [item.root_path for item in detectors.detect_tree(walk(tmp_path))]

        assert found == [str(tmp_path / "app")]

    def test_a_project_does_not_swallow_the_models_inside_it(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        project = tmp_path / "app"
        _write(project / "requirements.txt", "transformers\n")
        _write(project / "train.py", "pass\n")
        F.make_hf_model(project / "artifacts", "my-model", architecture="Qwen2ForCausalLM")

        kinds = {item.kind for item in detectors.detect_tree(walk(tmp_path))}

        assert kinds == {AssetKind.PROJECT, AssetKind.MODEL}

    def test_a_projects_size_excludes_the_assets_inside_it(
        self, tmp_path: Path, pipeline, walk
    ) -> None:
        """Otherwise every nested model is counted twice in the storage total."""
        project = tmp_path / "app"
        _write(project / "requirements.txt", "transformers\n")
        _write(project / "train.py", "pass\n")
        F.make_hf_model(project / "artifacts", "my-model", architecture="Qwen2ForCausalLM")

        records = {r.root_path: r for r in pipeline.process_tree(walk(tmp_path))}
        project_record = records[str(project)]
        model_record = records[str(project / "artifacts" / "my-model")]

        assert model_record.size_bytes > 0
        assert project_record.size_bytes < model_record.size_bytes


class TestExperimentDetection:
    def test_tensorboard_event_files(self, tmp_path: Path, detectors, walk) -> None:
        run = tmp_path / "logs" / "run1"
        run.mkdir(parents=True)
        (run / "events.out.tfevents.1700000000.host").write_bytes(b"0" * 512)

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.root_path) for item in found] == [
            (AssetKind.EXPERIMENT, str(run))
        ]

    def test_a_wandb_run_not_its_container(self, tmp_path: Path, detectors, walk) -> None:
        """``wandb/`` holds many runs; the run is the asset."""
        for name in ("run-20260515_011641-abc", "run-20260515_200744-def"):
            run = tmp_path / "wandb" / name
            _write(run / "files" / "wandb-metadata.json", "{}")

        found = detectors.detect_tree(walk(tmp_path))

        assert len(found) == 2
        assert all(item.kind is AssetKind.EXPERIMENT for item in found)
        assert str(tmp_path / "wandb") not in {item.root_path for item in found}

    def test_an_mlflow_run_needs_its_stores(self, tmp_path: Path, detectors, walk) -> None:
        # An MLflow *experiment* also carries meta.yaml; only a run has the stores.
        experiment = tmp_path / "mlruns" / "0"
        _write(experiment / "meta.yaml", "name: default\n")
        run = experiment / "abc123"
        _write(run / "meta.yaml", "run_id: abc123\n")
        (run / "metrics").mkdir()
        (run / "params").mkdir()

        found = [item.root_path for item in detectors.detect_tree(walk(tmp_path))]

        assert found == [str(run)]

    def test_an_ultralytics_run_without_weights(self, tmp_path: Path, detectors, walk) -> None:
        """Interrupted runs keep ``args.yaml`` and nothing else, and still count."""
        run = tmp_path / "runs" / "detect" / "train"
        _write(run / "args.yaml", "epochs: 100\n")
        (run / "weights").mkdir()

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.subkind) for item in found] == [
            (AssetKind.EXPERIMENT, "ultralytics")
        ]

    def test_a_lightning_version_directory(self, tmp_path: Path, detectors, walk) -> None:
        run = tmp_path / "lightning_logs" / "version_3"
        _write(run / "hparams.yaml", "lr: 0.001\n")

        found = detectors.detect_tree(walk(tmp_path))

        assert [item.root_path for item in found] == [str(run)]


class TestAnnotationDetection:
    def test_a_label_studio_workspace(self, tmp_path: Path, detectors, walk) -> None:
        workspace = tmp_path / "labelling"
        workspace.mkdir()
        (workspace / "label_studio.sqlite3").write_bytes(b"SQLite format 3\x00")

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.subkind) for item in found] == [
            (AssetKind.ANNOTATION_PROJECT, "label_studio")
        ]

    def test_a_roboflow_export_is_reported_as_annotation_work(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """Its layout is a YOLO dataset; the receipt says where it came from."""
        export = F.make_yolo_dataset(tmp_path, "traffic-signs", images=5)
        _write(export / "README.roboflow.txt", "exported from Roboflow\n")

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.subkind) for item in found] == [
            (AssetKind.ANNOTATION_PROJECT, "roboflow")
        ]

    def test_a_cvat_export_needs_both_markers(self, tmp_path: Path, detectors, walk) -> None:
        task = tmp_path / "task-7"
        _write(task / "task.json", "{}")

        assert detectors.detect_tree(walk(tmp_path)) == []

        _write(task / "annotations.xml", "<annotations/>")
        found = detectors.detect_tree(walk(tmp_path))
        assert [item.subkind for item in found] == ["cvat"]


class TestFormatGaps:
    def test_a_paddle_inference_bundle(self, tmp_path: Path, detectors, walk) -> None:
        """Neither file is a model on its own, so nothing used to look at the pair."""
        bundle = tmp_path / "arabic_PP-OCRv4_rec_infer"
        bundle.mkdir()
        (bundle / "inference.pdmodel").write_bytes(b"0" * 1024)
        (bundle / "inference.pdiparams").write_bytes(b"0" * 4096)

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.subkind) for item in found] == [(AssetKind.MODEL, "ocr")]

    def test_an_idx_dataset(self, tmp_path: Path, detectors, walk) -> None:
        """MNIST ships four binary files and no structure any other rule can see."""
        raw = tmp_path / "MNIST" / "raw"
        raw.mkdir(parents=True)
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"):
            (raw / name).write_bytes(b"\x00\x00\x08\x03" + b"0" * 256)

        found = detectors.detect_tree(walk(tmp_path))

        assert [(item.kind, item.name) for item in found] == [(AssetKind.DATASET, "MNIST")]

    def test_a_kraken_model(self, tmp_path: Path, detectors, walk) -> None:
        directory = tmp_path / "kraken_cache"
        directory.mkdir()
        (directory / "arabic_historical.mlmodel").write_bytes(b"0" * (2 * 1024 * 1024))

        found = detectors.detect_tree(walk(tmp_path))

        assert [item.kind for item in found] == [AssetKind.MODEL]

    def test_epoch_weights_are_checkpoints_not_models(
        self, tmp_path: Path, detectors, walk
    ) -> None:
        """Five epochs of one run are one run's checkpoints, not five models."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for name in ("dapt_epoch0.pt", "dapt_epoch1.pt", "best_dapt.pt"):
            # Over the 1 MiB floor, below which a weight file is treated as a companion.
            F.write_torch_checkpoint(artifacts / name, storages=2, storage_bytes=700_000)

        found = detectors.detect_tree(walk(tmp_path))

        assert {item.kind for item in found} == {AssetKind.CHECKPOINT}
        assert len(found) == 3


class TestRelationshipGraph:
    def _asset(self, session: Session, path: str, kind: AssetKind, name: str) -> Asset:
        asset = Asset(root_path=path, kind=kind.value, name=name)
        session.add(asset)
        session.flush()
        return asset

    def test_a_model_inside_a_project_belongs_to_it(self, session: Session) -> None:
        project = self._asset(session, r"D:\app", AssetKind.PROJECT, "app")
        model = self._asset(session, r"D:\app\artifacts\m", AssetKind.MODEL, "m")

        LinkingService(session).rebuild()

        edge = session.query(AssetLink).one()
        assert (edge.source_id, edge.target_id, edge.relation) == (
            model.id, project.id, "belongs_to"
        )

    def test_the_innermost_project_wins(self, session: Session) -> None:
        """A monorepo holds projects inside projects; the nearer one owns the asset."""
        self._asset(session, r"D:\app", AssetKind.PROJECT, "app")
        inner = self._asset(session, r"D:\app\services\clause", AssetKind.PROJECT, "clause")
        model = self._asset(session, r"D:\app\services\clause\m", AssetKind.MODEL, "m")

        LinkingService(session).rebuild()

        owners = {
            edge.target_id
            for edge in session.query(AssetLink).filter_by(source_id=model.id)
        }
        assert owners == {inner.id}

    def test_a_checkpoint_inside_a_run_was_produced_by_it(self, session: Session) -> None:
        run = self._asset(session, r"D:\runs\train", AssetKind.EXPERIMENT, "train")
        ckpt = self._asset(
            session, r"D:\runs\train\weights\best.pt", AssetKind.CHECKPOINT, "best"
        )

        LinkingService(session).rebuild()

        relations = {
            edge.relation
            for edge in session.query(AssetLink).filter_by(source_id=ckpt.id, target_id=run.id)
        }
        assert relations == {"produced_by"}

    def test_a_sibling_directory_is_not_treated_as_a_parent(self, session: Session) -> None:
        # "D:\app2" must never be read as living inside "D:\app".
        self._asset(session, r"D:\app", AssetKind.PROJECT, "app")
        self._asset(session, r"D:\app2\m", AssetKind.MODEL, "m")

        LinkingService(session).rebuild()

        assert session.query(AssetLink).count() == 0

    def test_rebuilding_replaces_rather_than_accumulates(self, session: Session) -> None:
        self._asset(session, r"D:\app", AssetKind.PROJECT, "app")
        self._asset(session, r"D:\app\m", AssetKind.MODEL, "m")

        service = LinkingService(session)
        service.rebuild()
        service.rebuild()

        assert session.query(AssetLink).count() == 1

    def test_a_missing_asset_is_left_out_of_the_graph(self, session: Session) -> None:
        self._asset(session, r"D:\app", AssetKind.PROJECT, "app")
        gone = self._asset(session, r"D:\app\m", AssetKind.MODEL, "m")
        gone.is_missing = True
        session.flush()

        LinkingService(session).rebuild()

        assert session.query(AssetLink).count() == 0
