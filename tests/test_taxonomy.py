"""Tests for the plugin taxonomy.

Two things are being checked. That the built-in plugins classify real assets correctly,
and — the load-bearing one — that the system is genuinely open: a new AI domain must be
addable without editing the registry, the engine, the schema or the CLI. The tests in
:class:`TestExtensibility` are the proof of that claim; if they need a core change to pass,
the architecture has failed whatever the other tests say.
"""

from __future__ import annotations

import pytest

from ai_asset_manager.backend.models.enums import Severity
from ai_asset_manager.backend.taxonomy import (
    AssetProfile,
    Category,
    Classification,
    Domain,
    Finding,
    Section,
    Task,
    TaxonomyRegistry,
    default_registry,
)
from tests.factories import make_profile


@pytest.fixture
def registry() -> TaxonomyRegistry:
    """Return the process-wide registry with every built-in plugin loaded."""
    return default_registry()


class TestPluginLoading:
    def test_every_builtin_plugin_loads(self, registry: TaxonomyRegistry) -> None:
        loaded = registry.plugins()

        for expected in ("core", "vision", "nlp", "ocr", "speech", "multimodal",
                         "generative", "adapters", "datasets", "models", "experiments",
                         "annotation", "medical", "autonomous", "geospatial", "tabular",
                         "documents"):
            assert expected in loaded

    def test_the_registry_itself_names_no_taxonomy(self) -> None:
        """The core must not know a single model family or dataset format.

        Checked against the source rather than the behaviour, because this is a claim
        about where knowledge lives, and it is the kind of claim that quietly stops being
        true the first time someone adds "just one" special case.
        """
        from pathlib import Path

        import ai_asset_manager.backend.taxonomy.registry as registry_module

        source = Path(registry_module.__file__).read_text(encoding="utf-8").lower()

        for forbidden in ("yolo", "coco", "llama", "qwen", "resnet", "whisper",
                          "safetensors", "imagenet", "kitti"):
            assert forbidden not in source

    def test_classifiers_are_ordered_by_priority(self, registry: TaxonomyRegistry) -> None:
        priorities = [classifier.priority for classifier in registry.classifiers()]

        assert priorities == sorted(priorities, reverse=True)

    def test_ocr_outranks_language(self, registry: TaxonomyRegistry) -> None:
        # Modern OCR models are architecturally language models. If this ordering ever
        # inverts, every OCR model on a machine silently becomes an LLM.
        order = [classifier.name for classifier in registry.classifiers()]

        assert order.index("ocr.model") < order.index("nlp.llm")


class TestExtensibility:
    """A new AI domain must cost one plugin and no core edit."""

    def test_a_plugin_adds_a_domain_a_category_and_a_task(self) -> None:
        registry = TaxonomyRegistry()
        registry.load_plugins()

        assert registry.resolve_alias("genomics") is None

        # Everything below is what a third-party plugin's register() would do.
        registry.add_domain(Domain(id="bioinformatics", label="Bioinformatics"))
        registry.add_task(
            Task(id="variant_calling", label="Variant Calling", domain="bioinformatics")
        )
        registry.add_category(
            Category(id="genomics_dataset", label="Genomics Dataset", section="datasets",
                     order=400, domain="bioinformatics", aliases=("genomics",))
        )

        def classify(profile: AssetProfile) -> Classification | None:
            if profile.files.count(".fastq", ".vcf", ".bam"):
                return Classification(
                    category="genomics_dataset", task="variant_calling",
                    domain="bioinformatics", evidence="sequencing files present",
                )
            return None

        registry.add_classifier(classify, name="genomics", priority=700)

        profile = make_profile(
            "hg38-cohort", kind="dataset", files=["sample1.vcf", "sample2.bam"]
        )
        verdict = registry.classify(profile)

        assert verdict.category == "genomics_dataset"
        assert verdict.task == "variant_calling"
        assert verdict.domain == "bioinformatics"
        assert verdict.source == "genomics"
        # The selector works immediately, with no alias table to update.
        assert registry.resolve_alias("genomics") == ("genomics_dataset",)
        assert "genomics_dataset" in (registry.resolve_alias("datasets") or ())
        assert "genomics_dataset" in (registry.resolve_alias("bioinformatics") or ())

    def test_a_plugin_adds_a_health_rule(self) -> None:
        registry = TaxonomyRegistry()

        def rule(profile: AssetProfile) -> list[Finding]:
            if not profile.files.has_name("checksums.txt"):
                return [Finding(code="x.no_checksums", severity=Severity.WARNING,
                                message="No checksum manifest")]
            return []

        registry.add_health_rule(rule, name="x.checksums")
        report = registry.check_health(make_profile("thing", files=["data.bin"]))

        assert report.score == 90
        assert report.status == "warning"
        assert "No checksum manifest" in report.messages()

    def test_a_plugin_adds_statistics(self) -> None:
        registry = TaxonomyRegistry()
        registry.add_statistic(lambda profile: {"custom": profile.file_count}, name="x")

        assert registry.statistics(make_profile("thing", files=["a", "b"])) == {"custom": 2}

    def test_forward_references_do_not_raise(self) -> None:
        """A category may name a section no plugin registered.

        Without this, plugin load order would matter and installing one plugin without its
        neighbours would crash the inventory.
        """
        registry = TaxonomyRegistry()
        registry.add_category(
            Category(id="widget", label="Widget", section="nowhere", domain="nothing")
        )

        assert registry.section("nowhere").label == "Nowhere"
        assert registry.category("widget").section == "nowhere"
        assert registry.task("never_registered").label == "Never Registered"

    def test_an_unknown_category_id_still_renders(self) -> None:
        # A catalogue written while a plugin was installed must stay readable after it is
        # removed.
        registry = TaxonomyRegistry()

        assert registry.label_of("quantum_circuit") == "Quantum Circuit"
        assert registry.section_of("quantum_circuit") == "other"

    def test_a_plugin_may_relabel_a_builtin(self) -> None:
        registry = TaxonomyRegistry()
        registry.load_plugins()

        registry.add_category(
            Category(id="llm", label="Large Language Model", section="models", order=5)
        )

        assert registry.label_of("llm") == "Large Language Model"

    def test_a_failing_classifier_does_not_break_the_inventory(self) -> None:
        registry = TaxonomyRegistry()

        def broken(profile: AssetProfile) -> Classification | None:
            raise RuntimeError("plugin bug")

        registry.add_classifier(broken, name="broken", priority=999)
        registry.add_classifier(
            lambda profile: Classification(category="ok"), name="good", priority=1
        )

        assert registry.classify(make_profile("thing")).category == "ok"

    def test_nothing_matching_still_returns_a_category(self) -> None:
        assert TaxonomyRegistry().classify(make_profile("x")).category == "unclassified"


class TestModelClassification:
    @pytest.mark.parametrize(
        ("name", "model_type", "architecture", "expected"),
        [
            ("Qwen2.5-0.5B-Instruct", "llm", "Qwen2ForCausalLM", "llm"),
            ("surya_rec2", None, None, "ocr"),
            ("PaddleOCR-VL", None, "PaddleOCRVLForConditionalGeneration", "ocr"),
            ("qwen2-vl-7b-instruct", "vision_language", None, "vision_language"),
            ("yolov8n", None, None, "object_detection"),
            ("sam2_hiera_large", None, None, "segmentation"),
            ("bytetrack_x_mot17", "object_detection", None, "tracking"),
            ("resnet18-f37072fd", None, None, "classification"),
            ("whisper-large-v3", None, None, "speech"),
            ("kokoro-82m", None, None, "text_to_speech"),
            ("bge-large-en-v1.5", None, None, "embedding"),
            ("bge-reranker-v2-m3", None, None, "reranker"),
            ("stable-diffusion-xl-base", None, None, "diffusion"),
            ("depth-anything-v2-large", None, None, "vision"),
        ],
    )
    def test_representative_models(
        self, registry: TaxonomyRegistry, name: str,
        model_type: str | None, architecture: str | None, expected: str,
    ) -> None:
        profile = make_profile(
            name, model={"model_type": model_type, "architecture": architecture}
        )

        assert registry.classify(profile).category == expected

    def test_ocr_is_not_mistaken_for_a_language_model(
        self, registry: TaxonomyRegistry
    ) -> None:
        # The architecture says causal LM; only the name says OCR. Getting this wrong is
        # the single most common way an OCR library becomes invisible.
        profile = make_profile(
            "Qari-OCR-v0.3-VL-2B-Instruct",
            model={"model_type": "llm", "architecture": "Qwen2VLForCausalLM"},
        )

        assert registry.classify(profile).category == "ocr"

    def test_a_sentiment_model_is_not_an_image_classifier(
        self, registry: TaxonomyRegistry
    ) -> None:
        """"Classification" is not a vision word.

        The scanner records a sentiment model and a ResNet under the same model type, so
        the head is what has to settle it.
        """
        profile = make_profile(
            "twitter-roberta-base-sentiment-latest",
            model={"model_type": "classification",
                   "architecture": "RobertaForSequenceClassification"},
        )
        verdict = registry.classify(profile)

        assert verdict.category == "text_classification"
        assert verdict.task == "sentiment_analysis"
        assert verdict.domain == "nlp"

    def test_family_follows_the_asset_name_not_the_architecture(
        self, registry: TaxonomyRegistry
    ) -> None:
        # deepseek-r1:8b is a Qwen3 distill: its architecture says Qwen, its name says
        # DeepSeek, and the name is what the user pulled and will look for.
        profile = make_profile(
            "deepseek-r1:8b",
            model={"model_type": "llm", "architecture": "Qwen3ForCausalLM"},
        )

        assert registry.classify(profile).family == "DeepSeek"

    def test_adapters_report_their_base_model(self, registry: TaxonomyRegistry) -> None:
        profile = make_profile(
            "my-lora", kind="adapter",
            model={"model_type": "lora", "base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        )
        verdict = registry.classify(profile)

        assert verdict.category == "adapter"
        assert verdict.family == "Qwen2.5-0.5B-Instruct"

    def test_classification_reports_its_evidence(self, registry: TaxonomyRegistry) -> None:
        # A surprising classification has to be explainable, or it cannot be trusted.
        verdict = registry.classify(make_profile("yolov8n"))

        assert verdict.evidence
        assert verdict.source == "vision.detector"


class TestDatasetClassification:
    @pytest.mark.parametrize(
        ("name", "dataset_format", "expected"),
        [
            ("coco2017", "coco", "detection_dataset"),
            ("my-yolo-set", "yolo", "detection_dataset"),
            ("cityscapes", "cityscapes", "segmentation_dataset"),
            ("mot17", "mot", "tracking_dataset"),
            ("flowers102", "image_classification", "image_dataset"),
            ("common_voice_17", "audio", "audio_dataset"),
            ("wikipedia", "hf_dataset", "nlp_dataset"),
        ],
    )
    def test_representative_datasets(
        self, registry: TaxonomyRegistry, name: str, dataset_format: str, expected: str
    ) -> None:
        profile = make_profile(name, kind="dataset", dataset={"dataset_format": dataset_format})

        assert registry.classify(profile).category == expected

    def test_ocr_corpora_beat_their_layout(self, registry: TaxonomyRegistry) -> None:
        # A TextOCR corpus is COCO-shaped. Filing it under detection puts it on a shelf
        # with COCO where nobody looking for text data would find it.
        profile = make_profile(
            "TextOCR-GT", kind="dataset", dataset={"dataset_format": "coco"}
        )
        verdict = registry.classify(profile)

        assert verdict.category == "ocr_dataset"
        assert verdict.domain == "document_ai"

    def test_driving_datasets_keep_their_shelf_but_gain_a_domain(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "kitti-3d", kind="dataset",
            dataset={"dataset_format": "kitti", "has_lidar": True},
        )
        verdict = registry.classify(profile)

        assert verdict.category == "detection_dataset"
        assert verdict.domain == "autonomous_driving"
        assert "lidar" in verdict.modalities
        assert verdict.task == "3d_object_detection"

    def test_medical_imaging_is_recognised_from_file_extensions_alone(
        self, registry: TaxonomyRegistry
    ) -> None:
        """The scanner has no DICOM detector, and it does not need one.

        The extensions it recorded are enough for a plugin to name the domain.
        """
        profile = make_profile(
            "chest-study", kind="dataset",
            files=["patient01/scan.dcm", "patient02/scan.dcm"],
            dataset={"dataset_format": "custom"},
        )
        verdict = registry.classify(profile)

        assert verdict.category == "medical_dataset"
        assert verdict.domain == "medical"

    def test_an_unrecognised_dataset_still_lands_under_datasets(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile("mystery", kind="dataset", dataset={"dataset_format": "custom"})
        verdict = registry.classify(profile)

        assert verdict.category == "other_dataset"
        assert registry.section_of(verdict.category) == "datasets"


class TestExperimentClassification:
    def test_a_tensorboard_run_with_weights_is_a_training_run(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "train7",
            files=["events.out.tfevents.1700000000.host", "weights/best.pt", "args.yaml"],
        )
        verdict = registry.classify(profile)

        assert verdict.category == "training_run"
        assert verdict.family == "TensorBoard"

    def test_logs_without_weights_are_a_log(self, registry: TaxonomyRegistry) -> None:
        profile = make_profile("train8", files=["events.out.tfevents.1700000000.host"])

        assert registry.classify(profile).category == "experiment_log"

    def test_an_annotation_export_is_recognised(self, registry: TaxonomyRegistry) -> None:
        profile = make_profile(
            "labelling-batch-3", kind="dataset",
            files=["annotations.xml", "images/frame_000001.jpg", "images/frame_000002.jpg"],
        )
        verdict = registry.classify(profile)

        assert verdict.category == "annotation_project"
        assert verdict.family == "CVAT"


class TestHealth:
    def test_a_complete_model_scores_full_marks(self, registry: TaxonomyRegistry) -> None:
        profile = make_profile(
            "good-model", framework="transformers", asset_format="safetensors",
            files=["config.json", "tokenizer.json", "model.safetensors"],
            model={"model_type": "llm", "architecture": "Qwen2ForCausalLM"},
        )
        report = registry.check_health(profile)

        assert report.score == 100
        assert report.is_healthy

    def test_missing_shards_are_caught_from_filenames(
        self, registry: TaxonomyRegistry
    ) -> None:
        """A half-downloaded sharded model looks healthy in a file browser.

        Every shard filename states the expected total, so counting the recorded names
        proves the download finished without opening anything.
        """
        profile = make_profile(
            "big-model", framework="transformers", asset_format="safetensors",
            files=["config.json", "tokenizer.json",
                   "model-00001-of-00004.safetensors",
                   "model-00002-of-00004.safetensors"],
            model={"model_type": "llm"},
        )
        report = registry.check_health(profile)

        codes = {finding.code for finding in report.findings}
        assert "model.missing_shards" in codes
        assert report.status == "error"
        assert "2 of 4" in " ".join(report.messages())

    def test_an_interrupted_download_is_an_error(self, registry: TaxonomyRegistry) -> None:
        profile = make_profile(
            "partial", framework="transformers", asset_format="safetensors",
            files=["config.json", "tokenizer.json", "model.safetensors.incomplete"],
            model={"model_type": "llm"},
        )

        assert "asset.incomplete_download" in {
            finding.code for finding in registry.check_health(profile).findings
        }

    def test_a_gguf_model_is_not_asked_for_a_tokenizer(
        self, registry: TaxonomyRegistry
    ) -> None:
        # GGUF embeds its tokenizer in the header; demanding one beside it would report
        # every quantised model on a machine as broken.
        profile = make_profile(
            "qwen-q4", framework="llama_cpp", asset_format="gguf",
            files=["qwen2.5-0.5b-instruct-q4_k_m.gguf"], model={"model_type": "llm"},
        )

        assert registry.check_health(profile).is_healthy

    def test_an_ollama_blob_counts_as_weights(self, registry: TaxonomyRegistry) -> None:
        # Ollama names its weight blob by digest, with no extension at all.
        profile = make_profile(
            "deepseek-r1:8b", framework="ollama", asset_format="gguf",
            files=["../../blobs/sha256-e6a7edc1a4d7", "."], model={"model_type": "llm"},
        )

        assert "model.no_weights" not in {
            finding.code for finding in registry.check_health(profile).findings
        }

    def test_a_dataset_missing_its_validation_split(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "half-set", kind="dataset",
            files=["readme.md", "license", "train/img1.jpg", "train/labels.json"],
            dataset={"dataset_format": "coco", "num_images": 1, "num_annotations": 4},
        )
        report = registry.check_health(profile)

        codes = {finding.code for finding in report.findings}
        assert "dataset.no_val_split" in codes
        assert "dataset.no_readme" not in codes
        assert "dataset.no_license" not in codes

    def test_missing_lidar_is_reported_for_a_driving_dataset(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "kitti", kind="dataset",
            files=["training/image_2/000000.png", "training/label_2/000000.txt"],
            dataset={"dataset_format": "kitti"},
        )

        codes = {finding.code for finding in registry.check_health(profile).findings}
        assert "driving.no_lidar" in codes
        assert "driving.no_calibration" in codes

    def test_health_is_not_pronounced_on_unloaded_files(
        self, registry: TaxonomyRegistry
    ) -> None:
        """An unevaluated asset must not be reported as perfect.

        Reporting 100/100 for something nobody looked at is worse than reporting nothing.
        """
        report = registry.check_health(make_profile("unknown-model"))

        assert not report.evaluated
        assert report.status == "unknown"

    def test_severities_cost_what_they_are_worth(self, registry: TaxonomyRegistry) -> None:
        error = Finding(code="a", severity=Severity.ERROR, message="")
        warning = Finding(code="b", severity=Severity.WARNING, message="")
        info = Finding(code="c", severity=Severity.INFO, message="")

        assert (error.penalty, warning.penalty, info.penalty) == (25, 10, 3)

    def test_the_score_floors_at_zero(self) -> None:
        registry = TaxonomyRegistry()
        for index in range(6):
            registry.add_health_rule(
                lambda profile, index=index: [
                    Finding(code=f"e{index}", severity=Severity.ERROR, message="bad")
                ],
                name=f"rule{index}",
            )

        assert registry.check_health(make_profile("x", files=["a"])).score == 0

    def test_duplicate_codes_are_penalised_once(self) -> None:
        registry = TaxonomyRegistry()
        finding = [Finding(code="same", severity=Severity.WARNING, message="dup")]
        registry.add_health_rule(lambda profile: finding, name="one")
        registry.add_health_rule(lambda profile: finding, name="two")

        assert registry.check_health(make_profile("x", files=["a"])).score == 90


class TestStatistics:
    def test_dataset_statistics_come_from_the_recorded_files(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "set", kind="dataset",
            files=["README.md", "LICENSE", "train/a.jpg", "train/b.jpg", "val/c.jpg"],
            file_sizes={"train/a.jpg": 3000, "train/b.jpg": 5000, "val/c.jpg": 4000},
            dataset={"dataset_format": "image_classification", "num_classes": 3},
        )
        stats = registry.statistics(profile)

        assert stats["images"] == 3
        assert stats["classes"] == 3
        assert sorted(stats["splits"]) == ["train", "val"]
        assert stats["has_readme"] is True
        assert stats["has_license"] is True
        assert stats["storage_format"] == "imagefolder"
        assert stats["avg_image_bytes"] == 4000

    def test_storage_format_recognises_webdataset_shards(
        self, registry: TaxonomyRegistry
    ) -> None:
        # One tar file is an archive; a run of numbered ones is a streaming format, and
        # which it is decides whether a training script can read it.
        profile = make_profile(
            "shards", kind="dataset",
            files=["shard-000000.tar", "shard-000001.tar"],
            dataset={"dataset_format": "custom"},
        )

        assert registry.statistics(profile)["storage_format"] == "webdataset"

    def test_model_statistics_report_shards_and_tokenizer(
        self, registry: TaxonomyRegistry
    ) -> None:
        profile = make_profile(
            "sharded", framework="transformers",
            files=["config.json", "tokenizer.json",
                   "model-00001-of-00002.safetensors",
                   "model-00002-of-00002.safetensors"],
            model={"model_type": "llm", "param_count": 8_190_735_360,
                   "param_count_is_exact": True, "quantization": "Q4_K_M"},
        )
        stats = registry.statistics(profile)

        assert stats["shards"] == "2/2"
        assert stats["tokenizer"] is True
        assert stats["parameters"] == 8_190_735_360
        assert stats["parameters_exact"] is True
        assert stats["weight_formats"] == ["safetensors"]

    def test_a_failing_provider_does_not_lose_the_others(self) -> None:
        registry = TaxonomyRegistry()

        def broken(profile: AssetProfile) -> dict[str, object]:
            raise RuntimeError("plugin bug")

        registry.add_statistic(broken, name="broken")
        registry.add_statistic(lambda profile: {"kept": 1}, name="good")

        assert registry.statistics(make_profile("x")) == {"kept": 1}


class TestVocabulary:
    def test_sections_categories_and_tasks_are_all_populated(
        self, registry: TaxonomyRegistry
    ) -> None:
        assert len(registry.sections()) >= 4
        assert len(registry.categories()) >= 25
        assert len(registry.tasks()) >= 50
        assert len(registry.domains()) >= 15

    def test_every_category_belongs_to_a_known_section(
        self, registry: TaxonomyRegistry
    ) -> None:
        known = {section.id for section in registry.sections()}

        for category in registry.categories():
            assert category.section in known, category.id

    def test_every_advertised_alias_resolves(self, registry: TaxonomyRegistry) -> None:
        for alias in registry.known_aliases():
            assert registry.resolve_alias(alias) is not None, alias

    def test_section_and_domain_names_work_as_selectors(
        self, registry: TaxonomyRegistry
    ) -> None:
        # Resolved live rather than stored, so a plugin adding a dataset category widens
        # "datasets" without anyone remembering to extend a list.
        datasets = registry.resolve_alias("datasets")
        assert datasets is not None and "detection_dataset" in datasets

        vision = registry.resolve_alias("document_ai")
        assert vision is not None and "ocr" in vision

    def test_unknown_selector_returns_none(self, registry: TaxonomyRegistry) -> None:
        # None rather than an empty tuple, so the CLI can tell "no such category" from
        # "that category is empty".
        assert registry.resolve_alias("banana") is None

    def test_alias_spelling_variants_are_equivalent(
        self, registry: TaxonomyRegistry
    ) -> None:
        assert (
            registry.resolve_alias("object_detection")
            == registry.resolve_alias("object-detection")
            == registry.resolve_alias("Object Detection")
        )

    def test_sections_are_registered_not_assumed(self) -> None:
        registry = TaxonomyRegistry()
        registry.add_section(Section(id="custom", label="Custom", order=1))

        assert registry.section("custom").label == "Custom"
