"""Deriving vendor, product, task and a usable name from a path."""

from __future__ import annotations

import pytest

from ai_asset_manager.backend.identity import identify, is_generic_name
from ai_asset_manager.backend.identity.naming import prettify


@pytest.mark.parametrize(
    "name",
    ["model", "Model", "model.onnx", "model_quantized", "pytorch_model", "weights",
     "best", "1.2.3", "20250218", "a3f9c1d20b4e5f6a7b8c9d0e1f2a3b4c", ""],
)
def test_generic_names_are_recognised(name):
    assert is_generic_name(name)


@pytest.mark.parametrize(
    "name",
    ["yolo11n", "resnet18-f37072fd", "Qwen2.5-0.5B-Instruct", "large-v3-turbo",
     "arabic_PP-OCRv4_rec_infer", "WordCombinedFloatieLreOnlineV3"],
)
def test_meaningful_names_are_left_alone(name):
    assert not is_generic_name(name)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            r"C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\1.0.3\model.tflite",
            "Chrome ScreenAI OCR Model",
        ),
        (
            r"C:\Users\pc\AppData\Local\Microsoft\Edge\User Data"
            r"\OptimizationGuidePredictionModels\4\model.tflite",
            "Edge Optimization Guide Model",
        ),
        (
            r"C:\Users\pc\.vscode\extensions\devsense.intelli-php-vscode-1.2\model.onnx",
            "VS Code Completion",
        ),
        (
            r"C:\Users\pc\AppData\Roaming\Zoom\bin\aomhost\virtual_background\model.onnx",
            "Zoom Background Segmentation",
        ),
        (
            r"C:\Users\pc\AppData\Local\Programs\cursor\resources\app\out\model.onnx",
            "Cursor Model",
        ),
    ],
)
def test_generic_names_become_meaningful(path, expected):
    assert identify(path, name="model", is_single_file=True).display_name == expected


def test_vendor_and_source_are_recorded_even_when_the_name_is_fine():
    identity = identify(
        r"C:\Users\pc\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct",
        name="Qwen/Qwen2.5-0.5B-Instruct",
    )

    assert identity.display_name is None, "a good name must not be replaced"
    assert identity.vendor == "HuggingFace"
    assert identity.source == "huggingface"


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (r"C:\Users\pc\.ollama\models\manifests\registry.ollama.ai\library\qwen3\8b", "ollama"),
        (r"C:\Users\pc\.cache\torch\hub\checkpoints\resnet18.pth", "pytorch_hub"),
        (r"C:\Users\pc\.cache\whisper\large-v3-turbo.pt", "whisper"),
        (r"C:\Users\pc\AppData\Local\datalab\datalab\Cache\models\layout\2025_02_18", "surya"),
        (r"F:\Downloads\yolo11n.pt", "manual"),
        (r"F:\scratch\thing.pt", "unknown"),
    ],
)
def test_software_source_is_identified(path, source):
    assert identify(path, name="x", is_single_file=True).source == source


def test_an_extension_id_does_not_outrank_the_application_holding_it():
    """A VS Code extension installed in Cursor is Cursor's, not Microsoft's.

    Both Cursor and Antigravity host VS Code extensions, so the deepest match in the path
    is an extension identifier that names the wrong application.
    """
    identity = identify(
        r"C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\7cd"
        r"\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\A\B\model.tflite",
        name="model",
        is_single_file=True,
    )
    assert identity.product == "Cursor"
    assert identity.display_name == "Cursor Optimization Guide Model"


def test_a_locale_folder_is_not_a_component():
    """`.../ModelResources/en-us/model.onnx` must not become "Microsoft En Us Model"."""
    identity = identify(
        r"C:\Users\pc\AppData\Local\Microsoft\AugLoop\Word\2.1"
        r"\CloudResources\ModelResources\en-us\model.onnx",
        name="model",
        is_single_file=True,
    )
    assert identity.display_name == "Microsoft Text Prediction"


def test_the_task_is_not_repeated_in_different_words():
    """"VS Code Reranker Reranking" reads as a mistake even though both halves are right."""
    identity = identify(
        r"C:\Users\pc\.vscode\extensions\devsense.phptools-1.0\out\models\deeprerank"
        r"\model.onnx",
        name="model",
        is_single_file=True,
    )
    assert identity.display_name == "VS Code Reranker"


def test_a_file_extension_is_not_a_vendor():
    """A regression: matching `onnx` in `model.onnx` attributed every model to a runtime."""
    identity = identify(r"F:\scratch\model.onnx", name="model", is_single_file=True)
    assert identity.vendor is None


def test_a_bare_vendor_is_not_a_name():
    """A vendor with nothing beside it says nothing that "model" did not.

    "Google Model" is not an improvement, so no rename is offered. The vendor is still
    recorded, because that part is genuinely known.
    """
    identity = identify(
        r"C:\Program Files\Google\model.tflite", name="model", is_single_file=True
    )
    assert identity.vendor == "Google"
    assert identity.component is None
    assert identity.display_name is None


def test_a_bare_component_is_a_name_even_without_a_vendor():
    r"""Two real assets kept a useless name because only a component was known.

    `catboost_info\test` was catalogued as "test". The guard that refuses a bare vendor
    was refusing a bare component too, though the two are not alike: nobody knows what
    "test" is, and "CatBoost Info Test" is exactly what it is.
    """
    identity = identify(
        r"C:\Users\pc\OneDrive\GCI\competition\catboost_info\test", name="test"
    )
    assert identity.vendor is None
    assert identity.component == "Catboost Info"
    # The discriminating word survives: `catboost_info` also holds `learn`, and collapsing
    # both to "Catboost Info" would recreate the duplicate-row problem one level up.
    assert identity.display_name == "Catboost Info Test"

    sibling = identify(
        r"C:\Users\pc\OneDrive\GCI\competition\catboost_info\learn", name="learn"
    )
    assert sibling.display_name != identity.display_name


def test_a_digest_is_not_kept_as_a_qualifier():
    """A hash distinguishes without informing, so it is dropped rather than appended."""
    identity = identify(
        r"F:\project\thorn-nlp\data\cache\coastalcph___lex_glue\unfair_tos"
        r"\0.0.0\c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
        name="c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
    )
    assert identity.display_name == "Unfair Tos"


def test_a_bare_component_is_not_called_a_model():
    r"""With no vendor and no product there is no evidence the asset is a model.

    `catboost_info\test` holds training metrics. Appending "Model" would state something
    the derivation does not know.
    """
    identity = identify(r"D:\runs\catboost_info\test", name="test")
    assert identity.display_name is not None
    assert not identity.display_name.endswith("Model")


def test_an_untabulated_folder_still_beats_model():
    identity = identify(
        r"C:\Users\pc\AppData\Roaming\Zoom\bin\frames_processor\model.onnx",
        name="model",
        is_single_file=True,
    )
    assert identity.display_name == "Zoom Frames Processor Model"


def test_identity_records_what_it_matched_on():
    identity = identify(
        r"C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\model.tflite",
        name="model",
        is_single_file=True,
    )
    assert any("Chrome" in signal for signal in identity.signals)
    assert any("screen_ai" in signal for signal in identity.signals)


def test_prettify_expands_known_acronyms():
    assert prettify("text_recognition") == "Text Recognition"
    assert prettify("screen_ai") == "Screen AI"
    assert prettify("ocr-det") == "OCR Det"


def test_identity_reaches_the_asset_record(tmp_path, pipeline, monkeypatch):
    """The end-to-end path: a generic weight file comes out of the pipeline named."""
    target = tmp_path / "AppData" / "Local" / "Google" / "Chrome" / "User Data" / "screen_ai"
    target.mkdir(parents=True)
    (target / "model.tflite").write_bytes(b"TFL3" + b"\0" * (2 * 1024 * 1024))

    records = pipeline.scan_root(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.name == "model", "the name on disk is preserved"
    assert record.display_name == "Chrome ScreenAI OCR Model"
    assert record.evidence["identity"]["source"] == "chrome"
