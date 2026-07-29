"""Explanations: why a detector concluded what it did."""

from __future__ import annotations

import json
from pathlib import Path

from ai_asset_manager.backend.detectors.explain import explanation_of, render_explanation


def test_every_record_carries_an_explanation(tmp_path: Path, pipeline):
    repo = tmp_path / "qwen"
    repo.mkdir()
    (repo / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896})
    )
    (repo / "tokenizer_config.json").write_text("{}")
    (repo / "model.safetensors").write_bytes(b"\0" * (2 * 1024 * 1024))

    records = pipeline.scan_root(tmp_path)

    assert len(records) == 1
    payload = records[0].evidence["explanation"]
    assert payload["detector"] == "hf_repo"
    assert 0.0 < payload["confidence"] <= 1.0
    assert payload["signals"], "an explanation with no evidence explains nothing"


def test_explanation_includes_what_the_parsers_found(tmp_path: Path, pipeline):
    repo = tmp_path / "bert"
    repo.mkdir()
    (repo / "config.json").write_text(
        json.dumps({"architectures": ["BertModel"], "hidden_size": 768, "num_hidden_layers": 12})
    )
    (repo / "model.safetensors").write_bytes(b"\0" * (2 * 1024 * 1024))

    records = pipeline.scan_root(tmp_path)
    signals = records[0].evidence["explanation"]["signals"]

    assert any("BertModel" in signal for signal in signals)


def test_detector_phrasing_is_preserved_verbatim():
    """A detector that phrased its own reasoning is quoted rather than reconstructed."""
    heading, signals = explanation_of(
        {
            "explanation": {
                "summary": "ocr",
                "confidence": 0.97,
                "signals": ["18 packet capture file(s)", "marker file: config.json"],
            }
        }
    )

    assert heading == "OCR — confidence 97%"
    assert signals == ["18 packet capture file(s)", "marker file: config.json"]


def test_the_explanation_agrees_with_the_detector_confidence():
    """One verdict, one percentage, however many places show it.

    `aam show 252` rendered "Detected by cybersecurity_dataset (82% confidence)" directly
    above "Network Capture — confidence 83%". Both were the same 0.825: the explanation
    rounded it to two places on the way in and the display rounded again on the way out.
    """
    confidence = 0.825
    heading, _ = explanation_of(
        {"explanation": {"summary": "network capture", "confidence": confidence}}
    )

    assert heading.endswith(f"{round(confidence * 100)}%")


def test_missing_explanation_is_not_an_error():
    assert explanation_of({}) is None
    assert render_explanation({}) == []


def test_rendered_lines_are_a_checklist():
    lines = render_explanation(
        {"explanation": {"summary": "model archive", "confidence": 0.95,
                         "signals": ["config.json", "3 weight file(s)"]}}
    )

    assert lines[0].startswith("Model Archive")
    assert lines[1:] == ["  ✓ config.json", "  ✓ 3 weight file(s)"]
