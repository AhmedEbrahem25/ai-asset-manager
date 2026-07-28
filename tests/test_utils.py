"""Tests for hashing, path and formatting helpers."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from ai_asset_manager.backend.utils.hashing import (
    HashCancelled,
    combine_hashes,
    fingerprint_entries,
    quick_signature,
    sha256_file,
)
from ai_asset_manager.backend.utils.humanize import (
    format_bytes,
    format_count,
    format_duration,
    parse_size,
)
from ai_asset_manager.backend.utils.paths import (
    classify_extension,
    get_drive,
    is_incomplete_marker,
    normalize_path,
    safe_relpath,
    shorten_path,
)


class TestQuickSignature:
    def test_identical_content_matches(self, tmp_path: Path) -> None:
        payload = os.urandom(3 * 1024 * 1024)
        (tmp_path / "a.bin").write_bytes(payload)
        (tmp_path / "b.bin").write_bytes(payload)

        assert quick_signature(tmp_path / "a.bin") == quick_signature(tmp_path / "b.bin")

    def test_different_content_differs(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"a" * 4096)
        (tmp_path / "b.bin").write_bytes(b"b" * 4096)

        assert quick_signature(tmp_path / "a.bin") != quick_signature(tmp_path / "b.bin")

    def test_same_head_and_tail_but_different_length_differs(self, tmp_path: Path) -> None:
        # Size is mixed into the digest precisely so this case cannot collide.
        head, tail = b"H" * 1024, b"T" * 1024
        (tmp_path / "short.bin").write_bytes(head + b"\x00" * 1024 + tail)
        (tmp_path / "long.bin").write_bytes(head + b"\x00" * 4096 + tail)

        signatures = {
            quick_signature(tmp_path / "short.bin", chunk_bytes=1024, min_full_hash_bytes=0),
            quick_signature(tmp_path / "long.bin", chunk_bytes=1024, min_full_hash_bytes=0),
        }
        assert len(signatures) == 2

    def test_differing_middle_is_caught_by_full_hash_of_small_files(
        self, tmp_path: Path
    ) -> None:
        # Files at or below the threshold are hashed whole, so a middle-only difference
        # is detected rather than missed by head/tail sampling.
        (tmp_path / "a.bin").write_bytes(b"X" * 100 + b"A" + b"X" * 100)
        (tmp_path / "b.bin").write_bytes(b"X" * 100 + b"B" + b"X" * 100)

        assert quick_signature(tmp_path / "a.bin") != quick_signature(tmp_path / "b.bin")

    def test_cancellation_is_honoured(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"\x00" * 4096)
        event = threading.Event()
        event.set()

        with pytest.raises(HashCancelled):
            quick_signature(tmp_path / "a.bin", cancel_event=event)


class TestSha256:
    def test_matches_known_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")

        assert sha256_file(path) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_empty_file_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "e.bin"
        path.write_bytes(b"")

        assert sha256_file(path) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_cancellation_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"\x00" * (4 * 1024 * 1024))
        event = threading.Event()
        event.set()

        with pytest.raises(HashCancelled):
            sha256_file(path, cancel_event=event)


class TestFingerprint:
    def test_is_order_independent(self) -> None:
        forward = fingerprint_entries([("a.bin", 10, 100.0), ("b.bin", 20, 200.0)])
        reverse = fingerprint_entries([("b.bin", 20, 200.0), ("a.bin", 10, 100.0)])

        assert forward == reverse

    def test_changes_when_size_changes(self) -> None:
        before = fingerprint_entries([("a.bin", 10, 100.0)])
        after = fingerprint_entries([("a.bin", 11, 100.0)])

        assert before != after

    def test_changes_when_mtime_changes(self) -> None:
        before = fingerprint_entries([("a.bin", 10, 100.0)])
        after = fingerprint_entries([("a.bin", 10, 200.0)])

        assert before != after

    def test_ignores_subsecond_mtime_drift(self) -> None:
        # Filesystems disagree on sub-second precision; a fingerprint that flickered
        # between scans would defeat the incrementality it exists to enable.
        before = fingerprint_entries([("a.bin", 10, 100.4)])
        after = fingerprint_entries([("a.bin", 10, 100.9)])

        assert before == after

    def test_empty_input_is_stable(self) -> None:
        assert fingerprint_entries([]) == fingerprint_entries([])

    def test_combine_hashes_is_order_independent(self) -> None:
        assert combine_hashes(["aa", "bb"]) == combine_hashes(["bb", "aa"])


class TestHumanize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"), (1536, "1.5 KiB"),
         (1024**3, "1.0 GiB"), (5 * 1024**4, "5.0 TiB")],
    )
    def test_format_bytes_binary(self, value: int, expected: str) -> None:
        assert format_bytes(value) == expected

    def test_format_bytes_decimal(self) -> None:
        assert format_bytes(1_000_000_000, binary=False) == "1.0 GB"

    def test_format_bytes_negative(self) -> None:
        assert format_bytes(-1024) == "-1.0 KiB"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("512", 512), ("1kb", 1000), ("1KiB", 1024), ("10GB", 10_000_000_000),
         ("1.5 GiB", int(1.5 * 1024**3)), ("2TB", 2 * 1000**4)],
    )
    def test_parse_size(self, text: str, expected: int) -> None:
        assert parse_size(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "10 quorks", "-5GB"])
    def test_parse_size_rejects_nonsense(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_size(text)

    def test_round_trips_with_format_bytes(self) -> None:
        assert parse_size(format_bytes(1024**3)) == 1024**3

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(512, "512"), (1500, "1.5K"), (494_032_768, "494.0M"), (8_190_735_360, "8.2B")],
    )
    def test_format_count(self, value: int, expected: str) -> None:
        assert format_count(value) == expected

    def test_format_duration(self) -> None:
        assert format_duration(0.4) == "0.4s"
        assert format_duration(95) == "1m 35s"
        assert format_duration(3725) == "1h 2m"


class TestPaths:
    def test_normalize_uppercases_windows_drive(self) -> None:
        if os.name != "nt":
            pytest.skip("drive-letter behaviour is Windows-specific")

        assert normalize_path("c:\\models\\x") == normalize_path("C:\\Models\\..\\models\\x")

    def test_normalize_is_absolute(self, tmp_path: Path) -> None:
        assert os.path.isabs(normalize_path("."))

    def test_get_drive(self) -> None:
        if os.name != "nt":
            pytest.skip("drive-letter behaviour is Windows-specific")

        assert get_drive("D:\\Models\\llama") == "D:"

    def test_safe_relpath_uses_forward_slashes(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"

        assert safe_relpath(target, tmp_path) == "a/b/c.txt"

    def test_safe_relpath_falls_back_across_drives(self) -> None:
        if os.name != "nt":
            pytest.skip("cross-drive paths are Windows-specific")

        # os.path.relpath raises for paths on different drives; the helper must not.
        assert safe_relpath("D:\\a\\b.txt", "C:\\root")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("m.safetensors", "payload"), ("a.jpg", "image"), ("v.mp4", "video"),
         ("s.wav", "audio"), ("d.jsonl", "text"), ("x.unknownext", None)],
    )
    def test_classify_extension(self, name: str, expected: str | None) -> None:
        assert classify_extension(name) == expected

    @pytest.mark.parametrize(
        "name",
        ["model.safetensors.incomplete", "weights.bin.part", "file.download", "x.tmp"],
    )
    def test_detects_incomplete_markers(self, name: str) -> None:
        assert is_incomplete_marker(name)

    def test_complete_file_is_not_an_incomplete_marker(self) -> None:
        assert not is_incomplete_marker("model.safetensors")

    def test_shorten_path_respects_limit(self) -> None:
        long_path = "/a/very/long/path/that/keeps/going/to/a/model/file.safetensors"

        assert len(shorten_path(long_path, 30)) <= 33
        assert shorten_path("short.txt", 30) == "short.txt"
