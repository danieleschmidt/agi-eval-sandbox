"""Tests for ResultTracker."""

import json
import tempfile
from pathlib import Path

import pytest
from agi_eval.tracker import ResultTracker


@pytest.fixture
def tmp_tracker(tmp_path):
    return ResultTracker(tmp_path / "results.json")


class TestResultTracker:
    def test_record_creates_file(self, tmp_tracker):
        tmp_tracker.record("gpt-4o", "hellaswag", 0.85, 120.0)
        assert tmp_tracker.path.exists()

    def test_record_fields(self, tmp_tracker):
        entry = tmp_tracker.record("gpt-4o", "hellaswag", 0.85, 120.0, {"questions": 20})
        assert entry["model"] == "gpt-4o"
        assert entry["benchmark"] == "hellaswag"
        assert entry["accuracy"] == 0.85
        assert entry["latency_ms"] == 120.0
        assert entry["metadata"]["questions"] == 20
        assert "timestamp" in entry

    def test_load_empty(self, tmp_tracker):
        assert tmp_tracker.load() == []

    def test_multiple_records(self, tmp_tracker):
        tmp_tracker.record("m1", "b1", 0.5, 10.0)
        tmp_tracker.record("m2", "b2", 0.7, 20.0)
        history = tmp_tracker.load()
        assert len(history) == 2

    def test_summary_deduplicates(self, tmp_tracker):
        tmp_tracker.record("gpt-4", "gsm8k", 0.5, 100.0)
        tmp_tracker.record("gpt-4", "gsm8k", 0.8, 90.0)  # same key, newer
        summary = tmp_tracker.summary()
        assert len(summary) == 1
        assert summary[0]["accuracy"] == 0.8

    def test_clear(self, tmp_tracker):
        tmp_tracker.record("m1", "b1", 0.5, 10.0)
        tmp_tracker.clear()
        assert tmp_tracker.load() == []

    def test_corrupt_file_returns_empty(self, tmp_tracker):
        tmp_tracker.path.write_text("not json{{{")
        assert tmp_tracker.load() == []
