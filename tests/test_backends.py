"""Tests for backends."""

import pytest
from agi_eval.backends import MockBackend, CompletionResponse


class TestMockBackend:
    def test_returns_default_answer(self):
        b = MockBackend(default_answer="C")
        resp = b.complete("some prompt")
        assert resp.text == "C"

    def test_per_question_override(self):
        b = MockBackend(answers={"q1": "A"}, default_answer="B")
        assert b.complete("p", question_id="q1").text == "A"
        assert b.complete("p", question_id="q2").text == "B"

    def test_model_name(self):
        b = MockBackend(model="test-model")
        assert b.model_name == "test-model"

    def test_latency_set(self):
        b = MockBackend(latency_ms=42.0)
        assert b.complete("hi").latency_ms == 42.0

    def test_returns_completion_response(self):
        b = MockBackend()
        resp = b.complete("test")
        assert isinstance(resp, CompletionResponse)
