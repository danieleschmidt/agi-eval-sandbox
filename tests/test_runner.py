"""Tests for BenchmarkSuite runner."""

import tempfile
from pathlib import Path

import pytest
from agi_eval.backends import MockBackend
from agi_eval.runner import BenchmarkSuite
from agi_eval.tracker import ResultTracker


@pytest.fixture
def mock_suite(tmp_path):
    backend = MockBackend(default_answer="B")
    tracker = ResultTracker(tmp_path / "results.json")
    return BenchmarkSuite(backend=backend, tracker=tracker)


class TestBenchmarkSuite:
    def test_available_benchmarks(self, mock_suite):
        benches = mock_suite.available_benchmarks()
        assert "hellaswag" in benches
        assert "gsm8k" in benches
        assert "truthfulqa" in benches
        assert "codeeval" in benches

    def test_run_hellaswag_returns_result(self, mock_suite):
        result = mock_suite.run("hellaswag")
        assert result.benchmark == "hellaswag"
        assert result.questions == 20
        assert 0.0 <= result.accuracy <= 1.0

    def test_run_with_limit(self, mock_suite):
        result = mock_suite.run("gsm8k", limit=3)
        assert result.questions == 3

    def test_run_records_to_tracker(self, mock_suite):
        mock_suite.run("hellaswag")
        history = mock_suite.tracker.load()
        assert len(history) == 1
        assert history[0]["benchmark"] == "hellaswag"

    def test_run_all_returns_4_benchmarks(self, mock_suite):
        results = mock_suite.run_all(limit=2)
        assert len(results) == 4

    def test_per_question_populated(self, mock_suite):
        result = mock_suite.run("gsm8k", limit=2)
        assert len(result.per_question) == 2
        for pq in result.per_question:
            assert "id" in pq
            assert "correct" in pq
            assert "latency_ms" in pq

    def test_latency_recorded(self, mock_suite):
        result = mock_suite.run("hellaswag", limit=5)
        assert result.latency_ms >= 0

    def test_suite_without_tracker(self):
        backend = MockBackend(default_answer="B")
        suite = BenchmarkSuite(backend=backend)
        result = suite.run("hellaswag", limit=3)
        assert result.questions == 3

    def test_mock_with_perfect_hellaswag_answers(self):
        """Perfect answers on hellaswag should give 100% accuracy."""
        from agi_eval.benchmarks import HellaSwagMini
        bench = HellaSwagMini()
        answers = {q.id: q.correct_answer for q in bench.questions()}
        backend = MockBackend(answers=answers)
        suite = BenchmarkSuite(backend=backend)
        result = suite.run("hellaswag")
        assert result.accuracy == 1.0
        assert result.correct == 20

    def test_mock_with_all_wrong_answers(self):
        """All wrong answers should give 0% accuracy on hellaswag."""
        from agi_eval.benchmarks import HellaSwagMini
        bench = HellaSwagMini()
        # Use wrong answer for every question
        answers = {}
        for q in bench.questions():
            wrong = next(l for l in "ABCD" if l != q.correct_answer)
            answers[q.id] = wrong
        backend = MockBackend(answers=answers)
        suite = BenchmarkSuite(backend=backend)
        result = suite.run("hellaswag")
        assert result.accuracy == 0.0

    def test_codeeval_with_correct_code(self):
        """Mock that returns correct code snippets should pass codeeval."""
        correct_codes = {
            "codeeval_0": "```python\ndef add(a, b):\n    return a + b\n```",
            "codeeval_1": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
            "codeeval_2": "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
            "codeeval_3": "```python\ndef factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)\n```",
            "codeeval_4": "```python\ndef fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n    return str(n)\n```",
        }
        backend = MockBackend(answers=correct_codes)
        suite = BenchmarkSuite(backend=backend)
        result = suite.run("codeeval")
        assert result.accuracy == 1.0
