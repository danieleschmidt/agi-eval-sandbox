"""BenchmarkSuite — orchestrates benchmark runs and tracks results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend
from .benchmarks import Benchmark, BenchmarkResult, Question, BENCHMARKS, get_benchmark
from .tracker import ResultTracker


@dataclass
class RunConfig:
    model: str
    benchmark: str
    limit: int | None = None          # Max questions per benchmark
    tracker_path: str = "results.json"


class BenchmarkSuite:
    """
    Collection of evaluation tasks.

    Usage::

        suite = BenchmarkSuite(backend)
        result = suite.run("hellaswag")
    """

    def __init__(self, backend: Backend, tracker: ResultTracker | None = None):
        self.backend = backend
        self.tracker = tracker

    def available_benchmarks(self) -> list[str]:
        return list(BENCHMARKS.keys())

    def run(self, benchmark_name: str, limit: int | None = None) -> BenchmarkResult:
        """Run a single benchmark and return results."""
        bench = get_benchmark(benchmark_name)
        questions = bench.questions()
        if limit:
            questions = questions[:limit]

        correct = 0
        total_latency = 0.0
        per_question: list[dict[str, Any]] = []

        for q in questions:
            resp = self.backend.complete(
                _build_prompt(bench, q),
                question_id=q.id,
            )
            total_latency += resp.latency_ms
            is_correct = bench.evaluate(q, resp.text)
            if is_correct:
                correct += 1
            per_question.append({
                "id": q.id,
                "correct": is_correct,
                "response": resp.text[:200],
                "latency_ms": resp.latency_ms,
            })

        n = len(questions)
        result = BenchmarkResult(
            benchmark=bench.name,
            questions=n,
            correct=correct,
            accuracy=correct / n if n else 0.0,
            latency_ms=total_latency / n if n else 0.0,
            per_question=per_question,
        )

        if self.tracker:
            self.tracker.record(
                model=self.backend.model_name,
                benchmark=result.benchmark,
                accuracy=result.accuracy,
                latency_ms=result.latency_ms,
                metadata={
                    "questions": result.questions,
                    "correct": result.correct,
                },
            )

        return result

    def run_all(self, limit: int | None = None) -> dict[str, BenchmarkResult]:
        """Run all built-in benchmarks."""
        return {name: self.run(name, limit=limit) for name in self.available_benchmarks()}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(bench: Benchmark, question: Question) -> str:
    from .benchmarks import HellaSwagMini, GSM8KMini, TruthfulQAMini, CodeEvalMini

    if isinstance(bench, HellaSwagMini):
        return (
            "Choose the most likely completion. Reply with only the letter A, B, C, or D.\n\n"
            + question.prompt
        )
    if isinstance(bench, GSM8KMini):
        return (
            "Solve the following math problem. Show your work, then state the final answer as a number.\n\n"
            + question.prompt
        )
    if isinstance(bench, TruthfulQAMini):
        return (
            "Answer the following question truthfully and concisely.\n\n"
            + question.prompt
        )
    if isinstance(bench, CodeEvalMini):
        return (
            "Write a Python function as described. Return only the function code in a ```python block.\n\n"
            + question.prompt
        )
    return question.prompt
