"""CLI: agi-eval run --benchmark hellaswag --model gpt-4o"""

from __future__ import annotations

import json
import os
import sys

import argparse

from .backends import MockBackend, OpenAIBackend
from .benchmarks import BENCHMARKS
from .runner import BenchmarkSuite
from .tracker import ResultTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agi-eval",
        description="AGI Eval Sandbox — run standardized LLM benchmarks",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run a benchmark")
    run_p.add_argument("--benchmark", "-b", default="all",
                       help=f"Benchmark name or 'all'. Options: {', '.join(BENCHMARKS)}")
    run_p.add_argument("--model", "-m", default="mock-model",
                       help="Model identifier (e.g. gpt-4o, mistral-7b)")
    run_p.add_argument("--api-key", dest="api_key",
                       help="API key (or set OPENAI_API_KEY env var)")
    run_p.add_argument("--base-url", dest="base_url",
                       default="https://api.openai.com/v1",
                       help="OpenAI-compatible base URL")
    run_p.add_argument("--limit", "-n", type=int, default=None,
                       help="Max questions per benchmark")
    run_p.add_argument("--mock", action="store_true",
                       help="Use mock backend (no API calls)")
    run_p.add_argument("--results-file", dest="results_file",
                       default="results.json",
                       help="Path to results JSON file")
    run_p.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    # ---- history ----
    hist_p = sub.add_parser("history", help="Show run history")
    hist_p.add_argument("--results-file", dest="results_file",
                        default="results.json")

    # ---- list ----
    sub.add_parser("list", help="List available benchmarks")

    return parser


def cmd_run(args) -> int:
    # Pick backend
    if args.mock or args.model == "mock-model":
        backend = MockBackend(model=args.model)
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: --api-key or OPENAI_API_KEY required for non-mock runs.", file=sys.stderr)
            return 1
        backend = OpenAIBackend(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
        )

    tracker = ResultTracker(args.results_file)
    suite = BenchmarkSuite(backend=backend, tracker=tracker)

    if args.benchmark.lower() == "all":
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    all_results = {}
    for bname in benchmarks:
        print(f"Running {bname}...", file=sys.stderr)
        try:
            result = suite.run(bname, limit=args.limit)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue
        all_results[bname] = result
        if not args.json:
            pct = result.accuracy * 100
            print(
                f"  {bname:15s}  {result.correct}/{result.questions}  "
                f"acc={pct:.1f}%  avg_latency={result.latency_ms:.0f}ms"
            )

    if args.json:
        output = {
            name: {
                "benchmark": r.benchmark,
                "questions": r.questions,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "latency_ms": r.latency_ms,
            }
            for name, r in all_results.items()
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_history(args) -> int:
    tracker = ResultTracker(args.results_file)
    history = tracker.load()
    if not history:
        print("No results yet.")
        return 0
    for entry in history:
        print(
            f"{entry['timestamp'][:19]}  model={entry['model']:20s}  "
            f"bench={entry['benchmark']:12s}  acc={entry['accuracy']*100:.1f}%  "
            f"lat={entry['latency_ms']:.0f}ms"
        )
    return 0


def cmd_list(_args) -> int:
    print("Available benchmarks:")
    descriptions = {
        "hellaswag": "HellaSwag-mini — 20 multiple-choice sentence completion questions",
        "gsm8k":     "GSM8K-mini    — 10 grade-school math problems",
        "truthfulqa": "TruthfulQA-mini — 10 factual/trick questions",
        "codeeval":  "CodeEval-mini — 5 Python coding tasks (execution-based)",
    }
    for name, desc in descriptions.items():
        print(f"  {name:12s}  {desc}")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "history":
        sys.exit(cmd_history(args))
    elif args.command == "list":
        sys.exit(cmd_list(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
