# agi-eval-sandbox

Run standardized LLM benchmarks against any OpenAI-compatible endpoint.

```
agi-eval run --benchmark hellaswag --model gpt-4o
```

## What it does

Evaluates language models on four built-in benchmarks with a single CLI command. Results are tracked in a local JSON file for comparison across runs.

Designed to work with **any OpenAI-compatible API** — OpenAI, Together AI, Groq, Ollama, LM Studio, vLLM, etc.

---

## Benchmarks

| Name | Questions | Type | Scoring |
|------|-----------|------|---------|
| `hellaswag` | 20 | Multiple-choice sentence completion | Exact letter match (A/B/C/D) |
| `gsm8k` | 10 | Grade-school math | Numeric answer extraction |
| `truthfulqa` | 10 | Factual / anti-hallucination | Keyword/numeric match |
| `codeeval` | 5 | Python coding tasks | Execution-based (all test cases must pass) |

All questions are hardcoded — no external dataset downloads required.

---

## Quick Start

```bash
pip install -e .
```

### Run with a real model

```bash
export OPENAI_API_KEY=sk-...

# Single benchmark
agi-eval run --benchmark hellaswag --model gpt-4o

# All benchmarks
agi-eval run --benchmark all --model gpt-4o

# Limit questions (useful for quick smoke tests)
agi-eval run --benchmark all --model gpt-4o --limit 5
```

### Run against a local / custom endpoint

```bash
agi-eval run \
  --benchmark hellaswag \
  --model mistral-7b \
  --api-key YOUR_KEY \
  --base-url http://localhost:11434/v1
```

### Run with mock backend (no API needed)

```bash
agi-eval run --benchmark all --mock
```

### JSON output

```bash
agi-eval run --benchmark hellaswag --model gpt-4o --json
```

### View run history

```bash
agi-eval history
```

### List available benchmarks

```bash
agi-eval list
```

---

## Architecture

```
src/agi_eval/
├── benchmarks.py   # Question banks + evaluation logic
├── backends.py     # MockBackend + OpenAIBackend
├── runner.py       # BenchmarkSuite orchestrator
├── tracker.py      # ResultTracker (JSON persistence)
└── cli.py          # CLI entry point

tests/
├── test_benchmarks.py
├── test_backends.py
├── test_runner.py
├── test_tracker.py
└── test_cli.py
```

**Key classes:**
- `BenchmarkSuite` — runs one or all benchmarks, records results
- `MockBackend` — deterministic answers for CI/testing
- `OpenAIBackend` — any OpenAI-compatible REST API
- `ResultTracker` — append-only JSON log of every run

---

## Development

```bash
# Install in editable mode
pip install -e .

# Run tests (uses conda python by default)
~/anaconda3/bin/python -m pytest tests/ -v
```

Zero runtime dependencies — stdlib only. Python 3.10+.

---

## Evaluation framing

These mini-benchmarks are calibration tools, not definitive scores. Use them to:

- **Catch regressions** when swapping models or fine-tunes
- **Compare endpoints** (OpenAI vs local vs distilled)
- **Verify a deployment** is behaving as expected

For production evaluation needs, see [HELM](https://crfm.stanford.edu/helm/), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), or [OpenAI Evals](https://github.com/openai/evals).

---

## License

MIT
