# agi-eval-sandbox

[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/agi-eval-sandbox/ci.yml?branch=main)](https://github.com/your-org/agi-eval-sandbox/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/docker/pulls/your-org/agi-eval-sandbox)](https://hub.docker.com/r/your-org/agi-eval-sandbox)
[![Notebooks](https://img.shields.io/badge/notebooks-20+-orange.svg)](notebooks/)

One-click evaluation environment bundling DeepEval, HELM-Lite, MT-Bench, and your custom benchmarks. Automatically track model improvements with CI badges and comprehensive dashboards.

## 🎯 Key Features

- **Unified Evaluation Suite**: 15+ benchmarks in a single environment
- **One-Click Setup**: Docker or cloud deployment in minutes
- **CI/CD Integration**: Auto-generate badges and reports for every commit
- **Longitudinal Tracking**: Monitor model drift and improvements over time
- **Custom Eval Builder**: Drag-and-drop interface for creating benchmarks
- **A/B Testing**: Compare models side-by-side with statistical significance

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Included Benchmarks](#included-benchmarks)
- [Running Evaluations](#running-evaluations)
- [Custom Benchmarks](#custom-benchmarks)
- [CI Integration](#ci-integration)
- [Visualization](#visualization)
- [A/B Testing](#ab-testing)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🚀 Installation

### Docker (Recommended)

```bash
# Pull and run the sandbox
docker run -it -p 8888:8888 -p 8080:8080 \
  -v $(pwd)/results:/results \
  your-org/agi-eval-sandbox:latest

# Access Jupyter at http://localhost:8888
# Access Dashboard at http://localhost:8080
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/your-org/agi-eval-sandbox
cd agi-eval-sandbox

# Install with all evaluators
pip install -e ".[all]"

# Start sandbox
agi-eval start
```

### Cloud Deployment

```bash
# Deploy to AWS/GCP/Azure
agi-eval deploy --provider aws --instance-type g4dn.xlarge

# Get access URL
agi-eval status
```

## ⚡ Quick Start

### Evaluate a Model

```python
from agi_eval_sandbox import EvalSuite, Model

# Initialize eval suite
suite = EvalSuite()

# Configure model
model = Model(
    provider="openai",
    name="gpt-4",
    api_key="your-key"
)

# Run standard evaluation
results = suite.evaluate(
    model=model,
    benchmarks=["mmlu", "humaneval", "truthfulqa"],
    save_results=True
)

# View results
print(results.summary())
results.to_dashboard()
```

### Jupyter Notebook Interface

```python
# In Jupyter notebook
from agi_eval_sandbox import NotebookEvaluator

evaluator = NotebookEvaluator()
evaluator.run_interactive()  # Opens GUI interface
```

## 📊 Included Benchmarks

### Core Benchmarks

| Benchmark | Type | Metrics | Description |
|-----------|------|---------|-------------|
| **MMLU** | Knowledge | Accuracy | 57 subjects across STEM, humanities |
| **HumanEval** | Coding | Pass@k | Python programming problems |
| **TruthfulQA** | Truthfulness | Truth%, Info% | Measures truthful generation |
| **MT-Bench** | Conversation | GPT-4 Judge | Multi-turn conversation quality |
| **HellaSwag** | Reasoning | Accuracy | Commonsense reasoning |
| **MATH** | Mathematical | Accuracy | Competition mathematics |
| **GSM8K** | Math Word | Accuracy | Grade school math problems |

### Safety Benchmarks

| Benchmark | Type | Metrics | Description |
|-----------|------|---------|-------------|
| **DeceptionBench** | Deception | DRS Score | From deception-redteam-bench |
| **HallucinationTest** | Factuality | Hallucination% | From hallucination-sentinel |
| **BiasEval** | Bias | Bias Score | Demographic bias detection |
| **ToxicityTest** | Safety | Toxicity% | Harmful content generation |

### Custom Extensions

| Benchmark | Type | Metrics | Description |
|-----------|------|---------|-------------|
| **DomainSpecific** | Custom | Variable | Your domain benchmarks |
| **InternalEvals** | Private | Custom | Company-specific tests |

## 🧪 Running Evaluations

### Configuration File

Create `eval_config.yaml`:

```yaml
model:
  provider: "anthropic"
  name: "claude-3-opus"
  temperature: 0.0
  max_tokens: 2048
  
benchmarks:
  mmlu:
    enabled: true
    subjects: ["all"]  # or specific like ["physics", "history"]
    shots: 5
    
  humaneval:
    enabled: true
    k_values: [1, 10, 100]
    timeout: 10
    
  truthfulqa:
    enabled: true
    model_judge: "gpt-4"
    categories: ["all"]
    
  deception_bench:
    enabled: true
    scenarios: ["sandbagging", "sycophancy"]
    
evaluation:
  parallel: true
  num_workers: 4
  seed: 42
  cache_results: true
  
output:
  format: ["json", "csv", "html"]
  dashboard: true
  badges: true
  
integration:
  wandb:
    enabled: true
    project: "model-evals"
  
  lang_observatory:
    enabled: true
    endpoint: "https://observatory.your-org.com"
```

### Command Line Interface

```bash
# Run evaluation with config
agi-eval run --config eval_config.yaml

# Run specific benchmarks
agi-eval run --model gpt-4 --benchmarks mmlu,humaneval

# Compare multiple models
agi-eval compare --models gpt-4,claude-3,llama-3 --benchmarks all

# Generate report
agi-eval report --run-id latest --format pdf
```

### Programmatic Usage

```python
from agi_eval_sandbox import EvalSuite, BenchmarkConfig

# Fine-grained control
suite = EvalSuite()

# Configure individual benchmarks
mmlu_config = BenchmarkConfig(
    name="mmlu",
    subjects=["physics", "computer_science"],
    few_shot=5,
    temperature=0.0
)

humaneval_config = BenchmarkConfig(
    name="humaneval",
    k_values=[1, 10],
    sandbox_mode=True
)

# Run with configs
results = suite.evaluate(
    model=model,
    configs=[mmlu_config, humaneval_config]
)
```

## 🛠️ Custom Benchmarks

### Creating a Custom Benchmark

```python
from agi_eval_sandbox import CustomBenchmark, Question

class DomainBenchmark(CustomBenchmark):
    """Custom benchmark for your domain."""
    
    def __init__(self):
        super().__init__(name="domain_specific")
        
    def load_questions(self):
        return [
            Question(
                id="q1",
                prompt="What is the optimal strategy for...",
                correct_answer="The optimal strategy is...",
                category="strategy"
            ),
            # Add more questions
        ]
    
    def evaluate_response(self, response, correct_answer):
        # Custom evaluation logic
        score = self.similarity(response, correct_answer)
        return {"score": score, "passed": score > 0.8}

# Register benchmark
suite.register_benchmark(DomainBenchmark())
```

### Benchmark Builder GUI

```python
# Launch interactive builder
from agi_eval_sandbox import BenchmarkBuilder

builder = BenchmarkBuilder()
builder.launch()  # Opens web interface

# Or programmatically
benchmark = builder.create(
    name="customer_service",
    question_source="csv",
    file_path="questions.csv",
    evaluation_method="llm_judge",
    judge_model="gpt-4"
)
```

## 🔄 CI Integration

### GitHub Actions

```yaml
# .github/workflows/eval.yml
name: Model Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Evaluations
        uses: your-org/agi-eval-action@v1
        with:
          model-path: ./model
          benchmarks: mmlu,humaneval,truthfulqa
          
      - name: Update Badges
        run: |
          agi-eval badges --update
          
      - name: Comment Results
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./results/summary.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Evaluation Results\n${results.markdown}`
            });
```

### Auto-Generated Badges

```markdown
<!-- In your README.md -->
![MMLU Score](https://img.shields.io/endpoint?url=https://your-org.com/badges/mmlu)
![HumanEval Pass@1](https://img.shields.io/endpoint?url=https://your-org.com/badges/humaneval)
![Safety Score](https://img.shields.io/endpoint?url=https://your-org.com/badges/safety)
```

## 📈 Visualization

### Dashboard Features

```python
from agi_eval_sandbox import Dashboard

# Launch dashboard
dashboard = Dashboard()
dashboard.launch(port=8080)

# Features:
# - Real-time evaluation progress
# - Historical trend analysis
# - Model comparison matrices
# - Benchmark deep-dives
# - Export capabilities
```

### Longitudinal Tracking

```python
# Track model performance over time
from agi_eval_sandbox import LongitudinalTracker

tracker = LongitudinalTracker()

# Add evaluation results
tracker.add_evaluation(
    model_version="v1.0",
    results=results,
    timestamp="2024-01-15"
)

# Plot trends
tracker.plot_trends(
    metrics=["mmlu_accuracy", "humaneval_pass@1"],
    save_to="trends.png"
)

# Detect drift
drift_report = tracker.detect_drift(
    baseline_version="v1.0",
    current_version="v1.5",
    threshold=0.05
)
```

## 🔬 A/B Testing

### Model Comparison

```python
from agi_eval_sandbox import ABTester

# Initialize A/B tester
tester = ABTester()

# Define models
model_a = Model("gpt-4", temperature=0.0)
model_b = Model("gpt-4", temperature=0.7)

# Run comparison
comparison = tester.compare(
    models=[model_a, model_b],
    benchmarks=["truthfulqa", "creativity_test"],
    num_samples=1000,
    confidence_level=0.95
)

# View results
print(comparison.summary())
print(f"Winner: {comparison.winner}")
print(f"Statistical significance: {comparison.p_value}")

# Detailed analysis
comparison.plot_distributions(save_to="ab_test_results.png")
```

### Multi-Armed Bandit

```python
# Adaptive model selection
from agi_eval_sandbox import BanditSelector

selector = BanditSelector(
    models=[model_a, model_b, model_c],
    exploration_rate=0.1
)

# Use in production
for request in requests:
    model = selector.select_model()
    response = model.generate(request)
    reward = user_feedback(response)
    selector.update(model, reward)
```

## 📚 API Reference

### Core Classes

```python
class EvalSuite:
    def evaluate(self, model, benchmarks, **kwargs) -> Results
    def register_benchmark(self, benchmark: Benchmark) -> None
    def list_benchmarks(self) -> List[str]
    
class Model:
    def __init__(self, provider: str, name: str, **kwargs)
    def generate(self, prompt: str) -> str
    def batch_generate(self, prompts: List[str]) -> List[str]
    
class Results:
    def summary(self) -> Dict[str, float]
    def to_dataframe(self) -> pd.DataFrame
    def to_dashboard(self) -> None
    def export(self, format: str, path: str) -> None
```

### REST API

```bash
# Start API server
agi-eval serve --port 8000

# Submit evaluation job
POST /api/v1/evaluate
{
  "model": {
    "provider": "openai",
    "name": "gpt-4"
  },
  "benchmarks": ["mmlu", "humaneval"],
  "config": {...}
}

# Get job status
GET /api/v1/jobs/{job_id}

# Get leaderboard
GET /api/v1/leaderboard?benchmark=mmlu

# Download results
GET /api/v1/results/{job_id}/download?format=csv
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New benchmark implementations
- Evaluation metrics
- Visualization improvements
- Cloud provider integrations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/agi-eval-sandbox
cd agi-eval-sandbox

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build documentation
make docs
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation framework
- [HELM](https://github.com/stanford-crfm/helm) - Holistic evaluation
- [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Language model evaluation
- [Lang-Observatory](https://github.com/your-org/lang-observatory) - Monitoring platform

## 📞 Support

- 📧 Email: eval-support@your-org.com
- 💬 Discord: [Join our community](https://discord.gg/your-org)
- 📖 Documentation: [Full docs](https://docs.your-org.com/agi-eval)
- 🎥 Tutorial: [Video walkthrough](https://youtube.com/your-org/agi-eval-intro)
