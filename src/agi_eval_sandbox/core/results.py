"""Results management and analysis."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import statistics

try:
    import pandas as pd
except ImportError:
    pd = None

from .benchmarks import Benchmark, Question, Score


@dataclass
class EvaluationResult:
    """Single evaluation result for a question."""
    question_id: str
    question_prompt: str
    model_response: str
    score: Score
    benchmark_name: str
    category: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class BenchmarkResult:
    """Results for an entire benchmark."""
    benchmark_name: str
    model_name: str
    model_provider: str
    results: List[EvaluationResult]
    timestamp: datetime = field(default_factory=datetime.now)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_questions(self) -> int:
        """Total number of questions evaluated."""
        return len(self.results)
    
    @property
    def average_score(self) -> float:
        """Average score across all questions."""
        if not self.results:
            return 0.0
        return statistics.mean(result.score.value for result in self.results)
    
    @property
    def pass_rate(self) -> float:
        """Percentage of questions that passed."""
        if not self.results:
            return 0.0
        passed = sum(1 for result in self.results if result.score.passed)
        return (passed / len(self.results)) * 100
    
    @property
    def category_scores(self) -> Dict[str, float]:
        """Average scores by category."""
        category_results = {}
        for result in self.results:
            category = result.category or "uncategorized"
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result.score.value)
        
        return {
            category: statistics.mean(scores)
            for category, scores in category_results.items()
        }
    
    def get_failed_questions(self) -> List[EvaluationResult]:
        """Get all questions that failed evaluation."""
        return [result for result in self.results if not result.score.passed]
    
    def get_results_by_category(self, category: str) -> List[EvaluationResult]:
        """Get results filtered by category."""
        return [result for result in self.results if result.category == category]


class Results:
    """Main results container for evaluation runs."""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp: datetime = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_benchmark_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the collection."""
        self.benchmark_results.append(result)
    
    def get_benchmark_result(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Get results for a specific benchmark."""
        for result in self.benchmark_results:
            if result.benchmark_name == benchmark_name:
                return result
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.benchmark_results:
            return {
                "total_benchmarks": 0,
                "total_questions": 0,
                "overall_score": 0.0,
                "overall_pass_rate": 0.0,
                "benchmark_scores": {}
            }
        
        total_questions = sum(result.total_questions for result in self.benchmark_results)
        all_scores = []
        all_passed = []
        
        for result in self.benchmark_results:
            for eval_result in result.results:
                all_scores.append(eval_result.score.value)
                all_passed.append(eval_result.score.passed)
        
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        overall_pass_rate = (sum(all_passed) / len(all_passed)) * 100 if all_passed else 0.0
        
        benchmark_scores = {
            result.benchmark_name: {
                "average_score": result.average_score,
                "pass_rate": result.pass_rate,
                "total_questions": result.total_questions,
                "category_scores": result.category_scores
            }
            for result in self.benchmark_results
        }
        
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "total_benchmarks": len(self.benchmark_results),
            "total_questions": total_questions,
            "overall_score": overall_score,
            "overall_pass_rate": overall_pass_rate,
            "benchmark_scores": benchmark_scores,
            "metadata": self.metadata
        }
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        if pd is None:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")
            
        rows = []
        for benchmark_result in self.benchmark_results:
            for eval_result in benchmark_result.results:
                rows.append({
                    "run_id": self.run_id,
                    "benchmark": benchmark_result.benchmark_name,
                    "model_name": benchmark_result.model_name,
                    "model_provider": benchmark_result.model_provider,
                    "question_id": eval_result.question_id,
                    "question_prompt": eval_result.question_prompt[:100] + "..." if len(eval_result.question_prompt) > 100 else eval_result.question_prompt,
                    "model_response": eval_result.model_response[:200] + "..." if len(eval_result.model_response) > 200 else eval_result.model_response,
                    "score": eval_result.score.value,
                    "max_score": eval_result.score.max_value,
                    "passed": eval_result.score.passed,
                    "percentage": eval_result.score.percentage,
                    "category": eval_result.category,
                    "timestamp": eval_result.timestamp.isoformat(),
                    "explanation": eval_result.score.explanation
                })
        
        return pd.DataFrame(rows)
    
    def export(self, format: str, path: str) -> None:
        """Export results to various formats."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(path_obj, "w") as f:
                json.dump(self.summary(), f, indent=2, default=str)
        
        elif format.lower() == "csv":
            df = self.to_dataframe()
            df.to_csv(path_obj, index=False)
        
        elif format.lower() == "html":
            self._export_html(path_obj)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_html(self, path: Path) -> None:
        """Export results as HTML report."""
        summary = self.summary()
        df = self.to_dataframe()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AGI Evaluation Results - {self.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .benchmark {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AGI Evaluation Results</h1>
        <p><strong>Run ID:</strong> {self.run_id}</p>
        <p><strong>Timestamp:</strong> {summary['timestamp']}</p>
        <p><strong>Total Benchmarks:</strong> {summary['total_benchmarks']}</p>
        <p><strong>Total Questions:</strong> {summary['total_questions']}</p>
        <p><strong>Overall Score:</strong> {summary['overall_score']:.2f}</p>
        <p><strong>Overall Pass Rate:</strong> {summary['overall_pass_rate']:.1f}%</p>
    </div>
    
    <div class="summary">
        <h2>Benchmark Summary</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Average Score</th>
                <th>Pass Rate</th>
                <th>Total Questions</th>
            </tr>
"""
        
        for benchmark_name, scores in summary['benchmark_scores'].items():
            html_content += f"""
            <tr>
                <td>{benchmark_name}</td>
                <td>{scores['average_score']:.3f}</td>
                <td>{scores['pass_rate']:.1f}%</td>
                <td>{scores['total_questions']}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="details">
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Question ID</th>
                <th>Score</th>
                <th>Passed</th>
                <th>Category</th>
                <th>Explanation</th>
            </tr>
"""
        
        for _, row in df.iterrows():
            passed_class = "passed" if row['passed'] else "failed"
            passed_text = "✓" if row['passed'] else "✗"
            html_content += f"""
            <tr>
                <td>{row['benchmark']}</td>
                <td>{row['question_id']}</td>
                <td>{row['score']:.3f}</td>
                <td class="{passed_class}">{passed_text}</td>
                <td>{row['category'] or 'N/A'}</td>
                <td>{row['explanation'] or 'N/A'}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(path, "w") as f:
            f.write(html_content)
    
    def to_dashboard(self) -> None:
        """Display results in dashboard (placeholder)."""
        print("📊 Results Dashboard")
        print("=" * 50)
        
        summary = self.summary()
        print(f"Run ID: {summary['run_id']}")
        print(f"Overall Score: {summary['overall_score']:.3f}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
        print()
        
        print("Benchmark Results:")
        for benchmark_name, scores in summary['benchmark_scores'].items():
            print(f"  {benchmark_name}:")
            print(f"    Score: {scores['average_score']:.3f}")
            print(f"    Pass Rate: {scores['pass_rate']:.1f}%")
            print(f"    Questions: {scores['total_questions']}")
            if scores['category_scores']:
                print("    Categories:")
                for category, score in scores['category_scores'].items():
                    print(f"      {category}: {score:.3f}")
            print()
    
    def get_comparison(self, other: "Results") -> Dict[str, Any]:
        """Compare with another Results instance."""
        self_summary = self.summary()
        other_summary = other.summary()
        
        comparison = {
            "runs": {
                "current": self_summary['run_id'],
                "comparison": other_summary['run_id']
            },
            "overall_score_change": self_summary['overall_score'] - other_summary['overall_score'],
            "pass_rate_change": self_summary['overall_pass_rate'] - other_summary['overall_pass_rate'],
            "benchmark_changes": {}
        }
        
        # Compare individual benchmarks
        for benchmark in self_summary['benchmark_scores']:
            if benchmark in other_summary['benchmark_scores']:
                current = self_summary['benchmark_scores'][benchmark]
                previous = other_summary['benchmark_scores'][benchmark]
                
                comparison['benchmark_changes'][benchmark] = {
                    "score_change": current['average_score'] - previous['average_score'],
                    "pass_rate_change": current['pass_rate'] - previous['pass_rate']
                }
        
        return comparison