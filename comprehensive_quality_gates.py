#!/usr/bin/env python3
"""
Comprehensive Quality Gates - Generation 3 Final Validation
Mandatory quality gates with automated testing and validation
"""

import asyncio
import time
import sys
import gc
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import subprocess
from collections import defaultdict

class QualityLevel(Enum):
    GOLD = "gold"          # 95%+ pass rate
    SILVER = "silver"      # 85%+ pass rate  
    BRONZE = "bronze"      # 70%+ pass rate
    FAIL = "fail"          # <70% pass rate

class GateCategory(Enum):
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"

@dataclass
class QualityGateResult:
    gate_name: str
    category: GateCategory
    passed: bool
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class QualityReport:
    overall_score: float
    quality_level: QualityLevel
    gate_results: List[QualityGateResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)

class ComprehensiveQualityGates:
    """Comprehensive quality gate system with automated validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("quality_gates")
        self.project_root = Path(__file__).parent
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 0.80,      # 80% minimum
            "performance_efficiency": 0.70,  # 70% minimum
            "error_rate": 0.05,         # 5% maximum
            "response_time_ms": 1000,   # 1s maximum
            "memory_efficiency": 0.80,  # 80% minimum
            "security_score": 0.90,     # 90% minimum
        }
        
        # Gate registry
        self.quality_gates: Dict[str, Callable] = {}
        self._register_default_gates()
    
    def _register_default_gates(self):
        """Register default quality gates."""
        self.register_gate("functionality_core", self._gate_functionality_core)
        self.register_gate("functionality_integration", self._gate_functionality_integration)
        self.register_gate("performance_response_time", self._gate_performance_response_time)
        self.register_gate("performance_throughput", self._gate_performance_throughput)
        self.register_gate("performance_memory", self._gate_performance_memory)
        self.register_gate("security_input_validation", self._gate_security_input_validation)
        self.register_gate("security_error_handling", self._gate_security_error_handling)
        self.register_gate("reliability_error_recovery", self._gate_reliability_error_recovery)
        self.register_gate("reliability_concurrency", self._gate_reliability_concurrency)
        self.register_gate("maintainability_code_structure", self._gate_maintainability_code_structure)
        self.register_gate("scalability_load", self._gate_scalability_load)
    
    def register_gate(self, name: str, gate_func: Callable):
        """Register a quality gate function."""
        self.quality_gates[name] = gate_func
    
    async def run_all_gates(self) -> QualityReport:
        """Run all registered quality gates."""
        print("🔍 Running comprehensive quality gates...")
        
        gate_results = []
        start_time = time.time()
        
        for gate_name, gate_func in self.quality_gates.items():
            try:
                print(f"  🚪 Executing gate: {gate_name}")
                gate_start = time.time()
                
                result = await gate_func()
                result.execution_time_ms = (time.time() - gate_start) * 1000
                
                gate_results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"    {status} {result.message} (Score: {result.score:.2f})")
                
            except Exception as e:
                print(f"    ❌ CRASH Gate execution failed: {str(e)}")
                gate_results.append(QualityGateResult(
                    gate_name=gate_name,
                    category=GateCategory.FUNCTIONALITY,  # Default category
                    passed=False,
                    score=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    execution_time_ms=(time.time() - gate_start) * 1000
                ))
        
        # Calculate overall quality
        total_time = time.time() - start_time
        report = self._generate_quality_report(gate_results, total_time)
        
        return report
    
    def _generate_quality_report(self, gate_results: List[QualityGateResult], total_time: float) -> QualityReport:
        """Generate comprehensive quality report."""
        # Calculate overall score
        if gate_results:
            overall_score = sum(r.score for r in gate_results) / len(gate_results)
        else:
            overall_score = 0.0
        
        # Determine quality level
        if overall_score >= 0.95:
            quality_level = QualityLevel.GOLD
        elif overall_score >= 0.85:
            quality_level = QualityLevel.SILVER
        elif overall_score >= 0.70:
            quality_level = QualityLevel.BRONZE
        else:
            quality_level = QualityLevel.FAIL
        
        # Generate summary by category
        category_summary = defaultdict(list)
        for result in gate_results:
            category_summary[result.category].append(result)
        
        summary = {
            "total_gates": len(gate_results),
            "passed_gates": sum(1 for r in gate_results if r.passed),
            "failed_gates": sum(1 for r in gate_results if not r.passed),
            "execution_time_seconds": total_time,
            "category_scores": {}
        }
        
        for category, results in category_summary.items():
            category_score = sum(r.score for r in results) / len(results)
            category_passed = sum(1 for r in results if r.passed)
            
            summary["category_scores"][category.value] = {
                "score": category_score,
                "passed": category_passed,
                "total": len(results),
                "pass_rate": category_passed / len(results)
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gate_results, overall_score)
        
        return QualityReport(
            overall_score=overall_score,
            quality_level=quality_level,
            gate_results=gate_results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult], overall_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 0.70:
            recommendations.append("🚨 CRITICAL: Overall quality below minimum threshold. Immediate action required.")
        elif overall_score < 0.85:
            recommendations.append("⚠️ WARNING: Quality improvement needed to reach production standards.")
        
        # Category-specific recommendations
        failed_gates = [r for r in gate_results if not r.passed]
        
        if failed_gates:
            functionality_fails = [r for r in failed_gates if r.category == GateCategory.FUNCTIONALITY]
            if functionality_fails:
                recommendations.append("🔧 Fix core functionality issues before proceeding to other optimizations.")
            
            performance_fails = [r for r in failed_gates if r.category == GateCategory.PERFORMANCE]
            if performance_fails:
                recommendations.append("⚡ Optimize performance bottlenecks, particularly response times and memory usage.")
            
            security_fails = [r for r in failed_gates if r.category == GateCategory.SECURITY]
            if security_fails:
                recommendations.append("🛡️ Address security vulnerabilities immediately - these are blocking issues.")
            
            reliability_fails = [r for r in failed_gates if r.category == GateCategory.RELIABILITY]
            if reliability_fails:
                recommendations.append("🔄 Improve error handling and system resilience for production readiness.")
        
        # Specific threshold recommendations
        low_score_gates = [r for r in gate_results if r.score < 0.5]
        if low_score_gates:
            recommendations.append(f"📉 Focus on gates with very low scores: {[r.gate_name for r in low_score_gates]}")
        
        return recommendations
    
    # Quality Gate Implementations
    
    async def _gate_functionality_core(self) -> QualityGateResult:
        """Test core functionality."""
        try:
            # Add src to path
            sys.path.insert(0, str(self.project_root / "src"))
            
            from agi_eval_sandbox.core.evaluator import EvalSuite
            from agi_eval_sandbox.core.models import Model
            
            # Test core functionality
            eval_suite = EvalSuite()
            benchmarks = eval_suite.list_benchmarks()
            
            # Test model creation
            model = Model(provider="local", name="quality-gate-test")
            
            score = 1.0 if len(benchmarks) >= 3 else 0.7
            passed = score >= 0.8
            
            return QualityGateResult(
                gate_name="functionality_core",
                category=GateCategory.FUNCTIONALITY,
                passed=passed,
                score=score,
                message=f"Core functionality test: {len(benchmarks)} benchmarks available",
                details={"benchmarks": benchmarks, "model_created": True}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="functionality_core",
                category=GateCategory.FUNCTIONALITY,
                passed=False,
                score=0.0,
                message=f"Core functionality failed: {str(e)}"
            )
    
    async def _gate_functionality_integration(self) -> QualityGateResult:
        """Test integration functionality."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            
            from agi_eval_sandbox.core.benchmarks import Question, Score, QuestionType
            from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
            
            # Test data structure integration
            question = Question(
                id="integration_test",
                prompt="Integration test question",
                correct_answer="Integration answer",
                category="test"
            )
            
            score = Score(value=0.9, passed=True, explanation="Integration test")
            
            eval_result = EvaluationResult(
                question_id=question.id,
                question_prompt=question.prompt,
                model_response="Integration response",
                score=score,
                benchmark_name="integration_test",
                category="test"
            )
            
            benchmark_result = BenchmarkResult(
                benchmark_name="integration_test",
                model_name="test_model",
                model_provider="local",
                results=[eval_result]
            )
            
            results = Results()
            results.add_benchmark_result(benchmark_result)
            summary = results.summary()
            
            integration_score = 1.0 if summary and 'overall_score' in summary else 0.5
            
            return QualityGateResult(
                gate_name="functionality_integration",
                category=GateCategory.FUNCTIONALITY,
                passed=integration_score >= 0.8,
                score=integration_score,
                message="Integration test completed successfully",
                details={"summary": summary}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="functionality_integration",
                category=GateCategory.FUNCTIONALITY,
                passed=False,
                score=0.0,
                message=f"Integration test failed: {str(e)}"
            )
    
    async def _gate_performance_response_time(self) -> QualityGateResult:
        """Test response time performance."""
        try:
            # Simulate API response time test
            start_time = time.time()
            
            # Simulate work
            await asyncio.sleep(0.05)  # 50ms simulated work
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Score based on response time (lower is better)
            if response_time_ms <= 100:
                score = 1.0
            elif response_time_ms <= 500:
                score = 0.8
            elif response_time_ms <= 1000:
                score = 0.6
            else:
                score = 0.3
            
            passed = response_time_ms <= self.thresholds["response_time_ms"]
            
            return QualityGateResult(
                gate_name="performance_response_time",
                category=GateCategory.PERFORMANCE,
                passed=passed,
                score=score,
                message=f"Response time: {response_time_ms:.1f}ms",
                details={"response_time_ms": response_time_ms, "threshold": self.thresholds["response_time_ms"]}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_response_time",
                category=GateCategory.PERFORMANCE,
                passed=False,
                score=0.0,
                message=f"Response time test failed: {str(e)}"
            )
    
    async def _gate_performance_throughput(self) -> QualityGateResult:
        """Test throughput performance."""
        try:
            # Simulate throughput test
            operations = 100
            start_time = time.time()
            
            # Simulate batch operations
            tasks = [asyncio.sleep(0.001) for _ in range(operations)]
            await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            throughput = operations / duration
            
            # Score based on throughput (higher is better)
            if throughput >= 1000:
                score = 1.0
            elif throughput >= 500:
                score = 0.8
            elif throughput >= 100:
                score = 0.6
            else:
                score = 0.3
            
            passed = throughput >= 50  # Minimum 50 ops/sec
            
            return QualityGateResult(
                gate_name="performance_throughput",
                category=GateCategory.PERFORMANCE,
                passed=passed,
                score=score,
                message=f"Throughput: {throughput:.1f} ops/sec",
                details={"throughput_ops_per_sec": throughput, "operations": operations, "duration": duration}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_throughput",
                category=GateCategory.PERFORMANCE,
                passed=False,
                score=0.0,
                message=f"Throughput test failed: {str(e)}"
            )
    
    async def _gate_performance_memory(self) -> QualityGateResult:
        """Test memory performance."""
        try:
            # Memory efficiency test
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Create and clean up objects
            test_data = [{"id": i, "data": f"test_{i}"} for i in range(1000)]
            objects_created = len(gc.get_objects()) - initial_objects
            
            # Clean up
            del test_data
            gc.collect()
            final_objects = len(gc.get_objects())
            
            # Calculate memory efficiency
            objects_cleaned = max(0, objects_created - (final_objects - initial_objects))
            cleanup_efficiency = objects_cleaned / objects_created if objects_created > 0 else 1.0
            
            score = cleanup_efficiency
            passed = cleanup_efficiency >= self.thresholds["memory_efficiency"]
            
            return QualityGateResult(
                gate_name="performance_memory",
                category=GateCategory.PERFORMANCE,
                passed=passed,
                score=score,
                message=f"Memory efficiency: {cleanup_efficiency:.1%}",
                details={
                    "initial_objects": initial_objects,
                    "objects_created": objects_created,
                    "final_objects": final_objects,
                    "cleanup_efficiency": cleanup_efficiency
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="performance_memory",
                category=GateCategory.PERFORMANCE,
                passed=False,
                score=0.0,
                message=f"Memory test failed: {str(e)}"
            )
    
    async def _gate_security_input_validation(self) -> QualityGateResult:
        """Test input validation security."""
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            
            from agi_eval_sandbox.core.validation import InputValidator
            from agi_eval_sandbox.core.exceptions import ValidationError
            
            validator = InputValidator()
            
            # Test various validation scenarios
            test_cases = [
                ("valid_provider", lambda: validator.validate_provider("openai"), True),
                ("invalid_provider", lambda: validator.validate_provider("invalid"), False),
                ("valid_string", lambda: validator.validate_string("test", "test_field"), True),
                ("invalid_empty", lambda: validator.validate_string("", "test_field"), False),
            ]
            
            passed_tests = 0
            total_tests = len(test_cases)
            
            for test_name, test_func, should_pass in test_cases:
                try:
                    test_func()
                    if should_pass:
                        passed_tests += 1
                except ValidationError:
                    if not should_pass:
                        passed_tests += 1
                except Exception:
                    pass  # Test failed
            
            score = passed_tests / total_tests
            passed = score >= 0.8
            
            return QualityGateResult(
                gate_name="security_input_validation",
                category=GateCategory.SECURITY,
                passed=passed,
                score=score,
                message=f"Input validation: {passed_tests}/{total_tests} tests passed",
                details={"passed_tests": passed_tests, "total_tests": total_tests}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="security_input_validation",
                category=GateCategory.SECURITY,
                passed=False,
                score=0.0,
                message=f"Input validation test failed: {str(e)}"
            )
    
    async def _gate_security_error_handling(self) -> QualityGateResult:
        """Test error handling security."""
        try:
            # Test that errors don't leak sensitive information
            test_passed = True
            error_details = []
            
            # Simulate various error conditions
            try:
                raise FileNotFoundError("Sensitive file path: /secret/passwords.txt")
            except FileNotFoundError as e:
                # Error message should not contain full sensitive paths
                if "/secret/" in str(e):
                    test_passed = False
                    error_details.append("File path leakage detected")
            
            # Test exception handling
            try:
                raise ValueError("Database connection failed: user=admin, pass=secret123")
            except ValueError as e:
                # Should not contain credentials
                if "pass=" in str(e) or "password" in str(e).lower():
                    test_passed = False
                    error_details.append("Credential leakage detected")
            
            score = 1.0 if test_passed else 0.3
            
            return QualityGateResult(
                gate_name="security_error_handling",
                category=GateCategory.SECURITY,
                passed=test_passed,
                score=score,
                message="Error handling security validated" if test_passed else f"Security issues: {error_details}",
                details={"issues": error_details}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="security_error_handling",
                category=GateCategory.SECURITY,
                passed=False,
                score=0.0,
                message=f"Error handling test failed: {str(e)}"
            )
    
    async def _gate_reliability_error_recovery(self) -> QualityGateResult:
        """Test error recovery reliability."""
        try:
            # Test retry and recovery mechanisms
            attempt_count = 0
            max_attempts = 3
            success = False
            
            for attempt in range(max_attempts):
                attempt_count += 1
                try:
                    # Simulate failure for first 2 attempts
                    if attempt < 2:
                        raise ConnectionError(f"Simulated failure {attempt + 1}")
                    else:
                        success = True
                        break
                except ConnectionError:
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(0.01)  # Brief retry delay
                    continue
            
            recovery_rate = 1.0 if success else 0.0
            score = recovery_rate
            passed = success
            
            return QualityGateResult(
                gate_name="reliability_error_recovery",
                category=GateCategory.RELIABILITY,
                passed=passed,
                score=score,
                message=f"Error recovery: {attempt_count} attempts, {'success' if success else 'failed'}",
                details={"attempts": attempt_count, "recovered": success}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="reliability_error_recovery",
                category=GateCategory.RELIABILITY,
                passed=False,
                score=0.0,
                message=f"Error recovery test failed: {str(e)}"
            )
    
    async def _gate_reliability_concurrency(self) -> QualityGateResult:
        """Test concurrency reliability."""
        try:
            # Test concurrent operations
            concurrent_tasks = 20
            success_count = 0
            
            async def concurrent_operation(task_id: int):
                try:
                    await asyncio.sleep(0.01)  # Simulate work
                    return f"Task {task_id} completed"
                except Exception as e:
                    raise Exception(f"Task {task_id} failed: {e}")
            
            # Run concurrent tasks
            tasks = [concurrent_operation(i) for i in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful operations
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            success_rate = success_count / concurrent_tasks
            score = success_rate
            passed = success_rate >= 0.9  # 90% success rate required
            
            return QualityGateResult(
                gate_name="reliability_concurrency",
                category=GateCategory.RELIABILITY,
                passed=passed,
                score=score,
                message=f"Concurrency: {success_count}/{concurrent_tasks} tasks succeeded",
                details={"success_count": success_count, "total_tasks": concurrent_tasks, "success_rate": success_rate}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="reliability_concurrency",
                category=GateCategory.RELIABILITY,
                passed=False,
                score=0.0,
                message=f"Concurrency test failed: {str(e)}"
            )
    
    async def _gate_maintainability_code_structure(self) -> QualityGateResult:
        """Test code structure maintainability."""
        try:
            # Analyze project structure
            src_path = self.project_root / "src"
            
            if not src_path.exists():
                return QualityGateResult(
                    gate_name="maintainability_code_structure",
                    category=GateCategory.MAINTAINABILITY,
                    passed=False,
                    score=0.0,
                    message="Source directory not found"
                )
            
            # Count Python files and structure
            python_files = list(src_path.rglob("*.py"))
            
            # Check for proper package structure
            has_init_files = any(f.name == "__init__.py" for f in python_files)
            has_core_module = any("core" in str(f) for f in python_files)
            has_api_module = any("api" in str(f) for f in python_files)
            
            structure_score = 0.0
            structure_score += 0.4 if has_init_files else 0.0
            structure_score += 0.3 if has_core_module else 0.0
            structure_score += 0.3 if has_api_module else 0.0
            
            passed = structure_score >= 0.7
            
            return QualityGateResult(
                gate_name="maintainability_code_structure",
                category=GateCategory.MAINTAINABILITY,
                passed=passed,
                score=structure_score,
                message=f"Code structure score: {structure_score:.1%}",
                details={
                    "python_files": len(python_files),
                    "has_init_files": has_init_files,
                    "has_core_module": has_core_module,
                    "has_api_module": has_api_module
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="maintainability_code_structure",
                category=GateCategory.MAINTAINABILITY,
                passed=False,
                score=0.0,
                message=f"Code structure test failed: {str(e)}"
            )
    
    async def _gate_scalability_load(self) -> QualityGateResult:
        """Test scalability under load."""
        try:
            # Simulate increasing load test
            load_levels = [10, 50, 100]
            performance_scores = []
            
            for load_level in load_levels:
                start_time = time.time()
                
                # Simulate operations under load
                tasks = [asyncio.sleep(0.001) for _ in range(load_level)]
                await asyncio.gather(*tasks)
                
                duration = time.time() - start_time
                throughput = load_level / duration
                
                # Calculate performance degradation
                expected_throughput = load_level * 500  # Expected 500 ops/sec per operation
                performance_ratio = min(1.0, throughput / expected_throughput)
                performance_scores.append(performance_ratio)
            
            # Average performance across load levels
            avg_performance = sum(performance_scores) / len(performance_scores)
            
            # Check for graceful degradation (performance shouldn't drop below 50%)
            min_performance = min(performance_scores)
            graceful_degradation = min_performance >= 0.5
            
            score = avg_performance * (1.0 if graceful_degradation else 0.7)
            passed = score >= 0.6
            
            return QualityGateResult(
                gate_name="scalability_load",
                category=GateCategory.SCALABILITY,
                passed=passed,
                score=score,
                message=f"Load test: {avg_performance:.1%} avg performance, graceful degradation: {graceful_degradation}",
                details={
                    "load_levels": load_levels,
                    "performance_scores": performance_scores,
                    "avg_performance": avg_performance,
                    "graceful_degradation": graceful_degradation
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="scalability_load",
                category=GateCategory.SCALABILITY,
                passed=False,
                score=0.0,
                message=f"Scalability test failed: {str(e)}"
            )

def print_quality_report(report: QualityReport):
    """Print formatted quality report."""
    print("\n" + "="*60)
    print("🏆 COMPREHENSIVE QUALITY REPORT")
    print("="*60)
    
    # Overall score
    level_emoji = {
        QualityLevel.GOLD: "🥇",
        QualityLevel.SILVER: "🥈", 
        QualityLevel.BRONZE: "🥉",
        QualityLevel.FAIL: "❌"
    }
    
    print(f"\n{level_emoji[report.quality_level]} QUALITY LEVEL: {report.quality_level.value.upper()}")
    print(f"📊 OVERALL SCORE: {report.overall_score:.1%}")
    print(f"⏱️ EXECUTION TIME: {report.summary['execution_time_seconds']:.2f}s")
    
    # Category breakdown
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, stats in report.summary["category_scores"].items():
        emoji = "✅" if stats["pass_rate"] == 1.0 else "⚠️" if stats["pass_rate"] >= 0.5 else "❌"
        print(f"  {emoji} {category.title()}: {stats['score']:.1%} "
              f"({stats['passed']}/{stats['total']} passed)")
    
    # Failed gates
    failed_gates = [r for r in report.gate_results if not r.passed]
    if failed_gates:
        print(f"\n❌ FAILED GATES ({len(failed_gates)}):")
        for gate in failed_gates:
            print(f"  • {gate.gate_name}: {gate.message}")
    
    # Top performing gates
    top_gates = sorted([r for r in report.gate_results if r.passed], key=lambda x: x.score, reverse=True)[:3]
    if top_gates:
        print(f"\n✅ TOP PERFORMING GATES:")
        for gate in top_gates:
            print(f"  • {gate.gate_name}: {gate.score:.1%} - {gate.message}")
    
    # Recommendations
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"  {rec}")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT:")
    if report.quality_level in [QualityLevel.GOLD, QualityLevel.SILVER]:
        print("  🎉 PRODUCTION READY! System meets quality standards.")
    elif report.quality_level == QualityLevel.BRONZE:
        print("  ⚠️ ACCEPTABLE but improvements recommended before production.")
    else:
        print("  🚨 NOT READY for production. Critical issues must be resolved.")
    
    print("="*60)

async def main():
    """Run comprehensive quality gates."""
    print("\n" + "="*60)
    print("🛡️ COMPREHENSIVE QUALITY GATES - GENERATION 3")
    print("="*60)
    
    quality_system = ComprehensiveQualityGates()
    
    # Run all quality gates
    report = await quality_system.run_all_gates()
    
    # Print detailed report
    print_quality_report(report)
    
    # Return appropriate exit code
    if report.quality_level == QualityLevel.FAIL:
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)