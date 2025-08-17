#!/usr/bin/env python3
"""
Direct Validation Test - Generation 1 Implementation

Simple direct test to validate core functionality.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🚀 Starting Direct Validation Test")
print("Generation 1: Testing core functionality...")

try:
    # Test core imports
    print("\n📦 Testing imports...")
    from agi_eval_sandbox.core.models import Model
    from agi_eval_sandbox.core.benchmarks import CustomBenchmark, Question, QuestionType, Score
    from agi_eval_sandbox.core.results import BenchmarkResult, EvaluationResult
    print("✅ Core imports successful")
    
    # Test basic model creation
    print("\n🤖 Testing model creation...")
    model = Model(provider="local", name="test_model")
    print(f"✅ Model created: {model.name} ({model.provider})")
    
    # Test question creation
    print("\n❓ Testing question creation...")
    question = Question(
        id="test_q1",
        prompt="What is 2+2?",
        correct_answer="4",
        question_type=QuestionType.SHORT_ANSWER
    )
    print(f"✅ Question created: {question.id}")
    
    # Test benchmark creation
    print("\n🏗️ Testing benchmark creation...")
    questions = [question]
    benchmark = CustomBenchmark(name="test_benchmark", questions=questions)
    print(f"✅ Benchmark created: {benchmark.name}")
    
    # Test score creation
    print("\n📊 Testing score creation...")
    score = Score(value=0.85, passed=True, explanation="Good response")
    print(f"✅ Score created: {score.value}")
    
    # Test evaluation result creation
    print("\n📋 Testing evaluation result creation...")
    eval_result = EvaluationResult(
        question_id=question.id,
        question_prompt=question.prompt,
        model_response="4",
        score=score,
        benchmark_name=benchmark.name
    )
    print(f"✅ Evaluation result created: {eval_result.question_id}")
    
    # Test benchmark result creation
    print("\n📈 Testing benchmark result creation...")
    benchmark_result = BenchmarkResult(
        benchmark_name=benchmark.name,
        model_name=model.name,
        model_provider=model.provider,
        results=[eval_result]
    )
    print(f"✅ Benchmark result created: {benchmark_result.average_score:.3f}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Core research framework is working")
    print("🚀 GENERATION 1 COMPLETE - Ready for Generation 2!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n💥 VALIDATION FAILED")
    sys.exit(1)

print("\n🏁 Direct validation completed successfully!")
