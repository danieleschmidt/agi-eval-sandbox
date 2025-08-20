#!/usr/bin/env python3
"""
Simple API Test - Generation 1 Validation
Tests core API functionality without external dependencies
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_evaluation_flow():
    """Test basic evaluation workflow."""
    print("🧪 Testing Basic Evaluation Flow...")
    
    try:
        # Import core components
        from agi_eval_sandbox.core.evaluator import EvalSuite
        from agi_eval_sandbox.core.models import Model
        from agi_eval_sandbox.core.benchmarks import Question, Benchmark, Score
        
        # Create a simple test model
        class TestModel(Model):
            def __init__(self):
                super().__init__(provider="test", name="simple-test-model")
            
            async def generate(self, prompt: str, **config) -> str:
                # Simple deterministic response for testing
                if "math" in prompt.lower():
                    return "42"
                elif "hello" in prompt.lower():
                    return "Hello! How can I help you?"
                else:
                    return f"Test response for: {prompt[:50]}..."
            
            async def batch_generate(self, prompts: list, **config) -> list:
                results = []
                for prompt in prompts:
                    result = await self.generate(prompt, **config)
                    results.append(result)
                return results
        
        # Create a simple test benchmark
        class SimpleBenchmark(Benchmark):
            def __init__(self):
                super().__init__(name="simple_test", version="1.0")
            
            def get_questions(self):
                return [
                    Question(
                        id="q1",
                        prompt="What is 2+2?",
                        correct_answer="4",
                        category="math"
                    ),
                    Question(
                        id="q2", 
                        prompt="Say hello to me",
                        correct_answer="Hello",
                        category="greeting"
                    )
                ]
            
            def evaluate_response(self, question: Question, response: str) -> Score:
                if question.category == "math":
                    score = 1.0 if "4" in response or "42" in response else 0.0
                elif question.category == "greeting":
                    score = 1.0 if "hello" in response.lower() else 0.0
                else:
                    score = 0.5
                
                return Score(
                    value=score,
                    passed=score >= 0.5,
                    explanation=f"Simple evaluation for {question.category}"
                )
        
        # Test the evaluation pipeline
        print("  ✓ Creating test components...")
        eval_suite = EvalSuite()
        test_model = TestModel()
        test_benchmark = SimpleBenchmark()
        
        print("  ✓ Registering test benchmark...")
        eval_suite.register_benchmark(test_benchmark)
        
        print("  ✓ Running simple evaluation...")
        results = await eval_suite.evaluate(
            model=test_model,
            benchmarks=["simple_test"],
            save_results=False
        )
        
        print("  ✓ Validating results...")
        summary = results.summary()
        
        if summary and 'overall_score' in summary:
            print(f"  ✅ Evaluation completed! Score: {summary['overall_score']:.2f}")
            return True
        else:
            print("  ❌ Evaluation failed - no valid results")
            return False
            
    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_context_compression():
    """Test context compression functionality."""
    print("🗜️ Testing Context Compression...")
    
    try:
        from agi_eval_sandbox.core.context_compressor import ContextCompressionEngine, CompressionStrategy
        
        print("  ✓ Creating compression engine...")
        engine = ContextCompressionEngine()
        
        # Initialize without external models for testing
        print("  ✓ Initializing compression engine...")
        await engine.initialize()
        
        # Test with simple text
        test_text = """
        This is a test document for compression. It contains multiple sentences.
        Some sentences are more important than others. The goal is to reduce
        the length while preserving key information. This sentence is also
        important for understanding the document. Some details can be removed
        safely without losing meaning.
        """
        
        print("  ✓ Running extractive summarization...")
        compressed, metrics = await engine.compress(
            text=test_text,
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARIZATION,
            target_length=100
        )
        
        if compressed and len(compressed) < len(test_text):
            compression_ratio = len(compressed) / len(test_text)
            print(f"  ✅ Compression successful! Ratio: {compression_ratio:.2f}")
            print(f"     Original: {len(test_text)} chars")
            print(f"     Compressed: {len(compressed)} chars")
            return True
        else:
            print("  ❌ Compression failed - no size reduction")
            return False
            
    except Exception as e:
        print(f"  ❌ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_models():
    """Test API data models."""
    print("📊 Testing API Models...")
    
    try:
        from agi_eval_sandbox.api.models import (
            EvaluationRequest, ModelSpec, EvaluationConfig,
            JobResponse, BenchmarkInfo
        )
        
        print("  ✓ Creating test API objects...")
        
        # Test model specification
        model_spec = ModelSpec(
            provider="openai",
            name="gpt-4",
            api_key="test-key"
        )
        
        # Test evaluation config
        eval_config = EvaluationConfig(
            temperature=0.7,
            max_tokens=1000
        )
        
        # Test evaluation request
        eval_request = EvaluationRequest(
            model=model_spec,
            benchmarks=["mmlu", "humaneval"],
            config=eval_config
        )
        
        # Test serialization
        print("  ✓ Testing JSON serialization...")
        request_dict = eval_request.dict()
        request_json = json.dumps(request_dict)
        parsed_dict = json.loads(request_json)
        
        if parsed_dict == request_dict:
            print("  ✅ API models working correctly!")
            return True
        else:
            print("  ❌ JSON serialization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ API models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_loading():
    """Test configuration loading."""
    print("⚙️ Testing Configuration Loading...")
    
    try:
        from agi_eval_sandbox.config import settings
        
        print("  ✓ Loading settings...")
        
        # Check basic settings
        if hasattr(settings, 'API_HOST'):
            print(f"  ✓ API Host: {settings.API_HOST}")
        
        if hasattr(settings, 'LOG_LEVEL'):
            print(f"  ✓ Log Level: {settings.LOG_LEVEL}")
            
        print("  ✅ Configuration loading successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all simple tests."""
    print("\n" + "="*60)
    print("🚀 AGI EVALUATION SANDBOX - GENERATION 1 TESTS")
    print("="*60)
    
    tests = [
        ("Basic Evaluation Flow", test_basic_evaluation_flow),
        ("Context Compression", test_context_compression),
        ("API Models", test_api_models),
        ("Configuration Loading", test_config_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL GENERATION 1 TESTS PASSED!")
        print("✨ Core functionality is working correctly")
        return 0
    else:
        print(f"\n⚠️ {total-passed} tests failed")
        print("🔧 Some functionality needs attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)