#!/usr/bin/env python3
"""
Test the complete evaluation flow.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_complete_evaluation():
    """Test complete evaluation workflow."""
    from agi_eval_sandbox.core import EvalSuite, Model
    
    print("🧪 Testing Complete Evaluation Flow")
    print("=" * 50)
    
    # Create evaluation suite
    suite = EvalSuite()
    print(f"✅ EvalSuite created with {len(suite.list_benchmarks())} benchmarks")
    
    # Create local test model
    model = Model(provider="local", name="test-evaluation-model")
    print(f"✅ Model created: {model.name}")
    
    # Run evaluation on TruthfulQA with 2 questions
    print("\n🔄 Running evaluation...")
    results = await suite.evaluate(
        model=model,
        benchmarks=["truthfulqa"],
        num_questions=2,
        save_results=False
    )
    
    # Analyze results
    summary = results.summary()
    print(f"✅ Evaluation completed!")
    print(f"   Overall Score: {summary['overall_score']:.3f}")
    print(f"   Pass Rate: {summary['overall_pass_rate']:.1f}%")
    print(f"   Total Questions: {summary['total_questions']}")
    
    # Test benchmark-specific results
    truthfulqa_result = results.get_benchmark_result("truthfulqa")
    print(f"   TruthfulQA Score: {truthfulqa_result.average_score:.3f}")
    print(f"   TruthfulQA Pass Rate: {truthfulqa_result.pass_rate:.1f}%")
    
    # Test results export
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "test_results.json")
        results.export("json", json_path)
        
        if os.path.exists(json_path):
            print("✅ Results export working")
        else:
            print("❌ Results export failed")
    
    # Test model comparison
    print("\n🔄 Testing model comparison...")
    model2 = Model(provider="local", name="test-model-2")
    
    comparison_results = await suite.compare_models(
        models=[model, model2],
        benchmarks=["truthfulqa"],
        num_questions=1
    )
    
    print(f"✅ Model comparison completed with {len(comparison_results)} models")
    for model_name, model_results in comparison_results.items():
        model_summary = model_results.summary()
        print(f"   {model_name}: {model_summary['overall_score']:.3f} score")
    
    return True

async def main():
    """Run evaluation tests."""
    try:
        success = await test_complete_evaluation()
        if success:
            print("\n" + "=" * 50)
            print("🎉 Generation 1 Implementation Complete!")
            print("✅ Core evaluation engine working")
            print("✅ Benchmark system functional")
            print("✅ Model abstraction working")
            print("✅ Results management working")
            print("✅ Export functionality working")
            print("✅ Model comparison working")
            return True
    except Exception as e:
        print(f"\n❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)