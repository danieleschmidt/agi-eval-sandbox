#!/usr/bin/env python3
"""
Quick Start Demo for AGI Evaluation Sandbox
Generation 1: Make It Work (Simple)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demo_basic_evaluation():
    """Demo basic evaluation functionality."""
    print("🚀 AGI Evaluation Sandbox - Quick Start Demo")
    print("=" * 50)
    
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        from agi_eval_sandbox.core.models import Model
        
        # Initialize evaluation suite
        suite = EvalSuite()
        print(f"✅ Evaluation suite initialized")
        print(f"📊 Available benchmarks: {suite.list_benchmarks()}")
        
        # Create a local model for demonstration
        mock_model = Model(
            provider="local",
            name="demo-model",
            api_key="demo-key-1234567890"  # Meets minimum length requirement
        )
        print(f"🤖 Created demo model: {mock_model.name}")
        
        # List available benchmarks
        benchmarks = suite.list_benchmarks()
        print(f"\n📋 {len(benchmarks)} benchmarks available:")
        for benchmark in benchmarks:
            try:
                bench_obj = suite.get_benchmark(benchmark)
                questions = bench_obj.get_questions()
                print(f"  - {benchmark}: {len(questions)} questions")
            except Exception as e:
                print(f"  - {benchmark}: Error loading ({e})")
        
        print("\n✅ Basic functionality verified!")
        print("🎯 Ready for full evaluation runs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_benchmark_details():
    """Demo benchmark inspection functionality."""
    print("\n🔍 Benchmark Details Demo")
    print("=" * 30)
    
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        
        suite = EvalSuite()
        
        # Inspect first benchmark
        benchmark_name = suite.list_benchmarks()[0]
        benchmark = suite.get_benchmark(benchmark_name)
        questions = benchmark.get_questions()
        
        print(f"📊 Benchmark: {benchmark_name}")
        print(f"📝 Total questions: {len(questions)}")
        print(f"🏷️  Version: {benchmark.version}")
        
        # Show sample questions
        if questions:
            print(f"\n📋 Sample questions (showing first 2):")
            for i, question in enumerate(questions[:2]):
                print(f"\n{i+1}. ID: {question.id}")
                print(f"   Type: {question.question_type.value}")
                print(f"   Prompt: {question.prompt[:100]}...")
                if hasattr(question, 'category') and question.category:
                    print(f"   Category: {question.category}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during benchmark demo: {e}")
        return False

async def demo_cli_functionality():
    """Demo CLI integration."""
    print("\n💻 CLI Integration Demo")
    print("=" * 25)
    
    try:
        # Test settings (CLI has external dependencies)
        from agi_eval_sandbox.config import settings
        print("✅ Settings module loaded")
        
        print("💡 CLI commands available:")
        print("  - python3 -m agi_eval_sandbox.cli run --model <model> --provider <provider>")
        print("  - python3 -m agi_eval_sandbox.cli list")
        print("  - python3 -m agi_eval_sandbox.cli compare --models <models>")
        print("  - python3 -m agi_eval_sandbox.cli serve")
        print("  - python3 -m agi_eval_sandbox.cli status")
        print("⚠️  Note: CLI requires additional dependencies (click, uvicorn)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during CLI demo: {e}")
        return False

async def main():
    """Main demo function."""
    print("🎯 AGI Evaluation Sandbox - Generation 1 Demo")
    print("=" * 60)
    
    success = True
    
    # Run basic evaluation demo
    success &= await demo_basic_evaluation()
    
    # Run benchmark details demo
    success &= await demo_benchmark_details()
    
    # Run CLI demo
    success &= await demo_cli_functionality()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All demos completed successfully!")
        print("✨ Generation 1 implementation is working!")
        print("\n📋 Next steps:")
        print("  1. Run: python3 simple_quick_start.py")
        print("  2. Install dependencies: pip install -e .")
        print("  3. Try CLI: python3 -m agi_eval_sandbox.cli list")
        print("  4. Start API: python3 -m agi_eval_sandbox.cli serve")
    else:
        print("❌ Some demos failed - check error messages above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)