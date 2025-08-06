#!/usr/bin/env python3
"""
Comprehensive tests for the context compression functionality.
This script tests all compression strategies and ensures they work correctly.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agi_eval_sandbox.core.context_compressor import (
    ContextCompressionEngine,
    CompressionStrategy,
    CompressionConfig
)


def test_sample_texts():
    """Sample texts for testing compression."""
    return {
        "short": "This is a short text. It has only a few sentences. Should not be compressed much.",
        
        "medium": """
        This is a medium-length text that should demonstrate context compression capabilities.
        It contains multiple sentences with different levels of importance.
        Some sentences provide crucial information about the topic.
        Others are less critical and could potentially be removed during compression.
        The compression algorithm should identify the most important sentences.
        It should preserve the core meaning while reducing the overall length.
        This text serves as a good test case for evaluating compression quality.
        Different compression strategies may handle this text differently.
        The goal is to maintain semantic coherence while achieving size reduction.
        """,
        
        "long": """
        Context compression is a critical technique in modern natural language processing systems.
        As language models become more powerful, they also consume more computational resources.
        The need for efficient context management has never been more important than it is today.
        
        Traditional approaches to context compression often rely on external retrieval systems.
        These systems require additional infrastructure and can introduce latency.
        Retrieval-free compression offers a more streamlined alternative approach.
        
        There are several key strategies for implementing retrieval-free context compression.
        Extractive summarization selects the most important sentences from the original text.
        This approach maintains the exact wording of important information.
        It works particularly well for documents with clear hierarchical structure.
        
        Sentence clustering groups similar sentences together for analysis.
        Representatives from each cluster are selected for the final output.
        This method can identify redundant information effectively.
        It helps eliminate repetitive content while preserving unique insights.
        
        Semantic filtering uses embedding-based similarity measurements.
        Sentences are evaluated based on their semantic relationship to key topics.
        This approach can identify conceptually important content.
        It works well when there's a clear thematic focus in the text.
        
        Token-level pruning operates at a more granular level than sentence selection.
        It removes individual words or phrases that contribute little to meaning.
        This technique can achieve higher compression ratios.
        However, it requires more sophisticated analysis to avoid degrading readability.
        
        Importance sampling uses probabilistic methods for content selection.
        Multiple factors contribute to calculating importance scores for each segment.
        Position within the document, semantic similarity, and novelty all play roles.
        This comprehensive approach can produce high-quality compressed output.
        
        Hierarchical compression applies different strategies at multiple levels.
        Document structure is analyzed to identify major sections and subsections.
        Different compression techniques may be applied to different parts.
        This multi-level approach can handle complex documents more effectively.
        
        The choice of compression strategy depends on several important factors.
        Document length, structure, and intended use case all influence the decision.
        Some strategies work better for technical documentation.
        Others are more suitable for narrative or conversational text.
        
        Evaluation of compression quality involves multiple metrics and considerations.
        Compression ratio indicates how much size reduction was achieved.
        Semantic similarity measures how well meaning was preserved.
        Information retention assesses whether critical details were maintained.
        Processing time is important for real-world applications and systems.
        
        Future developments in context compression will likely incorporate more advanced techniques.
        Machine learning models trained specifically for compression tasks show promise.
        Integration with domain-specific knowledge could improve results significantly.
        Real-time compression capabilities will become increasingly important as applications scale.
        """
    }


async def test_compression_strategy(engine, strategy, text, test_name):
    """Test a specific compression strategy."""
    print(f"\n{'='*60}")
    print(f"Testing {strategy.value} on {test_name} text")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        compressed_text, metrics = await engine.compress(
            text=text,
            strategy=strategy,
            target_length=None,
            preserve_structure=True
        )
        elapsed_time = time.time() - start_time
        
        print(f"✅ Success! Processed in {elapsed_time:.2f}s")
        print(f"📊 Original tokens: {metrics.original_tokens:,}")
        print(f"🗜️  Compressed tokens: {metrics.compressed_tokens:,}")
        print(f"📉 Compression ratio: {metrics.compression_ratio:.3f} ({(1-metrics.compression_ratio)*100:.1f}% reduction)")
        print(f"⚡ Processing time: {metrics.processing_time:.2f}s")
        
        if metrics.semantic_similarity:
            print(f"🎯 Semantic similarity: {metrics.semantic_similarity:.3f}")
        
        if metrics.information_retention:
            print(f"📈 Information retention: {metrics.information_retention:.3f}")
        
        # Show compressed text preview
        preview_length = min(200, len(compressed_text))
        print(f"\n📝 Compressed text preview:")
        print(f"{compressed_text[:preview_length]}{'...' if len(compressed_text) > preview_length else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_benchmark_strategies(engine, text, test_name):
    """Test benchmark functionality."""
    print(f"\n{'='*60}")
    print(f"Benchmarking all strategies on {test_name} text")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        results = await engine.benchmark_strategies(text)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Benchmark completed in {elapsed_time:.2f}s")
        print(f"\n🏆 Results Summary:")
        print(f"{'Strategy':<25} {'Ratio':<8} {'Tokens':<12} {'Similarity':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        # Sort by compression ratio
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1][1].compression_ratio
        )
        
        for strategy, (compressed_text, metrics) in sorted_results:
            similarity = metrics.semantic_similarity or 0.0
            print(
                f"{strategy.value:<25} {metrics.compression_ratio:<8.3f} "
                f"{metrics.compressed_tokens:<12,} {similarity:<12.3f} "
                f"{metrics.processing_time:<10.2f}"
            )
        
        # Show best results
        best_compression = min(results.items(), key=lambda x: x[1][1].compression_ratio)
        print(f"\n🏆 Best compression: {best_compression[0].value} "
               f"({best_compression[1][1].compression_ratio:.3f} ratio)")
        
        if any(metrics.semantic_similarity for _, (_, metrics) in results.items()):
            best_similarity = max(
                results.items(), 
                key=lambda x: x[1][1].semantic_similarity or 0
            )
            print(f"🎯 Best similarity: {best_similarity[0].value} "
                   f"({best_similarity[1][1].semantic_similarity:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling(engine):
    """Test error handling capabilities."""
    print(f"\n{'='*60}")
    print("Testing Error Handling")
    print(f"{'='*60}")
    
    test_cases = [
        ("Empty text", ""),
        ("Whitespace only", "   \n\t  "),
        ("Single word", "Hello"),
        ("Very short", "Hi there."),
    ]
    
    for test_name, text in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        try:
            compressed_text, metrics = await engine.compress(text=text)
            print(f"✅ Handled gracefully: {metrics.compression_ratio:.3f} ratio")
        except Exception as e:
            print(f"⚠️  Exception (expected for some cases): {str(e)}")


async def test_different_configurations():
    """Test different compression configurations."""
    print(f"\n{'='*60}")
    print("Testing Different Configurations")
    print(f"{'='*60}")
    
    test_text = test_sample_texts()["medium"]
    
    configs = [
        ("Conservative", CompressionConfig(target_ratio=0.8, semantic_threshold=0.8)),
        ("Balanced", CompressionConfig(target_ratio=0.5, semantic_threshold=0.7)),
        ("Aggressive", CompressionConfig(target_ratio=0.3, semantic_threshold=0.5)),
    ]
    
    for config_name, config in configs:
        print(f"\n🔧 Configuration: {config_name}")
        engine = ContextCompressionEngine(config)
        await engine.initialize()
        
        try:
            compressed_text, metrics = await engine.compress(text=test_text)
            print(f"✅ Target ratio: {config.target_ratio:.1f}, "
                  f"Actual ratio: {metrics.compression_ratio:.3f}")
            print(f"📊 Tokens: {metrics.original_tokens} → {metrics.compressed_tokens}")
        except Exception as e:
            print(f"❌ Failed: {str(e)}")


async def run_comprehensive_tests():
    """Run all tests comprehensively."""
    print("🚀 Starting Comprehensive Context Compression Tests")
    print("="*80)
    
    # Initialize compression engine
    print("🔧 Initializing compression engine...")
    engine = ContextCompressionEngine()
    
    try:
        await engine.initialize()
        print("✅ Compression engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize compression engine: {e}")
        return False
    
    available_strategies = engine.get_available_strategies()
    print(f"📋 Available strategies: {[s.value for s in available_strategies]}")
    
    sample_texts = test_sample_texts()
    
    # Test each strategy on different text lengths
    total_tests = 0
    passed_tests = 0
    
    for text_name, text in sample_texts.items():
        for strategy in available_strategies:
            total_tests += 1
            success = await test_compression_strategy(engine, strategy, text, text_name)
            if success:
                passed_tests += 1
    
    # Test benchmarking functionality
    for text_name, text in sample_texts.items():
        if text_name != "short":  # Skip benchmark for very short text
            total_tests += 1
            success = await test_benchmark_strategies(engine, text, text_name)
            if success:
                passed_tests += 1
    
    # Test error handling
    await test_error_handling(engine)
    
    # Test different configurations
    await test_different_configurations()
    
    # Final results
    print(f"\n{'='*80}")
    print("🏁 FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    print(f"📊 Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Context compression is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


async def main():
    """Main test function."""
    print("Context Compression Test Suite")
    print("=============================")
    
    success = await run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)