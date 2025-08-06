"""Command line interface for AGI Evaluation Sandbox."""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import List, Optional
import logging

from .core import EvalSuite, Model
from .core.context_compressor import ContextCompressionEngine, CompressionStrategy, CompressionConfig
from .config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(debug: bool):
    """AGI Evaluation Sandbox - Comprehensive LLM evaluation platform."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option('--model', '-m', required=True, help='Model name (e.g., gpt-4, claude-3-opus)')
@click.option('--provider', '-p', required=True, help='Model provider (openai, anthropic, local)')
@click.option('--api-key', help='API key for the model provider')
@click.option('--benchmarks', '-b', default='all', help='Comma-separated benchmark names or "all"')
@click.option('--num-questions', '-n', type=int, help='Limit number of questions per benchmark')
@click.option('--temperature', '-t', type=float, default=0.0, help='Model temperature')
@click.option('--max-tokens', type=int, default=2048, help='Maximum tokens per response')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'html']), default='json', help='Output format')
@click.option('--parallel/--sequential', default=True, help='Run benchmarks in parallel or sequentially')
def run(
    model: str,
    provider: str,
    api_key: Optional[str],
    benchmarks: str,
    num_questions: Optional[int],
    temperature: float,
    max_tokens: int,
    output: Optional[str],
    output_format: str,
    parallel: bool
):
    """Run evaluation on specified benchmarks."""
    
    # Handle API key from environment if not provided
    if not api_key:
        if provider.lower() == 'openai':
            api_key = settings.openai_api_key
        elif provider.lower() == 'anthropic':
            api_key = settings.anthropic_api_key
    
    if not api_key and provider.lower() != 'local':
        click.echo(f"Error: API key required for provider '{provider}'", err=True)
        sys.exit(1)
    
    try:
        # Initialize model
        model_instance = Model(
            provider=provider,
            name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Parse benchmarks
        if benchmarks.lower() == 'all':
            benchmark_list = 'all'
        else:
            benchmark_list = [b.strip() for b in benchmarks.split(',')]
        
        # Run evaluation
        click.echo(f"🚀 Starting evaluation of {model} ({provider})")
        click.echo(f"📊 Benchmarks: {benchmarks}")
        if num_questions:
            click.echo(f"❓ Questions per benchmark: {num_questions}")
        
        suite = EvalSuite()
        results = asyncio.run(
            suite.evaluate(
                model=model_instance,
                benchmarks=benchmark_list,
                num_questions=num_questions,
                parallel=parallel,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📈 EVALUATION RESULTS")
        click.echo("="*60)
        
        summary = results.summary()
        click.echo(f"🎯 Overall Score: {summary['overall_score']:.3f}")
        click.echo(f"✅ Pass Rate: {summary['overall_pass_rate']:.1f}%")
        click.echo(f"📝 Total Questions: {summary['total_questions']}")
        click.echo()
        
        # Benchmark breakdown
        for benchmark_name, scores in summary['benchmark_scores'].items():
            click.echo(f"📊 {benchmark_name.upper()}:")
            click.echo(f"   Score: {scores['average_score']:.3f}")
            click.echo(f"   Pass Rate: {scores['pass_rate']:.1f}%")
            click.echo(f"   Questions: {scores['total_questions']}")
            
            if scores['category_scores']:
                click.echo("   Categories:")
                for category, score in scores['category_scores'].items():
                    click.echo(f"     {category}: {score:.3f}")
            click.echo()
        
        # Save results if output specified
        if output:
            results.export(output_format, output)
            click.echo(f"💾 Results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--models', '-m', required=True, help='Comma-separated model specs (format: provider:model:api_key)')
@click.option('--benchmarks', '-b', default='all', help='Comma-separated benchmark names or "all"')
@click.option('--output', '-o', help='Output file path for comparison results')
def compare(models: str, benchmarks: str, output: Optional[str]):
    """Compare multiple models on the same benchmarks."""
    
    try:
        # Parse model specifications
        model_specs = []
        for model_spec in models.split(','):
            parts = model_spec.strip().split(':')
            if len(parts) < 2:
                click.echo(f"Error: Invalid model spec '{model_spec}'. Use format: provider:model[:api_key]", err=True)
                sys.exit(1)
            
            provider = parts[0]
            model_name = parts[1]
            api_key = parts[2] if len(parts) > 2 else None
            
            # Get API key from settings if not provided
            if not api_key and provider.lower() != 'local':
                if provider.lower() == 'openai':
                    api_key = settings.openai_api_key
                elif provider.lower() == 'anthropic':
                    api_key = settings.anthropic_api_key
            
            model_instance = Model(provider=provider, name=model_name, api_key=api_key)
            model_specs.append(model_instance)
        
        # Parse benchmarks
        if benchmarks.lower() == 'all':
            benchmark_list = 'all'
        else:
            benchmark_list = [b.strip() for b in benchmarks.split(',')]
        
        click.echo(f"🔄 Comparing {len(model_specs)} models on benchmarks: {benchmarks}")
        
        suite = EvalSuite()
        comparison_results = suite.compare_models(model_specs, benchmark_list)
        
        # Display comparison
        click.echo("\n" + "="*80)
        click.echo("🏆 MODEL COMPARISON RESULTS")
        click.echo("="*80)
        
        # Create comparison table
        benchmark_names = set()
        for results in comparison_results.values():
            benchmark_names.update(results.summary()['benchmark_scores'].keys())
        
        # Header
        click.echo(f"{'Model':<20} {'Overall Score':<15} {'Pass Rate':<12} ", end='')
        for benchmark in sorted(benchmark_names):
            click.echo(f"{benchmark:<12}", end='')
        click.echo()
        
        click.echo("-" * (47 + 12 * len(benchmark_names)))
        
        # Model rows
        for model_name, results in comparison_results.items():
            summary = results.summary()
            click.echo(f"{model_name:<20} {summary['overall_score']:<15.3f} {summary['overall_pass_rate']:<12.1f}% ", end='')
            
            for benchmark in sorted(benchmark_names):
                if benchmark in summary['benchmark_scores']:
                    score = summary['benchmark_scores'][benchmark]['average_score']
                    click.echo(f"{score:<12.3f}", end='')
                else:
                    click.echo(f"{'N/A':<12}", end='')
            click.echo()
        
        # Save comparison if output specified
        if output:
            comparison_data = {
                model_name: results.summary()
                for model_name, results in comparison_results.items()
            }
            with open(output, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            click.echo(f"\n💾 Comparison results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command(name='list')
def list_benchmarks():
    """List all available benchmarks."""
    suite = EvalSuite()
    benchmarks = suite.list_benchmarks()
    
    click.echo("📋 Available Benchmarks:")
    click.echo("="*30)
    
    for benchmark_name in sorted(benchmarks):
        benchmark = suite.get_benchmark(benchmark_name)
        questions = benchmark.get_questions()
        click.echo(f"• {benchmark_name} ({len(questions)} questions)")
    
    click.echo(f"\nTotal: {len(benchmarks)} benchmarks available")


@main.command()
@click.argument('benchmark_name')
@click.option('--sample-size', '-s', type=int, default=3, help='Number of sample questions to show')
def describe(benchmark_name: str, sample_size: int):
    """Describe a specific benchmark and show sample questions."""
    suite = EvalSuite()
    benchmark = suite.get_benchmark(benchmark_name)
    
    if not benchmark:
        click.echo(f"❌ Benchmark '{benchmark_name}' not found", err=True)
        sys.exit(1)
    
    questions = benchmark.get_questions()
    sample_questions = questions[:sample_size]
    
    click.echo(f"📊 Benchmark: {benchmark_name}")
    click.echo("="*50)
    click.echo(f"Version: {benchmark.version}")
    click.echo(f"Total Questions: {len(questions)}")
    
    # Show categories if available
    categories = set()
    for q in questions:
        if q.category:
            categories.add(q.category)
    
    if categories:
        click.echo(f"Categories: {', '.join(sorted(categories))}")
    
    click.echo(f"\n📝 Sample Questions (showing {len(sample_questions)} of {len(questions)}):")
    click.echo("-" * 50)
    
    for i, question in enumerate(sample_questions, 1):
        click.echo(f"\n{i}. ID: {question.id}")
        if question.category:
            click.echo(f"   Category: {question.category}")
        click.echo(f"   Type: {question.question_type.value}")
        click.echo(f"   Prompt: {question.prompt[:200]}{'...' if len(question.prompt) > 200 else ''}")
        
        if question.choices:
            click.echo("   Choices:")
            for j, choice in enumerate(question.choices):
                click.echo(f"     {chr(65+j)}. {choice}")
        
        click.echo(f"   Correct Answer: {question.correct_answer}")


@main.command()
@click.option('--benchmark', '-b', help='Filter by specific benchmark')
@click.option('--metric', '-m', type=click.Choice(['average_score', 'pass_rate']), default='average_score', help='Metric to sort by')
@click.option('--limit', '-l', type=int, default=10, help='Maximum number of results to show')
def leaderboard(benchmark: Optional[str], metric: str, limit: int):
    """Show leaderboard of model performance."""
    suite = EvalSuite()
    leaderboard_data = suite.get_leaderboard(benchmark=benchmark, metric=metric)
    
    if not leaderboard_data:
        click.echo("📭 No evaluation results found. Run some evaluations first!")
        return
    
    title = f"🏆 Leaderboard"
    if benchmark:
        title += f" - {benchmark}"
    title += f" (sorted by {metric})"
    
    click.echo(title)
    click.echo("="*80)
    
    # Header
    click.echo(f"{'Rank':<6} {'Model':<20} {'Provider':<12} {'Benchmark':<15} {'Score':<10} {'Pass Rate':<12}")
    click.echo("-" * 80)
    
    # Show top results
    for rank, record in enumerate(leaderboard_data[:limit], 1):
        click.echo(
            f"{rank:<6} {record['model_name']:<20} {record['model_provider']:<12} "
            f"{record['benchmark']:<15} {record['average_score']:<10.3f} {record['pass_rate']:<12.1f}%"
        )


@main.command()
@click.option('--api-url', default='http://localhost:8080', help='API server URL')
@click.option('--job-id', help='Specific job ID to check')
@click.option('--watch', '-w', is_flag=True, help='Watch mode - continuously poll for updates')
@click.option('--interval', type=int, default=5, help='Poll interval in seconds for watch mode')
def status(api_url: str, job_id: Optional[str], watch: bool, interval: int):
    """Check status of running evaluations."""
    try:
        import requests
        import time
    except ImportError:
        click.echo("❌ 'requests' package is required for status command. Install with: pip install requests", err=True)
        sys.exit(1)
    
    try:
        if job_id:
            # Check specific job status
            url = f"{api_url.rstrip('/')}/api/v1/jobs/{job_id}"
            
            while True:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 404:
                        click.echo(f"❌ Job '{job_id}' not found", err=True)
                        sys.exit(1)
                    
                    response.raise_for_status()
                    job_data = response.json()
                    
                    click.clear()
                    click.echo(f"📊 Job Status: {job_id}")
                    click.echo("="*50)
                    click.echo(f"Status: {job_data['status']}")
                    click.echo(f"Progress: {job_data['progress']:.1%}")
                    click.echo(f"Created: {job_data['created_at']}")
                    
                    if job_data.get('completed_at'):
                        click.echo(f"Completed: {job_data['completed_at']}")
                    
                    if job_data.get('error'):
                        click.echo(f"❌ Error: {job_data['error']}")
                    
                    # Get results if completed
                    if job_data['status'] == 'completed':
                        try:
                            results_url = f"{api_url.rstrip('/')}/api/v1/jobs/{job_id}/results"
                            results_response = requests.get(results_url, timeout=10)
                            if results_response.status_code == 200:
                                results = results_response.json()
                                if results.get('results'):
                                    summary = results['results']
                                    click.echo("\n📈 Results:")
                                    click.echo(f"Overall Score: {summary.get('overall_score', 0):.3f}")
                                    click.echo(f"Pass Rate: {summary.get('overall_pass_rate', 0):.1f}%")
                                    click.echo(f"Total Questions: {summary.get('total_questions', 0)}")
                        except Exception as e:
                            click.echo(f"⚠️ Could not fetch results: {e}")
                    
                    if not watch or job_data['status'] in ['completed', 'failed', 'cancelled']:
                        break
                    
                    click.echo(f"\n🔄 Refreshing in {interval}s... (Ctrl+C to stop)")
                    time.sleep(interval)
                    
                except requests.RequestException as e:
                    click.echo(f"❌ Failed to connect to API: {e}", err=True)
                    if not watch:
                        sys.exit(1)
                    time.sleep(interval)
                except KeyboardInterrupt:
                    click.echo("\n👋 Stopped watching")
                    break
        
        else:
            # List all jobs status
            url = f"{api_url.rstrip('/')}/api/v1/jobs"
            
            while True:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    jobs = data.get('jobs', [])
                    
                    click.clear()
                    click.echo("📋 Evaluation Jobs Status")
                    click.echo("="*60)
                    
                    if not jobs:
                        click.echo("No evaluation jobs found.")
                    else:
                        # Header
                        click.echo(f"{'Job ID':<12} {'Status':<12} {'Model':<20} {'Provider':<12} {'Created':<20}")
                        click.echo("-" * 60)
                        
                        # Jobs
                        for job in jobs:
                            job_id_short = job['job_id'][:8] + '...' if len(job['job_id']) > 8 else job['job_id']
                            status_icon = {
                                'pending': '⏳',
                                'running': '🔄',
                                'completed': '✅',
                                'failed': '❌',
                                'cancelled': '🚫'
                            }.get(job['status'], '❓')
                            
                            click.echo(
                                f"{job_id_short:<12} {status_icon + job['status']:<12} {job['model']:<20} "
                                f"{job['provider']:<12} {job['created_at'][:19]:<20}"
                            )
                        
                        # Summary
                        running_count = sum(1 for j in jobs if j['status'] == 'running')
                        completed_count = sum(1 for j in jobs if j['status'] == 'completed')
                        failed_count = sum(1 for j in jobs if j['status'] == 'failed')
                        
                        click.echo("\n📊 Summary:")
                        click.echo(f"Running: {running_count}, Completed: {completed_count}, Failed: {failed_count}")
                    
                    if not watch:
                        break
                    
                    click.echo(f"\n🔄 Refreshing in {interval}s... (Ctrl+C to stop)")
                    time.sleep(interval)
                    
                except requests.RequestException as e:
                    click.echo(f"❌ Failed to connect to API server at {api_url}", err=True)
                    click.echo(f"Make sure the server is running with: agi-eval serve")
                    if not watch:
                        sys.exit(1)
                    time.sleep(interval)
                except KeyboardInterrupt:
                    click.echo("\n👋 Stopped watching")
                    break
    
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server')
@click.option('--port', default=8080, type=int, help='Port to bind the server')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    try:
        import uvicorn
        from .api.main import app
        
        click.echo(f"🚀 Starting AGI Evaluation Sandbox API server")
        click.echo(f"📡 Server: http://{host}:{port}")
        click.echo(f"📖 Docs: http://{host}:{port}/docs")
        
        uvicorn.run(
            "agi_eval_sandbox.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        click.echo("❌ API server dependencies not installed. Install with: pip install 'agi-eval-sandbox[api]'", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}", err=True)
        sys.exit(1)


@main.group()
def compress():
    """Context compression commands."""
    pass


@compress.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file for compressed text')
@click.option('--strategy', '-s', 
              type=click.Choice([s.value for s in CompressionStrategy]), 
              default='extractive_summarization',
              help='Compression strategy to use')
@click.option('--target-ratio', '-r', type=float, default=0.5, 
              help='Target compression ratio (0.1 = 90% reduction)')
@click.option('--target-length', '-l', type=int, 
              help='Target length in tokens (overrides ratio)')
@click.option('--preserve-structure/--no-preserve-structure', default=True,
              help='Preserve document structure')
@click.option('--model-name', '-m', default='sentence-transformers/all-MiniLM-L6-v2',
              help='Model name for semantic analysis')
@click.option('--semantic-threshold', type=float, default=0.7,
              help='Semantic similarity threshold for filtering')
def compress_file(
    input_file: str,
    output: Optional[str],
    strategy: str,
    target_ratio: float,
    target_length: Optional[int],
    preserve_structure: bool,
    model_name: str,
    semantic_threshold: float
):
    """Compress text from a file using retrieval-free compression."""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            click.echo("❌ Input file is empty", err=True)
            sys.exit(1)
        
        # Configure compression
        config = CompressionConfig(
            strategy=CompressionStrategy(strategy),
            target_ratio=target_ratio,
            semantic_threshold=semantic_threshold,
            model_name=model_name
        )
        
        # Initialize compression engine
        click.echo(f"🔧 Initializing {strategy} compressor...")
        engine = ContextCompressionEngine(config)
        
        async def run_compression():
            await engine.initialize()
            return await engine.compress(
                text=text,
                target_length=target_length,
                preserve_structure=preserve_structure
            )
        
        # Run compression
        click.echo(f"🗜️  Compressing text using {strategy}...")
        compressed_text, metrics = asyncio.run(run_compression())
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 COMPRESSION RESULTS")
        click.echo("="*60)
        click.echo(f"📝 Original tokens: {metrics.original_tokens:,}")
        click.echo(f"🗜️  Compressed tokens: {metrics.compressed_tokens:,}")
        click.echo(f"📉 Compression ratio: {metrics.compression_ratio:.3f} ({(1-metrics.compression_ratio)*100:.1f}% reduction)")
        click.echo(f"⚡ Processing time: {metrics.processing_time:.2f}s")
        
        if metrics.semantic_similarity:
            click.echo(f"🎯 Semantic similarity: {metrics.semantic_similarity:.3f}")
        
        if metrics.information_retention:
            click.echo(f"📊 Information retention: {metrics.information_retention:.3f}")
        
        # Output compressed text
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(compressed_text)
            click.echo(f"\n💾 Compressed text saved to: {output}")
        else:
            click.echo("\n" + "="*60)
            click.echo("🗜️  COMPRESSED TEXT")
            click.echo("="*60)
            click.echo(compressed_text)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@compress.command()
@click.argument('text')
@click.option('--strategy', '-s', 
              type=click.Choice([s.value for s in CompressionStrategy]), 
              default='extractive_summarization',
              help='Compression strategy to use')
@click.option('--target-ratio', '-r', type=float, default=0.5,
              help='Target compression ratio')
@click.option('--target-length', '-l', type=int,
              help='Target length in tokens')
def compress_text(
    text: str,
    strategy: str,
    target_ratio: float,
    target_length: Optional[int]
):
    """Compress text directly from command line."""
    try:
        config = CompressionConfig(
            strategy=CompressionStrategy(strategy),
            target_ratio=target_ratio
        )
        
        engine = ContextCompressionEngine(config)
        
        async def run_compression():
            await engine.initialize()
            return await engine.compress(
                text=text,
                target_length=target_length
            )
        
        compressed_text, metrics = asyncio.run(run_compression())
        
        click.echo(f"📝 Original: {metrics.original_tokens} tokens")
        click.echo(f"🗜️  Compressed: {metrics.compressed_tokens} tokens ({metrics.compression_ratio:.3f} ratio)")
        click.echo(f"📊 Semantic similarity: {metrics.semantic_similarity:.3f}")
        click.echo("\n" + compressed_text)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@compress.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--target-length', '-l', type=int, help='Target length for benchmarking')
def benchmark(input_file: str, target_length: Optional[int]):
    """Benchmark all compression strategies on a file."""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Run benchmark
        click.echo("🏁 Benchmarking all compression strategies...")
        
        async def run_benchmark():
            engine = ContextCompressionEngine()
            await engine.initialize()
            return await engine.benchmark_strategies(text, target_length)
        
        results = asyncio.run(run_benchmark())
        
        # Display results
        click.echo("\n" + "="*80)
        click.echo("🏆 COMPRESSION STRATEGY BENCHMARK")
        click.echo("="*80)
        
        # Header
        click.echo(f"{'Strategy':<25} {'Ratio':<8} {'Tokens':<12} {'Similarity':<12} {'Time (s)':<10}")
        click.echo("-" * 80)
        
        # Sort by compression ratio
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1][1].compression_ratio
        )
        
        for strategy, (compressed_text, metrics) in sorted_results:
            click.echo(
                f"{strategy.value:<25} {metrics.compression_ratio:<8.3f} "
                f"{metrics.compressed_tokens:<12,} {metrics.semantic_similarity or 0:<12.3f} "
                f"{metrics.processing_time:<10.2f}"
            )
        
        # Show best strategy
        best_strategy = min(results.items(), key=lambda x: x[1][1].compression_ratio)
        click.echo(f"\n🏆 Best compression: {best_strategy[0].value} "
                   f"({best_strategy[1][1].compression_ratio:.3f} ratio)")
        
        # Show best semantic similarity
        best_similarity = max(
            results.items(), 
            key=lambda x: x[1][1].semantic_similarity or 0
        )
        click.echo(f"🎯 Best similarity: {best_similarity[0].value} "
                   f"({best_similarity[1][1].semantic_similarity:.3f} similarity)")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@compress.command(name='list')
def list_strategies():
    """List all available compression strategies."""
    click.echo("🗜️  Available Compression Strategies:")
    click.echo("="*40)
    
    for strategy in CompressionStrategy:
        descriptions = {
            CompressionStrategy.EXTRACTIVE_SUMMARIZATION: "Select most important sentences",
            CompressionStrategy.SENTENCE_CLUSTERING: "Group similar sentences and select representatives", 
            CompressionStrategy.SEMANTIC_FILTERING: "Filter by semantic similarity to key topics",
            CompressionStrategy.TOKEN_PRUNING: "Remove less important tokens",
            CompressionStrategy.IMPORTANCE_SAMPLING: "Probabilistic sampling based on importance scores",
            CompressionStrategy.HIERARCHICAL_COMPRESSION: "Multi-level compression approach"
        }
        
        desc = descriptions.get(strategy, "Advanced compression technique")
        click.echo(f"• {strategy.value:<25} - {desc}")


@main.command()
def start():
    """Start the complete sandbox environment (API + Dashboard)."""
    click.echo("🏗️  Starting AGI Evaluation Sandbox...")
    click.echo("This will start both the API server and web dashboard")
    click.echo("Use 'agi-eval serve' for API only")
    # Implementation would start both API and dashboard
    click.echo("⚠️  Full environment startup not yet implemented")


if __name__ == '__main__':
    main()