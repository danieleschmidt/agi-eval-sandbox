"""Simple command line interface for AGI Evaluation Sandbox."""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import List, Optional
import logging

from .core import EvalSuite, Model
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
            api_key = getattr(settings, 'openai_api_key', None)
        elif provider.lower() == 'anthropic':
            api_key = getattr(settings, 'anthropic_api_key', None)
    
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
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            elif output_format == 'csv':
                results.to_csv(str(output_path))
            
            click.echo(f"💾 Results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
def list_benchmarks():
    """List all available benchmarks."""
    suite = EvalSuite()
    benchmarks = suite.list_benchmarks()
    
    click.echo("📋 Available Benchmarks:")
    click.echo("=" * 40)
    for i, benchmark in enumerate(benchmarks, 1):
        click.echo(f"{i}. {benchmark}")


@main.command()
@click.option('--benchmark', '-b', required=True, help='Benchmark name')
@click.option('--num-questions', '-n', type=int, default=5, help='Number of sample questions')
def sample(benchmark: str, num_questions: int):
    """Show sample questions from a benchmark."""
    suite = EvalSuite()
    bench = suite.get_benchmark(benchmark)
    
    if not bench:
        click.echo(f"❌ Unknown benchmark: {benchmark}", err=True)
        sys.exit(1)
    
    questions = bench.get_sample_questions(num_questions)
    
    click.echo(f"📋 Sample Questions from {benchmark.upper()}:")
    click.echo("=" * 50)
    
    for i, q in enumerate(questions, 1):
        click.echo(f"\n{i}. {q.prompt}")
        click.echo(f"   Category: {q.category or 'N/A'}")
        click.echo(f"   Type: {q.question_type.value}")
        if q.choices:
            for j, choice in enumerate(q.choices, 1):
                click.echo(f"   {chr(64+j)}. {choice}")


if __name__ == '__main__':
    main()