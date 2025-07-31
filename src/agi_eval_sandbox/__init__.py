"""
AGI Evaluation Sandbox

A comprehensive evaluation platform for large language models and AI systems.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core imports for easy access
from .core.evaluator import EvalSuite
from .core.models import Model
from .core.benchmarks import Benchmark, CustomBenchmark
from .core.results import Results

# Configuration
from .config import settings

__all__ = [
    "__version__",
    "EvalSuite", 
    "Model",
    "Benchmark",
    "CustomBenchmark", 
    "Results",
    "settings"
]