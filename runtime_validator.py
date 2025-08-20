#!/usr/bin/env python3
"""
Runtime Environment Validator for AGI Evaluation Sandbox
Validates core functionality without external dependencies
"""

import sys
import os
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class ValidationResult:
    name: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class RuntimeValidator:
    """Validates runtime environment and core functionality."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent
    
    def validate_python_environment(self) -> ValidationResult:
        """Validate Python version and core modules."""
        try:
            version = sys.version_info
            if version.major == 3 and version.minor >= 9:
                return ValidationResult(
                    name="python_version",
                    level=ValidationLevel.CRITICAL,
                    passed=True,
                    message=f"Python {version.major}.{version.minor}.{version.micro} ✓"
                )
            else:
                return ValidationResult(
                    name="python_version", 
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message=f"Python {version.major}.{version.minor} unsupported. Requires 3.9+"
                )
        except Exception as e:
            return ValidationResult(
                name="python_version",
                level=ValidationLevel.CRITICAL, 
                passed=False,
                message=f"Python validation failed: {e}"
            )
    
    def validate_project_structure(self) -> ValidationResult:
        """Validate project directory structure."""
        required_paths = [
            "src/agi_eval_sandbox",
            "src/agi_eval_sandbox/core",
            "src/agi_eval_sandbox/api", 
            "tests",
            "pyproject.toml",
            "README.md"
        ]
        
        missing_paths = []
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                missing_paths.append(path)
        
        if missing_paths:
            return ValidationResult(
                name="project_structure",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Missing required paths: {missing_paths}",
                details={"missing_paths": missing_paths}
            )
        else:
            return ValidationResult(
                name="project_structure",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="Project structure valid ✓"
            )
    
    def validate_core_imports(self) -> ValidationResult:
        """Validate core module imports work."""
        try:
            # Test core imports without external dependencies
            sys.path.insert(0, str(self.project_root / "src"))
            
            # Import core modules
            from agi_eval_sandbox import __version__
            from agi_eval_sandbox.core.models import Model
            from agi_eval_sandbox.core.benchmarks import Benchmark
            
            return ValidationResult(
                name="core_imports",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="Core module imports successful ✓",
                details={"version": __version__}
            )
        except ImportError as e:
            return ValidationResult(
                name="core_imports",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Core import failed: {e}",
                details={"import_error": str(e)}
            )
        except Exception as e:
            return ValidationResult(
                name="core_imports",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Import validation error: {e}"
            )
    
    def validate_async_support(self) -> ValidationResult:
        """Validate async/await functionality."""
        try:
            async def test_async():
                await asyncio.sleep(0.001)
                return "async_works"
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test_async())
            loop.close()
            
            if result == "async_works":
                return ValidationResult(
                    name="async_support",
                    level=ValidationLevel.CRITICAL,
                    passed=True,
                    message="Async/await support validated ✓"
                )
            else:
                return ValidationResult(
                    name="async_support",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message="Async test returned unexpected result"
                )
        except Exception as e:
            return ValidationResult(
                name="async_support",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Async validation failed: {e}"
            )
    
    def validate_json_handling(self) -> ValidationResult:
        """Validate JSON serialization/deserialization."""
        try:
            test_data = {
                "model": "test-model",
                "temperature": 0.7,
                "benchmarks": ["mmlu", "humaneval"],
                "metadata": {
                    "timestamp": time.time(),
                    "version": "1.0.0"
                }
            }
            
            # Test JSON roundtrip
            json_str = json.dumps(test_data)
            parsed_data = json.loads(json_str)
            
            if parsed_data == test_data:
                return ValidationResult(
                    name="json_handling",
                    level=ValidationLevel.CRITICAL,
                    passed=True,
                    message="JSON serialization/deserialization ✓"
                )
            else:
                return ValidationResult(
                    name="json_handling",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message="JSON roundtrip failed - data mismatch"
                )
        except Exception as e:
            return ValidationResult(
                name="json_handling",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"JSON validation failed: {e}"
            )
    
    def validate_file_permissions(self) -> ValidationResult:
        """Validate file system permissions."""
        try:
            # Test read permissions
            config_files = [
                "pyproject.toml",
                "README.md"
            ]
            
            for config_file in config_files:
                file_path = self.project_root / config_file
                if file_path.exists():
                    content = file_path.read_text()
                    if not content:
                        return ValidationResult(
                            name="file_permissions",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"Cannot read {config_file}"
                        )
            
            # Test write permissions in temp directory
            temp_file = self.project_root / "temp_validation_test.txt"
            temp_file.write_text("validation test")
            content = temp_file.read_text()
            temp_file.unlink()
            
            if content == "validation test":
                return ValidationResult(
                    name="file_permissions",
                    level=ValidationLevel.CRITICAL,
                    passed=True,
                    message="File system permissions valid ✓"
                )
            else:
                return ValidationResult(
                    name="file_permissions",
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message="File write/read test failed"
                )
        except Exception as e:
            return ValidationResult(
                name="file_permissions",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"File permission validation failed: {e}"
            )
    
    def validate_memory_usage(self) -> ValidationResult:
        """Basic memory usage validation."""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Test large data structure creation
            test_data = [{"id": i, "data": f"test_data_{i}"} for i in range(1000)]
            
            if len(test_data) == 1000:
                # Clean up
                del test_data
                gc.collect()
                
                return ValidationResult(
                    name="memory_usage",
                    level=ValidationLevel.INFO,
                    passed=True,
                    message="Memory allocation test passed ✓"
                )
            else:
                return ValidationResult(
                    name="memory_usage",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message="Memory allocation test failed"
                )
        except Exception as e:
            return ValidationResult(
                name="memory_usage",
                level=ValidationLevel.INFO,
                passed=False,
                message=f"Memory validation failed: {e}"
            )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation checks."""
        validators = [
            self.validate_python_environment,
            self.validate_project_structure,
            self.validate_core_imports,
            self.validate_async_support,
            self.validate_json_handling,
            self.validate_file_permissions,
            self.validate_memory_usage
        ]
        
        results = []
        for validator in validators:
            try:
                result = validator()
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    name=validator.__name__,
                    level=ValidationLevel.CRITICAL,
                    passed=False,
                    message=f"Validator crashed: {e}"
                ))
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.results:
            return {"error": "No validations run"}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        critical_failures = [
            r for r in self.results 
            if not r.passed and r.level == ValidationLevel.CRITICAL
        ]
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total) * 100 if total > 0 else 0,
            "critical_failures": len(critical_failures),
            "ready_for_execution": len(critical_failures) == 0,
            "timestamp": time.time()
        }
    
    def print_results(self):
        """Print validation results to console."""
        print("\n" + "="*60)
        print("🔍 AGI EVALUATION SANDBOX - RUNTIME VALIDATION")
        print("="*60)
        
        if not self.results:
            print("❌ No validation results available")
            return
        
        # Group by level
        critical_results = [r for r in self.results if r.level == ValidationLevel.CRITICAL]
        warning_results = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info_results = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        # Print critical checks
        print("\n🚨 CRITICAL CHECKS:")
        for result in critical_results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name}: {result.message}")
        
        # Print warnings
        if warning_results:
            print("\n⚠️  WARNINGS:")
            for result in warning_results:
                status = "✅" if result.passed else "⚠️"
                print(f"  {status} {result.name}: {result.message}")
        
        # Print info
        if info_results:
            print("\nℹ️  INFO:")
            for result in info_results:
                status = "✅" if result.passed else "ℹ️"
                print(f"  {status} {result.name}: {result.message}")
        
        # Print summary
        summary = self.get_summary()
        print(f"\n📊 SUMMARY:")
        print(f"  Total Checks: {summary['total_checks']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['ready_for_execution']:
            print("\n🎉 ENVIRONMENT READY FOR EXECUTION!")
        else:
            print(f"\n❌ CRITICAL FAILURES: {summary['critical_failures']}")
            print("   Please resolve critical issues before proceeding.")
        
        print("="*60)

def main():
    """Main validation entry point."""
    validator = RuntimeValidator()
    validator.run_all_validations()
    validator.print_results()
    
    summary = validator.get_summary()
    
    # Return appropriate exit code
    if summary['ready_for_execution']:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)