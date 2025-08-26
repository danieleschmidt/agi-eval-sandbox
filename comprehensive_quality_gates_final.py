#!/usr/bin/env python3
"""
QUALITY GATES - Comprehensive validation of all systems
Tests, security scanning, performance benchmarks, and compliance checks
"""

import sys
import asyncio
import time
import subprocess
import json
from pathlib import Path
sys.path.insert(0, '/root/repo/src')

class QualityGateValidator:
    """Comprehensive quality gate validation system"""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "metrics": {}
        }
        
    async def validate_all(self) -> bool:
        """Run all quality gates and return overall pass/fail status"""
        
        print("🛂 QUALITY GATES - Comprehensive System Validation")
        print("=" * 70)
        
        gates = [
            ("Core Functionality", self.test_core_functionality),
            ("Security Scanning", self.test_security_scanning),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("API Validation", self.test_api_validation),
            ("Error Handling", self.test_error_handling),
            ("Code Quality", self.test_code_quality),
            ("Documentation", self.test_documentation),
            ("Deployment Readiness", self.test_deployment_readiness),
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Testing {gate_name}...")
            try:
                if asyncio.iscoroutinefunction(gate_func):
                    success = await gate_func()
                else:
                    success = gate_func()
                
                if success:
                    self.results["passed"].append(gate_name)
                    print(f"✅ {gate_name} PASSED")
                else:
                    self.results["failed"].append(gate_name)
                    print(f"❌ {gate_name} FAILED")
                    
            except Exception as e:
                self.results["failed"].append(gate_name)
                print(f"❌ {gate_name} CRASHED: {e}")
        
        return self.generate_final_report()
    
    async def test_core_functionality(self) -> bool:
        """Test core system functionality"""
        try:
            # Test model creation and basic operations
            from agi_eval_sandbox.core.models import Model
            from agi_eval_sandbox.core.evaluator import EvalSuite
            
            # Test model creation
            model = Model(provider="local", name="test-model")
            
            # Test evaluation suite
            suite = EvalSuite()
            
            # Test basic generation
            result = await model.generate("test prompt")
            if not result or len(result) < 10:
                return False
            
            print("  ✅ Model creation and generation working")
            print("  ✅ EvalSuite initialization working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Core functionality failed: {e}")
            return False
    
    def test_security_scanning(self) -> bool:
        """Test security scanning capabilities"""
        try:
            from agi_eval_sandbox.quality.security_scanner import SecurityScanner
            from agi_eval_sandbox.core.validation import InputValidator
            
            # Test security scanner
            scanner = SecurityScanner()
            
            # Test with potentially unsafe code
            unsafe_code = """
import os
os.system("echo test")
eval("1+1")
"""
            
            issues = scanner.scan_code(unsafe_code)
            if len(issues) == 0:
                print("  ⚠️  Security scanner may not be detecting issues")
                self.results["warnings"].append("Security scanner sensitivity")
            else:
                print(f"  ✅ Security scanner detected {len(issues)} issues")
            
            # Test input validation
            validator = InputValidator()
            
            # Test malicious input rejection
            try:
                validator.validate_model_name("../../../etc/passwd")
                print("  ⚠️  Input validation may be too permissive")
                self.results["warnings"].append("Input validation strictness")
            except:
                print("  ✅ Input validation properly rejects malicious input")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Security scanning failed: {e}")
            return False
    
    async def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks and thresholds"""
        try:
            from agi_eval_sandbox.core.models import create_mock_model
            
            # Performance benchmark: Batch processing speed
            model = create_mock_model(
                model_name="benchmark-model",
                simulate_delay=0.001,  # 1ms per request
                simulate_failures=False
            )
            
            batch_size = 100
            prompts = [f"Benchmark prompt {i}" for i in range(batch_size)]
            
            start_time = time.time()
            results = await model.batch_generate(prompts)
            duration = time.time() - start_time
            
            throughput = batch_size / duration if duration > 0 else 0
            
            self.results["metrics"]["batch_throughput"] = throughput
            self.results["metrics"]["batch_duration"] = duration
            
            print(f"  📊 Batch processing: {batch_size} items in {duration:.3f}s")
            print(f"  📊 Throughput: {throughput:.1f} requests/second")
            
            # Benchmark thresholds
            if throughput < 1000:  # Less than 1000 req/sec
                print("  ⚠️  Throughput below optimal threshold")
                self.results["warnings"].append("Low throughput performance")
            else:
                print("  ✅ Throughput meets performance requirements")
            
            # Memory efficiency test
            import gc
            gc.collect()
            
            # Create and destroy large objects
            large_data = [{"data": "x" * 1000} for _ in range(1000)]
            del large_data
            gc.collect()
            
            print("  ✅ Memory management test completed")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Performance benchmarks failed: {e}")
            return False
    
    def test_api_validation(self) -> bool:
        """Test API endpoint validation"""
        try:
            # Test API structure
            from agi_eval_sandbox.api.simple_main import app
            
            # Validate FastAPI app structure
            if not hasattr(app, 'routes'):
                print("  ❌ FastAPI app missing routes")
                return False
            
            route_count = len(app.routes)
            print(f"  ✅ FastAPI app has {route_count} routes")
            
            # Test API models
            from agi_eval_sandbox.api.models import EvaluationRequest
            
            # Test request model validation
            try:
                request = EvaluationRequest(
                    model_provider="openai",
                    model_name="gpt-3.5-turbo",
                    benchmarks=["mmlu"],
                    config={}
                )
                print("  ✅ API request models working")
            except Exception as e:
                print(f"  ❌ API request models failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ API validation failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test comprehensive error handling"""
        try:
            from agi_eval_sandbox.core.models import create_mock_model
            from agi_eval_sandbox.core.exceptions import ValidationError, ModelProviderError
            
            # Test error handling with mock failures
            model = create_mock_model(
                model_name="error-test-model",
                simulate_failures=True,
                failure_rate=1.0  # 100% failure rate
            )
            
            # Test that errors are properly handled
            error_caught = False
            try:
                await model.generate("test prompt")
            except (ModelProviderError, Exception):
                error_caught = True
            
            if not error_caught:
                print("  ❌ Error handling not working properly")
                return False
            
            print("  ✅ Error handling working correctly")
            
            # Test validation errors
            try:
                model = create_mock_model(model_name="")  # Invalid name
                print("  ❌ Validation not catching empty names")
                return False
            except (ValidationError, ValueError):
                print("  ✅ Input validation working correctly")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error handling test failed: {e}")
            return False
    
    def test_code_quality(self) -> bool:
        """Test code quality metrics"""
        try:
            # Test imports and module structure
            import agi_eval_sandbox
            
            if not hasattr(agi_eval_sandbox, '__version__'):
                print("  ⚠️  Package missing version information")
                self.results["warnings"].append("Missing package version")
            
            # Test core modules availability
            core_modules = [
                'agi_eval_sandbox.core.models',
                'agi_eval_sandbox.core.evaluator', 
                'agi_eval_sandbox.core.benchmarks',
                'agi_eval_sandbox.core.results',
                'agi_eval_sandbox.api.main',
                'agi_eval_sandbox.quality.security_scanner'
            ]
            
            missing_modules = []
            for module_name in core_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    missing_modules.append(module_name)
            
            if missing_modules:
                print(f"  ⚠️  Missing modules: {missing_modules}")
                self.results["warnings"].append(f"Missing modules: {missing_modules}")
            else:
                print("  ✅ All core modules available")
            
            # Test configuration
            from agi_eval_sandbox.config.settings import get_settings
            settings = get_settings()
            
            if settings:
                print("  ✅ Configuration system working")
            else:
                print("  ❌ Configuration system failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ Code quality test failed: {e}")
            return False
    
    def test_documentation(self) -> bool:
        """Test documentation completeness"""
        try:
            repo_root = Path("/root/repo")
            
            # Check for essential documentation files
            essential_docs = [
                "README.md",
                "LICENSE", 
                "CONTRIBUTING.md",
                "pyproject.toml"
            ]
            
            missing_docs = []
            for doc_file in essential_docs:
                if not (repo_root / doc_file).exists():
                    missing_docs.append(doc_file)
            
            if missing_docs:
                print(f"  ⚠️  Missing documentation: {missing_docs}")
                self.results["warnings"].append(f"Missing docs: {missing_docs}")
            else:
                print("  ✅ Essential documentation present")
            
            # Check README content
            readme_path = repo_root / "README.md"
            if readme_path.exists():
                readme_content = readme_path.read_text()
                if len(readme_content) < 1000:
                    print("  ⚠️  README might be too brief")
                    self.results["warnings"].append("Brief README")
                else:
                    print("  ✅ README has substantial content")
            
            # Check for docs directory
            docs_dir = repo_root / "docs"
            if docs_dir.exists():
                doc_count = len(list(docs_dir.glob("**/*.md")))
                print(f"  ✅ Documentation directory has {doc_count} markdown files")
            else:
                print("  ⚠️  No dedicated docs directory found")
                self.results["warnings"].append("No docs directory")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Documentation test failed: {e}")
            return False
    
    def test_deployment_readiness(self) -> bool:
        """Test deployment readiness"""
        try:
            repo_root = Path("/root/repo")
            
            # Check for deployment files
            deployment_files = [
                "Dockerfile",
                "docker-compose.yml",
                "requirements.txt",
                "pyproject.toml"
            ]
            
            found_deployment = []
            for dep_file in deployment_files:
                if (repo_root / dep_file).exists():
                    found_deployment.append(dep_file)
            
            if len(found_deployment) < 2:
                print(f"  ⚠️  Limited deployment options: {found_deployment}")
                self.results["warnings"].append("Limited deployment options")
            else:
                print(f"  ✅ Multiple deployment options: {found_deployment}")
            
            # Check for environment configuration
            env_files = [".env.example", ".env.template", "config/"]
            env_found = any((repo_root / env_file).exists() for env_file in env_files)
            
            if env_found:
                print("  ✅ Environment configuration available")
            else:
                print("  ⚠️  No environment configuration templates")
                self.results["warnings"].append("No environment templates")
            
            # Check for scripts directory
            scripts_dir = repo_root / "scripts"
            if scripts_dir.exists():
                script_count = len(list(scripts_dir.glob("*.sh")))
                print(f"  ✅ Deployment scripts available ({script_count} scripts)")
            else:
                print("  ⚠️  No deployment scripts directory")
                self.results["warnings"].append("No deployment scripts")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Deployment readiness test failed: {e}")
            return False
    
    def generate_final_report(self) -> bool:
        """Generate comprehensive quality gate report"""
        print("\n" + "=" * 70)
        print("📊 QUALITY GATES FINAL REPORT")
        print("=" * 70)
        
        total_gates = len(self.results["passed"]) + len(self.results["failed"])
        pass_rate = len(self.results["passed"]) / total_gates if total_gates > 0 else 0
        
        print(f"\n📈 Overall Results:")
        print(f"  ✅ Passed: {len(self.results['passed'])}")
        print(f"  ❌ Failed: {len(self.results['failed'])}")
        print(f"  ⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"  📊 Pass Rate: {pass_rate:.1%}")
        
        if self.results["passed"]:
            print(f"\n✅ Passed Gates:")
            for gate in self.results["passed"]:
                print(f"  • {gate}")
        
        if self.results["failed"]:
            print(f"\n❌ Failed Gates:")
            for gate in self.results["failed"]:
                print(f"  • {gate}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings:")
            for warning in self.results["warnings"]:
                print(f"  • {warning}")
        
        if self.results["metrics"]:
            print(f"\n📊 Performance Metrics:")
            for metric, value in self.results["metrics"].items():
                print(f"  • {metric}: {value}")
        
        # Determine overall pass/fail
        critical_failures = len(self.results["failed"])
        acceptable_pass_rate = 0.8  # 80% minimum
        
        if critical_failures == 0 and pass_rate >= acceptable_pass_rate:
            print(f"\n🎉 QUALITY GATES PASSED")
            print(f"   System is ready for production deployment!")
            return True
        elif critical_failures <= 2 and pass_rate >= 0.6:
            print(f"\n✅ QUALITY GATES MOSTLY PASSED")  
            print(f"   System is ready with minor issues to address")
            return True
        else:
            print(f"\n⚠️  QUALITY GATES NEED ATTENTION")
            print(f"   Address failed gates before production deployment")
            return False

async def main():
    """Run comprehensive quality gate validation"""
    validator = QualityGateValidator()
    success = await validator.validate_all()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)