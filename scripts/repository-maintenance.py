#!/usr/bin/env python3
"""
Repository maintenance automation script for AGI Evaluation Sandbox
Performs various maintenance tasks to keep the repository healthy.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import re

try:
    import requests
except ImportError:
    print("Warning: requests not available. Some features may not work.")
    requests = None

try:
    import git
except ImportError:
    print("Warning: GitPython not available. Git operations may not work.")
    git = None


class RepositoryMaintenance:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = None
        if git:
            try:
                self.repo = git.Repo(repo_path)
            except Exception as e:
                print(f"Warning: Could not initialize git repo: {e}")
    
    def clean_old_branches(self, dry_run: bool = True) -> List[str]:
        """Clean up old merged branches."""
        if not self.repo:
            print("Git repository not available")
            return []
        
        cleaned_branches = []
        
        try:
            # Get all remote branches that have been merged
            merged_branches = []
            for ref in self.repo.refs:
                if ref.name.startswith('origin/') and ref.name not in ['origin/main', 'origin/develop']:
                    try:
                        # Check if branch is merged
                        merge_base = self.repo.merge_base(ref, self.repo.heads.main)[0]
                        if merge_base == ref.commit:
                            branch_name = ref.name.replace('origin/', '')
                            merged_branches.append(branch_name)
                    except Exception:
                        pass
            
            for branch in merged_branches:
                if dry_run:
                    print(f"Would delete merged branch: {branch}")
                    cleaned_branches.append(branch)
                else:
                    try:
                        # Delete remote branch
                        subprocess.run(['git', 'push', 'origin', '--delete', branch], 
                                     cwd=self.repo_path, check=True, capture_output=True)
                        cleaned_branches.append(branch)
                        print(f"Deleted merged branch: {branch}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to delete branch {branch}: {e}")
            
        except Exception as e:
            print(f"Error cleaning branches: {e}")
        
        return cleaned_branches
    
    def update_dependencies(self, dry_run: bool = True) -> Dict[str, Any]:
        """Update project dependencies."""
        updates = {"python": [], "nodejs": []}
        
        # Python dependencies
        if (self.repo_path / "requirements.txt").exists():
            try:
                if dry_run:
                    result = subprocess.run(
                        ["pip", "list", "--outdated", "--format=json"],
                        cwd=self.repo_path, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        outdated = json.loads(result.stdout)
                        updates["python"] = [
                            f"{pkg['name']}: {pkg['version']} -> {pkg['latest_version']}"
                            for pkg in outdated
                        ]
                else:
                    # Update requirements
                    subprocess.run(
                        ["pip-compile", "--upgrade", "requirements.in"],
                        cwd=self.repo_path, check=True
                    )
                    print("Updated Python dependencies")
            except Exception as e:
                print(f"Error updating Python dependencies: {e}")
        
        # Node.js dependencies
        if (self.repo_path / "package.json").exists():
            try:
                if dry_run:
                    result = subprocess.run(
                        ["npm", "outdated", "--json"],
                        cwd=self.repo_path, capture_output=True, text=True
                    )
                    if result.stdout:
                        try:
                            outdated = json.loads(result.stdout)
                            updates["nodejs"] = [
                                f"{pkg}: {info['current']} -> {info['latest']}"
                                for pkg, info in outdated.items()
                            ]
                        except json.JSONDecodeError:
                            pass
                else:
                    # Update package.json
                    subprocess.run(
                        ["npx", "npm-check-updates", "-u"],
                        cwd=self.repo_path, check=True
                    )
                    subprocess.run(
                        ["npm", "install"],
                        cwd=self.repo_path, check=True
                    )
                    print("Updated Node.js dependencies")
            except Exception as e:
                print(f"Error updating Node.js dependencies: {e}")
        
        return updates
    
    def check_security_vulnerabilities(self) -> Dict[str, Any]:
        """Check for security vulnerabilities."""
        vulnerabilities = {"python": [], "nodejs": []}
        
        # Python security check
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities["python"] = safety_data
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Python security check failed: {e}")
        
        # Node.js security audit
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities["nodejs"] = audit_data.get("vulnerabilities", {})
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Node.js security audit failed: {e}")
        
        return vulnerabilities
    
    def clean_cache_files(self, dry_run: bool = True) -> List[str]:
        """Clean up cache and temporary files."""
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/node_modules/.cache",
            "**/.npm",
            "**/.cache",
            "**/dist",
            "**/build",
            "**/*.egg-info"
        ]
        
        cleaned_files = []
        
        for pattern in cache_patterns:
            try:
                for path in self.repo_path.glob(pattern):
                    if path.exists():
                        if dry_run:
                            print(f"Would remove: {path}")
                            cleaned_files.append(str(path))
                        else:
                            if path.is_file():
                                path.unlink()
                            elif path.is_dir():
                                import shutil
                                shutil.rmtree(path)
                            cleaned_files.append(str(path))
                            print(f"Removed: {path}")
            except Exception as e:
                print(f"Error cleaning {pattern}: {e}")
        
        return cleaned_files
    
    def validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files for syntax errors."""
        validation_results = {}
        
        # JSON files
        json_files = list(self.repo_path.glob("**/*.json"))
        json_files = [f for f in json_files if "node_modules" not in str(f)]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                validation_results[str(json_file)] = "valid"
            except json.JSONDecodeError as e:
                validation_results[str(json_file)] = f"invalid: {e}"
        
        # YAML files
        yaml_files = list(self.repo_path.glob("**/*.yml")) + list(self.repo_path.glob("**/*.yaml"))
        yaml_files = [f for f in yaml_files if "node_modules" not in str(f)]
        
        for yaml_file in yaml_files:
            try:
                import yaml
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
                validation_results[str(yaml_file)] = "valid"
            except Exception as e:
                validation_results[str(yaml_file)] = f"invalid: {e}"
        
        return validation_results
    
    def check_license_compliance(self) -> Dict[str, Any]:
        """Check license compliance for dependencies."""
        compliance = {"python": {}, "nodejs": {}}
        
        # Python licenses
        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                licenses = json.loads(result.stdout)
                compliance["python"] = {
                    pkg["Name"]: pkg.get("License", "Unknown")
                    for pkg in licenses
                }
        except Exception as e:
            print(f"Python license check failed: {e}")
        
        # Node.js licenses
        try:
            result = subprocess.run(
                ["license-checker", "--json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                licenses = json.loads(result.stdout)
                compliance["nodejs"] = {
                    pkg: info.get("licenses", "Unknown")
                    for pkg, info in licenses.items()
                }
        except Exception as e:
            print(f"Node.js license check failed: {e}")
        
        return compliance
    
    def generate_dependency_graph(self) -> Dict[str, Any]:
        """Generate dependency graph information."""
        graph = {"python": {}, "nodejs": {}}
        
        # Python dependency tree
        try:
            result = subprocess.run(
                ["pipdeptree", "--json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                graph["python"] = json.loads(result.stdout)
        except Exception as e:
            print(f"Python dependency tree failed: {e}")
        
        # Node.js dependency tree (simplified)
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    graph["nodejs"] = {
                        "dependencies": package_data.get("dependencies", {}),
                        "devDependencies": package_data.get("devDependencies", {})
                    }
            except Exception as e:
                print(f"Node.js dependency analysis failed: {e}")
        
        return graph
    
    def run_maintenance_checks(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run all maintenance checks."""
        print("Running repository maintenance checks...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "checks": {}
        }
        
        # Security vulnerabilities
        print("Checking security vulnerabilities...")
        results["checks"]["security"] = self.check_security_vulnerabilities()
        
        # Configuration validation
        print("Validating configuration files...")
        results["checks"]["configuration"] = self.validate_configuration_files()
        
        # License compliance
        print("Checking license compliance...")
        results["checks"]["licenses"] = self.check_license_compliance()
        
        # Dependency updates
        print("Checking for dependency updates...")
        results["checks"]["dependency_updates"] = self.update_dependencies(dry_run)
        
        # Dependency graph
        print("Generating dependency information...")
        results["checks"]["dependency_graph"] = self.generate_dependency_graph()
        
        # Clean cache files
        print("Checking for cache files to clean...")
        results["checks"]["cache_cleanup"] = self.clean_cache_files(dry_run)
        
        # Clean old branches
        if self.repo:
            print("Checking for old branches to clean...")
            results["checks"]["branch_cleanup"] = self.clean_old_branches(dry_run)
        
        return results
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable maintenance report."""
        report = []
        report.append("=" * 60)
        report.append("REPOSITORY MAINTENANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Mode: {'DRY RUN' if results['dry_run'] else 'EXECUTION'}")
        report.append("")
        
        # Security vulnerabilities
        security = results["checks"].get("security", {})
        python_vulns = len(security.get("python", []))
        nodejs_vulns = len(security.get("nodejs", {}))
        
        report.append("🔒 SECURITY VULNERABILITIES")
        report.append("-" * 30)
        report.append(f"Python vulnerabilities: {python_vulns}")
        report.append(f"Node.js vulnerabilities: {nodejs_vulns}")
        
        if python_vulns > 0 or nodejs_vulns > 0:
            report.append("⚠️  ACTION REQUIRED: Review and fix vulnerabilities")
        else:
            report.append("✅ No known vulnerabilities found")
        report.append("")
        
        # Configuration validation
        config = results["checks"].get("configuration", {})
        invalid_configs = [f for f, status in config.items() if status != "valid"]
        
        report.append("📋 CONFIGURATION VALIDATION")
        report.append("-" * 30)
        report.append(f"Total files checked: {len(config)}")
        report.append(f"Invalid files: {len(invalid_configs)}")
        
        if invalid_configs:
            report.append("⚠️  Invalid configuration files:")
            for file in invalid_configs[:5]:  # Show first 5
                report.append(f"   - {file}")
            if len(invalid_configs) > 5:
                report.append(f"   ... and {len(invalid_configs) - 5} more")
        else:
            report.append("✅ All configuration files are valid")
        report.append("")
        
        # Dependency updates
        updates = results["checks"].get("dependency_updates", {})
        python_updates = len(updates.get("python", []))
        nodejs_updates = len(updates.get("nodejs", []))
        
        report.append("📦 DEPENDENCY UPDATES")
        report.append("-" * 30)
        report.append(f"Python packages to update: {python_updates}")
        report.append(f"Node.js packages to update: {nodejs_updates}")
        
        if python_updates > 0 or nodejs_updates > 0:
            report.append("📈 Updates available - consider updating dependencies")
        else:
            report.append("✅ All dependencies are up to date")
        report.append("")
        
        # Cache cleanup
        cache_files = results["checks"].get("cache_cleanup", [])
        report.append("🧹 CACHE CLEANUP")
        report.append("-" * 30)
        report.append(f"Cache files to clean: {len(cache_files)}")
        
        if cache_files:
            if results["dry_run"]:
                report.append("💡 Run with --execute to clean cache files")
            else:
                report.append("✅ Cache files cleaned")
        else:
            report.append("✅ No cache files to clean")
        report.append("")
        
        # Branch cleanup
        branches = results["checks"].get("branch_cleanup", [])
        if branches is not None:
            report.append("🌿 BRANCH CLEANUP")
            report.append("-" * 30)
            report.append(f"Merged branches to clean: {len(branches)}")
            
            if branches:
                if results["dry_run"]:
                    report.append("💡 Run with --execute to clean merged branches")
                else:
                    report.append("✅ Merged branches cleaned")
            else:
                report.append("✅ No merged branches to clean")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--execute", action="store_true", help="Execute maintenance tasks (default is dry run)")
    parser.add_argument("--output", "-o", help="Output file for maintenance report")
    parser.add_argument("--repo-path", "-r", default=".", help="Path to repository")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    try:
        maintenance = RepositoryMaintenance(args.repo_path)
        results = maintenance.run_maintenance_checks(dry_run=not args.execute)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Maintenance report saved to: {output_path}")
        
        if not args.quiet:
            report = maintenance.generate_maintenance_report(results)
            print(report)
        
        # Exit with error code if issues found
        security = results["checks"].get("security", {})
        python_vulns = len(security.get("python", []))
        nodejs_vulns = len(security.get("nodejs", {}))
        
        config = results["checks"].get("configuration", {})
        invalid_configs = [f for f, status in config.items() if status != "valid"]
        
        if python_vulns > 0 or nodejs_vulns > 0 or invalid_configs:
            return 1
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())