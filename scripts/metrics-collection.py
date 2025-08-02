#!/usr/bin/env python3
"""
Automated metrics collection script for AGI Evaluation Sandbox
Collects and reports various project metrics for monitoring and improvement.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

try:
    import requests
except ImportError:
    print("Warning: requests not available. Some metrics may not be collected.")
    requests = None

try:
    import git
except ImportError:
    print("Warning: GitPython not available. Git metrics may not be collected.")
    git = None


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = None
        if git:
            try:
                self.repo = git.Repo(repo_path)
            except Exception as e:
                print(f"Warning: Could not initialize git repo: {e}")
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        if not self.repo:
            return {}
        
        try:
            # Basic repository info
            metrics = {
                "branch": self.repo.active_branch.name,
                "latest_commit": self.repo.head.commit.hexsha[:8],
                "commit_count": len(list(self.repo.iter_commits())),
                "contributor_count": len(set(commit.author.email for commit in self.repo.iter_commits())),
                "last_commit_date": self.repo.head.commit.committed_datetime.isoformat()
            }
            
            # Recent activity (last 30 days)
            since = datetime.now(timezone.utc).timestamp() - (30 * 24 * 60 * 60)
            recent_commits = list(self.repo.iter_commits(since=since))
            metrics["commits_last_30_days"] = len(recent_commits)
            
            # Files changed in recent commits
            if recent_commits:
                changed_files = set()
                for commit in recent_commits[:50]:  # Last 50 commits
                    try:
                        for item in commit.stats.files:
                            changed_files.add(item)
                    except Exception:
                        pass
                metrics["files_changed_recently"] = len(changed_files)
            
            return metrics
        except Exception as e:
            print(f"Error collecting git metrics: {e}")
            return {}
    
    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Count lines of code
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-o", "-name", "*.ts", "-o", "-name", "*.tsx", "-o", "-name", "*.js", "-o", "-name", "*.jsx"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                files = [f for f in files if f and not any(ignore in f for ignore in ['node_modules', '__pycache__', '.git', 'dist', 'build'])]
                
                total_lines = 0
                for file_path in files:
                    try:
                        with open(self.repo_path / file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += len(f.readlines())
                    except Exception:
                        pass
                
                metrics["lines_of_code"] = total_lines
                metrics["file_count"] = len(files)
        except Exception as e:
            print(f"Error counting lines of code: {e}")
        
        # Test coverage (if coverage report exists)
        coverage_file = self.repo_path / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    line_rate = coverage_elem.get("line-rate")
                    if line_rate:
                        metrics["test_coverage"] = round(float(line_rate) * 100, 2)
            except Exception as e:
                print(f"Error parsing coverage report: {e}")
        
        return metrics
    
    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker-related metrics."""
        metrics = {}
        
        # Check if Dockerfile exists
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            metrics["dockerfile_exists"] = True
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    metrics["dockerfile_lines"] = len(content.split('\n'))
                    metrics["dockerfile_multistage"] = "FROM" in content and content.count("FROM") > 1
            except Exception:
                pass
        else:
            metrics["dockerfile_exists"] = False
        
        # Check docker-compose
        compose_file = self.repo_path / "docker-compose.yml"
        if compose_file.exists():
            metrics["docker_compose_exists"] = True
            try:
                import yaml
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                    services = compose_data.get('services', {})
                    metrics["docker_compose_services"] = len(services)
            except Exception:
                pass
        else:
            metrics["docker_compose_exists"] = False
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency metrics."""
        metrics = {}
        
        # Python dependencies
        requirements_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        python_deps = 0
        
        for req_file in requirements_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        content = f.read()
                        if req_file.endswith('.toml'):
                            # Simple count for pyproject.toml
                            python_deps += content.count('"')
                        else:
                            # Count non-comment, non-empty lines
                            lines = [line.strip() for line in content.split('\n')]
                            python_deps += len([line for line in lines if line and not line.startswith('#')])
                except Exception:
                    pass
        
        metrics["python_dependencies"] = python_deps
        
        # Node.js dependencies
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    deps = len(package_data.get('dependencies', {}))
                    dev_deps = len(package_data.get('devDependencies', {}))
                    metrics["nodejs_dependencies"] = deps
                    metrics["nodejs_dev_dependencies"] = dev_deps
            except Exception:
                pass
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {}
        
        # Check for security files
        security_files = [
            "SECURITY.md",
            ".github/workflows/security.yml",
            ".github/workflows/security-scan.yml"
        ]
        
        security_file_count = 0
        for sec_file in security_files:
            if (self.repo_path / sec_file).exists():
                security_file_count += 1
        
        metrics["security_files_present"] = security_file_count
        metrics["security_files_total"] = len(security_files)
        
        # Check for secrets in code (basic check)
        try:
            result = subprocess.run(
                ["grep", "-r", "-i", "--include=*.py", "--include=*.js", "--include=*.ts", 
                 "password\\|secret\\|key\\|token", "."],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Filter out obvious false positives
                lines = result.stdout.split('\n')
                potential_secrets = [
                    line for line in lines 
                    if line and not any(safe in line.lower() for safe in [
                        'example', 'placeholder', 'test', 'mock', 'fake', 'dummy',
                        'your_', 'change_', 'replace_', 'secret_key_here'
                    ])
                ]
                metrics["potential_secrets_found"] = len(potential_secrets)
            else:
                metrics["potential_secrets_found"] = 0
                
        except Exception:
            metrics["potential_secrets_found"] = 0
        
        return metrics
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        metrics = {}
        
        # Count documentation files
        doc_extensions = ['.md', '.rst', '.txt']
        doc_files = []
        
        for ext in doc_extensions:
            try:
                result = subprocess.run(
                    ["find", ".", "-name", f"*{ext}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    files = [f for f in result.stdout.strip().split('\n') if f]
                    doc_files.extend(files)
            except Exception:
                pass
        
        metrics["documentation_files"] = len(doc_files)
        
        # Check for specific important docs
        important_docs = [
            "README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE",
            "SECURITY.md", "CODE_OF_CONDUCT.md"
        ]
        
        present_docs = 0
        for doc in important_docs:
            if (self.repo_path / doc).exists():
                present_docs += 1
        
        metrics["important_docs_present"] = present_docs
        metrics["important_docs_total"] = len(important_docs)
        
        # Calculate documentation coverage
        if metrics["documentation_files"] > 0:
            coverage = (present_docs / len(important_docs)) * 100
            metrics["documentation_coverage"] = round(coverage, 2)
        
        return metrics
    
    def collect_automation_metrics(self) -> Dict[str, Any]:
        """Collect automation and CI/CD metrics."""
        metrics = {}
        
        # GitHub Actions workflows
        workflows_dir = self.repo_path / ".github" / "workflows"
        if workflows_dir.exists():
            try:
                workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
                metrics["github_workflows"] = len(workflow_files)
                
                # Count workflow jobs
                total_jobs = 0
                for workflow_file in workflow_files:
                    try:
                        import yaml
                        with open(workflow_file, 'r') as f:
                            workflow_data = yaml.safe_load(f)
                            jobs = workflow_data.get('jobs', {})
                            total_jobs += len(jobs)
                    except Exception:
                        pass
                
                metrics["github_workflow_jobs"] = total_jobs
            except Exception:
                metrics["github_workflows"] = 0
        
        # Pre-commit hooks
        precommit_file = self.repo_path / ".pre-commit-config.yaml"
        if precommit_file.exists():
            metrics["precommit_configured"] = True
            try:
                import yaml
                with open(precommit_file, 'r') as f:
                    precommit_data = yaml.safe_load(f)
                    repos = precommit_data.get('repos', [])
                    hooks = sum(len(repo.get('hooks', [])) for repo in repos)
                    metrics["precommit_hooks"] = hooks
            except Exception:
                pass
        else:
            metrics["precommit_configured"] = False
        
        # Makefile automation
        makefile = self.repo_path / "Makefile"
        if makefile.exists():
            metrics["makefile_exists"] = True
            try:
                with open(makefile, 'r') as f:
                    content = f.read()
                    # Count targets (lines that start with word followed by :)
                    lines = content.split('\n')
                    targets = [line for line in lines if ':' in line and not line.startswith('\t') and not line.startswith('#')]
                    metrics["makefile_targets"] = len(targets)
            except Exception:
                pass
        else:
            metrics["makefile_exists"] = False
        
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        print("Collecting project metrics...")
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "agi-eval-sandbox",
            "version": "1.0.0",
            "git": self.collect_git_metrics(),
            "code": self.collect_code_metrics(),
            "docker": self.collect_docker_metrics(),
            "dependencies": self.collect_dependency_metrics(),
            "security": self.collect_security_metrics(),
            "documentation": self.collect_documentation_metrics(),
            "automation": self.collect_automation_metrics(),
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save metrics report to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Metrics report saved to: {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of key metrics."""
        print("\n" + "="*50)
        print("PROJECT METRICS SUMMARY")
        print("="*50)
        
        # Git metrics
        git_metrics = report.get("git", {})
        print(f"📊 Repository: {git_metrics.get('branch', 'unknown')} branch")
        print(f"📊 Total commits: {git_metrics.get('commit_count', 0)}")
        print(f"📊 Contributors: {git_metrics.get('contributor_count', 0)}")
        print(f"📊 Recent activity: {git_metrics.get('commits_last_30_days', 0)} commits (30 days)")
        
        # Code metrics
        code_metrics = report.get("code", {})
        print(f"\n💻 Lines of code: {code_metrics.get('lines_of_code', 0):,}")
        print(f"💻 Files: {code_metrics.get('file_count', 0)}")
        if 'test_coverage' in code_metrics:
            print(f"💻 Test coverage: {code_metrics['test_coverage']}%")
        
        # Dependencies
        deps = report.get("dependencies", {})
        print(f"\n📦 Python dependencies: {deps.get('python_dependencies', 0)}")
        print(f"📦 Node.js dependencies: {deps.get('nodejs_dependencies', 0)}")
        
        # Security
        security = report.get("security", {})
        print(f"\n🔒 Security files: {security.get('security_files_present', 0)}/{security.get('security_files_total', 0)}")
        potential_secrets = security.get('potential_secrets_found', 0)
        if potential_secrets > 0:
            print(f"⚠️  Potential secrets found: {potential_secrets} (review required)")
        else:
            print("✅ No obvious secrets detected")
        
        # Documentation
        docs = report.get("documentation", {})
        print(f"\n📚 Documentation files: {docs.get('documentation_files', 0)}")
        print(f"📚 Important docs: {docs.get('important_docs_present', 0)}/{docs.get('important_docs_total', 0)}")
        if 'documentation_coverage' in docs:
            print(f"📚 Documentation coverage: {docs['documentation_coverage']}%")
        
        # Automation
        automation = report.get("automation", {})
        print(f"\n🤖 GitHub workflows: {automation.get('github_workflows', 0)}")
        print(f"🤖 Workflow jobs: {automation.get('github_workflow_jobs', 0)}")
        print(f"🤖 Pre-commit configured: {'✅' if automation.get('precommit_configured') else '❌'}")
        print(f"🤖 Makefile: {'✅' if automation.get('makefile_exists') else '❌'}")
        
        print("\n" + "="*50)


def main():
    parser = argparse.ArgumentParser(description="Collect AGI Evaluation Sandbox project metrics")
    parser.add_argument("--output", "-o", default="metrics-report.json", help="Output file for metrics report")
    parser.add_argument("--repo-path", "-r", default=".", help="Path to repository")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.repo_path)
        report = collector.generate_report()
        
        collector.save_report(report, args.output)
        
        if not args.quiet:
            collector.print_summary(report)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())