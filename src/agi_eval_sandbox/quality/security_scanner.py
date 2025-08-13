"""Security scanning and validation for AGI Evaluation Sandbox."""

import os
import re
import ast
import hashlib
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import tempfile

from ..core.logging_config import get_logger, security_logger
from ..core.exceptions import SecurityError

logger = get_logger("security_scanner")


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str  # "injection", "crypto", "secrets", "permissions", etc.
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


@dataclass
class SecurityScanResult:
    """Results of a security scan."""
    scan_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    issues: List[SecurityIssue] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration_seconds: float = 0.0
    
    @property
    def critical_issues(self) -> List[SecurityIssue]:
        return [i for i in self.issues if i.severity == "critical"]
    
    @property
    def high_issues(self) -> List[SecurityIssue]:
        return [i for i in self.issues if i.severity == "high"]
    
    @property
    def total_high_critical(self) -> int:
        return len(self.critical_issues) + len(self.high_issues)
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        weights = {"critical": 10, "high": 7, "medium": 4, "low": 2, "info": 1}
        total_score = sum(weights.get(issue.severity, 0) for issue in self.issues)
        # Normalize to 0-100 scale
        max_possible = len(self.issues) * 10 if self.issues else 1
        return min(100.0, (total_score / max_possible) * 100)


class SecurityScanner:
    """Comprehensive security scanner for code and configuration."""
    
    def __init__(self):
        self.patterns = self._load_security_patterns()
        self.secret_patterns = self._load_secret_patterns()
        self.safe_functions = {
            'subprocess.run', 'subprocess.Popen', 'os.system', 'eval', 'exec',
            'compile', 'open', '__import__', 'getattr', 'setattr', 'delattr',
            'pickle.loads', 'pickle.load', 'yaml.load', 'yaml.unsafe_load'
        }
    
    def _load_security_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security patterns for different vulnerability types."""
        return {
            "injection": [
                {
                    "pattern": r"subprocess\.(run|call|Popen|check_output)\s*\([^)]*shell\s*=\s*True",
                    "severity": "high",
                    "title": "Command Injection via shell=True",
                    "description": "Using shell=True with subprocess can lead to command injection",
                    "cwe_id": "CWE-78",
                    "remediation": "Use shell=False and pass command as list of arguments"
                },
                {
                    "pattern": r"os\.system\s*\(",
                    "severity": "high", 
                    "title": "Command Injection via os.system",
                    "description": "os.system() is vulnerable to command injection",
                    "cwe_id": "CWE-78",
                    "remediation": "Use subprocess.run() with shell=False instead"
                },
                {
                    "pattern": r"eval\s*\(",
                    "severity": "critical",
                    "title": "Code Injection via eval()",
                    "description": "eval() can execute arbitrary code",
                    "cwe_id": "CWE-94",
                    "remediation": "Use ast.literal_eval() for safe evaluation or avoid eval()"
                },
                {
                    "pattern": r"exec\s*\(",
                    "severity": "critical",
                    "title": "Code Injection via exec()",
                    "description": "exec() can execute arbitrary code",
                    "cwe_id": "CWE-94",
                    "remediation": "Avoid exec() or use strict input validation"
                }
            ],
            "crypto": [
                {
                    "pattern": r"hashlib\.md5\s*\(",
                    "severity": "medium",
                    "title": "Weak Cryptographic Hash (MD5)",
                    "description": "MD5 is cryptographically broken",
                    "cwe_id": "CWE-327",
                    "remediation": "Use SHA-256 or stronger hash algorithms"
                },
                {
                    "pattern": r"hashlib\.sha1\s*\(",
                    "severity": "medium",
                    "title": "Weak Cryptographic Hash (SHA1)",
                    "description": "SHA1 is considered weak",
                    "cwe_id": "CWE-327",
                    "remediation": "Use SHA-256 or stronger hash algorithms"
                },
                {
                    "pattern": r"random\.random\s*\(|random\.randint\s*\(",
                    "severity": "low",
                    "title": "Weak Random Number Generator",
                    "description": "random module is not cryptographically secure",
                    "cwe_id": "CWE-338",
                    "remediation": "Use secrets module for cryptographic operations"
                }
            ],
            "deserialization": [
                {
                    "pattern": r"pickle\.loads?\s*\(",
                    "severity": "high",
                    "title": "Unsafe Deserialization",
                    "description": "pickle.load() can execute arbitrary code",
                    "cwe_id": "CWE-502",
                    "remediation": "Use json or implement safe deserialization"
                },
                {
                    "pattern": r"yaml\.load\s*\((?!.*Loader=yaml\.SafeLoader)",
                    "severity": "high",
                    "title": "Unsafe YAML Deserialization",
                    "description": "yaml.load() without SafeLoader can execute arbitrary code",
                    "cwe_id": "CWE-502",
                    "remediation": "Use yaml.safe_load() or yaml.load() with SafeLoader"
                }
            ],
            "path_traversal": [
                {
                    "pattern": r"open\s*\([^)]*\.\./",
                    "severity": "medium",
                    "title": "Potential Path Traversal",
                    "description": "File path contains '../' which could lead to path traversal",
                    "cwe_id": "CWE-22",
                    "remediation": "Validate and sanitize file paths"
                }
            ]
        }
    
    def _load_secret_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for detecting secrets and sensitive data."""
        return [
            {
                "pattern": r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]([a-zA-Z0-9]{20,})['\"]",
                "severity": "critical",
                "title": "API Key in Code",
                "description": "Hardcoded API key found",
                "cwe_id": "CWE-798"
            },
            {
                "pattern": r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"]([a-zA-Z0-9]{16,})['\"]",
                "severity": "critical",
                "title": "Secret Key in Code",
                "description": "Hardcoded secret key found",
                "cwe_id": "CWE-798"
            },
            {
                "pattern": r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]([a-zA-Z0-9]{8,})['\"]",
                "severity": "high",
                "title": "Password in Code",
                "description": "Hardcoded password found",
                "cwe_id": "CWE-798"
            },
            {
                "pattern": r"(?i)(token)\s*[=:]\s*['\"]([a-zA-Z0-9]{20,})['\"]",
                "severity": "high",
                "title": "Token in Code",
                "description": "Hardcoded token found",
                "cwe_id": "CWE-798"
            },
            {
                "pattern": r"-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----",
                "severity": "critical",
                "title": "Private Key in Code",
                "description": "Private key found in source code",
                "cwe_id": "CWE-798"
            }
        ]
    
    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Scan for pattern-based vulnerabilities
            issues.extend(self._scan_patterns(file_path, content))
            
            # Scan for secrets
            issues.extend(self._scan_secrets(file_path, content))
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                issues.extend(self._scan_ast(file_path, content))
            
        except Exception as e:
            logger.warning(f"Failed to scan file {file_path}: {e}")
            issues.append(SecurityIssue(
                severity="info",
                category="scan_error",
                title="File Scan Error",
                description=f"Could not scan file: {str(e)}",
                file_path=str(file_path)
            ))
        
        return issues
    
    def _scan_patterns(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan content using regex patterns."""
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        issues.append(SecurityIssue(
                            severity=pattern_info["severity"],
                            category=category,
                            title=pattern_info["title"],
                            description=pattern_info["description"],
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            remediation=pattern_info.get("remediation"),
                            cwe_id=pattern_info.get("cwe_id")
                        ))
        
        return issues
    
    def _scan_secrets(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan for hardcoded secrets and sensitive data."""
        issues = []
        lines = content.split('\n')
        
        for pattern_info in self.secret_patterns:
            pattern = pattern_info["pattern"]
            
            for line_num, line in enumerate(lines, 1):
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Obfuscate the secret in the description
                    secret_value = match.group(2) if len(match.groups()) >= 2 else match.group(0)
                    obfuscated = secret_value[:4] + "*" * (len(secret_value) - 4)
                    
                    issues.append(SecurityIssue(
                        severity=pattern_info["severity"],
                        category="secrets",
                        title=pattern_info["title"],
                        description=f"{pattern_info['description']}: {obfuscated}",
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.replace(secret_value, obfuscated),
                        cwe_id=pattern_info.get("cwe_id")
                    ))
        
        return issues
    
    def _scan_ast(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Advanced AST-based security analysis for Python files."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Attribute):
                        func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                    elif isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    else:
                        func_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else str(node.func)
                    
                    # Check for shell=True in subprocess calls
                    if "subprocess" in func_name:
                        for keyword in node.keywords:
                            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                                if keyword.value.value is True:
                                    self.issues.append(SecurityIssue(
                                        severity="high",
                                        category="injection",
                                        title="subprocess with shell=True",
                                        description="Subprocess call with shell=True detected",
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        cwe_id="CWE-78"
                                    ))
                    
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    # Check for dangerous imports
                    for alias in node.names:
                        if alias.name in ['pickle', 'cPickle', 'dill']:
                            self.issues.append(SecurityIssue(
                                severity="medium",
                                category="deserialization",
                                title="Dangerous Import",
                                description=f"Import of potentially unsafe module: {alias.name}",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                cwe_id="CWE-502"
                            ))
                    
                    self.generic_visit(node)
            
            visitor = SecurityVisitor()
            visitor.visit(tree)
            issues.extend(visitor.issues)
            
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"AST analysis failed for {file_path}: {e}")
        
        return issues
    
    def scan_directory(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> SecurityScanResult:
        """Scan an entire directory for security issues."""
        scan_id = hashlib.sha256(f"{directory}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        start_time = datetime.now()
        
        exclude_patterns = exclude_patterns or [
            "*.pyc", "__pycache__", ".git", ".venv", "venv", "node_modules",
            "*.log", "*.tmp", ".pytest_cache"
        ]
        
        all_issues = []
        files_scanned = 0
        
        # Get all Python files
        for file_path in directory.rglob("*.py"):
            # Skip excluded patterns
            if any(file_path.match(pattern) for pattern in exclude_patterns):
                continue
            
            if file_path.is_file():
                issues = self.scan_file(file_path)
                all_issues.extend(issues)
                files_scanned += 1
        
        # Also scan configuration files
        config_files = list(directory.rglob("*.json")) + list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
        for file_path in config_files:
            if any(file_path.match(pattern) for pattern in exclude_patterns):
                continue
            
            if file_path.is_file():
                issues = self.scan_file(file_path)
                all_issues.extend(issues)
                files_scanned += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = SecurityScanResult(
            scan_id=scan_id,
            timestamp=start_time,
            issues=all_issues,
            files_scanned=files_scanned,
            scan_duration_seconds=duration
        )
        
        # Log scan results
        logger.info(
            f"Security scan completed: {files_scanned} files, "
            f"{len(all_issues)} issues found, "
            f"{result.total_high_critical} high/critical issues"
        )
        
        # Log critical issues to security logger
        for issue in result.critical_issues:
            security_logger.log_security_violation(
                f"Critical security issue: {issue.title}",
                {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "category": issue.category,
                    "cwe_id": issue.cwe_id
                }
            )
        
        return result
    
    def generate_report(self, scan_result: SecurityScanResult, output_format: str = "json") -> str:
        """Generate a security scan report."""
        if output_format == "json":
            return self._generate_json_report(scan_result)
        elif output_format == "html":
            return self._generate_html_report(scan_result)
        elif output_format == "markdown":
            return self._generate_markdown_report(scan_result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_json_report(self, scan_result: SecurityScanResult) -> str:
        """Generate JSON security report."""
        report_data = {
            "scan_id": scan_result.scan_id,
            "timestamp": scan_result.timestamp.isoformat(),
            "summary": {
                "files_scanned": scan_result.files_scanned,
                "total_issues": len(scan_result.issues),
                "critical_issues": len(scan_result.critical_issues),
                "high_issues": len(scan_result.high_issues),
                "risk_score": scan_result.risk_score,
                "scan_duration_seconds": scan_result.scan_duration_seconds
            },
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "code_snippet": issue.code_snippet,
                    "remediation": issue.remediation,
                    "cwe_id": issue.cwe_id
                }
                for issue in scan_result.issues
            ]
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_markdown_report(self, scan_result: SecurityScanResult) -> str:
        """Generate Markdown security report."""
        report = f"""# Security Scan Report

**Scan ID:** {scan_result.scan_id}  
**Timestamp:** {scan_result.timestamp.isoformat()}  
**Files Scanned:** {scan_result.files_scanned}  
**Scan Duration:** {scan_result.scan_duration_seconds:.2f} seconds  

## Summary

- **Total Issues:** {len(scan_result.issues)}
- **Critical Issues:** {len(scan_result.critical_issues)}
- **High Issues:** {len(scan_result.high_issues)}
- **Risk Score:** {scan_result.risk_score:.1f}/100

## Issues by Severity

"""
        
        # Group issues by severity
        severity_groups = {}
        for issue in scan_result.issues:
            if issue.severity not in severity_groups:
                severity_groups[issue.severity] = []
            severity_groups[issue.severity].append(issue)
        
        for severity in ["critical", "high", "medium", "low", "info"]:
            if severity in severity_groups:
                issues = severity_groups[severity]
                report += f"### {severity.title()} ({len(issues)} issues)\n\n"
                
                for issue in issues:
                    report += f"**{issue.title}**\n"
                    report += f"- **File:** {issue.file_path}:{issue.line_number or 'N/A'}\n"
                    report += f"- **Category:** {issue.category}\n"
                    report += f"- **Description:** {issue.description}\n"
                    if issue.remediation:
                        report += f"- **Remediation:** {issue.remediation}\n"
                    if issue.cwe_id:
                        report += f"- **CWE ID:** {issue.cwe_id}\n"
                    report += "\n"
        
        return report


# Global security scanner instance
security_scanner = SecurityScanner()