#!/usr/bin/env python3
"""
Generation 2: Robust Validation and Security System
Comprehensive input validation, security scanning, and data protection
"""

import asyncio
import re
import hashlib
import secrets
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check."""
    field: str
    valid: bool
    severity: ValidationSeverity
    message: str
    suggestions: List[str]
    sanitized_value: Optional[Any] = None

@dataclass
class SecurityScanResult:
    """Result of security scanning."""
    scan_type: str
    threat_level: SecurityLevel
    threats_found: List[Dict[str, Any]]
    recommendations: List[str]
    scan_duration_ms: float
    timestamp: datetime

class RobustInputValidator:
    """Comprehensive input validation with security features."""
    
    def __init__(self):
        self.logger = logging.getLogger("input_validator")
        
        # Security patterns to detect
        self.security_patterns = {
            "sql_injection": [
                r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                r"(\bdrop\b.*\btable\b)|(\btable\b.*\bdrop\b)",
                r"(\binsert\b.*\binto\b)|(\binto\b.*\binsert\b)",
                r"(\bdelete\b.*\bfrom\b)|(\bfrom\b.*\bdelete\b)",
                r"(\bupdate\b.*\bset\b)|(\bset\b.*\bupdate\b)",
                r"['\"];.*(\b(or|and)\b.*['\"].*=.*['\"])",
                r"['\"].*\b(or|and)\b.*['\"].*[=<>]"
            ],
            "xss_injection": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
                r"<object[^>]*>.*?</object>",
                r"<embed[^>]*>.*?</embed>"
            ],
            "command_injection": [
                r"[;&|`$(){}[\]\\]",
                r"\b(eval|exec|system|shell_exec|passthru)\b",
                r"['\"].*[;&|`].*['\"]",
                r"\$\{.*\}",
                r"`.*`"
            ],
            "path_traversal": [
                r"\.\./.*",
                r"\.\.\\.*",
                r"~[/\\]",
                r"/etc/passwd",
                r"/proc/.*",
                r"C:\\Windows\\System32"
            ],
            "code_injection": [
                r"\b(import|exec|eval|compile|__import__)\b",
                r"['\"].*\b(exec|eval)\b.*['\"]",
                r"getattr\s*\(",
                r"setattr\s*\(",
                r"__.*__"
            ]
        }
        
        # Validation rules
        self.validation_rules = {
            "api_key": {
                "min_length": 10,
                "max_length": 200,
                "pattern": r"^[a-zA-Z0-9_\-\.]+$",
                "required": True
            },
            "model_name": {
                "min_length": 1,
                "max_length": 100,
                "pattern": r"^[a-zA-Z0-9_\-\.]+$",
                "required": True
            },
            "provider": {
                "allowed_values": ["openai", "anthropic", "local", "huggingface", "google"],
                "required": True
            },
            "benchmark_name": {
                "min_length": 1,
                "max_length": 100,
                "pattern": r"^[a-zA-Z0-9_\-]+$",
                "required": True
            },
            "temperature": {
                "min_value": 0.0,
                "max_value": 2.0,
                "type": float
            },
            "max_tokens": {
                "min_value": 1,
                "max_value": 32000,
                "type": int
            },
            "prompt": {
                "min_length": 1,
                "max_length": 50000,
                "type": str
            }
        }
    
    def sanitize_input(self, value: Any, field_name: str) -> Any:
        """Sanitize input value to remove potentially dangerous content."""
        if not isinstance(value, str):
            return value
        
        sanitized = value
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove/escape special characters for specific fields
        if field_name in ["model_name", "benchmark_name", "provider"]:
            # Only allow alphanumeric, underscore, hyphen, and dot
            sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '', sanitized)
        
        elif field_name == "prompt":
            # For prompts, remove script tags and dangerous HTML
            for pattern in self.security_patterns["xss_injection"]:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        elif field_name == "api_key":
            # For API keys, remove any non-allowed characters
            sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def detect_security_threats(self, value: str, field_name: str) -> List[Dict[str, Any]]:
        """Detect potential security threats in input."""
        if not isinstance(value, str):
            return []
        
        threats = []
        value_lower = value.lower()
        
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, value_lower, re.IGNORECASE | re.DOTALL)
                if matches:
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "matches": matches,
                        "severity": self._get_threat_severity(threat_type),
                        "field": field_name
                    })
        
        return threats
    
    def _get_threat_severity(self, threat_type: str) -> SecurityLevel:
        """Get severity level for threat type."""
        severity_map = {
            "sql_injection": SecurityLevel.HIGH,
            "xss_injection": SecurityLevel.HIGH,
            "command_injection": SecurityLevel.CRITICAL,
            "path_traversal": SecurityLevel.HIGH,
            "code_injection": SecurityLevel.CRITICAL
        }
        return severity_map.get(threat_type, SecurityLevel.MEDIUM)
    
    def validate_field(self, value: Any, field_name: str) -> ValidationResult:
        """Validate a single field with comprehensive checks."""
        rules = self.validation_rules.get(field_name, {})
        suggestions = []
        
        # Check if required
        if rules.get("required", False) and (value is None or value == ""):
            return ValidationResult(
                field=field_name,
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} is required",
                suggestions=["Provide a valid value for this field"]
            )
        
        if value is None or value == "":
            return ValidationResult(
                field=field_name,
                valid=True,
                severity=ValidationSeverity.INFO,
                message=f"{field_name} is empty (optional)",
                suggestions=[]
            )
        
        # Type validation
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            try:
                value = expected_type(value)
            except (ValueError, TypeError):
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be of type {expected_type.__name__}",
                    suggestions=[f"Convert value to {expected_type.__name__}"]
                )
        
        # String-specific validations
        if isinstance(value, str):
            # Length validation
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")
            
            if min_length and len(value) < min_length:
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be at least {min_length} characters",
                    suggestions=[f"Increase length to at least {min_length} characters"]
                )
            
            if max_length and len(value) > max_length:
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be at most {max_length} characters",
                    suggestions=[f"Reduce length to at most {max_length} characters"]
                )
            
            # Pattern validation
            pattern = rules.get("pattern")
            if pattern and not re.match(pattern, value):
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} does not match required pattern",
                    suggestions=["Use only allowed characters", f"Pattern: {pattern}"]
                )
            
            # Security threat detection
            threats = self.detect_security_threats(value, field_name)
            if threats:
                critical_threats = [t for t in threats if t["severity"] == SecurityLevel.CRITICAL]
                high_threats = [t for t in threats if t["severity"] == SecurityLevel.HIGH]
                
                if critical_threats:
                    return ValidationResult(
                        field=field_name,
                        valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Critical security threats detected in {field_name}",
                        suggestions=["Remove potentially malicious content", "Use safe input only"]
                    )
                elif high_threats:
                    sanitized_value = self.sanitize_input(value, field_name)
                    return ValidationResult(
                        field=field_name,
                        valid=True,
                        severity=ValidationSeverity.WARNING,
                        message=f"Potential security risks detected in {field_name}",
                        suggestions=["Review input for safety", "Consider using sanitized version"],
                        sanitized_value=sanitized_value
                    )
        
        # Numeric validations
        if isinstance(value, (int, float)):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")
            
            if min_value is not None and value < min_value:
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be at least {min_value}",
                    suggestions=[f"Increase value to at least {min_value}"]
                )
            
            if max_value is not None and value > max_value:
                return ValidationResult(
                    field=field_name,
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be at most {max_value}",
                    suggestions=[f"Reduce value to at most {max_value}"]
                )
        
        # Allowed values validation
        allowed_values = rules.get("allowed_values")
        if allowed_values and value not in allowed_values:
            return ValidationResult(
                field=field_name,
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be one of: {', '.join(map(str, allowed_values))}",
                suggestions=[f"Use one of: {', '.join(map(str, allowed_values))}"]
            )
        
        # If we reach here, validation passed
        sanitized_value = self.sanitize_input(value, field_name)
        return ValidationResult(
            field=field_name,
            valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} validation passed",
            suggestions=[],
            sanitized_value=sanitized_value if sanitized_value != value else None
        )
    
    def validate_batch(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate multiple fields at once."""
        results = {}
        
        for field_name, value in data.items():
            results[field_name] = self.validate_field(value, field_name)
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results."""
        total = len(results)
        valid_count = sum(1 for r in results.values() if r.valid)
        
        severity_counts = {severity.value: 0 for severity in ValidationSeverity}
        for result in results.values():
            severity_counts[result.severity.value] += 1
        
        critical_fields = [name for name, result in results.items() 
                          if result.severity == ValidationSeverity.CRITICAL]
        error_fields = [name for name, result in results.items() 
                       if result.severity == ValidationSeverity.ERROR]
        
        return {
            "total_fields": total,
            "valid_fields": valid_count,
            "invalid_fields": total - valid_count,
            "severity_counts": severity_counts,
            "critical_fields": critical_fields,
            "error_fields": error_fields,
            "overall_valid": len(critical_fields) == 0 and len(error_fields) == 0
        }

class DataProtectionSystem:
    """Data protection and privacy compliance system."""
    
    def __init__(self):
        self.logger = logging.getLogger("data_protection")
        
        # PII patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information in text."""
        if not isinstance(text, str):
            return {}
        
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize text by replacing PII with placeholders."""
        if not isinstance(text, str):
            return text
        
        anonymized = text
        
        for pii_type, pattern in self.pii_patterns.items():
            placeholder = f"[{pii_type.upper()}_REDACTED]"
            anonymized = re.sub(pattern, placeholder, anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data with salt for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted_data = f"{salt}{data}"
        hashed = hashlib.sha256(salted_data.encode()).hexdigest()
        
        return f"{salt}:{hashed}"
    
    def verify_hashed_data(self, data: str, hashed_data: str) -> bool:
        """Verify data against its hash."""
        try:
            salt, expected_hash = hashed_data.split(":", 1)
            salted_data = f"{salt}{data}"
            actual_hash = hashlib.sha256(salted_data.encode()).hexdigest()
            return actual_hash == expected_hash
        except ValueError:
            return False

async def demonstrate_robust_validation():
    """Demonstrate Generation 2 robust validation system."""
    print("🛡️  Generation 2: Robust Validation & Security System")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize systems
    validator = RobustInputValidator()
    data_protection = DataProtectionSystem()
    
    print("🔍 Testing Input Validation...")
    print("-" * 40)
    
    # Test data with various validation scenarios
    test_data = {
        "api_key": "sk-1234567890abcdef",  # Valid
        "model_name": "gpt-4-turbo",       # Valid
        "provider": "openai",              # Valid
        "temperature": 0.7,                # Valid
        "max_tokens": 2048,                # Valid
        "prompt": "What is the capital of France?",  # Valid
        "benchmark_name": "mmlu",          # Valid
        
        # Invalid cases
        "invalid_provider": "unknown_provider",
        "invalid_temperature": 3.0,
        "invalid_api_key": "short",
        "malicious_prompt": "<script>alert('xss')</script>What is 2+2?",
        "sql_injection": "'; DROP TABLE users; --",
        "command_injection": "ls; rm -rf /",
    }
    
    # Validate all fields
    validation_results = validator.validate_batch(test_data)
    
    # Display results
    for field_name, result in validation_results.items():
        status_icon = "✅" if result.valid else "❌"
        severity_icon = {
            ValidationSeverity.INFO: "ℹ️ ",
            ValidationSeverity.WARNING: "⚠️ ",
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.CRITICAL: "🚨"
        }[result.severity]
        
        print(f"{status_icon} {field_name}: {result.severity.value}")
        print(f"   {severity_icon} {result.message}")
        
        if result.suggestions:
            print(f"   💡 Suggestions: {'; '.join(result.suggestions)}")
        
        if result.sanitized_value is not None:
            print(f"   🧹 Sanitized: {result.sanitized_value}")
        
        print()
    
    # Validation summary
    print("📊 Validation Summary:")
    print("-" * 25)
    summary = validator.get_validation_summary(validation_results)
    print(f"Total Fields: {summary['total_fields']}")
    print(f"Valid Fields: {summary['valid_fields']}")
    print(f"Invalid Fields: {summary['invalid_fields']}")
    print(f"Critical Issues: {len(summary['critical_fields'])}")
    print(f"Errors: {len(summary['error_fields'])}")
    print(f"Overall Valid: {'✅' if summary['overall_valid'] else '❌'}")
    
    if summary['critical_fields']:
        print(f"🚨 Critical Fields: {', '.join(summary['critical_fields'])}")
    
    if summary['error_fields']:
        print(f"❌ Error Fields: {', '.join(summary['error_fields'])}")
    
    # Test data protection
    print("\n🔒 Testing Data Protection...")
    print("-" * 35)
    
    sensitive_text = """
    Contact John Doe at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
    Server IP: 192.168.1.100
    """
    
    # Detect PII
    detected_pii = data_protection.detect_pii(sensitive_text)
    print("🔍 PII Detection Results:")
    for pii_type, matches in detected_pii.items():
        print(f"  {pii_type}: {matches}")
    
    # Anonymize text
    anonymized = data_protection.anonymize_text(sensitive_text)
    print(f"\n🔒 Anonymized Text:")
    print(f"  {anonymized.strip()}")
    
    # Test data hashing
    print(f"\n🔐 Secure Data Hashing:")
    sensitive_data = "user_password_123"
    hashed = data_protection.hash_sensitive_data(sensitive_data)
    print(f"  Original: {sensitive_data}")
    print(f"  Hashed: {hashed[:50]}...")
    
    # Verify hash
    is_valid = data_protection.verify_hashed_data(sensitive_data, hashed)
    is_invalid = data_protection.verify_hashed_data("wrong_password", hashed)
    print(f"  Verification (correct): {'✅' if is_valid else '❌'}")
    print(f"  Verification (wrong): {'✅' if not is_invalid else '❌'}")
    
    print("\n✅ Generation 2 robust validation system demonstration complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(demonstrate_robust_validation())
    sys.exit(0 if success else 1)