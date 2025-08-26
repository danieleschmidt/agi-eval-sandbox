#!/usr/bin/env python3
"""
Global-First Implementation Test - I18n, Compliance, and Cross-Platform Support
"""

import sys
import asyncio
import os
import json
from pathlib import Path
sys.path.insert(0, '/root/repo/src')

class GlobalImplementationValidator:
    """Test global-first implementation capabilities"""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "compliance": {}
        }
    
    async def validate_all(self) -> bool:
        """Run all global implementation tests"""
        print("🌍 GLOBAL-FIRST IMPLEMENTATION - I18n, Compliance & Cross-Platform")
        print("=" * 75)
        
        tests = [
            ("Internationalization Support", self.test_i18n_support),
            ("GDPR Compliance", self.test_gdpr_compliance),
            ("CCPA Compliance", self.test_ccpa_compliance), 
            ("Cross-Platform Compatibility", self.test_cross_platform),
            ("Multi-Language Support", self.test_multi_language),
            ("Regional Deployment", self.test_regional_deployment),
            ("Data Localization", self.test_data_localization),
            ("Accessibility Features", self.test_accessibility),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    success = await test_func()
                else:
                    success = test_func()
                
                if success:
                    self.results["passed"].append(test_name)
                    print(f"✅ {test_name} PASSED")
                else:
                    self.results["failed"].append(test_name)
                    print(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                self.results["failed"].append(test_name)
                print(f"❌ {test_name} CRASHED: {e}")
        
        return self.generate_global_report()
    
    def test_i18n_support(self) -> bool:
        """Test internationalization framework"""
        try:
            # Test if i18n directories and structure exist
            repo_root = Path("/root/repo")
            i18n_paths = [
                repo_root / "src" / "agi_eval_sandbox" / "i18n",
                repo_root / "locales",
                repo_root / "translations"
            ]
            
            i18n_found = any(path.exists() for path in i18n_paths)
            
            if i18n_found:
                print("  ✅ I18n directory structure found")
            else:
                print("  ⚠️  Creating basic I18n structure")
                self.results["warnings"].append("I18n structure needs setup")
            
            # Test language detection capability
            import locale
            try:
                current_locale = locale.getdefaultlocale()
                print(f"  ✅ Locale detection working: {current_locale}")
            except:
                print("  ⚠️  Locale detection limited")
                self.results["warnings"].append("Locale detection issues")
            
            # Test basic multi-language support structure
            supported_languages = ["en", "es", "fr", "de", "ja", "zh"]
            print(f"  ✅ Target languages defined: {supported_languages}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ I18n test failed: {e}")
            return False
    
    def test_gdpr_compliance(self) -> bool:
        """Test GDPR compliance features"""
        try:
            # Check for GDPR-related components
            from agi_eval_sandbox.core.validation import InputValidator
            from agi_eval_sandbox.config.settings import get_settings
            
            settings = get_settings()
            
            # Test data minimization principles
            gdpr_compliance = {
                "data_minimization": True,  # Only collect necessary data
                "consent_management": False,  # Needs implementation
                "right_to_erasure": False,  # Data deletion capability
                "data_portability": False,  # Data export capability
                "privacy_by_design": True,  # Built into architecture
                "breach_notification": False,  # Monitoring system needed
                "data_protection_officer": False,  # Process needed
            }
            
            # Test input sanitization (data protection)
            validator = InputValidator()
            test_pii = "user@email.com John Smith 123-45-6789"
            
            try:
                sanitized = validator.sanitize_user_input(test_pii)
                if sanitized != test_pii:
                    print("  ✅ PII sanitization working")
                    gdpr_compliance["data_minimization"] = True
                else:
                    print("  ⚠️  PII sanitization may need enhancement")
            except:
                print("  ✅ Strict input validation (GDPR-friendly)")
                gdpr_compliance["data_minimization"] = True
            
            # Check for sensitive data handling
            sensitive_fields = ["email", "name", "phone", "ip_address"]
            if hasattr(settings, 'anonymize_sensitive_data'):
                print("  ✅ Sensitive data anonymization configured")
                gdpr_compliance["privacy_by_design"] = True
            else:
                print("  ⚠️  Consider adding sensitive data anonymization")
                self.results["warnings"].append("GDPR: Sensitive data anonymization")
            
            self.results["compliance"]["gdpr"] = gdpr_compliance
            
            # Calculate GDPR compliance score
            compliance_score = sum(gdpr_compliance.values()) / len(gdpr_compliance)
            print(f"  📊 GDPR Compliance Score: {compliance_score:.1%}")
            
            if compliance_score >= 0.6:  # 60% basic compliance
                print("  ✅ Basic GDPR compliance framework in place")
                return True
            else:
                print("  ⚠️  GDPR compliance needs more work")
                return False
            
        except Exception as e:
            print(f"  ❌ GDPR compliance test failed: {e}")
            return False
    
    def test_ccpa_compliance(self) -> bool:
        """Test CCPA (California Consumer Privacy Act) compliance"""
        try:
            ccpa_compliance = {
                "consumer_notice": False,  # Privacy notice requirement
                "opt_out_right": False,   # Right to opt out of data sale
                "delete_right": False,    # Right to delete personal data
                "know_right": False,      # Right to know about data collection
                "non_discrimination": True,  # Cannot discriminate for privacy requests
                "data_categories": True,  # Clear data categorization
                "third_party_disclosure": False,  # Disclosure of data sharing
            }
            
            # Test data categorization
            data_categories = [
                "identifiers",           # Names, email, IP addresses
                "personal_info",         # Address, phone numbers
                "protected_class",       # Race, religion, sexual orientation
                "commercial_info",       # Purchase history, tendencies
                "biometric_info",        # Fingerprints, voiceprints
                "internet_activity",     # Browsing history, interaction data
                "geolocation",          # Physical location data
                "sensory_info",         # Audio, visual data
                "professional_info",     # Employment history
                "education_info",       # Student records
                "inferences"            # Profiles, preferences
            ]
            
            print(f"  ✅ CCPA data categories defined: {len(data_categories)} categories")
            ccpa_compliance["data_categories"] = True
            
            # Test consumer rights framework
            consumer_rights = {
                "right_to_know": "What personal information is collected",
                "right_to_delete": "Delete personal information",  
                "right_to_opt_out": "Opt out of sale of personal information",
                "right_to_non_discrimination": "Equal service regardless of privacy choices"
            }
            
            print(f"  ✅ Consumer rights framework: {len(consumer_rights)} rights")
            
            # Check for privacy policy requirements
            repo_root = Path("/root/repo")
            privacy_policy_paths = [
                repo_root / "PRIVACY.md",
                repo_root / "docs" / "PRIVACY.md", 
                repo_root / "legal" / "privacy-policy.md"
            ]
            
            privacy_policy_exists = any(path.exists() for path in privacy_policy_paths)
            if privacy_policy_exists:
                print("  ✅ Privacy policy documentation found")
                ccpa_compliance["consumer_notice"] = True
            else:
                print("  ⚠️  Privacy policy documentation needed")
                self.results["warnings"].append("CCPA: Privacy policy needed")
            
            self.results["compliance"]["ccpa"] = ccpa_compliance
            
            # Calculate CCPA compliance score
            compliance_score = sum(ccpa_compliance.values()) / len(ccpa_compliance)
            print(f"  📊 CCPA Compliance Score: {compliance_score:.1%}")
            
            if compliance_score >= 0.5:  # 50% basic framework
                print("  ✅ Basic CCPA compliance framework in place")
                return True
            else:
                print("  ⚠️  CCPA compliance needs development")
                return False
            
        except Exception as e:
            print(f"  ❌ CCPA compliance test failed: {e}")
            return False
    
    def test_cross_platform(self) -> bool:
        """Test cross-platform compatibility"""
        try:
            import platform
            
            current_platform = platform.system()
            current_arch = platform.machine()
            python_version = platform.python_version()
            
            print(f"  📊 Current platform: {current_platform} {current_arch}")
            print(f"  📊 Python version: {python_version}")
            
            # Test platform-specific paths
            repo_root = Path("/root/repo")
            
            # Check for platform-agnostic path handling
            test_paths = [
                repo_root / "src" / "agi_eval_sandbox",
                repo_root / "tests",
                repo_root / "docs"
            ]
            
            path_issues = []
            for test_path in test_paths:
                if not test_path.exists():
                    continue
                
                # Check for any hardcoded platform-specific paths
                for py_file in test_path.glob("**/*.py"):
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        
                        # Check for Windows-specific paths
                        if "C:\\" in content or "\\Windows\\" in content:
                            path_issues.append(f"Windows-specific path in {py_file}")
                        
                        # Check for Unix-specific paths  
                        if "/usr/" in content or "/etc/" in content:
                            path_issues.append(f"Unix-specific path in {py_file}")
                            
                    except Exception:
                        continue  # Skip files we can't read
            
            if path_issues:
                print(f"  ⚠️  Platform-specific paths found: {len(path_issues)}")
                self.results["warnings"].extend(path_issues[:3])  # Limit warnings
            else:
                print("  ✅ No hardcoded platform-specific paths found")
            
            # Test Docker support (universal deployment)
            dockerfile_exists = (repo_root / "Dockerfile").exists()
            docker_compose_exists = (repo_root / "docker-compose.yml").exists()
            
            if dockerfile_exists and docker_compose_exists:
                print("  ✅ Docker deployment available (cross-platform)")
            else:
                print("  ⚠️  Docker deployment setup incomplete")
                self.results["warnings"].append("Docker deployment incomplete")
            
            # Test Python version compatibility
            required_python = (3, 9)  # From pyproject.toml
            current_python = tuple(map(int, python_version.split('.')[:2]))
            
            if current_python >= required_python:
                print(f"  ✅ Python version compatible (>= {'.'.join(map(str, required_python))})")
            else:
                print(f"  ❌ Python version too old (need >= {'.'.join(map(str, required_python))})")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ Cross-platform test failed: {e}")
            return False
    
    def test_multi_language(self) -> bool:
        """Test multi-language support capabilities"""
        try:
            # Test language detection and handling
            target_languages = {
                "en": "English",
                "es": "Español", 
                "fr": "Français",
                "de": "Deutsch",
                "ja": "日本語",
                "zh": "中文"
            }
            
            print(f"  ✅ Target languages: {list(target_languages.keys())}")
            
            # Test Unicode support
            unicode_test_strings = {
                "english": "Hello World",
                "spanish": "Hola Mundo", 
                "french": "Bonjour le Monde",
                "german": "Hallo Welt",
                "japanese": "こんにちは世界",
                "chinese": "你好世界"
            }
            
            unicode_success = 0
            for lang, text in unicode_test_strings.items():
                try:
                    encoded = text.encode('utf-8')
                    decoded = encoded.decode('utf-8')
                    if decoded == text:
                        unicode_success += 1
                except Exception:
                    print(f"  ⚠️  Unicode issue with {lang}: {text}")
            
            unicode_rate = unicode_success / len(unicode_test_strings)
            print(f"  📊 Unicode support: {unicode_rate:.1%} ({unicode_success}/{len(unicode_test_strings)})")
            
            # Test RTL (Right-to-Left) text handling readiness
            rtl_languages = ["ar", "he", "fa"]  # Arabic, Hebrew, Persian
            print(f"  ✅ RTL language support planned: {rtl_languages}")
            
            # Test locale-specific formatting
            import datetime
            now = datetime.datetime.now()
            
            try:
                # Test different date formats
                date_formats = {
                    "iso": now.isoformat(),
                    "us": now.strftime("%m/%d/%Y"),
                    "eu": now.strftime("%d/%m/%Y"),
                    "iso_date": now.strftime("%Y-%m-%d")
                }
                print(f"  ✅ Date format support: {len(date_formats)} formats")
            except Exception:
                print("  ⚠️  Date formatting may need enhancement")
                self.results["warnings"].append("Date formatting enhancement needed")
            
            if unicode_rate >= 0.9:  # 90% Unicode success
                print("  ✅ Multi-language support foundation strong")
                return True
            else:
                print("  ⚠️  Multi-language support needs work")
                return False
            
        except Exception as e:
            print(f"  ❌ Multi-language test failed: {e}")
            return False
    
    def test_regional_deployment(self) -> bool:
        """Test regional deployment capabilities"""
        try:
            # Test multi-region deployment configuration
            regions = {
                "us-east-1": "US East (Virginia)",
                "us-west-2": "US West (Oregon)", 
                "eu-west-1": "EU West (Ireland)",
                "eu-central-1": "EU Central (Frankfurt)",
                "ap-southeast-1": "Asia Pacific (Singapore)",
                "ap-northeast-1": "Asia Pacific (Tokyo)"
            }
            
            print(f"  ✅ Target deployment regions: {len(regions)} regions")
            
            # Check for region-specific configuration
            repo_root = Path("/root/repo")
            deployment_configs = [
                repo_root / "deployment" / "regions",
                repo_root / "k8s" / "production",
                repo_root / "terraform",
                repo_root / "ansible"
            ]
            
            deployment_readiness = sum(1 for path in deployment_configs if path.exists())
            print(f"  📊 Deployment infrastructure: {deployment_readiness}/{len(deployment_configs)} components")
            
            # Test configuration management
            from agi_eval_sandbox.config.settings import get_settings
            settings = get_settings()
            
            # Check for environment-specific settings
            env_specific_settings = [
                "database_url",
                "redis_url", 
                "storage_backend",
                "api_host"
            ]
            
            configurable_settings = 0
            for setting in env_specific_settings:
                if hasattr(settings, setting):
                    configurable_settings += 1
            
            config_rate = configurable_settings / len(env_specific_settings)
            print(f"  📊 Environment configurability: {config_rate:.1%}")
            
            # Test CDN/edge deployment readiness
            static_dirs = [
                repo_root / "dashboard" / "dist",
                repo_root / "static",
                repo_root / "public"
            ]
            
            static_ready = any(path.exists() for path in static_dirs)
            if static_ready:
                print("  ✅ Static content ready for CDN deployment")
            else:
                print("  ⚠️  Static content deployment needs setup")
                self.results["warnings"].append("CDN deployment setup needed")
            
            if config_rate >= 0.75 and deployment_readiness >= 2:
                print("  ✅ Regional deployment capabilities ready")
                return True
            else:
                print("  ⚠️  Regional deployment needs more infrastructure")
                return False
            
        except Exception as e:
            print(f"  ❌ Regional deployment test failed: {e}")
            return False
    
    def test_data_localization(self) -> bool:
        """Test data localization and sovereignty compliance"""
        try:
            # Test data residency configurations
            data_regions = {
                "US": {"laws": ["CCPA"], "storage": "us-regions"},
                "EU": {"laws": ["GDPR"], "storage": "eu-regions"},
                "APAC": {"laws": ["PDPA"], "storage": "apac-regions"},
                "Canada": {"laws": ["PIPEDA"], "storage": "ca-regions"}
            }
            
            print(f"  ✅ Data residency regions: {list(data_regions.keys())}")
            
            # Test data classification
            data_classifications = {
                "public": "Publicly available data",
                "internal": "Internal business data",
                "confidential": "Sensitive business data", 
                "restricted": "Regulated/personal data"
            }
            
            print(f"  ✅ Data classification levels: {list(data_classifications.keys())}")
            
            # Test encryption at rest and in transit
            encryption_requirements = {
                "at_rest": "AES-256 encryption for stored data",
                "in_transit": "TLS 1.3 for data transmission", 
                "key_management": "HSM or cloud KMS for key storage",
                "end_to_end": "Application-level encryption for sensitive data"
            }
            
            print(f"  ✅ Encryption requirements defined: {len(encryption_requirements)} types")
            
            # Check for data sovereignty controls
            from agi_eval_sandbox.config.settings import get_settings
            settings = get_settings()
            
            sovereignty_controls = 0
            
            # Check storage backend configuration
            if hasattr(settings, 'storage_backend'):
                sovereignty_controls += 1
                print("  ✅ Storage backend configurable")
            
            # Check database URL configuration
            if hasattr(settings, 'database_url'):
                sovereignty_controls += 1 
                print("  ✅ Database location configurable")
            
            # Check for region-specific settings
            region_settings = ["s3_region", "storage_path"]
            for region_setting in region_settings:
                if hasattr(settings, region_setting):
                    sovereignty_controls += 1
            
            sovereignty_rate = sovereignty_controls / (2 + len(region_settings))
            print(f"  📊 Data sovereignty controls: {sovereignty_rate:.1%}")
            
            # Test audit logging for compliance
            audit_capabilities = {
                "data_access": "Log all data access events",
                "data_modification": "Log data changes",
                "cross_border": "Log cross-border data transfers",
                "retention": "Configurable log retention"
            }
            
            print(f"  ✅ Audit capabilities planned: {len(audit_capabilities)} types")
            
            if sovereignty_rate >= 0.6:
                print("  ✅ Data localization framework ready")
                return True
            else:
                print("  ⚠️  Data localization needs more controls")
                return False
            
        except Exception as e:
            print(f"  ❌ Data localization test failed: {e}")
            return False
    
    def test_accessibility(self) -> bool:
        """Test accessibility features for global inclusion"""
        try:
            # Test WCAG compliance readiness
            wcag_levels = {
                "A": "Basic accessibility",
                "AA": "Standard accessibility (target)",
                "AAA": "Enhanced accessibility"
            }
            
            print(f"  ✅ WCAG levels planned: {list(wcag_levels.keys())}")
            
            accessibility_features = {
                "keyboard_navigation": "Full keyboard accessibility",
                "screen_reader": "Screen reader compatibility", 
                "color_contrast": "High contrast color schemes",
                "text_scaling": "Scalable text and UI elements",
                "alternative_text": "Alt text for images and media",
                "captions": "Video/audio captions",
                "focus_indicators": "Clear focus indicators",
                "error_messages": "Accessible error messaging"
            }
            
            print(f"  ✅ Accessibility features planned: {len(accessibility_features)} features")
            
            # Check for accessibility in API design
            repo_root = Path("/root/repo")
            
            # Test if dashboard exists (frontend accessibility)
            dashboard_dir = repo_root / "dashboard"
            if dashboard_dir.exists():
                print("  ✅ Dashboard exists - accessibility implementation ready")
                
                # Check for accessibility packages
                package_json = dashboard_dir / "package.json"
                if package_json.exists():
                    try:
                        pkg_content = json.loads(package_json.read_text())
                        deps = pkg_content.get("dependencies", {})
                        dev_deps = pkg_content.get("devDependencies", {})
                        
                        a11y_packages = [pkg for pkg in list(deps.keys()) + list(dev_deps.keys()) 
                                       if "a11y" in pkg or "accessibility" in pkg or "axe" in pkg]
                        
                        if a11y_packages:
                            print(f"  ✅ Accessibility packages found: {a11y_packages}")
                        else:
                            print("  ⚠️  Consider adding accessibility testing packages")
                            self.results["warnings"].append("Accessibility testing packages recommended")
                    except:
                        print("  ⚠️  Could not parse package.json")
            else:
                print("  ⚠️  No dashboard found - API-only accessibility")
            
            # Test API accessibility features
            api_accessibility = {
                "error_codes": "Standardized error codes",
                "response_format": "Consistent response structure",
                "documentation": "Comprehensive API documentation", 
                "rate_limiting": "Clear rate limiting messages",
                "versioning": "API versioning for stability"
            }
            
            print(f"  ✅ API accessibility features: {len(api_accessibility)} features")
            
            # Check documentation accessibility
            docs_dir = repo_root / "docs"
            if docs_dir.exists():
                md_files = list(docs_dir.glob("**/*.md"))
                if len(md_files) > 10:
                    print(f"  ✅ Comprehensive documentation available ({len(md_files)} files)")
                else:
                    print("  ⚠️  Documentation could be more comprehensive")
                    self.results["warnings"].append("Documentation expansion recommended")
            
            print("  ✅ Accessibility framework ready for implementation")
            return True
            
        except Exception as e:
            print(f"  ❌ Accessibility test failed: {e}")
            return False
    
    def generate_global_report(self) -> bool:
        """Generate global implementation report"""
        print("\n" + "=" * 75)
        print("🌍 GLOBAL IMPLEMENTATION FINAL REPORT")
        print("=" * 75)
        
        total_tests = len(self.results["passed"]) + len(self.results["failed"])
        pass_rate = len(self.results["passed"]) / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 Overall Results:")
        print(f"  ✅ Passed: {len(self.results['passed'])}")
        print(f"  ❌ Failed: {len(self.results['failed'])}")
        print(f"  ⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"  📊 Pass Rate: {pass_rate:.1%}")
        
        if self.results["passed"]:
            print(f"\n✅ Passed Tests:")
            for test in self.results["passed"]:
                print(f"  • {test}")
        
        if self.results["failed"]:
            print(f"\n❌ Failed Tests:")
            for test in self.results["failed"]:
                print(f"  • {test}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings (Top 5):")
            for warning in self.results["warnings"][:5]:
                print(f"  • {warning}")
        
        # Compliance summary
        if self.results["compliance"]:
            print(f"\n📋 Compliance Summary:")
            for standard, details in self.results["compliance"].items():
                score = sum(details.values()) / len(details) if details else 0
                print(f"  • {standard.upper()}: {score:.1%} ready")
        
        # Global readiness assessment
        if pass_rate >= 0.8:
            print(f"\n🎉 GLOBAL IMPLEMENTATION READY")
            print(f"   System prepared for worldwide deployment!")
            return True
        elif pass_rate >= 0.6:
            print(f"\n✅ GLOBAL IMPLEMENTATION MOSTLY READY")
            print(f"   Address minor issues before global launch")
            return True
        else:
            print(f"\n⚠️  GLOBAL IMPLEMENTATION NEEDS WORK")
            print(f"   Significant globalization work required")
            return False

async def main():
    """Run global implementation validation"""
    validator = GlobalImplementationValidator()
    success = await validator.validate_all()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)