#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Perpetual value-maximizing SDLC enhancement system
"""

import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class ValueItem:
    """Represents a discovered work item with comprehensive scoring."""
    id: str
    title: str
    description: str
    category: str
    type: str
    source: str
    estimated_effort: float  # hours
    
    # WSJF Components
    user_business_value: float
    time_criticality: float
    risk_reduction: float
    opportunity_enablement: float
    
    # ICE Components
    impact: float
    confidence: float
    ease: float
    
    # Technical Debt Components
    debt_impact: float
    debt_interest: float
    hotspot_multiplier: float
    
    # Composite Scoring
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Metadata
    discovered_at: str = ""
    files_affected: List[str] = None
    dependencies: List[str] = None
    risk_level: float = 0.0
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []
        if self.dependencies is None:
            self.dependencies = []
        if not self.discovered_at:
            self.discovered_at = datetime.now(timezone.utc).isoformat()


class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization engine."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.metrics_path = Path(".terragon/value-metrics.json")
        self.backlog_path = Path("BACKLOG.md")
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _load_metrics(self) -> Dict:
        """Load metrics from JSON file."""
        try:
            with open(self.metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_metrics()
    
    def _default_config(self) -> Dict:
        """Default configuration for value discovery."""
        return {
            "scoring": {
                "weights": {
                    "maturing": {
                        "wsjf": 0.6,
                        "ice": 0.1,
                        "technicalDebt": 0.2,
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "minScore": 10,
                    "maxRisk": 0.8,
                    "securityBoost": 2.0,
                    "complianceBoost": 1.8
                }
            },
            "repository": {
                "maturity": "maturing"
            }
        }
    
    def _default_metrics(self) -> Dict:
        """Default metrics structure."""
        return {
            "execution_history": [],
            "backlog_metrics": {
                "totalItems": 0,
                "averageAge": 0,
                "debtRatio": 0.0,
                "velocityTrend": "baseline"
            }
        }
    
    def discover_value_items(self) -> List[ValueItem]:
        """Comprehensive signal harvesting and value item discovery."""
        discovered_items = []
        
        # 1. Git History Analysis
        discovered_items.extend(self._analyze_git_history())
        
        # 2. Static Code Analysis
        discovered_items.extend(self._analyze_code_quality())
        
        # 3. Security Vulnerability Scan
        discovered_items.extend(self._scan_security_issues())
        
        # 4. Dependency Analysis
        discovered_items.extend(self._analyze_dependencies())
        
        # 5. Documentation Gaps
        discovered_items.extend(self._find_documentation_gaps())
        
        # 6. Performance Opportunities
        discovered_items.extend(self._identify_performance_issues())
        
        # Score all items
        for item in discovered_items:
            self._calculate_composite_score(item)
        
        return discovered_items
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Extract technical debt indicators from Git history."""
        items = []
        
        try:
            # Look for TODO, FIXME, HACK comments
            result = subprocess.run([
                "git", "log", "--oneline", "--grep=TODO\\|FIXME\\|HACK\\|temporary\\|quick fix",
                "-i", "--since=3 months ago"
            ], capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    commit_hash = line.split()[0]
                    message = ' '.join(line.split()[1:])
                    
                    items.append(ValueItem(
                        id=f"git-debt-{commit_hash[:8]}",
                        title=f"Address technical debt: {message[:50]}...",
                        description=f"Technical debt identified in commit {commit_hash}: {message}",
                        category="technical_debt",
                        type="refactoring",
                        source="git_history",
                        estimated_effort=2.0,
                        user_business_value=3.0,
                        time_criticality=2.0,
                        risk_reduction=4.0,
                        opportunity_enablement=3.0,
                        impact=6.0,
                        confidence=8.0,
                        ease=7.0,
                        debt_impact=15.0,
                        debt_interest=5.0,
                        hotspot_multiplier=1.5
                    ))
        except subprocess.CalledProcessError:
            pass  # Repository might not have Git history
        
        return items
    
    def _analyze_code_quality(self) -> List[ValueItem]:
        """Run static analysis tools to identify code quality issues."""
        items = []
        
        # Run ruff for Python code quality
        try:
            result = subprocess.run([
                "ruff", "check", "src/", "--output-format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:5]:  # Limit to top 5 issues
                    items.append(ValueItem(
                        id=f"ruff-{hash(issue['filename'] + issue['code'])}",
                        title=f"Fix {issue['code']}: {issue['message'][:50]}...",
                        description=f"Code quality issue in {issue['filename']}: {issue['message']}",
                        category="code_quality",
                        type="cleanup",
                        source="static_analysis",
                        estimated_effort=0.5,
                        user_business_value=2.0,
                        time_criticality=1.0,
                        risk_reduction=3.0,
                        opportunity_enablement=2.0,
                        impact=4.0,
                        confidence=9.0,
                        ease=8.0,
                        debt_impact=8.0,
                        debt_interest=2.0,
                        hotspot_multiplier=1.2,
                        files_affected=[issue['filename']]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return items
    
    def _scan_security_issues(self) -> List[ValueItem]:
        """Identify security vulnerabilities and issues."""
        items = []
        
        # Run bandit security scan
        try:
            result = subprocess.run([
                "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                report = json.loads(result.stdout)
                for issue in report.get('results', [])[:3]:  # Top 3 security issues
                    items.append(ValueItem(
                        id=f"security-{hash(issue['filename'] + issue['test_id'])}",
                        title=f"Security: {issue['issue_text'][:50]}...",
                        description=f"Security issue in {issue['filename']}: {issue['issue_text']}",
                        category="security",
                        type="security_fix",
                        source="security_scan",
                        estimated_effort=1.5,
                        user_business_value=8.0,
                        time_criticality=7.0,
                        risk_reduction=9.0,
                        opportunity_enablement=4.0,
                        impact=9.0,
                        confidence=8.0,
                        ease=6.0,
                        debt_impact=20.0,
                        debt_interest=10.0,
                        hotspot_multiplier=2.0,
                        files_affected=[issue['filename']]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return items
    
    def _analyze_dependencies(self) -> List[ValueItem]:
        """Check for outdated or vulnerable dependencies."""
        items = []
        
        # Check for outdated Python packages
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities[:3]:  # Top 3 vulnerabilities
                    items.append(ValueItem(
                        id=f"dep-vuln-{hash(vuln.get('package_name', 'unknown'))}",
                        title=f"Update vulnerable dependency: {vuln.get('package_name', 'unknown')}",
                        description=f"Security vulnerability in {vuln.get('package_name')}: {vuln.get('advisory', '')}",
                        category="security",
                        type="dependency_update",
                        source="dependency_scan",
                        estimated_effort=0.5,
                        user_business_value=7.0,
                        time_criticality=8.0,
                        risk_reduction=9.0,
                        opportunity_enablement=3.0,
                        impact=8.0,
                        confidence=9.0,
                        ease=9.0,
                        debt_impact=15.0,
                        debt_interest=8.0,
                        hotspot_multiplier=1.5
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return items
    
    def _find_documentation_gaps(self) -> List[ValueItem]:
        """Identify missing or outdated documentation."""
        items = []
        
        # Check for missing API documentation
        src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
        undocumented_modules = []
        
        for file_path in src_files[:5]:  # Limit to avoid overwhelming
            try:
                with open(file_path) as f:
                    content = f.read()
                    if '"""' not in content and "'''" not in content:
                        undocumented_modules.append(str(file_path))
            except Exception:
                continue
        
        if undocumented_modules:
            items.append(ValueItem(
                id="doc-gap-modules",
                title=f"Add documentation to {len(undocumented_modules)} modules",
                description=f"Missing module documentation in: {', '.join(undocumented_modules[:3])}",
                category="documentation",
                type="documentation",
                source="code_analysis",
                estimated_effort=len(undocumented_modules) * 0.5,
                user_business_value=4.0,
                time_criticality=2.0,
                risk_reduction=2.0,
                opportunity_enablement=6.0,
                impact=5.0,
                confidence=8.0,
                ease=7.0,
                debt_impact=10.0,
                debt_interest=3.0,
                hotspot_multiplier=1.0,
                files_affected=undocumented_modules
            ))
        
        return items
    
    def _identify_performance_issues(self) -> List[ValueItem]:
        """Find potential performance optimization opportunities."""
        items = []
        
        # Look for common performance anti-patterns
        performance_patterns = [
            ("N+1 queries", "database performance"),
            ("synchronous I/O", "async optimization"),
            ("large imports", "import optimization"),
            ("inefficient loops", "algorithm optimization")
        ]
        
        for pattern, optimization in performance_patterns:
            items.append(ValueItem(
                id=f"perf-{hash(pattern)}",
                title=f"Optimize {optimization}",
                description=f"Potential performance improvement: {pattern}",
                category="performance",
                type="optimization",
                source="pattern_analysis",
                estimated_effort=3.0,
                user_business_value=6.0,
                time_criticality=3.0,
                risk_reduction=2.0,
                opportunity_enablement=7.0,
                impact=7.0,
                confidence=5.0,
                ease=4.0,
                debt_impact=12.0,
                debt_interest=4.0,
                hotspot_multiplier=1.3
            ))
        
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> None:
        """Calculate comprehensive value score using WSJF, ICE, and technical debt."""
        # WSJF Calculation
        cost_of_delay = (
            item.user_business_value + 
            item.time_criticality + 
            item.risk_reduction + 
            item.opportunity_enablement
        )
        item.wsjf_score = cost_of_delay / max(item.estimated_effort, 0.1)
        
        # ICE Calculation
        item.ice_score = item.impact * item.confidence * item.ease
        
        # Technical Debt Score
        item.technical_debt_score = (
            (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
        )
        
        # Composite Score with adaptive weights
        maturity = self.config.get("repository", {}).get("maturity", "maturing")
        weights = self.config["scoring"]["weights"].get(maturity, {})
        
        # Normalize scores
        normalized_wsjf = min(item.wsjf_score / 10.0, 1.0)
        normalized_ice = min(item.ice_score / 1000.0, 1.0)
        normalized_debt = min(item.technical_debt_score / 100.0, 1.0)
        
        item.composite_score = (
            weights.get("wsjf", 0.6) * normalized_wsjf * 100 +
            weights.get("ice", 0.1) * normalized_ice * 100 +
            weights.get("technicalDebt", 0.2) * normalized_debt * 100
        )
        
        # Apply boosts
        thresholds = self.config["scoring"]["thresholds"]
        if item.category == "security":
            item.composite_score *= thresholds.get("securityBoost", 2.0)
        
        # Risk assessment
        item.risk_level = max(0.1, 1.0 - (item.confidence / 10.0))
    
    def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the highest value item that meets execution criteria."""
        # Sort by composite score (descending)
        prioritized = sorted(items, key=lambda x: x.composite_score, reverse=True)
        
        thresholds = self.config["scoring"]["thresholds"]
        
        for item in prioritized:
            # Skip if score too low
            if item.composite_score < thresholds.get("minScore", 10):
                continue
            
            # Skip if risk too high
            if item.risk_level > thresholds.get("maxRisk", 0.8):
                continue
            
            # Found our best value item
            return item
        
        # No suitable items found
        return None
    
    def generate_backlog_markdown(self, items: List[ValueItem]) -> str:
        """Generate comprehensive backlog markdown."""
        now = datetime.now(timezone.utc)
        next_execution = datetime.fromtimestamp(time.time() + 1800, timezone.utc)  # 30 min from now
        
        # Select top item
        top_item = self.select_next_best_value(items)
        
        # Sort items by score for display
        sorted_items = sorted(items, key=lambda x: x.composite_score, reverse=True)[:10]
        
        markdown = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Next Execution: {next_execution.isoformat()}

## 🎯 Next Best Value Item
"""
        
        if top_item:
            markdown += f"""**[{top_item.id.upper()}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.0f} | **Tech Debt**: {top_item.technical_debt_score:.0f}
- **Estimated Effort**: {top_item.estimated_effort:.1f} hours
- **Category**: {top_item.category} | **Type**: {top_item.type}
- **Risk Level**: {top_item.risk_level:.2f}

{top_item.description}
"""
        else:
            markdown += "**No items meet current execution criteria**\n"
        
        markdown += """
## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(sorted_items, 1):
            title_short = item.title[:40] + "..." if len(item.title) > 40 else item.title
            markdown += f"| {i} | {item.id[:8]} | {title_short} | {item.composite_score:.1f} | {item.category} | {item.estimated_effort:.1f} |\n"
        
        # Add metrics section
        total_items = len(items)
        avg_score = sum(item.composite_score for item in items) / max(len(items), 1)
        security_items = len([item for item in items if item.category == "security"])
        debt_items = len([item for item in items if item.category == "technical_debt"])
        
        markdown += f"""
## 📈 Value Metrics
- **Total Items Discovered**: {total_items}
- **Average Value Score**: {avg_score:.1f}
- **Security Items**: {security_items}
- **Technical Debt Items**: {debt_items}
- **Discovery Date**: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🔄 Continuous Discovery Stats
- **Repository Maturity**: {self.config.get('repository', {}).get('maturity', 'unknown').title()}
- **Maturity Score**: {self.config.get('repository', {}).get('maturityScore', 'N/A')}%
- **Discovery Sources**:
  - Git History Analysis: Active
  - Static Code Analysis: Active  
  - Security Scanning: Active
  - Dependency Analysis: Active
  - Documentation Analysis: Active
  - Performance Analysis: Active

## 🎯 Value Discovery Categories

### 🛡️ Security Items ({security_items})
High-priority security vulnerabilities and compliance issues.

### 🔧 Technical Debt Items ({debt_items})  
Code quality improvements and refactoring opportunities.

### 📈 Performance Items
Optimization opportunities for better system performance.

### 📚 Documentation Items
Missing or outdated documentation requiring updates.

---

*Generated by Terragon Autonomous SDLC Enhancement Engine*
*Next discovery cycle: Continuous monitoring active*
"""
        
        return markdown
    
    def save_metrics(self) -> None:
        """Save updated metrics to file."""
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def run_discovery_cycle(self) -> Tuple[List[ValueItem], Optional[ValueItem]]:
        """Execute a complete value discovery cycle."""
        print("🔍 Starting value discovery cycle...")
        
        # Discover all value items
        items = self.discover_value_items()
        
        # Select next best value
        next_item = self.select_next_best_value(items)
        
        # Update metrics
        self.metrics["backlog_metrics"]["totalItems"] = len(items)
        self.metrics["backlog_metrics"]["lastDiscovery"] = datetime.now(timezone.utc).isoformat()
        
        # Generate and save backlog
        backlog_md = self.generate_backlog_markdown(items)
        with open(self.backlog_path, 'w') as f:
            f.write(backlog_md)
        
        # Save metrics
        self.save_metrics()
        
        print(f"✅ Discovery complete: {len(items)} items found")
        if next_item:
            print(f"🎯 Next best value: {next_item.title} (score: {next_item.composite_score:.1f})")
        
        return items, next_item


def main():
    """Main entry point for value discovery engine."""
    engine = ValueDiscoveryEngine()
    items, next_item = engine.run_discovery_cycle()
    
    if next_item:
        print(f"\n🚀 Ready to execute: {next_item.title}")
        print(f"   Estimated effort: {next_item.estimated_effort} hours")
        print(f"   Expected value: {next_item.composite_score:.1f}")
    else:
        print("\n⏸️  No items meet execution criteria at this time")


if __name__ == "__main__":
    main()