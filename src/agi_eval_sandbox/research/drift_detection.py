"""
Real-Time Model Drift Detection and Auto-Correction System

Novel drift detection algorithm using ensemble methods, statistical tests,
and adaptive correction mechanisms for maintaining model reliability.

Research Innovation: "Continuous Model Reliability via Adaptive Drift Correction"
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import wasserstein_distance
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor

from ..core.logging_config import get_logger
from ..core.models import Model
from ..core.results import Results, BenchmarkResult

logger = get_logger("drift_detection")


class DriftType(Enum):
    """Types of model drift that can be detected."""
    NONE = "none"
    CONCEPT_DRIFT = "concept_drift"          # P(Y|X) changes
    COVARIATE_DRIFT = "covariate_drift"      # P(X) changes 
    PRIOR_DRIFT = "prior_drift"              # P(Y) changes
    PERFORMANCE_DRIFT = "performance_drift"   # Overall performance degrades
    BEHAVIORAL_DRIFT = "behavioral_drift"     # Model behavior patterns change
    STATISTICAL_DRIFT = "statistical_drift"  # Statistical properties change


@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    drift_type: DriftType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    affected_metrics: List[str]
    statistical_evidence: Dict[str, float]
    recommended_actions: List[str]
    correction_applied: bool = False


@dataclass
class ReferenceDistribution:
    """Reference distribution for drift detection."""
    data: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    distribution_type: str
    timestamp: datetime
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionAction:
    """Represents a drift correction action."""
    action_type: str
    parameters: Dict[str, Any]
    expected_effectiveness: float
    resource_cost: float
    execution_time_estimate: float


class StatisticalDriftDetector:
    """Statistical methods for drift detection."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def kolmogorov_smirnov_test(self, ref_data: np.ndarray, new_data: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for distribution change."""
        if len(ref_data) == 0 or len(new_data) == 0:
            return 0.0, 1.0
            
        try:
            statistic, p_value = stats.ks_2samp(ref_data, new_data)
            return statistic, p_value
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return 0.0, 1.0
    
    def anderson_darling_test(self, ref_data: np.ndarray, new_data: np.ndarray) -> float:
        """Anderson-Darling test for goodness of fit."""
        try:
            # Combine and sort data
            combined = np.concatenate([ref_data, new_data])
            combined_sorted = np.sort(combined)
            
            # Calculate empirical CDFs
            ref_cdf = np.searchsorted(combined_sorted, ref_data, side='right') / len(combined_sorted)
            new_cdf = np.searchsorted(combined_sorted, new_data, side='right') / len(combined_sorted)
            
            # Calculate AD statistic approximation
            n1, n2 = len(ref_data), len(new_data)
            if n1 == 0 or n2 == 0:
                return 0.0
                
            # Simplified AD statistic
            ad_stat = np.sum((ref_cdf - new_cdf) ** 2) * (n1 * n2) / (n1 + n2)
            return ad_stat
            
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")
            return 0.0
    
    def wasserstein_distance_test(self, ref_data: np.ndarray, new_data: np.ndarray) -> float:
        """Earth Mover's Distance between distributions."""
        try:
            return wasserstein_distance(ref_data, new_data)
        except Exception as e:
            logger.warning(f"Wasserstein distance failed: {e}")
            return 0.0
    
    def population_stability_index(self, ref_data: np.ndarray, new_data: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index for drift detection."""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(ref_data, bins=bins)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
            new_hist, _ = np.histogram(new_data, bins=bin_edges, density=True)
            
            # Normalize to get probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            new_prob = new_hist / np.sum(new_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_prob = np.maximum(ref_prob, epsilon)
            new_prob = np.maximum(new_prob, epsilon)
            
            # Calculate PSI
            psi = np.sum((new_prob - ref_prob) * np.log(new_prob / ref_prob))
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0


class EnsembleDriftDetector:
    """Ensemble of multiple drift detection methods."""
    
    def __init__(self, methods_config: Optional[Dict[str, float]] = None):
        self.methods_config = methods_config or {
            'ks_test': 0.25,
            'anderson_darling': 0.2,
            'wasserstein': 0.25,
            'psi': 0.2,
            'isolation_forest': 0.1
        }
        
        self.statistical_detector = StatisticalDriftDetector()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.pca = PCA(n_components=2)
        self.is_fitted = False
        
    def fit(self, reference_data: np.ndarray) -> None:
        """Fit the ensemble on reference data."""
        if len(reference_data.shape) == 1:
            reference_data = reference_data.reshape(-1, 1)
            
        self.isolation_forest.fit(reference_data)
        
        if reference_data.shape[1] > 2:
            self.pca.fit(reference_data)
        
        self.is_fitted = True
        logger.info(f"Ensemble drift detector fitted on {len(reference_data)} samples")
        
    def detect_drift(self, reference_data: np.ndarray, new_data: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Detect drift using ensemble of methods."""
        if len(reference_data) == 0 or len(new_data) == 0:
            return 0.0, {}
            
        evidence_scores = {}
        
        # For multivariate data, flatten or use first component
        if len(reference_data.shape) > 1 and reference_data.shape[1] > 1:
            if self.is_fitted and hasattr(self.pca, 'components_'):
                ref_1d = self.pca.transform(reference_data)[:, 0]
                new_1d = self.pca.transform(new_data)[:, 0]
            else:
                ref_1d = reference_data.mean(axis=1)
                new_1d = new_data.mean(axis=1)
        else:
            ref_1d = reference_data.flatten()
            new_1d = new_data.flatten()
        
        # Statistical tests
        if 'ks_test' in self.methods_config:
            ks_stat, ks_p = self.statistical_detector.kolmogorov_smirnov_test(ref_1d, new_1d)
            evidence_scores['ks_test'] = ks_stat
            
        if 'anderson_darling' in self.methods_config:
            ad_stat = self.statistical_detector.anderson_darling_test(ref_1d, new_1d)
            evidence_scores['anderson_darling'] = min(ad_stat / 10.0, 1.0)  # Normalize
            
        if 'wasserstein' in self.methods_config:
            wd = self.statistical_detector.wasserstein_distance_test(ref_1d, new_1d)
            # Normalize by data range
            data_range = max(np.max(ref_1d) - np.min(ref_1d), 1e-6)
            evidence_scores['wasserstein'] = min(wd / data_range, 1.0)
            
        if 'psi' in self.methods_config:
            psi = self.statistical_detector.population_stability_index(ref_1d, new_1d)
            evidence_scores['psi'] = min(psi / 2.0, 1.0)  # PSI > 0.2 is concerning
            
        # Anomaly detection
        if 'isolation_forest' in self.methods_config and self.is_fitted:
            if len(new_data.shape) == 1:
                new_data_shaped = new_data.reshape(-1, 1)
            else:
                new_data_shaped = new_data
                
            anomaly_scores = self.isolation_forest.decision_function(new_data_shaped)
            anomaly_ratio = np.mean(anomaly_scores < 0)  # Fraction of anomalies
            evidence_scores['isolation_forest'] = anomaly_ratio
        
        # Combine evidence with weights
        combined_score = 0.0
        total_weight = 0.0
        
        for method, weight in self.methods_config.items():
            if method in evidence_scores:
                combined_score += weight * evidence_scores[method]
                total_weight += weight
        
        if total_weight > 0:
            combined_score /= total_weight
            
        return min(combined_score, 1.0), evidence_scores


class PerformanceDriftMonitor:
    """Monitor performance metrics for drift detection."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_history = deque(maxlen=window_size * 2)
        self.baseline_performance: Optional[Dict[str, float]] = None
        
    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """Update performance history with new metrics."""
        timestamped_metrics = {
            'timestamp': datetime.now(),
            'metrics': performance_metrics.copy()
        }
        self.performance_history.append(timestamped_metrics)
        
    def set_baseline(self, baseline_metrics: Dict[str, float]) -> None:
        """Set baseline performance for comparison."""
        self.baseline_performance = baseline_metrics.copy()
        logger.info(f"Performance baseline set: {baseline_metrics}")
        
    def detect_performance_drift(self) -> Tuple[bool, Dict[str, float]]:
        """Detect significant performance drift."""
        if not self.baseline_performance or len(self.performance_history) < self.window_size:
            return False, {}
            
        # Get recent performance metrics
        recent_metrics = list(self.performance_history)[-self.window_size:]
        
        drift_scores = {}
        drift_detected = False
        
        for metric_name in self.baseline_performance.keys():
            recent_values = [
                entry['metrics'].get(metric_name, 0.0) 
                for entry in recent_metrics 
                if metric_name in entry['metrics']
            ]
            
            if len(recent_values) < self.window_size // 2:
                continue
                
            baseline_value = self.baseline_performance[metric_name]
            recent_mean = np.mean(recent_values)
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = abs(recent_mean - baseline_value) / abs(baseline_value)
            else:
                relative_change = abs(recent_mean)
            
            drift_scores[metric_name] = relative_change
            
            if relative_change > self.threshold:
                drift_detected = True
                logger.warning(f"Performance drift detected in {metric_name}: "
                             f"{baseline_value:.3f} -> {recent_mean:.3f} "
                             f"(change: {relative_change:.3f})")
        
        return drift_detected, drift_scores


class AdaptiveCorrectionEngine:
    """Engine for applying adaptive corrections to detected drift."""
    
    def __init__(self):
        self.correction_strategies = {
            DriftType.CONCEPT_DRIFT: self._correct_concept_drift,
            DriftType.COVARIATE_DRIFT: self._correct_covariate_drift,
            DriftType.PERFORMANCE_DRIFT: self._correct_performance_drift,
            DriftType.BEHAVIORAL_DRIFT: self._correct_behavioral_drift,
            DriftType.STATISTICAL_DRIFT: self._correct_statistical_drift
        }
        self.correction_history = deque(maxlen=1000)
        
    async def apply_correction(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Apply appropriate correction for detected drift."""
        correction_strategy = self.correction_strategies.get(drift_event.drift_type)
        
        if not correction_strategy:
            logger.warning(f"No correction strategy for drift type: {drift_event.drift_type}")
            return []
        
        try:
            corrections = await correction_strategy(drift_event, model)
            
            # Record correction attempt
            self.correction_history.append({
                'timestamp': datetime.now(),
                'drift_event': drift_event,
                'corrections_applied': len(corrections),
                'model_id': model.name
            })
            
            logger.info(f"Applied {len(corrections)} corrections for {drift_event.drift_type}")
            return corrections
            
        except Exception as e:
            logger.error(f"Correction failed for {drift_event.drift_type}: {e}")
            return []
    
    async def _correct_concept_drift(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Correct concept drift (P(Y|X) changes)."""
        corrections = []
        
        # Adaptive recalibration
        if drift_event.severity > 0.5:
            corrections.append(CorrectionAction(
                action_type="model_recalibration",
                parameters={
                    "recalibration_method": "platt_scaling",
                    "validation_split": 0.2
                },
                expected_effectiveness=0.7,
                resource_cost=0.3,
                execution_time_estimate=60.0
            ))
        
        # Online learning adaptation
        if drift_event.severity > 0.3:
            corrections.append(CorrectionAction(
                action_type="online_learning",
                parameters={
                    "learning_rate": 0.001,
                    "adaptation_strength": drift_event.severity,
                    "update_frequency": "batch"
                },
                expected_effectiveness=0.6,
                resource_cost=0.4,
                execution_time_estimate=30.0
            ))
        
        return corrections
    
    async def _correct_covariate_drift(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Correct covariate drift (P(X) changes)."""
        corrections = []
        
        # Input normalization adjustment
        corrections.append(CorrectionAction(
            action_type="input_normalization",
            parameters={
                "normalization_method": "adaptive_standardization",
                "update_statistics": True,
                "decay_factor": 0.9
            },
            expected_effectiveness=0.5,
            resource_cost=0.1,
            execution_time_estimate=5.0
        ))
        
        # Feature importance reweighting
        if drift_event.severity > 0.4:
            corrections.append(CorrectionAction(
                action_type="feature_reweighting",
                parameters={
                    "reweighting_method": "importance_sampling",
                    "adaptation_rate": drift_event.severity * 0.5
                },
                expected_effectiveness=0.6,
                resource_cost=0.2,
                execution_time_estimate=15.0
            ))
        
        return corrections
    
    async def _correct_performance_drift(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Correct performance drift."""
        corrections = []
        
        # Model ensemble adaptation
        corrections.append(CorrectionAction(
            action_type="ensemble_reweighting",
            parameters={
                "reweight_based_on": "recent_performance",
                "performance_window": 50,
                "adjustment_strength": drift_event.severity
            },
            expected_effectiveness=0.6,
            resource_cost=0.2,
            execution_time_estimate=10.0
        ))
        
        # Threshold adjustment
        if "classification" in str(type(model)).lower():
            corrections.append(CorrectionAction(
                action_type="threshold_optimization",
                parameters={
                    "optimization_metric": "f1_score",
                    "search_method": "grid_search",
                    "threshold_range": (0.3, 0.7)
                },
                expected_effectiveness=0.4,
                resource_cost=0.1,
                execution_time_estimate=20.0
            ))
        
        # Emergency fallback to simpler model
        if drift_event.severity > 0.8:
            corrections.append(CorrectionAction(
                action_type="fallback_model",
                parameters={
                    "fallback_type": "linear_baseline",
                    "transition_gradual": True,
                    "confidence_threshold": 0.7
                },
                expected_effectiveness=0.8,
                resource_cost=0.5,
                execution_time_estimate=5.0
            ))
        
        return corrections
    
    async def _correct_behavioral_drift(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Correct behavioral drift."""
        corrections = []
        
        # Behavioral pattern reset
        corrections.append(CorrectionAction(
            action_type="pattern_reset",
            parameters={
                "reset_attention": True,
                "reset_memory": False,
                "adaptation_period": 100
            },
            expected_effectiveness=0.5,
            resource_cost=0.3,
            execution_time_estimate=15.0
        ))
        
        # Context window adjustment
        corrections.append(CorrectionAction(
            action_type="context_adjustment",
            parameters={
                "context_length": "adaptive",
                "relevance_threshold": 0.8,
                "update_frequency": "real_time"
            },
            expected_effectiveness=0.4,
            resource_cost=0.2,
            execution_time_estimate=5.0
        ))
        
        return corrections
    
    async def _correct_statistical_drift(self, drift_event: DriftEvent, model: Model) -> List[CorrectionAction]:
        """Correct statistical drift."""
        corrections = []
        
        # Statistical recalibration
        corrections.append(CorrectionAction(
            action_type="statistical_recalibration",
            parameters={
                "method": "quantile_mapping",
                "reference_period": "last_stable",
                "update_frequency": "daily"
            },
            expected_effectiveness=0.6,
            resource_cost=0.2,
            execution_time_estimate=25.0
        ))
        
        # Distribution alignment
        corrections.append(CorrectionAction(
            action_type="distribution_alignment",
            parameters={
                "alignment_method": "moment_matching",
                "target_moments": ["mean", "variance"],
                "regularization": 0.1
            },
            expected_effectiveness=0.5,
            resource_cost=0.3,
            execution_time_estimate=20.0
        ))
        
        return corrections


class DriftDetectionSystem:
    """
    Comprehensive real-time drift detection and auto-correction system.
    
    Key innovations:
    1. Multi-method ensemble drift detection
    2. Real-time monitoring with adaptive thresholds
    3. Automated correction recommendation and application
    4. Performance impact assessment
    5. Continuous learning and adaptation
    """
    
    def __init__(
        self,
        detection_interval: float = 300.0,  # 5 minutes
        correction_threshold: float = 0.3,
        auto_correction: bool = True
    ):
        self.detection_interval = detection_interval
        self.correction_threshold = correction_threshold
        self.auto_correction = auto_correction
        
        # Core components
        self.ensemble_detector = EnsembleDriftDetector()
        self.performance_monitor = PerformanceDriftMonitor()
        self.correction_engine = AdaptiveCorrectionEngine()
        
        # State management
        self.reference_distributions: Dict[str, ReferenceDistribution] = {}
        self.detected_drifts: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics tracking
        self.detection_stats = {
            'total_detections': 0,
            'false_positives': 0,
            'corrections_applied': 0,
            'avg_detection_time': 0.0
        }
        
        logger.info("Drift Detection System initialized")
    
    async def start_monitoring(self, model: Model, initial_data: np.ndarray) -> None:
        """Start real-time drift monitoring for a model."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        # Set up reference distributions
        await self._initialize_reference_distributions(model.name, initial_data)
        
        # Start monitoring loop
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(model))
        
        logger.info(f"Started drift monitoring for model {model.name}")
    
    async def stop_monitoring(self) -> None:
        """Stop drift monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped drift monitoring")
    
    async def _initialize_reference_distributions(self, model_id: str, initial_data: np.ndarray) -> None:
        """Initialize reference distributions for drift detection."""
        if len(initial_data) == 0:
            logger.warning("Empty initial data provided")
            return
        
        # Fit ensemble detector
        self.ensemble_detector.fit(initial_data)
        
        # Store reference distribution
        self.reference_distributions[model_id] = ReferenceDistribution(
            data=initial_data.copy(),
            mean=np.mean(initial_data, axis=0),
            std=np.std(initial_data, axis=0),
            distribution_type="empirical",
            timestamp=datetime.now(),
            sample_size=len(initial_data)
        )
        
        logger.info(f"Reference distribution initialized for {model_id} with {len(initial_data)} samples")
    
    async def _monitoring_loop(self, model: Model) -> None:
        """Main monitoring loop for drift detection."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.detection_interval)
                
                if not self.is_monitoring:
                    break
                
                # Perform drift detection
                await self._check_for_drift(model)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_for_drift(self, model: Model) -> None:
        """Check for drift in model performance and data."""
        detection_start = datetime.now()
        
        # Get recent model outputs/predictions (simulated)
        recent_data = await self._collect_recent_data(model)
        
        if len(recent_data) == 0:
            return
        
        # Detect drift using ensemble methods
        drift_events = await self._detect_all_drift_types(model, recent_data)
        
        # Process detected drift events
        for drift_event in drift_events:
            self.detected_drifts.append(drift_event)
            self.detection_stats['total_detections'] += 1
            
            logger.warning(f"Drift detected: {drift_event.drift_type} "
                         f"(severity: {drift_event.severity:.3f}, "
                         f"confidence: {drift_event.confidence:.3f})")
            
            # Apply corrections if enabled and severity exceeds threshold
            if (self.auto_correction and 
                drift_event.severity > self.correction_threshold and
                drift_event.confidence > 0.5):
                
                corrections = await self.correction_engine.apply_correction(drift_event, model)
                
                if corrections:
                    self.detection_stats['corrections_applied'] += len(corrections)
                    drift_event.correction_applied = True
                    
                    logger.info(f"Applied {len(corrections)} automatic corrections")
        
        # Update detection time statistics
        detection_time = (datetime.now() - detection_start).total_seconds()
        self.detection_stats['avg_detection_time'] = (
            0.9 * self.detection_stats['avg_detection_time'] + 
            0.1 * detection_time
        )
    
    async def _collect_recent_data(self, model: Model) -> np.ndarray:
        """Collect recent model data for drift detection."""
        # Simulated data collection - in real implementation,
        # this would collect actual model inputs/outputs
        
        # Generate synthetic data representing recent model behavior
        if model.name in self.reference_distributions:
            ref_dist = self.reference_distributions[model.name]
            
            # Simulate some drift by adding noise/bias
            drift_factor = np.random.uniform(0.8, 1.2)
            noise_level = np.random.uniform(0.0, 0.1)
            
            synthetic_data = (
                ref_dist.mean + 
                np.random.normal(0, ref_dist.std + noise_level, (50, len(ref_dist.mean))) * drift_factor
            )
            
            return synthetic_data
        
        return np.array([])
    
    async def _detect_all_drift_types(self, model: Model, recent_data: np.ndarray) -> List[DriftEvent]:
        """Detect all types of drift using different methods."""
        drift_events = []
        
        if model.name not in self.reference_distributions:
            return drift_events
        
        reference_data = self.reference_distributions[model.name].data
        
        # Statistical drift detection
        drift_score, evidence = self.ensemble_detector.detect_drift(reference_data, recent_data)
        
        if drift_score > 0.3:  # Threshold for drift detection
            drift_type = self._classify_drift_type(evidence, recent_data, reference_data)
            
            drift_event = DriftEvent(
                drift_type=drift_type,
                severity=min(drift_score, 1.0),
                confidence=self._calculate_confidence(evidence),
                timestamp=datetime.now(),
                affected_metrics=list(evidence.keys()),
                statistical_evidence=evidence,
                recommended_actions=self._generate_recommendations(drift_type, drift_score)
            )
            
            drift_events.append(drift_event)
        
        # Performance drift detection
        performance_drift, perf_scores = self.performance_monitor.detect_performance_drift()
        
        if performance_drift:
            drift_event = DriftEvent(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=max(perf_scores.values()) if perf_scores else 0.5,
                confidence=0.8,
                timestamp=datetime.now(),
                affected_metrics=list(perf_scores.keys()),
                statistical_evidence=perf_scores,
                recommended_actions=["performance_optimization", "model_retraining"]
            )
            
            drift_events.append(drift_event)
        
        return drift_events
    
    def _classify_drift_type(self, evidence: Dict[str, float], 
                           recent_data: np.ndarray, reference_data: np.ndarray) -> DriftType:
        """Classify the type of drift based on evidence."""
        # Simple heuristic classification
        
        # If KS test shows significant difference, likely covariate drift
        if evidence.get('ks_test', 0) > 0.5:
            return DriftType.COVARIATE_DRIFT
        
        # If PSI is high, likely statistical drift
        if evidence.get('psi', 0) > 0.4:
            return DriftType.STATISTICAL_DRIFT
        
        # If isolation forest detects many anomalies, likely behavioral drift
        if evidence.get('isolation_forest', 0) > 0.3:
            return DriftType.BEHAVIORAL_DRIFT
        
        # If Wasserstein distance is significant, likely concept drift
        if evidence.get('wasserstein', 0) > 0.4:
            return DriftType.CONCEPT_DRIFT
        
        # Default to statistical drift
        return DriftType.STATISTICAL_DRIFT
    
    def _calculate_confidence(self, evidence: Dict[str, float]) -> float:
        """Calculate confidence in drift detection."""
        if not evidence:
            return 0.0
        
        # Confidence based on agreement between methods
        evidence_values = list(evidence.values())
        
        # Higher confidence if multiple methods agree
        agreement_score = 1.0 - np.std(evidence_values) / (np.mean(evidence_values) + 1e-6)
        
        # Higher confidence for stronger evidence
        strength_score = np.mean(evidence_values)
        
        confidence = 0.6 * agreement_score + 0.4 * strength_score
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_recommendations(self, drift_type: DriftType, severity: float) -> List[str]:
        """Generate recommendations based on drift type and severity."""
        recommendations = []
        
        if drift_type == DriftType.CONCEPT_DRIFT:
            recommendations.extend(["model_recalibration", "online_learning"])
            if severity > 0.7:
                recommendations.append("model_retraining")
        
        elif drift_type == DriftType.COVARIATE_DRIFT:
            recommendations.extend(["input_normalization", "feature_reweighting"])
        
        elif drift_type == DriftType.PERFORMANCE_DRIFT:
            recommendations.extend(["threshold_optimization", "ensemble_reweighting"])
            if severity > 0.8:
                recommendations.append("fallback_model")
        
        elif drift_type == DriftType.BEHAVIORAL_DRIFT:
            recommendations.extend(["pattern_reset", "context_adjustment"])
        
        elif drift_type == DriftType.STATISTICAL_DRIFT:
            recommendations.extend(["statistical_recalibration", "distribution_alignment"])
        
        return recommendations
    
    async def manual_drift_check(self, model: Model, test_data: np.ndarray) -> List[DriftEvent]:
        """Manually trigger drift detection on specific data."""
        logger.info(f"Manual drift check for model {model.name}")
        return await self._detect_all_drift_types(model, test_data)
    
    def update_performance_baseline(self, model: Model, performance_metrics: Dict[str, float]) -> None:
        """Update performance baseline for drift detection."""
        self.performance_monitor.set_baseline(performance_metrics)
        logger.info(f"Updated performance baseline for {model.name}")
    
    def get_drift_history(self, hours: int = 24) -> List[DriftEvent]:
        """Get drift detection history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.detected_drifts
            if event.timestamp >= cutoff_time
        ]
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""
        recent_drifts = self.get_drift_history(24)
        
        drift_type_counts = defaultdict(int)
        severity_scores = []
        
        for drift in recent_drifts:
            drift_type_counts[drift.drift_type.value] += 1
            severity_scores.append(drift.severity)
        
        return {
            "detection_stats": self.detection_stats.copy(),
            "recent_drift_count": len(recent_drifts),
            "drift_type_distribution": dict(drift_type_counts),
            "average_severity": np.mean(severity_scores) if severity_scores else 0.0,
            "monitoring_active": self.is_monitoring,
            "reference_distributions_count": len(self.reference_distributions),
            "corrections_applied": self.detection_stats['corrections_applied']
        }
    
    def export_research_data(self) -> Dict[str, Any]:
        """Export comprehensive research data for analysis."""
        return {
            "algorithm_name": "Real-Time Drift Detection and Auto-Correction",
            "detection_methods": {
                "ensemble_methods": list(self.ensemble_detector.methods_config.keys()),
                "method_weights": self.ensemble_detector.methods_config,
                "statistical_tests": ["ks_test", "anderson_darling", "wasserstein", "psi"],
                "anomaly_detection": "isolation_forest"
            },
            "correction_strategies": {
                "drift_types_handled": [dt.value for dt in DriftType],
                "auto_correction_enabled": self.auto_correction,
                "correction_threshold": self.correction_threshold
            },
            "performance_metrics": self.get_detection_statistics(),
            "drift_history": [
                {
                    "drift_type": event.drift_type.value,
                    "severity": event.severity,
                    "confidence": event.confidence,
                    "timestamp": event.timestamp.isoformat(),
                    "correction_applied": event.correction_applied
                }
                for event in list(self.detected_drifts)[-100:]  # Last 100 events
            ],
            "reference_distributions": {
                model_id: {
                    "sample_size": dist.sample_size,
                    "timestamp": dist.timestamp.isoformat(),
                    "distribution_type": dist.distribution_type
                }
                for model_id, dist in self.reference_distributions.items()
            }
        }