"""
Adaptive Benchmark Selection with Meta-Learning

Novel meta-learning algorithm that adapts benchmark selection based on model
characteristics, performance patterns, and transfer learning insights.

Research Innovation: "Dynamic Benchmark Adaptation via Meta-Learning Transfer"
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from ..core.models import Model
from ..core.benchmarks import Benchmark
from ..core.results import Results
from ..core.logging_config import get_logger

logger = get_logger("adaptive_benchmark")


@dataclass
class ModelProfile:
    """Comprehensive model performance profile."""
    model_id: str
    architecture_type: str  # transformer, cnn, rnn, etc.
    parameter_count: int
    training_data_size: int
    performance_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    capability_scores: Dict[str, float] = field(default_factory=dict)
    weakness_areas: List[str] = field(default_factory=list)
    strength_areas: List[str] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    
@dataclass
class BenchmarkMetadata:
    """Enhanced benchmark metadata for adaptive selection."""
    benchmark_id: str
    cognitive_domains: List[str]  # reasoning, memory, language, etc.
    difficulty_level: float  # 0.0 to 1.0
    discriminative_power: float  # Ability to differentiate models
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    computational_cost: float = 1.0
    reliability_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TransferLearningInsight:
    """Insights from transfer learning between benchmarks."""
    source_benchmark: str
    target_benchmark: str
    transfer_coefficient: float  # How well performance transfers
    knowledge_domains: List[str]
    confidence_score: float
    sample_efficiency: float  # How many samples needed for reliable transfer


class MetaLearningEngine:
    """Meta-learning engine for benchmark adaptation."""
    
    def __init__(self, adaptation_rate: float = 0.1, memory_size: int = 1000):
        self.adaptation_rate = adaptation_rate
        self.memory_size = memory_size
        self.performance_memory = deque(maxlen=memory_size)
        self.model_embeddings: Dict[str, np.ndarray] = {}
        self.benchmark_embeddings: Dict[str, np.ndarray] = {}
        self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def update_memory(self, model_id: str, benchmark_id: str, performance: float,
                     metadata: Dict[str, Any]) -> None:
        """Update meta-learning memory with new performance data."""
        memory_entry = {
            'model_id': model_id,
            'benchmark_id': benchmark_id,
            'performance': performance,
            'timestamp': datetime.now(),
            'metadata': metadata
        }
        self.performance_memory.append(memory_entry)
        
    def learn_model_embedding(self, model_profile: ModelProfile) -> np.ndarray:
        """Learn embedding representation for model characteristics."""
        # Create feature vector from model characteristics
        features = [
            model_profile.parameter_count / 1e9,  # Normalize parameter count
            len(model_profile.capability_scores),
            np.mean(list(model_profile.capability_scores.values())) if model_profile.capability_scores else 0,
            len(model_profile.strength_areas),
            len(model_profile.weakness_areas)
        ]
        
        # Add performance vector if available
        if len(model_profile.performance_vector) > 0:
            features.extend(model_profile.performance_vector[:10])  # Limit to first 10 dimensions
        else:
            features.extend([0] * 10)
            
        embedding = np.array(features, dtype=np.float32)
        self.model_embeddings[model_profile.model_id] = embedding
        return embedding
        
    def learn_benchmark_embedding(self, benchmark_metadata: BenchmarkMetadata) -> np.ndarray:
        """Learn embedding representation for benchmark characteristics."""
        features = [
            benchmark_metadata.difficulty_level,
            benchmark_metadata.discriminative_power,
            benchmark_metadata.computational_cost,
            benchmark_metadata.reliability_score,
            len(benchmark_metadata.cognitive_domains)
        ]
        
        # Add one-hot encoding for cognitive domains
        all_domains = ['reasoning', 'memory', 'language', 'mathematics', 'coding', 'knowledge']
        domain_encoding = [1 if domain in benchmark_metadata.cognitive_domains else 0 
                          for domain in all_domains]
        features.extend(domain_encoding)
        
        embedding = np.array(features, dtype=np.float32)
        self.benchmark_embeddings[benchmark_metadata.benchmark_id] = embedding
        return embedding
        
    def train_meta_model(self) -> None:
        """Train meta-learning model on accumulated performance data."""
        if len(self.performance_memory) < 50:  # Need minimum data
            logger.warning("Insufficient data for meta-learning training")
            return
            
        # Prepare training data
        X, y = [], []
        
        for entry in self.performance_memory:
            model_id = entry['model_id']
            benchmark_id = entry['benchmark_id']
            
            if model_id in self.model_embeddings and benchmark_id in self.benchmark_embeddings:
                # Concatenate model and benchmark embeddings
                model_emb = self.model_embeddings[model_id]
                benchmark_emb = self.benchmark_embeddings[benchmark_id]
                combined_features = np.concatenate([model_emb, benchmark_emb])
                
                X.append(combined_features)
                y.append(entry['performance'])
                
        if len(X) < 10:
            logger.warning("Insufficient feature data for training")
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train meta-model
        self.meta_model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Meta-learning model trained on {len(X)} samples")
        
    def predict_performance(self, model_id: str, benchmark_id: str) -> float:
        """Predict model performance on benchmark using meta-learning."""
        if not self.is_trained:
            return 0.5  # Default prediction
            
        if model_id not in self.model_embeddings or benchmark_id not in self.benchmark_embeddings:
            return 0.5
            
        # Prepare features for prediction
        model_emb = self.model_embeddings[model_id]
        benchmark_emb = self.benchmark_embeddings[benchmark_id]
        features = np.concatenate([model_emb, benchmark_emb]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict performance
        prediction = self.meta_model.predict(features_scaled)[0]
        return max(0.0, min(1.0, prediction))  # Clamp to [0, 1]


class AdaptiveBenchmarkSelector:
    """
    Adaptive benchmark selection system with meta-learning capabilities.
    
    Key innovations:
    1. Meta-learning for performance prediction
    2. Dynamic benchmark portfolio optimization
    3. Transfer learning between related benchmarks
    4. Adaptive difficulty adjustment
    5. Multi-objective optimization (accuracy, efficiency, coverage)
    """
    
    def __init__(self, adaptation_strength: float = 0.3, diversity_weight: float = 0.4):
        self.adaptation_strength = adaptation_strength
        self.diversity_weight = diversity_weight
        self.meta_learner = MetaLearningEngine()
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.benchmark_metadata: Dict[str, BenchmarkMetadata] = {}
        self.transfer_insights: List[TransferLearningInsight] = []
        self.selection_history: List[Dict[str, Any]] = []
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        
    async def adaptive_select(
        self,
        model: Model,
        available_benchmarks: List[Benchmark],
        target_count: int = 5,
        optimization_objective: str = "balanced"  # balanced, accuracy, speed, coverage
    ) -> List[Benchmark]:
        """
        Adaptively select optimal benchmarks for a given model.
        
        Args:
            model: Model to evaluate
            available_benchmarks: Pool of available benchmarks
            target_count: Number of benchmarks to select
            optimization_objective: Optimization strategy
            
        Returns:
            Optimally selected benchmarks for the model
        """
        logger.info(f"Adaptive benchmark selection for model {model.name}")
        
        # Step 1: Update model profile
        await self._update_model_profile(model)
        
        # Step 2: Calculate benchmark scores
        benchmark_scores = await self._calculate_benchmark_scores(
            model, available_benchmarks, optimization_objective
        )
        
        # Step 3: Apply diversity constraints
        selected_benchmarks = await self._optimize_benchmark_portfolio(
            available_benchmarks, benchmark_scores, target_count
        )
        
        # Step 4: Record selection for learning
        self._record_selection(model, selected_benchmarks, benchmark_scores)
        
        logger.info(f"Selected {len(selected_benchmarks)} benchmarks: {[b.name for b in selected_benchmarks]}")
        return selected_benchmarks
        
    async def _update_model_profile(self, model: Model) -> None:
        """Update or create model profile with latest characteristics."""
        model_id = model.name
        
        if model_id not in self.model_profiles:
            # Create new profile
            profile = ModelProfile(
                model_id=model_id,
                architecture_type=getattr(model, 'architecture_type', 'unknown'),
                parameter_count=getattr(model, 'parameter_count', 0),
                training_data_size=getattr(model, 'training_data_size', 0)
            )
        else:
            profile = self.model_profiles[model_id]
            
        # Update capability scores based on recent performance
        if hasattr(model, 'recent_performance'):
            profile.capability_scores.update(model.recent_performance)
            
        # Learn embedding representation
        self.meta_learner.learn_model_embedding(profile)
        self.model_profiles[model_id] = profile
        
    async def _calculate_benchmark_scores(
        self,
        model: Model,
        benchmarks: List[Benchmark],
        objective: str
    ) -> Dict[str, float]:
        """Calculate adaptive scores for each benchmark."""
        scores = {}
        
        for benchmark in benchmarks:
            # Ensure benchmark metadata exists
            await self._update_benchmark_metadata(benchmark)
            
            # Get base score from meta-learning prediction
            base_score = self.meta_learner.predict_performance(model.name, benchmark.name)
            
            # Apply objective-specific adjustments
            objective_score = await self._apply_objective_weighting(
                benchmark, base_score, objective
            )
            
            # Add transfer learning insights
            transfer_score = await self._calculate_transfer_value(
                model.name, benchmark.name
            )
            
            # Combine scores
            final_score = (
                0.5 * objective_score + 
                0.3 * transfer_score + 
                0.2 * base_score
            )
            
            scores[benchmark.name] = final_score
            
        return scores
        
    async def _update_benchmark_metadata(self, benchmark: Benchmark) -> None:
        """Update or create benchmark metadata."""
        benchmark_id = benchmark.name
        
        if benchmark_id not in self.benchmark_metadata:
            # Analyze benchmark characteristics
            metadata = BenchmarkMetadata(
                benchmark_id=benchmark_id,
                cognitive_domains=self._infer_cognitive_domains(benchmark),
                difficulty_level=self._estimate_difficulty(benchmark),
                discriminative_power=self._calculate_discriminative_power(benchmark),
                computational_cost=getattr(benchmark, 'computational_cost', 1.0),
                reliability_score=getattr(benchmark, 'reliability_score', 1.0)
            )
            
            # Learn benchmark embedding
            self.meta_learner.learn_benchmark_embedding(metadata)
            self.benchmark_metadata[benchmark_id] = metadata
            
    def _infer_cognitive_domains(self, benchmark: Benchmark) -> List[str]:
        """Infer cognitive domains tested by benchmark."""
        benchmark_name = benchmark.name.lower()
        
        domains = []
        if any(term in benchmark_name for term in ['math', 'arithmetic', 'calculation']):
            domains.append('mathematics')
        if any(term in benchmark_name for term in ['code', 'programming', 'eval']):
            domains.append('coding')
        if any(term in benchmark_name for term in ['reasoning', 'logic', 'inference']):
            domains.append('reasoning')
        if any(term in benchmark_name for term in ['language', 'nlp', 'text']):
            domains.append('language')
        if any(term in benchmark_name for term in ['knowledge', 'qa', 'truth']):
            domains.append('knowledge')
        if any(term in benchmark_name for term in ['memory', 'recall', 'context']):
            domains.append('memory')
            
        return domains if domains else ['general']
        
    def _estimate_difficulty(self, benchmark: Benchmark) -> float:
        """Estimate benchmark difficulty based on characteristics."""
        # Simple heuristic based on benchmark name and type
        difficulty_indicators = {
            'advanced': 0.9,
            'hard': 0.8,
            'complex': 0.8,
            'expert': 0.9,
            'professional': 0.7,
            'intermediate': 0.6,
            'basic': 0.3,
            'simple': 0.2,
            'easy': 0.2
        }
        
        benchmark_name = benchmark.name.lower()
        for indicator, level in difficulty_indicators.items():
            if indicator in benchmark_name:
                return level
                
        return 0.5  # Default medium difficulty
        
    def _calculate_discriminative_power(self, benchmark: Benchmark) -> float:
        """Calculate how well benchmark discriminates between models."""
        # Placeholder - would analyze variance in historical performance
        return 0.7  # Default discriminative power
        
    async def _apply_objective_weighting(
        self,
        benchmark: Benchmark,
        base_score: float,
        objective: str
    ) -> float:
        """Apply objective-specific weighting to benchmark scores."""
        metadata = self.benchmark_metadata[benchmark.name]
        
        if objective == "accuracy":
            # Prefer high discriminative power and reliability
            weight = 0.6 * metadata.discriminative_power + 0.4 * metadata.reliability_score
        elif objective == "speed":
            # Prefer low computational cost
            weight = 1.0 / max(metadata.computational_cost, 0.1)
        elif objective == "coverage":
            # Prefer diverse cognitive domains
            domain_diversity = len(metadata.cognitive_domains) / 6.0  # Normalize by max domains
            weight = domain_diversity
        else:  # balanced
            weight = (
                0.3 * metadata.discriminative_power +
                0.3 * metadata.reliability_score +
                0.2 * (1.0 / max(metadata.computational_cost, 0.1)) +
                0.2 * (len(metadata.cognitive_domains) / 6.0)
            )
            
        return base_score * weight
        
    async def _calculate_transfer_value(self, model_id: str, benchmark_id: str) -> float:
        """Calculate transfer learning value for benchmark selection."""
        transfer_value = 0.0
        
        for insight in self.transfer_insights:
            if insight.target_benchmark == benchmark_id:
                # Check if we have performance data for source benchmark
                source_performance = self._get_model_benchmark_performance(
                    model_id, insight.source_benchmark
                )
                if source_performance is not None:
                    transfer_contribution = (
                        insight.transfer_coefficient * 
                        insight.confidence_score * 
                        source_performance
                    )
                    transfer_value = max(transfer_value, transfer_contribution)
                    
        return min(transfer_value, 1.0)
        
    def _get_model_benchmark_performance(self, model_id: str, benchmark_id: str) -> Optional[float]:
        """Get historical performance of model on benchmark."""
        for entry in self.meta_learner.performance_memory:
            if entry['model_id'] == model_id and entry['benchmark_id'] == benchmark_id:
                return entry['performance']
        return None
        
    async def _optimize_benchmark_portfolio(
        self,
        available_benchmarks: List[Benchmark],
        scores: Dict[str, float],
        target_count: int
    ) -> List[Benchmark]:
        """Optimize benchmark portfolio for diversity and performance."""
        if len(available_benchmarks) <= target_count:
            return available_benchmarks
            
        # Create feature matrix for clustering
        features = []
        benchmark_names = []
        
        for benchmark in available_benchmarks:
            if benchmark.name in self.benchmark_metadata:
                metadata = self.benchmark_metadata[benchmark.name]
                feature_vector = [
                    metadata.difficulty_level,
                    metadata.discriminative_power,
                    metadata.computational_cost,
                    len(metadata.cognitive_domains)
                ]
                features.append(feature_vector)
                benchmark_names.append(benchmark.name)
                
        if len(features) < target_count:
            return available_benchmarks[:target_count]
            
        # Cluster benchmarks for diversity
        features_array = np.array(features)
        n_clusters = min(target_count, len(features))
        self.clustering_model.n_clusters = n_clusters
        cluster_labels = self.clustering_model.fit_predict(features_array)
        
        # Select best benchmark from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_benchmarks = [
                (benchmark_names[i], scores.get(benchmark_names[i], 0.0))
                for i in range(len(benchmark_names))
                if cluster_labels[i] == cluster_id
            ]
            
            if cluster_benchmarks:
                # Select highest scoring benchmark from cluster
                best_name, _ = max(cluster_benchmarks, key=lambda x: x[1])
                selected_benchmark = next(
                    b for b in available_benchmarks if b.name == best_name
                )
                selected.append(selected_benchmark)
                
        # Fill remaining slots with highest scoring benchmarks
        while len(selected) < target_count and len(selected) < len(available_benchmarks):
            remaining = [b for b in available_benchmarks if b not in selected]
            if not remaining:
                break
                
            best_remaining = max(remaining, key=lambda b: scores.get(b.name, 0.0))
            selected.append(best_remaining)
            
        return selected[:target_count]
        
    def _record_selection(
        self,
        model: Model,
        selected_benchmarks: List[Benchmark],
        scores: Dict[str, float]
    ) -> None:
        """Record benchmark selection for learning."""
        selection_record = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model.name,
            'selected_benchmarks': [b.name for b in selected_benchmarks],
            'benchmark_scores': scores,
            'selection_rationale': {
                'diversity_weight': self.diversity_weight,
                'adaptation_strength': self.adaptation_strength
            }
        }
        
        self.selection_history.append(selection_record)
        
    async def learn_transfer_insights(
        self,
        performance_data: Dict[str, Dict[str, float]]
    ) -> None:
        """Learn transfer learning insights from performance data."""
        logger.info("Learning transfer insights from performance data")
        
        benchmark_pairs = []
        for model_id, benchmarks in performance_data.items():
            benchmark_list = list(benchmarks.keys())
            for i in range(len(benchmark_list)):
                for j in range(i + 1, len(benchmark_list)):
                    benchmark_pairs.append((benchmark_list[i], benchmark_list[j]))
                    
        # Calculate transfer coefficients
        for source_bench, target_bench in set(benchmark_pairs):
            transfer_coeff = self._calculate_transfer_coefficient(
                source_bench, target_bench, performance_data
            )
            
            if transfer_coeff > 0.3:  # Threshold for meaningful transfer
                insight = TransferLearningInsight(
                    source_benchmark=source_bench,
                    target_benchmark=target_bench,
                    transfer_coefficient=transfer_coeff,
                    knowledge_domains=self._find_common_domains(source_bench, target_bench),
                    confidence_score=min(transfer_coeff * 2, 1.0),
                    sample_efficiency=transfer_coeff
                )
                self.transfer_insights.append(insight)
                
        logger.info(f"Learned {len(self.transfer_insights)} transfer insights")
        
    def _calculate_transfer_coefficient(
        self,
        source_bench: str,
        target_bench: str,
        performance_data: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate transfer coefficient between benchmarks."""
        source_perfs = []
        target_perfs = []
        
        for model_id, benchmarks in performance_data.items():
            if source_bench in benchmarks and target_bench in benchmarks:
                source_perfs.append(benchmarks[source_bench])
                target_perfs.append(benchmarks[target_bench])
                
        if len(source_perfs) < 3:  # Need minimum data points
            return 0.0
            
        # Calculate correlation
        correlation = np.corrcoef(source_perfs, target_perfs)[0, 1]
        return max(0.0, correlation)
        
    def _find_common_domains(self, bench1: str, bench2: str) -> List[str]:
        """Find common cognitive domains between benchmarks."""
        if bench1 in self.benchmark_metadata and bench2 in self.benchmark_metadata:
            domains1 = set(self.benchmark_metadata[bench1].cognitive_domains)
            domains2 = set(self.benchmark_metadata[bench2].cognitive_domains)
            return list(domains1 & domains2)
        return []
        
    def export_research_data(self) -> Dict[str, Any]:
        """Export comprehensive research data for analysis."""
        return {
            "algorithm_name": "Adaptive Benchmark Selection with Meta-Learning",
            "model_profiles": {
                model_id: {
                    "architecture_type": profile.architecture_type,
                    "parameter_count": profile.parameter_count,
                    "capability_scores": profile.capability_scores,
                    "adaptation_history_count": len(profile.adaptation_history)
                }
                for model_id, profile in self.model_profiles.items()
            },
            "benchmark_metadata": {
                bench_id: {
                    "cognitive_domains": metadata.cognitive_domains,
                    "difficulty_level": metadata.difficulty_level,
                    "discriminative_power": metadata.discriminative_power,
                    "computational_cost": metadata.computational_cost
                }
                for bench_id, metadata in self.benchmark_metadata.items()
            },
            "transfer_insights": [
                {
                    "source_benchmark": insight.source_benchmark,
                    "target_benchmark": insight.target_benchmark,
                    "transfer_coefficient": insight.transfer_coefficient,
                    "confidence_score": insight.confidence_score
                }
                for insight in self.transfer_insights
            ],
            "selection_history": self.selection_history,
            "meta_learning_trained": self.meta_learner.is_trained,
            "memory_size": len(self.meta_learner.performance_memory)
        }