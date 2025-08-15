"""Intelligent caching with predictive prefetching and adaptive eviction."""

import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import json
import heapq
from enum import Enum

from .logging_config import get_logger, performance_logger
from .cache import CacheBackend, MemoryCache

logger = get_logger("intelligent_cache")


class AccessPattern(Enum):
    """Types of access patterns for intelligent caching."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    LOCALITY = "locality"


@dataclass
class CacheAccess:
    """Represents a cache access event."""
    key: str
    timestamp: datetime
    hit: bool
    access_time_ms: float
    user_context: Optional[str] = None


@dataclass
class PredictionScore:
    """Score for cache prediction algorithms."""
    key: str
    score: float
    prediction_type: str
    confidence: float


class AccessPatternAnalyzer:
    """Analyzes access patterns to predict future cache needs."""
    
    def __init__(self, window_size: int = 1000):
        self.access_history: List[CacheAccess] = []
        self.window_size = window_size
        self.pattern_weights = {
            AccessPattern.SEQUENTIAL: 0.3,
            AccessPattern.TEMPORAL: 0.4,
            AccessPattern.LOCALITY: 0.2,
            AccessPattern.RANDOM: 0.1
        }
    
    def record_access(self, access: CacheAccess):
        """Record a cache access for pattern analysis."""
        self.access_history.append(access)
        
        # Keep only recent accesses within window
        if len(self.access_history) > self.window_size:
            self.access_history = self.access_history[-self.window_size:]
    
    def analyze_patterns(self) -> Dict[AccessPattern, float]:
        """Analyze access patterns and return confidence scores."""
        if len(self.access_history) < 10:
            return {pattern: 0.0 for pattern in AccessPattern}
        
        patterns = {}
        
        # Sequential pattern analysis
        patterns[AccessPattern.SEQUENTIAL] = self._analyze_sequential_pattern()
        
        # Temporal pattern analysis
        patterns[AccessPattern.TEMPORAL] = self._analyze_temporal_pattern()
        
        # Locality pattern analysis
        patterns[AccessPattern.LOCALITY] = self._analyze_locality_pattern()
        
        # Random pattern (inversely correlated with others)
        other_patterns_avg = sum(patterns.values()) / len(patterns)
        patterns[AccessPattern.RANDOM] = max(0.0, 1.0 - other_patterns_avg)
        
        return patterns
    
    def _analyze_sequential_pattern(self) -> float:
        """Analyze for sequential access patterns."""
        if len(self.access_history) < 5:
            return 0.0
        
        sequential_count = 0
        total_pairs = 0
        
        for i in range(1, len(self.access_history)):
            current_key = self.access_history[i].key
            prev_key = self.access_history[i-1].key
            
            # Check if keys are sequential (simple heuristic)
            if self._are_keys_sequential(prev_key, current_key):
                sequential_count += 1
            total_pairs += 1
        
        return sequential_count / total_pairs if total_pairs > 0 else 0.0
    
    def _analyze_temporal_pattern(self) -> float:
        """Analyze for temporal patterns (recurring access times)."""
        if len(self.access_history) < 20:
            return 0.0
        
        # Group accesses by hour of day
        hour_counts = defaultdict(int)
        for access in self.access_history:
            hour = access.timestamp.hour
            hour_counts[hour] += 1
        
        # Calculate variance in hourly access
        access_counts = list(hour_counts.values())
        if not access_counts:
            return 0.0
        
        mean_count = sum(access_counts) / len(access_counts)
        variance = sum((count - mean_count) ** 2 for count in access_counts) / len(access_counts)
        
        # Higher variance indicates temporal patterns
        max_variance = mean_count ** 2  # Theoretical maximum
        return min(1.0, variance / max_variance) if max_variance > 0 else 0.0
    
    def _analyze_locality_pattern(self) -> float:
        """Analyze for spatial locality patterns."""
        if len(self.access_history) < 10:
            return 0.0
        
        # Track key reuse within sliding window
        window_size = 10
        locality_score = 0.0
        
        for i in range(window_size, len(self.access_history)):
            current_key = self.access_history[i].key
            recent_keys = {
                self.access_history[j].key 
                for j in range(i - window_size, i)
            }
            
            if current_key in recent_keys:
                locality_score += 1.0
        
        total_checks = max(1, len(self.access_history) - window_size)
        return locality_score / total_checks
    
    def _are_keys_sequential(self, key1: str, key2: str) -> bool:
        """Heuristic to determine if two keys are sequential."""
        try:
            # Extract numeric parts from keys
            import re
            
            nums1 = re.findall(r'\d+', key1)
            nums2 = re.findall(r'\d+', key2)
            
            if nums1 and nums2:
                # Check if numeric parts are consecutive
                for n1, n2 in zip(nums1, nums2):
                    if abs(int(n1) - int(n2)) == 1:
                        return True
            
            return False
        except (ValueError, IndexError):
            return False
    
    def predict_next_keys(self, limit: int = 10) -> List[PredictionScore]:
        """Predict the next keys likely to be accessed."""
        if len(self.access_history) < 5:
            return []
        
        patterns = self.analyze_patterns()
        predictions = []
        
        # Sequential predictions
        if patterns[AccessPattern.SEQUENTIAL] > 0.3:
            seq_predictions = self._predict_sequential_keys(limit // 2)
            predictions.extend(seq_predictions)
        
        # Temporal predictions
        if patterns[AccessPattern.TEMPORAL] > 0.3:
            temp_predictions = self._predict_temporal_keys(limit // 2)
            predictions.extend(temp_predictions)
        
        # Locality predictions
        if patterns[AccessPattern.LOCALITY] > 0.3:
            loc_predictions = self._predict_locality_keys(limit // 2)
            predictions.extend(loc_predictions)
        
        # Sort by score and return top predictions
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:limit]
    
    def _predict_sequential_keys(self, limit: int) -> List[PredictionScore]:
        """Predict sequential keys based on recent access patterns."""
        predictions = []
        
        if not self.access_history:
            return predictions
        
        last_key = self.access_history[-1].key
        
        try:
            # Extract numeric parts and predict next values
            import re
            nums = re.findall(r'\d+', last_key)
            
            for i, num_str in enumerate(nums):
                num = int(num_str)
                for offset in range(1, limit + 1):
                    predicted_key = last_key.replace(num_str, str(num + offset), 1)
                    score = 1.0 / (offset + 1)  # Higher score for closer predictions
                    
                    predictions.append(PredictionScore(
                        key=predicted_key,
                        score=score,
                        prediction_type="sequential",
                        confidence=0.8
                    ))
        except (ValueError, AttributeError):
            pass
        
        return predictions[:limit]
    
    def _predict_temporal_keys(self, limit: int) -> List[PredictionScore]:
        """Predict keys based on temporal access patterns."""
        predictions = []
        current_hour = datetime.now().hour
        
        # Find keys frequently accessed at this hour
        hour_key_counts = defaultdict(int)
        
        for access in self.access_history:
            if access.timestamp.hour == current_hour:
                hour_key_counts[access.key] += 1
        
        # Sort by frequency and create predictions
        sorted_keys = sorted(hour_key_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (key, count) in enumerate(sorted_keys[:limit]):
            score = count / len(self.access_history)
            predictions.append(PredictionScore(
                key=key,
                score=score,
                prediction_type="temporal",
                confidence=0.7
            ))
        
        return predictions
    
    def _predict_locality_keys(self, limit: int) -> List[PredictionScore]:
        """Predict keys based on locality patterns."""
        predictions = []
        
        if len(self.access_history) < 5:
            return predictions
        
        # Get recently accessed keys
        recent_keys = [access.key for access in self.access_history[-10:]]
        key_counts = defaultdict(int)
        
        for key in recent_keys:
            key_counts[key] += 1
        
        # Predict re-access of recent keys
        for key, count in key_counts.items():
            score = count / len(recent_keys)
            predictions.append(PredictionScore(
                key=key,
                score=score,
                prediction_type="locality",
                confidence=0.6
            ))
        
        return sorted(predictions, key=lambda x: x.score, reverse=True)[:limit]


class AdaptiveEvictionPolicy:
    """Adaptive cache eviction policy based on access patterns and prediction."""
    
    def __init__(self):
        self.access_frequencies = defaultdict(int)
        self.last_access_times = {}
        self.prediction_scores = {}
        self.eviction_weights = {
            "frequency": 0.3,
            "recency": 0.3,
            "prediction": 0.4
        }
    
    def record_access(self, key: str):
        """Record access for eviction policy."""
        self.access_frequencies[key] += 1
        self.last_access_times[key] = time.time()
    
    def update_predictions(self, predictions: List[PredictionScore]):
        """Update prediction scores for eviction decisions."""
        self.prediction_scores.clear()
        for pred in predictions:
            self.prediction_scores[pred.key] = pred.score
    
    def select_eviction_candidates(self, keys: List[str], count: int) -> List[str]:
        """Select keys for eviction based on adaptive policy."""
        if not keys or count <= 0:
            return []
        
        key_scores = []
        current_time = time.time()
        
        for key in keys:
            # Frequency score (normalized)
            freq_score = self.access_frequencies.get(key, 0)
            max_freq = max(self.access_frequencies.values()) if self.access_frequencies else 1
            freq_score = freq_score / max_freq
            
            # Recency score (normalized)
            last_access = self.last_access_times.get(key, 0)
            time_since_access = current_time - last_access
            recency_score = 1.0 / (1.0 + time_since_access / 3600.0)  # Decay over hours
            
            # Prediction score
            pred_score = self.prediction_scores.get(key, 0)
            
            # Combined score (lower is better for eviction)
            combined_score = (
                self.eviction_weights["frequency"] * freq_score +
                self.eviction_weights["recency"] * recency_score +
                self.eviction_weights["prediction"] * pred_score
            )
            
            key_scores.append((key, combined_score))
        
        # Sort by score (ascending) and return worst candidates for eviction
        key_scores.sort(key=lambda x: x[1])
        return [key for key, score in key_scores[:count]]


class IntelligentCacheManager:
    """Intelligent cache manager with predictive prefetching and adaptive eviction."""
    
    def __init__(self, 
                 backend: Optional[CacheBackend] = None,
                 max_size: int = 10000,
                 prefetch_enabled: bool = True,
                 prefetch_workers: int = 2):
        self.backend = backend or MemoryCache(max_size=max_size)
        self.max_size = max_size
        self.prefetch_enabled = prefetch_enabled
        
        # Intelligence components
        self.pattern_analyzer = AccessPatternAnalyzer()
        self.eviction_policy = AdaptiveEvictionPolicy()
        
        # Prefetching
        self.prefetch_queue = asyncio.Queue()
        self.prefetch_workers = prefetch_workers
        self.prefetch_tasks: List[asyncio.Task] = []
        self.prefetch_functions: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "prefetch_hits": 0,
            "evictions": 0,
            "predictions_made": 0,
            "prefetch_success": 0,
            "prefetch_failures": 0
        }
        
        self._running = False
    
    async def start(self):
        """Start the intelligent cache manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start prefetch workers
        if self.prefetch_enabled:
            for i in range(self.prefetch_workers):
                task = asyncio.create_task(self._prefetch_worker(f"worker_{i}"))
                self.prefetch_tasks.append(task)
        
        logger.info(f"Started intelligent cache manager with {self.prefetch_workers} prefetch workers")
    
    async def stop(self):
        """Stop the intelligent cache manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop prefetch workers
        for task in self.prefetch_tasks:
            task.cancel()
        
        await asyncio.gather(*self.prefetch_tasks, return_exceptions=True)
        self.prefetch_tasks.clear()
        
        logger.info("Stopped intelligent cache manager")
    
    async def get(self, key: str, user_context: Optional[str] = None) -> Any:
        """Get value from cache with intelligent access tracking."""
        start_time = time.time()
        
        try:
            value = await self.backend.get(key)
            
            # Record cache hit
            access = CacheAccess(
                key=key,
                timestamp=datetime.now(),
                hit=True,
                access_time_ms=(time.time() - start_time) * 1000,
                user_context=user_context
            )
            self._record_access(access)
            
            self.stats["hits"] += 1
            
            # Trigger predictive analysis
            await self._trigger_prediction_analysis()
            
            return value
            
        except KeyError:
            # Record cache miss
            access = CacheAccess(
                key=key,
                timestamp=datetime.now(),
                hit=False,
                access_time_ms=(time.time() - start_time) * 1000,
                user_context=user_context
            )
            self._record_access(access)
            
            self.stats["misses"] += 1
            raise
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with intelligent eviction."""
        # Check if eviction is needed
        current_size = await self.backend.size()
        if current_size >= self.max_size:
            await self._intelligent_eviction()
        
        await self.backend.set(key, value, ttl)
        self.eviction_policy.record_access(key)
    
    async def _intelligent_eviction(self):
        """Perform intelligent cache eviction."""
        keys = await self.backend.keys()
        if not keys:
            return
        
        # Determine how many items to evict (10% of cache size)
        evict_count = max(1, len(keys) // 10)
        
        # Select candidates using adaptive policy
        candidates = self.eviction_policy.select_eviction_candidates(keys, evict_count)
        
        # Evict selected candidates
        for key in candidates:
            await self.backend.delete(key)
            self.stats["evictions"] += 1
        
        logger.debug(f"Evicted {len(candidates)} items using intelligent policy")
    
    def _record_access(self, access: CacheAccess):
        """Record access for pattern analysis."""
        self.pattern_analyzer.record_access(access)
        self.eviction_policy.record_access(access.key)
        
        # Track prefetch hits
        if access.hit and access.key in getattr(self, '_prefetched_keys', set()):
            self.stats["prefetch_hits"] += 1
    
    async def _trigger_prediction_analysis(self):
        """Trigger predictive analysis and prefetching."""
        if not self.prefetch_enabled or not self._running:
            return
        
        # Analyze patterns periodically (every 100 accesses)
        total_accesses = self.stats["hits"] + self.stats["misses"]
        if total_accesses % 100 != 0:
            return
        
        try:
            # Get predictions
            predictions = self.pattern_analyzer.predict_next_keys(limit=20)
            self.stats["predictions_made"] += len(predictions)
            
            # Update eviction policy with predictions
            self.eviction_policy.update_predictions(predictions)
            
            # Queue high-confidence predictions for prefetching
            for pred in predictions:
                if pred.confidence > 0.6 and pred.score > 0.3:
                    await self.prefetch_queue.put(pred)
            
            logger.debug(f"Generated {len(predictions)} cache predictions")
            
        except Exception as e:
            logger.error(f"Error in prediction analysis: {e}")
    
    async def _prefetch_worker(self, worker_id: str):
        """Worker for prefetching predicted cache items."""
        if not hasattr(self, '_prefetched_keys'):
            self._prefetched_keys = set()
        
        while self._running:
            try:
                # Get prediction from queue
                prediction = await asyncio.wait_for(
                    self.prefetch_queue.get(), 
                    timeout=5.0
                )
                
                # Check if already cached
                try:
                    await self.backend.get(prediction.key)
                    continue  # Already cached
                except KeyError:
                    pass
                
                # Try to prefetch
                if prediction.key in self.prefetch_functions:
                    prefetch_func = self.prefetch_functions[prediction.key]
                    try:
                        value = await prefetch_func(prediction.key)
                        await self.set(prediction.key, value)
                        self._prefetched_keys.add(prediction.key)
                        self.stats["prefetch_success"] += 1
                        
                        logger.debug(f"Prefetched {prediction.key} (score: {prediction.score:.2f})")
                        
                    except Exception as e:
                        self.stats["prefetch_failures"] += 1
                        logger.warning(f"Prefetch failed for {prediction.key}: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    def register_prefetch_function(self, key_pattern: str, func: Callable):
        """Register a function to prefetch data for keys matching pattern."""
        self.prefetch_functions[key_pattern] = func
        logger.info(f"Registered prefetch function for pattern: {key_pattern}")
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get intelligence and performance statistics."""
        patterns = self.pattern_analyzer.analyze_patterns()
        
        total_accesses = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_accesses * 100) if total_accesses > 0 else 0
        prefetch_hit_rate = (self.stats["prefetch_hits"] / self.stats["hits"] * 100) if self.stats["hits"] > 0 else 0
        
        return {
            "cache_stats": {
                "hit_rate_percent": round(hit_rate, 2),
                "total_hits": self.stats["hits"],
                "total_misses": self.stats["misses"],
                "prefetch_hits": self.stats["prefetch_hits"],
                "prefetch_hit_rate_percent": round(prefetch_hit_rate, 2),
                "evictions": self.stats["evictions"]
            },
            "intelligence_stats": {
                "access_patterns": {pattern.value: round(score, 3) for pattern, score in patterns.items()},
                "predictions_made": self.stats["predictions_made"],
                "prefetch_success": self.stats["prefetch_success"],
                "prefetch_failures": self.stats["prefetch_failures"],
                "prefetch_success_rate": round(
                    self.stats["prefetch_success"] / max(1, self.stats["prefetch_success"] + self.stats["prefetch_failures"]) * 100, 2
                )
            },
            "system_stats": {
                "prefetch_enabled": self.prefetch_enabled,
                "prefetch_workers": self.prefetch_workers,
                "max_cache_size": self.max_size,
                "registered_prefetch_functions": len(self.prefetch_functions)
            }
        }


# Global intelligent cache manager
intelligent_cache = IntelligentCacheManager()