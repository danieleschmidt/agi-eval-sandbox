#!/usr/bin/env python3
"""
Generation 3: Scalable Performance System
Advanced optimization, auto-scaling, intelligent caching, and performance monitoring
"""

import asyncio
import time
import logging
import sys
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import statistics
from collections import deque
import weakref

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class PerformanceLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    OPTIMIZED = "optimized"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    timestamp: datetime
    operation: str
    duration_ms: float
    cpu_percent: float
    memory_mb: float
    throughput_ops_sec: float
    success_rate: float
    error_count: int
    cache_hit_rate: float
    
@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration."""
    min_workers: int = 1
    max_workers: int = 10
    scale_up_threshold: float = 0.8  # CPU/Memory threshold to scale up
    scale_down_threshold: float = 0.3  # CPU/Memory threshold to scale down
    scale_up_cooldown: int = 60  # Seconds to wait before scaling up again
    scale_down_cooldown: int = 120  # Seconds to wait before scaling down again
    target_cpu_percent: float = 70.0
    target_response_time_ms: float = 1000.0

class IntelligentCache:
    """Advanced caching system with multiple strategies and adaptive behavior."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = {}
        self.ttl_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.performance_history = deque(maxlen=100)
        self.strategy_performance: Dict[CacheStrategy, float] = {}
        
        self.logger = logging.getLogger("intelligent_cache")
    
    def _should_evict_lru(self, key: str) -> bool:
        """Check if key should be evicted using LRU strategy."""
        if len(self.cache) < self.max_size:
            return False
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        return key == oldest_key
    
    def _should_evict_lfu(self, key: str) -> bool:
        """Check if key should be evicted using LFU strategy."""
        if len(self.cache) < self.max_size:
            return False
        
        least_frequent_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        return key == least_frequent_key
    
    def _should_evict_ttl(self, key: str) -> bool:
        """Check if key should be evicted due to TTL expiration."""
        if key not in self.ttl_times:
            return False
        
        return datetime.now() > self.ttl_times[key]
    
    def _evict_if_needed(self):
        """Evict items based on current strategy."""
        if len(self.cache) < self.max_size:
            return
        
        # Always evict expired TTL items first
        expired_keys = [k for k in self.ttl_times.keys() if self._should_evict_ttl(k)]
        for key in expired_keys:
            self._remove_key(key)
        
        if len(self.cache) < self.max_size:
            return
        
        # Apply strategy-specific eviction
        if self.strategy == CacheStrategy.LRU:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_key(oldest_key)
        elif self.strategy == CacheStrategy.LFU:
            least_frequent_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self._remove_key(least_frequent_key)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Use the best performing strategy based on recent history
            best_strategy = self._get_best_strategy()
            if best_strategy == CacheStrategy.LRU:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                self._remove_key(oldest_key)
            else:  # LFU
                least_frequent_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
                self._remove_key(least_frequent_key)
    
    def _remove_key(self, key: str):
        """Remove a key from all cache structures."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.ttl_times:
            del self.ttl_times[key]
    
    def _get_best_strategy(self) -> CacheStrategy:
        """Determine the best performing cache strategy."""
        if not self.strategy_performance:
            return CacheStrategy.LRU  # Default fallback
        
        return max(self.strategy_performance.keys(), key=lambda k: self.strategy_performance[k])
    
    def _update_strategy_performance(self, strategy: CacheStrategy, hit_rate: float):
        """Update performance metrics for cache strategies."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = hit_rate
        else:
            # Exponential moving average
            alpha = 0.1
            self.strategy_performance[strategy] = (
                alpha * hit_rate + (1 - alpha) * self.strategy_performance[strategy]
            )
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent tracking."""
        start_time = time.time()
        
        with self.lock:
            # Check TTL expiration
            if self._should_evict_ttl(key):
                self._remove_key(key)
                self.miss_count += 1
                return None
            
            if key in self.cache:
                # Cache hit
                self.hit_count += 1
                self.access_times[key] = datetime.now()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                # Record performance
                duration_ms = (time.time() - start_time) * 1000
                self.performance_history.append({
                    "operation": "cache_hit",
                    "duration_ms": duration_ms,
                    "timestamp": datetime.now()
                })
                
                return self.cache[key]
            else:
                # Cache miss
                self.miss_count += 1
                
                # Record performance
                duration_ms = (time.time() - start_time) * 1000
                self.performance_history.append({
                    "operation": "cache_miss",
                    "duration_ms": duration_ms,
                    "timestamp": datetime.now()
                })
                
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set item in cache with intelligent eviction."""
        with self.lock:
            self._evict_if_needed()
            
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            if ttl_seconds:
                self.ttl_times[key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        recent_performance = list(self.performance_history)[-10:]  # Last 10 operations
        avg_access_time = statistics.mean([p["duration_ms"] for p in recent_performance]) if recent_performance else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "strategy": self.strategy.value,
            "avg_access_time_ms": avg_access_time,
            "strategy_performance": {k.value: v for k, v in self.strategy_performance.items()}
        }

class AdaptiveWorkerPool:
    """Auto-scaling worker pool with performance-based scaling decisions."""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=config.min_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.min_workers)
        
        self.task_queue = asyncio.Queue()
        self.performance_metrics = deque(maxlen=100)
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        
        self.logger = logging.getLogger("adaptive_worker_pool")
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start performance monitoring and auto-scaling."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_performance())
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    async def _monitor_performance(self):
        """Monitor performance and trigger scaling decisions."""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                self.performance_metrics.append(current_metrics)
                
                # Make scaling decision
                await self._evaluate_scaling_decision(current_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        try:
            # Try to get actual system metrics
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
            except ImportError:
                # Fallback to simulated metrics
                cpu_percent = 50.0 + (len(self.performance_metrics) % 20)  # Simulate fluctuation
                memory_percent = 60.0 + (len(self.performance_metrics) % 15)
            
            # Calculate task queue utilization
            queue_size = self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
            queue_utilization = min(queue_size / (self.current_workers * 2), 1.0)  # Assume 2 tasks per worker is full
            
            # Calculate average response time
            recent_metrics = list(self.performance_metrics)[-10:]
            avg_response_time = statistics.mean([m.get("response_time_ms", 1000) for m in recent_metrics]) if recent_metrics else 1000
            
            return {
                "timestamp": datetime.now(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "queue_utilization": queue_utilization,
                "avg_response_time_ms": avg_response_time,
                "current_workers": self.current_workers,
                "queue_size": queue_size
            }
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return {
                "timestamp": datetime.now(),
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "queue_utilization": 0.5,
                "avg_response_time_ms": 1000.0,
                "current_workers": self.current_workers,
                "queue_size": 0
            }
    
    async def _evaluate_scaling_decision(self, metrics: Dict[str, Any]):
        """Evaluate whether to scale up or down based on metrics."""
        now = datetime.now()
        
        # Check if we should scale up
        should_scale_up = (
            (metrics["cpu_percent"] > self.config.scale_up_threshold * 100 or
             metrics["queue_utilization"] > self.config.scale_up_threshold or
             metrics["avg_response_time_ms"] > self.config.target_response_time_ms * 1.5) and
            self.current_workers < self.config.max_workers and
            (now - self.last_scale_up).total_seconds() > self.config.scale_up_cooldown
        )
        
        # Check if we should scale down
        should_scale_down = (
            metrics["cpu_percent"] < self.config.scale_down_threshold * 100 and
            metrics["queue_utilization"] < self.config.scale_down_threshold and
            metrics["avg_response_time_ms"] < self.config.target_response_time_ms * 0.5 and
            self.current_workers > self.config.min_workers and
            (now - self.last_scale_down).total_seconds() > self.config.scale_down_cooldown
        )
        
        if should_scale_up:
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up the worker pool."""
        new_worker_count = min(self.current_workers + 1, self.config.max_workers)
        
        if new_worker_count > self.current_workers:
            self.logger.info(f"Scaling up from {self.current_workers} to {new_worker_count} workers")
            
            # Recreate thread pool with more workers
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = ThreadPoolExecutor(max_workers=new_worker_count)
            
            self.current_workers = new_worker_count
            self.last_scale_up = datetime.now()
    
    async def _scale_down(self):
        """Scale down the worker pool."""
        new_worker_count = max(self.current_workers - 1, self.config.min_workers)
        
        if new_worker_count < self.current_workers:
            self.logger.info(f"Scaling down from {self.current_workers} to {new_worker_count} workers")
            
            # Recreate thread pool with fewer workers
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = ThreadPoolExecutor(max_workers=new_worker_count)
            
            self.current_workers = new_worker_count
            self.last_scale_down = datetime.now()
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the worker pool."""
        start_time = time.time()
        
        try:
            # Submit to thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
            
            # Record performance
            duration_ms = (time.time() - start_time) * 1000
            self.performance_metrics.append({
                "response_time_ms": duration_ms,
                "timestamp": datetime.now(),
                "success": True
            })
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_metrics.append({
                "response_time_ms": duration_ms,
                "timestamp": datetime.now(),
                "success": False,
                "error": str(e)
            })
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        recent_metrics = list(self.performance_metrics)[-20:]
        
        success_rate = sum(1 for m in recent_metrics if m.get("success", False)) / len(recent_metrics) if recent_metrics else 1.0
        avg_response_time = statistics.mean([m["response_time_ms"] for m in recent_metrics]) if recent_metrics else 0.0
        
        return {
            "current_workers": self.current_workers,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "queue_size": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "total_tasks_processed": len(self.performance_metrics)
        }

class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self, level: PerformanceLevel = PerformanceLevel.ADAPTIVE):
        self.level = level
        self.cache = IntelligentCache(max_size=10000, strategy=CacheStrategy.ADAPTIVE)
        self.worker_pool = AdaptiveWorkerPool(AutoScalingConfig())
        self.optimization_history = deque(maxlen=1000)
        self.active_optimizations: Dict[str, Any] = {}
        
        self.logger = logging.getLogger("performance_optimizer")
    
    async def initialize(self):
        """Initialize the performance optimization system."""
        await self.worker_pool.start_monitoring()
        self.logger.info(f"Performance optimizer initialized with level: {self.level.value}")
    
    async def shutdown(self):
        """Shutdown the performance optimization system."""
        await self.worker_pool.stop_monitoring()
        self.worker_pool.thread_pool.shutdown(wait=True)
        self.worker_pool.process_pool.shutdown(wait=True)
    
    async def optimize_function_call(self, func: Callable, *args, cache_key: Optional[str] = None, **kwargs) -> Any:
        """Optimize a function call with caching and worker pooling."""
        start_time = time.time()
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for {func.__name__}")
            return cached_result
        
        # Execute function with worker pool
        try:
            result = await self.worker_pool.submit_task(func, *args, **kwargs)
            
            # Cache successful results
            self.cache.set(cache_key, result, ttl_seconds=3600)  # Cache for 1 hour
            
            # Record optimization metrics
            duration_ms = (time.time() - start_time) * 1000
            self.optimization_history.append({
                "function": func.__name__,
                "duration_ms": duration_ms,
                "cache_hit": False,
                "timestamp": datetime.now()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing {func.__name__}: {e}")
            raise
    
    def enable_optimization(self, optimization_type: str, config: Dict[str, Any]):
        """Enable a specific optimization type."""
        self.active_optimizations[optimization_type] = {
            "config": config,
            "enabled_at": datetime.now()
        }
        self.logger.info(f"Enabled optimization: {optimization_type}")
    
    def disable_optimization(self, optimization_type: str):
        """Disable a specific optimization type."""
        if optimization_type in self.active_optimizations:
            del self.active_optimizations[optimization_type]
            self.logger.info(f"Disabled optimization: {optimization_type}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        recent_optimizations = list(self.optimization_history)[-50:]
        
        if recent_optimizations:
            avg_duration = statistics.mean([o["duration_ms"] for o in recent_optimizations])
            cache_hit_rate = sum(1 for o in recent_optimizations if o["cache_hit"]) / len(recent_optimizations)
            function_counts = {}
            for opt in recent_optimizations:
                func_name = opt["function"]
                function_counts[func_name] = function_counts.get(func_name, 0) + 1
        else:
            avg_duration = 0.0
            cache_hit_rate = 0.0
            function_counts = {}
        
        return {
            "optimization_level": self.level.value,
            "total_optimizations": len(self.optimization_history),
            "avg_duration_ms": avg_duration,
            "cache_hit_rate": cache_hit_rate,
            "active_optimizations": list(self.active_optimizations.keys()),
            "function_usage": function_counts,
            "cache_stats": self.cache.get_stats(),
            "worker_pool_stats": self.worker_pool.get_stats(),
            "timestamp": datetime.now().isoformat()
        }

async def benchmark_performance_system():
    """Benchmark the performance optimization system."""
    print("🚀 Generation 3: Scalable Performance System Benchmark")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(PerformanceLevel.ADAPTIVE)
    await optimizer.initialize()
    
    print("🔧 Testing Intelligent Cache...")
    print("-" * 35)
    
    # Test cache performance
    cache = IntelligentCache(max_size=100, strategy=CacheStrategy.ADAPTIVE)
    
    # Simulate cache operations
    test_operations = [
        ("set", "key1", "value1"),
        ("set", "key2", "value2"),
        ("get", "key1"),
        ("get", "key2"),
        ("get", "key3"),  # Miss
        ("set", "key3", "value3"),
        ("get", "key1"),  # Hit
        ("get", "key2"),  # Hit
        ("get", "key3"),  # Hit
    ]
    
    for operation, key, *args in test_operations:
        if operation == "set":
            cache.set(key, args[0])
        elif operation == "get":
            result = cache.get(key)
            status = "HIT" if result is not None else "MISS"
            print(f"  Cache {operation.upper()} {key}: {status}")
    
    cache_stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Avg Access Time: {cache_stats['avg_access_time_ms']:.2f}ms")
    
    print("\n⚙️  Testing Auto-Scaling Worker Pool...")
    print("-" * 40)
    
    # Test worker pool performance
    def cpu_intensive_task(n: int) -> int:
        """Simulate CPU-intensive task."""
        result = 0
        for i in range(n * 1000):
            result += i * i
        return result
    
    # Submit tasks to test scaling
    tasks = []
    for i in range(10):
        task = optimizer.optimize_function_call(cpu_intensive_task, 100 + i * 50)
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    execution_time = time.time() - start_time
    
    print(f"  Executed {len(tasks)} tasks in {execution_time:.2f}s")
    print(f"  Average task result: {statistics.mean(results):,.0f}")
    
    # Get worker pool stats
    worker_stats = optimizer.worker_pool.get_stats()
    print(f"  Workers: {worker_stats['current_workers']}")
    print(f"  Success Rate: {worker_stats['success_rate']:.1%}")
    print(f"  Avg Response Time: {worker_stats['avg_response_time_ms']:.1f}ms")
    
    print("\n📈 Testing Performance Optimization...")
    print("-" * 42)
    
    # Test function optimization with caching
    def expensive_computation(x: int, y: int) -> int:
        """Simulate expensive computation."""
        time.sleep(0.1)  # Simulate work
        return x * y + (x ** 2) + (y ** 2)
    
    # Test with and without optimization
    print("  Testing without optimization...")
    start_time = time.time()
    result1 = expensive_computation(10, 20)
    result2 = expensive_computation(10, 20)  # Same computation
    no_opt_time = time.time() - start_time
    
    print("  Testing with optimization...")
    start_time = time.time()
    opt_result1 = await optimizer.optimize_function_call(expensive_computation, 10, 20)
    opt_result2 = await optimizer.optimize_function_call(expensive_computation, 10, 20)  # Should hit cache
    opt_time = time.time() - start_time
    
    print(f"  Without optimization: {no_opt_time:.3f}s")
    print(f"  With optimization: {opt_time:.3f}s")
    print(f"  Speedup: {no_opt_time / opt_time:.1f}x")
    print(f"  Results match: {'✅' if result1 == opt_result1 else '❌'}")
    
    print("\n📋 Performance Report:")
    print("-" * 25)
    
    # Generate comprehensive performance report
    report = optimizer.get_performance_report()
    print(f"  Optimization Level: {report['optimization_level']}")
    print(f"  Total Optimizations: {report['total_optimizations']}")
    print(f"  Average Duration: {report['avg_duration_ms']:.1f}ms")
    print(f"  Cache Hit Rate: {report['cache_hit_rate']:.1%}")
    print(f"  Active Optimizations: {len(report['active_optimizations'])}")
    
    # Cache statistics
    cache_stats = report['cache_stats']
    print(f"\n🗄️  Cache Performance:")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Strategy: {cache_stats['strategy']}")
    
    # Worker pool statistics
    worker_stats = report['worker_pool_stats']
    print(f"\n👷 Worker Pool Performance:")
    print(f"  Current Workers: {worker_stats['current_workers']}")
    print(f"  Success Rate: {worker_stats['success_rate']:.1%}")
    print(f"  Tasks Processed: {worker_stats['total_tasks_processed']}")
    
    # Export performance report
    report_path = "/tmp/performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📊 Performance report exported to: {report_path}")
    
    # Cleanup
    await optimizer.shutdown()
    
    print("\n✅ Generation 3 scalable performance system benchmark complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(benchmark_performance_system())
    sys.exit(0 if success else 1)