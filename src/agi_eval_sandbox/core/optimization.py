"""Advanced performance optimization and scaling features."""

import asyncio
import time
import math
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import threading
import multiprocessing as mp
from datetime import datetime, timedelta

from .logging_config import get_logger, performance_logger
from .cache import cache_manager
from .exceptions import ResourceError, ConfigurationError

logger = get_logger("optimization")


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    avg_response_time: float = 0.0
    throughput_qps: float = 0.0
    success_rate: float = 100.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_requests: int = 0
    queue_depth: int = 0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_adaptive_batching: bool = True
    enable_response_streaming: bool = True
    enable_connection_pooling: bool = True
    enable_smart_caching: bool = True
    enable_load_balancing: bool = True
    enable_auto_scaling: bool = True
    
    # Adaptive batching
    min_batch_size: int = 1
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    
    # Connection pooling
    max_connections_per_host: int = 20
    connection_timeout_seconds: int = 30
    
    # Auto-scaling
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_workers: int = 2
    max_workers: int = 16


class AdaptiveBatchProcessor:
    """Adaptive batching system that optimizes batch sizes based on performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.pending_requests: deque = deque()
        self.processing_queue = asyncio.Queue()
        self.performance_history: deque = deque(maxlen=100)
        self.lock = asyncio.Lock()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.last_optimization = time.time()
        self.optimization_interval = 30.0  # seconds
    
    async def add_request(self, request_id: str, request_data: Any, priority: int = 0) -> Any:
        """Add request to adaptive batch processing queue."""
        future = asyncio.Future()
        request = {
            'id': request_id,
            'data': request_data,
            'future': future,
            'priority': priority,
            'timestamp': time.time()
        }
        
        async with self.lock:
            self.pending_requests.append(request)
            await self.processing_queue.put(request)
        
        # Trigger batch processing if queue is full or timeout reached
        await self._maybe_process_batch()
        
        return await future
    
    async def _maybe_process_batch(self):
        """Check if we should process a batch now."""
        current_time = time.time()
        
        async with self.lock:
            if not self.pending_requests:
                return
            
            # Check batch size threshold
            if len(self.pending_requests) >= self.current_batch_size:
                await self._process_current_batch()
                return
            
            # Check timeout threshold
            oldest_request = self.pending_requests[0]
            if (current_time - oldest_request['timestamp']) * 1000 > self.config.batch_timeout_ms:
                await self._process_current_batch()
                return
    
    async def _process_current_batch(self):
        """Process the current batch of requests."""
        if not self.pending_requests:
            return
        
        # Extract batch
        batch_size = min(len(self.pending_requests), self.current_batch_size)
        batch = []
        for _ in range(batch_size):
            batch.append(self.pending_requests.popleft())
        
        # Sort by priority
        batch.sort(key=lambda x: x['priority'], reverse=True)
        
        # Process batch
        start_time = time.time()
        try:
            results = await self._execute_batch([req['data'] for req in batch])
            
            # Set results
            for request, result in zip(batch, results):
                request['future'].set_result(result)
            
            # Record performance
            duration = time.time() - start_time
            self._record_batch_performance(batch_size, duration, True)
            
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch:
                request['future'].set_exception(e)
            
            # Record failure
            duration = time.time() - start_time
            self._record_batch_performance(batch_size, duration, False)
        
        # Optimize batch size if needed
        await self._optimize_batch_size()
    
    async def _execute_batch(self, batch_data: List[Any]) -> List[Any]:
        """Execute batch processing logic. Override in subclasses."""
        # Default: return input data (placeholder)
        await asyncio.sleep(0.001)  # Simulate processing
        return batch_data
    
    def _record_batch_performance(self, batch_size: int, duration: float, success: bool):
        """Record performance metrics for batch processing."""
        throughput = batch_size / duration if duration > 0 else 0
        
        perf_record = {
            'batch_size': batch_size,
            'duration': duration,
            'throughput': throughput,
            'success': success,
            'timestamp': time.time()
        }
        
        self.performance_history.append(perf_record)
        
        # Update current metrics
        if self.performance_history:
            recent_records = list(self.performance_history)[-10:]  # Last 10 batches
            self.metrics.avg_response_time = statistics.mean(r['duration'] for r in recent_records)
            self.metrics.throughput_qps = statistics.mean(r['throughput'] for r in recent_records)
            self.metrics.success_rate = (
                sum(1 for r in recent_records if r['success']) / len(recent_records) * 100
            )
    
    async def _optimize_batch_size(self):
        """Optimize batch size based on performance history."""
        current_time = time.time()
        
        # Only optimize periodically
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        if len(self.performance_history) < 5:
            return  # Need more data
        
        # Analyze recent performance trends
        recent_records = list(self.performance_history)[-20:]
        
        # Calculate performance score (throughput / latency)
        def performance_score(records):
            if not records:
                return 0
            avg_throughput = statistics.mean(r['throughput'] for r in records)
            avg_latency = statistics.mean(r['duration'] for r in records)
            return avg_throughput / (avg_latency + 0.001)  # Add small epsilon
        
        current_score = performance_score(recent_records)
        
        # Try different batch sizes in simulation
        best_batch_size = self.current_batch_size
        best_score = current_score
        
        for test_size in [
            max(self.config.min_batch_size, self.current_batch_size - 2),
            self.current_batch_size,
            min(self.config.max_batch_size, self.current_batch_size + 2)
        ]:
            # Simulate performance with this batch size
            simulated_score = self._simulate_batch_performance(test_size)
            if simulated_score > best_score:
                best_score = simulated_score
                best_batch_size = test_size
        
        # Update batch size if improvement is significant
        if best_batch_size != self.current_batch_size:
            old_size = self.current_batch_size
            self.current_batch_size = best_batch_size
            self.last_optimization = current_time
            
            logger.info(
                f"Optimized batch size: {old_size} -> {self.current_batch_size} "
                f"(score improvement: {(best_score - current_score):.3f})"
            )
    
    def _simulate_batch_performance(self, batch_size: int) -> float:
        """Simulate performance for a given batch size."""
        # Simple heuristic based on observed patterns
        # Larger batches generally have better throughput but higher latency
        
        # Base throughput increases with batch size (up to a point)
        base_throughput = min(batch_size * 0.8, batch_size * (1 - batch_size / 50))
        
        # Latency increases with batch size
        base_latency = 0.1 + (batch_size * 0.01)
        
        # Apply variance based on historical data
        if self.performance_history:
            recent_avg_latency = statistics.mean(
                r['duration'] for r in list(self.performance_history)[-10:]
            )
            base_latency = (base_latency + recent_avg_latency) / 2
        
        return base_throughput / (base_latency + 0.001)


class ConnectionPoolOptimizer:
    """Optimizes connection pooling for external services."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.pools: Dict[str, Any] = {}
        self.connection_metrics: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    async def get_optimized_session(self, service_name: str, base_url: str):
        """Get optimized HTTP session for a service."""
        import aiohttp
        
        pool_key = f"{service_name}_{hash(base_url)}"
        
        with self.lock:
            if pool_key not in self.pools:
                # Calculate optimal connection limits based on service performance
                max_connections = self._calculate_optimal_connections(service_name)
                
                connector = aiohttp.TCPConnector(
                    limit=max_connections,
                    limit_per_host=min(max_connections, self.config.max_connections_per_host),
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=self.config.connection_timeout_seconds,
                    connect=10
                )
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
                
                self.pools[pool_key] = session
                logger.info(f"Created optimized connection pool for {service_name}")
        
        return self.pools[pool_key]
    
    def _calculate_optimal_connections(self, service_name: str) -> int:
        """Calculate optimal number of connections for a service."""
        # Base on historical performance data
        if service_name in self.connection_metrics:
            recent_latencies = self.connection_metrics[service_name][-50:]
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                
                # More connections for higher latency services
                if avg_latency > 1.0:  # High latency
                    return min(self.config.max_connections_per_host, 15)
                elif avg_latency > 0.5:  # Medium latency
                    return min(self.config.max_connections_per_host, 10)
                else:  # Low latency
                    return min(self.config.max_connections_per_host, 5)
        
        # Default for new services
        return 8
    
    def record_connection_performance(self, service_name: str, latency: float):
        """Record connection performance for optimization."""
        with self.lock:
            self.connection_metrics[service_name].append(latency)
            # Keep only recent measurements
            if len(self.connection_metrics[service_name]) > 100:
                self.connection_metrics[service_name] = self.connection_metrics[service_name][-50:]


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.metrics_history: deque = deque(maxlen=60)  # 1 minute of data
        self.last_scale_action = time.time()
        self.scale_cooldown = 30.0  # seconds
        
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize worker pools."""
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="agi_eval_worker"
        )
        
        # Process pool for CPU-intensive tasks
        cpu_count = mp.cpu_count()
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(cpu_count, 4)
        )
        
        logger.info(f"Initialized worker pools: {self.current_workers} threads, {min(cpu_count, 4)} processes")
    
    async def submit_task(self, func: Callable, *args, use_process_pool: bool = False, **kwargs) -> Any:
        """Submit task to appropriate worker pool."""
        loop = asyncio.get_event_loop()
        
        if use_process_pool:
            future = self.process_pool.submit(func, *args, **kwargs)
        else:
            future = self.worker_pool.submit(func, *args, **kwargs)
        
        return await loop.run_in_executor(None, future.result)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for scaling decisions."""
        self.metrics_history.append({
            'timestamp': time.time(),
            'cpu_usage': metrics.cpu_usage_percent,
            'memory_usage': metrics.memory_usage_mb,
            'response_time': metrics.avg_response_time,
            'throughput': metrics.throughput_qps,
            'queue_depth': metrics.queue_depth,
            'concurrent_requests': metrics.concurrent_requests
        })
        
        # Check if scaling is needed
        asyncio.create_task(self._check_scaling_needed())
    
    async def _check_scaling_needed(self):
        """Check if scaling up or down is needed."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scale_action < self.scale_cooldown:
            return
        
        if len(self.metrics_history) < 5:
            return  # Need more data
        
        # Analyze recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = statistics.mean(m['cpu_usage'] for m in recent_metrics)
        avg_memory = statistics.mean(m['memory_usage'] for m in recent_metrics)
        avg_response_time = statistics.mean(m['response_time'] for m in recent_metrics)
        avg_queue_depth = statistics.mean(m['queue_depth'] for m in recent_metrics)
        
        # Scale up conditions
        scale_up = (
            avg_cpu > self.config.target_cpu_percent * self.config.scale_up_threshold or
            avg_queue_depth > 10 or
            avg_response_time > 2.0
        )
        
        # Scale down conditions
        scale_down = (
            avg_cpu < self.config.target_cpu_percent * self.config.scale_down_threshold and
            avg_queue_depth < 2 and
            avg_response_time < 0.5
        )
        
        if scale_up and self.current_workers < self.config.max_workers:
            await self._scale_up()
        elif scale_down and self.current_workers > self.config.min_workers:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up worker pool."""
        new_workers = min(self.current_workers + 2, self.config.max_workers)
        
        # Shutdown old pool and create new one
        old_pool = self.worker_pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=new_workers,
            thread_name_prefix="agi_eval_worker"
        )
        
        # Shutdown old pool gracefully
        if old_pool:
            old_pool.shutdown(wait=False)
        
        old_workers = self.current_workers
        self.current_workers = new_workers
        self.last_scale_action = time.time()
        
        logger.info(f"Scaled up workers: {old_workers} -> {new_workers}")
        
        performance_logger.log_resource_usage(
            memory_mb=0,  # Will be updated by monitoring
            cpu_percent=0,  # Will be updated by monitoring
            disk_usage_mb=0
        )
    
    async def _scale_down(self):
        """Scale down worker pool."""
        new_workers = max(self.current_workers - 1, self.config.min_workers)
        
        # Shutdown old pool and create new one
        old_pool = self.worker_pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=new_workers,
            thread_name_prefix="agi_eval_worker"
        )
        
        # Shutdown old pool gracefully
        if old_pool:
            old_pool.shutdown(wait=False)
        
        old_workers = self.current_workers
        self.current_workers = new_workers
        self.last_scale_action = time.time()
        
        logger.info(f"Scaled down workers: {old_workers} -> {new_workers}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'current_workers': self.current_workers,
            'thread_pool_size': self.current_workers,
            'process_pool_size': self.process_pool._max_workers,
            'metrics_history_size': len(self.metrics_history),
            'last_scale_action': datetime.fromtimestamp(self.last_scale_action).isoformat()
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Initialize optimization components
        self.batch_processor = AdaptiveBatchProcessor(self.config)
        self.connection_optimizer = ConnectionPoolOptimizer(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.optimization_enabled = True
        
        logger.info("Performance optimizer initialized with adaptive features")
    
    async def optimize_evaluation_pipeline(
        self,
        evaluation_func: Callable,
        requests: List[Any],
        **kwargs
    ) -> List[Any]:
        """Optimize evaluation pipeline with adaptive batching and scaling."""
        if not self.optimization_enabled:
            # Fallback to sequential processing
            return [await evaluation_func(req, **kwargs) for req in requests]
        
        start_time = time.time()
        
        try:
            # Use adaptive batching for optimal throughput
            tasks = []
            for i, request in enumerate(requests):
                task = self.batch_processor.add_request(
                    request_id=f"eval_{i}",
                    request_data=(evaluation_func, request, kwargs),
                    priority=getattr(request, 'priority', 0)
                )
                tasks.append(task)
            
            # Wait for all results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update metrics
            duration = time.time() - start_time
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            self.metrics.avg_response_time = duration
            self.metrics.throughput_qps = len(requests) / duration if duration > 0 else 0
            self.metrics.success_rate = (success_count / len(requests)) * 100 if requests else 100
            
            # Record metrics for auto-scaling
            self.auto_scaler.record_metrics(self.metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            # Fallback to sequential processing
            return [await evaluation_func(req, **kwargs) for req in requests]
    
    async def get_optimized_connection(self, service_name: str, base_url: str):
        """Get optimized connection for external service."""
        return await self.connection_optimizer.get_optimized_session(service_name, base_url)
    
    async def submit_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit CPU-intensive task to process pool."""
        return await self.auto_scaler.submit_task(func, *args, use_process_pool=True, **kwargs)
    
    async def submit_io_intensive_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit I/O-intensive task to thread pool."""
        return await self.auto_scaler.submit_task(func, *args, use_process_pool=False, **kwargs)
    
    def enable_optimization(self):
        """Enable performance optimizations."""
        self.optimization_enabled = True
        logger.info("Performance optimizations enabled")
    
    def disable_optimization(self):
        """Disable performance optimizations for debugging."""
        self.optimization_enabled = False
        logger.info("Performance optimizations disabled")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            'optimization_enabled': self.optimization_enabled,
            'current_metrics': {
                'avg_response_time': self.metrics.avg_response_time,
                'throughput_qps': self.metrics.throughput_qps,
                'success_rate': self.metrics.success_rate,
                'cache_hit_rate': self.metrics.cache_hit_rate
            },
            'batch_processor': {
                'current_batch_size': self.batch_processor.current_batch_size,
                'performance_history_size': len(self.batch_processor.performance_history)
            },
            'auto_scaler': self.auto_scaler.get_pool_stats(),
            'config': {
                'min_batch_size': self.config.min_batch_size,
                'max_batch_size': self.config.max_batch_size,
                'batch_timeout_ms': self.config.batch_timeout_ms,
                'target_cpu_percent': self.config.target_cpu_percent,
                'max_workers': self.config.max_workers
            }
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()