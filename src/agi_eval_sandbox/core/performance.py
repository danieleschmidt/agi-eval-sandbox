"""
Performance optimization and caching module.

This module provides advanced performance optimizations including:
- Intelligent caching with TTL and LRU eviction
- Connection pooling for databases and external APIs
- Request batching and deduplication
- Memory-efficient processing patterns
- Async processing optimizations
"""

import asyncio
import time
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from contextlib import asynccontextmanager
import threading
from datetime import datetime, timedelta

import redis
from sqlalchemy.pool import QueuePool, StaticPool

from .logging_config import get_logger, performance_logger
from .exceptions import ConfigurationError, ResourceError

logger = get_logger("performance")


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour default
    enable_compression: bool = True
    enable_serialization: bool = True
    eviction_policy: str = "lru"  # lru, lfu, fifo
    redis_url: Optional[str] = None
    enable_distributed: bool = False


class IntelligentCache:
    """High-performance cache with multiple eviction policies and compression."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict = OrderedDict()
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self.stats = CacheStats()
        
        # Redis for distributed caching
        self._redis_client = None
        if self.config.enable_distributed and self.config.redis_url:
            try:
                self._redis_client = redis.from_url(self.config.redis_url)
                self._redis_client.ping()  # Test connection
                logger.info("Distributed caching enabled with Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.config.enable_distributed = False
    
    def _generate_key(self, key: Union[str, tuple]) -> str:
        """Generate cache key from input."""
        if isinstance(key, str):
            return key
        elif isinstance(key, tuple):
            return hashlib.md5(str(key).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if not self.config.enable_serialization:
            return value
        
        serialized = pickle.dumps(value)
        
        if self.config.enable_compression and len(serialized) > 1024:
            import gzip
            serialized = gzip.compress(serialized)
        
        return serialized
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not self.config.enable_serialization:
            return data
        
        try:
            # Try to decompress first
            if self.config.enable_compression:
                import gzip
                try:
                    data = gzip.decompress(data)
                except gzip.BadGzipFile:
                    pass  # Not compressed
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize cached value: {e}")
            return None
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._access_times:
            return True
        
        age = datetime.now() - self._access_times[key]
        return age.total_seconds() > self.config.ttl_seconds
    
    def _evict_if_needed(self):
        """Evict items if cache is full."""
        while len(self._cache) >= self.config.max_size:
            if self.config.eviction_policy == "lru":
                # Remove least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._access_times.pop(oldest_key, None)
                self._access_counts.pop(oldest_key, None)
            elif self.config.eviction_policy == "lfu":
                # Remove least frequently used
                lfu_key = min(self._access_counts.keys(), 
                             key=lambda k: self._access_counts[k])
                del self._cache[lfu_key]
                self._access_times.pop(lfu_key, None)
                self._access_counts.pop(lfu_key, None)
            elif self.config.eviction_policy == "fifo":
                # Remove first in
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._access_times.pop(oldest_key, None)
                self._access_counts.pop(oldest_key, None)
            
            self.stats.evictions += 1
    
    async def get(self, key: Union[str, tuple]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key)
        self.stats.total_requests += 1
        
        with self._lock:
            # Check local cache first
            if cache_key in self._cache and not self._is_expired(cache_key):
                # Move to end for LRU
                value = self._cache.pop(cache_key)
                self._cache[cache_key] = value
                self._access_times[cache_key] = datetime.now()
                self._access_counts[cache_key] += 1
                self.stats.hits += 1
                
                return self._deserialize_value(value)
        
        # Try distributed cache if enabled
        if self._redis_client:
            try:
                redis_value = await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.get, f"cache:{cache_key}"
                )
                if redis_value:
                    # Store in local cache
                    with self._lock:
                        self._evict_if_needed()
                        self._cache[cache_key] = redis_value
                        self._access_times[cache_key] = datetime.now()
                        self._access_counts[cache_key] += 1
                    
                    self.stats.hits += 1
                    return self._deserialize_value(redis_value)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: Union[str, tuple], value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        cache_key = self._generate_key(key)
        serialized_value = self._serialize_value(value)
        ttl = ttl or self.config.ttl_seconds
        
        with self._lock:
            self._evict_if_needed()
            self._cache[cache_key] = serialized_value
            self._access_times[cache_key] = datetime.now()
            self._access_counts[cache_key] = 1
            self.stats.total_memory_usage += len(serialized_value)
        
        # Also store in distributed cache
        if self._redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._redis_client.setex(
                        f"cache:{cache_key}", 
                        ttl, 
                        serialized_value
                    )
                )
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
    
    async def delete(self, key: Union[str, tuple]):
        """Delete value from cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._access_times.pop(cache_key, None)
                self._access_counts.pop(cache_key, None)
        
        # Also delete from distributed cache
        if self._redis_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._redis_client.delete, f"cache:{cache_key}"
                )
            except Exception as e:
                logger.warning(f"Redis cache delete failed: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self.stats = CacheStats()
        
        # Also clear distributed cache
        if self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_rate": self.stats.hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "cache_size": len(self._cache),
            "max_size": self.config.max_size,
            "memory_usage_bytes": self.stats.total_memory_usage
        }


class ConnectionPoolManager:
    """Manages connection pools for various external services."""
    
    def __init__(self):
        self._pools: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def get_database_pool(self, database_url: str, pool_size: int = 10) -> QueuePool:
        """Get or create database connection pool."""
        with self._lock:
            pool_key = f"db_{hashlib.md5(database_url.encode()).hexdigest()}"
            
            if pool_key not in self._pools:
                from sqlalchemy import create_engine
                
                engine = create_engine(
                    database_url,
                    poolclass=QueuePool,
                    pool_size=pool_size,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
                
                self._pools[pool_key] = engine
                logger.info(f"Created database pool with size {pool_size}")
            
            return self._pools[pool_key]
    
    async def get_http_session(self, max_connections: int = 100):
        """Get HTTP session with connection pooling."""
        import aiohttp
        
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )


class BatchProcessor:
    """Batch and deduplicate requests for better performance."""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._pending_requests: List[Tuple[str, Any, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
    
    async def add_request(self, request_id: str, request_data: Any) -> Any:
        """Add request to batch and wait for result."""
        future = asyncio.Future()
        
        async with self._lock:
            self._pending_requests.append((request_id, request_data, future))
            
            # Start batch processing if needed
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated batch requests."""
        await asyncio.sleep(self.batch_timeout)
        
        async with self._lock:
            if not self._pending_requests:
                return
            
            # Take current batch
            current_batch = self._pending_requests[:]
            self._pending_requests.clear()
        
        # Group by similar requests for deduplication
        grouped_requests = self._group_similar_requests(current_batch)
        
        # Process each group
        for group in grouped_requests:
            try:
                results = await self._process_request_group(group)
                
                # Set results for all futures in group
                for i, (_, _, future) in enumerate(group):
                    if i < len(results):
                        future.set_result(results[i])
                    else:
                        future.set_exception(Exception("Insufficient results"))
            
            except Exception as e:
                # Set exception for all futures in group
                for _, _, future in group:
                    if not future.done():
                        future.set_exception(e)
    
    def _group_similar_requests(
        self, 
        requests: List[Tuple[str, Any, asyncio.Future]]
    ) -> List[List[Tuple[str, Any, asyncio.Future]]]:
        """Group similar requests together."""
        groups = []
        
        # Simple grouping by request type
        type_groups = defaultdict(list)
        for request in requests:
            request_type = type(request[1]).__name__
            type_groups[request_type].append(request)
        
        # Split large groups into batch_size chunks
        for group in type_groups.values():
            for i in range(0, len(group), self.batch_size):
                groups.append(group[i:i + self.batch_size])
        
        return groups
    
    async def _process_request_group(
        self, 
        group: List[Tuple[str, Any, asyncio.Future]]
    ) -> List[Any]:
        """Process a group of similar requests."""
        # Extract request data
        request_data = [req[1] for req in group]
        
        # This would be implemented by subclasses for specific processing
        # For now, just return the input data
        return request_data


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, cache_config: CacheConfig = None):
        self.cache = IntelligentCache(cache_config or CacheConfig())
        self.connection_manager = ConnectionPoolManager()
        self.batch_processor = BatchProcessor()
        
        # Performance monitoring
        self._metrics = {
            "request_count": 0,
            "average_response_time": 0.0,
            "error_count": 0,
            "cache_hit_rate": 0.0
        }
    
    @asynccontextmanager
    async def performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        
        try:
            yield
            # Success
            duration = time.time() - start_time
            self._metrics["request_count"] += 1
            self._update_average_response_time(duration)
            
            performance_logger.log_performance_metric(
                metric_name=f"{operation_name}_duration",
                value=duration,
                labels={"operation": operation_name, "status": "success"}
            )
            
        except Exception as e:
            # Error
            duration = time.time() - start_time
            self._metrics["error_count"] += 1
            
            performance_logger.log_performance_metric(
                metric_name=f"{operation_name}_duration",
                value=duration,
                labels={"operation": operation_name, "status": "error"}
            )
            
            logger.error(f"Performance context error in {operation_name}: {e}")
            raise
    
    def _update_average_response_time(self, duration: float):
        """Update average response time metric."""
        current_avg = self._metrics["average_response_time"]
        count = self._metrics["request_count"]
        
        # Exponential moving average
        alpha = 0.1
        self._metrics["average_response_time"] = (
            alpha * duration + (1 - alpha) * current_avg
        )
    
    async def cached_operation(
        self, 
        key: Union[str, tuple], 
        operation: Callable,
        ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with caching."""
        # Try to get from cache first
        cached_result = await self.cache.get(key)
        if cached_result is not None:
            return cached_result
        
        # Execute operation
        async with self.performance_context("cached_operation"):
            result = await operation(*args, **kwargs)
            await self.cache.set(key, result, ttl)
            return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_stats = self.cache.get_stats()
        
        return {
            **self._metrics,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["cache_size"],
            "cache_memory_usage": cache_stats["memory_usage_bytes"]
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


def cached(key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            return await performance_optimizer.cached_operation(
                cache_key, func, ttl, *args, **kwargs
            )
        
        return wrapper
    return decorator


@asynccontextmanager
async def performance_monitoring(operation_name: str):
    """Context manager for performance monitoring."""
    async with performance_optimizer.performance_context(operation_name):
        yield