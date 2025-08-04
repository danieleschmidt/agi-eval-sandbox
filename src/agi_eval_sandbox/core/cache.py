"""Advanced caching system with multiple backends and intelligent strategies."""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading

from .exceptions import ResourceError
from .logging_config import get_logger, performance_logger

logger = get_logger("cache")

try:
    import pickle
except ImportError:
    pickle = None


@dataclass
class CacheItem:
    """Individual cache item with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of item in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def access(self) -> None:
        """Update access metadata."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set item in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all items from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.items: Dict[str, CacheItem] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[CacheItem]:
        """Get item from memory cache."""
        with self.lock:
            if key not in self.items:
                self.misses += 1
                return None
            
            item = self.items[key]
            if item.is_expired:
                del self.items[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.misses += 1
                return None
            
            # Update access tracking
            item.access()
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            return item
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set item in memory cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(json.dumps(value, default=str).encode('utf-8'))
            except:
                size_bytes = 1024  # Fallback estimate
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            # Check memory limits
            current_memory = sum(item.size_bytes for item in self.items.values())
            if current_memory + size_bytes > self.max_memory_bytes:
                await self._evict_to_fit(size_bytes)
            
            # Store item
            if key in self.items and key in self.access_order:
                self.access_order.remove(key)
            
            self.items[key] = item
            self.access_order.append(key)
            
            # Enforce size limits
            while len(self.items) > self.max_size and self.access_order:
                await self._evict_lru()
    
    async def delete(self, key: str) -> bool:
        """Delete item from memory cache."""
        with self.lock:
            if key in self.items:
                del self.items[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from memory cache."""
        with self.lock:
            self.items.clear()
            self.access_order.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        with self.lock:
            return key in self.items and not self.items[key].is_expired
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.items:
                del self.items[lru_key]
                self.evictions += 1
    
    async def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict items to fit new item."""
        current_memory = sum(item.size_bytes for item in self.items.values())
        
        while (current_memory + needed_bytes > self.max_memory_bytes and 
               self.access_order):
            lru_key = self.access_order[0]
            if lru_key in self.items:
                current_memory -= self.items[lru_key].size_bytes
            await self._evict_lru()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        with self.lock:
            total_memory = sum(item.size_bytes for item in self.items.values())
            hit_rate = (self.hits / (self.hits + self.misses)) * 100 if (self.hits + self.misses) > 0 else 0
            
            return {
                "backend": "memory",
                "items": len(self.items),
                "max_size": self.max_size,
                "memory_bytes": total_memory,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_usage_percent": (total_memory / self.max_memory_bytes) * 100,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate_percent": hit_rate
            }


class FileCache(CacheBackend):
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: str = "./cache", max_files: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[CacheItem]:
        """Get item from file cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            self.misses += 1
            return None
        
        try:
            with open(file_path, 'rb') as f:
                if pickle:
                    data = pickle.load(f)
                else:
                    data = json.load(f)
                
                item = CacheItem(**data)
                
                if item.is_expired:
                    file_path.unlink(missing_ok=True)
                    self.misses += 1
                    return None
                
                item.access()
                self.hits += 1
                return item
                
        except Exception as e:
            logger.warning(f"Failed to load cache file {file_path}: {e}")
            file_path.unlink(missing_ok=True)
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set item in file cache."""
        file_path = self._get_file_path(key)
        
        # Create cache item
        item = CacheItem(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds
        )
        
        # Enforce file limits
        await self._cleanup_old_files()
        
        try:
            with open(file_path, 'wb') as f:
                if pickle:
                    pickle.dump(item.__dict__, f)
                else:
                    # JSON fallback (less efficient but more compatible)
                    json.dump(item.__dict__, f, default=str)
                    
        except Exception as e:
            logger.error(f"Failed to write cache file {file_path}: {e}")
            raise ResourceError(f"Cache write failed: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete item from file cache."""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all items from file cache."""
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink(missing_ok=True)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in file cache."""
        item = await self.get(key)
        return item is not None
    
    async def _cleanup_old_files(self) -> None:
        """Clean up old cache files to stay within limits."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        if len(cache_files) >= self.max_files:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda p: p.stat().st_mtime)
            
            files_to_remove = len(cache_files) - self.max_files + 1
            for file_path in cache_files[:files_to_remove]:
                file_path.unlink(missing_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        hit_rate = (self.hits / (self.hits + self.misses)) * 100 if (self.hits + self.misses) > 0 else 0
        
        return {
            "backend": "file",
            "files": len(cache_files),
            "max_files": self.max_files,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate
        }


class MultiTierCache(CacheBackend):
    """Multi-tier cache combining memory and file storage."""
    
    def __init__(
        self,
        l1_cache: Optional[CacheBackend] = None,
        l2_cache: Optional[CacheBackend] = None
    ):
        self.l1_cache = l1_cache or MemoryCache(max_size=500, max_memory_mb=50)
        self.l2_cache = l2_cache or FileCache()
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.total_misses = 0
    
    async def get(self, key: str) -> Optional[CacheItem]:
        """Get item from multi-tier cache."""
        # Try L1 cache first
        item = await self.l1_cache.get(key)
        if item:
            self.l1_hits += 1
            return item
        
        # Try L2 cache
        item = await self.l2_cache.get(key)
        if item:
            self.l2_hits += 1
            # Promote to L1 cache
            await self.l1_cache.set(key, item.value, item.ttl_seconds)
            return item
        
        self.total_misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set item in multi-tier cache."""
        # Set in both caches
        await asyncio.gather(
            self.l1_cache.set(key, value, ttl_seconds),
            self.l2_cache.set(key, value, ttl_seconds)
        )
    
    async def delete(self, key: str) -> bool:
        """Delete item from multi-tier cache."""
        results = await asyncio.gather(
            self.l1_cache.delete(key),
            self.l2_cache.delete(key)
        )
        return any(results)
    
    async def clear(self) -> None:
        """Clear all items from multi-tier cache."""
        await asyncio.gather(
            self.l1_cache.clear(),
            self.l2_cache.clear()
        )
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in multi-tier cache."""
        return (await self.l1_cache.exists(key)) or (await self.l2_cache.exists(key))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-tier cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_requests = self.l1_hits + self.l2_hits + self.total_misses
        l1_hit_rate = (self.l1_hits / total_requests) * 100 if total_requests > 0 else 0
        l2_hit_rate = (self.l2_hits / total_requests) * 100 if total_requests > 0 else 0
        overall_hit_rate = ((self.l1_hits + self.l2_hits) / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "backend": "multi_tier",
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "total_misses": self.total_misses,
            "l1_hit_rate_percent": l1_hit_rate,
            "l2_hit_rate_percent": l2_hit_rate,
            "overall_hit_rate_percent": overall_hit_rate,
            "l1_cache": l1_stats,
            "l2_cache": l2_stats
        }


class SmartCacheManager:
    """Intelligent cache manager with adaptive strategies."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or MultiTierCache()
        self.request_patterns: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False
    ) -> Any:
        """Get from cache or execute factory function."""
        if not force_refresh:
            item = await self.backend.get(key)
            if item:
                self._track_access_pattern(key)
                return item.value
        
        # Execute factory function
        start_time = time.time()
        try:
            value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            execution_time = time.time() - start_time
            
            # Adaptive TTL based on execution time
            if ttl_seconds is None:
                if execution_time > 10:  # Expensive operations
                    ttl_seconds = 3600  # 1 hour
                elif execution_time > 1:  # Moderate operations
                    ttl_seconds = 900   # 15 minutes
                else:  # Fast operations
                    ttl_seconds = 300   # 5 minutes
            
            await self.backend.set(key, value, ttl_seconds)
            self._track_access_pattern(key)
            
            performance_logger.log_api_performance(
                endpoint="cache_factory",
                method="GET",
                duration_ms=execution_time * 1000,
                status_code=200,
                response_size_bytes=len(str(value))
            )
            
            return value
            
        except Exception as e:
            logger.error(f"Cache factory function failed for key {key}: {e}")
            raise
    
    def cache(
        self,
        ttl_seconds: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            
            async def async_wrapper(*args, **kwargs):
                cache_key = self._generate_cache_key(prefix, *args, **kwargs)
                
                return await self.get_or_set(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl_seconds
                )
            
            def sync_wrapper(*args, **kwargs):
                cache_key = self._generate_cache_key(prefix, *args, **kwargs)
                
                # For sync functions, we need to run async operations in event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(
                    self.get_or_set(
                        cache_key,
                        lambda: func(*args, **kwargs),
                        ttl_seconds
                    )
                )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _track_access_pattern(self, key: str) -> None:
        """Track access patterns for intelligent caching."""
        with self.lock:
            current_time = time.time()
            
            if key not in self.request_patterns:
                self.request_patterns[key] = []
            
            # Keep only recent access times (last hour)
            cutoff_time = current_time - 3600
            self.request_patterns[key] = [
                t for t in self.request_patterns[key] 
                if t > cutoff_time
            ]
            
            self.request_patterns[key].append(current_time)
    
    def get_access_frequency(self, key: str) -> float:
        """Get access frequency for a key (requests per hour)."""
        with self.lock:
            if key not in self.request_patterns:
                return 0.0
            
            return len(self.request_patterns[key])
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        backend_stats = self.backend.get_stats()
        
        # Analyze access patterns
        with self.lock:
            total_keys = len(self.request_patterns)
            if total_keys > 0:
                frequencies = [len(accesses) for accesses in self.request_patterns.values()]
                avg_frequency = sum(frequencies) / len(frequencies)
                max_frequency = max(frequencies)
            else:
                avg_frequency = 0
                max_frequency = 0
        
        return {
            "cache_manager": {
                "tracked_keys": total_keys,
                "average_access_frequency": avg_frequency,
                "max_access_frequency": max_frequency
            },
            "backend": backend_stats
        }


# Global cache manager instance
cache_manager = SmartCacheManager()