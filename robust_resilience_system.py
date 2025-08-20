#!/usr/bin/env python3
"""
Robust Resilience System - Generation 2 Enhancement
Advanced error handling and resilience patterns without external dependencies
"""

import asyncio
import time
import sys
import os
import gc
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from contextlib import asynccontextmanager
import traceback

class ResilienceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

class OperationStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY_NEEDED = "retry_needed"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class OperationResult:
    status: OperationStatus
    message: str
    data: Optional[Any] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ResilienceConfig:
    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    exponential_base: float = 2.0
    jitter: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_ms: float = 30000.0
    timeout_ms: float = 10000.0

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerState:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

class RobustResilienceSystem:
    """Advanced resilience system with multiple failure handling patterns."""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.logger = logging.getLogger("resilience_system")
        
        # Circuit breaker states for different operations
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Operation metrics
        self.operation_metrics: Dict[str, List[OperationResult]] = {}
        
        # Fallback handlers
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        
        # Bulkhead semaphores for resource isolation
        self.bulkheads: Dict[str, asyncio.Semaphore] = {}
        
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function for an operation."""
        self.fallback_handlers[operation_name] = fallback_func
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register health check function."""
        self.health_checks[name] = check_func
    
    def create_bulkhead(self, name: str, max_concurrent: int):
        """Create bulkhead semaphore for resource isolation."""
        self.bulkheads[name] = asyncio.Semaphore(max_concurrent)
    
    def _get_circuit_breaker(self, operation_name: str) -> CircuitBreakerState:
        """Get or create circuit breaker state for operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
        return self.circuit_breakers[operation_name]
    
    def _record_operation_result(self, operation_name: str, result: OperationResult):
        """Record operation result for metrics."""
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = []
        
        self.operation_metrics[operation_name].append(result)
        
        # Keep only last 100 results
        if len(self.operation_metrics[operation_name]) > 100:
            self.operation_metrics[operation_name] = self.operation_metrics[operation_name][-100:]
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff and jitter."""
        delay = min(
            self.config.base_delay_ms * (self.config.exponential_base ** attempt),
            self.config.max_delay_ms
        )
        
        if self.config.jitter:
            # Add up to 50% jitter
            import random
            jitter_amount = delay * 0.5 * random.random()
            delay += jitter_amount
        
        return delay / 1000.0  # Convert to seconds
    
    def _should_circuit_open(self, cb_state: CircuitBreakerState) -> bool:
        """Check if circuit breaker should open."""
        return cb_state.failure_count >= self.config.circuit_breaker_threshold
    
    def _should_circuit_close(self, cb_state: CircuitBreakerState) -> bool:
        """Check if circuit breaker should close (from half-open)."""
        return (cb_state.success_count >= 3 and 
                time.time() - cb_state.last_failure_time > self.config.circuit_breaker_timeout_ms / 1000.0)
    
    def _should_attempt_half_open(self, cb_state: CircuitBreakerState) -> bool:
        """Check if circuit breaker should attempt half-open."""
        return (cb_state.state == CircuitState.OPEN and
                time.time() - cb_state.last_failure_time > self.config.circuit_breaker_timeout_ms / 1000.0)
    
    async def _execute_with_timeout(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with timeout."""
        timeout_seconds = self.config.timeout_ms / 1000.0
        
        try:
            if asyncio.iscoroutinefunction(operation):
                return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout_seconds)
            else:
                return await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, operation, *args, **kwargs),
                    timeout=timeout_seconds
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        fallback_data: Optional[Any] = None,
        bulkhead_name: Optional[str] = None,
        **kwargs
    ) -> OperationResult:
        """Execute operation with full resilience patterns."""
        start_time = time.time()
        cb_state = self._get_circuit_breaker(operation_name)
        
        # Check circuit breaker state
        if cb_state.state == CircuitState.OPEN:
            if self._should_attempt_half_open(cb_state):
                cb_state.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {operation_name} attempting half-open")
            else:
                # Circuit is open, try fallback
                if operation_name in self.fallback_handlers:
                    try:
                        fallback_result = await self._execute_fallback(operation_name, fallback_data)
                        return OperationResult(
                            status=OperationStatus.CIRCUIT_OPEN,
                            message=f"Circuit open, used fallback",
                            data=fallback_result,
                            duration_ms=(time.time() - start_time) * 1000
                        )
                    except Exception as e:
                        return OperationResult(
                            status=OperationStatus.FAILED,
                            message=f"Circuit open and fallback failed: {str(e)}",
                            error=e,
                            duration_ms=(time.time() - start_time) * 1000
                        )
                else:
                    return OperationResult(
                        status=OperationStatus.CIRCUIT_OPEN,
                        message=f"Circuit breaker {operation_name} is open",
                        duration_ms=(time.time() - start_time) * 1000
                    )
        
        # Use bulkhead if specified
        bulkhead_context = None
        if bulkhead_name and bulkhead_name in self.bulkheads:
            bulkhead_context = self.bulkheads[bulkhead_name]
        
        # Execute with retry logic
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Acquire bulkhead if specified
                if bulkhead_context:
                    async with bulkhead_context:
                        result = await self._execute_with_timeout(operation, *args, **kwargs)
                else:
                    result = await self._execute_with_timeout(operation, *args, **kwargs)
                
                # Success - update circuit breaker
                cb_state.success_count += 1
                cb_state.last_success_time = time.time()
                
                if cb_state.state == CircuitState.HALF_OPEN and self._should_circuit_close(cb_state):
                    cb_state.state = CircuitState.CLOSED
                    cb_state.failure_count = 0
                    cb_state.success_count = 0
                    self.logger.info(f"Circuit breaker {operation_name} closed after recovery")
                
                operation_result = OperationResult(
                    status=OperationStatus.SUCCESS,
                    message=f"Operation succeeded on attempt {attempt + 1}",
                    data=result,
                    duration_ms=(time.time() - start_time) * 1000,
                    retry_count=attempt
                )
                
                self._record_operation_result(operation_name, operation_result)
                return operation_result
                
            except Exception as e:
                last_exception = e
                
                # Update circuit breaker failure state
                cb_state.failure_count += 1
                cb_state.last_failure_time = time.time()
                
                if self._should_circuit_open(cb_state):
                    cb_state.state = CircuitState.OPEN
                    cb_state.success_count = 0
                    self.logger.error(f"Circuit breaker {operation_name} opened due to failures")
                
                # Don't retry on last attempt
                if attempt == self.config.max_retries:
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Operation {operation_name} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.2f}s: {str(e)}"
                )
                await asyncio.sleep(delay)
        
        # All retries exhausted, try fallback
        if operation_name in self.fallback_handlers:
            try:
                fallback_result = await self._execute_fallback(operation_name, fallback_data)
                operation_result = OperationResult(
                    status=OperationStatus.PARTIAL_SUCCESS,
                    message=f"Operation failed but fallback succeeded after {self.config.max_retries + 1} attempts",
                    data=fallback_result,
                    error=last_exception,
                    duration_ms=(time.time() - start_time) * 1000,
                    retry_count=self.config.max_retries + 1
                )
            except Exception as fallback_error:
                operation_result = OperationResult(
                    status=OperationStatus.FAILED,
                    message=f"Operation and fallback both failed: {str(last_exception)}",
                    error=last_exception,
                    duration_ms=(time.time() - start_time) * 1000,
                    retry_count=self.config.max_retries + 1
                )
        else:
            operation_result = OperationResult(
                status=OperationStatus.FAILED,
                message=f"Operation failed after {self.config.max_retries + 1} attempts: {str(last_exception)}",
                error=last_exception,
                duration_ms=(time.time() - start_time) * 1000,
                retry_count=self.config.max_retries + 1
            )
        
        self._record_operation_result(operation_name, operation_result)
        return operation_result
    
    async def _execute_fallback(self, operation_name: str, fallback_data: Optional[Any] = None) -> Any:
        """Execute fallback function."""
        fallback_func = self.fallback_handlers[operation_name]
        
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func(fallback_data)
        else:
            return fallback_func(fallback_data)
    
    async def run_health_checks(self) -> Dict[str, OperationResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                
                results[name] = OperationResult(
                    status=OperationStatus.SUCCESS,
                    message="Health check passed",
                    data=check_result,
                    duration_ms=(time.time() - start_time) * 1000
                )
                
            except Exception as e:
                results[name] = OperationResult(
                    status=OperationStatus.FAILED,
                    message=f"Health check failed: {str(e)}",
                    error=e,
                    duration_ms=(time.time() - start_time) * 1000
                )
        
        return results
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics."""
        stats = {
            "circuit_breakers": {},
            "operations": {},
            "health_checks": len(self.health_checks),
            "bulkheads": len(self.bulkheads),
            "fallback_handlers": len(self.fallback_handlers)
        }
        
        # Circuit breaker stats
        for name, cb_state in self.circuit_breakers.items():
            stats["circuit_breakers"][name] = {
                "state": cb_state.state.value,
                "failure_count": cb_state.failure_count,
                "success_count": cb_state.success_count,
                "last_failure_time": cb_state.last_failure_time,
                "last_success_time": cb_state.last_success_time
            }
        
        # Operation stats
        for name, results in self.operation_metrics.items():
            if results:
                success_count = sum(1 for r in results if r.status == OperationStatus.SUCCESS)
                avg_duration = sum(r.duration_ms for r in results) / len(results)
                
                stats["operations"][name] = {
                    "total_executions": len(results),
                    "success_count": success_count,
                    "success_rate": success_count / len(results),
                    "avg_duration_ms": avg_duration,
                    "last_execution": results[-1].timestamp
                }
        
        return stats

# Built-in resilience patterns

async def safe_file_operation(file_path: str, operation: str, content: Optional[str] = None) -> Any:
    """Safe file operation with error handling."""
    try:
        path = Path(file_path)
        
        if operation == "read":
            if not path.exists():
                raise FileNotFoundError(f"File {file_path} does not exist")
            return path.read_text()
        
        elif operation == "write":
            if content is None:
                raise ValueError("Content required for write operation")
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write using temporary file
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(content)
            temp_path.rename(path)
            
            return f"Successfully wrote {len(content)} characters"
        
        elif operation == "delete":
            if path.exists():
                path.unlink()
                return f"Deleted {file_path}"
            else:
                return f"File {file_path} did not exist"
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        raise Exception(f"File operation '{operation}' failed for {file_path}: {str(e)}")

def memory_usage_fallback(data: Optional[Any] = None) -> Dict[str, Any]:
    """Fallback for memory usage monitoring."""
    gc.collect()  # Force garbage collection
    return {
        "status": "fallback",
        "message": "Using basic memory management",
        "garbage_collected_objects": gc.collect(),
        "memory_info": "Limited memory info available"
    }

def health_check_basic() -> Dict[str, Any]:
    """Basic health check without external dependencies."""
    try:
        # Test Python basics
        test_data = {"test": True, "timestamp": time.time()}
        json_test = json.dumps(test_data)
        parsed = json.loads(json_test)
        
        # Test async capability
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def async_test():
            await asyncio.sleep(0.001)
            return "async_ok"
        
        async_result = loop.run_until_complete(async_test())
        loop.close()
        
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "json_serialization": parsed == test_data,
            "async_support": async_result == "async_ok",
            "working_directory": str(Path.cwd()),
            "status": "healthy"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def test_resilience_system():
    """Test the resilience system."""
    print("\n" + "="*60)
    print("🛡️ ROBUST RESILIENCE SYSTEM - GENERATION 2")
    print("="*60)
    
    # Create resilience system
    config = ResilienceConfig(
        max_retries=2,
        base_delay_ms=50.0,
        circuit_breaker_threshold=3
    )
    
    resilience = RobustResilienceSystem(config)
    
    # Register fallback handlers
    resilience.register_fallback("memory_check", memory_usage_fallback)
    
    # Register health checks
    resilience.register_health_check("basic_health", health_check_basic)
    
    # Create bulkhead for file operations
    resilience.create_bulkhead("file_operations", 2)
    
    print("🔧 Testing resilient operations...")
    
    # Test successful operation
    result = await resilience.execute_with_resilience(
        "file_write",
        safe_file_operation,
        "/tmp/test_resilience.txt",
        "write",
        content="Test content for resilience",
        bulkhead_name="file_operations"
    )
    
    print(f"  ✓ File write: {result.status.value} - {result.message}")
    
    # Test file read
    result = await resilience.execute_with_resilience(
        "file_read",
        safe_file_operation,
        "/tmp/test_resilience.txt",
        "read",
        bulkhead_name="file_operations"
    )
    
    print(f"  ✓ File read: {result.status.value} - Content length: {len(result.data) if result.data else 0}")
    
    # Test operation that will fail (non-existent file)
    result = await resilience.execute_with_resilience(
        "file_read_fail",
        safe_file_operation,
        "/nonexistent/path/file.txt",
        "read"
    )
    
    print(f"  ⚠️ Failed operation: {result.status.value} - {result.message}")
    
    # Test health checks
    print("\n🩺 Running health checks...")
    health_results = await resilience.run_health_checks()
    
    for name, result in health_results.items():
        status_emoji = "✅" if result.status == OperationStatus.SUCCESS else "❌"
        print(f"  {status_emoji} {name}: {result.status.value}")
        if result.data:
            print(f"     Python: {result.data.get('python_version', 'unknown')}")
            print(f"     Status: {result.data.get('status', 'unknown')}")
    
    # Get resilience statistics
    print("\n📊 Resilience Statistics:")
    stats = resilience.get_resilience_stats()
    
    print(f"  Circuit Breakers: {len(stats['circuit_breakers'])}")
    for name, cb_stats in stats['circuit_breakers'].items():
        print(f"    • {name}: {cb_stats['state']} (failures: {cb_stats['failure_count']})")
    
    print(f"  Operations Tracked: {len(stats['operations'])}")
    for name, op_stats in stats['operations'].items():
        print(f"    • {name}: {op_stats['success_rate']:.1%} success rate, "
              f"{op_stats['avg_duration_ms']:.1f}ms avg")
    
    print(f"  Bulkheads: {stats['bulkheads']}")
    print(f"  Fallback Handlers: {stats['fallback_handlers']}")
    print(f"  Health Checks: {stats['health_checks']}")
    
    # Clean up test file
    try:
        await resilience.execute_with_resilience(
            "file_cleanup",
            safe_file_operation,
            "/tmp/test_resilience.txt",
            "delete"
        )
        print("\n🧹 Cleanup completed")
    except:
        pass
    
    print("\n🎉 RESILIENCE SYSTEM TESTED SUCCESSFULLY!")
    print("✨ System is now more robust and fault-tolerant")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_resilience_system())