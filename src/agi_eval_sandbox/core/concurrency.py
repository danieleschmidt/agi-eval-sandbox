"""Advanced concurrency and resource management utilities."""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, PriorityQueue
import weakref

from .exceptions import ResourceError, TimeoutError
from .logging_config import get_logger, performance_logger
from .health import health_monitor

logger = get_logger("concurrency")

T = TypeVar('T')


@dataclass
class TaskResult(Generic[T]):
    """Result of a concurrent task."""
    task_id: str
    result: Optional[T] = None
    error: Optional[Exception] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.error is None
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.completed_at is not None


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_seconds: float = 0.0
    current_task: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    
    @property
    def average_duration(self) -> float:
        """Average task duration in seconds."""
        total_tasks = self.tasks_completed + self.tasks_failed
        return self.total_duration_seconds / total_tasks if total_tasks > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Task success rate as percentage."""
        total_tasks = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total_tasks) * 100 if total_tasks > 0 else 0.0


class AdaptiveThreadPool:
    """Thread pool that adapts size based on workload and system resources."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 20,
        scale_factor: float = 1.5,
        idle_timeout: int = 300
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_factor = scale_factor
        self.idle_timeout = idle_timeout
        
        self.executor = ThreadPoolExecutor(max_workers=min_workers)
        self.current_workers = min_workers
        self.pending_tasks = 0
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.last_scale_time = time.time()
        
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_workload, daemon=True)
        self.monitor_thread.start()
    
    async def submit(self, func: Callable, *args, **kwargs) -> TaskResult[T]:
        """Submit task to adaptive thread pool."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.pending_tasks += 1
        
        # Check if we need to scale up
        await self._maybe_scale_up()
        
        start_time = time.time()
        try:
            # Submit to thread pool
            future = self.executor.submit(func, *args, **kwargs)
            result = await asyncio.wrap_future(future)
            
            duration = time.time() - start_time
            
            # Update stats
            self._update_worker_stats(task_id, duration, success=True)
            
            return TaskResult(
                task_id=task_id,
                result=result,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_worker_stats(task_id, duration, success=False)
            
            return TaskResult(
                task_id=task_id,
                error=e,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                duration_seconds=duration
            )
        finally:
            with self.lock:
                self.pending_tasks = max(0, self.pending_tasks - 1)
    
    async def _maybe_scale_up(self) -> None:
        """Scale up workers if needed."""
        with self.lock:
            current_time = time.time()
            
            # Don't scale too frequently
            if current_time - self.last_scale_time < 30:
                return
            
            # Check if we need more workers
            if (self.pending_tasks > self.current_workers * 2 and 
                self.current_workers < self.max_workers):
                
                # Check system resources
                try:
                    metrics = health_monitor.collect_metrics()
                    if metrics.cpu_percent < 80 and metrics.memory_percent < 85:
                        new_workers = min(
                            int(self.current_workers * self.scale_factor),
                            self.max_workers
                        )
                        
                        logger.info(f"Scaling thread pool from {self.current_workers} to {new_workers} workers")
                        
                        # Create new executor with more workers
                        old_executor = self.executor
                        self.executor = ThreadPoolExecutor(max_workers=new_workers)
                        self.current_workers = new_workers
                        self.last_scale_time = current_time
                        
                        # Shutdown old executor gracefully
                        threading.Thread(target=lambda: old_executor.shutdown(wait=True), daemon=True).start()
                        
                except Exception as e:
                    logger.warning(f"Failed to check system resources for scaling: {e}")
    
    def _maybe_scale_down(self) -> None:
        """Scale down workers if they're idle."""
        with self.lock:
            if (self.pending_tasks < self.current_workers / 3 and 
                self.current_workers > self.min_workers):
                
                new_workers = max(
                    int(self.current_workers / self.scale_factor),
                    self.min_workers
                )
                
                logger.info(f"Scaling thread pool down from {self.current_workers} to {new_workers} workers")
                
                # Create new executor with fewer workers
                old_executor = self.executor
                self.executor = ThreadPoolExecutor(max_workers=new_workers)
                self.current_workers = new_workers
                self.last_scale_time = time.time()
                
                # Shutdown old executor gracefully
                threading.Thread(target=lambda: old_executor.shutdown(wait=True), daemon=True).start()
    
    def _monitor_workload(self) -> None:
        """Monitor workload and scale down if needed."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                # Check if we should scale down
                if time.time() - self.last_scale_time > self.idle_timeout:
                    self._maybe_scale_down()
                    
            except Exception as e:
                logger.error(f"Error in workload monitor: {e}")
    
    def _update_worker_stats(self, task_id: str, duration: float, success: bool) -> None:
        """Update worker statistics."""
        with self.lock:
            worker_id = f"worker_{threading.current_thread().ident}"
            
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
            
            stats = self.worker_stats[worker_id]
            if success:
                stats.tasks_completed += 1
            else:
                stats.tasks_failed += 1
            
            stats.total_duration_seconds += duration
            stats.current_task = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            total_completed = sum(stats.tasks_completed for stats in self.worker_stats.values())
            total_failed = sum(stats.tasks_failed for stats in self.worker_stats.values())
            avg_duration = sum(stats.average_duration for stats in self.worker_stats.values()) / len(self.worker_stats) if self.worker_stats else 0
            
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "pending_tasks": self.pending_tasks,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "average_duration_seconds": avg_duration,
                "success_rate_percent": (total_completed / (total_completed + total_failed)) * 100 if (total_completed + total_failed) > 0 else 0
            }


class TaskQueue:
    """Priority-based task queue with advanced scheduling."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: PriorityQueue = PriorityQueue(maxsize=max_size)
        self.results: Dict[str, TaskResult] = {}
        self.task_counter = 0
        self.lock = threading.RLock()
    
    async def enqueue(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0,
        timeout_seconds: Optional[int] = None
    ) -> str:
        """Enqueue a task with priority."""
        kwargs = kwargs or {}
        
        with self.lock:
            if self.queue.full():
                raise ResourceError("Task queue is full")
            
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time() * 1000)}"
            
            task_data = {
                "task_id": task_id,
                "func": func,
                "args": args,
                "kwargs": kwargs,
                "timeout_seconds": timeout_seconds,
                "enqueued_at": datetime.now()
            }
            
            # Higher priority values are processed first (negated for min-heap)
            self.queue.put((-priority, self.task_counter, task_data))
            
            logger.debug(f"Enqueued task {task_id} with priority {priority}")
            return task_id
    
    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Dequeue the highest priority task."""
        try:
            priority, counter, task_data = self.queue.get_nowait()
            return task_data
        except:
            return None
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task."""
        return self.results.get(task_id)
    
    def set_result(self, task_id: str, result: TaskResult) -> None:
        """Set result of a completed task."""
        self.results[task_id] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            completed_tasks = len([r for r in self.results.values() if r.is_completed])
            successful_tasks = len([r for r in self.results.values() if r.is_success])
            
            return {
                "queue_size": self.queue.qsize(),
                "max_size": self.max_size,
                "total_enqueued": self.task_counter,
                "completed_tasks": completed_tasks,
                "successful_tasks": successful_tasks,
                "pending_results": len(self.results)
            }


class ConcurrentEvaluationManager:
    """Manages concurrent evaluation execution with intelligent resource allocation."""
    
    def __init__(
        self,
        max_concurrent_evaluations: int = 10,
        max_concurrent_questions: int = 50,
        thread_pool: Optional[AdaptiveThreadPool] = None
    ):
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.max_concurrent_questions = max_concurrent_questions
        self.thread_pool = thread_pool or AdaptiveThreadPool()
        self.task_queue = TaskQueue()
        
        self.active_evaluations: Dict[str, Dict[str, Any]] = {}
        self.evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)
        self.question_semaphore = asyncio.Semaphore(max_concurrent_questions)
        
        self.lock = asyncio.Lock()
    
    async def run_evaluation_batch(
        self,
        evaluation_func: Callable,
        evaluation_configs: List[Dict[str, Any]],
        priority: int = 0
    ) -> List[TaskResult]:
        """Run multiple evaluations concurrently."""
        evaluation_id = f"batch_{int(time.time() * 1000)}"
        
        async with self.lock:
            self.active_evaluations[evaluation_id] = {
                "total_evaluations": len(evaluation_configs),
                "completed_evaluations": 0,
                "started_at": datetime.now(),
                "status": "running"
            }
        
        try:
            # Create semaphore-controlled evaluation tasks
            async def run_single_evaluation(config):
                async with self.evaluation_semaphore:
                    return await self._run_evaluation_with_monitoring(
                        evaluation_func,
                        config,
                        evaluation_id
                    )
            
            # Execute evaluations concurrently
            tasks = [run_single_evaluation(config) for config in evaluation_configs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(TaskResult(
                        task_id=f"{evaluation_id}_eval_{i}",
                        error=result,
                        completed_at=datetime.now()
                    ))
                else:
                    processed_results.append(result)
            
            # Update evaluation status
            async with self.lock:
                if evaluation_id in self.active_evaluations:
                    self.active_evaluations[evaluation_id]["status"] = "completed"
                    self.active_evaluations[evaluation_id]["completed_at"] = datetime.now()
            
            return processed_results
            
        except Exception as e:
            async with self.lock:
                if evaluation_id in self.active_evaluations:
                    self.active_evaluations[evaluation_id]["status"] = "failed"
                    self.active_evaluations[evaluation_id]["error"] = str(e)
            raise
    
    async def _run_evaluation_with_monitoring(
        self,
        evaluation_func: Callable,
        config: Dict[str, Any],
        evaluation_id: str
    ) -> TaskResult:
        """Run single evaluation with monitoring."""
        task_id = f"{evaluation_id}_eval_{config.get('model_name', 'unknown')}"
        start_time = time.time()
        
        try:
            # Execute evaluation
            if asyncio.iscoroutinefunction(evaluation_func):
                result = await evaluation_func(**config)
            else:
                result = await self.thread_pool.submit(evaluation_func, **config)
            
            duration = time.time() - start_time
            
            # Log performance
            performance_logger.log_evaluation_performance(
                model_name=config.get('model_name', 'unknown'),
                benchmark=config.get('benchmark', 'unknown'),
                duration_seconds=duration,
                questions_count=config.get('num_questions', 0),
                success=True
            )
            
            # Update progress
            async with self.lock:
                if evaluation_id in self.active_evaluations:
                    self.active_evaluations[evaluation_id]["completed_evaluations"] += 1
            
            return TaskResult(
                task_id=task_id,
                result=result,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(f"Evaluation failed for task {task_id}: {e}")
            performance_logger.log_evaluation_performance(
                model_name=config.get('model_name', 'unknown'),
                benchmark=config.get('benchmark', 'unknown'),
                duration_seconds=duration,
                questions_count=config.get('num_questions', 0),
                success=False
            )
            
            return TaskResult(
                task_id=task_id,
                error=e,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                duration_seconds=duration
            )
    
    async def run_question_batch(
        self,
        question_func: Callable,
        questions: List[Any],
        batch_size: int = 10
    ) -> List[TaskResult]:
        """Run questions in concurrent batches."""
        results = []
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            
            async def process_question(question):
                async with self.question_semaphore:
                    if asyncio.iscoroutinefunction(question_func):
                        return await question_func(question)
                    else:
                        return await self.thread_pool.submit(question_func, question)
            
            # Execute batch concurrently
            batch_tasks = [process_question(q) for q in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Convert to TaskResult objects
            for j, result in enumerate(batch_results):
                task_id = f"question_{i + j}"
                if isinstance(result, Exception):
                    results.append(TaskResult(task_id=task_id, error=result, completed_at=datetime.now()))
                else:
                    results.append(TaskResult(task_id=task_id, result=result, completed_at=datetime.now()))
        
        return results
    
    def get_evaluation_status(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running evaluation."""
        return self.active_evaluations.get(evaluation_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        async with self.lock:
            active_count = len([e for e in self.active_evaluations.values() if e["status"] == "running"])
            completed_count = len([e for e in self.active_evaluations.values() if e["status"] == "completed"])
            failed_count = len([e for e in self.active_evaluations.values() if e["status"] == "failed"])
        
        thread_pool_stats = self.thread_pool.get_stats()
        queue_stats = self.task_queue.get_stats()
        
        return {
            "evaluations": {
                "active": active_count,
                "completed": completed_count,
                "failed": failed_count,
                "max_concurrent": self.max_concurrent_evaluations
            },
            "questions": {
                "max_concurrent": self.max_concurrent_questions,
                "available_slots": self.question_semaphore._value
            },
            "thread_pool": thread_pool_stats,
            "task_queue": queue_stats
        }


# Global concurrency manager
concurrency_manager = ConcurrentEvaluationManager()