"""
Tests for performance optimization and caching system.

Tests lazy loading, caching, parallel processing, GPU acceleration,
and benchmarking functionality.
"""

import os
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from neurolite.core.performance import (
    LazyLoader,
    CacheManager,
    ParallelProcessor,
    GPUAccelerator,
    get_cache_manager,
    get_gpu_accelerator,
    lazy_load,
    cached,
    parallel_map,
    gpu_context
)
from neurolite.core.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    PerformanceMonitor,
    BenchmarkRunner,
    get_benchmark_runner
)


class TestLazyLoader:
    """Test lazy loading functionality."""
    
    def test_lazy_loader_basic(self):
        """Test basic lazy loading functionality."""
        call_count = 0
        
        def expensive_function():
            nonlocal call_count
            call_count += 1
            return "expensive_result"
        
        loader = LazyLoader(expensive_function)
        
        # Object should not be loaded initially
        assert not loader.is_loaded
        assert loader.load_time is None
        
        # First call should load the object
        result = loader()
        assert result == "expensive_result"
        assert call_count == 1
        assert loader.is_loaded
        assert loader.load_time is not None
        
        # Second call should return cached object
        result2 = loader()
        assert result2 == "expensive_result"
        assert call_count == 1  # Should not increment
    
    def test_lazy_loader_with_args(self):
        """Test lazy loader with arguments."""
        def function_with_args(x, y, z=10):
            return x + y + z
        
        loader = LazyLoader(function_with_args, 5, 3, z=2)
        result = loader()
        assert result == 10
    
    def test_lazy_loader_unload(self):
        """Test lazy loader unload functionality."""
        def expensive_function():
            return "expensive_result"
        
        loader = LazyLoader(expensive_function)
        
        # Load the object
        result = loader()
        assert loader.is_loaded
        
        # Unload the object
        loader.unload()
        assert not loader.is_loaded
        assert loader.load_time is None
    
    def test_lazy_loader_thread_safety(self):
        """Test lazy loader thread safety."""
        call_count = 0
        
        def expensive_function():
            nonlocal call_count
            time.sleep(0.1)  # Simulate expensive operation
            call_count += 1
            return "expensive_result"
        
        loader = LazyLoader(expensive_function)
        results = []
        
        def worker():
            results.append(loader())
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should only be called once despite multiple threads
        assert call_count == 1
        assert all(result == "expensive_result" for result in results)


class TestCacheManager:
    """Test caching functionality."""
    
    def setup_method(self):
        """Set up test cache manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.cache_manager = CacheManager(
                cache_dir=temp_dir,
                max_memory_size=1024 * 1024,  # 1MB
                max_disk_size=10 * 1024 * 1024,  # 10MB
                default_ttl=None
            )
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        # Test set and get
        assert self.cache_manager.set("test_key", "test_value")
        result = self.cache_manager.get("test_key")
        assert result == "test_value"
        
        # Test non-existent key
        result = self.cache_manager.get("non_existent")
        assert result is None
    
    def test_cache_namespaces(self):
        """Test cache namespaces."""
        # Set values in different namespaces
        self.cache_manager.set("key", "value1", namespace="ns1")
        self.cache_manager.set("key", "value2", namespace="ns2")
        
        # Values should be separate
        assert self.cache_manager.get("key", namespace="ns1") == "value1"
        assert self.cache_manager.get("key", namespace="ns2") == "value2"
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        # Set with short TTL
        self.cache_manager.set("ttl_key", "ttl_value", ttl=0.1)
        
        # Should be available immediately
        assert self.cache_manager.get("ttl_key") == "ttl_value"
        
        # Should expire after TTL
        time.sleep(0.2)
        assert self.cache_manager.get("ttl_key") is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Set values in different namespaces
        self.cache_manager.set("key1", "value1", namespace="ns1")
        self.cache_manager.set("key2", "value2", namespace="ns2")
        
        # Clear specific namespace
        self.cache_manager.clear(namespace="ns1")
        assert self.cache_manager.get("key1", namespace="ns1") is None
        assert self.cache_manager.get("key2", namespace="ns2") == "value2"
        
        # Clear all
        self.cache_manager.clear()
        assert self.cache_manager.get("key2", namespace="ns2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some data
        self.cache_manager.set("key1", "value1")
        self.cache_manager.set("key2", "value2")
        
        stats = self.cache_manager.get_stats()
        assert "memory_entries" in stats
        assert "disk_entries" in stats
        assert "total_entries" in stats
        assert stats["total_entries"] >= 2


class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    def test_parallel_map_basic(self):
        """Test basic parallel map functionality."""
        def square(x):
            return x * x
        
        processor = ParallelProcessor(max_workers=2)
        data = [1, 2, 3, 4, 5]
        results = processor.map(square, data)
        
        assert results == [1, 4, 9, 16, 25]
    
    def test_parallel_map_with_progress(self):
        """Test parallel map with progress callback."""
        def slow_function(x):
            time.sleep(0.01)
            return x * 2
        
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        processor = ParallelProcessor(max_workers=2)
        data = [1, 2, 3, 4]
        results = processor.map(slow_function, data, progress_callback)
        
        assert results == [2, 4, 6, 8]
        assert len(progress_calls) == 4
        assert progress_calls[-1] == (4, 4)
    
    def test_parallel_map_empty_input(self):
        """Test parallel map with empty input."""
        processor = ParallelProcessor(max_workers=2)
        results = processor.map(lambda x: x, [])
        assert results == []
    
    def test_parallel_map_single_item(self):
        """Test parallel map with single item."""
        processor = ParallelProcessor(max_workers=2)
        results = processor.map(lambda x: x * 2, [5])
        assert results == [10]
    
    def test_parallel_map_batches(self):
        """Test parallel batch processing."""
        def sum_batch(batch):
            return sum(batch)
        
        processor = ParallelProcessor(max_workers=2, chunk_size=3)
        data = list(range(10))
        results = processor.map_batches(sum_batch, data)
        
        # Should process in batches and return flattened results
        assert isinstance(results, list)
    
    def test_parallel_processor_processes(self):
        """Test parallel processor with processes."""
        # Use a simple function that can be pickled
        import math
        
        processor = ParallelProcessor(max_workers=2, use_processes=True)
        data = [1, 2, 3]
        results = processor.map(math.sqrt, data)
        
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)
        assert results[0] == 1.0
        assert results[1] == math.sqrt(2)
        assert results[2] == math.sqrt(3)


class TestGPUAccelerator:
    """Test GPU acceleration functionality."""
    
    def setup_method(self):
        """Set up GPU accelerator."""
        self.accelerator = GPUAccelerator()
    
    def test_gpu_detection(self):
        """Test GPU detection."""
        # Should not raise errors
        device_info = self.accelerator.device_info
        assert isinstance(device_info, dict)
        assert "cuda_available" in device_info
        assert "mps_available" in device_info
        assert "recommended_device" in device_info
    
    def test_recommended_device(self):
        """Test recommended device selection."""
        device = self.accelerator.recommended_device
        assert device in ["cpu", "cuda", "mps"]
    
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        batch_size = self.accelerator.get_optimal_batch_size(
            model_size_mb=100,
            input_size_mb=1
        )
        assert isinstance(batch_size, int)
        assert batch_size >= 1
    
    def test_device_context(self):
        """Test device context manager."""
        with self.accelerator.device_context() as device:
            assert device in ["cpu", "cuda", "mps"]
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        # Should not raise errors
        self.accelerator.optimize_memory_usage()


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_cache_manager(self):
        """Test global cache manager."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2  # Should be singleton
    
    def test_get_gpu_accelerator(self):
        """Test global GPU accelerator."""
        acc1 = get_gpu_accelerator()
        acc2 = get_gpu_accelerator()
        assert acc1 is acc2  # Should be singleton
    
    def test_lazy_load_function(self):
        """Test lazy_load utility function."""
        def expensive_func():
            return "result"
        
        loader = lazy_load(expensive_func)
        assert isinstance(loader, LazyLoader)
        assert loader() == "result"
    
    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = 0
        
        @cached(namespace="test", ttl=1.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1
    
    def test_parallel_map_function(self):
        """Test parallel_map utility function."""
        results = parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2)
        assert results == [2, 4, 6]
    
    def test_gpu_context_function(self):
        """Test gpu_context utility function."""
        with gpu_context() as device:
            assert device in ["cpu", "cuda", "mps"]


class TestBenchmarkResult:
    """Test benchmark result functionality."""
    
    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        result = BenchmarkResult(
            name="test_benchmark",
            duration=1.5,
            memory_usage={"delta_rss": 100},
            gpu_memory_usage=None,
            throughput=1000.0,
            accuracy=0.95,
            metadata={"test": True},
            timestamp=time.time()
        )
        
        assert result.name == "test_benchmark"
        assert result.duration == 1.5
        assert result.throughput == 1000.0
        assert result.accuracy == 0.95
    
    def test_benchmark_result_serialization(self):
        """Test benchmark result serialization."""
        result = BenchmarkResult(
            name="test",
            duration=1.0,
            memory_usage={},
            gpu_memory_usage=None,
            throughput=None,
            accuracy=None,
            metadata={},
            timestamp=time.time()
        )
        
        # Test to_dict and from_dict
        data = result.to_dict()
        restored = BenchmarkResult.from_dict(data)
        
        assert restored.name == result.name
        assert restored.duration == result.duration


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_basic(self):
        """Test basic performance monitoring."""
        monitor = PerformanceMonitor(interval=0.01)
        
        monitor.start_monitoring()
        time.sleep(0.05)  # Let it collect some measurements
        summary = monitor.stop_monitoring()
        
        assert isinstance(summary, dict)
        if summary:  # May be empty if monitoring was too brief
            assert "duration" in summary
            assert "samples" in summary
    
    def test_performance_monitor_not_started(self):
        """Test stopping monitor that wasn't started."""
        monitor = PerformanceMonitor()
        summary = monitor.stop_monitoring()
        assert summary == {}


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""
    
    def setup_method(self):
        """Set up benchmark runner."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(
            results_dir=self.temp_dir,
            warmup_runs=1,
            measurement_runs=2
        )
    
    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_context(self):
        """Test benchmark context manager."""
        with self.runner.benchmark_context("test_benchmark") as results:
            time.sleep(0.01)  # Simulate work
            results["custom_metric"] = 42
        
        assert "name" in results
        assert "duration" in results
        assert "memory_usage" in results
        assert results["custom_metric"] == 42
    
    def test_run_benchmark(self):
        """Test running a single benchmark."""
        def test_function(x):
            time.sleep(0.01)
            return x * 2
        
        result = self.runner.run_benchmark(
            test_function,
            "test_benchmark",
            args=(5,),
            metadata={"input": 5}
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_benchmark"
        assert result.duration > 0
        assert "input" in result.metadata
    
    def test_run_benchmark_with_error(self):
        """Test benchmark with function that raises error."""
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(RuntimeError):
            self.runner.run_benchmark(failing_function, "failing_test")
    
    def test_run_suite(self):
        """Test running a benchmark suite."""
        def func1():
            return "result1"
        
        def func2(x):
            return x * 2
        
        benchmarks = [
            (func1, "benchmark1", (), {}),
            (func2, "benchmark2", (10,), {})
        ]
        
        suite = self.runner.run_suite(benchmarks, "test_suite", save_results=False)
        
        assert isinstance(suite, BenchmarkSuite)
        assert suite.name == "test_suite"
        assert len(suite.results) == 2
        assert "total_benchmarks" in suite.summary
    
    def test_save_and_load_suite(self):
        """Test saving and loading benchmark suite."""
        # Create a simple suite
        suite = BenchmarkSuite(
            name="test_suite",
            results=[],
            summary={"test": True},
            timestamp=time.time()
        )
        
        # Save suite
        self.runner.save_suite(suite)
        
        # Find the saved file - check in the benchmarks subdirectory
        benchmarks_dir = Path(self.temp_dir) / "benchmarks"
        saved_files = list(benchmarks_dir.glob("test_suite_*.json"))
        assert len(saved_files) == 1
        
        # Load suite
        loaded_suite = self.runner.load_suite(saved_files[0])
        assert loaded_suite.name == suite.name
        assert loaded_suite.summary == suite.summary
    
    def test_compare_suites(self):
        """Test suite comparison for regression detection."""
        # Create baseline suite
        baseline_result = BenchmarkResult(
            name="test_benchmark",
            duration=1.0,
            memory_usage={"avg_delta_rss": 100},
            gpu_memory_usage=None,
            throughput=None,
            accuracy=None,
            metadata={},
            timestamp=time.time()
        )
        baseline_suite = BenchmarkSuite(
            name="baseline",
            results=[baseline_result],
            summary={},
            timestamp=time.time()
        )
        
        # Create current suite with regression
        current_result = BenchmarkResult(
            name="test_benchmark",
            duration=1.5,  # 50% slower
            memory_usage={"avg_delta_rss": 150},  # 50% more memory
            gpu_memory_usage=None,
            throughput=None,
            accuracy=None,
            metadata={},
            timestamp=time.time()
        )
        current_suite = BenchmarkSuite(
            name="current",
            results=[current_result],
            summary={},
            timestamp=time.time()
        )
        
        comparison = self.runner.compare_suites(baseline_suite, current_suite, threshold=0.1)
        
        assert "regressions" in comparison
        assert "improvements" in comparison
        assert len(comparison["regressions"]) == 1  # Should detect regression
    
    def test_get_benchmark_runner(self):
        """Test global benchmark runner."""
        runner1 = get_benchmark_runner()
        runner2 = get_benchmark_runner()
        assert runner1 is runner2  # Should be singleton