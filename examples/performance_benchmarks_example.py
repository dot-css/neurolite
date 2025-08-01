#!/usr/bin/env python3
"""
Performance Benchmarks Example

This example demonstrates how to use NeuroLite's performance optimization
and benchmarking features including lazy loading, caching, parallel processing,
and GPU acceleration.
"""

import time
import numpy as np
from pathlib import Path

# Import NeuroLite performance utilities
from neurolite.core.performance import (
    LazyLoader, 
    CacheManager, 
    ParallelProcessor, 
    GPUAccelerator,
    lazy_load,
    cached,
    parallel_map,
    gpu_context
)
from neurolite.core.benchmarks import (
    BenchmarkRunner,
    get_benchmark_runner
)


def expensive_computation(n: int) -> float:
    """Simulate an expensive computation."""
    result = 0.0
    for i in range(n * 1000):
        result += np.sqrt(i + 1)
    return result


def data_processing_task(data_chunk: list) -> list:
    """Simulate data processing on a chunk."""
    return [x * 2 + 1 for x in data_chunk]


@cached(namespace="examples", ttl=300)  # Cache for 5 minutes
def cached_expensive_function(x: int) -> int:
    """Example of cached function."""
    print(f"Computing expensive function for {x}")
    time.sleep(0.1)  # Simulate expensive computation
    return x ** 2 + x + 1


def demonstrate_lazy_loading():
    """Demonstrate lazy loading functionality."""
    print("\n=== Lazy Loading Demo ===")
    
    # Create a lazy loader for expensive computation
    lazy_computation = lazy_load(expensive_computation, 1000)
    
    print("Lazy loader created (computation not executed yet)")
    print(f"Is loaded: {lazy_computation.is_loaded}")
    
    # First access triggers computation
    start_time = time.time()
    result = lazy_computation()
    end_time = time.time()
    
    print(f"First access result: {result:.2f}")
    print(f"Computation time: {end_time - start_time:.3f}s")
    print(f"Is loaded: {lazy_computation.is_loaded}")
    print(f"Load time: {lazy_computation.load_time:.3f}s")
    
    # Second access uses cached result
    start_time = time.time()
    result2 = lazy_computation()
    end_time = time.time()
    
    print(f"Second access result: {result2:.2f}")
    print(f"Access time: {end_time - start_time:.6f}s (much faster!)")


def demonstrate_caching():
    """Demonstrate caching functionality."""
    print("\n=== Caching Demo ===")
    
    # Test cached decorator
    print("Testing cached decorator:")
    
    # First call - will compute
    result1 = cached_expensive_function(5)
    print(f"First call result: {result1}")
    
    # Second call - will use cache
    result2 = cached_expensive_function(5)
    print(f"Second call result: {result2} (from cache)")
    
    # Different parameter - will compute
    result3 = cached_expensive_function(10)
    print(f"Different parameter result: {result3}")
    
    # Manual cache management
    cache_manager = CacheManager()
    
    # Store some data
    cache_manager.set("test_key", {"data": [1, 2, 3, 4, 5]}, namespace="demo")
    
    # Retrieve data
    cached_data = cache_manager.get("test_key", namespace="demo")
    print(f"Retrieved from cache: {cached_data}")
    
    # Show cache statistics
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats['total_entries']} entries, {stats['total_size_formatted']} total size")


def demonstrate_parallel_processing():
    """Demonstrate parallel processing functionality."""
    print("\n=== Parallel Processing Demo ===")
    
    # Create test data
    data = list(range(100))
    chunks = [data[i:i+10] for i in range(0, len(data), 10)]
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [data_processing_task(chunk) for chunk in chunks]
    sequential_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.3f}s")
    
    # Parallel processing with threads
    start_time = time.time()
    parallel_results = parallel_map(
        data_processing_task, 
        chunks, 
        max_workers=4, 
        use_processes=False
    )
    parallel_time = time.time() - start_time
    
    print(f"Parallel processing (threads): {parallel_time:.3f}s")
    
    # Calculate speedup, avoiding division by zero
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Parallel processing was too fast to measure accurately")
    
    # Verify results are the same
    assert sequential_results == parallel_results
    print("Results verified: sequential and parallel outputs match")
    
    # Demonstrate progress callback
    def progress_callback(completed, total):
        if completed % 2 == 0:  # Print every 2nd update
            print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
    
    print("\nParallel processing with progress tracking:")
    parallel_map(
        lambda x: time.sleep(0.1) or x * 2,
        list(range(10)),
        max_workers=3,
        progress_callback=progress_callback
    )


def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration detection."""
    print("\n=== GPU Acceleration Demo ===")
    
    accelerator = GPUAccelerator()
    
    print(f"GPU available: {accelerator.is_available}")
    print(f"Recommended device: {accelerator.recommended_device}")
    
    device_info = accelerator.device_info
    print(f"CUDA available: {device_info['cuda_available']}")
    print(f"MPS available: {device_info['mps_available']}")
    
    if device_info['cuda_devices']:
        for i, device in enumerate(device_info['cuda_devices']):
            print(f"CUDA Device {i}: {device['name']}")
            print(f"  Total memory: {device['memory_total'] / (1024**3):.1f} GB")
    
    # Demonstrate optimal batch size calculation
    batch_size = accelerator.get_optimal_batch_size(
        model_size_mb=100,  # 100MB model
        input_size_mb=1     # 1MB per input
    )
    print(f"Recommended batch size: {batch_size}")
    
    # Demonstrate device context
    with gpu_context() as device:
        print(f"Using device: {device}")
        # Your GPU-accelerated code would go here
        
    # Optimize memory usage
    accelerator.optimize_memory_usage()
    print("GPU memory optimized")


def demonstrate_benchmarking():
    """Demonstrate benchmarking functionality."""
    print("\n=== Benchmarking Demo ===")
    
    runner = get_benchmark_runner()
    
    # Benchmark a simple function
    def test_function(n):
        return sum(i**2 for i in range(n))
    
    print("Running benchmark...")
    result = runner.run_benchmark(
        test_function,
        "sum_of_squares",
        args=(1000,),
        metadata={"input_size": 1000}
    )
    
    print(f"Benchmark: {result.name}")
    print(f"Duration: {result.duration:.4f}s")
    print(f"Memory delta: {result.memory_usage.get('avg_delta_rss', 0):.1f} MB")
    
    # Benchmark multiple functions
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    def fibonacci_iterative(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    benchmarks = [
        (fibonacci, "fibonacci_recursive", (20,), {}),
        (fibonacci_iterative, "fibonacci_iterative", (20,), {})
    ]
    
    print("\nRunning benchmark suite...")
    suite = runner.run_suite(benchmarks, "fibonacci_comparison", save_results=False)
    
    print(f"Suite: {suite.name}")
    print(f"Results: {len(suite.results)}/{suite.summary['total_benchmarks']} successful")
    
    for result in suite.results:
        print(f"  {result.name}: {result.duration:.4f}s")
    
    # Compare performance
    if len(suite.results) == 2:
        recursive_time = suite.results[0].duration
        iterative_time = suite.results[1].duration
        speedup = recursive_time / iterative_time
        print(f"Iterative is {speedup:.2f}x faster than recursive")


def main():
    """Run all performance demonstrations."""
    print("NeuroLite Performance Optimization Examples")
    print("=" * 50)
    
    try:
        demonstrate_lazy_loading()
        demonstrate_caching()
        demonstrate_parallel_processing()
        demonstrate_gpu_acceleration()
        demonstrate_benchmarking()
        
        print("\n" + "=" * 50)
        print("All performance demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()