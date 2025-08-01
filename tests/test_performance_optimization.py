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