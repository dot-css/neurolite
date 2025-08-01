"""
Performance benchmarking and regression testing utilities.

Provides tools for measuring and tracking performance metrics
across different components and configurations.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import psutil

from .logger import get_logger
from .config import get_config
from .utils import ensure_dir, format_duration, format_bytes, get_memory_usage
from .performance import get_gpu_accelerator


logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    duration: float
    memory_usage: Dict[str, float]
    gpu_memory_usage: Optional[Dict[str, float]]
    throughput: Optional[float]
    accuracy: Optional[float]
    metadata: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkSuite':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            results=[BenchmarkResult.from_dict(r) for r in data['results']],
            summary=data['summary'],
            timestamp=data['timestamp']
        )


class PerformanceMonitor:
    """
    Real-time performance monitoring during benchmark execution.
    
    Tracks CPU, memory, and GPU usage during benchmark runs.
    """
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize performance monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.measurements = []
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.debug("Started performance monitoring")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop performance monitoring and return summary.
        
        Returns:
            Performance summary statistics
        """
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug("Stopped performance monitoring")
        return self._calculate_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        gpu_accelerator = get_gpu_accelerator()
        
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory_info = get_memory_usage()
                
                measurement = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_rss': memory_info['rss'],
                    'memory_percent': memory_info['percent']
                }
                
                # GPU usage if available
                if gpu_accelerator.is_available:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            measurement['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024)
                            measurement['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024 * 1024)
                    except Exception:
                        pass
                
                with self._lock:
                    self.measurements.append(measurement)
                
                time.sleep(self.interval)
            
            except Exception as e:
                logger.warning(f"Error in performance monitoring: {e}")
                break
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from measurements."""
        if not self.measurements:
            return {}
        
        with self._lock:
            measurements = self.measurements.copy()
        
        # Extract time series data
        cpu_values = [m['cpu_percent'] for m in measurements]
        memory_values = [m['memory_rss'] for m in measurements]
        memory_percent_values = [m['memory_percent'] for m in measurements]
        
        summary = {
            'duration': measurements[-1]['timestamp'] - measurements[0]['timestamp'],
            'samples': len(measurements),
            'cpu': {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'stdev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_rss': {
                'mean': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'stdev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'memory_percent': {
                'mean': statistics.mean(memory_percent_values),
                'max': max(memory_percent_values),
                'min': min(memory_percent_values),
                'stdev': statistics.stdev(memory_percent_values) if len(memory_percent_values) > 1 else 0
            }
        }
        
        # GPU statistics if available
        gpu_allocated_values = [m.get('gpu_memory_allocated', 0) for m in measurements if 'gpu_memory_allocated' in m]
        if gpu_allocated_values:
            summary['gpu_memory_allocated'] = {
                'mean': statistics.mean(gpu_allocated_values),
                'max': max(gpu_allocated_values),
                'min': min(gpu_allocated_values),
                'stdev': statistics.stdev(gpu_allocated_values) if len(gpu_allocated_values) > 1 else 0
            }
        
        return summary


class BenchmarkRunner:
    """
    Main benchmark runner for performance testing.
    
    Executes benchmarks with proper measurement and result collection.
    """
    
    def __init__(self, 
                 results_dir: Optional[Union[str, Path]] = None,
                 warmup_runs: int = 1,
                 measurement_runs: int = 3):
        """
        Initialize benchmark runner.
        
        Args:
            results_dir: Directory to store benchmark results
            warmup_runs: Number of warmup runs before measurement
            measurement_runs: Number of measurement runs for averaging
        """
        self.results_dir = Path(results_dir or get_config().data.cache_dir) / "benchmarks"
        ensure_dir(self.results_dir)
        
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.monitor = PerformanceMonitor()
        
        logger.debug(f"Initialized BenchmarkRunner with {measurement_runs} runs")
    
    @contextmanager
    def benchmark_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for benchmarking code blocks.
        
        Args:
            name: Benchmark name
            metadata: Additional metadata
            
        Yields:
            Dictionary to store benchmark results
        """
        metadata = metadata or {}
        results = {}
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # GPU memory if available
        gpu_start_memory = None
        gpu_accelerator = get_gpu_accelerator()
        if gpu_accelerator.is_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_start_memory = {
                        'allocated': torch.cuda.memory_allocated(),
                        'reserved': torch.cuda.memory_reserved()
                    }
            except Exception:
                pass
        
        try:
            yield results
        finally:
            # Stop monitoring and collect results
            end_time = time.time()
            end_memory = get_memory_usage()
            monitoring_summary = self.monitor.stop_monitoring()
            
            # GPU memory if available
            gpu_end_memory = None
            if gpu_start_memory is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_end_memory = {
                            'allocated': torch.cuda.memory_allocated(),
                            'reserved': torch.cuda.memory_reserved()
                        }
                except Exception:
                    pass
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory['rss'] - start_memory['rss']
            
            gpu_memory_usage = None
            if gpu_start_memory and gpu_end_memory:
                gpu_memory_usage = {
                    'allocated_delta': (gpu_end_memory['allocated'] - gpu_start_memory['allocated']) / (1024 * 1024),
                    'reserved_delta': (gpu_end_memory['reserved'] - gpu_start_memory['reserved']) / (1024 * 1024),
                    'peak_allocated': monitoring_summary.get('gpu_memory_allocated', {}).get('max', 0)
                }
            
            # Store results
            results.update({
                'name': name,
                'duration': duration,
                'memory_usage': {
                    'start_rss': start_memory['rss'],
                    'end_rss': end_memory['rss'],
                    'delta_rss': memory_delta,
                    'peak_rss': monitoring_summary.get('memory_rss', {}).get('max', end_memory['rss'])
                },
                'gpu_memory_usage': gpu_memory_usage,
                'monitoring_summary': monitoring_summary,
                'metadata': metadata,
                'timestamp': start_time
            })
    
    def run_benchmark(self,
                     func: Callable,
                     name: str,
                     args: Tuple = (),
                     kwargs: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """
        Run a single benchmark function.
        
        Args:
            func: Function to benchmark
            name: Benchmark name
            args: Function arguments
            kwargs: Function keyword arguments
            metadata: Additional metadata
            
        Returns:
            Benchmark result
        """
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        logger.info(f"Running benchmark: {name}")
        
        # Warmup runs
        for i in range(self.warmup_runs):
            logger.debug(f"Warmup run {i + 1}/{self.warmup_runs}")
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
        
        # Measurement runs
        durations = []
        memory_deltas = []
        results = []
        
        for i in range(self.measurement_runs):
            logger.debug(f"Measurement run {i + 1}/{self.measurement_runs}")
            
            with self.benchmark_context(f"{name}_run_{i}", metadata) as result:
                try:
                    output = func(*args, **kwargs)
                    result['output'] = output
                    result['success'] = True
                except Exception as e:
                    logger.error(f"Benchmark run failed: {e}")
                    result['error'] = str(e)
                    result['success'] = False
            
            if result['success']:
                durations.append(result['duration'])
                memory_deltas.append(result['memory_usage']['delta_rss'])
                results.append(result)
        
        if not results:
            raise RuntimeError(f"All benchmark runs failed for {name}")
        
        # Calculate summary statistics
        avg_duration = statistics.mean(durations)
        avg_memory_delta = statistics.mean(memory_deltas)
        
        # Calculate throughput if applicable
        throughput = None
        if 'throughput_items' in metadata:
            throughput = metadata['throughput_items'] / avg_duration
        
        # Extract accuracy if available
        accuracy = None
        if 'accuracy' in metadata:
            accuracy = metadata['accuracy']
        elif results and 'output' in results[0] and isinstance(results[0]['output'], dict):
            accuracy = results[0]['output'].get('accuracy')
        
        # GPU memory usage summary
        gpu_memory_usage = None
        if results[0]['gpu_memory_usage']:
            gpu_memory_usage = {
                'avg_allocated_delta': statistics.mean([
                    r['gpu_memory_usage']['allocated_delta'] for r in results
                ]),
                'max_peak_allocated': max([
                    r['gpu_memory_usage']['peak_allocated'] for r in results
                ])
            }
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration=avg_duration,
            memory_usage={
                'avg_delta_rss': avg_memory_delta,
                'max_peak_rss': max([r['memory_usage']['peak_rss'] for r in results])
            },
            gpu_memory_usage=gpu_memory_usage,
            throughput=throughput,
            accuracy=accuracy,
            metadata={
                **metadata,
                'runs': self.measurement_runs,
                'duration_stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
                'memory_stdev': statistics.stdev(memory_deltas) if len(memory_deltas) > 1 else 0
            },
            timestamp=time.time()
        )
        
        logger.info(
            f"Benchmark {name} completed: "
            f"{format_duration(avg_duration)} "
            f"(±{format_duration(benchmark_result.metadata['duration_stdev'])}), "
            f"Memory: {avg_memory_delta:+.1f} MB"
        )
        
        return benchmark_result
    
    def run_suite(self, 
                  benchmarks: List[Tuple[Callable, str, Tuple, Dict[str, Any]]],
                  suite_name: str,
                  save_results: bool = True) -> BenchmarkSuite:
        """
        Run a suite of benchmarks.
        
        Args:
            benchmarks: List of (func, name, args, kwargs) tuples
            suite_name: Suite name
            save_results: Whether to save results to disk
            
        Returns:
            Benchmark suite results
        """
        logger.info(f"Running benchmark suite: {suite_name}")
        start_time = time.time()
        
        results = []
        for func, name, args, kwargs in benchmarks:
            try:
                result = self.run_benchmark(func, name, args, kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
        
        # Calculate suite summary
        if results:
            total_duration = sum(r.duration for r in results)
            avg_duration = statistics.mean([r.duration for r in results])
            total_memory = sum(r.memory_usage.get('avg_delta_rss', 0) for r in results)
            
            summary = {
                'total_benchmarks': len(benchmarks),
                'successful_benchmarks': len(results),
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'total_memory_delta': total_memory,
                'suite_execution_time': time.time() - start_time
            }
        else:
            summary = {
                'total_benchmarks': len(benchmarks),
                'successful_benchmarks': 0,
                'suite_execution_time': time.time() - start_time
            }
        
        suite = BenchmarkSuite(
            name=suite_name,
            results=results,
            summary=summary,
            timestamp=start_time
        )
        
        if save_results:
            self.save_suite(suite)
        
        logger.info(
            f"Benchmark suite {suite_name} completed: "
            f"{len(results)}/{len(benchmarks)} successful, "
            f"Total time: {format_duration(summary['suite_execution_time'])}"
        )
        
        return suite
    
    def save_suite(self, suite: BenchmarkSuite):
        """Save benchmark suite to disk."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(suite.timestamp))
        filename = f"{suite.name}_{timestamp_str}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(suite.to_dict(), f, indent=2)
            logger.info(f"Saved benchmark results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def load_suite(self, filepath: Union[str, Path]) -> BenchmarkSuite:
        """Load benchmark suite from disk."""
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return BenchmarkSuite.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {e}")
            raise
    
    def compare_suites(self, 
                      baseline: BenchmarkSuite,
                      current: BenchmarkSuite,
                      threshold: float = 0.1) -> Dict[str, Any]:
        """
        Compare two benchmark suites for regression detection.
        
        Args:
            baseline: Baseline benchmark suite
            current: Current benchmark suite
            threshold: Regression threshold (10% by default)
            
        Returns:
            Comparison results with regression flags
        """
        comparison = {
            'baseline_suite': baseline.name,
            'current_suite': current.name,
            'baseline_timestamp': baseline.timestamp,
            'current_timestamp': current.timestamp,
            'regressions': [],
            'improvements': [],
            'summary': {}
        }
        
        # Create lookup for baseline results
        baseline_lookup = {r.name: r for r in baseline.results}
        
        total_regressions = 0
        total_improvements = 0
        
        for current_result in current.results:
            if current_result.name not in baseline_lookup:
                continue
            
            baseline_result = baseline_lookup[current_result.name]
            
            # Compare duration
            duration_change = (current_result.duration - baseline_result.duration) / baseline_result.duration
            
            # Compare memory usage
            current_memory = current_result.memory_usage.get('avg_delta_rss', 0)
            baseline_memory = baseline_result.memory_usage.get('avg_delta_rss', 0)
            memory_change = 0
            if baseline_memory != 0:
                memory_change = (current_memory - baseline_memory) / abs(baseline_memory)
            
            # Check for regressions
            is_regression = duration_change > threshold or memory_change > threshold
            is_improvement = duration_change < -threshold or memory_change < -threshold
            
            result_comparison = {
                'name': current_result.name,
                'duration_change': duration_change,
                'memory_change': memory_change,
                'baseline_duration': baseline_result.duration,
                'current_duration': current_result.duration,
                'baseline_memory': baseline_memory,
                'current_memory': current_memory,
                'is_regression': is_regression,
                'is_improvement': is_improvement
            }
            
            if is_regression:
                comparison['regressions'].append(result_comparison)
                total_regressions += 1
            elif is_improvement:
                comparison['improvements'].append(result_comparison)
                total_improvements += 1
        
        comparison['summary'] = {
            'total_compared': len([r for r in current.results if r.name in baseline_lookup]),
            'regressions': total_regressions,
            'improvements': total_improvements,
            'threshold': threshold
        }
        
        return comparison


# Predefined benchmark functions
def benchmark_data_loading(data_path: str, batch_size: int = 32) -> Dict[str, Any]:
    """Benchmark data loading performance."""
    from ..data.loader import DataLoader
    
    loader = DataLoader()
    dataset = loader.load_data(data_path)
    
    # Simulate batch loading
    batches_loaded = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batches_loaded += 1
    
    return {
        'batches_loaded': batches_loaded,
        'total_samples': len(dataset),
        'batch_size': batch_size
    }


def benchmark_preprocessing(data, preprocessor_config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark preprocessing performance."""
    from ..data.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor(**preprocessor_config)
    processed_data = preprocessor.preprocess(data)
    
    return {
        'input_size': len(data) if hasattr(data, '__len__') else 1,
        'output_size': len(processed_data) if hasattr(processed_data, '__len__') else 1,
        'config': preprocessor_config
    }


def benchmark_model_inference(model, test_data, batch_size: int = 32) -> Dict[str, Any]:
    """Benchmark model inference performance."""
    total_samples = len(test_data)
    predictions = []
    
    for i in range(0, total_samples, batch_size):
        batch = test_data[i:i + batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    
    return {
        'total_samples': total_samples,
        'batch_size': batch_size,
        'predictions_count': len(predictions)
    }


# Global benchmark runner instance
_benchmark_runner = None


def get_benchmark_runner() -> BenchmarkRunner:
    """Get global benchmark runner instance."""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = BenchmarkRunner()
    return _benchmark_runner