"""
Performance tests for NeuroLite library to validate 5-second target.

These tests ensure that the library meets performance requirements for various
dataset sizes and types, with a target of 5 seconds for datasets up to 1GB.
"""

import time
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import os

from neurolite.core.data_profiler import DataProfiler
from neurolite.core.performance import (
    LazyDataLoader, SamplingStrategy, ParallelProcessor, 
    OptimizedDataProfiler, MemoryMonitor
)


class TestPerformanceOptimizations:
    """Test performance optimization components."""
    
    def test_lazy_data_loader_csv(self):
        """Test lazy loading for CSV files."""
        # Create test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write('col1,col2,col3,col4,col5\n')
            # Write 10000 rows
            for i in range(10000):
                f.write(f'{i},{i*2},{i*3},{i*4},{i*5}\n')
            temp_file = f.name
        
        try:
            loader = LazyDataLoader(temp_file, chunk_size=1000)
            
            # Test metadata extraction
            metadata = loader.get_metadata()
            assert metadata['total_rows'] == 10000
            assert len(metadata['columns']) == 5
            assert metadata['estimated_chunks'] == 11  # 10000/1000 + 1
            
            # Test iteration
            chunk_count = 0
            total_rows = 0
            for chunk in loader:
                chunk_count += 1
                total_rows += len(chunk)
                assert len(chunk.columns) == 5
                assert len(chunk) <= 1000
            
            assert chunk_count == 10
            assert total_rows == 10000
            
        finally:
            os.unlink(temp_file)
    
    def test_sampling_strategies(self):
        """Test various sampling strategies."""
        # Create test DataFrame
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(10000),
            'feature2': np.random.randn(10000),
            'target': np.random.choice(['A', 'B', 'C'], 10000, p=[0.5, 0.3, 0.2])
        })
        
        # Test random sampling
        sample = SamplingStrategy.random_sample(df, 1000)
        assert len(sample) == 1000
        assert set(sample.columns) == set(df.columns)
        
        # Test stratified sampling
        stratified_sample = SamplingStrategy.stratified_sample(df, 'target', 1000)
        assert len(stratified_sample) <= 1000  # Allow for rounding differences
        assert len(stratified_sample) >= 990   # Should be close to target
        
        # Check that class distribution is approximately maintained
        original_dist = df['target'].value_counts(normalize=True)
        sample_dist = stratified_sample['target'].value_counts(normalize=True)
        
        for class_val in original_dist.index:
            assert abs(original_dist[class_val] - sample_dist[class_val]) < 0.1
        
        # Test systematic sampling
        systematic_sample = SamplingStrategy.systematic_sample(df, 1000)
        assert len(systematic_sample) == 1000
        
        # Test adaptive sample size calculation
        sample_size = SamplingStrategy.adaptive_sample_size(100000, 50, 500)
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size <= 100000
    
    def test_parallel_processing(self):
        """Test parallel processing capabilities."""
        # Create test data chunks
        chunks = [
            pd.DataFrame({'x': np.random.randn(1000), 'y': np.random.randn(1000)})
            for _ in range(4)
        ]
        
        def sum_columns(chunk):
            return {
                'x_sum': chunk['x'].sum(),
                'y_sum': chunk['y'].sum(),
                'count': len(chunk)
            }
        
        processor = ParallelProcessor(max_workers=2)
        
        # Test parallel processing
        start_time = time.time()
        results = processor.process_chunks_parallel(chunks, sum_columns)
        parallel_time = time.time() - start_time
        
        assert len(results) == 4
        assert all('x_sum' in result for result in results)
        assert all('y_sum' in result for result in results)
        assert all(result['count'] == 1000 for result in results)
        
        # Test map-reduce
        def reduce_sums(results_list):
            total_x = sum(r['x_sum'] for r in results_list)
            total_y = sum(r['y_sum'] for r in results_list)
            total_count = sum(r['count'] for r in results_list)
            return {'total_x': total_x, 'total_y': total_y, 'total_count': total_count}
        
        reduced_result = processor.map_reduce_parallel(chunks, sum_columns, reduce_sums)
        assert reduced_result['total_count'] == 4000
        assert 'total_x' in reduced_result
        assert 'total_y' in reduced_result
    
    def test_memory_monitor(self):
        """Test memory monitoring functionality."""
        monitor = MemoryMonitor(max_memory_mb=1000)  # 1GB limit
        
        # Test memory usage tracking
        initial_usage = monitor.get_memory_usage()
        assert initial_usage >= 0
        
        # Create some data to use memory
        large_array = np.random.randn(100000, 10)
        
        # Check memory usage increased
        new_usage = monitor.get_memory_usage()
        assert new_usage > initial_usage
        
        # Clean up
        del large_array
    
    def test_optimized_data_profiler(self):
        """Test optimized data profiler with performance features."""
        # Create test DataFrame
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric1': np.random.randn(5000),
            'numeric2': np.random.randint(0, 100, 5000),
            'categorical': np.random.choice(['A', 'B', 'C', 'D'], 5000),
            'text': [f'text_{i}' for i in range(5000)],
            'missing': np.random.choice([1, 2, np.nan], 5000, p=[0.4, 0.4, 0.2])
        })
        
        profiler = OptimizedDataProfiler(
            chunk_size=1000,
            target_memory_mb=100,
            enable_parallel=True
        )
        
        start_time = time.time()
        results = profiler.analyze_large_dataset(df, sample_for_analysis=True)
        execution_time = time.time() - start_time
        
        # Verify results structure
        assert 'metadata' in results
        assert 'performance' in results
        
        # Check if sampling was used or full analysis
        if 'sample_analysis' in results:
            assert 'sampling_info' in results
        else:
            assert 'full_analysis' in results
        
        # Verify metadata
        metadata = results['metadata']
        assert metadata['total_rows'] == 5000
        assert len(metadata['columns']) == 5
        
        # Verify performance metrics
        performance = results['performance']
        assert 'total_execution_time' in performance
        assert 'memory_used_mb' in performance
        assert performance['processing_mode'] in ['sampled', 'full']
        
        # Performance should be reasonable for this size
        assert execution_time < 10  # Should be much faster than 10 seconds


class TestPerformanceTargets:
    """Test performance targets for various dataset sizes."""
    
    def create_test_dataset(self, rows: int, cols: int) -> pd.DataFrame:
        """Create a test dataset with specified dimensions."""
        np.random.seed(42)
        data = {}
        
        for i in range(cols):
            if i % 4 == 0:
                # Numerical column
                data[f'num_{i}'] = np.random.randn(rows)
            elif i % 4 == 1:
                # Integer column
                data[f'int_{i}'] = np.random.randint(0, 1000, rows)
            elif i % 4 == 2:
                # Categorical column
                data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], rows)
            else:
                # Text column with some missing values
                data[f'text_{i}'] = [
                    f'text_{j}' if np.random.random() > 0.1 else np.nan 
                    for j in range(rows)
                ]
        
        return pd.DataFrame(data)
    
    @pytest.mark.parametrize("rows,cols,expected_time", [
        (1000, 10, 1.0),      # Small dataset - should be very fast
        (10000, 20, 2.0),     # Medium dataset
        (50000, 30, 4.0),     # Large dataset
        (100000, 50, 5.0),    # Very large dataset - target limit
    ])
    def test_performance_targets(self, rows, cols, expected_time):
        """Test that analysis meets performance targets for various dataset sizes."""
        df = self.create_test_dataset(rows, cols)
        
        profiler = DataProfiler(
            confidence_threshold=0.8,
            enable_graceful_degradation=True,
            max_processing_time=expected_time * 2  # Allow 2x expected time as limit
        )
        
        start_time = time.time()
        
        try:
            # Use quick_analyze for performance testing
            result = profiler.quick_analyze(df)
            execution_time = time.time() - start_time
            
            # Verify result is valid
            assert result is not None
            assert hasattr(result, 'execution_time')
            assert result.execution_time > 0
            
            # Check performance target
            print(f"Dataset ({rows}x{cols}): {execution_time:.2f}s (target: {expected_time}s)")
            
            # Allow some flexibility in performance targets
            assert execution_time <= expected_time * 1.5, f"Execution time {execution_time:.2f}s exceeded target {expected_time}s"
            
        except Exception as e:
            pytest.fail(f"Analysis failed for dataset ({rows}x{cols}): {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large datasets."""
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Create a moderately large dataset
        df = self.create_test_dataset(20000, 25)
        
        profiler = DataProfiler(
            confidence_threshold=0.8,
            enable_graceful_degradation=True
        )
        
        # Perform analysis
        result = profiler.quick_analyze(df)
        
        peak_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable (less than 500MB for this test)
        assert memory_used < 500, f"Memory usage {memory_used:.1f}MB is too high"
        
        # Clean up
        del df, result
    
    def test_large_file_performance(self):
        """Test performance with large file processing."""
        # Create a temporary large CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write('id,value1,value2,value3,category,text\n')
            
            # Write 50000 rows (approximately 10MB file)
            for i in range(50000):
                f.write(f'{i},{np.random.randn():.4f},{np.random.randn():.4f},'
                       f'{np.random.randint(0, 1000)},{np.random.choice(["A", "B", "C"])},'
                       f'text_data_{i}\n')
            temp_file = f.name
        
        try:
            profiler = DataProfiler(
                confidence_threshold=0.8,
                enable_graceful_degradation=True,
                max_processing_time=10.0  # 10 second limit
            )
            
            start_time = time.time()
            result = profiler.quick_analyze(temp_file)
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert execution_time < 8.0, f"File processing took {execution_time:.2f}s, too slow"
            
            # Verify result
            assert result is not None
            assert result.file_info.format_type.upper() == 'CSV'
            
            print(f"Large file processing: {execution_time:.2f}s")
            
        finally:
            os.unlink(temp_file)
    
    def test_concurrent_analysis(self):
        """Test performance with concurrent analysis requests."""
        import threading
        
        # Create multiple test datasets
        datasets = [
            self.create_test_dataset(5000, 15) for _ in range(3)
        ]
        
        results = []
        execution_times = []
        
        def analyze_dataset(df, index):
            profiler = DataProfiler(confidence_threshold=0.8)
            start_time = time.time()
            result = profiler.quick_analyze(df)
            execution_time = time.time() - start_time
            
            results.append((index, result))
            execution_times.append(execution_time)
        
        # Run concurrent analyses
        threads = []
        start_time = time.time()
        
        for i, df in enumerate(datasets):
            thread = threading.Thread(target=analyze_dataset, args=(df, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All analyses should complete
        assert len(results) == 3
        assert len(execution_times) == 3
        
        # Concurrent execution should be efficient
        avg_individual_time = sum(execution_times) / len(execution_times)
        print(f"Concurrent analysis - Total: {total_time:.2f}s, Avg individual: {avg_individual_time:.2f}s")
        
        # Total time should be less than sum of individual times (due to parallelism)
        assert total_time < sum(execution_times), "No parallelism benefit observed"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])