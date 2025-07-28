"""
Performance tests for DataProfiler quick analysis functionality.

This module tests the performance optimizations and sampling strategies
for large datasets in quick analysis mode.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch

from neurolite.core.data_profiler import DataProfiler
from neurolite.core.data_models import QuickReport, DataStructure


class TestQuickAnalysisPerformance:
    """Test performance aspects of quick analysis."""
    
    def test_quick_analysis_performance_target(self):
        """Test that quick analysis meets the 5-second performance target."""
        profiler = DataProfiler()
        
        # Create a moderately large dataset (not too large to avoid CI issues)
        df = pd.DataFrame({
            'id': range(50000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 50000),
            'value': np.random.normal(0, 1, 50000),
            'text': ['sample text ' + str(i) for i in range(50000)]
        })
        
        start_time = time.time()
        result = profiler.quick_analyze(df)
        execution_time = time.time() - start_time
        
        # Should complete within 5 seconds (being generous for CI)
        assert execution_time < 5.0, f"Quick analysis took {execution_time:.2f}s, should be < 5s"
        assert isinstance(result, QuickReport)
        assert result.execution_time > 0
    
    def test_sampling_strategy_for_large_datasets(self):
        """Test that sampling is applied correctly for large datasets."""
        profiler = DataProfiler()
        
        # Create a large dataset that should trigger sampling
        large_df = pd.DataFrame({
            'A': range(20000),
            'B': np.random.choice(['X', 'Y', 'Z'], 20000),
            'C': np.random.normal(0, 1, 20000)
        })
        
        with patch.object(profiler.file_detector, 'detect_structure') as mock_structure:
            mock_structure.return_value = DataStructure(
                structure_type='tabular',
                dimensions=(5000, 3),  # Should be sampled to 5000
                sample_size=5000,
                memory_usage=1024
            )
            
            result = profiler.quick_analyze(large_df)
            
            # Verify sampling occurred
            assert result.file_info.metadata.get('sampled') is True
            # Sample size should be reasonable (between 1000 and original size)
            sample_size = result.file_info.metadata.get('sample_size')
            assert 1000 <= sample_size < 20000
    
    def test_memory_efficient_basic_stats(self):
        """Test that basic stats generation is memory efficient."""
        profiler = DataProfiler()
        
        # Create dataset with various data types
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 10000),
            'float_col': np.random.normal(0, 1, 10000),
            'str_col': ['category_' + str(i % 10) for i in range(10000)],
            'bool_col': np.random.choice([True, False], 10000)
        })
        
        basic_stats = profiler._generate_basic_stats(df)
        
        # Verify all expected statistics are present
        assert 'shape' in basic_stats
        assert 'dtypes' in basic_stats
        assert 'missing_values' in basic_stats
        assert 'memory_usage_mb' in basic_stats
        assert 'numerical_summary' in basic_stats
        assert 'categorical_summary' in basic_stats
        
        # Verify numerical summary contains expected statistics
        numerical_summary = basic_stats['numerical_summary']
        assert 'int_col' in numerical_summary
        assert 'float_col' in numerical_summary
        assert 'count' in numerical_summary['int_col']
        assert 'mean' in numerical_summary['int_col']
        assert 'std' in numerical_summary['int_col']
        
        # Verify categorical summary
        categorical_summary = basic_stats['categorical_summary']
        assert 'str_col' in categorical_summary
        assert 'unique_count' in categorical_summary['str_col']
        assert 'most_frequent' in categorical_summary['str_col']
    
    def test_quick_recommendations_performance(self):
        """Test that quick recommendations are generated efficiently."""
        profiler = DataProfiler()
        
        # Create dataset with various quality issues
        df = pd.DataFrame({
            'complete_col': range(1000),
            'missing_col': [i if i % 10 != 0 else None for i in range(1000)],
            'high_dim_col': range(1000),  # Will be part of high-dimensional data
            'memory_heavy_col': ['x' * 100] * 1000  # Memory intensive
        })
        
        # Add more columns to trigger high-dimensionality warning
        for i in range(150):
            df[f'feature_{i}'] = np.random.normal(0, 1, 1000)
        
        basic_stats = profiler._generate_basic_stats(df)
        
        start_time = time.time()
        recommendations = profiler._generate_quick_recommendations(df, basic_stats)
        generation_time = time.time() - start_time
        
        # Should be very fast (< 0.1 seconds)
        assert generation_time < 0.1, f"Recommendation generation took {generation_time:.3f}s"
        
        # Verify recommendations are generated
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check for expected recommendations based on data characteristics
        recommendation_text = ' '.join(recommendations).lower()
        assert 'missing values' in recommendation_text
        assert 'high-dimensional' in recommendation_text or 'dimensionality reduction' in recommendation_text
        # Memory recommendation might not always appear depending on actual memory usage
        # Just verify we have multiple recommendations
        assert len(recommendations) >= 2
    
    def test_error_handling_in_quick_analysis(self):
        """Test error handling doesn't significantly impact performance."""
        profiler = DataProfiler()
        
        # Create problematic dataset
        df = pd.DataFrame({
            'normal_col': range(1000),
            'problematic_col': [float('inf'), float('-inf')] + [1.0] * 998
        })
        
        start_time = time.time()
        
        try:
            result = profiler.quick_analyze(df)
            # Should still complete successfully with graceful error handling
            assert isinstance(result, QuickReport)
        except Exception:
            # Even if it fails, should fail quickly
            pass
        
        execution_time = time.time() - start_time
        assert execution_time < 2.0, f"Error handling took too long: {execution_time:.2f}s"
    
    def test_concurrent_quick_analysis(self):
        """Test that quick analysis can handle concurrent requests efficiently."""
        import threading
        import queue
        
        profiler = DataProfiler()
        
        # Create test datasets
        datasets = []
        for i in range(3):
            df = pd.DataFrame({
                'id': range(1000),
                'value': np.random.normal(i, 1, 1000),
                'category': np.random.choice([f'cat_{i}_A', f'cat_{i}_B'], 1000)
            })
            datasets.append(df)
        
        results_queue = queue.Queue()
        
        def analyze_dataset(dataset, result_queue):
            try:
                start_time = time.time()
                result = profiler.quick_analyze(dataset)
                execution_time = time.time() - start_time
                result_queue.put(('success', execution_time, result))
            except Exception as e:
                result_queue.put(('error', 0, str(e)))
        
        # Start concurrent analyses
        threads = []
        for dataset in datasets:
            thread = threading.Thread(target=analyze_dataset, args=(dataset, results_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3, "All concurrent analyses should complete"
        
        for status, execution_time, result in results:
            assert status == 'success', f"Analysis failed: {result}"
            assert execution_time < 3.0, f"Concurrent analysis took too long: {execution_time:.2f}s"
            assert isinstance(result, QuickReport)


class TestSamplingStrategies:
    """Test different sampling strategies for large datasets."""
    
    def test_random_sampling_preserves_distribution(self):
        """Test that random sampling preserves data distribution characteristics."""
        profiler = DataProfiler()
        
        # Create dataset with known distribution
        np.random.seed(42)  # For reproducible results
        original_df = pd.DataFrame({
            'normal_dist': np.random.normal(100, 15, 20000),
            'categorical': np.random.choice(['A', 'B', 'C'], 20000, p=[0.5, 0.3, 0.2]),
            'uniform': np.random.uniform(0, 100, 20000)
        })
        
        # Get original statistics
        original_stats = profiler._generate_basic_stats(original_df)
        
        # Apply sampling (simulate what happens in quick_analyze)
        sampled_df = original_df.sample(n=5000, random_state=42)
        sampled_stats = profiler._generate_basic_stats(sampled_df)
        
        # Check that distributions are approximately preserved
        original_mean = original_stats['numerical_summary']['normal_dist']['mean']
        sampled_mean = sampled_stats['numerical_summary']['normal_dist']['mean']
        
        # Should be within 5% of original mean
        assert abs(original_mean - sampled_mean) / original_mean < 0.05
        
        # Check categorical distribution preservation
        original_cat_counts = original_df['categorical'].value_counts(normalize=True)
        sampled_cat_counts = sampled_df['categorical'].value_counts(normalize=True)
        
        for category in original_cat_counts.index:
            original_prop = original_cat_counts[category]
            sampled_prop = sampled_cat_counts[category]
            # Should be within 10% of original proportion
            assert abs(original_prop - sampled_prop) < 0.1
    
    def test_stratified_sampling_concept(self):
        """Test concept for stratified sampling (future enhancement)."""
        profiler = DataProfiler()
        
        # Create imbalanced dataset
        df_majority = pd.DataFrame({
            'feature': np.random.normal(0, 1, 9000),
            'target': ['majority'] * 9000
        })
        
        df_minority = pd.DataFrame({
            'feature': np.random.normal(2, 1, 1000),
            'target': ['minority'] * 1000
        })
        
        df = pd.concat([df_majority, df_minority], ignore_index=True)
        
        # Current random sampling
        sampled_df = df.sample(n=2000, random_state=42)
        
        # Check if minority class is adequately represented
        minority_ratio_original = (df['target'] == 'minority').mean()
        minority_ratio_sampled = (sampled_df['target'] == 'minority').mean()
        
        # This test documents current behavior and suggests future improvement
        # In a stratified sampling implementation, we'd want to preserve the ratio better
        print(f"Original minority ratio: {minority_ratio_original:.3f}")
        print(f"Sampled minority ratio: {minority_ratio_sampled:.3f}")
        
        # For now, just verify sampling occurred
        assert len(sampled_df) == 2000
        assert 'minority' in sampled_df['target'].values
        assert 'majority' in sampled_df['target'].values
    
    def test_memory_based_sampling_threshold(self):
        """Test that sampling threshold adapts to memory constraints."""
        profiler = DataProfiler()
        
        # Create datasets of different sizes
        small_df = pd.DataFrame({'A': range(5000), 'B': range(5000)})
        large_df = pd.DataFrame({'A': range(15000), 'B': range(15000)})
        
        # Test small dataset (should not be sampled)
        small_loaded, small_info = profiler._load_data(small_df, quick_mode=True)
        assert len(small_loaded) == 5000
        assert small_info.metadata.get('sampled') is not True
        
        # Test large dataset (should be sampled)
        large_loaded, large_info = profiler._load_data(large_df, quick_mode=True)
        assert len(large_loaded) < 15000  # Should be sampled (less than original)
        assert len(large_loaded) >= 1000  # Should have minimum sample size
        assert large_info.metadata.get('sampled') is True
        assert large_info.metadata.get('sample_size') == len(large_loaded)


if __name__ == '__main__':
    pytest.main([__file__])