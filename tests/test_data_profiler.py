"""
Unit tests for DataProfiler class.

Tests the main interface for NeuroLite data profiling.
"""

import pytest
import pandas as pd
import numpy as np

from neurolite import DataProfiler
from neurolite.core.exceptions import InsufficientDataError


class TestDataProfiler:
    """Test cases for DataProfiler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = DataProfiler()
    
    def test_init_default_confidence(self):
        """Test DataProfiler initialization with default confidence."""
        profiler = DataProfiler()
        assert profiler.confidence_threshold == 0.8
        assert profiler.file_detector is not None
        assert profiler.type_detector is not None
        assert profiler.quality_detector is not None
    
    def test_init_custom_confidence(self):
        """Test DataProfiler initialization with custom confidence."""
        profiler = DataProfiler(confidence_threshold=0.9)
        assert profiler.confidence_threshold == 0.9
    
    def test_analyze_dataframe_basic(self):
        """Test basic analysis with DataFrame input."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        results = self.profiler.analyze(df)
        
        # Check that all expected keys are present
        assert 'data_structure' in results
        assert 'quality_metrics' in results
        assert 'column_analysis' in results
        assert 'missing_analysis' in results
        
        # Check data structure
        assert results['data_structure'].structure_type == 'tabular'
        assert results['data_structure'].dimensions == (5, 3)
        
        # Check quality metrics
        assert results['quality_metrics'].completeness == 1.0  # No missing data
        assert results['quality_metrics'].uniqueness == 1.0    # No duplicates
    
    def test_analyze_dataframe_quick(self):
        """Test quick analysis with DataFrame input."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        results = self.profiler.analyze(df, quick=True)
        
        # Check that only basic keys are present
        assert 'data_structure' in results
        assert 'quality_metrics' in results
        assert 'column_analysis' not in results
        assert 'missing_analysis' not in results
    
    def test_analyze_with_missing_data(self):
        """Test analysis with missing data."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', np.nan, 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, np.nan, 5.5]
        })
        
        results = self.profiler.analyze(df)
        
        # Check quality metrics reflect missing data
        assert results['quality_metrics'].completeness < 1.0
        
        # Check missing analysis
        assert results['missing_analysis'].missing_percentage > 0.0
        assert len(results['missing_analysis'].missing_columns) > 0
        assert results['missing_analysis'].imputation_strategy != 'none'
    
    def test_analyze_with_duplicates(self):
        """Test analysis with duplicate data."""
        df = pd.DataFrame({
            'A': [1, 2, 1, 4, 2],
            'B': ['a', 'b', 'a', 'd', 'b'],
            'C': [1.1, 2.2, 1.1, 4.4, 2.2]
        })
        
        results = self.profiler.analyze(df)
        
        # Check quality metrics reflect duplicates
        assert results['quality_metrics'].uniqueness < 1.0
        assert results['quality_metrics'].duplicate_count > 0
    
    def test_analyze_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        df = pd.DataFrame()
        
        with pytest.raises(InsufficientDataError):
            self.profiler.analyze(df)
    
    def test_analyze_insufficient_data_raises_error(self):
        """Test that insufficient data raises appropriate error."""
        df = pd.DataFrame({'A': [1]})  # Only one row
        
        with pytest.raises(InsufficientDataError):
            self.profiler.analyze(df)
    
    def test_column_analysis_types(self):
        """Test that column analysis correctly identifies types."""
        df = pd.DataFrame({
            'integer_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'date_col': pd.date_range('2023-01-01', periods=5)
        })
        
        results = self.profiler.analyze(df)
        
        column_analysis = results['column_analysis']
        
        # Check that different column types are detected
        assert 'integer_col' in column_analysis
        assert 'float_col' in column_analysis
        assert 'string_col' in column_analysis
        assert 'date_col' in column_analysis
        
        # Check that each column has the expected structure
        for col_name, col_info in column_analysis.items():
            assert hasattr(col_info, 'primary_type')
            assert hasattr(col_info, 'subtype')
            assert hasattr(col_info, 'confidence')
            assert 0.0 <= col_info.confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])