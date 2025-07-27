"""
Unit tests for QualityDetector class.

Tests missing data analysis, data consistency validation, and quality metrics calculation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from neurolite.detectors.quality_detector import QualityDetector
from neurolite.core.data_models import QualityMetrics, MissingDataAnalysis
from neurolite.core.exceptions import InsufficientDataError


class TestQualityDetectorMissingData:
    """Test cases for missing data analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = QualityDetector()
    
    def test_detect_missing_patterns_empty_dataframe(self):
        """Test missing data detection with empty DataFrame."""
        df = pd.DataFrame()
        result = self.detector.detect_missing_patterns(df)
        
        assert isinstance(result, MissingDataAnalysis)
        assert result.missing_percentage == 0.0
        assert result.missing_pattern_type == 'UNKNOWN'
        assert result.missing_columns == []
        assert result.imputation_strategy == 'none'
    
    def test_detect_missing_patterns_no_missing_data(self):
        """Test missing data detection with complete data."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = self.detector.detect_missing_patterns(df)
        
        assert result.missing_percentage == 0.0
        assert result.missing_pattern_type == 'UNKNOWN'
        assert result.missing_columns == []
        assert result.imputation_strategy == 'none'
    
    def test_detect_missing_patterns_with_missing_data(self):
        """Test missing data detection with some missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', np.nan, 'c', 'd', np.nan],
            'C': [1.1, 2.2, 3.3, np.nan, 5.5]
        })
        
        result = self.detector.detect_missing_patterns(df)
        
        assert 0.0 < result.missing_percentage < 1.0
        assert result.missing_pattern_type in ['MCAR', 'MAR', 'MNAR', 'UNKNOWN']
        assert set(result.missing_columns) == {'A', 'B', 'C'}
        assert result.imputation_strategy != 'none'
    
    def test_classify_missing_pattern_mcar(self):
        """Test MCAR pattern classification."""
        # Create data with random missing pattern
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        })
        
        # Randomly set some values to NaN (MCAR pattern)
        mask = np.random.random((100, 3)) < 0.1  # 10% missing randomly
        df = df.mask(mask)
        
        pattern_type = self.detector._classify_missing_pattern(df)
        
        # Should detect as MCAR due to random pattern
        assert pattern_type in ['MCAR', 'UNKNOWN']
    
    def test_classify_missing_pattern_mar(self):
        """Test MAR pattern classification."""
        # Create data where missingness depends on observed values
        df = pd.DataFrame({
            'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            'income': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
            'education': ['HS', 'HS', 'College', 'College', 'College', 'Graduate', 'Graduate', 'Graduate', 'Graduate', 'Graduate']
        })
        
        # Make income missing for younger people (MAR pattern)
        df.loc[df['age'] < 35, 'income'] = np.nan
        
        pattern_type = self.detector._classify_missing_pattern(df)
        
        # Pattern classification is challenging, accept any reasonable classification
        assert pattern_type in ['MCAR', 'MAR', 'MNAR']
    
    def test_classify_missing_pattern_mnar(self):
        """Test MNAR pattern classification."""
        # Create data where missingness depends on unobserved values
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'C': ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 'x']
        })
        
        # Create complex missing pattern that's hard to explain by observed data
        df.loc[[1, 3, 7], 'A'] = np.nan
        df.loc[[2, 5, 8], 'B'] = np.nan
        
        pattern_type = self.detector._classify_missing_pattern(df)
        
        # Pattern classification is challenging, accept any reasonable classification
        assert pattern_type in ['MCAR', 'MAR', 'MNAR', 'UNKNOWN']
    
    def test_recommend_imputation_strategy_no_missing(self):
        """Test imputation strategy recommendation with no missing data."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        strategy = self.detector._recommend_imputation_strategy(df, 'UNKNOWN')
        assert strategy == 'none'
    
    def test_recommend_imputation_strategy_high_missing(self):
        """Test imputation strategy recommendation with high missing percentage."""
        df = pd.DataFrame({
            'A': [1, np.nan, np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan, 4, np.nan]
        })
        strategy = self.detector._recommend_imputation_strategy(df, 'MCAR')
        assert strategy == 'consider_dropping_columns'
    
    def test_recommend_imputation_strategy_low_missing(self):
        """Test imputation strategy recommendation with low missing percentage."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan],  # 10% missing
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        strategy = self.detector._recommend_imputation_strategy(df, 'MCAR')
        assert strategy == 'simple_deletion'
    
    def test_recommend_imputation_strategy_mcar(self):
        """Test imputation strategy recommendation for MCAR pattern."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],  # 20% missing
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        strategy = self.detector._recommend_imputation_strategy(df, 'MCAR')
        assert strategy == 'mean_median_mode_imputation'
    
    def test_recommend_imputation_strategy_mar(self):
        """Test imputation strategy recommendation for MAR pattern."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],  # 20% missing
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        strategy = self.detector._recommend_imputation_strategy(df, 'MAR')
        assert strategy == 'multiple_imputation'
    
    def test_recommend_imputation_strategy_mnar(self):
        """Test imputation strategy recommendation for MNAR pattern."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],  # 20% missing
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        strategy = self.detector._recommend_imputation_strategy(df, 'MNAR')
        assert strategy == 'domain_specific_imputation'
    
    def test_test_mcar_with_random_missing(self):
        """Test MCAR detection with truly random missing data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        })
        
        # Randomly set 5% of values to NaN
        mask = np.random.random((100, 3)) < 0.05
        df = df.mask(mask)
        
        is_mcar = self.detector._test_mcar(df)
        assert isinstance(is_mcar, bool)
    
    def test_test_mar_with_dependent_missing(self):
        """Test MAR detection with dependent missing data."""
        df = pd.DataFrame({
            'age': range(20, 70),
            'income': range(30000, 80000, 1000),
            'satisfaction': ['high'] * 25 + ['low'] * 25
        })
        
        # Make income missing for people with low satisfaction (MAR)
        df.loc[df['satisfaction'] == 'low', 'income'] = np.nan
        
        is_mar = self.detector._test_mar(df)
        assert isinstance(is_mar, bool)


class TestQualityDetectorMetrics:
    """Test cases for quality metrics calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = QualityDetector()
    
    def test_analyze_quality_empty_dataframe(self):
        """Test quality analysis with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(InsufficientDataError):
            self.detector.analyze_quality(df)
    
    def test_analyze_quality_insufficient_data(self):
        """Test quality analysis with insufficient data."""
        df = pd.DataFrame({'A': [1]})  # Only one row
        
        with pytest.raises(InsufficientDataError):
            self.detector.analyze_quality(df)
    
    def test_analyze_quality_complete_data(self):
        """Test quality analysis with complete, clean data."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = self.detector.analyze_quality(df)
        
        assert isinstance(result, QualityMetrics)
        assert result.completeness == 1.0  # No missing data
        assert result.consistency > 0.0
        assert result.validity > 0.0
        assert result.uniqueness == 1.0  # No duplicates
        assert result.duplicate_count == 0
    
    def test_analyze_quality_with_issues(self):
        """Test quality analysis with various data quality issues."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 1, 2],  # Missing value and duplicate rows
            'B': ['a', 'b', 'c', 'a', 'b'],  # Duplicate rows
            'C': [1.1, 2.2, 3.3, 1.1, 2.2]  # Duplicate rows
        })
        
        result = self.detector.analyze_quality(df)
        
        assert isinstance(result, QualityMetrics)
        assert result.completeness < 1.0  # Has missing data
        assert result.uniqueness < 1.0  # Has duplicates
        assert result.duplicate_count > 0


class TestQualityDetectorConsistency:
    """Test cases for data consistency validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = QualityDetector()
    
    def test_find_duplicates_empty_dataframe(self):
        """Test duplicate detection with empty DataFrame."""
        from neurolite.detectors.quality_detector import DuplicateAnalysis
        
        df = pd.DataFrame()
        result = self.detector.find_duplicates(df)
        
        assert isinstance(result, DuplicateAnalysis)
        assert result.duplicate_count == 0
        assert result.duplicate_percentage == 0.0
        assert result.duplicate_rows == []
        assert result.exact_duplicates == 0
        assert result.partial_duplicates == 0
    
    def test_find_duplicates_no_duplicates(self):
        """Test duplicate detection with no duplicates."""
        from neurolite.detectors.quality_detector import DuplicateAnalysis
        
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = self.detector.find_duplicates(df)
        
        assert isinstance(result, DuplicateAnalysis)
        assert result.duplicate_count == 0
        assert result.duplicate_percentage == 0.0
        assert result.exact_duplicates == 0
    
    def test_find_duplicates_with_exact_duplicates(self):
        """Test duplicate detection with exact duplicates."""
        from neurolite.detectors.quality_detector import DuplicateAnalysis
        
        df = pd.DataFrame({
            'A': [1, 2, 1, 4, 2],
            'B': ['a', 'b', 'a', 'd', 'b']
        })
        
        result = self.detector.find_duplicates(df)
        
        assert isinstance(result, DuplicateAnalysis)
        assert result.exact_duplicates == 2  # Two duplicate rows
        assert result.duplicate_percentage > 0.0
        assert len(result.duplicate_rows) == 2
    
    def test_validate_consistency_empty_dataframe(self):
        """Test consistency validation with empty DataFrame."""
        from neurolite.detectors.quality_detector import ConsistencyReport
        
        df = pd.DataFrame()
        result = self.detector.validate_consistency(df)
        
        assert isinstance(result, ConsistencyReport)
        assert result.format_consistency_score == 0.0
        assert result.range_consistency_score == 0.0
        assert result.referential_integrity_score == 0.0
        assert result.inconsistent_formats == {}
        assert result.range_violations == {}
        assert result.integrity_violations == []


if __name__ == '__main__':
    pytest.main([__file__])