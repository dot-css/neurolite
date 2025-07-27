"""
Unit tests for DataTypeDetector class.

Tests cover numerical, categorical, temporal, and text data classification
with various edge cases and data patterns.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from neurolite.detectors.data_type_detector import DataTypeDetector
from neurolite.core.data_models import (
    ColumnType, NumericalAnalysis, CategoricalAnalysis, TemporalAnalysis, TextAnalysis
)
from neurolite.core.exceptions import NeuroLiteException


class TestDataTypeDetector:
    """Test suite for DataTypeDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_init(self):
        """Test DataTypeDetector initialization."""
        assert self.detector.confidence_threshold == 0.8
        assert self.detector.max_categorical_cardinality == 50
        assert self.detector.min_samples_for_analysis == 10


class TestNumericalAnalysis:
    """Test suite for numerical data analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_analyze_numerical_integer_discrete(self):
        """Test analysis of discrete integer data."""
        # Create discrete integer data
        data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        
        result = self.detector.analyze_numerical(data)
        
        assert isinstance(result, NumericalAnalysis)
        assert result.data_type == 'integer'
        assert result.is_continuous == False
        assert result.range_min == 1.0
        assert result.range_max == 3.0
        assert result.outlier_count >= 0
        assert result.distribution_type in ['normal', 'uniform', 'skewed_left', 'skewed_right', 'multimodal', 'unknown']
    
    def test_analyze_numerical_float_continuous(self):
        """Test analysis of continuous float data."""
        # Create continuous float data
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 100))
        
        result = self.detector.analyze_numerical(data)
        
        assert isinstance(result, NumericalAnalysis)
        assert result.data_type == 'float'
        assert result.is_continuous == True
        assert result.range_min < result.range_max
        assert result.outlier_count >= 0
    
    def test_analyze_numerical_with_outliers(self):
        """Test numerical analysis with outliers."""
        # Create data with obvious outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        
        result = self.detector.analyze_numerical(data)
        
        assert result.outlier_count > 0
        assert result.range_min == 1.0
        assert result.range_max == 100.0
    
    def test_analyze_numerical_normal_distribution(self):
        """Test detection of normal distribution."""
        # Create normally distributed data
        np.random.seed(42)
        data = pd.Series(np.random.normal(50, 10, 1000))
        
        result = self.detector.analyze_numerical(data)
        
        # Should detect normal distribution with large sample
        assert result.distribution_type in ['normal', 'unknown']
    
    def test_analyze_numerical_uniform_distribution(self):
        """Test detection of uniform distribution."""
        # Create uniformly distributed data
        np.random.seed(42)
        data = pd.Series(np.random.uniform(0, 100, 1000))
        
        result = self.detector.analyze_numerical(data)
        
        assert result.is_continuous == True
        assert result.range_min >= 0
        assert result.range_max <= 100
    
    def test_analyze_numerical_skewed_data(self):
        """Test detection of skewed distribution."""
        # Create right-skewed data
        np.random.seed(42)
        data = pd.Series(np.random.exponential(2, 1000))
        
        result = self.detector.analyze_numerical(data)
        
        assert result.is_continuous == True
        # Exponential distribution should be detected as skewed or unknown
        assert result.distribution_type in ['skewed_right', 'unknown']
    
    def test_analyze_numerical_with_missing_values(self):
        """Test numerical analysis with missing values."""
        data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        
        result = self.detector.analyze_numerical(data)
        
        # Should analyze only non-missing values
        assert result.range_min == 1.0
        assert result.range_max == 10.0
    
    def test_analyze_numerical_non_numerical_raises_error(self):
        """Test that non-numerical data raises error."""
        data = pd.Series(['a', 'b', 'c', 'd', 'e'])
        
        with pytest.raises(NeuroLiteException):
            self.detector.analyze_numerical(data)
    
    def test_is_numerical_true_cases(self):
        """Test _is_numerical method with valid numerical data."""
        # Integer data
        assert self.detector._is_numerical(pd.Series([1, 2, 3, 4, 5]))
        
        # Float data
        assert self.detector._is_numerical(pd.Series([1.1, 2.2, 3.3, 4.4, 5.5]))
        
        # Mixed int/float
        assert self.detector._is_numerical(pd.Series([1, 2.5, 3, 4.7, 5]))
        
        # Numeric strings
        assert self.detector._is_numerical(pd.Series(['1', '2', '3', '4', '5']))
    
    def test_is_numerical_false_cases(self):
        """Test _is_numerical method with non-numerical data."""
        # Text data
        assert not self.detector._is_numerical(pd.Series(['a', 'b', 'c']))
        
        # Mixed text and numbers
        assert not self.detector._is_numerical(pd.Series(['1', 'b', '3']))
        
        # Boolean data
        assert not self.detector._is_numerical(pd.Series([True, False, True]))
    
    def test_determine_numerical_type_integer(self):
        """Test integer type detection."""
        # All whole numbers within integer range
        data = pd.Series([1, 2, 3, 4, 5])
        assert self.detector._determine_numerical_type(data) == 'integer'
        
        # Large integers within range
        data = pd.Series([1000000, 2000000, 3000000])
        assert self.detector._determine_numerical_type(data) == 'integer'
    
    def test_determine_numerical_type_float(self):
        """Test float type detection."""
        # Decimal numbers
        data = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert self.detector._determine_numerical_type(data) == 'float'
        
        # Very large numbers (outside integer range)
        data = pd.Series([2**32, 2**33, 2**34])
        assert self.detector._determine_numerical_type(data) == 'float'
        
        # Mix of integers and floats with decimals
        data = pd.Series([1, 2.5, 3, 4.7])
        assert self.detector._determine_numerical_type(data) == 'float'
    
    def test_is_continuous_true_cases(self):
        """Test continuous data detection."""
        # High unique ratio
        data = pd.Series(np.random.random(100))
        assert self.detector._is_continuous(data) == True
        
        # Float data with decimals
        data = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1])
        assert self.detector._is_continuous(data) == True
    
    def test_is_continuous_false_cases(self):
        """Test discrete data detection."""
        # Low unique ratio
        data = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1])
        assert self.detector._is_continuous(data) == False
        
        # Few unique values
        data = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        assert self.detector._is_continuous(data) == False
    
    def test_count_outliers_iqr_method(self):
        """Test outlier counting using IQR method."""
        # Data with clear outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 5])
        outlier_count = self.detector._count_outliers(data)
        assert outlier_count > 0
        
        # Normal data without outliers
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        outlier_count = self.detector._count_outliers(data)
        assert outlier_count >= 0  # May or may not have outliers depending on distribution
    
    def test_classify_numerical_column(self):
        """Test classification of numerical columns."""
        # Integer discrete data
        data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        result = self.detector._classify_numerical(data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'numerical'
        assert 'integer' in result.subtype
        assert 'discrete' in result.subtype
        assert 0.8 <= result.confidence <= 1.0
        assert 'range' in result.properties
        assert 'distribution' in result.properties
        assert 'outlier_count' in result.properties
        assert 'is_continuous' in result.properties
    
    def test_classify_columns_mixed_types(self):
        """Test classification of DataFrame with mixed column types."""
        df = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            'categories': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'text': ['hello', 'world', 'test', 'data', 'analysis', 'machine', 'learning', 'python', 'pandas', 'numpy']
        })
        
        result = self.detector.classify_columns(df)
        
        assert len(result) == 4
        assert all(isinstance(col_type, ColumnType) for col_type in result.values())
        
        # Check that numerical columns are detected
        assert result['integers'].primary_type == 'numerical'
        assert result['floats'].primary_type == 'numerical'
    
    def test_classify_columns_empty_dataframe(self):
        """Test classification of empty DataFrame raises error."""
        df = pd.DataFrame()
        
        with pytest.raises(NeuroLiteException):
            self.detector.classify_columns(df)
    
    def test_classify_single_column_insufficient_data(self):
        """Test classification with insufficient data."""
        # Less than minimum samples
        data = pd.Series([1, 2, 3])  # Only 3 samples, less than min_samples_for_analysis
        
        result = self.detector._classify_single_column(data)
        
        assert result.primary_type == 'text'
        assert result.subtype == 'insufficient_data'
        assert result.confidence == 0.1
        assert result.properties['sample_size'] == 3


class TestCategoricalAnalysis:
    """Test suite for categorical data analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_analyze_categorical_nominal(self):
        """Test analysis of nominal categorical data."""
        # Create nominal categorical data
        data = pd.Series(['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'])
        
        result = self.detector.analyze_categorical(data)
        
        assert isinstance(result, CategoricalAnalysis)
        assert result.category_type == 'nominal'
        assert result.cardinality == 3
        assert set(result.unique_values) == {'red', 'blue', 'green'}
        assert result.encoding_recommendation == 'one_hot_encoding'
        assert len(result.frequency_distribution) == 3
    
    def test_analyze_categorical_ordinal(self):
        """Test analysis of ordinal categorical data."""
        # Create ordinal categorical data
        data = pd.Series(['low', 'medium', 'high', 'low', 'medium', 'high', 'low', 'medium', 'high', 'low'])
        
        result = self.detector.analyze_categorical(data)
        
        assert isinstance(result, CategoricalAnalysis)
        assert result.category_type == 'ordinal'
        assert result.cardinality == 3
        assert set(result.unique_values) == {'low', 'medium', 'high'}
        assert result.encoding_recommendation == 'ordinal_encoding'
    
    def test_analyze_categorical_binary(self):
        """Test analysis of binary categorical data."""
        # Create binary categorical data
        data = pd.Series(['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'])
        
        result = self.detector.analyze_categorical(data)
        
        assert isinstance(result, CategoricalAnalysis)
        assert result.cardinality == 2
        assert set(result.unique_values) == {'yes', 'no'}
        assert result.encoding_recommendation == 'binary_encoding'
    
    def test_analyze_categorical_high_cardinality(self):
        """Test analysis of high cardinality categorical data."""
        # Create high cardinality categorical data
        data = pd.Series([f'category_{i}' for i in range(15)] * 2)  # 15 unique categories
        
        result = self.detector.analyze_categorical(data)
        
        assert isinstance(result, CategoricalAnalysis)
        assert result.cardinality == 15
        assert result.encoding_recommendation == 'target_encoding'
    
    def test_analyze_categorical_with_missing_values(self):
        """Test categorical analysis with missing values."""
        data = pd.Series(['A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'B', 'A', 'B'])
        
        result = self.detector.analyze_categorical(data)
        
        # Should analyze only non-missing values
        assert result.cardinality == 2
        assert set(result.unique_values) == {'A', 'B'}
    
    def test_analyze_categorical_non_categorical_raises_error(self):
        """Test that non-categorical data raises error."""
        # High cardinality, high unique ratio data
        data = pd.Series([f'unique_{i}' for i in range(100)])
        
        with pytest.raises(NeuroLiteException):
            self.detector.analyze_categorical(data)
    
    def test_is_categorical_true_cases(self):
        """Test _is_categorical method with valid categorical data."""
        # Low cardinality data
        assert self.detector._is_categorical(pd.Series(['A', 'B', 'A', 'B', 'A', 'B']))
        
        # Pandas categorical dtype
        cat_data = pd.Categorical(['A', 'B', 'A', 'B'])
        assert self.detector._is_categorical(pd.Series(cat_data))
        
        # Repeated values
        assert self.detector._is_categorical(pd.Series(['red', 'blue', 'red', 'blue', 'red', 'blue']))
    
    def test_is_categorical_false_cases(self):
        """Test _is_categorical method with non-categorical data."""
        # High cardinality data
        assert not self.detector._is_categorical(pd.Series([f'unique_{i}' for i in range(100)]))
        
        # High unique ratio
        assert not self.detector._is_categorical(pd.Series([f'val_{i}' for i in range(20)]))
        
        # Numerical data
        assert not self.detector._is_categorical(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    def test_determine_categorical_type_ordinal_patterns(self):
        """Test ordinal pattern detection."""
        # Size pattern
        data = pd.Series(['small', 'medium', 'large'])
        assert self.detector._determine_categorical_type(data) == 'ordinal'
        
        # Quality pattern
        data = pd.Series(['poor', 'fair', 'good', 'excellent'])
        assert self.detector._determine_categorical_type(data) == 'ordinal'
        
        # Frequency pattern
        data = pd.Series(['never', 'rarely', 'sometimes', 'often', 'always'])
        assert self.detector._determine_categorical_type(data) == 'ordinal'
        
        # Numeric strings
        data = pd.Series(['1', '2', '3', '4', '5'])
        assert self.detector._determine_categorical_type(data) == 'ordinal'
    
    def test_determine_categorical_type_nominal(self):
        """Test nominal category detection."""
        # Color names
        data = pd.Series(['red', 'blue', 'green', 'yellow'])
        assert self.detector._determine_categorical_type(data) == 'nominal'
        
        # Random categories
        data = pd.Series(['apple', 'banana', 'cherry'])
        assert self.detector._determine_categorical_type(data) == 'nominal'
    
    def test_recommend_encoding_strategies(self):
        """Test encoding recommendation logic."""
        # Binary encoding for 2 categories
        assert self.detector._recommend_encoding(2, 'nominal') == 'binary_encoding'
        
        # One-hot encoding for low cardinality
        assert self.detector._recommend_encoding(5, 'nominal') == 'one_hot_encoding'
        
        # Ordinal encoding for ordinal data
        assert self.detector._recommend_encoding(5, 'ordinal') == 'ordinal_encoding'
        
        # Target encoding for high cardinality
        assert self.detector._recommend_encoding(20, 'nominal') == 'target_encoding'
    
    def test_classify_categorical_column(self):
        """Test classification of categorical columns."""
        # Nominal categorical data
        data = pd.Series(['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue'])
        result = self.detector._classify_categorical(data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'categorical'
        assert 'nominal' in result.subtype
        assert 'cardinality_3' in result.subtype
        assert 0.6 <= result.confidence <= 1.0
        assert 'cardinality' in result.properties
        assert 'category_type' in result.properties
        assert 'encoding_recommendation' in result.properties
        assert 'most_frequent' in result.properties


class TestTemporalAnalysis:
    """Test suite for temporal data analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_analyze_temporal_daily_data(self):
        """Test analysis of daily temporal data."""
        # Create daily date series
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = pd.Series(dates.strftime('%Y-%m-%d'))
        
        result = self.detector.analyze_temporal(data)
        
        assert isinstance(result, TemporalAnalysis)
        assert result.datetime_format == '%Y-%m-%d'
        assert result.frequency == 'D'
        assert result.has_seasonality == False  # Less than 2 years
        assert result.has_trend == True
        assert result.is_stationary in [True, False]
        assert result.time_range[0] < result.time_range[1]
    
    def test_analyze_temporal_monthly_data(self):
        """Test analysis of monthly temporal data."""
        # Create monthly date series
        dates = pd.date_range('2020-01-01', periods=36, freq='M')
        data = pd.Series(dates.strftime('%Y-%m-%d'))
        
        result = self.detector.analyze_temporal(data)
        
        assert isinstance(result, TemporalAnalysis)
        assert result.frequency == 'M'
        assert result.has_seasonality == True  # More than 2 years
        assert result.has_trend == True
    
    def test_analyze_temporal_datetime_with_time(self):
        """Test analysis of datetime data with time component."""
        # Create datetime series with time
        dates = pd.date_range('2023-01-01 10:00:00', periods=24, freq='H')
        data = pd.Series(dates.strftime('%Y-%m-%d %H:%M:%S'))
        
        result = self.detector.analyze_temporal(data)
        
        assert isinstance(result, TemporalAnalysis)
        assert result.datetime_format == '%Y-%m-%d %H:%M:%S'
        assert result.frequency == 'H'
    
    def test_analyze_temporal_different_formats(self):
        """Test analysis of different datetime formats."""
        # Test various formats
        formats_and_data = [
            ('%m/%d/%Y', ['01/15/2023', '01/16/2023', '01/17/2023']),
            ('%d/%m/%Y', ['15/01/2023', '16/01/2023', '17/01/2023']),
            ('%Y%m%d', ['20230115', '20230116', '20230117']),
        ]
        
        for expected_format, date_strings in formats_and_data:
            data = pd.Series(date_strings * 5)  # Repeat to have enough data
            result = self.detector.analyze_temporal(data)
            
            assert result.datetime_format == expected_format
    
    def test_analyze_temporal_with_missing_values(self):
        """Test temporal analysis with missing values."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.Series(dates.strftime('%Y-%m-%d'))
        data.iloc[2] = np.nan
        data.iloc[5] = np.nan
        
        result = self.detector.analyze_temporal(data)
        
        # Should analyze only non-missing values
        assert isinstance(result, TemporalAnalysis)
        assert result.datetime_format == '%Y-%m-%d'
    
    def test_analyze_temporal_non_temporal_raises_error(self):
        """Test that non-temporal data raises error."""
        data = pd.Series(['not', 'a', 'date', 'series', 'at', 'all'])
        
        with pytest.raises(NeuroLiteException):
            self.detector.analyze_temporal(data)
    
    def test_is_temporal_true_cases(self):
        """Test _is_temporal method with valid temporal data."""
        # Standard date format
        assert self.detector._is_temporal(pd.Series(['2023-01-01', '2023-01-02', '2023-01-03']))
        
        # Datetime format
        assert self.detector._is_temporal(pd.Series(['2023-01-01 10:00:00', '2023-01-01 11:00:00']))
        
        # Different format
        assert self.detector._is_temporal(pd.Series(['01/15/2023', '01/16/2023', '01/17/2023']))
        
        # ISO format
        assert self.detector._is_temporal(pd.Series(['2023-01-01T10:00:00', '2023-01-01T11:00:00']))
    
    def test_is_temporal_false_cases(self):
        """Test _is_temporal method with non-temporal data."""
        # Text data
        assert not self.detector._is_temporal(pd.Series(['hello', 'world', 'test']))
        
        # Numbers
        assert not self.detector._is_temporal(pd.Series([1, 2, 3, 4, 5]))
        
        # Mixed data
        assert not self.detector._is_temporal(pd.Series(['2023-01-01', 'not a date', '2023-01-03']))
    
    def test_detect_datetime_format(self):
        """Test datetime format detection."""
        format_tests = [
            ('2023-01-15', '%Y-%m-%d'),
            ('2023-01-15 10:30:00', '%Y-%m-%d %H:%M:%S'),
            ('01/15/2023', '%m/%d/%Y'),
            ('15/01/2023', '%d/%m/%Y'),
            ('20230115', '%Y%m%d'),
            ('2023-01-15T10:30:00', '%Y-%m-%dT%H:%M:%S'),
        ]
        
        for date_string, expected_format in format_tests:
            result = self.detector._detect_datetime_format(date_string)
            assert result == expected_format
    
    def test_detect_datetime_format_unknown(self):
        """Test datetime format detection for unknown formats."""
        result = self.detector._detect_datetime_format('not a date')
        assert result == 'unknown'
    
    def test_detect_frequency(self):
        """Test frequency detection."""
        # Daily frequency
        daily_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        assert self.detector._detect_frequency(daily_dates) == 'D'
        
        # Weekly frequency
        weekly_dates = pd.date_range('2023-01-01', periods=10, freq='W')
        assert self.detector._detect_frequency(weekly_dates) == 'W'
        
        # Monthly frequency
        monthly_dates = pd.date_range('2023-01-01', periods=10, freq='M')
        assert self.detector._detect_frequency(monthly_dates) == 'M'
        
        # Yearly frequency
        yearly_dates = pd.date_range('2020-01-01', periods=5, freq='Y')
        assert self.detector._detect_frequency(yearly_dates) == 'Y'
    
    def test_detect_frequency_insufficient_data(self):
        """Test frequency detection with insufficient data."""
        # Less than 3 data points
        dates = pd.date_range('2023-01-01', periods=2, freq='D')
        assert self.detector._detect_frequency(dates) is None
    
    def test_detect_seasonality(self):
        """Test seasonality detection."""
        # Short time series (no seasonality)
        short_dates = pd.date_range('2023-01-01', periods=12, freq='M')
        assert self.detector._detect_seasonality(short_dates) == False
        
        # Long time series (potential seasonality)
        long_dates = pd.date_range('2020-01-01', periods=36, freq='M')
        assert self.detector._detect_seasonality(long_dates) == True
    
    def test_detect_trend(self):
        """Test trend detection."""
        # Sequential dates should show trend
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        assert self.detector._detect_trend(dates) == True
        
        # Insufficient data
        short_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        assert self.detector._detect_trend(short_dates) == False
    
    def test_test_stationarity(self):
        """Test stationarity testing."""
        # Regular sequence should be stationary
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        result = self.detector._test_stationarity(dates)
        assert isinstance(result, bool)
        
        # Insufficient data should be considered stationary
        short_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        assert self.detector._test_stationarity(short_dates) == True
    
    def test_classify_temporal_column(self):
        """Test classification of temporal columns."""
        # Daily data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = pd.Series(dates.strftime('%Y-%m-%d'))
        result = self.detector._classify_temporal(data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'temporal'
        assert 'datetime' in result.subtype
        assert 'freq_D' in result.subtype
        assert result.confidence == 0.9
        assert 'datetime_format' in result.properties
        assert 'frequency' in result.properties
        assert 'has_seasonality' in result.properties
        assert 'has_trend' in result.properties
        assert 'is_stationary' in result.properties
        assert 'time_span_days' in result.properties


class TestTextAnalysis:
    """Test suite for text data analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
    
    def test_analyze_text_natural_language(self):
        """Test analysis of natural language text."""
        # Create natural language text data
        data = pd.Series([
            'This is a sample sentence with common English words.',
            'Natural language processing is an important field in AI.',
            'Machine learning algorithms can analyze text data effectively.',
            'The quick brown fox jumps over the lazy dog.',
            'Data science involves extracting insights from various data types.',
            'Text analysis helps understand patterns in written communication.',
            'Python is a popular programming language for data analysis.',
            'Statistical methods are used to validate research findings.',
            'Deep learning models can process complex text structures.',
            'Information retrieval systems help find relevant documents.'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.text_type == 'natural_language'
        assert result.language == 'en'
        assert result.encoding in ['ascii', 'utf-8']
        assert result.avg_length > 20
        assert result.unique_ratio > 0.8  # Each sentence is unique
        assert result.contains_special_chars == True  # Contains punctuation
        assert result.contains_numbers == False  # No numbers in these sentences
        assert result.readability_score is not None
        assert 0 <= result.readability_score <= 100
    
    def test_analyze_text_categorical_text(self):
        """Test analysis of categorical text data."""
        # Create categorical text data
        data = pd.Series([
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.text_type == 'categorical_text'
        assert result.unique_ratio < 0.5  # Low unique ratio
        assert result.avg_length < 50  # Short text
        assert result.readability_score is None  # No readability for categorical
    
    def test_analyze_text_structured_text(self):
        """Test analysis of structured text data."""
        # Create structured text data (email addresses)
        data = pd.Series([
            'user1@example.com', 'user2@example.com', 'user3@example.com',
            'admin@company.org', 'support@service.net', 'info@business.co',
            'contact@website.com', 'sales@store.com', 'help@platform.io',
            'team@startup.ai', 'dev@tech.com', 'data@analytics.com'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.text_type == 'structured_text'
        assert result.contains_special_chars == True  # Contains @ and .
        assert result.unique_ratio > 0.8  # Each email is unique
        assert result.readability_score is None  # No readability for structured text
    
    def test_analyze_text_mixed_type(self):
        """Test analysis of mixed text data."""
        # Create mixed text data
        data = pd.Series([
            'This is a natural language sentence.',
            'CODE123',
            'user@example.com',
            'Another sentence with words and meaning.',
            'ID-456-XYZ',
            'Short text',
            'A longer sentence that contains multiple words and punctuation.',
            'CATEGORY_A',
            'Final sentence in natural language.',
            'STRUCT_789'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.text_type in ['mixed', 'natural_language']  # Could be either
        assert result.unique_ratio == 1.0  # All unique
        assert result.contains_special_chars == True
    
    def test_analyze_text_with_numbers(self):
        """Test analysis of text containing numbers."""
        data = pd.Series([
            'Product 123 is available',
            'Order number 456789',
            'Version 2.1.0 released',
            'Price is $29.99',
            'Call 555-1234 for support',
            'Address: 123 Main St',
            'Year 2023 statistics',
            'Model ABC-789',
            'Quantity: 15 items',
            'Code: XYZ123'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.contains_numbers == True
        assert result.contains_special_chars == True
    
    def test_analyze_text_different_languages(self):
        """Test language detection for different languages."""
        # Test various languages
        language_tests = [
            (['Hello world', 'This is English text', 'Natural language'], 'en'),
            (['Привет мир', 'Это русский текст', 'Естественный язык'], 'ru'),
            (['Hola mundo', 'Este es texto español', 'Lenguaje natural'], 'es'),
            (['Bonjour monde', 'Ceci est du texte français', 'Langage naturel'], 'fr'),
            (['Hallo Welt', 'Das ist deutscher Text', 'Natürliche Sprache'], 'de'),
        ]
        
        for texts, expected_lang in language_tests:
            data = pd.Series(texts * 4)  # Repeat to have enough samples
            result = self.detector.analyze_text(data)
            assert result.language == expected_lang
    
    def test_analyze_text_with_special_characters(self):
        """Test text with various special characters."""
        data = pd.Series([
            'Text with symbols: !@#$%^&*()',
            'Punctuation marks: .,;:?!',
            'Brackets and braces: []{}()',
            'Mathematical symbols: +-*/=<>',
            'Currency symbols: $€£¥',
            'Quotes and apostrophes: "\'`',
            'Dashes and underscores: -_',
            'Forward and back slashes: /\\',
            'Pipe and tilde: |~',
            'Percentage and hash: %#'
        ])
        
        result = self.detector.analyze_text(data)
        
        assert isinstance(result, TextAnalysis)
        assert result.contains_special_chars == True
        assert result.text_type in ['natural_language', 'mixed']
    
    def test_analyze_text_encoding_detection(self):
        """Test encoding detection."""
        # ASCII text
        ascii_data = pd.Series(['Hello', 'World', 'Test', 'Data', 'Analysis'])
        result = self.detector.analyze_text(ascii_data)
        assert result.encoding == 'ascii'
        
        # Text with extended characters
        extended_data = pd.Series(['Café', 'Naïve', 'Résumé', 'Piñata', 'Jalapeño'])
        result = self.detector.analyze_text(extended_data)
        assert result.encoding in ['utf-8', 'latin-1']
    
    def test_analyze_text_readability_score(self):
        """Test readability score calculation."""
        # Simple text (should have higher readability)
        simple_data = pd.Series([
            'The cat sat on the mat.',
            'Dogs like to play in the park.',
            'Birds fly in the sky.',
            'Fish swim in the water.',
            'The sun is bright today.'
        ])
        
        result = self.detector.analyze_text(simple_data)
        
        if result.text_type == 'natural_language':
            assert result.readability_score is not None
            assert result.readability_score > 50  # Simple text should be readable
        
        # Complex text (should have lower readability)
        complex_data = pd.Series([
            'The implementation of sophisticated algorithms necessitates comprehensive understanding.',
            'Multidimensional optimization techniques require extensive computational resources.',
            'Hierarchical clustering methodologies facilitate pattern recognition capabilities.',
            'Statistical significance testing validates experimental hypothesis formulations.',
            'Probabilistic inference mechanisms enable uncertainty quantification procedures.'
        ])
        
        result = self.detector.analyze_text(complex_data)
        
        if result.text_type == 'natural_language':
            assert result.readability_score is not None
            # Complex text might have lower readability, but not always
    
    def test_analyze_text_insufficient_data_raises_error(self):
        """Test that insufficient data raises error."""
        # Less than minimum samples
        data = pd.Series(['short', 'text'])  # Only 2 samples
        
        with pytest.raises(NeuroLiteException):
            self.detector.analyze_text(data)
    
    def test_analyze_text_with_missing_values(self):
        """Test text analysis with missing values."""
        data = pd.Series([
            'Valid text entry',
            np.nan,
            'Another valid entry',
            'Third entry',
            np.nan,
            'Fourth entry',
            'Fifth entry',
            'Sixth entry',
            'Seventh entry',
            'Eighth entry'
        ])
        
        result = self.detector.analyze_text(data)
        
        # Should analyze only non-missing values
        assert isinstance(result, TextAnalysis)
        assert result.unique_ratio > 0  # Should have some unique values
    
    def test_determine_text_type_categorical(self):
        """Test categorical text type determination."""
        # Low unique ratio, short length
        data = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'] * 3)
        assert self.detector._determine_text_type(data) == 'categorical_text'
    
    def test_determine_text_type_structured(self):
        """Test structured text type determination."""
        # Email pattern
        email_data = pd.Series(['user1@test.com', 'user2@test.com', 'user3@test.com'] * 4)
        assert self.detector._determine_text_type(email_data) == 'structured_text'
        
        # Phone pattern
        phone_data = pd.Series(['555-1234', '555-5678', '555-9012'] * 4)
        assert self.detector._determine_text_type(phone_data) == 'structured_text'
        
        # Code pattern
        code_data = pd.Series(['ABC-123', 'DEF-456', 'GHI-789'] * 4)
        assert self.detector._determine_text_type(code_data) == 'structured_text'
    
    def test_is_structured_text_patterns(self):
        """Test structured text pattern detection."""
        # Email pattern
        email_data = pd.Series(['user@example.com', 'admin@test.org', 'info@company.net'])
        assert self.detector._is_structured_text(email_data) == True
        
        # URL pattern
        url_data = pd.Series(['https://example.com', 'http://test.org', 'https://company.net'])
        assert self.detector._is_structured_text(url_data) == True
        
        # Phone pattern
        phone_data = pd.Series(['555-123-4567', '555-987-6543', '555-456-7890'])
        assert self.detector._is_structured_text(phone_data) == True
        
        # Code pattern
        code_data = pd.Series(['ABC-123', 'DEF-456', 'GHI-789'])
        assert self.detector._is_structured_text(code_data) == True
        
        # Non-structured text
        natural_data = pd.Series(['This is natural language', 'Another sentence', 'More text'])
        assert self.detector._is_structured_text(natural_data) == False
    
    def test_is_natural_language_indicators(self):
        """Test natural language detection indicators."""
        # Text with common words and sentence structure
        natural_data = pd.Series([
            'The quick brown fox jumps over the lazy dog',
            'This is a test of natural language detection',
            'Machine learning algorithms can process text data'
        ])
        assert self.detector._is_natural_language(natural_data) == True
        
        # Non-natural language text
        code_data = pd.Series(['ABC123', 'DEF456', 'GHI789'])
        assert self.detector._is_natural_language(code_data) == False
        
        # Short categorical text
        cat_data = pd.Series(['A', 'B', 'C'])
        assert self.detector._is_natural_language(cat_data) == False
    
    def test_detect_language_patterns(self):
        """Test language detection patterns."""
        # English
        en_data = pd.Series(['Hello world this is English text'])
        assert self.detector._detect_language(en_data) == 'en'
        
        # Russian (Cyrillic)
        ru_data = pd.Series(['Привет мир это русский текст'])
        assert self.detector._detect_language(ru_data) == 'ru'
        
        # Spanish
        es_data = pd.Series(['Hola mundo este es texto español'])
        assert self.detector._detect_language(es_data) == 'es'
        
        # German (with special characters)
        de_data = pd.Series(['Hallo Welt das ist schöner deutscher Text'])
        assert self.detector._detect_language(de_data) == 'de'
        
        # French
        fr_data = pd.Series(['Bonjour monde ceci est du texte français'])
        assert self.detector._detect_language(fr_data) == 'fr'
        
        # Short text (should return None)
        short_data = pd.Series(['Hi'])
        assert self.detector._detect_language(short_data) is None
    
    def test_contains_special_characters(self):
        """Test special character detection."""
        # Text with special characters
        special_data = pd.Series(['Hello!', 'Test@example.com', 'Price: $29.99'])
        assert self.detector._contains_special_characters(special_data) == True
        
        # Text without special characters
        plain_data = pd.Series(['Hello', 'World', 'Test'])
        assert self.detector._contains_special_characters(plain_data) == False
    
    def test_contains_numbers(self):
        """Test number detection in text."""
        # Text with numbers
        number_data = pd.Series(['Product 123', 'Version 2.0', 'Year 2023'])
        assert self.detector._contains_numbers(number_data) == True
        
        # Text without numbers
        text_data = pd.Series(['Hello', 'World', 'Test'])
        assert self.detector._contains_numbers(text_data) == False
    
    def test_calculate_readability_score(self):
        """Test readability score calculation."""
        # Simple text
        simple_data = pd.Series(['The cat sat on the mat'])
        score = self.detector._calculate_readability_score(simple_data)
        assert 0 <= score <= 100
        
        # Complex text
        complex_data = pd.Series(['The implementation of sophisticated methodologies'])
        score = self.detector._calculate_readability_score(complex_data)
        assert 0 <= score <= 100
        
        # Very short text
        short_data = pd.Series(['Hi'])
        score = self.detector._calculate_readability_score(short_data)
        assert score == 50.0  # Neutral score for short text
    
    def test_calculate_text_confidence(self):
        """Test text classification confidence calculation."""
        # High confidence categorical text
        cat_analysis = TextAnalysis(
            text_type='categorical_text',
            language='en',
            encoding='ascii',
            avg_length=10.0,
            max_length=15,
            min_length=5,
            unique_ratio=0.03,  # Very low unique ratio
            contains_special_chars=False,
            contains_numbers=False
        )
        confidence = self.detector._calculate_text_confidence(cat_analysis)
        assert confidence > 0.8
        
        # High confidence natural language
        nl_analysis = TextAnalysis(
            text_type='natural_language',
            language='en',
            encoding='ascii',
            avg_length=60.0,
            max_length=100,
            min_length=20,
            unique_ratio=0.9,
            contains_special_chars=True,
            contains_numbers=False,
            readability_score=70.0
        )
        confidence = self.detector._calculate_text_confidence(nl_analysis)
        assert confidence > 0.8
        
        # Lower confidence mixed type
        mixed_analysis = TextAnalysis(
            text_type='mixed',
            language=None,
            encoding='ascii',
            avg_length=20.0,
            max_length=50,
            min_length=5,
            unique_ratio=0.5,
            contains_special_chars=True,
            contains_numbers=True
        )
        confidence = self.detector._calculate_text_confidence(mixed_analysis)
        assert 0.6 <= confidence <= 0.8
    
    def test_classify_text_column(self):
        """Test classification of text columns."""
        # Natural language text
        nl_data = pd.Series([
            'This is a sample sentence with natural language.',
            'Machine learning is used for text analysis.',
            'Data science involves processing various data types.',
            'Natural language processing helps understand text.',
            'Statistical methods validate research findings.',
            'Python is popular for data analysis tasks.',
            'Text mining extracts insights from documents.',
            'Information retrieval finds relevant content.',
            'Deep learning models process complex structures.',
            'Pattern recognition identifies text characteristics.'
        ])
        
        result = self.detector._classify_text(nl_data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'text'
        assert 'natural_language' in result.subtype
        assert result.confidence > 0.7
        assert 'text_type' in result.properties
        assert 'language' in result.properties
        assert 'encoding' in result.properties
        assert 'avg_length' in result.properties
        assert 'unique_ratio' in result.properties
        assert 'contains_special_chars' in result.properties
        assert 'contains_numbers' in result.properties
        assert 'readability_score' in result.properties
    
    def test_classify_text_fallback(self):
        """Test text classification fallback for insufficient data."""
        # Insufficient data should trigger fallback
        short_data = pd.Series(['A', 'B'])  # Less than min_samples_for_analysis
        
        result = self.detector._classify_text(short_data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'text'
        assert result.confidence == 0.5  # Fallback confidence
        assert 'avg_length' in result.properties
        assert 'unique_ratio' in result.properties
    
    def test_classify_text_categorical(self):
        """Test classification of categorical text."""
        # Categorical text data
        cat_data = pd.Series([
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C',
            'Category A', 'Category B', 'Category C'
        ])
        
        result = self.detector._classify_text(cat_data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'text'
        assert 'categorical_text' in result.subtype
        assert result.properties['text_type'] == 'categorical_text'
        assert result.properties['unique_ratio'] < 0.5
    
    def test_classify_text_structured(self):
        """Test classification of structured text."""
        # Structured text data (emails)
        struct_data = pd.Series([
            'user1@example.com', 'user2@example.com', 'user3@example.com',
            'admin@company.org', 'support@service.net', 'info@business.co',
            'contact@website.com', 'sales@store.com', 'help@platform.io',
            'team@startup.ai', 'dev@tech.com', 'data@analytics.com'
        ])
        
        result = self.detector._classify_text(struct_data)
        
        assert isinstance(result, ColumnType)
        assert result.primary_type == 'text'
        assert 'structured_text' in result.subtype
        assert result.properties['text_type'] == 'structured_text'
        assert result.properties['contains_special_chars'] == True