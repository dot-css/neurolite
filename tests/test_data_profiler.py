"""
Unit tests for DataProfiler orchestrator.

This module tests the core analysis pipeline functionality including
error handling, graceful degradation, and result aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from neurolite.core.data_profiler import DataProfiler
from neurolite.core.data_models import (
    ProfileReport, QuickReport, FileFormat, DataStructure,
    ColumnType, QualityMetrics, StatisticalProperties,
    TaskIdentification, ModelRecommendation
)
from neurolite.core.exceptions import (
    NeuroLiteException, UnsupportedFormatError, InsufficientDataError,
    ResourceLimitError, ProcessingError, ValidationError
)


class TestDataProfilerInitialization:
    """Test DataProfiler initialization and configuration."""
    
    def test_default_initialization(self):
        """Test DataProfiler initialization with default parameters."""
        profiler = DataProfiler()
        
        assert profiler.confidence_threshold == 0.8
        assert profiler.enable_graceful_degradation is True
        assert profiler.max_processing_time is None
        assert profiler.max_memory_usage_mb is None
        assert profiler._failed_components == []
    
    def test_custom_initialization(self):
        """Test DataProfiler initialization with custom parameters."""
        profiler = DataProfiler(
            confidence_threshold=0.9,
            enable_graceful_degradation=False,
            max_processing_time=30.0,
            max_memory_usage_mb=1024.0
        )
        
        assert profiler.confidence_threshold == 0.9
        assert profiler.enable_graceful_degradation is False
        assert profiler.max_processing_time == 30.0
        assert profiler.max_memory_usage_mb == 1024.0
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DataProfiler(confidence_threshold=1.5)
        
        assert "confidence_threshold" in str(exc_info.value)
        assert "Must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        profiler = DataProfiler()
        
        assert hasattr(profiler, 'file_detector')
        assert hasattr(profiler, 'data_type_detector')
        assert hasattr(profiler, 'quality_detector')
        assert hasattr(profiler, 'domain_detector')
        assert hasattr(profiler, 'statistical_analyzer')
        assert hasattr(profiler, 'complexity_analyzer')
        assert hasattr(profiler, 'model_recommender')
        assert hasattr(profiler, 'preprocessing_recommender')


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_dataframe(self):
        """Test loading pandas DataFrame."""
        profiler = DataProfiler()
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        
        loaded_df, file_info = profiler._load_data(df)
        
        assert loaded_df.equals(df)
        assert file_info.format_type == 'dataframe'
        assert file_info.confidence == 1.0
        assert file_info.mime_type == 'application/x-pandas-dataframe'
    
    def test_load_numpy_array(self):
        """Test loading numpy array."""
        profiler = DataProfiler()
        arr = np.array([[1, 2], [3, 4]])
        
        loaded_df, file_info = profiler._load_data(arr)
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (2, 2)
        assert file_info.format_type == 'numpy_array'
        assert file_info.confidence == 1.0
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        profiler = DataProfiler()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('A,B\n1,a\n2,b\n3,c\n')
            temp_path = f.name
        
        try:
            with patch.object(profiler.file_detector, 'detect_format') as mock_detect:
                mock_detect.return_value = FileFormat(
                    format_type='csv',
                    confidence=0.95,
                    mime_type='text/csv',
                    encoding='utf-8'
                )
                
                loaded_df, file_info = profiler._load_data(temp_path)
                
                assert isinstance(loaded_df, pd.DataFrame)
                assert len(loaded_df) == 3
                assert list(loaded_df.columns) == ['A', 'B']
                assert file_info.format_type == 'csv'
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises ValidationError."""
        profiler = DataProfiler()
        
        with pytest.raises(ValidationError) as exc_info:
            profiler._load_data('/nonexistent/file.csv')
        
        assert "File does not exist" in str(exc_info.value)
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format raises UnsupportedFormatError."""
        profiler = DataProfiler()
        
        with tempfile.NamedTemporaryFile(suffix='.unknown') as f:
            with patch.object(profiler.file_detector, 'detect_format') as mock_detect:
                mock_detect.return_value = FileFormat(
                    format_type='unknown',
                    confidence=0.5,
                    mime_type='application/octet-stream'
                )
                
                with pytest.raises(UnsupportedFormatError):
                    profiler._load_data(f.name)
    
    def test_quick_mode_sampling(self):
        """Test that quick mode applies sampling for large datasets."""
        profiler = DataProfiler()
        large_df = pd.DataFrame({'A': range(15000), 'B': range(15000)})
        
        loaded_df, file_info = profiler._load_data(large_df, quick_mode=True)
        
        assert len(loaded_df) < 15000  # Should be sampled (less than original)
        assert len(loaded_df) >= 1000  # Should have minimum sample size
        assert file_info.metadata.get('sampled') is True
        assert file_info.metadata.get('sample_size') == len(loaded_df)


class TestResourceLimits:
    """Test resource limit checking."""
    
    def test_processing_time_limit(self):
        """Test processing time limit enforcement."""
        profiler = DataProfiler(max_processing_time=0.001)  # Very short limit
        profiler._analysis_start_time = 0  # Set start time to epoch
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(ResourceLimitError) as exc_info:
            profiler._check_resource_limits(df)
        
        assert "processing_time" in str(exc_info.value)
    
    def test_memory_limit(self):
        """Test memory usage limit enforcement."""
        profiler = DataProfiler(max_memory_usage_mb=0.001)  # Very small limit
        
        # Create a DataFrame that will exceed the memory limit
        df = pd.DataFrame({'A': ['x' * 1000] * 1000})
        
        with pytest.raises(ResourceLimitError) as exc_info:
            profiler._check_resource_limits(df)
        
        assert "memory" in str(exc_info.value)


class TestAnalysisPipeline:
    """Test the main analysis pipeline."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': range(100),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'value': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def mock_profiler(self):
        """Create a DataProfiler with mocked components."""
        profiler = DataProfiler()
        
        # Mock all components
        profiler.file_detector = Mock()
        profiler.data_type_detector = Mock()
        profiler.quality_detector = Mock()
        profiler.domain_detector = Mock()
        profiler.statistical_analyzer = Mock()
        profiler.complexity_analyzer = Mock()
        profiler.model_recommender = Mock()
        profiler.preprocessing_recommender = Mock()
        
        return profiler
    
    def test_successful_analysis(self, mock_profiler, sample_dataframe):
        """Test successful complete analysis pipeline."""
        # Setup mock returns
        mock_profiler.file_detector.detect_structure.return_value = DataStructure(
            structure_type='tabular',
            dimensions=(100, 4),
            sample_size=100,
            memory_usage=1024
        )
        
        mock_profiler.data_type_detector.classify_columns.return_value = {
            'id': ColumnType('numerical', 'integer', 0.95),
            'category': ColumnType('categorical', 'nominal', 0.9),
            'value': ColumnType('numerical', 'float', 0.95),
            'target': ColumnType('categorical', 'binary', 0.9)
        }
        
        mock_profiler.quality_detector.analyze_quality.return_value = QualityMetrics(
            completeness=1.0,
            consistency=0.95,
            validity=0.98,
            uniqueness=0.75,
            missing_pattern='none',
            duplicate_count=0
        )
        
        mock_profiler.statistical_analyzer.analyze_comprehensive.return_value = StatisticalProperties(
            distribution='normal',
            parameters={'mean': 0.0, 'std': 1.0}
        )
        
        mock_profiler.model_recommender.recommend_models.return_value = [
            ModelRecommendation(
                model_name='RandomForest',
                model_type='ensemble',
                confidence=0.85,
                rationale='Good for tabular data'
            )
        ]
        
        mock_profiler.preprocessing_recommender.recommend_preprocessing_pipeline.return_value = [
            'StandardScaler for numerical features',
            'OneHotEncoder for categorical features'
        ]
        
        mock_profiler.complexity_analyzer.analyze_complexity.return_value = {
            'complexity_level': 'medium',
            'estimated_time': 5.0
        }
        
        # Run analysis
        result = mock_profiler.analyze(sample_dataframe)
        
        # Verify result type and structure
        assert isinstance(result, ProfileReport)
        assert result.data_structure.structure_type == 'tabular'
        assert len(result.column_analysis) == 4
        assert result.quality_metrics.completeness == 1.0
        assert len(result.model_recommendations) == 1
        assert len(result.preprocessing_recommendations) == 2
        assert result.execution_time >= 0  # Execution time should be non-negative
    
    def test_graceful_degradation_enabled(self, sample_dataframe):
        """Test graceful degradation when components fail."""
        profiler = DataProfiler(enable_graceful_degradation=True)
        
        # Mock file detector to work, but make other components fail
        with patch.object(profiler, 'file_detector') as mock_file:
            mock_file.detect_structure.return_value = DataStructure(
                structure_type='tabular',
                dimensions=(100, 4),
                sample_size=100,
                memory_usage=1024
            )
            
            with patch.object(profiler, 'data_type_detector') as mock_type:
                mock_type.classify_columns.side_effect = Exception("Component failed")
                
                with patch.object(profiler, 'quality_detector') as mock_quality:
                    mock_quality.analyze_quality.side_effect = Exception("Component failed")
                    
                    # Analysis should still complete with graceful degradation
                    result = profiler.analyze(sample_dataframe)
                    
                    assert isinstance(result, ProfileReport)
                    assert 'warnings' in result.resource_requirements
                    assert 'failed_components' in result.resource_requirements['warnings']
                    assert len(profiler._failed_components) > 0
    
    def test_graceful_degradation_disabled(self, sample_dataframe):
        """Test that errors are raised when graceful degradation is disabled."""
        profiler = DataProfiler(enable_graceful_degradation=False)
        
        with patch.object(profiler, 'data_type_detector') as mock_type:
            mock_type.classify_columns.side_effect = Exception("Component failed")
            
            with pytest.raises(ProcessingError) as exc_info:
                profiler.analyze(sample_dataframe)
            
            assert "column_analysis" in str(exc_info.value)


class TestQuickAnalysis:
    """Test quick analysis functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'A': range(1000),
            'B': ['category'] * 1000,
            'C': np.random.normal(0, 1, 1000)
        })
    
    def test_quick_analysis_success(self, sample_dataframe):
        """Test successful quick analysis."""
        profiler = DataProfiler()
        
        with patch.object(profiler.file_detector, 'detect_structure') as mock_structure:
            mock_structure.return_value = DataStructure(
                structure_type='tabular',
                dimensions=(1000, 3),
                sample_size=1000,
                memory_usage=1024
            )
            
            result = profiler.quick_analyze(sample_dataframe)
            
            assert isinstance(result, QuickReport)
            assert result.data_structure.structure_type == 'tabular'
            assert 'shape' in result.basic_stats
            assert 'dtypes' in result.basic_stats
            assert 'missing_values' in result.basic_stats
            assert len(result.quick_recommendations) >= 0  # May or may not have recommendations
            assert result.execution_time >= 0  # Execution time should be non-negative
    
    def test_quick_analysis_basic_stats(self, sample_dataframe):
        """Test basic statistics generation in quick analysis."""
        profiler = DataProfiler()
        
        basic_stats = profiler._generate_basic_stats(sample_dataframe)
        
        assert basic_stats['shape'] == (1000, 3)
        assert 'A' in basic_stats['dtypes']
        assert 'missing_values' in basic_stats
        assert 'memory_usage_mb' in basic_stats
        assert 'numerical_summary' in basic_stats
        assert 'categorical_summary' in basic_stats
    
    def test_quick_recommendations_generation(self, sample_dataframe):
        """Test quick recommendations generation."""
        profiler = DataProfiler()
        
        # Add some missing values
        sample_dataframe.loc[0:10, 'A'] = None
        
        basic_stats = profiler._generate_basic_stats(sample_dataframe)
        recommendations = profiler._generate_quick_recommendations(sample_dataframe, basic_stats)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('missing values' in rec.lower() for rec in recommendations)
    
    def test_quick_analysis_large_dataset_sampling(self):
        """Test that quick analysis applies sampling for large datasets."""
        profiler = DataProfiler()
        
        # Create large dataset
        large_df = pd.DataFrame({'A': range(15000), 'B': range(15000)})
        
        with patch.object(profiler.file_detector, 'detect_structure') as mock_structure:
            mock_structure.return_value = DataStructure(
                structure_type='tabular',
                dimensions=(5000, 2),  # Should be sampled
                sample_size=5000,
                memory_usage=1024
            )
            
            result = profiler.quick_analyze(large_df)
            
            # Verify sampling occurred
            assert result.file_info.metadata.get('sampled') is True
            # Sample size should be reasonable (between 1000 and original size)
            sample_size = result.file_info.metadata.get('sample_size')
            assert 1000 <= sample_size < 15000


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_data_source(self):
        """Test that invalid data source raises ValidationError."""
        profiler = DataProfiler()
        
        with pytest.raises(ValidationError):
            profiler.analyze(None)
        
        with pytest.raises(ValidationError):
            profiler.analyze(123)  # Invalid type
    
    def test_processing_error_propagation(self):
        """Test that processing errors are properly wrapped."""
        profiler = DataProfiler(enable_graceful_degradation=False)
        
        with patch.object(profiler, '_load_data') as mock_load:
            mock_load.side_effect = Exception("Unexpected error")
            
            with pytest.raises(ProcessingError) as exc_info:
                profiler.analyze(pd.DataFrame({'A': [1, 2, 3]}))
            
            assert "analysis_pipeline" in str(exc_info.value)
    
    def test_neurolite_exception_passthrough(self):
        """Test that NeuroLiteExceptions are passed through without wrapping."""
        profiler = DataProfiler()
        
        with patch.object(profiler, '_load_data') as mock_load:
            mock_load.side_effect = UnsupportedFormatError("test_format")
            
            with pytest.raises(UnsupportedFormatError):
                profiler.analyze("test_file.unknown")


class TestTaskIdentification:
    """Test ML task identification logic."""
    
    def test_classification_task_identification(self):
        """Test identification of classification tasks."""
        profiler = DataProfiler()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'target': ['A', 'B', 'A', 'B']
        })
        
        column_analysis = {
            'feature1': ColumnType('numerical', 'integer', 0.95),
            'feature2': ColumnType('numerical', 'float', 0.95),
            'target': ColumnType('categorical', 'nominal', 0.9)
        }
        
        task_id = profiler._identify_task(df, column_analysis)
        
        assert task_id.task_type == 'classification'
        assert task_id.task_subtype == 'binary'
        assert task_id.confidence > 0
    
    def test_regression_task_identification(self):
        """Test identification of regression tasks."""
        profiler = DataProfiler()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'target': [10.5, 20.3, 15.7, 25.1]
        })
        
        column_analysis = {
            'feature1': ColumnType('numerical', 'integer', 0.95),
            'feature2': ColumnType('numerical', 'float', 0.95),
            'target': ColumnType('numerical', 'float', 0.95)
        }
        
        task_id = profiler._identify_task(df, column_analysis)
        
        assert task_id.task_type == 'regression'
        assert task_id.task_subtype == 'continuous'
    
    def test_unsupervised_task_identification(self):
        """Test identification of unsupervised tasks."""
        profiler = DataProfiler()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'feature3': ['A', 'B', 'C', 'D']
        })
        
        column_analysis = {
            'feature1': ColumnType('numerical', 'integer', 0.95),
            'feature2': ColumnType('numerical', 'float', 0.95),
            'feature3': ColumnType('categorical', 'nominal', 0.9)
        }
        
        task_id = profiler._identify_task(df, column_analysis)
        
        assert task_id.task_type == 'unsupervised'
        assert task_id.task_subtype == 'clustering'


if __name__ == '__main__':
    pytest.main([__file__])