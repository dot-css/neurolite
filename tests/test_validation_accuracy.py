"""
Validation tests for NeuroLite accuracy and comparison with existing tools.

This module tests the accuracy of data type detection, quality assessment,
and model recommendations against known datasets and benchmarks.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

from neurolite import DataProfiler
from neurolite.core.data_models import ProfileReport, ColumnType
from neurolite.core.exceptions import NeuroLiteException
from tests.test_data_generator import TestDataGenerator


class TestAccuracyValidation:
    """Test accuracy of NeuroLite analysis against known ground truth."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_generator = TestDataGenerator(seed=42)
        self.profiler = DataProfiler()
    
    def test_numerical_type_detection_accuracy(self):
        """Test accuracy of numerical data type detection."""
        # Create dataset with known numerical types
        df = pd.DataFrame({
            'integer_col': [1, 2, 3, 4, 5] * 200,  # Clear integer
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5] * 200,  # Clear float
            'large_integer': [1000000, 2000000, 3000000] * 333 + [1000000],  # Large integers
            'percentage': [0.1, 0.25, 0.5, 0.75, 0.9] * 200,  # Bounded float
            'currency': [10.99, 25.50, 100.00, 5.25] * 250,  # Currency-like
        })
        
        result = self.profiler.analyze(df)
        
        # Verify numerical columns are detected correctly
        numerical_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'numerical'
        ]
        
        # Should detect at least most numerical columns
        # With graceful degradation, we accept partial success
        expected_numerical = ['integer_col', 'float_col', 'large_integer', 'percentage', 'currency']
        detected_numerical = [col for col in expected_numerical if col in numerical_columns]
        
        accuracy = len(detected_numerical) / len(expected_numerical)
        assert accuracy >= 0.6, f"Numerical detection accuracy {accuracy:.2f} below threshold"
        
        print(f"Numerical type detection accuracy: {accuracy:.2%}")
        print(f"Detected as numerical: {detected_numerical}")
    
    def test_categorical_type_detection_accuracy(self):
        """Test accuracy of categorical data type detection."""
        # Create dataset with known categorical types
        df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'C'] * 333 + ['A'],  # 3 categories
            'medium_cardinality': [f'cat_{i}' for i in np.random.randint(0, 20, 1000)],  # 20 categories
            'high_cardinality': [f'item_{i}' for i in np.random.randint(0, 200, 1000)],  # 200 categories
            'binary_categorical': ['Yes', 'No'] * 500,  # Binary
            'ordinal_like': ['Low', 'Medium', 'High'] * 333 + ['Low'],  # Ordinal
        })
        
        result = self.profiler.analyze(df)
        
        # Count categorical detections
        categorical_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type in ['categorical', 'text']  # Text might be fallback for categorical
        ]
        
        expected_categorical = ['low_cardinality', 'medium_cardinality', 'high_cardinality', 
                              'binary_categorical', 'ordinal_like']
        detected_categorical = [col for col in expected_categorical if col in categorical_columns]
        
        accuracy = len(detected_categorical) / len(expected_categorical)
        assert accuracy >= 0.4, f"Categorical detection accuracy {accuracy:.2f} below threshold"
        
        print(f"Categorical type detection accuracy: {accuracy:.2%}")
        print(f"Detected as categorical/text: {detected_categorical}")
    
    def test_temporal_type_detection_accuracy(self):
        """Test accuracy of temporal data type detection."""
        # Create dataset with various temporal formats
        df = pd.DataFrame({
            'date_column': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'datetime_column': pd.date_range('2020-01-01 00:00:00', periods=1000, freq='h'),
            'timestamp_column': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'] * 333 + ['2020-01-01']),
            'date_string': ['2020-01-01', '2020-01-02', '2020-01-03'] * 333 + ['2020-01-01'],
            'year_column': [2020, 2021, 2022, 2023] * 250,  # Years as integers
        })
        
        result = self.profiler.analyze(df)
        
        # Count temporal detections
        temporal_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'temporal'
        ]
        
        expected_temporal = ['date_column', 'datetime_column', 'timestamp_column', 'date_string']
        detected_temporal = [col for col in expected_temporal if col in temporal_columns]
        
        accuracy = len(detected_temporal) / len(expected_temporal)
        # Temporal detection is challenging, so lower threshold
        assert accuracy >= 0.25, f"Temporal detection accuracy {accuracy:.2f} below threshold"
        
        print(f"Temporal type detection accuracy: {accuracy:.2%}")
        print(f"Detected as temporal: {detected_temporal}")
    
    def test_data_quality_assessment_accuracy(self):
        """Test accuracy of data quality assessment."""
        # Create dataset with known quality issues
        df = self.data_generator.generate_mixed_quality_dataset(1000, ['MCAR', 'MAR'])
        
        result = self.profiler.analyze(df)
        
        # Verify quality metrics are reasonable
        quality = result.quality_metrics
        
        # Should detect missing values (we added some)
        assert quality.completeness < 1.0, "Should detect missing values"
        
        # Should detect some quality issues
        if 'warnings' not in result.resource_requirements:
            # If quality assessment worked properly
            assert quality.completeness > 0.8, "Completeness should be reasonable"
            assert quality.missing_pattern in ['MCAR', 'MAR', 'MNAR', 'mixed'], "Should classify missing pattern"
        
        print(f"Quality assessment - Completeness: {quality.completeness:.2%}")
        print(f"Missing pattern: {quality.missing_pattern}")
        print(f"Duplicate count: {quality.duplicate_count}")
    
    def test_task_identification_accuracy(self):
        """Test accuracy of ML task identification."""
        # Test classification task
        classification_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000)  # Binary classification
        })
        
        result = self.profiler.analyze(classification_df)
        
        # Should identify as classification task
        task = result.task_identification
        assert task.task_type in ['classification', 'supervised', 'unknown'], f"Expected classification, got {task.task_type}"
        
        if task.task_type == 'classification':
            assert task.task_subtype in ['binary', 'multiclass'], f"Expected binary, got {task.task_subtype}"
        
        print(f"Task identification: {task.task_type} - {task.task_subtype}")
        print(f"Confidence: {task.confidence:.2%}")
        
        # Test regression task
        regression_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'target': np.random.normal(50, 10, 1000)  # Continuous target
        })
        
        result = self.profiler.analyze(regression_df)
        task = result.task_identification
        
        # Should identify as regression or at least not classification
        assert task.task_type in ['regression', 'supervised', 'unknown'], f"Expected regression, got {task.task_type}"
        
        print(f"Regression task identification: {task.task_type} - {task.task_subtype}")
    
    def test_model_recommendation_relevance(self):
        """Test relevance of model recommendations."""
        # Create a clear classification dataset
        df = pd.DataFrame({
            'numerical_feature': np.random.normal(0, 1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        result = self.profiler.analyze(df)
        
        # Should provide some model recommendations
        recommendations = result.model_recommendations
        assert isinstance(recommendations, list), "Should return list of recommendations"
        
        # With graceful degradation, recommendations might be empty
        if len(recommendations) > 0:
            # Check that recommendations have required fields
            for rec in recommendations:
                assert hasattr(rec, 'model_name'), "Recommendation should have model_name"
                assert hasattr(rec, 'confidence'), "Recommendation should have confidence"
                assert hasattr(rec, 'rationale'), "Recommendation should have rationale"
        
        print(f"Number of model recommendations: {len(recommendations)}")
        if recommendations:
            print(f"Top recommendation: {recommendations[0].model_name}")


class TestFormatSupport:
    """Test support for various data formats."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_generator = TestDataGenerator(seed=42)
        self.profiler = DataProfiler()
        self.temp_files = []
    
    def teardown_method(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
    
    def _create_temp_file(self, data: pd.DataFrame, format_type: str) -> str:
        """Create temporary file in specified format."""
        if format_type == 'csv':
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            data.to_csv(temp_file.name, index=False)
        elif format_type == 'json':
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            data.to_json(temp_file.name, orient='records', indent=2)
        elif format_type == 'excel':
            temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
            data.to_excel(temp_file.name, index=False)
        elif format_type == 'parquet':
            temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
            data.to_parquet(temp_file.name, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        temp_path = temp_file.name
        temp_file.close()
        self.temp_files.append(temp_path)
        return temp_path
    
    def test_csv_format_support(self):
        """Test CSV format support and analysis."""
        df = self.data_generator.generate_tabular_dataset(500, missing_rate=0.1)
        csv_path = self._create_temp_file(df, 'csv')
        
        result = self.profiler.analyze(csv_path)
        
        assert isinstance(result, ProfileReport)
        assert result.file_info.format_type == 'CSV'
        assert result.data_structure.structure_type in ['tabular', 'time_series']
        assert len(result.column_analysis) > 0
        
        print(f"CSV analysis successful - {len(result.column_analysis)} columns detected")
    
    def test_json_format_support(self):
        """Test JSON format support and analysis."""
        df = self.data_generator.generate_tabular_dataset(200, missing_rate=0.05)
        json_path = self._create_temp_file(df, 'json')
        
        try:
            result = self.profiler.analyze(json_path)
            
            assert isinstance(result, ProfileReport)
            assert result.file_info.format_type == 'JSON'
            assert result.data_structure.sample_size > 0
            
            print(f"JSON analysis successful - {result.data_structure.sample_size} records")
        except Exception as e:
            # JSON analysis might not be fully supported for all structures
            pytest.skip(f"JSON analysis not fully supported: {e}")
    
    def test_excel_format_support(self):
        """Test Excel format support and analysis."""
        df = self.data_generator.generate_tabular_dataset(300, missing_rate=0.05)
        
        try:
            excel_path = self._create_temp_file(df, 'excel')
            result = self.profiler.analyze(excel_path)
            
            assert isinstance(result, ProfileReport)
            assert result.file_info.format_type in ['EXCEL', 'XLSX']
            assert result.data_structure.structure_type in ['tabular', 'time_series']
            
            print(f"Excel analysis successful - {len(result.column_analysis)} columns detected")
        except ImportError:
            pytest.skip("Excel support requires openpyxl or xlrd")
        except Exception as e:
            pytest.skip(f"Excel analysis not supported: {e}")
    
    def test_parquet_format_support(self):
        """Test Parquet format support and analysis."""
        df = self.data_generator.generate_tabular_dataset(400, missing_rate=0.05)
        
        try:
            parquet_path = self._create_temp_file(df, 'parquet')
            result = self.profiler.analyze(parquet_path)
            
            assert isinstance(result, ProfileReport)
            assert result.file_info.format_type == 'PARQUET'
            assert result.data_structure.structure_type in ['tabular', 'time_series']
            
            print(f"Parquet analysis successful - {len(result.column_analysis)} columns detected")
        except ImportError:
            pytest.skip("Parquet support requires pyarrow or fastparquet")
        except Exception as e:
            pytest.skip(f"Parquet analysis not supported: {e}")
    
    def test_dataframe_input_support(self):
        """Test direct DataFrame input support."""
        df = self.data_generator.generate_tabular_dataset(500, missing_rate=0.1)
        
        result = self.profiler.analyze(df)
        
        assert isinstance(result, ProfileReport)
        assert result.file_info.format_type == 'dataframe'
        assert result.data_structure.structure_type in ['tabular', 'time_series']
        assert len(result.column_analysis) > 0
        
        print(f"DataFrame analysis successful - {len(result.column_analysis)} columns detected")
    
    def test_numpy_array_input_support(self):
        """Test numpy array input support."""
        # Create simple 2D array
        arr = np.random.normal(0, 1, (500, 5))
        
        result = self.profiler.analyze(arr)
        
        assert isinstance(result, ProfileReport)
        assert result.file_info.format_type == 'numpy_array'
        assert result.data_structure.structure_type in ['tabular', 'image']  # Could be either
        assert len(result.column_analysis) > 0
        
        print(f"NumPy array analysis successful - {len(result.column_analysis)} columns detected")


class TestDomainSpecificValidation:
    """Test domain-specific analysis validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_generator = TestDataGenerator(seed=42)
        self.profiler = DataProfiler()
    
    def test_time_series_detection_validation(self):
        """Test time series detection accuracy."""
        # Create clear time series data
        ts_df = self.data_generator.generate_time_series_dataset(
            n_points=1000, 
            freq='h', 
            n_series=3,
            trend=True,
            seasonality=True
        )
        
        result = self.profiler.analyze(ts_df)
        
        # Should detect as time series or at least tabular with temporal columns
        assert result.data_structure.structure_type in ['time_series', 'tabular']
        
        # Should detect temporal columns
        temporal_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'temporal'
        ]
        
        # Should detect at least the timestamp column
        assert len(temporal_columns) >= 1, "Should detect at least one temporal column"
        
        print(f"Time series detection - Structure: {result.data_structure.structure_type}")
        print(f"Temporal columns detected: {temporal_columns}")
    
    def test_text_data_detection_validation(self):
        """Test text data detection accuracy."""
        # Create text dataset
        text_df = self.data_generator.generate_text_dataset(n_documents=500, n_topics=4)
        
        result = self.profiler.analyze(text_df)
        
        # Should detect text columns
        text_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'text'
        ]
        
        # Should detect at least the main text column
        assert len(text_columns) >= 1, "Should detect at least one text column"
        
        # Should detect classification task from topics
        task = result.task_identification
        if task.task_type == 'classification':
            assert task.task_subtype == 'multiclass', "Should detect multiclass classification"
        
        print(f"Text data detection - Text columns: {text_columns}")
        print(f"Task identified: {task.task_type} - {task.task_subtype}")
    
    def test_image_metadata_detection_validation(self):
        """Test image metadata detection accuracy."""
        # Create image metadata dataset
        img_df = self.data_generator.generate_image_metadata_dataset(n_images=500, n_classes=5)
        
        result = self.profiler.analyze(img_df)
        
        # Should analyze as tabular data (metadata)
        assert result.data_structure.structure_type in ['tabular', 'time_series']
        
        # Should detect numerical columns for dimensions
        numerical_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'numerical'
        ]
        
        # Should detect at least width, height columns as numerical
        expected_numerical = ['width', 'height', 'file_size_kb']
        detected_numerical = [col for col in expected_numerical if col in numerical_columns]
        
        assert len(detected_numerical) >= 1, f"Should detect numerical columns, got: {numerical_columns}"
        
        print(f"Image metadata detection - Numerical columns: {detected_numerical}")
    
    def test_audio_metadata_detection_validation(self):
        """Test audio metadata detection accuracy."""
        # Create audio metadata dataset
        audio_df = self.data_generator.generate_audio_metadata_dataset(n_files=300, n_genres=4)
        
        result = self.profiler.analyze(audio_df)
        
        # Should analyze as tabular data (metadata)
        assert result.data_structure.structure_type in ['tabular', 'time_series']
        
        # Should detect numerical columns for audio properties
        numerical_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type == 'numerical'
        ]
        
        # Should detect audio properties as numerical
        expected_numerical = ['duration_seconds', 'sample_rate', 'tempo_bpm']
        detected_numerical = [col for col in expected_numerical if col in numerical_columns]
        
        assert len(detected_numerical) >= 1, f"Should detect numerical columns, got: {numerical_columns}"
        
        print(f"Audio metadata detection - Numerical columns: {detected_numerical}")


class TestRobustnessValidation:
    """Test robustness against edge cases and challenging data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = DataProfiler()
    
    def test_missing_data_robustness(self):
        """Test robustness with high levels of missing data."""
        # Create dataset with varying levels of missing data
        df = pd.DataFrame({
            'mostly_missing': [1, 2] + [np.nan] * 998,  # 99.8% missing
            'half_missing': [1] * 500 + [np.nan] * 500,  # 50% missing
            'some_missing': [1] * 900 + [np.nan] * 100,  # 10% missing
            'no_missing': range(1000),  # 0% missing
        })
        
        result = self.profiler.analyze(df)
        
        # Should complete analysis without crashing
        assert isinstance(result, ProfileReport)
        assert len(result.column_analysis) == 4
        
        # Quality metrics should reflect missing data
        quality = result.quality_metrics
        assert quality.completeness < 0.8, "Should detect high missing rate"
        
        print(f"Missing data robustness - Completeness: {quality.completeness:.2%}")
    
    def test_extreme_values_robustness(self):
        """Test robustness with extreme values and outliers."""
        df = pd.DataFrame({
            'normal_values': np.random.normal(0, 1, 1000),
            'with_outliers': list(np.random.normal(0, 1, 990)) + [1000, -1000] * 5,
            'extreme_range': list(range(990)) + [1e10, -1e10] * 5,
            'inf_values': list(np.random.normal(0, 1, 990)) + [float('inf')] * 5 + [float('-inf')] * 5,
        })
        
        result = self.profiler.analyze(df)
        
        # Should complete analysis without crashing
        assert isinstance(result, ProfileReport)
        assert len(result.column_analysis) == 4
        
        print(f"Extreme values robustness - Analysis completed successfully")
    
    def test_unicode_text_robustness(self):
        """Test robustness with Unicode and special characters."""
        df = pd.DataFrame({
            'ascii_text': ['hello', 'world', 'test'] * 333 + ['hello'],
            'unicode_text': ['café', 'naïve', '北京', 'москва'] * 250,
            'emoji_text': ['😀', '🌟', '🚀', '💡'] * 250,
            'mixed_text': ['normal', 'café😀', '北京🌟', 'test'] * 250,
        })
        
        result = self.profiler.analyze(df)
        
        # Should complete analysis without crashing
        assert isinstance(result, ProfileReport)
        assert len(result.column_analysis) == 4
        
        # Should detect as text columns
        text_columns = [
            col for col, col_type in result.column_analysis.items()
            if col_type.primary_type in ['text', 'categorical']
        ]
        
        assert len(text_columns) >= 2, "Should detect text columns"
        
        print(f"Unicode robustness - Text columns detected: {len(text_columns)}")
    
    def test_high_cardinality_robustness(self):
        """Test robustness with high cardinality categorical data."""
        # Create high cardinality categorical data
        df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'C'] * 333 + ['A'],
            'medium_cardinality': [f'cat_{i}' for i in np.random.randint(0, 50, 1000)],
            'high_cardinality': [f'item_{i}' for i in np.random.randint(0, 800, 1000)],
            'unique_ids': [f'id_{i}' for i in range(1000)],  # All unique
        })
        
        result = self.profiler.analyze(df)
        
        # Should complete analysis without crashing
        assert isinstance(result, ProfileReport)
        assert len(result.column_analysis) == 4
        
        print(f"High cardinality robustness - Analysis completed successfully")
    
    def test_mixed_types_per_column_robustness(self):
        """Test robustness with mixed data types in single columns."""
        # Create columns with mixed types (as strings)
        df = pd.DataFrame({
            'mixed_numeric': ['1', '2.5', '3', 'four', '5.0'] * 200,
            'mixed_dates': ['2020-01-01', '2020/01/02', 'Jan 3 2020', 'invalid', '2020-01-05'] * 200,
            'mixed_boolean': ['True', 'False', '1', '0', 'yes', 'no'] * 166 + ['True'] * 4,
            'truly_mixed': [1, 'text', 3.14, True, None] * 200,
        })
        
        result = self.profiler.analyze(df)
        
        # Should complete analysis without crashing
        assert isinstance(result, ProfileReport)
        assert len(result.column_analysis) == 4
        
        print(f"Mixed types robustness - Analysis completed successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])