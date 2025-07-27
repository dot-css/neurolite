"""
Unit tests for core data models.

This module contains comprehensive tests for all data model classes
including validation, serialization, and edge cases.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from neurolite.core.data_models import (
    FileFormat,
    DataStructure,
    ColumnType,
    QualityMetrics,
    StatisticalProperties,
    TaskIdentification,
    ModelRecommendation,
    ProfileReport,
    QuickReport,
    NumericalAnalysis,
    CategoricalAnalysis,
    TemporalAnalysis,
    MissingDataAnalysis
)


class TestFileFormat:
    """Test cases for FileFormat data model."""
    
    def test_valid_file_format(self):
        """Test creation of valid FileFormat instance."""
        file_format = FileFormat(
            format_type="CSV",
            confidence=0.95,
            mime_type="text/csv",
            encoding="utf-8",
            metadata={"delimiter": ","}
        )
        
        assert file_format.format_type == "CSV"
        assert file_format.confidence == 0.95
        assert file_format.mime_type == "text/csv"
        assert file_format.encoding == "utf-8"
        assert file_format.metadata["delimiter"] == ","
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        FileFormat("CSV", 0.0, "text/csv")
        FileFormat("CSV", 1.0, "text/csv")
        FileFormat("CSV", 0.5, "text/csv")
        
        # Invalid confidence scores
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            FileFormat("CSV", -0.1, "text/csv")
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            FileFormat("CSV", 1.1, "text/csv")
    
    def test_empty_format_type(self):
        """Test validation of empty format type."""
        with pytest.raises(ValueError, match="Format type cannot be empty"):
            FileFormat("", 0.5, "text/csv")


class TestDataStructure:
    """Test cases for DataStructure data model."""
    
    def test_valid_data_structure(self):
        """Test creation of valid DataStructure instance."""
        data_structure = DataStructure(
            structure_type="tabular",
            dimensions=(100, 10),
            sample_size=100,
            memory_usage=1024
        )
        
        assert data_structure.structure_type == "tabular"
        assert data_structure.dimensions == (100, 10)
        assert data_structure.sample_size == 100
        assert data_structure.memory_usage == 1024
    
    def test_negative_sample_size(self):
        """Test validation of negative sample size."""
        with pytest.raises(ValueError, match="Sample size cannot be negative"):
            DataStructure("tabular", (100, 10), -1, 1024)
    
    def test_negative_memory_usage(self):
        """Test validation of negative memory usage."""
        with pytest.raises(ValueError, match="Memory usage cannot be negative"):
            DataStructure("tabular", (100, 10), 100, -1)
    
    def test_empty_dimensions(self):
        """Test validation of empty dimensions."""
        with pytest.raises(ValueError, match="Dimensions cannot be empty"):
            DataStructure("tabular", (), 100, 1024)


class TestColumnType:
    """Test cases for ColumnType data model."""
    
    def test_valid_column_type(self):
        """Test creation of valid ColumnType instance."""
        column_type = ColumnType(
            primary_type="numerical",
            subtype="integer",
            confidence=0.9,
            properties={"range": (0, 100)}
        )
        
        assert column_type.primary_type == "numerical"
        assert column_type.subtype == "integer"
        assert column_type.confidence == 0.9
        assert column_type.properties["range"] == (0, 100)
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ColumnType("numerical", "integer", 1.5)
    
    def test_empty_subtype(self):
        """Test validation of empty subtype."""
        with pytest.raises(ValueError, match="Subtype cannot be empty"):
            ColumnType("numerical", "", 0.9)


class TestQualityMetrics:
    """Test cases for QualityMetrics data model."""
    
    def test_valid_quality_metrics(self):
        """Test creation of valid QualityMetrics instance."""
        quality_metrics = QualityMetrics(
            completeness=0.95,
            consistency=0.90,
            validity=0.85,
            uniqueness=0.80,
            missing_pattern="MCAR",
            duplicate_count=5
        )
        
        assert quality_metrics.completeness == 0.95
        assert quality_metrics.consistency == 0.90
        assert quality_metrics.validity == 0.85
        assert quality_metrics.uniqueness == 0.80
        assert quality_metrics.missing_pattern == "MCAR"
        assert quality_metrics.duplicate_count == 5
    
    def test_metric_validation(self):
        """Test validation of quality metrics."""
        # Test invalid completeness
        with pytest.raises(ValueError, match="Quality metrics must be between 0.0 and 1.0"):
            QualityMetrics(1.5, 0.9, 0.8, 0.7, "MCAR", 0)
        
        # Test invalid consistency
        with pytest.raises(ValueError, match="Quality metrics must be between 0.0 and 1.0"):
            QualityMetrics(0.9, -0.1, 0.8, 0.7, "MCAR", 0)
    
    def test_negative_duplicate_count(self):
        """Test validation of negative duplicate count."""
        with pytest.raises(ValueError, match="Duplicate count cannot be negative"):
            QualityMetrics(0.9, 0.8, 0.7, 0.6, "MCAR", -1)


class TestStatisticalProperties:
    """Test cases for StatisticalProperties data model."""
    
    def test_valid_statistical_properties(self):
        """Test creation of valid StatisticalProperties instance."""
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        stats = StatisticalProperties(
            distribution="normal",
            parameters={"mean": 0.0, "std": 1.0},
            correlation_matrix=correlation_matrix,
            feature_importance={"feature1": 0.8, "feature2": 0.6},
            outlier_indices=[1, 5, 10]
        )
        
        assert stats.distribution == "normal"
        assert stats.parameters["mean"] == 0.0
        assert np.array_equal(stats.correlation_matrix, correlation_matrix)
        assert stats.feature_importance["feature1"] == 0.8
        assert stats.outlier_indices == [1, 5, 10]
    
    def test_empty_distribution(self):
        """Test validation of empty distribution."""
        with pytest.raises(ValueError, match="Distribution cannot be empty"):
            StatisticalProperties("", {"mean": 0.0})
    
    def test_invalid_correlation_matrix(self):
        """Test validation of correlation matrix dimensions."""
        with pytest.raises(ValueError, match="Correlation matrix must be 2-dimensional"):
            StatisticalProperties(
                "normal",
                {"mean": 0.0},
                correlation_matrix=np.array([1, 2, 3])  # 1D array
            )


class TestTaskIdentification:
    """Test cases for TaskIdentification data model."""
    
    def test_valid_task_identification(self):
        """Test creation of valid TaskIdentification instance."""
        task = TaskIdentification(
            task_type="classification",
            task_subtype="binary",
            complexity="medium",
            confidence=0.85,
            characteristics={"balanced": True, "features": 10}
        )
        
        assert task.task_type == "classification"
        assert task.task_subtype == "binary"
        assert task.complexity == "medium"
        assert task.confidence == 0.85
        assert task.characteristics["balanced"] is True
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            TaskIdentification("classification", "binary", "medium", 2.0)
    
    def test_empty_task_type(self):
        """Test validation of empty task type."""
        with pytest.raises(ValueError, match="Task type cannot be empty"):
            TaskIdentification("", "binary", "medium", 0.8)
    
    def test_empty_task_subtype(self):
        """Test validation of empty task subtype."""
        with pytest.raises(ValueError, match="Task subtype cannot be empty"):
            TaskIdentification("classification", "", "medium", 0.8)


class TestModelRecommendation:
    """Test cases for ModelRecommendation data model."""
    
    def test_valid_model_recommendation(self):
        """Test creation of valid ModelRecommendation instance."""
        recommendation = ModelRecommendation(
            model_name="RandomForest",
            model_type="ensemble",
            confidence=0.9,
            rationale="Good for tabular data with mixed features",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            expected_performance={"accuracy": 0.85, "f1_score": 0.82}
        )
        
        assert recommendation.model_name == "RandomForest"
        assert recommendation.model_type == "ensemble"
        assert recommendation.confidence == 0.9
        assert recommendation.rationale == "Good for tabular data with mixed features"
        assert recommendation.hyperparameters["n_estimators"] == 100
        assert recommendation.expected_performance["accuracy"] == 0.85
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ModelRecommendation("RandomForest", "ensemble", 1.5, "Good model")
    
    def test_empty_model_name(self):
        """Test validation of empty model name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelRecommendation("", "ensemble", 0.9, "Good model")
    
    def test_empty_model_type(self):
        """Test validation of empty model type."""
        with pytest.raises(ValueError, match="Model type cannot be empty"):
            ModelRecommendation("RandomForest", "", 0.9, "Good model")


class TestProfileReport:
    """Test cases for ProfileReport data model."""
    
    def test_valid_profile_report(self):
        """Test creation of valid ProfileReport instance."""
        # Create required components
        file_info = FileFormat("CSV", 0.95, "text/csv")
        data_structure = DataStructure("tabular", (100, 10), 100, 1024)
        column_analysis = {"col1": ColumnType("numerical", "integer", 0.9)}
        quality_metrics = QualityMetrics(0.95, 0.90, 0.85, 0.80, "MCAR", 5)
        statistical_properties = StatisticalProperties("normal", {"mean": 0.0})
        task_identification = TaskIdentification("classification", "binary", "medium", 0.85)
        model_recommendations = [ModelRecommendation("RandomForest", "ensemble", 0.9, "Good model")]
        
        report = ProfileReport(
            file_info=file_info,
            data_structure=data_structure,
            column_analysis=column_analysis,
            quality_metrics=quality_metrics,
            statistical_properties=statistical_properties,
            domain_analysis={"domain": "general"},
            task_identification=task_identification,
            model_recommendations=model_recommendations,
            preprocessing_recommendations=["normalize", "encode_categorical"],
            resource_requirements={"memory": "2GB", "time": "30s"},
            execution_time=25.5
        )
        
        assert report.file_info.format_type == "CSV"
        assert report.data_structure.structure_type == "tabular"
        assert len(report.column_analysis) == 1
        assert report.execution_time == 25.5
        assert isinstance(report.timestamp, datetime)
    
    def test_negative_execution_time(self):
        """Test validation of negative execution time."""
        file_info = FileFormat("CSV", 0.95, "text/csv")
        data_structure = DataStructure("tabular", (100, 10), 100, 1024)
        column_analysis = {"col1": ColumnType("numerical", "integer", 0.9)}
        quality_metrics = QualityMetrics(0.95, 0.90, 0.85, 0.80, "MCAR", 5)
        statistical_properties = StatisticalProperties("normal", {"mean": 0.0})
        task_identification = TaskIdentification("classification", "binary", "medium", 0.85)
        
        with pytest.raises(ValueError, match="Execution time cannot be negative"):
            ProfileReport(
                file_info=file_info,
                data_structure=data_structure,
                column_analysis=column_analysis,
                quality_metrics=quality_metrics,
                statistical_properties=statistical_properties,
                domain_analysis={},
                task_identification=task_identification,
                model_recommendations=[],
                preprocessing_recommendations=[],
                resource_requirements={},
                execution_time=-1.0
            )
    
    def test_empty_column_analysis(self):
        """Test validation of empty column analysis."""
        file_info = FileFormat("CSV", 0.95, "text/csv")
        data_structure = DataStructure("tabular", (100, 10), 100, 1024)
        quality_metrics = QualityMetrics(0.95, 0.90, 0.85, 0.80, "MCAR", 5)
        statistical_properties = StatisticalProperties("normal", {"mean": 0.0})
        task_identification = TaskIdentification("classification", "binary", "medium", 0.85)
        
        with pytest.raises(ValueError, match="Column analysis cannot be empty"):
            ProfileReport(
                file_info=file_info,
                data_structure=data_structure,
                column_analysis={},  # Empty column analysis
                quality_metrics=quality_metrics,
                statistical_properties=statistical_properties,
                domain_analysis={},
                task_identification=task_identification,
                model_recommendations=[],
                preprocessing_recommendations=[],
                resource_requirements={},
                execution_time=1.0
            )


class TestSpecializedDataModels:
    """Test cases for specialized data model classes."""
    
    def test_numerical_analysis(self):
        """Test NumericalAnalysis data model."""
        analysis = NumericalAnalysis(
            data_type="float",
            is_continuous=True,
            range_min=0.0,
            range_max=100.0,
            distribution_type="normal",
            outlier_count=3
        )
        
        assert analysis.data_type == "float"
        assert analysis.is_continuous is True
        assert analysis.range_min == 0.0
        assert analysis.range_max == 100.0
        
        # Test invalid range
        with pytest.raises(ValueError, match="Range minimum cannot be greater than maximum"):
            NumericalAnalysis("float", True, 100.0, 0.0, "normal", 0)
        
        # Test negative outlier count
        with pytest.raises(ValueError, match="Outlier count cannot be negative"):
            NumericalAnalysis("float", True, 0.0, 100.0, "normal", -1)
    
    def test_categorical_analysis(self):
        """Test CategoricalAnalysis data model."""
        analysis = CategoricalAnalysis(
            category_type="nominal",
            cardinality=3,
            unique_values=["A", "B", "C"],
            frequency_distribution={"A": 10, "B": 15, "C": 5},
            encoding_recommendation="one_hot"
        )
        
        assert analysis.category_type == "nominal"
        assert analysis.cardinality == 3
        assert len(analysis.unique_values) == 3
        
        # Test cardinality mismatch
        with pytest.raises(ValueError, match="Unique values count must match cardinality"):
            CategoricalAnalysis("nominal", 2, ["A", "B", "C"], {}, "one_hot")
    
    def test_temporal_analysis(self):
        """Test TemporalAnalysis data model."""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=30)
        
        analysis = TemporalAnalysis(
            datetime_format="%Y-%m-%d",
            frequency="daily",
            has_seasonality=True,
            has_trend=False,
            is_stationary=True,
            time_range=(start_time, end_time)
        )
        
        assert analysis.datetime_format == "%Y-%m-%d"
        assert analysis.frequency == "daily"
        assert analysis.has_seasonality is True
        
        # Test invalid time range
        with pytest.raises(ValueError, match="Start time cannot be after end time"):
            TemporalAnalysis("%Y-%m-%d", "daily", True, False, True, (end_time, start_time))
    
    def test_missing_data_analysis(self):
        """Test MissingDataAnalysis data model."""
        analysis = MissingDataAnalysis(
            missing_percentage=0.15,
            missing_pattern_type="MCAR",
            missing_columns=["col1", "col3"],
            imputation_strategy="mean"
        )
        
        assert analysis.missing_percentage == 0.15
        assert analysis.missing_pattern_type == "MCAR"
        assert len(analysis.missing_columns) == 2
        
        # Test invalid missing percentage
        with pytest.raises(ValueError, match="Missing percentage must be between 0.0 and 1.0"):
            MissingDataAnalysis(1.5, "MCAR", [], "mean")