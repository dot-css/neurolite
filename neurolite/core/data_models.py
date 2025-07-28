"""
Core data models for NeuroLite library.

This module contains all the dataclasses and type definitions used throughout
the NeuroLite library for representing analysis results and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Literal
import numpy as np
from datetime import datetime


@dataclass
class FileFormat:
    """Represents detected file format information."""
    format_type: str
    confidence: float
    mime_type: str
    encoding: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate FileFormat data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.format_type:
            raise ValueError("Format type cannot be empty")


@dataclass
class DataStructure:
    """Represents the structure and characteristics of the dataset."""
    structure_type: Literal['tabular', 'time_series', 'image', 'text', 'audio', 'video']
    dimensions: Tuple[int, ...]
    sample_size: int
    memory_usage: int
    
    def __post_init__(self):
        """Validate DataStructure data after initialization."""
        if self.sample_size < 0:
            raise ValueError("Sample size cannot be negative")
        if self.memory_usage < 0:
            raise ValueError("Memory usage cannot be negative")
        if not self.dimensions:
            raise ValueError("Dimensions cannot be empty")


@dataclass
class ColumnType:
    """Represents the type and properties of a data column."""
    primary_type: Literal['numerical', 'categorical', 'temporal', 'text', 'binary']
    subtype: str
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ColumnType data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.subtype:
            raise ValueError("Subtype cannot be empty")


@dataclass
class QualityMetrics:
    """Represents data quality assessment metrics."""
    completeness: float
    consistency: float
    validity: float
    uniqueness: float
    missing_pattern: str
    duplicate_count: int
    
    def __post_init__(self):
        """Validate QualityMetrics data after initialization."""
        metrics = [self.completeness, self.consistency, self.validity, self.uniqueness]
        for metric in metrics:
            if not 0.0 <= metric <= 1.0:
                raise ValueError("Quality metrics must be between 0.0 and 1.0")
        if self.duplicate_count < 0:
            raise ValueError("Duplicate count cannot be negative")


@dataclass
class StatisticalProperties:
    """Represents statistical properties of the dataset."""
    distribution: str
    parameters: Dict[str, float]
    correlation_matrix: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    outlier_indices: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate StatisticalProperties data after initialization."""
        if not self.distribution:
            raise ValueError("Distribution cannot be empty")
        if self.correlation_matrix is not None and self.correlation_matrix.ndim != 2:
            raise ValueError("Correlation matrix must be 2-dimensional")


@dataclass
class TaskIdentification:
    """Represents identified ML task characteristics."""
    task_type: str
    task_subtype: str
    complexity: str
    confidence: float
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate TaskIdentification data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.task_type:
            raise ValueError("Task type cannot be empty")
        if not self.task_subtype:
            raise ValueError("Task subtype cannot be empty")


@dataclass
class ModelRecommendation:
    """Represents a recommended ML model with metadata."""
    model_name: str
    model_type: str
    confidence: float
    rationale: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    expected_performance: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ModelRecommendation data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.model_type:
            raise ValueError("Model type cannot be empty")


@dataclass
class ProfileReport:
    """Comprehensive analysis report containing all detection and analysis results."""
    file_info: FileFormat
    data_structure: DataStructure
    column_analysis: Dict[str, ColumnType]
    quality_metrics: QualityMetrics
    statistical_properties: StatisticalProperties
    domain_analysis: Dict[str, Any]
    task_identification: TaskIdentification
    model_recommendations: List[ModelRecommendation]
    preprocessing_recommendations: List[str]
    resource_requirements: Dict[str, Any]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate ProfileReport data after initialization."""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        if not self.column_analysis:
            raise ValueError("Column analysis cannot be empty")


@dataclass
class QuickReport:
    """Simplified analysis report for quick analysis mode."""
    file_info: FileFormat
    data_structure: DataStructure
    basic_stats: Dict[str, Any]
    quick_recommendations: List[str]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate QuickReport data after initialization."""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")


# Additional specialized data models for specific analysis types

@dataclass
class NumericalAnalysis:
    """Detailed analysis results for numerical data."""
    data_type: Literal['integer', 'float']
    is_continuous: bool
    range_min: float
    range_max: float
    distribution_type: str
    outlier_count: int
    
    def __post_init__(self):
        """Validate NumericalAnalysis data after initialization."""
        if self.range_min > self.range_max:
            raise ValueError("Range minimum cannot be greater than maximum")
        if self.outlier_count < 0:
            raise ValueError("Outlier count cannot be negative")


@dataclass
class CategoricalAnalysis:
    """Detailed analysis results for categorical data."""
    category_type: Literal['nominal', 'ordinal']
    cardinality: int
    unique_values: List[str]
    frequency_distribution: Dict[str, int]
    encoding_recommendation: str
    
    def __post_init__(self):
        """Validate CategoricalAnalysis data after initialization."""
        if self.cardinality < 0:
            raise ValueError("Cardinality cannot be negative")
        if len(self.unique_values) != self.cardinality:
            raise ValueError("Unique values count must match cardinality")


@dataclass
class TemporalAnalysis:
    """Detailed analysis results for temporal data."""
    datetime_format: str
    frequency: Optional[str]
    has_seasonality: bool
    has_trend: bool
    is_stationary: bool
    time_range: Tuple[datetime, datetime]
    
    def __post_init__(self):
        """Validate TemporalAnalysis data after initialization."""
        if not self.datetime_format:
            raise ValueError("Datetime format cannot be empty")
        if self.time_range[0] > self.time_range[1]:
            raise ValueError("Start time cannot be after end time")


@dataclass
class TextAnalysis:
    """Detailed analysis results for text data."""
    text_type: Literal['natural_language', 'categorical_text', 'structured_text', 'mixed']
    language: Optional[str]
    encoding: str
    avg_length: float
    max_length: int
    min_length: int
    unique_ratio: float
    contains_special_chars: bool
    contains_numbers: bool
    readability_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate TextAnalysis data after initialization."""
        if self.avg_length < 0:
            raise ValueError("Average length cannot be negative")
        if self.max_length < 0 or self.min_length < 0:
            raise ValueError("Length values cannot be negative")
        if self.min_length > self.max_length:
            raise ValueError("Minimum length cannot be greater than maximum length")
        if not 0.0 <= self.unique_ratio <= 1.0:
            raise ValueError("Unique ratio must be between 0.0 and 1.0")
        if self.readability_score is not None and not 0.0 <= self.readability_score <= 100.0:
            raise ValueError("Readability score must be between 0.0 and 100.0")


@dataclass
class MissingDataAnalysis:
    """Analysis results for missing data patterns."""
    missing_percentage: float
    missing_pattern_type: Literal['MCAR', 'MAR', 'MNAR', 'UNKNOWN']
    missing_columns: List[str]
    imputation_strategy: str
    
    def __post_init__(self):
        """Validate MissingDataAnalysis data after initialization."""
        if not 0.0 <= self.missing_percentage <= 1.0:
            raise ValueError("Missing percentage must be between 0.0 and 1.0")
        if not self.imputation_strategy:
            raise ValueError("Imputation strategy cannot be empty")


# Domain-specific analysis data models

@dataclass
class CVTaskAnalysis:
    """Analysis results for computer vision tasks."""
    task_type: Literal['classification', 'object_detection', 'segmentation', 'unknown']
    task_subtype: str
    confidence: float
    num_classes: Optional[int] = None
    annotation_format: Optional[str] = None
    image_characteristics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate CVTaskAnalysis data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError("Number of classes must be positive")


@dataclass
class NLPTaskAnalysis:
    """Analysis results for NLP tasks."""
    task_type: Literal['sentiment', 'classification', 'ner', 'qa', 'conversation', 'unknown']
    task_subtype: str
    confidence: float
    text_characteristics: Dict[str, Any] = field(default_factory=dict)
    sequence_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate NLPTaskAnalysis data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class TimeSeriesAnalysis:
    """Analysis results for time series data."""
    series_type: Literal['univariate', 'multivariate']
    has_trend: bool
    has_seasonality: bool
    is_stationary: bool
    frequency: Optional[str]
    seasonality_period: Optional[int] = None
    trend_strength: Optional[float] = None
    seasonal_strength: Optional[float] = None
    recommended_task: Literal['forecasting', 'classification', 'anomaly_detection'] = 'forecasting'
    
    def __post_init__(self):
        """Validate TimeSeriesAnalysis data after initialization."""
        if self.trend_strength is not None and not 0.0 <= self.trend_strength <= 1.0:
            raise ValueError("Trend strength must be between 0.0 and 1.0")
        if self.seasonal_strength is not None and not 0.0 <= self.seasonal_strength <= 1.0:
            raise ValueError("Seasonal strength must be between 0.0 and 1.0")
        if self.seasonality_period is not None and self.seasonality_period < 1:
            raise ValueError("Seasonality period must be positive")


# Preprocessing recommendation data models

@dataclass
class ScalingRecommendation:
    """Recommendation for feature scaling strategies."""
    scaling_type: Literal['standardization', 'normalization', 'robust_scaling', 'none']
    rationale: str
    confidence: float
    affected_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ScalingRecommendation data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.rationale:
            raise ValueError("Rationale cannot be empty")


@dataclass
class EncodingRecommendation:
    """Recommendation for categorical encoding strategies."""
    encoding_type: Literal['one_hot', 'label_encoding', 'target_encoding', 'binary_encoding', 'none']
    rationale: str
    confidence: float
    affected_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate EncodingRecommendation data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.rationale:
            raise ValueError("Rationale cannot be empty")


@dataclass
class FeatureEngineeringRecommendation:
    """Recommendation for feature engineering strategies."""
    technique: str
    rationale: str
    confidence: float
    affected_columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_benefit: str = ""
    
    def __post_init__(self):
        """Validate FeatureEngineeringRecommendation data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.rationale:
            raise ValueError("Rationale cannot be empty")
        if not self.technique:
            raise ValueError("Technique cannot be empty")


@dataclass
class PreprocessingPipeline:
    """Complete preprocessing pipeline recommendation."""
    scaling_recommendations: List[ScalingRecommendation] = field(default_factory=list)
    encoding_recommendations: List[EncodingRecommendation] = field(default_factory=list)
    feature_engineering_recommendations: List[FeatureEngineeringRecommendation] = field(default_factory=list)
    pipeline_order: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    estimated_processing_time: float = 0.0
    
    def __post_init__(self):
        """Validate PreprocessingPipeline data after initialization."""
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError("Overall confidence must be between 0.0 and 1.0")
        if self.estimated_processing_time < 0:
            raise ValueError("Estimated processing time cannot be negative")


# Task detection specific data models

@dataclass
class SupervisedTaskAnalysis:
    """Analysis results for supervised learning tasks."""
    task_type: Literal['classification', 'regression']
    task_subtype: str  # 'binary', 'multiclass', 'linear', 'non_linear'
    confidence: float
    target_characteristics: Dict[str, Any] = field(default_factory=dict)
    dataset_balance: Dict[str, Any] = field(default_factory=dict)
    complexity_indicators: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate SupervisedTaskAnalysis data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.task_subtype:
            raise ValueError("Task subtype cannot be empty")


@dataclass
class UnsupervisedTaskAnalysis:
    """Analysis results for unsupervised learning tasks."""
    clustering_potential: float
    optimal_clusters: Optional[int]
    dimensionality_reduction_needed: bool
    confidence: float
    clustering_characteristics: Dict[str, Any] = field(default_factory=dict)
    dimensionality_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate UnsupervisedTaskAnalysis data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.clustering_potential <= 1.0:
            raise ValueError("Clustering potential must be between 0.0 and 1.0")
        if self.optimal_clusters is not None and self.optimal_clusters < 2:
            raise ValueError("Optimal clusters must be at least 2")