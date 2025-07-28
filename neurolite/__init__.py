"""
NeuroLite - Automated AI/ML library for intelligent data detection and model recommendation.
"""

from .core.data_models import (
    FileFormat,
    DataStructure,
    ColumnType,
    QualityMetrics,
    StatisticalProperties,
    TaskIdentification,
    ModelRecommendation,
    ProfileReport,
    QuickReport
)

from .core.exceptions import (
    NeuroLiteException,
    UnsupportedFormatError,
    InsufficientDataError,
    ResourceLimitError
)

from .core.data_profiler import DataProfiler

from .detectors import (
    FileDetector,
    DataTypeDetector,
    QualityDetector
)

from .analyzers import (
    StatisticalAnalyzer,
    ComplexityAnalyzer
)

__version__ = "0.1.0"
__all__ = [
    "DataProfiler",
    "FileFormat",
    "DataStructure", 
    "ColumnType",
    "QualityMetrics",
    "StatisticalProperties",
    "TaskIdentification",
    "ModelRecommendation",
    "ProfileReport",
    "QuickReport",
    "FileDetector",
    "DataTypeDetector",
    "QualityDetector",
    "StatisticalAnalyzer",
    "ComplexityAnalyzer",
    "NeuroLiteException",
    "UnsupportedFormatError",
    "InsufficientDataError",
    "ResourceLimitError"
]