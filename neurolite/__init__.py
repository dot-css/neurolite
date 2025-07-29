"""
NeuroLite - Automated AI/ML library for intelligent data detection and model recommendation.

Simple 3-line usage:
    import neurolite as nl
    report = nl.analyze('your_data.csv')
    print(report.model_recommendations)
"""

# Main public API - simple functions for common use cases
from .api import (
    analyze,
    quick_analyze,
    get_recommendations,
    detect_data_types,
    assess_data_quality,
    profile,
    scan
)

# Visualization and formatting functions
from .visualization import (
    format_summary,
    create_dataframe_summary,
    export_report
)

# Core data models for advanced users
from .core.data_models import (
    ProfileReport,
    QuickReport,
    ModelRecommendation,
    TaskIdentification,
    QualityMetrics
)

# Exceptions
from .core.exceptions import (
    NeuroLiteException,
    UnsupportedFormatError,
    InsufficientDataError,
    ResourceLimitError
)

# Advanced API for power users
from .core.data_profiler import DataProfiler

__version__ = "0.1.0"

# Public API - prioritize simple functions
__all__ = [
    # Main API functions (most users will only need these)
    "analyze",
    "quick_analyze", 
    "get_recommendations",
    "detect_data_types",
    "assess_data_quality",
    
    # Convenience aliases
    "profile",
    "scan",
    
    # Visualization and formatting
    "format_summary",
    "create_dataframe_summary",
    "export_report",
    
    # Result objects
    "ProfileReport",
    "QuickReport",
    "ModelRecommendation",
    "TaskIdentification",
    "QualityMetrics",
    
    # Exceptions
    "NeuroLiteException",
    "UnsupportedFormatError",
    "InsufficientDataError",
    "ResourceLimitError",
    
    # Advanced API
    "DataProfiler"
]