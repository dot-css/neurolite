"""
NeuroLite Public API - Simple interface for automated data analysis and ML recommendations.

This module provides the main public API that allows users to analyze any dataset
with minimal code (maximum 3 lines) and get comprehensive insights.
"""

from typing import Union, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

from .core.data_profiler import DataProfiler
from .core.data_models import ProfileReport, QuickReport
from .core.exceptions import NeuroLiteException


def analyze(data_source: Union[str, pd.DataFrame, np.ndarray], 
           quick: bool = False,
           confidence_threshold: float = 0.8,
           max_processing_time: Optional[float] = 5.0,
           enable_parallel: bool = True) -> Union[ProfileReport, QuickReport]:
    """
    Analyze any dataset with comprehensive AI/ML insights in one function call.
    
    This is the main entry point for NeuroLite. It automatically detects data types,
    assesses quality, performs statistical analysis, identifies ML tasks, and 
    recommends appropriate models and preprocessing steps.
    
    Args:
        data_source: File path (str), pandas DataFrame, or numpy array to analyze
        quick: If True, performs fast basic analysis. If False, comprehensive analysis
        confidence_threshold: Minimum confidence for classifications (0.0-1.0)
        max_processing_time: Maximum processing time in seconds (None for no limit)
        enable_parallel: Whether to use parallel processing for better performance
        
    Returns:
        ProfileReport: Comprehensive analysis results (if quick=False)
        QuickReport: Basic analysis results (if quick=True)
        
    Raises:
        NeuroLiteException: If analysis fails or data format is unsupported
        
    Examples:
        >>> import neurolite as nl
        >>> 
        >>> # Analyze a CSV file
        >>> report = nl.analyze('data.csv')
        >>> print(f"Detected task: {report.task_identification.task_type}")
        >>> print(f"Recommended models: {[m.model_name for m in report.model_recommendations]}")
        >>> 
        >>> # Quick analysis of a DataFrame
        >>> import pandas as pd
        >>> df = pd.read_csv('data.csv')
        >>> quick_report = nl.analyze(df, quick=True)
        >>> print(f"Dataset shape: {quick_report.basic_stats['shape']}")
        >>> 
        >>> # Analyze with custom settings
        >>> report = nl.analyze('large_data.csv', 
        ...                    confidence_threshold=0.9,
        ...                    max_processing_time=10.0)
    """
    try:
        # Initialize profiler with optimized settings
        profiler = DataProfiler(
            confidence_threshold=confidence_threshold,
            enable_graceful_degradation=True,
            max_processing_time=max_processing_time,
            max_memory_usage_mb=1024  # 1GB default limit
        )
        
        if quick:
            return profiler.quick_analyze(data_source)
        else:
            # Use optimized analysis for large datasets
            if isinstance(data_source, str):
                # For files, check size and use appropriate method
                file_path = Path(data_source)
                file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
                
                if file_size_mb > 100:  # > 100MB, use optimized analysis
                    return profiler.analyze_large_dataset(data_source, use_sampling=True)
                else:
                    return profiler.analyze(data_source)
            elif isinstance(data_source, (pd.DataFrame, np.ndarray)):
                # For in-memory data, check size
                if isinstance(data_source, pd.DataFrame):
                    memory_mb = data_source.memory_usage(deep=True).sum() / (1024 * 1024)
                    row_count = len(data_source)
                else:
                    memory_mb = data_source.nbytes / (1024 * 1024)
                    row_count = data_source.shape[0] if data_source.ndim > 0 else 0
                
                if memory_mb > 100 or row_count > 50000:  # Large dataset
                    return profiler.analyze_large_dataset(data_source, use_sampling=True)
                else:
                    return profiler.analyze(data_source)
            else:
                return profiler.analyze(data_source)
                
    except Exception as e:
        if isinstance(e, NeuroLiteException):
            raise
        else:
            raise NeuroLiteException(f"Analysis failed: {str(e)}")


def quick_analyze(data_source: Union[str, pd.DataFrame, np.ndarray]) -> QuickReport:
    """
    Perform quick analysis with basic metrics for fast initial assessment.
    
    This is a convenience function equivalent to analyze(data_source, quick=True).
    Provides essential information about the dataset with minimal processing time.
    
    Args:
        data_source: File path, pandas DataFrame, or numpy array to analyze
        
    Returns:
        QuickReport: Basic analysis results including shape, types, and quick recommendations
        
    Examples:
        >>> import neurolite as nl
        >>> 
        >>> # Quick analysis of any data source
        >>> report = nl.quick_analyze('data.csv')
        >>> print(f"Shape: {report.basic_stats['shape']}")
        >>> print(f"Missing values: {report.basic_stats['missing_values']}")
    """
    return analyze(data_source, quick=True)


def get_recommendations(data_source: Union[str, pd.DataFrame, np.ndarray],
                       task_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get ML model and preprocessing recommendations for a dataset.
    
    This function focuses specifically on providing actionable recommendations
    for machine learning workflows.
    
    Args:
        data_source: File path, pandas DataFrame, or numpy array to analyze
        task_type: Optional task type hint ('classification', 'regression', 'clustering')
        
    Returns:
        Dictionary with 'models' and 'preprocessing' recommendation lists
        
    Examples:
        >>> import neurolite as nl
        >>> 
        >>> # Get recommendations for any dataset
        >>> recs = nl.get_recommendations('data.csv')
        >>> print("Recommended models:", recs['models'])
        >>> print("Preprocessing steps:", recs['preprocessing'])
        >>> 
        >>> # Get recommendations with task hint
        >>> recs = nl.get_recommendations('data.csv', task_type='classification')
    """
    try:
        report = analyze(data_source, quick=False)
        
        return {
            'models': [rec.model_name for rec in report.model_recommendations],
            'preprocessing': report.preprocessing_recommendations,
            'task_detected': report.task_identification.task_type,
            'confidence': report.task_identification.confidence
        }
        
    except Exception as e:
        raise NeuroLiteException(f"Failed to get recommendations: {str(e)}")


def detect_data_types(data_source: Union[str, pd.DataFrame, np.ndarray]) -> Dict[str, str]:
    """
    Detect data types for all columns in a dataset.
    
    This function focuses specifically on data type detection and classification.
    
    Args:
        data_source: File path, pandas DataFrame, or numpy array to analyze
        
    Returns:
        Dictionary mapping column names to detected data types
        
    Examples:
        >>> import neurolite as nl
        >>> 
        >>> # Detect data types
        >>> types = nl.detect_data_types('data.csv')
        >>> print(types)
        >>> # {'age': 'numerical', 'name': 'text', 'category': 'categorical'}
    """
    try:
        report = analyze(data_source, quick=False)
        
        return {
            col_name: col_type.primary_type 
            for col_name, col_type in report.column_analysis.items()
        }
        
    except Exception as e:
        raise NeuroLiteException(f"Failed to detect data types: {str(e)}")


def assess_data_quality(data_source: Union[str, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    Assess data quality metrics for a dataset.
    
    This function focuses specifically on data quality assessment.
    
    Args:
        data_source: File path, pandas DataFrame, or numpy array to analyze
        
    Returns:
        Dictionary with quality metrics and recommendations
        
    Examples:
        >>> import neurolite as nl
        >>> 
        >>> # Assess data quality
        >>> quality = nl.assess_data_quality('data.csv')
        >>> print(f"Completeness: {quality['completeness']:.2%}")
        >>> print(f"Missing pattern: {quality['missing_pattern']}")
    """
    try:
        report = analyze(data_source, quick=False)
        quality = report.quality_metrics
        
        return {
            'completeness': quality.completeness,
            'consistency': quality.consistency,
            'validity': quality.validity,
            'uniqueness': quality.uniqueness,
            'missing_pattern': quality.missing_pattern,
            'duplicate_count': quality.duplicate_count,
            'overall_score': (quality.completeness + quality.consistency + 
                            quality.validity + quality.uniqueness) / 4
        }
        
    except Exception as e:
        raise NeuroLiteException(f"Failed to assess data quality: {str(e)}")


# Convenience aliases for common use cases
profile = analyze  # Alias for main function
scan = quick_analyze  # Alias for quick analysis


# Version info
__version__ = "0.1.0"
__all__ = [
    'analyze',
    'quick_analyze', 
    'get_recommendations',
    'detect_data_types',
    'assess_data_quality',
    'profile',
    'scan'
]