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
    ProfileReport
)

from .core.exceptions import (
    NeuroLiteException,
    UnsupportedFormatError,
    InsufficientDataError,
    ResourceLimitError
)

from .detectors import (
    FileDetector,
    DataTypeDetector,
    QualityDetector
)

from .analyzers import (
    StatisticalAnalyzer,
    ComplexityAnalyzer
)

# Simple DataProfiler class for easy usage
class DataProfiler:
    """
    Main interface for NeuroLite data profiling.
    
    Provides a simple interface to analyze datasets with comprehensive
    data detection, quality assessment, and type classification.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize DataProfiler.
        
        Args:
            confidence_threshold: Minimum confidence threshold for classifications
        """
        self.confidence_threshold = confidence_threshold
        self.file_detector = FileDetector()
        self.type_detector = DataTypeDetector()
        self.quality_detector = QualityDetector(confidence_threshold=confidence_threshold)
        self.statistical_analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    def analyze(self, data_source, quick: bool = False):
        """
        Analyze a data source and return comprehensive results.
        
        Args:
            data_source: File path or pandas DataFrame to analyze
            quick: If True, perform quick analysis with basic metrics only
            
        Returns:
            Dictionary with analysis results (simplified ProfileReport)
        """
        import pandas as pd
        
        # Load data if it's a file path
        if isinstance(data_source, str):
            # Simple file loading - in full implementation this would use FileDetector
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_source)
            else:
                df = pd.read_csv(data_source)  # Default to CSV
        else:
            df = data_source
        
        # Perform analysis
        results = {}
        
        # Data structure analysis
        data_structure = self.file_detector.detect_structure(df)
        results['data_structure'] = data_structure
        
        # Quality analysis
        quality_metrics = self.quality_detector.analyze_quality(df)
        results['quality_metrics'] = quality_metrics
        
        if not quick:
            # Detailed type analysis
            column_analysis = self.type_detector.classify_columns(df)
            results['column_analysis'] = column_analysis
            
            # Missing data analysis
            missing_analysis = self.quality_detector.detect_missing_patterns(df)
            results['missing_analysis'] = missing_analysis
            
            # Statistical analysis
            statistical_properties = self.statistical_analyzer.analyze_comprehensive(df)
            results['statistical_properties'] = statistical_properties
        
        return results

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