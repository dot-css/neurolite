"""
Central DataProfiler orchestrator for NeuroLite library.

This module provides the main DataProfiler class that coordinates all detectors,
analyzers, and recommenders to provide comprehensive data analysis with minimal code.
"""

import time
import warnings
from typing import Union, Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .data_models import (
    ProfileReport, QuickReport, FileFormat, DataStructure, 
    ColumnType, QualityMetrics, StatisticalProperties,
    TaskIdentification, ModelRecommendation
)
from .exceptions import (
    NeuroLiteException, UnsupportedFormatError, InsufficientDataError,
    ResourceLimitError, ProcessingError, ValidationError,
    handle_graceful_degradation, validate_input_data
)
from ..detectors import FileDetector, DataTypeDetector, QualityDetector, DomainDetector
from ..analyzers import StatisticalAnalyzer, ComplexityAnalyzer
from ..recommenders import ModelRecommender, PreprocessingRecommender
from .performance import LazyDataLoader, SamplingStrategy, ParallelProcessor, MemoryMonitor


class DataProfiler:
    """
    Central orchestrator for comprehensive data analysis and ML recommendations.
    
    The DataProfiler coordinates all detection, analysis, and recommendation components
    to provide a unified interface for data profiling with minimal user code.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.8,
                 enable_graceful_degradation: bool = True,
                 max_processing_time: Optional[float] = None,
                 max_memory_usage_mb: Optional[float] = None):
        """
        Initialize the DataProfiler with configuration options.
        
        Args:
            confidence_threshold: Minimum confidence threshold for classifications (0.0-1.0)
            enable_graceful_degradation: Whether to continue analysis if components fail
            max_processing_time: Maximum processing time in seconds (None for no limit)
            max_memory_usage_mb: Maximum memory usage in MB (None for no limit)
        """
        # Validate configuration
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValidationError("confidence_threshold", confidence_threshold, 
                                "Must be between 0.0 and 1.0")
        
        self.confidence_threshold = confidence_threshold
        self.enable_graceful_degradation = enable_graceful_degradation
        self.max_processing_time = max_processing_time
        self.max_memory_usage_mb = max_memory_usage_mb
        
        # Initialize components
        self._initialize_components()
        
        # Performance optimization components
        self.parallel_processor = ParallelProcessor(max_workers=min(mp.cpu_count(), 4))
        self.memory_monitor = MemoryMonitor(max_memory_usage_mb)
        
        # Analysis state
        self._analysis_start_time = None
        self._failed_components = []
        
    def _initialize_components(self):
        """Initialize all detector, analyzer, and recommender components."""
        try:
            # Detectors
            self.file_detector = FileDetector()
            self.data_type_detector = DataTypeDetector()
            self.quality_detector = QualityDetector(
                confidence_threshold=self.confidence_threshold
            )
            self.domain_detector = DomainDetector()
            
            # Analyzers
            self.statistical_analyzer = StatisticalAnalyzer(
                confidence_level=0.95
            )
            self.complexity_analyzer = ComplexityAnalyzer()
            
            # Recommenders
            self.model_recommender = ModelRecommender()
            self.preprocessing_recommender = PreprocessingRecommender()
            
        except Exception as e:
            raise ProcessingError("component_initialization", e)
    
    def analyze(self, data_source: Union[str, pd.DataFrame, np.ndarray]) -> ProfileReport:
        """
        Perform comprehensive data analysis and return detailed results.
        
        This method coordinates all detectors and analyzers to provide a complete
        analysis of the dataset including format detection, type classification,
        quality assessment, statistical analysis, and ML recommendations.
        
        Args:
            data_source: File path, pandas DataFrame, or numpy array to analyze
            
        Returns:
            ProfileReport: Comprehensive analysis results
            
        Raises:
            UnsupportedFormatError: If the data format is not supported
            InsufficientDataError: If the dataset is too small for analysis
            ResourceLimitError: If resource limits are exceeded
            ProcessingError: If analysis fails
        """
        self._analysis_start_time = time.time()
        self._failed_components = []
        
        try:
            # Validate input
            validate_input_data(data_source, "data_source")
            
            # Load and prepare data
            df, file_info = self._load_data(data_source)
            
            # Check resource limits
            self._check_resource_limits(df)
            
            # Detect data structure
            data_structure = self._detect_structure(df, file_info)
            
            # Perform column analysis
            column_analysis = self._analyze_columns(df)
            
            # Assess data quality
            quality_metrics = self._assess_quality(df)
            
            # Perform statistical analysis
            statistical_properties = self._analyze_statistics(df)
            
            # Domain-specific analysis
            domain_analysis = self._analyze_domain(df, data_structure)
            
            # Task identification
            task_identification = self._identify_task(df, column_analysis)
            
            # Generate model recommendations
            model_recommendations = self._recommend_models(
                df, task_identification, statistical_properties
            )
            
            # Generate preprocessing recommendations
            preprocessing_recommendations = self._recommend_preprocessing(
                df, column_analysis, quality_metrics
            )
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resources(df, task_identification)
            
            # Calculate execution time
            execution_time = time.time() - self._analysis_start_time
            
            # Create comprehensive report
            report = ProfileReport(
                file_info=file_info,
                data_structure=data_structure,
                column_analysis=column_analysis,
                quality_metrics=quality_metrics,
                statistical_properties=statistical_properties,
                domain_analysis=domain_analysis,
                task_identification=task_identification,
                model_recommendations=model_recommendations,
                preprocessing_recommendations=preprocessing_recommendations,
                resource_requirements=resource_requirements,
                execution_time=execution_time
            )
            
            # Add warnings for failed components
            if self._failed_components and self.enable_graceful_degradation:
                report.resource_requirements['warnings'] = {
                    'failed_components': self._failed_components,
                    'message': 'Some analysis components failed but results are still valid'
                }
            
            return report
            
        except Exception as e:
            if isinstance(e, NeuroLiteException):
                raise
            else:
                raise ProcessingError("analysis_pipeline", e)
    
    def quick_analyze(self, data_source: Union[str, pd.DataFrame, np.ndarray]) -> QuickReport:
        """
        Perform quick analysis with basic metrics for fast initial assessment.
        
        This method provides essential information about the dataset with minimal
        processing time, suitable for large datasets or initial exploration.
        
        Args:
            data_source: File path, pandas DataFrame, or numpy array to analyze
            
        Returns:
            QuickReport: Basic analysis results
            
        Raises:
            UnsupportedFormatError: If the data format is not supported
            ProcessingError: If analysis fails
        """
        self._analysis_start_time = time.time()
        
        try:
            # Validate input
            validate_input_data(data_source, "data_source")
            
            # Load and prepare data (with sampling for large datasets)
            df, file_info = self._load_data(data_source, quick_mode=True)
            
            # Detect data structure
            data_structure = self._detect_structure(df, file_info)
            
            # Generate basic statistics
            basic_stats = self._generate_basic_stats(df)
            
            # Generate quick recommendations
            quick_recommendations = self._generate_quick_recommendations(df, basic_stats)
            
            # Calculate execution time
            execution_time = time.time() - self._analysis_start_time
            
            return QuickReport(
                file_info=file_info,
                data_structure=data_structure,
                basic_stats=basic_stats,
                quick_recommendations=quick_recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            if isinstance(e, NeuroLiteException):
                raise
            else:
                raise ProcessingError("quick_analysis_pipeline", e)
    
    def _load_data(self, data_source: Union[str, pd.DataFrame, np.ndarray], 
                   quick_mode: bool = False) -> tuple[pd.DataFrame, FileFormat]:
        """Load data from various sources and detect file format."""
        try:
            if isinstance(data_source, str):
                # File path provided
                file_path = Path(data_source)
                if not file_path.exists():
                    raise ValidationError("file_path", data_source, "File does not exist")
                
                # Detect file format
                file_info = self.file_detector.detect_format(str(file_path))
                
                # Load data based on format (normalize to uppercase for comparison)
                format_upper = file_info.format_type.upper()
                if format_upper == 'CSV':
                    df = pd.read_csv(file_path)
                elif format_upper == 'TSV':
                    df = pd.read_csv(file_path, sep='\t')
                elif format_upper in ['XLSX', 'XLS', 'EXCEL']:
                    df = pd.read_excel(file_path)
                elif format_upper in ['JSON', 'JSONL']:
                    df = pd.read_json(file_path)
                elif format_upper == 'PARQUET':
                    df = pd.read_parquet(file_path)
                else:
                    raise UnsupportedFormatError(file_info.format_type)
                
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
                file_info = FileFormat(
                    format_type='dataframe',
                    confidence=1.0,
                    mime_type='application/x-pandas-dataframe',
                    encoding='utf-8'
                )
                
            elif isinstance(data_source, np.ndarray):
                df = pd.DataFrame(data_source)
                file_info = FileFormat(
                    format_type='numpy_array',
                    confidence=1.0,
                    mime_type='application/x-numpy-array',
                    encoding='utf-8'
                )
                
            else:
                raise ValidationError("data_source", type(data_source), 
                                    "Must be file path, DataFrame, or numpy array")
            
            # Apply sampling for quick mode or large datasets
            if quick_mode and len(df) > 10000:
                sample_size = self._calculate_optimal_sample_size(df)
                df = df.sample(n=sample_size, random_state=42)
                file_info.metadata['sampled'] = True
                file_info.metadata['sample_size'] = len(df)
                file_info.metadata['original_size'] = len(df) if 'original_size' not in file_info.metadata else file_info.metadata['original_size']
            
            return df, file_info
            
        except Exception as e:
            if isinstance(e, NeuroLiteException):
                raise
            else:
                raise ProcessingError("data_loading", e)
    
    def _check_resource_limits(self, df: pd.DataFrame):
        """Check if the dataset exceeds configured resource limits."""
        if self.max_processing_time is not None:
            elapsed = time.time() - self._analysis_start_time
            if elapsed > self.max_processing_time:
                raise ResourceLimitError("processing_time", 
                                       f"Exceeded {self.max_processing_time}s", elapsed)
        
        if self.max_memory_usage_mb is not None:
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_usage_mb > self.max_memory_usage_mb:
                raise ResourceLimitError("memory", 
                                       f"Exceeded {self.max_memory_usage_mb}MB", 
                                       memory_usage_mb)
    
    def _detect_structure(self, df: pd.DataFrame, file_info: FileFormat) -> DataStructure:
        """Detect data structure using FileDetector."""
        try:
            return self.file_detector.detect_structure(df)
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("structure_detection")
                return handle_graceful_degradation(
                    e, "structure_detection",
                    DataStructure(
                        structure_type='tabular',
                        dimensions=(len(df), len(df.columns)),
                        sample_size=len(df),
                        memory_usage=int(df.memory_usage(deep=True).sum())
                    )
                )
            else:
                raise ProcessingError("structure_detection", e)
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        """Analyze column types using DataTypeDetector."""
        try:
            return self.data_type_detector.classify_columns(df)
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("column_analysis")
                # Provide basic fallback column analysis
                fallback_analysis = {}
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        fallback_analysis[col] = ColumnType('numerical', 'unknown', 0.1)
                    else:
                        fallback_analysis[col] = ColumnType('text', 'unknown', 0.1)
                return handle_graceful_degradation(e, "column_analysis", fallback_analysis)
            else:
                raise ProcessingError("column_analysis", e)
    
    def _assess_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """Assess data quality using QualityDetector."""
        try:
            return self.quality_detector.analyze_quality(df)
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("quality_assessment")
                return handle_graceful_degradation(
                    e, "quality_assessment",
                    QualityMetrics(
                        completeness=0.0,
                        consistency=0.0,
                        validity=0.0,
                        uniqueness=0.0,
                        missing_pattern="unknown",
                        duplicate_count=0
                    )
                )
            else:
                raise ProcessingError("quality_assessment", e)
    
    def _analyze_statistics(self, df: pd.DataFrame) -> StatisticalProperties:
        """Perform statistical analysis using StatisticalAnalyzer."""
        try:
            return self.statistical_analyzer.analyze_comprehensive(df)
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("statistical_analysis")
                return handle_graceful_degradation(
                    e, "statistical_analysis",
                    StatisticalProperties(
                        distribution="unknown",
                        parameters={}
                    )
                )
            else:
                raise ProcessingError("statistical_analysis", e)
    
    def _analyze_domain(self, df: pd.DataFrame, data_structure: DataStructure) -> Dict[str, Any]:
        """Perform domain-specific analysis using DomainDetector."""
        try:
            domain_analysis = {}
            
            # Computer vision analysis
            if data_structure.structure_type == 'image':
                cv_analysis = self.domain_detector.detect_cv_task(df)
                domain_analysis['computer_vision'] = cv_analysis
            
            # NLP analysis
            if data_structure.structure_type == 'text':
                nlp_analysis = self.domain_detector.detect_nlp_task(df)
                domain_analysis['natural_language_processing'] = nlp_analysis
            
            # Time series analysis
            if data_structure.structure_type == 'time_series':
                ts_analysis = self.domain_detector.detect_timeseries_characteristics(df)
                domain_analysis['time_series'] = ts_analysis
            
            return domain_analysis
            
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("domain_analysis")
                return handle_graceful_degradation(e, "domain_analysis", {})
            else:
                raise ProcessingError("domain_analysis", e)
    
    def _identify_task(self, df: pd.DataFrame, column_analysis: Dict[str, ColumnType]) -> TaskIdentification:
        """Identify ML task using available information."""
        try:
            # Simple task identification logic
            # In a full implementation, this would use TaskDetector
            
            # Check if there's a clear target variable
            target_candidates = []
            for col_name, col_type in column_analysis.items():
                if col_type.primary_type in ['categorical', 'numerical']:
                    if 'target' in col_name.lower() or 'label' in col_name.lower():
                        target_candidates.append(col_name)
            
            if target_candidates:
                # Supervised learning task
                target_col = target_candidates[0]
                target_type = column_analysis[target_col]
                
                if target_type.primary_type == 'categorical':
                    task_type = "classification"
                    unique_values = df[target_col].nunique()
                    task_subtype = "binary" if unique_values == 2 else "multiclass"
                else:
                    task_type = "regression"
                    task_subtype = "continuous"
            else:
                # Unsupervised learning task
                task_type = "unsupervised"
                task_subtype = "clustering"
            
            return TaskIdentification(
                task_type=task_type,
                task_subtype=task_subtype,
                complexity="medium",
                confidence=0.7,
                characteristics={
                    "num_features": len(df.columns),
                    "num_samples": len(df),
                    "target_candidates": target_candidates
                }
            )
            
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("task_identification")
                return handle_graceful_degradation(
                    e, "task_identification",
                    TaskIdentification(
                        task_type="unknown",
                        task_subtype="unknown",
                        complexity="unknown",
                        confidence=0.0
                    )
                )
            else:
                raise ProcessingError("task_identification", e)
    
    def _recommend_models(self, df: pd.DataFrame, task_identification: TaskIdentification,
                         statistical_properties: StatisticalProperties) -> List[ModelRecommendation]:
        """Generate model recommendations using ModelRecommender."""
        try:
            return self.model_recommender.recommend_models(
                task_identification, statistical_properties, df
            )
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("model_recommendations")
                return handle_graceful_degradation(e, "model_recommendations", [])
            else:
                raise ProcessingError("model_recommendations", e)
    
    def _recommend_preprocessing(self, df: pd.DataFrame, column_analysis: Dict[str, ColumnType],
                               quality_metrics: QualityMetrics) -> List[str]:
        """Generate preprocessing recommendations using PreprocessingRecommender."""
        try:
            return self.preprocessing_recommender.recommend_preprocessing_pipeline(
                df, column_analysis, quality_metrics
            )
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("preprocessing_recommendations")
                return handle_graceful_degradation(e, "preprocessing_recommendations", [])
            else:
                raise ProcessingError("preprocessing_recommendations", e)
    
    def _estimate_resources(self, df: pd.DataFrame, task_identification: TaskIdentification) -> Dict[str, Any]:
        """Estimate computational resource requirements using ComplexityAnalyzer."""
        try:
            complexity_analysis = self.complexity_analyzer.analyze_complexity(df, task_identification)
            return {
                "complexity_analysis": complexity_analysis,
                "estimated_memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "estimated_processing_time_seconds": len(df) * 0.001,  # Simple estimate
                "recommended_hardware": "cpu" if len(df) < 100000 else "gpu"
            }
        except Exception as e:
            if self.enable_graceful_degradation:
                self._failed_components.append("resource_estimation")
                return handle_graceful_degradation(
                    e, "resource_estimation",
                    {
                        "estimated_memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                        "estimated_processing_time_seconds": len(df) * 0.001,
                        "recommended_hardware": "cpu"
                    }
                )
            else:
                raise ProcessingError("resource_estimation", e)
    
    def _generate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics for quick analysis."""
        try:
            basic_stats = {
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Add basic descriptive statistics for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                basic_stats["numerical_summary"] = df[numerical_cols].describe().to_dict()
            
            # Add basic info for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                basic_stats["categorical_summary"] = {
                    col: {"unique_count": df[col].nunique(), "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None}
                    for col in categorical_cols
                }
            
            return basic_stats
            
        except Exception as e:
            return {"error": f"Failed to generate basic stats: {str(e)}"}
    
    def _generate_quick_recommendations(self, df: pd.DataFrame, basic_stats: Dict[str, Any]) -> List[str]:
        """Generate quick recommendations based on basic statistics."""
        recommendations = []
        
        try:
            # Check for missing values
            if "missing_values" in basic_stats:
                missing_cols = [col for col, count in basic_stats["missing_values"].items() if count > 0]
                if missing_cols:
                    recommendations.append(f"Handle missing values in columns: {', '.join(missing_cols[:3])}")
            
            # Check data size
            if basic_stats.get("shape", [0, 0])[0] < 1000:
                recommendations.append("Dataset is small - consider data augmentation or simpler models")
            elif basic_stats.get("shape", [0, 0])[0] > 100000:
                recommendations.append("Large dataset - consider sampling or distributed processing")
            
            # Check feature count
            if basic_stats.get("shape", [0, 0])[1] > 100:
                recommendations.append("High-dimensional data - consider dimensionality reduction")
            
            # Memory usage recommendation
            memory_mb = basic_stats.get("memory_usage_mb", 0)
            if memory_mb > 1000:
                recommendations.append("High memory usage - consider data optimization or chunked processing")
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations due to analysis error"]
    
    def _calculate_optimal_sample_size(self, df: pd.DataFrame) -> int:
        """
        Calculate optimal sample size for quick analysis based on data characteristics.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Optimal sample size for analysis
        """
        try:
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Base sample size
            base_sample_size = 5000
            
            # Adjust based on number of columns
            if total_cols > 50:
                # More columns need more samples for statistical significance
                base_sample_size = min(8000, base_sample_size + (total_cols - 50) * 20)
            elif total_cols < 10:
                # Fewer columns can work with smaller samples
                base_sample_size = max(2000, base_sample_size - (10 - total_cols) * 100)
            
            # Adjust based on memory usage
            memory_per_row = df.memory_usage(deep=True).sum() / total_rows
            if memory_per_row > 1024:  # More than 1KB per row
                # Reduce sample size for memory-intensive data
                base_sample_size = max(1000, int(base_sample_size * 0.7))
            
            # Ensure we don't sample more than available
            optimal_size = min(base_sample_size, total_rows)
            
            # Ensure minimum sample size for statistical validity
            return max(1000, optimal_size)
            
        except Exception:
            # Fallback to conservative sample size
            return min(5000, len(df))
    
    def analyze_large_dataset(self, data_source: Union[str, pd.DataFrame, np.ndarray],
                            use_sampling: bool = True, chunk_size: int = 10000) -> ProfileReport:
        """
        Analyze large datasets using streaming, sampling, and parallel processing.
        
        This method is optimized for datasets that are too large to fit in memory
        or require significant processing time. It uses lazy loading, intelligent
        sampling, and parallel processing to provide comprehensive analysis.
        
        Args:
            data_source: File path, pandas DataFrame, or numpy array to analyze
            use_sampling: Whether to use sampling for detailed analysis
            chunk_size: Size of chunks for streaming processing
            
        Returns:
            ProfileReport: Comprehensive analysis results with performance optimizations
            
        Raises:
            UnsupportedFormatError: If the data format is not supported
            ResourceLimitError: If resource limits are exceeded
            ProcessingError: If analysis fails
        """
        self._analysis_start_time = time.time()
        self._failed_components = []
        
        try:
            # Validate input
            validate_input_data(data_source, "data_source")
            
            # Initialize lazy loader for streaming
            if isinstance(data_source, str):
                lazy_loader = LazyDataLoader(data_source, chunk_size)
                metadata = lazy_loader.get_metadata()
                
                # Determine if we need sampling
                total_rows = metadata['total_rows']
                if use_sampling and total_rows > 50000:
                    # Use sampling for detailed analysis
                    sample_size = SamplingStrategy.adaptive_sample_size(
                        total_rows, len(metadata['columns']), 
                        self.max_memory_usage_mb or 500
                    )
                    
                    # Get representative sample using reservoir sampling
                    sample_df = SamplingStrategy.reservoir_sample(
                        iter(lazy_loader), sample_size
                    )
                    
                    # Perform detailed analysis on sample
                    df, file_info = sample_df, FileFormat(
                        format_type=Path(data_source).suffix[1:].upper(),
                        confidence=1.0,
                        mime_type='application/octet-stream',
                        encoding='utf-8',
                        metadata={
                            'sampled': True,
                            'sample_size': len(sample_df),
                            'original_size': total_rows
                        }
                    )
                else:
                    # Load full dataset if small enough
                    df, file_info = self._load_data(data_source)
            else:
                # For DataFrames and arrays, use existing logic with potential sampling
                df, file_info = self._load_data(data_source)
                
                # Apply sampling if dataset is large
                if use_sampling and len(df) > 50000:
                    sample_size = SamplingStrategy.adaptive_sample_size(
                        len(df), len(df.columns),
                        self.max_memory_usage_mb or 500
                    )
                    df = SamplingStrategy.random_sample(df, sample_size)
                    file_info.metadata['sampled'] = True
                    file_info.metadata['sample_size'] = len(df)
            
            # Check resource limits
            self._check_resource_limits(df)
            
            # Use parallel processing for analysis components
            analysis_tasks = [
                ('structure', self._detect_structure, df, file_info),
                ('columns', self._analyze_columns, df),
                ('quality', self._assess_quality, df),
                ('statistics', self._analyze_statistics, df)
            ]
            
            # Execute analysis tasks in parallel where possible
            analysis_results = self._execute_parallel_analysis(analysis_tasks)
            
            # Sequential tasks that depend on previous results
            data_structure = analysis_results['structure']
            column_analysis = analysis_results['columns']
            quality_metrics = analysis_results['quality']
            statistical_properties = analysis_results['statistics']
            
            # Domain-specific analysis
            domain_analysis = self._analyze_domain(df, data_structure)
            
            # Task identification
            task_identification = self._identify_task(df, column_analysis)
            
            # Generate recommendations in parallel
            recommendation_tasks = [
                ('models', self._recommend_models, df, task_identification, statistical_properties),
                ('preprocessing', self._recommend_preprocessing, df, column_analysis, quality_metrics),
                ('resources', self._estimate_resources, df, task_identification)
            ]
            
            recommendation_results = self._execute_parallel_recommendations(recommendation_tasks)
            
            # Calculate execution time
            execution_time = time.time() - self._analysis_start_time
            
            # Create comprehensive report
            report = ProfileReport(
                file_info=file_info,
                data_structure=data_structure,
                column_analysis=column_analysis,
                quality_metrics=quality_metrics,
                statistical_properties=statistical_properties,
                domain_analysis=domain_analysis,
                task_identification=task_identification,
                model_recommendations=recommendation_results['models'],
                preprocessing_recommendations=recommendation_results['preprocessing'],
                resource_requirements=recommendation_results['resources'],
                execution_time=execution_time
            )
            
            # Add performance metrics and warnings
            if self._failed_components and self.enable_graceful_degradation:
                report.resource_requirements['warnings'] = {
                    'failed_components': self._failed_components,
                    'message': 'Some analysis components failed but results are still valid'
                }
            
            # Add performance optimization info
            report.resource_requirements['performance_optimizations'] = {
                'used_sampling': file_info.metadata.get('sampled', False),
                'parallel_processing': True,
                'memory_usage_mb': self.memory_monitor.get_memory_usage(),
                'chunk_processing': isinstance(data_source, str) and use_sampling
            }
            
            return report
            
        except Exception as e:
            if isinstance(e, NeuroLiteException):
                raise
            else:
                raise ProcessingError("large_dataset_analysis", e)
    
    def _execute_parallel_analysis(self, analysis_tasks: List[tuple]) -> Dict[str, Any]:
        """Execute analysis tasks in parallel where possible."""
        results = {}
        
        try:
            # Separate independent tasks that can run in parallel
            independent_tasks = []
            dependent_tasks = []
            
            for task_name, task_func, *args in analysis_tasks:
                if task_name in ['structure', 'columns', 'quality', 'statistics']:
                    # These can run independently
                    independent_tasks.append((task_name, task_func, args))
                else:
                    dependent_tasks.append((task_name, task_func, args))
            
            # Execute independent tasks in parallel using threads (I/O bound)
            if len(independent_tasks) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(independent_tasks))) as executor:
                    future_to_task = {
                        executor.submit(task_func, *args): task_name
                        for task_name, task_func, args in independent_tasks
                    }
                    
                    for future in future_to_task:
                        task_name = future_to_task[future]
                        try:
                            results[task_name] = future.result()
                        except Exception as e:
                            if self.enable_graceful_degradation:
                                self._failed_components.append(f"parallel_{task_name}")
                                results[task_name] = self._get_fallback_result(task_name)
                            else:
                                raise ProcessingError(f"parallel_{task_name}", e)
            else:
                # Execute sequentially if only one task
                for task_name, task_func, args in independent_tasks:
                    try:
                        results[task_name] = task_func(*args)
                    except Exception as e:
                        if self.enable_graceful_degradation:
                            self._failed_components.append(task_name)
                            results[task_name] = self._get_fallback_result(task_name)
                        else:
                            raise ProcessingError(task_name, e)
            
            # Execute dependent tasks sequentially
            for task_name, task_func, args in dependent_tasks:
                try:
                    results[task_name] = task_func(*args)
                except Exception as e:
                    if self.enable_graceful_degradation:
                        self._failed_components.append(task_name)
                        results[task_name] = self._get_fallback_result(task_name)
                    else:
                        raise ProcessingError(task_name, e)
            
            return results
            
        except Exception as e:
            # Fallback to sequential execution
            warnings.warn(f"Parallel analysis failed: {e}. Falling back to sequential execution.")
            
            for task_name, task_func, *args in analysis_tasks:
                try:
                    results[task_name] = task_func(*args[0])
                except Exception as task_e:
                    if self.enable_graceful_degradation:
                        self._failed_components.append(task_name)
                        results[task_name] = self._get_fallback_result(task_name)
                    else:
                        raise ProcessingError(task_name, task_e)
            
            return results
    
    def _execute_parallel_recommendations(self, recommendation_tasks: List[tuple]) -> Dict[str, Any]:
        """Execute recommendation tasks in parallel."""
        results = {}
        
        try:
            # All recommendation tasks can run in parallel
            with ThreadPoolExecutor(max_workers=min(3, len(recommendation_tasks))) as executor:
                future_to_task = {}
                
                for task_name, task_func, *args in recommendation_tasks:
                    future = executor.submit(task_func, *args)
                    future_to_task[future] = task_name
                
                for future in future_to_task:
                    task_name = future_to_task[future]
                    try:
                        results[task_name] = future.result()
                    except Exception as e:
                        if self.enable_graceful_degradation:
                            self._failed_components.append(f"parallel_{task_name}")
                            results[task_name] = self._get_fallback_result(task_name)
                        else:
                            raise ProcessingError(f"parallel_{task_name}", e)
            
            return results
            
        except Exception as e:
            # Fallback to sequential execution
            warnings.warn(f"Parallel recommendations failed: {e}. Falling back to sequential execution.")
            
            for task_name, task_func, *args in recommendation_tasks:
                try:
                    results[task_name] = task_func(*args)
                except Exception as task_e:
                    if self.enable_graceful_degradation:
                        self._failed_components.append(task_name)
                        results[task_name] = self._get_fallback_result(task_name)
                    else:
                        raise ProcessingError(task_name, task_e)
            
            return results
    
    def _get_fallback_result(self, task_name: str) -> Any:
        """Get fallback result for failed tasks."""
        fallbacks = {
            'structure': DataStructure(
                structure_type='tabular',
                dimensions=(0, 0),
                sample_size=0,
                memory_usage=0
            ),
            'columns': {},
            'quality': QualityMetrics(
                completeness=0.0,
                consistency=0.0,
                validity=0.0,
                uniqueness=0.0,
                missing_pattern="unknown",
                duplicate_count=0
            ),
            'statistics': StatisticalProperties(
                distribution="unknown",
                parameters={}
            ),
            'models': [],
            'preprocessing': [],
            'resources': {
                "estimated_memory_mb": 0,
                "estimated_processing_time_seconds": 0,
                "recommended_hardware": "cpu"
            }
        }
        
        return fallbacks.get(task_name, None)