"""
Cross-validation utilities for NeuroLite.

Provides comprehensive cross-validation capabilities with stratified sampling,
performance analysis, and statistical significance testing.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import time
from copy import deepcopy

from ..models.base import BaseModel, TaskType
from ..data.detector import DataType
from ..core.exceptions import EvaluationError
from ..core import get_logger
from .metrics import MetricCalculator, MetricsCollection, MetricType
from .evaluator import EvaluationResults

logger = get_logger(__name__)


@dataclass
class CrossValidationResults:
    """
    Results from cross-validation evaluation.
    
    Contains fold-wise results, aggregated statistics, and performance metrics.
    """
    
    # Core results
    fold_results: List[EvaluationResults]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    
    # Configuration
    num_folds: int
    task_type: TaskType
    data_type: DataType
    model_name: str = ""
    
    # Performance information
    total_time: float = 0.0
    mean_fold_time: float = 0.0
    
    # Statistical information
    confidence_intervals: Dict[str, Tuple[float, float]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.metadata is None:
            self.metadata = {}
        
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        
        # Calculate mean fold time
        if self.fold_results and self.mean_fold_time == 0.0:
            fold_times = [result.evaluation_time for result in self.fold_results]
            self.mean_fold_time = np.mean(fold_times)
    
    def get_primary_score(self) -> float:
        """Get the mean primary metric score."""
        if self.fold_results:
            primary_metric = self.fold_results[0].metrics.primary_metric
            return self.mean_metrics.get(primary_metric, 0.0)
        return 0.0
    
    def get_primary_std(self) -> float:
        """Get the standard deviation of the primary metric."""
        if self.fold_results:
            primary_metric = self.fold_results[0].metrics.primary_metric
            return self.std_metrics.get(primary_metric, 0.0)
        return 0.0
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """
        Get summary statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with mean, std, min, max, and confidence interval
        """
        if metric_name not in self.mean_metrics:
            return {}
        
        # Extract metric values from all folds
        values = []
        for result in self.fold_results:
            metric_result = result.metrics.get_metric(metric_name)
            if metric_result:
                values.append(metric_result.value)
        
        if not values:
            return {}
        
        values = np.array(values)
        
        summary = {
            'mean': self.mean_metrics[metric_name],
            'std': self.std_metrics[metric_name],
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
        # Add confidence interval if available
        if metric_name in self.confidence_intervals:
            ci_lower, ci_upper = self.confidence_intervals[metric_name]
            summary['ci_lower'] = ci_lower
            summary['ci_upper'] = ci_upper
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cross-validation results to dictionary format."""
        result = {
            'num_folds': self.num_folds,
            'model_name': self.model_name,
            'task_type': self.task_type.value,
            'data_type': self.data_type.value,
            'total_time': self.total_time,
            'mean_fold_time': self.mean_fold_time,
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'confidence_intervals': self.confidence_intervals,
            'primary_score': self.get_primary_score(),
            'primary_std': self.get_primary_std(),
            'metadata': self.metadata
        }
        
        # Add fold-wise results summary
        result['fold_scores'] = []
        if self.fold_results:
            primary_metric = self.fold_results[0].metrics.primary_metric
            for i, fold_result in enumerate(self.fold_results):
                fold_score = fold_result.get_metric(primary_metric) or 0.0
                result['fold_scores'].append({
                    'fold': i + 1,
                    'score': fold_score,
                    'time': fold_result.evaluation_time
                })
        
        return result
    
    def __str__(self) -> str:
        """String representation of cross-validation results."""
        lines = [
            f"Cross-Validation Results for {self.model_name}",
            f"Task: {self.task_type.value}",
            f"Folds: {self.num_folds}",
            f"Total Time: {self.total_time:.3f}s (avg: {self.mean_fold_time:.3f}s per fold)",
            ""
        ]
        
        # Add primary metric summary
        if self.fold_results:
            primary_metric = self.fold_results[0].metrics.primary_metric
            primary_mean = self.get_primary_score()
            primary_std = self.get_primary_std()
            lines.append(f"Primary Metric ({primary_metric}): {primary_mean:.4f} ± {primary_std:.4f}")
            lines.append("")
        
        # Add all metrics summary
        lines.append("All Metrics:")
        for metric_name in sorted(self.mean_metrics.keys()):
            mean_val = self.mean_metrics[metric_name]
            std_val = self.std_metrics[metric_name]
            lines.append(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        return "\n".join(lines)


class CrossValidator:
    """
    Comprehensive cross-validation system with stratified sampling.
    
    Provides k-fold, stratified k-fold, and custom cross-validation
    with statistical analysis and performance benchmarking.
    """
    
    def __init__(self, metric_calculator: Optional[MetricCalculator] = None):
        """
        Initialize the cross-validator.
        
        Args:
            metric_calculator: Custom metric calculator (optional)
        """
        self.metric_calculator = metric_calculator or MetricCalculator()
        logger.debug("Initialized CrossValidator")
    
    def cross_validate(
        self,
        model: BaseModel,
        X: Union[np.ndarray, List, Any],
        y: Union[np.ndarray, List, Any],
        cv: Union[int, Any] = 5,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        metrics: Optional[List[MetricType]] = None,
        stratify: bool = True,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        return_fold_results: bool = True,
        n_jobs: int = 1
    ) -> CrossValidationResults:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate (will be cloned for each fold)
            X: Training features/inputs
            y: Training targets/labels
            cv: Number of folds or cross-validation splitter
            task_type: Type of ML task (auto-detected if None)
            data_type: Type of input data (auto-detected if None)
            metrics: Specific metrics to calculate (uses task defaults if None)
            stratify: Whether to use stratified sampling for classification
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            return_fold_results: Whether to return detailed fold results
            n_jobs: Number of parallel jobs (currently not implemented)
            
        Returns:
            CrossValidationResults with comprehensive validation information
            
        Raises:
            EvaluationError: If cross-validation fails
        """
        try:
            logger.info(f"Starting cross-validation of {model.__class__.__name__}")
            cv_start_time = time.time()
            
            # Convert inputs to numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)
            
            # Auto-detect task and data types if not provided
            if task_type is None:
                task_type = self._detect_task_type(y, model)
            
            if data_type is None:
                data_type = self._detect_data_type(X)
            
            logger.debug(f"Detected task type: {task_type.value}, data type: {data_type.value}")
            
            # Create cross-validation splitter
            cv_splitter = self._create_cv_splitter(
                cv=cv,
                y=y,
                task_type=task_type,
                stratify=stratify,
                shuffle=shuffle,
                random_state=random_state
            )
            
            # Perform cross-validation
            fold_results = []
            fold_times = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                logger.debug(f"Processing fold {fold_idx + 1}/{cv_splitter.get_n_splits()}")
                
                fold_start_time = time.time()
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone and train model for this fold
                fold_model = self._clone_model(model)
                fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                from .evaluator import EvaluationEngine
                evaluator = EvaluationEngine(self.metric_calculator)
                
                fold_result = evaluator.evaluate(
                    model=fold_model,
                    X_test=X_val,
                    y_test=y_val,
                    task_type=task_type,
                    data_type=data_type,
                    metrics=metrics,
                    return_predictions=return_fold_results
                )
                
                fold_time = time.time() - fold_start_time
                fold_times.append(fold_time)
                fold_results.append(fold_result)
                
                logger.debug(f"Fold {fold_idx + 1} completed in {fold_time:.3f}s")
            
            total_time = time.time() - cv_start_time
            
            # Aggregate results
            aggregated_results = self._aggregate_fold_results(
                fold_results=fold_results,
                task_type=task_type,
                data_type=data_type,
                model_name=model.__class__.__name__,
                total_time=total_time
            )
            
            logger.info(f"Cross-validation completed in {total_time:.3f}s")
            logger.info(f"Mean primary score: {aggregated_results.get_primary_score():.4f} ± {aggregated_results.get_primary_std():.4f}")
            
            return aggregated_results
            
        except Exception as e:
            raise EvaluationError(f"Cross-validation failed: {str(e)}")
    
    def cross_validate_multiple_models(
        self,
        models: Dict[str, BaseModel],
        X: Union[np.ndarray, List, Any],
        y: Union[np.ndarray, List, Any],
        cv: Union[int, Any] = 5,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        metrics: Optional[List[MetricType]] = None,
        **kwargs
    ) -> Dict[str, CrossValidationResults]:
        """
        Cross-validate multiple models on the same data.
        
        Args:
            models: Dictionary of model name -> model instance
            X: Training features/inputs
            y: Training targets/labels
            cv: Number of folds or cross-validation splitter
            task_type: Type of ML task (auto-detected if None)
            data_type: Type of input data (auto-detected if None)
            metrics: Specific metrics to calculate (uses task defaults if None)
            **kwargs: Additional arguments for cross_validate
            
        Returns:
            Dictionary of model name -> cross-validation results
        """
        logger.info(f"Cross-validating {len(models)} models")
        
        results = {}
        for model_name, model in models.items():
            try:
                logger.info(f"Cross-validating model: {model_name}")
                result = self.cross_validate(
                    model=model,
                    X=X,
                    y=y,
                    cv=cv,
                    task_type=task_type,
                    data_type=data_type,
                    metrics=metrics,
                    **kwargs
                )
                result.model_name = model_name
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to cross-validate model {model_name}: {e}")
                # Continue with other models
        
        # Log comparison summary
        if results:
            self._log_cv_comparison(results)
        
        return results
    
    def _create_cv_splitter(
        self,
        cv: Union[int, Any],
        y: np.ndarray,
        task_type: TaskType,
        stratify: bool,
        shuffle: bool,
        random_state: Optional[int]
    ) -> Any:
        """
        Create appropriate cross-validation splitter.
        
        Args:
            cv: Number of folds or existing splitter
            y: Target values
            task_type: Type of ML task
            stratify: Whether to use stratified sampling
            shuffle: Whether to shuffle data
            random_state: Random seed
            
        Returns:
            Cross-validation splitter
        """
        try:
            from sklearn.model_selection import (
                KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
            )
        except ImportError:
            raise EvaluationError("scikit-learn is required for cross-validation")
        
        # If cv is already a splitter, return it
        if hasattr(cv, 'split'):
            return cv
        
        # Determine if we should use stratified sampling
        use_stratified = (
            stratify and 
            self._is_classification_task(task_type) and
            len(np.unique(y)) > 1
        )
        
        # Create appropriate splitter
        if use_stratified:
            splitter = StratifiedKFold(
                n_splits=cv,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            splitter = KFold(
                n_splits=cv,
                shuffle=shuffle,
                random_state=random_state
            )
        
        logger.debug(f"Created {splitter.__class__.__name__} with {cv} folds")
        return splitter
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """
        Clone a model for cross-validation.
        
        Args:
            model: Model to clone
            
        Returns:
            Cloned model instance
        """
        try:
            # Try to use sklearn's clone if available
            from sklearn.base import clone
            if hasattr(model, 'sklearn_model'):
                # For sklearn adapters, clone the underlying model
                cloned_sklearn = clone(model.sklearn_model)
                cloned_model = model.__class__(cloned_sklearn, **model.get_config())
                return cloned_model
        except ImportError:
            pass
        
        # Fallback to deep copy
        try:
            return deepcopy(model)
        except Exception as e:
            logger.warning(f"Failed to clone model, creating new instance: {e}")
            # Last resort: create new instance with same config
            return model.__class__(**model.get_config())
    
    def _aggregate_fold_results(
        self,
        fold_results: List[EvaluationResults],
        task_type: TaskType,
        data_type: DataType,
        model_name: str,
        total_time: float
    ) -> CrossValidationResults:
        """
        Aggregate results from all folds.
        
        Args:
            fold_results: List of evaluation results from each fold
            task_type: Type of ML task
            data_type: Type of input data
            model_name: Name of the model
            total_time: Total cross-validation time
            
        Returns:
            Aggregated cross-validation results
        """
        if not fold_results:
            raise EvaluationError("No fold results to aggregate")
        
        # Collect all metric values
        all_metrics = {}
        for result in fold_results:
            for metric_name, metric_result in result.metrics.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_result.value)
        
        # Calculate mean and standard deviation
        mean_metrics = {}
        std_metrics = {}
        confidence_intervals = {}
        
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            mean_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            # Calculate 95% confidence interval
            if len(values) > 1:
                from scipy import stats
                try:
                    ci = stats.t.interval(
                        confidence=0.95,
                        df=len(values) - 1,
                        loc=mean_metrics[metric_name],
                        scale=stats.sem(values)
                    )
                    confidence_intervals[metric_name] = ci
                except ImportError:
                    # Fallback to normal approximation
                    margin = 1.96 * std_metrics[metric_name] / np.sqrt(len(values))
                    ci = (
                        mean_metrics[metric_name] - margin,
                        mean_metrics[metric_name] + margin
                    )
                    confidence_intervals[metric_name] = ci
        
        return CrossValidationResults(
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            num_folds=len(fold_results),
            task_type=task_type,
            data_type=data_type,
            model_name=model_name,
            total_time=total_time,
            confidence_intervals=confidence_intervals,
            metadata={
                'num_samples_per_fold': [len(result.predictions) for result in fold_results],
                'fold_times': [result.evaluation_time for result in fold_results]
            }
        )
    
    def _detect_task_type(self, y: np.ndarray, model: BaseModel) -> TaskType:
        """Auto-detect task type from labels and model capabilities."""
        # Analyze the target values first
        unique_values = np.unique(y)
        
        # Check if values are continuous (regression) or discrete (classification)
        if len(unique_values) > 20 or np.issubdtype(y.dtype, np.floating):
            # Likely regression
            detected_task = TaskType.REGRESSION
        else:
            # Likely classification
            if len(unique_values) == 2:
                detected_task = TaskType.BINARY_CLASSIFICATION
            else:
                detected_task = TaskType.MULTICLASS_CLASSIFICATION
        
        # Check model capabilities to validate
        if hasattr(model, 'capabilities') and model.capabilities.supported_tasks:
            supported_tasks = model.capabilities.supported_tasks
            
            # If detected task is supported, use it
            if detected_task in supported_tasks:
                return detected_task
            
            # If model only supports one task type, use that
            if len(supported_tasks) == 1:
                return supported_tasks[0]
        
        return detected_task
    
    def _detect_data_type(self, X: Any) -> DataType:
        """Auto-detect data type from features."""
        if hasattr(X, 'shape'):
            shape = X.shape
            if len(shape) >= 3:
                return DataType.IMAGE
            elif len(shape) == 2:
                return DataType.TABULAR
        
        return DataType.TABULAR
    
    def _is_classification_task(self, task_type: TaskType) -> bool:
        """Check if task type is classification."""
        classification_tasks = {
            TaskType.CLASSIFICATION,
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.TEXT_CLASSIFICATION,
            TaskType.SENTIMENT_ANALYSIS
        }
        return task_type in classification_tasks
    
    def _log_cv_comparison(self, results: Dict[str, CrossValidationResults]) -> None:
        """Log a comparison summary of cross-validation results."""
        if not results:
            return
        
        logger.info("Cross-Validation Comparison Summary:")
        logger.info("-" * 60)
        
        # Sort by primary metric - handle potential comparison issues
        try:
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1].get_primary_score(),
                reverse=True  # Assume higher is better for primary metric
            )
        except (TypeError, AttributeError):
            # Fallback to unsorted if comparison fails
            sorted_results = list(results.items())
        
        for model_name, result in sorted_results:
            try:
                primary_score = result.get_primary_score()
                primary_std = result.get_primary_std()
                primary_metric = result.fold_results[0].metrics.primary_metric if result.fold_results else "unknown"
                logger.info(f"{model_name}: {primary_metric} = {primary_score:.4f} ± {primary_std:.4f}")
            except (AttributeError, TypeError, IndexError):
                # Handle mock objects or incomplete results
                logger.info(f"{model_name}: evaluation completed")


# Convenience function
def cross_validate_model(
    model: BaseModel,
    X: Union[np.ndarray, List, Any],
    y: Union[np.ndarray, List, Any],
    cv: Union[int, Any] = 5,
    task_type: Optional[TaskType] = None,
    data_type: Optional[DataType] = None,
    metrics: Optional[List[MetricType]] = None,
    **kwargs
) -> CrossValidationResults:
    """
    Convenience function to cross-validate a single model.
    
    Args:
        model: Model to cross-validate
        X: Training features/inputs
        y: Training targets/labels
        cv: Number of folds or cross-validation splitter
        task_type: Type of ML task (auto-detected if None)
        data_type: Type of input data (auto-detected if None)
        metrics: Specific metrics to calculate (uses task defaults if None)
        **kwargs: Additional arguments for cross_validate
        
    Returns:
        CrossValidationResults with comprehensive validation information
    """
    validator = CrossValidator()
    return validator.cross_validate(
        model=model,
        X=X,
        y=y,
        cv=cv,
        task_type=task_type,
        data_type=data_type,
        metrics=metrics,
        **kwargs
    )