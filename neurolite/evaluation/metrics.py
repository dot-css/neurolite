"""
Metrics calculation and automatic metric selection for different task types.

Provides comprehensive metric calculation capabilities with automatic
selection based on task type and data characteristics.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ..models.base import TaskType
from ..core.exceptions import MetricError
from ..core import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics available."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"
    F1_WEIGHTED = "f1_weighted"
    ROC_AUC = "roc_auc"
    ROC_AUC_MACRO = "roc_auc_macro"
    ROC_AUC_MICRO = "roc_auc_micro"
    PRECISION_MACRO = "precision_macro"
    PRECISION_MICRO = "precision_micro"
    PRECISION_WEIGHTED = "precision_weighted"
    RECALL_MACRO = "recall_macro"
    RECALL_MICRO = "recall_micro"
    RECALL_WEIGHTED = "recall_weighted"
    LOG_LOSS = "log_loss"
    
    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mape"
    MEDIAN_ABSOLUTE_ERROR = "median_ae"
    EXPLAINED_VARIANCE = "explained_variance"
    
    # Multi-class specific
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"


@dataclass
class MetricResult:
    """Result of a single metric calculation."""
    name: str
    value: float
    description: str
    higher_is_better: bool = True


@dataclass
class MetricsCollection:
    """Collection of calculated metrics."""
    task_type: TaskType
    metrics: Dict[str, MetricResult]
    primary_metric: str
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get a specific metric by name."""
        return self.metrics.get(name)
    
    def get_primary_score(self) -> float:
        """Get the primary metric score."""
        primary = self.metrics.get(self.primary_metric)
        return primary.value if primary else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {name: result.value for name, result in self.metrics.items()}
    
    def __str__(self) -> str:
        """String representation of metrics."""
        lines = [f"Task Type: {self.task_type.value}"]
        lines.append(f"Primary Metric: {self.primary_metric}")
        lines.append("Metrics:")
        for name, result in self.metrics.items():
            marker = " *" if name == self.primary_metric else "  "
            lines.append(f"{marker} {name}: {result.value:.4f}")
        return "\n".join(lines)


class MetricCalculator:
    """
    Comprehensive metric calculation system with automatic metric selection.
    
    Provides task-appropriate metrics and handles different data types
    and prediction formats.
    """
    
    # Task-specific metric configurations
    TASK_METRICS = {
        TaskType.BINARY_CLASSIFICATION: {
            'primary': MetricType.F1_SCORE,
            'metrics': [
                MetricType.ACCURACY,
                MetricType.PRECISION,
                MetricType.RECALL,
                MetricType.F1_SCORE,
                MetricType.ROC_AUC,
                MetricType.LOG_LOSS
            ]
        },
        TaskType.MULTICLASS_CLASSIFICATION: {
            'primary': MetricType.F1_WEIGHTED,
            'metrics': [
                MetricType.ACCURACY,
                MetricType.PRECISION_WEIGHTED,
                MetricType.RECALL_WEIGHTED,
                MetricType.F1_WEIGHTED,
                MetricType.F1_MACRO,
                MetricType.ROC_AUC_MACRO
            ]
        },
        TaskType.CLASSIFICATION: {
            'primary': MetricType.ACCURACY,
            'metrics': [
                MetricType.ACCURACY,
                MetricType.PRECISION_WEIGHTED,
                MetricType.RECALL_WEIGHTED,
                MetricType.F1_WEIGHTED,
                MetricType.F1_MACRO
            ]
        },
        TaskType.REGRESSION: {
            'primary': MetricType.RMSE,
            'metrics': [
                MetricType.MAE,
                MetricType.MSE,
                MetricType.RMSE,
                MetricType.R2_SCORE,
                MetricType.MEAN_ABSOLUTE_PERCENTAGE_ERROR
            ]
        },
        TaskType.LINEAR_REGRESSION: {
            'primary': MetricType.R2_SCORE,
            'metrics': [
                MetricType.MAE,
                MetricType.MSE,
                MetricType.RMSE,
                MetricType.R2_SCORE,
                MetricType.EXPLAINED_VARIANCE
            ]
        }
    }
    
    def __init__(self):
        """Initialize the metric calculator."""
        self._sklearn_available = self._check_sklearn()
    
    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn.metrics
            return True
        except ImportError:
            logger.warning("scikit-learn not available, some metrics may not work")
            return False
    
    def get_metrics_for_task(self, task_type: TaskType) -> Dict[str, Any]:
        """
        Get appropriate metrics configuration for a task type.
        
        Args:
            task_type: Type of ML task
            
        Returns:
            Dictionary with primary metric and list of metrics
        """
        # Handle auto-detection and fallbacks
        if task_type == TaskType.AUTO:
            task_type = TaskType.CLASSIFICATION
        
        # Map specific task types to general categories
        task_mapping = {
            TaskType.IMAGE_CLASSIFICATION: TaskType.CLASSIFICATION,
            TaskType.TEXT_CLASSIFICATION: TaskType.CLASSIFICATION,
            TaskType.SENTIMENT_ANALYSIS: TaskType.BINARY_CLASSIFICATION,
            TaskType.POLYNOMIAL_REGRESSION: TaskType.REGRESSION,
        }
        
        mapped_task = task_mapping.get(task_type, task_type)
        
        if mapped_task in self.TASK_METRICS:
            return self.TASK_METRICS[mapped_task].copy()
        
        # Default fallback
        logger.warning(f"No specific metrics defined for task {task_type.value}, using classification defaults")
        return self.TASK_METRICS[TaskType.CLASSIFICATION].copy()
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: TaskType,
        y_proba: Optional[np.ndarray] = None,
        metrics: Optional[List[MetricType]] = None
    ) -> MetricsCollection:
        """
        Calculate comprehensive metrics for predictions.
        
        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted labels/values
            task_type: Type of ML task
            y_proba: Predicted probabilities (for classification)
            metrics: Specific metrics to calculate (if None, uses task defaults)
            
        Returns:
            MetricsCollection with calculated metrics
            
        Raises:
            MetricError: If metric calculation fails
        """
        if not self._sklearn_available:
            raise MetricError("sklearn", "scikit-learn is required for metric calculations")
        
        try:
            # Get task configuration
            task_config = self.get_metrics_for_task(task_type)
            metrics_to_calc = metrics or task_config['metrics']
            primary_metric = task_config['primary'].value
            
            # Validate inputs
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            if y_true.shape[0] != y_pred.shape[0]:
                raise MetricError(
                    "shape_mismatch",
                    f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
                )
            
            # Determine if this is classification or regression
            is_classification = self._is_classification_task(task_type)
            
            # Calculate metrics
            calculated_metrics = {}
            
            if is_classification:
                calculated_metrics = self._calculate_classification_metrics(
                    y_true, y_pred, y_proba, metrics_to_calc
                )
            else:
                calculated_metrics = self._calculate_regression_metrics(
                    y_true, y_pred, metrics_to_calc
                )
            
            return MetricsCollection(
                task_type=task_type,
                metrics=calculated_metrics,
                primary_metric=primary_metric
            )
            
        except Exception as e:
            raise MetricError("calculation", f"Failed to calculate metrics: {str(e)}")
    
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
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        metrics: List[MetricType]
    ) -> Dict[str, MetricResult]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
        )
        
        results = {}
        
        # Determine if binary or multiclass
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_labels) <= 2
        
        for metric in metrics:
            try:
                if metric == MetricType.ACCURACY:
                    value = accuracy_score(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Accuracy",
                        value=value,
                        description="Fraction of correct predictions",
                        higher_is_better=True
                    )
                
                elif metric == MetricType.PRECISION:
                    avg = 'binary' if is_binary else 'weighted'
                    value = precision_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name="Precision",
                        value=value,
                        description="Fraction of positive predictions that are correct",
                        higher_is_better=True
                    )
                
                elif metric == MetricType.RECALL:
                    avg = 'binary' if is_binary else 'weighted'
                    value = recall_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name="Recall",
                        value=value,
                        description="Fraction of positive cases that are identified",
                        higher_is_better=True
                    )
                
                elif metric == MetricType.F1_SCORE:
                    avg = 'binary' if is_binary else 'weighted'
                    value = f1_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name="F1 Score",
                        value=value,
                        description="Harmonic mean of precision and recall",
                        higher_is_better=True
                    )
                
                elif metric in [MetricType.PRECISION_MACRO, MetricType.PRECISION_MICRO, MetricType.PRECISION_WEIGHTED]:
                    avg = metric.value.split('_')[1]  # Extract 'macro', 'micro', or 'weighted'
                    value = precision_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name=f"Precision ({avg})",
                        value=value,
                        description=f"Precision with {avg} averaging",
                        higher_is_better=True
                    )
                
                elif metric in [MetricType.RECALL_MACRO, MetricType.RECALL_MICRO, MetricType.RECALL_WEIGHTED]:
                    avg = metric.value.split('_')[1]
                    value = recall_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name=f"Recall ({avg})",
                        value=value,
                        description=f"Recall with {avg} averaging",
                        higher_is_better=True
                    )
                
                elif metric in [MetricType.F1_MACRO, MetricType.F1_MICRO, MetricType.F1_WEIGHTED]:
                    avg = metric.value.split('_')[1]
                    value = f1_score(y_true, y_pred, average=avg, zero_division=0)
                    results[metric.value] = MetricResult(
                        name=f"F1 Score ({avg})",
                        value=value,
                        description=f"F1 score with {avg} averaging",
                        higher_is_better=True
                    )
                
                elif metric in [MetricType.ROC_AUC, MetricType.ROC_AUC_MACRO, MetricType.ROC_AUC_MICRO]:
                    if y_proba is not None:
                        if metric == MetricType.ROC_AUC:
                            if is_binary:
                                # For binary classification, use probabilities of positive class
                                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                                    proba_pos = y_proba[:, 1]
                                else:
                                    proba_pos = y_proba.ravel()
                                value = roc_auc_score(y_true, proba_pos)
                            else:
                                value = roc_auc_score(y_true, y_proba, multi_class='ovr')
                        else:
                            avg = 'macro' if 'macro' in metric.value else 'micro'
                            value = roc_auc_score(y_true, y_proba, multi_class='ovr', average=avg)
                        
                        results[metric.value] = MetricResult(
                            name="ROC AUC",
                            value=value,
                            description="Area under the ROC curve",
                            higher_is_better=True
                        )
                
                elif metric == MetricType.LOG_LOSS:
                    if y_proba is not None:
                        value = log_loss(y_true, y_proba)
                        results[metric.value] = MetricResult(
                            name="Log Loss",
                            value=value,
                            description="Logarithmic loss",
                            higher_is_better=False
                        )
                
                elif metric == MetricType.COHEN_KAPPA:
                    value = cohen_kappa_score(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Cohen's Kappa",
                        value=value,
                        description="Inter-rater agreement statistic",
                        higher_is_better=True
                    )
                
                elif metric == MetricType.MATTHEWS_CORRCOEF:
                    value = matthews_corrcoef(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Matthews Correlation Coefficient",
                        value=value,
                        description="Correlation between predicted and actual values",
                        higher_is_better=True
                    )
                
            except Exception as e:
                logger.warning(f"Failed to calculate {metric.value}: {e}")
        
        return results
    
    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[MetricType]
    ) -> Dict[str, MetricResult]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
            median_absolute_error, explained_variance_score
        )
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == MetricType.MAE:
                    value = mean_absolute_error(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Mean Absolute Error",
                        value=value,
                        description="Average absolute difference between predicted and actual values",
                        higher_is_better=False
                    )
                
                elif metric == MetricType.MSE:
                    value = mean_squared_error(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Mean Squared Error",
                        value=value,
                        description="Average squared difference between predicted and actual values",
                        higher_is_better=False
                    )
                
                elif metric == MetricType.RMSE:
                    value = np.sqrt(mean_squared_error(y_true, y_pred))
                    results[metric.value] = MetricResult(
                        name="Root Mean Squared Error",
                        value=value,
                        description="Square root of mean squared error",
                        higher_is_better=False
                    )
                
                elif metric == MetricType.R2_SCORE:
                    value = r2_score(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="R² Score",
                        value=value,
                        description="Coefficient of determination",
                        higher_is_better=True
                    )
                
                elif metric == MetricType.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                    # Calculate MAPE manually to handle zero values
                    mask = y_true != 0
                    if np.any(mask):
                        value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    else:
                        value = float('inf')
                    results[metric.value] = MetricResult(
                        name="Mean Absolute Percentage Error",
                        value=value,
                        description="Average percentage error",
                        higher_is_better=False
                    )
                
                elif metric == MetricType.MEDIAN_ABSOLUTE_ERROR:
                    value = median_absolute_error(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Median Absolute Error",
                        value=value,
                        description="Median absolute difference between predicted and actual values",
                        higher_is_better=False
                    )
                
                elif metric == MetricType.EXPLAINED_VARIANCE:
                    value = explained_variance_score(y_true, y_pred)
                    results[metric.value] = MetricResult(
                        name="Explained Variance",
                        value=value,
                        description="Fraction of variance explained by the model",
                        higher_is_better=True
                    )
                
            except Exception as e:
                logger.warning(f"Failed to calculate {metric.value}: {e}")
        
        return results


# Convenience functions
def get_metrics_for_task(task_type: TaskType) -> Dict[str, Any]:
    """
    Get appropriate metrics for a task type.
    
    Args:
        task_type: Type of ML task
        
    Returns:
        Dictionary with primary metric and list of metrics
    """
    calculator = MetricCalculator()
    return calculator.get_metrics_for_task(task_type)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.CLASSIFICATION
) -> MetricsCollection:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        task_type: Type of classification task
        
    Returns:
        MetricsCollection with calculated metrics
    """
    calculator = MetricCalculator()
    return calculator.calculate_metrics(y_true, y_pred, task_type, y_proba)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType = TaskType.REGRESSION
) -> MetricsCollection:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task_type: Type of regression task
        
    Returns:
        MetricsCollection with calculated metrics
    """
    calculator = MetricCalculator()
    return calculator.calculate_metrics(y_true, y_pred, task_type)