"""
Evaluation and metrics system for NeuroLite.

Provides comprehensive model evaluation capabilities including:
- Automatic metric selection based on task type
- Performance evaluation with appropriate metrics
- Visualization generation for results
- Cross-validation utilities
"""

from .metrics import (
    MetricCalculator,
    MetricsCollection,
    MetricResult,
    MetricType,
    get_metrics_for_task,
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .evaluator import (
    EvaluationEngine,
    EvaluationResults,
    evaluate_model
)
from .validator import (
    CrossValidator,
    CrossValidationResults,
    cross_validate_model
)
from .visualizer import (
    VisualizationEngine,
    generate_confusion_matrix,
    generate_roc_curve,
    generate_performance_plots
)

__all__ = [
    # Metrics
    'MetricCalculator',
    'MetricsCollection',
    'MetricResult',
    'MetricType',
    'get_metrics_for_task',
    'calculate_classification_metrics',
    'calculate_regression_metrics',
    
    # Evaluation
    'EvaluationEngine',
    'EvaluationResults',
    'evaluate_model',
    
    # Cross-validation
    'CrossValidator',
    'CrossValidationResults',
    'cross_validate_model',
    
    # Visualization
    'VisualizationEngine',
    'generate_confusion_matrix',
    'generate_roc_curve',
    'generate_performance_plots'
]