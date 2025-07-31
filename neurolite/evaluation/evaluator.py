"""
Comprehensive evaluation engine for NeuroLite.

Provides model evaluation capabilities with automatic metric selection,
performance analysis, and result reporting.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
import time
from pathlib import Path

from ..models.base import BaseModel, TaskType, PredictionResult
from ..data.detector import DataType
from ..core.exceptions import EvaluationError, MetricError
from ..core import get_logger
from .metrics import MetricCalculator, MetricsCollection, MetricType

logger = get_logger(__name__)


@dataclass
class EvaluationResults:
    """
    Comprehensive evaluation results for a model.
    
    Contains metrics, predictions, timing information, and metadata.
    """
    
    # Core results
    metrics: MetricsCollection
    predictions: np.ndarray
    ground_truth: np.ndarray
    probabilities: Optional[np.ndarray] = None
    
    # Performance information
    prediction_time: float = 0.0
    evaluation_time: float = 0.0
    samples_per_second: float = 0.0
    
    # Model and task information
    model_name: str = ""
    task_type: TaskType = TaskType.AUTO
    data_type: DataType = DataType.UNKNOWN
    num_samples: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate derived metrics
        if self.num_samples == 0:
            self.num_samples = len(self.predictions)
        
        if self.prediction_time > 0 and self.samples_per_second == 0:
            self.samples_per_second = self.num_samples / self.prediction_time
    
    def get_primary_score(self) -> float:
        """Get the primary metric score."""
        return self.metrics.get_primary_score()
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get a specific metric value by name."""
        metric_result = self.metrics.get_metric(name)
        return metric_result.value if metric_result else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation results to dictionary format."""
        result = {
            'metrics': self.metrics.to_dict(),
            'primary_score': self.get_primary_score(),
            'primary_metric': self.metrics.primary_metric,
            'prediction_time': self.prediction_time,
            'evaluation_time': self.evaluation_time,
            'samples_per_second': self.samples_per_second,
            'model_name': self.model_name,
            'task_type': self.task_type.value,
            'data_type': self.data_type.value,
            'num_samples': self.num_samples,
            'metadata': self.metadata
        }
        return result
    
    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save evaluation results to file.
        
        Args:
            path: Path to save results (supports .json, .yaml, .pkl)
        """
        import json
        from pathlib import Path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = self.to_dict()
        
        # Convert numpy arrays to lists for JSON serialization
        if path.suffix.lower() == '.json':
            data['predictions'] = self.predictions.tolist()
            data['ground_truth'] = self.ground_truth.tolist()
            if self.probabilities is not None:
                data['probabilities'] = self.probabilities.tolist()
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                data['predictions'] = self.predictions.tolist()
                data['ground_truth'] = self.ground_truth.tolist()
                if self.probabilities is not None:
                    data['probabilities'] = self.probabilities.tolist()
                
                with open(path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise EvaluationError("PyYAML is required to save YAML files")
        
        elif path.suffix.lower() == '.pkl':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        
        else:
            raise EvaluationError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Evaluation results saved to {path}")
    
    def __str__(self) -> str:
        """String representation of evaluation results."""
        lines = [
            f"Evaluation Results for {self.model_name}",
            f"Task: {self.task_type.value}",
            f"Samples: {self.num_samples}",
            f"Prediction Time: {self.prediction_time:.3f}s ({self.samples_per_second:.1f} samples/sec)",
            "",
            str(self.metrics)
        ]
        return "\n".join(lines)


class EvaluationEngine:
    """
    Comprehensive evaluation engine for model performance assessment.
    
    Provides automatic metric selection, performance benchmarking,
    and detailed result analysis.
    """
    
    def __init__(self, metric_calculator: Optional[MetricCalculator] = None):
        """
        Initialize the evaluation engine.
        
        Args:
            metric_calculator: Custom metric calculator (optional)
        """
        self.metric_calculator = metric_calculator or MetricCalculator()
        logger.debug("Initialized EvaluationEngine")
    
    def evaluate(
        self,
        model: BaseModel,
        X_test: Union[np.ndarray, List, Any],
        y_test: Union[np.ndarray, List, Any],
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        metrics: Optional[List[MetricType]] = None,
        return_predictions: bool = True,
        benchmark_performance: bool = True
    ) -> EvaluationResults:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features/inputs
            y_test: Test targets/labels
            task_type: Type of ML task (auto-detected if None)
            data_type: Type of input data (auto-detected if None)
            metrics: Specific metrics to calculate (uses task defaults if None)
            return_predictions: Whether to include predictions in results
            benchmark_performance: Whether to measure prediction performance
            
        Returns:
            EvaluationResults with comprehensive evaluation information
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            logger.info(f"Starting evaluation of {model.__class__.__name__}")
            eval_start_time = time.time()
            
            # Validate inputs
            if not model.is_trained:
                raise EvaluationError("Model must be trained before evaluation")
            
            # Convert inputs to numpy arrays
            y_test = np.asarray(y_test)
            num_samples = len(y_test)
            
            # Auto-detect task and data types if not provided
            if task_type is None:
                task_type = self._detect_task_type(y_test, model)
            
            if data_type is None:
                data_type = self._detect_data_type(X_test)
            
            logger.debug(f"Detected task type: {task_type.value}, data type: {data_type.value}")
            
            # Make predictions with timing
            pred_start_time = time.time()
            prediction_result = model.predict(X_test)
            prediction_time = time.time() - pred_start_time
            
            # Extract predictions and probabilities
            y_pred = prediction_result.predictions
            y_proba = prediction_result.probabilities
            
            logger.debug(f"Generated predictions for {num_samples} samples in {prediction_time:.3f}s")
            
            # Calculate metrics
            metrics_collection = self.metric_calculator.calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                task_type=task_type,
                y_proba=y_proba,
                metrics=metrics
            )
            
            evaluation_time = time.time() - eval_start_time
            
            # Create evaluation results
            results = EvaluationResults(
                metrics=metrics_collection,
                predictions=y_pred if return_predictions else np.array([]),
                ground_truth=y_test if return_predictions else np.array([]),
                probabilities=y_proba if return_predictions and y_proba is not None else None,
                prediction_time=prediction_time,
                evaluation_time=evaluation_time,
                model_name=model.__class__.__name__,
                task_type=task_type,
                data_type=data_type,
                num_samples=num_samples,
                metadata={
                    'model_config': model.get_config() if hasattr(model, 'get_config') else {},
                    'benchmark_performance': benchmark_performance
                }
            )
            
            logger.info(f"Evaluation completed in {evaluation_time:.3f}s")
            logger.info(f"Primary metric ({metrics_collection.primary_metric}): {results.get_primary_score():.4f}")
            
            return results
            
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {str(e)}")
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, BaseModel],
        X_test: Union[np.ndarray, List, Any],
        y_test: Union[np.ndarray, List, Any],
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        metrics: Optional[List[MetricType]] = None
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate multiple models on the same test data.
        
        Args:
            models: Dictionary of model name -> model instance
            X_test: Test features/inputs
            y_test: Test targets/labels
            task_type: Type of ML task (auto-detected if None)
            data_type: Type of input data (auto-detected if None)
            metrics: Specific metrics to calculate (uses task defaults if None)
            
        Returns:
            Dictionary of model name -> evaluation results
        """
        logger.info(f"Evaluating {len(models)} models")
        
        results = {}
        for model_name, model in models.items():
            try:
                logger.info(f"Evaluating model: {model_name}")
                result = self.evaluate(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    task_type=task_type,
                    data_type=data_type,
                    metrics=metrics
                )
                result.model_name = model_name
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                # Continue with other models
        
        # Log comparison summary
        if results:
            self._log_model_comparison(results)
        
        return results
    
    def compare_models(
        self,
        evaluation_results: Dict[str, EvaluationResults],
        sort_by_primary: bool = True
    ) -> List[Tuple[str, EvaluationResults]]:
        """
        Compare multiple model evaluation results.
        
        Args:
            evaluation_results: Dictionary of model name -> evaluation results
            sort_by_primary: Whether to sort by primary metric
            
        Returns:
            List of (model_name, results) tuples sorted by performance
        """
        if not evaluation_results:
            return []
        
        # Convert to list of tuples
        model_results = list(evaluation_results.items())
        
        if sort_by_primary:
            # Get the first result to determine if higher is better for primary metric
            first_result = next(iter(evaluation_results.values()))
            primary_metric_name = first_result.metrics.primary_metric
            primary_metric_result = first_result.metrics.get_metric(primary_metric_name)
            
            if primary_metric_result:
                higher_is_better = primary_metric_result.higher_is_better
                
                # Sort by primary metric score
                model_results.sort(
                    key=lambda x: x[1].get_primary_score(),
                    reverse=higher_is_better
                )
        
        return model_results
    
    def _detect_task_type(self, y_test: np.ndarray, model: BaseModel) -> TaskType:
        """
        Auto-detect task type from test labels and model capabilities.
        
        Args:
            y_test: Test labels/targets
            model: Model instance
            
        Returns:
            Detected task type
        """
        # Analyze the target values first
        unique_values = np.unique(y_test)
        
        # Check if values are continuous (regression) or discrete (classification)
        if len(unique_values) > 20 or np.issubdtype(y_test.dtype, np.floating):
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
    
    def _detect_data_type(self, X_test: Any) -> DataType:
        """
        Auto-detect data type from test features.
        
        Args:
            X_test: Test features/inputs
            
        Returns:
            Detected data type
        """
        # Simple heuristics for data type detection
        if hasattr(X_test, 'shape'):
            shape = X_test.shape
            
            # Image data typically has 3 or 4 dimensions
            if len(shape) >= 3:
                return DataType.IMAGE
            
            # Tabular data typically has 2 dimensions
            elif len(shape) == 2:
                return DataType.TABULAR
        
        # Default to tabular for most ML tasks
        return DataType.TABULAR
    
    def _log_model_comparison(self, results: Dict[str, EvaluationResults]) -> None:
        """Log a comparison summary of multiple models."""
        if not results:
            return
        
        logger.info("Model Comparison Summary:")
        logger.info("-" * 50)
        
        # Sort by primary metric
        sorted_results = self.compare_models(results)
        
        for model_name, result in sorted_results:
            primary_score = result.get_primary_score()
            primary_metric = result.metrics.primary_metric
            logger.info(f"{model_name}: {primary_metric} = {primary_score:.4f}")


# Convenience function
def evaluate_model(
    model: BaseModel,
    X_test: Union[np.ndarray, List, Any],
    y_test: Union[np.ndarray, List, Any],
    task_type: Optional[TaskType] = None,
    data_type: Optional[DataType] = None,
    metrics: Optional[List[MetricType]] = None
) -> EvaluationResults:
    """
    Convenience function to evaluate a single model.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features/inputs
        y_test: Test targets/labels
        task_type: Type of ML task (auto-detected if None)
        data_type: Type of input data (auto-detected if None)
        metrics: Specific metrics to calculate (uses task defaults if None)
        
    Returns:
        EvaluationResults with comprehensive evaluation information
    """
    engine = EvaluationEngine()
    return engine.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
        data_type=data_type,
        metrics=metrics
    )