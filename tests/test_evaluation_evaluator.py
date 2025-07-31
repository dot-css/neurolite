"""
Unit tests for evaluation engine.

Tests the EvaluationEngine, EvaluationResults, and related functionality
for comprehensive model evaluation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from neurolite.evaluation.evaluator import (
    EvaluationEngine, EvaluationResults, evaluate_model
)
from neurolite.evaluation.metrics import MetricsCollection, MetricResult
from neurolite.models.base import BaseModel, TaskType, PredictionResult, ModelCapabilities
from neurolite.data.detector import DataType
from neurolite.core.exceptions import EvaluationError


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, supported_tasks=None, **kwargs):
        super().__init__(**kwargs)
        self.is_trained = True
        self._predictions = None
        self._probabilities = None
        self._supported_tasks = supported_tasks or [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION]
    
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=self._supported_tasks,
            supported_data_types=[DataType.TABULAR],
            framework="mock"
        )
    
    def fit(self, X, y, validation_data=None, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        if self._predictions is not None:
            predictions = self._predictions
        else:
            predictions = np.random.randint(0, 2, len(X))
        
        probabilities = self._probabilities
        if probabilities is None and len(X) > 0:
            probabilities = np.random.rand(len(X), 2)
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self
    
    def set_predictions(self, predictions, probabilities=None):
        """Set predictions for testing."""
        self._predictions = predictions
        self._probabilities = probabilities


class TestEvaluationResults:
    """Test EvaluationResults functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock metrics collection
        metrics = {
            'accuracy': MetricResult('Accuracy', 0.85, 'Test accuracy'),
            'precision': MetricResult('Precision', 0.80, 'Test precision')
        }
        
        self.metrics_collection = MetricsCollection(
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=metrics,
            primary_metric='accuracy'
        )
        
        self.results = EvaluationResults(
            metrics=self.metrics_collection,
            predictions=np.array([0, 1, 1, 0, 1]),
            ground_truth=np.array([0, 1, 0, 0, 1]),
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]]),
            prediction_time=0.1,
            evaluation_time=0.2,
            model_name="TestModel",
            task_type=TaskType.BINARY_CLASSIFICATION,
            data_type=DataType.TABULAR
        )
    
    def test_post_init(self):
        """Test post-initialization calculations."""
        assert self.results.num_samples == 5
        assert self.results.samples_per_second == 50.0  # 5 samples / 0.1 seconds
        assert self.results.metadata is not None
    
    def test_get_primary_score(self):
        """Test getting primary metric score."""
        score = self.results.get_primary_score()
        assert score == 0.85
    
    def test_get_metric(self):
        """Test getting specific metric value."""
        accuracy = self.results.get_metric('accuracy')
        assert accuracy == 0.85
        
        missing = self.results.get_metric('missing')
        assert missing is None
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result_dict = self.results.to_dict()
        
        assert result_dict['primary_score'] == 0.85
        assert result_dict['primary_metric'] == 'accuracy'
        assert result_dict['model_name'] == 'TestModel'
        assert result_dict['task_type'] == 'binary_classification'
        assert result_dict['num_samples'] == 5
        assert 'metrics' in result_dict
    
    def test_save_results_json(self):
        """Test saving results to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "results.json"
            self.results.save_results(save_path)
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            assert data['primary_score'] == 0.85
            assert data['model_name'] == 'TestModel'
            assert isinstance(data['predictions'], list)
    
    def test_save_results_unsupported_format(self):
        """Test saving results with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "results.txt"
            
            with pytest.raises(EvaluationError):
                self.results.save_results(save_path)
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.results)
        
        assert 'TestModel' in str_repr
        assert 'binary_classification' in str_repr
        assert '5' in str_repr  # num_samples
        assert '0.100s' in str_repr  # prediction_time


class TestEvaluationEngine:
    """Test EvaluationEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = EvaluationEngine()
        self.model = MockModel()
        
        # Test data
        np.random.seed(42)
        self.X_test = np.random.rand(10, 4)
        self.y_test = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        
        # Set model predictions
        self.model.set_predictions(
            predictions=np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
            probabilities=np.random.rand(10, 2)
        )
    
    def test_evaluate_basic(self):
        """Test basic model evaluation."""
        results = self.engine.evaluate(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        assert isinstance(results, EvaluationResults)
        assert results.model_name == 'MockModel'
        assert results.num_samples == 10
        assert results.prediction_time >= 0  # Allow zero time for fast operations
        assert results.evaluation_time > 0
        assert len(results.predictions) == 10
        assert len(results.ground_truth) == 10
    
    def test_evaluate_untrained_model(self):
        """Test evaluating untrained model."""
        untrained_model = MockModel()
        untrained_model.is_trained = False
        
        with pytest.raises(EvaluationError):
            self.engine.evaluate(
                model=untrained_model,
                X_test=self.X_test,
                y_test=self.y_test
            )
    
    def test_evaluate_with_task_type(self):
        """Test evaluation with specified task type."""
        results = self.engine.evaluate(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        
        assert results.task_type == TaskType.BINARY_CLASSIFICATION
    
    def test_evaluate_without_predictions(self):
        """Test evaluation without returning predictions."""
        results = self.engine.evaluate(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            return_predictions=False
        )
        
        assert len(results.predictions) == 0
        assert len(results.ground_truth) == 0
    
    def test_evaluate_multiple_models(self):
        """Test evaluating multiple models."""
        model1 = MockModel()
        model1.set_predictions(np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0]))
        
        model2 = MockModel()
        model2.set_predictions(np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0]))
        
        models = {'Model1': model1, 'Model2': model2}
        
        results = self.engine.evaluate_multiple_models(
            models=models,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        assert len(results) == 2
        assert 'Model1' in results
        assert 'Model2' in results
        assert results['Model1'].model_name == 'Model1'
        assert results['Model2'].model_name == 'Model2'
    
    def test_evaluate_multiple_models_with_failure(self):
        """Test evaluating multiple models with one failure."""
        model1 = MockModel()
        model1.set_predictions(np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0]))
        
        # Create a model that will fail
        failing_model = Mock()
        failing_model.is_trained = True
        failing_model.predict.side_effect = Exception("Prediction failed")
        
        models = {'Model1': model1, 'FailingModel': failing_model}
        
        results = self.engine.evaluate_multiple_models(
            models=models,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        # Should only have results for the successful model
        assert len(results) == 1
        assert 'Model1' in results
        assert 'FailingModel' not in results
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock results
        result1 = Mock()
        result1.get_primary_score.return_value = 0.85
        result1.metrics.primary_metric = 'accuracy'
        result1.metrics.get_metric.return_value = Mock(higher_is_better=True)
        
        result2 = Mock()
        result2.get_primary_score.return_value = 0.90
        result2.metrics.primary_metric = 'accuracy'
        result2.metrics.get_metric.return_value = Mock(higher_is_better=True)
        
        results = {'Model1': result1, 'Model2': result2}
        
        sorted_results = self.engine.compare_models(results)
        
        assert len(sorted_results) == 2
        # Model2 should be first (higher score)
        assert sorted_results[0][0] == 'Model2'
        assert sorted_results[1][0] == 'Model1'
    
    def test_compare_models_empty(self):
        """Test comparing empty results."""
        results = self.engine.compare_models({})
        assert results == []
    
    def test_detect_task_type_binary(self):
        """Test task type detection for binary classification."""
        y_binary = np.array([0, 1, 0, 1, 0])
        task_type = self.engine._detect_task_type(y_binary, self.model)
        assert task_type == TaskType.BINARY_CLASSIFICATION
    
    def test_detect_task_type_multiclass(self):
        """Test task type detection for multiclass classification."""
        y_multiclass = np.array([0, 1, 2, 0, 1, 2])
        task_type = self.engine._detect_task_type(y_multiclass, self.model)
        assert task_type == TaskType.MULTICLASS_CLASSIFICATION
    
    def test_detect_task_type_regression(self):
        """Test task type detection for regression."""
        y_regression = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
        task_type = self.engine._detect_task_type(y_regression, self.model)
        assert task_type == TaskType.REGRESSION
    
    def test_detect_data_type_tabular(self):
        """Test data type detection for tabular data."""
        X_tabular = np.random.rand(10, 5)
        data_type = self.engine._detect_data_type(X_tabular)
        assert data_type == DataType.TABULAR
    
    def test_detect_data_type_image(self):
        """Test data type detection for image data."""
        X_image = np.random.rand(10, 32, 32, 3)
        data_type = self.engine._detect_data_type(X_image)
        assert data_type == DataType.IMAGE
    
    @patch('neurolite.evaluation.evaluator.logger')
    def test_log_model_comparison(self, mock_logger):
        """Test logging model comparison."""
        result1 = Mock()
        result1.get_primary_score.return_value = 0.85
        result1.metrics.primary_metric = 'accuracy'
        result1.metrics.get_metric.return_value = Mock(higher_is_better=True)
        
        results = {'Model1': result1}
        
        self.engine._log_model_comparison(results)
        
        # Verify logging was called
        assert mock_logger.info.called
    
    def test_evaluation_with_exception(self):
        """Test evaluation with exception during prediction."""
        failing_model = Mock()
        failing_model.is_trained = True
        failing_model.predict.side_effect = Exception("Prediction failed")
        
        with pytest.raises(EvaluationError):
            self.engine.evaluate(
                model=failing_model,
                X_test=self.X_test,
                y_test=self.y_test
            )


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_evaluate_model_function(self):
        """Test evaluate_model convenience function."""
        model = MockModel()
        X_test = np.random.rand(5, 3)
        y_test = np.array([0, 1, 0, 1, 0])
        
        model.set_predictions(np.array([0, 1, 1, 1, 0]))
        
        results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test
        )
        
        assert isinstance(results, EvaluationResults)
        assert results.num_samples == 5


if __name__ == '__main__':
    pytest.main([__file__])