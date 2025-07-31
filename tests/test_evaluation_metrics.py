"""
Unit tests for evaluation metrics system.

Tests the MetricCalculator, MetricsCollection, and related functionality
for automatic metric selection and calculation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from neurolite.evaluation.metrics import (
    MetricCalculator, MetricsCollection, MetricResult, MetricType,
    get_metrics_for_task, calculate_classification_metrics, calculate_regression_metrics
)
from neurolite.models.base import TaskType
from neurolite.core.exceptions import MetricError


class TestMetricResult:
    """Test MetricResult dataclass."""
    
    def test_metric_result_creation(self):
        """Test creating a MetricResult."""
        result = MetricResult(
            name="Accuracy",
            value=0.85,
            description="Test accuracy",
            higher_is_better=True
        )
        
        assert result.name == "Accuracy"
        assert result.value == 0.85
        assert result.description == "Test accuracy"
        assert result.higher_is_better is True
    
    def test_metric_result_defaults(self):
        """Test MetricResult with default values."""
        result = MetricResult(
            name="Test Metric",
            value=0.5,
            description="Test description"
        )
        
        assert result.higher_is_better is True  # Default value


class TestMetricsCollection:
    """Test MetricsCollection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = {
            'accuracy': MetricResult('Accuracy', 0.85, 'Test accuracy'),
            'precision': MetricResult('Precision', 0.80, 'Test precision'),
            'recall': MetricResult('Recall', 0.90, 'Test recall')
        }
        
        self.collection = MetricsCollection(
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=self.metrics,
            primary_metric='accuracy'
        )
    
    def test_get_metric(self):
        """Test getting a specific metric."""
        accuracy = self.collection.get_metric('accuracy')
        assert accuracy is not None
        assert accuracy.value == 0.85
        
        # Test non-existent metric
        missing = self.collection.get_metric('missing')
        assert missing is None
    
    def test_get_primary_score(self):
        """Test getting primary metric score."""
        score = self.collection.get_primary_score()
        assert score == 0.85
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result_dict = self.collection.to_dict()
        
        expected = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90
        }
        
        assert result_dict == expected
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.collection)
        
        assert 'Task Type: binary_classification' in str_repr
        assert 'Primary Metric: accuracy' in str_repr
        assert 'accuracy: 0.8500' in str_repr
        assert 'precision: 0.8000' in str_repr
        assert 'recall: 0.9000' in str_repr


class TestMetricCalculator:
    """Test MetricCalculator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricCalculator()
        
        # Create sample data
        np.random.seed(42)
        self.y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.y_pred_binary = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        self.y_proba_binary = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8],
            [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2]
        ])
        
        self.y_true_multiclass = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred_multiclass = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
        self.y_proba_multiclass = np.random.rand(10, 3)
        
        self.y_true_regression = np.array([1.5, 2.3, 3.1, 4.2, 5.0, 2.8, 3.9, 1.2, 4.5, 3.7])
        self.y_pred_regression = np.array([1.4, 2.1, 3.3, 4.0, 5.2, 2.9, 3.8, 1.3, 4.3, 3.5])
    
    def test_get_metrics_for_task_binary(self):
        """Test getting metrics for binary classification."""
        config = self.calculator.get_metrics_for_task(TaskType.BINARY_CLASSIFICATION)
        
        assert 'primary' in config
        assert 'metrics' in config
        assert config['primary'] == MetricType.F1_SCORE
        assert MetricType.ACCURACY in config['metrics']
        assert MetricType.ROC_AUC in config['metrics']
    
    def test_get_metrics_for_task_multiclass(self):
        """Test getting metrics for multiclass classification."""
        config = self.calculator.get_metrics_for_task(TaskType.MULTICLASS_CLASSIFICATION)
        
        assert config['primary'] == MetricType.F1_WEIGHTED
        assert MetricType.ACCURACY in config['metrics']
        assert MetricType.F1_MACRO in config['metrics']
    
    def test_get_metrics_for_task_regression(self):
        """Test getting metrics for regression."""
        config = self.calculator.get_metrics_for_task(TaskType.REGRESSION)
        
        assert config['primary'] == MetricType.RMSE
        assert MetricType.MAE in config['metrics']
        assert MetricType.R2_SCORE in config['metrics']
    
    def test_get_metrics_for_task_unknown(self):
        """Test getting metrics for unknown task type."""
        config = self.calculator.get_metrics_for_task(TaskType.AUTO)
        
        # Should fallback to classification
        assert MetricType.ACCURACY in config['metrics']
    
    @patch('neurolite.evaluation.metrics.MetricCalculator._check_sklearn')
    def test_calculate_metrics_no_sklearn(self, mock_check):
        """Test metric calculation without sklearn."""
        mock_check.return_value = False
        calculator = MetricCalculator()
        
        with pytest.raises(MetricError):
            calculator.calculate_metrics(
                self.y_true_binary,
                self.y_pred_binary,
                TaskType.BINARY_CLASSIFICATION
            )
    
    def test_calculate_binary_classification_metrics(self):
        """Test calculating binary classification metrics."""
        metrics = self.calculator.calculate_metrics(
            y_true=self.y_true_binary,
            y_pred=self.y_pred_binary,
            task_type=TaskType.BINARY_CLASSIFICATION,
            y_proba=self.y_proba_binary
        )
        
        assert isinstance(metrics, MetricsCollection)
        assert metrics.task_type == TaskType.BINARY_CLASSIFICATION
        assert 'accuracy' in metrics.metrics
        assert 'precision' in metrics.metrics
        assert 'recall' in metrics.metrics
        assert 'f1' in metrics.metrics
        assert 'roc_auc' in metrics.metrics
        
        # Check that values are reasonable
        accuracy = metrics.get_metric('accuracy')
        assert 0 <= accuracy.value <= 1
    
    def test_calculate_multiclass_classification_metrics(self):
        """Test calculating multiclass classification metrics."""
        metrics = self.calculator.calculate_metrics(
            y_true=self.y_true_multiclass,
            y_pred=self.y_pred_multiclass,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            y_proba=self.y_proba_multiclass
        )
        
        assert isinstance(metrics, MetricsCollection)
        assert metrics.task_type == TaskType.MULTICLASS_CLASSIFICATION
        assert 'accuracy' in metrics.metrics
        assert 'f1_weighted' in metrics.metrics
        assert 'f1_macro' in metrics.metrics
    
    def test_calculate_regression_metrics(self):
        """Test calculating regression metrics."""
        metrics = self.calculator.calculate_metrics(
            y_true=self.y_true_regression,
            y_pred=self.y_pred_regression,
            task_type=TaskType.REGRESSION
        )
        
        assert isinstance(metrics, MetricsCollection)
        assert metrics.task_type == TaskType.REGRESSION
        assert 'mae' in metrics.metrics
        assert 'mse' in metrics.metrics
        assert 'rmse' in metrics.metrics
        assert 'r2' in metrics.metrics
        
        # Check that RMSE is positive
        rmse = metrics.get_metric('rmse')
        assert rmse.value >= 0
    
    def test_calculate_metrics_shape_mismatch(self):
        """Test metric calculation with mismatched shapes."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 0])  # Different length
        
        with pytest.raises(MetricError):
            self.calculator.calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                task_type=TaskType.BINARY_CLASSIFICATION
            )
    
    def test_calculate_metrics_custom_metrics(self):
        """Test calculating specific metrics only."""
        custom_metrics = [MetricType.ACCURACY, MetricType.PRECISION]
        
        metrics = self.calculator.calculate_metrics(
            y_true=self.y_true_binary,
            y_pred=self.y_pred_binary,
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=custom_metrics
        )
        
        assert 'accuracy' in metrics.metrics
        assert 'precision' in metrics.metrics
        assert 'recall' not in metrics.metrics  # Not requested
    
    def test_is_classification_task(self):
        """Test classification task detection."""
        assert self.calculator._is_classification_task(TaskType.BINARY_CLASSIFICATION)
        assert self.calculator._is_classification_task(TaskType.MULTICLASS_CLASSIFICATION)
        assert self.calculator._is_classification_task(TaskType.TEXT_CLASSIFICATION)
        assert not self.calculator._is_classification_task(TaskType.REGRESSION)
    
    def test_calculate_classification_metrics_no_probabilities(self):
        """Test classification metrics without probabilities."""
        metrics = self.calculator.calculate_metrics(
            y_true=self.y_true_binary,
            y_pred=self.y_pred_binary,
            task_type=TaskType.BINARY_CLASSIFICATION
            # No y_proba provided
        )
        
        # Should still calculate basic metrics
        assert 'accuracy' in metrics.metrics
        assert 'precision' in metrics.metrics
        # ROC AUC should not be calculated without probabilities
        assert 'roc_auc' not in metrics.metrics
    
    def test_calculate_regression_mape_with_zeros(self):
        """Test MAPE calculation with zero values."""
        y_true = np.array([0.0, 1.0, 2.0, 0.0])
        y_pred = np.array([0.1, 1.1, 1.9, 0.2])
        
        metrics = self.calculator.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            task_type=TaskType.REGRESSION,
            metrics=[MetricType.MEAN_ABSOLUTE_PERCENTAGE_ERROR]
        )
        
        # Should handle zero values gracefully
        mape = metrics.get_metric('mape')
        assert mape is not None
        assert mape.value >= 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1])
        self.y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])
    
    def test_get_metrics_for_task_function(self):
        """Test get_metrics_for_task convenience function."""
        config = get_metrics_for_task(TaskType.BINARY_CLASSIFICATION)
        
        assert 'primary' in config
        assert 'metrics' in config
        assert isinstance(config['metrics'], list)
    
    def test_calculate_classification_metrics_function(self):
        """Test calculate_classification_metrics convenience function."""
        metrics = calculate_classification_metrics(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_proba=self.y_proba
        )
        
        assert isinstance(metrics, MetricsCollection)
        assert 'accuracy' in metrics.metrics
    
    def test_calculate_regression_metrics_function(self):
        """Test calculate_regression_metrics convenience function."""
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = calculate_regression_metrics(
            y_true=y_true_reg,
            y_pred=y_pred_reg
        )
        
        assert isinstance(metrics, MetricsCollection)
        assert 'mae' in metrics.metrics
        assert 'rmse' in metrics.metrics


if __name__ == '__main__':
    pytest.main([__file__])