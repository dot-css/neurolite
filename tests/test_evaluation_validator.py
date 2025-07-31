"""
Unit tests for cross-validation utilities.

Tests the CrossValidator, CrossValidationResults, and related functionality
for comprehensive cross-validation with stratified sampling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

from neurolite.evaluation.validator import (
    CrossValidator, CrossValidationResults, cross_validate_model
)
from neurolite.evaluation.evaluator import EvaluationResults
from neurolite.evaluation.metrics import MetricsCollection, MetricResult
from neurolite.models.base import BaseModel, TaskType, PredictionResult, ModelCapabilities
from neurolite.data.detector import DataType
from neurolite.core.exceptions import EvaluationError


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, supported_tasks=None, **kwargs):
        super().__init__(**kwargs)
        self.is_trained = False
        self._fit_called = False
        self._predictions = None
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
        self._fit_called = True
        return self
    
    def predict(self, X, **kwargs):
        if self._predictions is not None:
            predictions = self._predictions[:len(X)]
        else:
            predictions = np.random.randint(0, 2, len(X))
        
        probabilities = np.random.rand(len(X), 2)
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self
    
    def set_predictions(self, predictions):
        """Set predictions for testing."""
        self._predictions = predictions


class TestCrossValidationResults:
    """Test CrossValidationResults functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock fold results
        self.fold_results = []
        for i in range(3):
            metrics = {
                'accuracy': MetricResult('Accuracy', 0.8 + i * 0.05, 'Test accuracy'),
                'precision': MetricResult('Precision', 0.75 + i * 0.05, 'Test precision')
            }
            
            metrics_collection = MetricsCollection(
                task_type=TaskType.BINARY_CLASSIFICATION,
                metrics=metrics,
                primary_metric='accuracy'
            )
            
            fold_result = EvaluationResults(
                metrics=metrics_collection,
                predictions=np.array([0, 1, 0, 1]),
                ground_truth=np.array([0, 1, 1, 1]),
                evaluation_time=0.1 + i * 0.01
            )
            
            self.fold_results.append(fold_result)
        
        self.cv_results = CrossValidationResults(
            fold_results=self.fold_results,
            mean_metrics={'accuracy': 0.825, 'precision': 0.775},
            std_metrics={'accuracy': 0.025, 'precision': 0.025},
            num_folds=3,
            task_type=TaskType.BINARY_CLASSIFICATION,
            data_type=DataType.TABULAR,
            model_name="TestModel",
            total_time=0.5
        )
    
    def test_post_init(self):
        """Test post-initialization calculations."""
        assert self.cv_results.mean_fold_time > 0
        assert self.cv_results.metadata is not None
        assert self.cv_results.confidence_intervals is not None
    
    def test_get_primary_score(self):
        """Test getting primary metric score."""
        score = self.cv_results.get_primary_score()
        assert score == 0.825
    
    def test_get_primary_std(self):
        """Test getting primary metric standard deviation."""
        std = self.cv_results.get_primary_std()
        assert std == 0.025
    
    def test_get_metric_summary(self):
        """Test getting metric summary statistics."""
        summary = self.cv_results.get_metric_summary('accuracy')
        
        assert 'mean' in summary
        assert 'std' in summary
        assert 'min' in summary
        assert 'max' in summary
        assert 'median' in summary
        assert summary['mean'] == 0.825
        assert summary['std'] == 0.025
    
    def test_get_metric_summary_missing(self):
        """Test getting summary for missing metric."""
        summary = self.cv_results.get_metric_summary('missing_metric')
        assert summary == {}
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result_dict = self.cv_results.to_dict()
        
        assert result_dict['num_folds'] == 3
        assert result_dict['model_name'] == 'TestModel'
        assert result_dict['task_type'] == 'binary_classification'
        assert result_dict['primary_score'] == 0.825
        assert result_dict['primary_std'] == 0.025
        assert 'fold_scores' in result_dict
        assert len(result_dict['fold_scores']) == 3
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.cv_results)
        
        assert 'TestModel' in str_repr
        assert 'binary_classification' in str_repr
        assert 'Folds: 3' in str_repr
        assert 'accuracy: 0.8250 ± 0.0250' in str_repr


class TestCrossValidator:
    """Test CrossValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CrossValidator()
        self.model = MockModel()
        
        # Test data
        np.random.seed(42)
        self.X = np.random.rand(20, 4)
        self.y = np.array([0, 1] * 10)  # Balanced binary classification
        
        # Set model predictions
        self.model.set_predictions(np.array([0, 1, 0, 1] * 5))
    
    @patch('neurolite.evaluation.evaluator.EvaluationEngine')
    def test_cross_validate_basic(self, mock_evaluator_class):
        """Test basic cross-validation."""
        # Mock the evaluator
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create mock evaluation result
        metrics = {
            'accuracy': MetricResult('Accuracy', 0.8, 'Test accuracy')
        }
        metrics_collection = MetricsCollection(
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=metrics,
            primary_metric='accuracy'
        )
        
        mock_result = EvaluationResults(
            metrics=metrics_collection,
            predictions=np.array([0, 1, 0, 1]),
            ground_truth=np.array([0, 1, 1, 1]),
            evaluation_time=0.1
        )
        
        mock_evaluator.evaluate.return_value = mock_result
        
        # Run cross-validation
        results = self.validator.cross_validate(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=3
        )
        
        assert isinstance(results, CrossValidationResults)
        assert results.num_folds == 3
        assert len(results.fold_results) == 3
        assert 'accuracy' in results.mean_metrics
    
    def test_cross_validate_invalid_sklearn(self):
        """Test cross-validation without sklearn."""
        with patch('neurolite.evaluation.validator.CrossValidator._create_cv_splitter') as mock_create:
            mock_create.side_effect = EvaluationError("scikit-learn is required")
            
            with pytest.raises(EvaluationError):
                self.validator.cross_validate(
                    model=self.model,
                    X=self.X,
                    y=self.y
                )
    
    @patch('sklearn.model_selection.StratifiedKFold')
    def test_create_cv_splitter_stratified(self, mock_stratified_kfold):
        """Test creating stratified CV splitter."""
        mock_splitter = Mock()
        mock_stratified_kfold.return_value = mock_splitter
        
        splitter = self.validator._create_cv_splitter(
            cv=5,
            y=self.y,
            task_type=TaskType.BINARY_CLASSIFICATION,
            stratify=True,
            shuffle=True,
            random_state=42
        )
        
        assert splitter == mock_splitter
        mock_stratified_kfold.assert_called_once_with(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
    
    @patch('sklearn.model_selection.KFold')
    def test_create_cv_splitter_regular(self, mock_kfold):
        """Test creating regular CV splitter."""
        mock_splitter = Mock()
        mock_kfold.return_value = mock_splitter
        
        splitter = self.validator._create_cv_splitter(
            cv=5,
            y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Regression data
            task_type=TaskType.REGRESSION,
            stratify=False,
            shuffle=True,
            random_state=42
        )
        
        assert splitter == mock_splitter
        mock_kfold.assert_called_once_with(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
    
    def test_create_cv_splitter_existing(self):
        """Test using existing CV splitter."""
        existing_splitter = Mock()
        existing_splitter.split = Mock()
        
        splitter = self.validator._create_cv_splitter(
            cv=existing_splitter,
            y=self.y,
            task_type=TaskType.BINARY_CLASSIFICATION,
            stratify=True,
            shuffle=True,
            random_state=42
        )
        
        assert splitter == existing_splitter
    
    def test_clone_model_sklearn(self):
        """Test cloning sklearn model."""
        # Create mock sklearn model
        mock_sklearn_model = Mock()
        mock_model = Mock()
        mock_model.sklearn_model = mock_sklearn_model
        mock_model.get_config.return_value = {'param': 'value'}
        mock_model.__class__ = Mock()
        
        with patch('sklearn.base.clone') as mock_clone:
            mock_clone.return_value = Mock()
            
            cloned = self.validator._clone_model(mock_model)
            
            mock_clone.assert_called_once_with(mock_sklearn_model)
    
    def test_clone_model_deepcopy(self):
        """Test cloning model with deepcopy."""
        with patch('neurolite.evaluation.validator.deepcopy') as mock_deepcopy:
            mock_deepcopy.return_value = Mock()
            
            cloned = self.validator._clone_model(self.model)
            
            mock_deepcopy.assert_called_once_with(self.model)
    
    def test_clone_model_fallback(self):
        """Test cloning model with fallback to new instance."""
        with patch('neurolite.evaluation.validator.deepcopy') as mock_deepcopy:
            mock_deepcopy.side_effect = Exception("Deepcopy failed")
            
            cloned = self.validator._clone_model(self.model)
            
            # Should create new instance
            assert isinstance(cloned, MockModel)
    
    def test_aggregate_fold_results(self):
        """Test aggregating fold results."""
        # Create mock fold results
        fold_results = []
        for i in range(3):
            metrics = {
                'accuracy': MetricResult('Accuracy', 0.8 + i * 0.1, 'Test accuracy'),
                'precision': MetricResult('Precision', 0.7 + i * 0.1, 'Test precision')
            }
            
            metrics_collection = MetricsCollection(
                task_type=TaskType.BINARY_CLASSIFICATION,
                metrics=metrics,
                primary_metric='accuracy'
            )
            
            fold_result = EvaluationResults(
                metrics=metrics_collection,
                predictions=np.array([0, 1]),
                ground_truth=np.array([0, 1]),
                evaluation_time=0.1
            )
            
            fold_results.append(fold_result)
        
        aggregated = self.validator._aggregate_fold_results(
            fold_results=fold_results,
            task_type=TaskType.BINARY_CLASSIFICATION,
            data_type=DataType.TABULAR,
            model_name="TestModel",
            total_time=0.5
        )
        
        assert isinstance(aggregated, CrossValidationResults)
        assert aggregated.num_folds == 3
        assert 'accuracy' in aggregated.mean_metrics
        assert 'precision' in aggregated.mean_metrics
        assert aggregated.mean_metrics['accuracy'] == 0.9  # (0.8 + 0.9 + 1.0) / 3
    
    def test_aggregate_fold_results_empty(self):
        """Test aggregating empty fold results."""
        with pytest.raises(EvaluationError):
            self.validator._aggregate_fold_results(
                fold_results=[],
                task_type=TaskType.BINARY_CLASSIFICATION,
                data_type=DataType.TABULAR,
                model_name="TestModel",
                total_time=0.5
            )
    
    def test_cross_validate_multiple_models(self):
        """Test cross-validating multiple models."""
        model1 = MockModel()
        model2 = MockModel()
        
        models = {'Model1': model1, 'Model2': model2}
        
        with patch.object(self.validator, 'cross_validate') as mock_cv:
            mock_result = Mock()
            mock_result.model_name = 'TestModel'
            mock_cv.return_value = mock_result
            
            results = self.validator.cross_validate_multiple_models(
                models=models,
                X=self.X,
                y=self.y
            )
            
            assert len(results) == 2
            assert 'Model1' in results
            assert 'Model2' in results
            assert mock_cv.call_count == 2
    
    def test_cross_validate_multiple_models_with_failure(self):
        """Test cross-validating multiple models with one failure."""
        model1 = MockModel()
        failing_model = Mock()
        
        models = {'Model1': model1, 'FailingModel': failing_model}
        
        with patch.object(self.validator, 'cross_validate') as mock_cv:
            def side_effect(model, **kwargs):
                if model == failing_model:
                    raise Exception("CV failed")
                mock_result = Mock()
                mock_result.model_name = 'Model1'
                return mock_result
            
            mock_cv.side_effect = side_effect
            
            results = self.validator.cross_validate_multiple_models(
                models=models,
                X=self.X,
                y=self.y
            )
            
            # Should only have results for successful model
            assert len(results) == 1
            assert 'Model1' in results
            assert 'FailingModel' not in results
    
    def test_detect_task_type_binary(self):
        """Test task type detection for binary classification."""
        y_binary = np.array([0, 1, 0, 1, 0])
        task_type = self.validator._detect_task_type(y_binary, self.model)
        assert task_type == TaskType.BINARY_CLASSIFICATION
    
    def test_detect_task_type_multiclass(self):
        """Test task type detection for multiclass classification."""
        y_multiclass = np.array([0, 1, 2, 0, 1, 2])
        task_type = self.validator._detect_task_type(y_multiclass, self.model)
        assert task_type == TaskType.MULTICLASS_CLASSIFICATION
    
    def test_detect_task_type_regression(self):
        """Test task type detection for regression."""
        y_regression = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
        task_type = self.validator._detect_task_type(y_regression, self.model)
        assert task_type == TaskType.REGRESSION
    
    def test_detect_data_type_tabular(self):
        """Test data type detection for tabular data."""
        X_tabular = np.random.rand(10, 5)
        data_type = self.validator._detect_data_type(X_tabular)
        assert data_type == DataType.TABULAR
    
    def test_detect_data_type_image(self):
        """Test data type detection for image data."""
        X_image = np.random.rand(10, 32, 32, 3)
        data_type = self.validator._detect_data_type(X_image)
        assert data_type == DataType.IMAGE
    
    def test_is_classification_task(self):
        """Test classification task detection."""
        assert self.validator._is_classification_task(TaskType.BINARY_CLASSIFICATION)
        assert self.validator._is_classification_task(TaskType.MULTICLASS_CLASSIFICATION)
        assert self.validator._is_classification_task(TaskType.TEXT_CLASSIFICATION)
        assert not self.validator._is_classification_task(TaskType.REGRESSION)
    
    @patch('neurolite.evaluation.validator.logger')
    def test_log_cv_comparison(self, mock_logger):
        """Test logging cross-validation comparison."""
        result1 = Mock()
        result1.get_primary_score.return_value = 0.85
        result1.get_primary_std.return_value = 0.05
        result1.fold_results = [Mock()]
        result1.fold_results[0].metrics.primary_metric = 'accuracy'
        
        results = {'Model1': result1}
        
        self.validator._log_cv_comparison(results)
        
        # Verify logging was called
        assert mock_logger.info.called


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_cross_validate_model_function(self):
        """Test cross_validate_model convenience function."""
        model = MockModel()
        X = np.random.rand(10, 3)
        y = np.array([0, 1] * 5)
        
        with patch('neurolite.evaluation.validator.CrossValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            
            mock_result = Mock()
            mock_validator.cross_validate.return_value = mock_result
            
            result = cross_validate_model(
                model=model,
                X=X,
                y=y,
                cv=3
            )
            
            assert result == mock_result
            mock_validator.cross_validate.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])