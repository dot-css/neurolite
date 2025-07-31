"""
Unit tests for hyperparameter optimization functionality.

Tests the Optuna integration for automated hyperparameter search
across different model types with resource-aware termination.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional

from neurolite.training.optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationBounds,
    ResourceConstraints,
    SearchStrategy,
    PruningStrategy,
    optimize_hyperparameters
)
from neurolite.training.config import TrainingConfiguration, OptimizerType, LossFunction
from neurolite.training.trainer import TrainingEngine, TrainedModel, TrainingHistory
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, ModelMetadata, PredictionResult
from neurolite.data.detector import DataType
from neurolite.core.exceptions import TrainingError, ConfigurationError


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_called = False
        self.predict_called = False
    
    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
    
    def fit(self, X: Any, y: Any, validation_data: Optional[Any] = None, **kwargs) -> 'MockModel':
        self.fit_called = True
        self.is_trained = True
        return self
    
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        self.predict_called = True
        # Return mock predictions
        n_samples = len(X) if hasattr(X, '__len__') else 10
        predictions = np.random.random(n_samples)
        probabilities = np.random.random((n_samples, 2))
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def save(self, path: str) -> None:
        pass
    
    def load(self, path: str) -> 'MockModel':
        return self


def mock_model_factory(**kwargs) -> MockModel:
    """Factory function for creating mock models."""
    return MockModel(**kwargs)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)
    
    # Split into train/validation
    split_idx = 80
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_val, y_val


@pytest.fixture
def optimization_config():
    """Create optimization configuration for testing."""
    return OptimizationConfig(
        search_strategy=SearchStrategy.RANDOM,  # Use random for faster testing
        pruning_strategy=PruningStrategy.NONE,  # Disable pruning for testing
        resource_constraints=ResourceConstraints(
            max_trials=5,  # Small number for fast testing
            max_time_minutes=1,
            early_stopping_rounds=3
        ),
        objective_metric="val_loss",
        objective_direction="minimize",
        use_cross_validation=False,  # Disable CV for faster testing
        verbose=False
    )


class TestOptimizationBounds:
    """Test optimization bounds configuration."""
    
    def test_default_bounds(self):
        """Test default optimization bounds."""
        bounds = OptimizationBounds()
        
        assert bounds.learning_rate == (1e-5, 1e-1)
        assert bounds.batch_size == [8, 16, 32, 64, 128]
        assert bounds.epochs == (10, 200)
        assert bounds.dropout_rate == (0.0, 0.5)
        assert bounds.l1_regularization == (0.0, 0.1)
        assert bounds.l2_regularization == (0.0, 0.1)
        assert bounds.weight_decay == (0.0, 0.01)
    
    def test_custom_bounds(self):
        """Test custom optimization bounds."""
        bounds = OptimizationBounds(
            learning_rate=(1e-4, 1e-2),
            batch_size=[16, 32],
            epochs=(5, 50)
        )
        
        assert bounds.learning_rate == (1e-4, 1e-2)
        assert bounds.batch_size == [16, 32]
        assert bounds.epochs == (5, 50)


class TestResourceConstraints:
    """Test resource constraints configuration."""
    
    def test_default_constraints(self):
        """Test default resource constraints."""
        constraints = ResourceConstraints()
        
        assert constraints.max_trials == 100
        assert constraints.max_time_minutes is None
        assert constraints.max_memory_gb is None
        assert constraints.early_stopping_rounds == 10
        assert constraints.min_trials_before_pruning == 5
    
    def test_custom_constraints(self):
        """Test custom resource constraints."""
        constraints = ResourceConstraints(
            max_trials=50,
            max_time_minutes=30,
            max_memory_gb=8.0,
            early_stopping_rounds=5
        )
        
        assert constraints.max_trials == 50
        assert constraints.max_time_minutes == 30
        assert constraints.max_memory_gb == 8.0
        assert constraints.early_stopping_rounds == 5


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_default_config(self):
        """Test default optimization configuration."""
        config = OptimizationConfig()
        
        assert config.search_strategy == SearchStrategy.TPE
        assert config.pruning_strategy == PruningStrategy.MEDIAN
        assert config.objective_metric == "val_loss"
        assert config.objective_direction == "minimize"
        assert config.cv_folds == 3
        assert config.use_cross_validation is True
        assert config.verbose is True
    
    def test_custom_config(self):
        """Test custom optimization configuration."""
        config = OptimizationConfig(
            search_strategy=SearchStrategy.RANDOM,
            pruning_strategy=PruningStrategy.HYPERBAND,
            objective_metric="val_accuracy",
            objective_direction="maximize",
            cv_folds=5,
            use_cross_validation=False
        )
        
        assert config.search_strategy == SearchStrategy.RANDOM
        assert config.pruning_strategy == PruningStrategy.HYPERBAND
        assert config.objective_metric == "val_accuracy"
        assert config.objective_direction == "maximize"
        assert config.cv_folds == 5
        assert config.use_cross_validation is False


@pytest.mark.skipif(
    not hasattr(pytest, 'importorskip') or 
    pytest.importorskip('optuna', reason="Optuna not available") is None,
    reason="Optuna not available"
)
class TestHyperparameterOptimizer:
    """Test hyperparameter optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizationConfig()
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.study is None
        assert optimizer._start_time is None
        assert optimizer._best_model is None
    
    def test_optimizer_initialization_without_optuna(self):
        """Test optimizer initialization when Optuna is not available."""
        with patch('neurolite.training.optimizer.OPTUNA_AVAILABLE', False):
            with pytest.raises(ImportError, match="Optuna is required"):
                HyperparameterOptimizer()
    
    @patch('neurolite.training.optimizer.optuna')
    def test_create_study_tpe(self, mock_optuna):
        """Test study creation with TPE sampler."""
        config = OptimizationConfig(search_strategy=SearchStrategy.TPE)
        optimizer = HyperparameterOptimizer(config)
        
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        
        study = optimizer._create_study()
        
        assert study == mock_study
        mock_optuna.create_study.assert_called_once()
        call_args = mock_optuna.create_study.call_args
        assert call_args[1]['direction'] == 'minimize'
    
    @patch('neurolite.training.optimizer.optuna')
    def test_create_study_random(self, mock_optuna):
        """Test study creation with random sampler."""
        config = OptimizationConfig(
            search_strategy=SearchStrategy.RANDOM,
            objective_direction="maximize"
        )
        optimizer = HyperparameterOptimizer(config)
        
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        
        study = optimizer._create_study()
        
        assert study == mock_study
        mock_optuna.create_study.assert_called_once()
        call_args = mock_optuna.create_study.call_args
        assert call_args[1]['direction'] == 'maximize'
    
    @patch('neurolite.training.optimizer.optuna')
    def test_sample_hyperparameters(self, mock_optuna):
        """Test hyperparameter sampling."""
        config = OptimizationConfig()
        optimizer = HyperparameterOptimizer(config)
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.1, 0.0, 0.0, 0.9, 0.999, 0.9, 0.1]
        mock_trial.suggest_categorical.side_effect = [32, 'adam', 'reduce_on_plateau']
        mock_trial.suggest_int.side_effect = [50, 10]
        
        params = optimizer._sample_hyperparameters(mock_trial)
        
        assert 'learning_rate' in params
        assert 'batch_size' in params
        assert 'epochs' in params
        assert 'optimizer_type' in params
        assert 'scheduler_type' in params
        
        # Verify trial methods were called
        assert mock_trial.suggest_float.call_count >= 4
        assert mock_trial.suggest_categorical.call_count >= 2
        assert mock_trial.suggest_int.call_count >= 1
    
    def test_create_training_config(self):
        """Test training configuration creation from parameters."""
        config = OptimizationConfig()
        optimizer = HyperparameterOptimizer(config)
        
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'optimizer_type': OptimizerType.ADAM,
            'dropout_rate': 0.1,
            'l1_regularization': 0.01,
            'l2_regularization': 0.01,
            'weight_decay': 0.001
        }
        
        training_config = optimizer._create_training_config(params)
        
        assert training_config.batch_size == 32
        assert training_config.epochs == 50
        assert training_config.optimizer.learning_rate == 0.001
        assert training_config.optimizer.type == OptimizerType.ADAM
        assert training_config.dropout_rate == 0.1
        assert training_config.l1_regularization == 0.01
        assert training_config.l2_regularization == 0.01
        assert training_config.optimizer.weight_decay == 0.001
    
    def test_get_objective_value(self):
        """Test objective value extraction from trained model."""
        config = OptimizationConfig(objective_metric="val_loss")
        optimizer = HyperparameterOptimizer(config)
        
        # Create mock trained model
        history = TrainingHistory()
        history.val_loss = [1.0, 0.8, 0.6, 0.5]
        
        trained_model = Mock()
        trained_model.history = history
        
        mock_trial = Mock()
        
        value = optimizer._get_objective_value(trained_model, mock_trial)
        assert value == 0.5  # Last validation loss
    
    def test_get_objective_value_with_metrics(self):
        """Test objective value extraction with custom metrics."""
        config = OptimizationConfig(objective_metric="val_accuracy")
        optimizer = HyperparameterOptimizer(config)
        
        # Create mock trained model
        history = TrainingHistory()
        history.metrics = {"val_accuracy": [0.7, 0.8, 0.85, 0.9]}
        
        trained_model = Mock()
        trained_model.history = history
        
        mock_trial = Mock()
        
        value = optimizer._get_objective_value(trained_model, mock_trial)
        assert value == 0.9  # Last validation accuracy
    
    def test_get_objective_value_fallback(self):
        """Test objective value fallback when metric not found."""
        config = OptimizationConfig(objective_metric="nonexistent_metric")
        optimizer = HyperparameterOptimizer(config)
        
        # Create mock trained model with no metrics
        history = TrainingHistory()
        history.val_loss = [0.5]
        
        trained_model = Mock()
        trained_model.history = history
        
        mock_trial = Mock()
        
        value = optimizer._get_objective_value(trained_model, mock_trial)
        assert value == 0.5  # Fallback to validation loss
    
    @patch('neurolite.training.optimizer.optuna')
    def test_optimize_basic(self, mock_optuna, sample_data, optimization_config):
        """Test basic optimization functionality."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'learning_rate': 0.001, 'batch_size': 32}
        mock_study.best_value = 0.5
        mock_study.best_trial = Mock()
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(optimization_config)
        
        # Mock the training process
        with patch.object(optimizer, '_create_objective_function') as mock_objective:
            mock_objective_func = Mock(return_value=0.5)
            mock_objective.return_value = mock_objective_func
            
            result = optimizer.optimize(
                model_factory=mock_model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task_type=TaskType.CLASSIFICATION,
                data_type=DataType.TABULAR
            )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params == {'learning_rate': 0.001, 'batch_size': 32}
        assert result.best_value == 0.5
        assert result.optimization_time >= 0  # Can be 0 for mocked tests
        
        # Verify study was created and optimized
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
    
    @patch('neurolite.training.optimizer.optuna')
    def test_optimize_with_cross_validation(self, mock_optuna, sample_data):
        """Test optimization with cross-validation."""
        X_train, y_train, _, _ = sample_data
        
        config = OptimizationConfig(
            use_cross_validation=True,
            cv_folds=3,
            resource_constraints=ResourceConstraints(max_trials=3),
            verbose=False
        )
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'learning_rate': 0.001}
        mock_study.best_value = 0.6
        mock_study.best_trial = Mock()
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(config)
        
        # Mock cross-validation
        with patch.object(optimizer, '_cross_validate', return_value=0.6) as mock_cv:
            with patch.object(optimizer, '_create_objective_function') as mock_objective:
                mock_objective_func = Mock(return_value=0.6)
                mock_objective.return_value = mock_objective_func
                
                result = optimizer.optimize(
                    model_factory=mock_model_factory,
                    X_train=X_train,
                    y_train=y_train,
                    task_type=TaskType.CLASSIFICATION,
                    data_type=DataType.TABULAR
                )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_value == 0.6
    
    def test_get_optimization_insights(self):
        """Test optimization insights generation."""
        config = OptimizationConfig()
        optimizer = HyperparameterOptimizer(config)
        optimizer._start_time = 1000.0
        
        # Mock study with trials
        mock_trial1 = Mock()
        mock_trial1.value = 0.8
        mock_trial2 = Mock()
        mock_trial2.value = 0.6
        
        mock_study = Mock()
        mock_study.best_params = {'learning_rate': 0.001}
        mock_study.best_value = 0.6
        mock_study.trials = [mock_trial1, mock_trial2]
        
        optimizer.study = mock_study
        
        with patch('time.time', return_value=1010.0):
            insights = optimizer.get_optimization_insights()
        
        assert insights['n_trials'] == 2
        assert insights['best_value'] == 0.6
        assert insights['best_params'] == {'learning_rate': 0.001}
        assert insights['optimization_time'] == 10.0
        assert 'optimization_history' in insights
        assert len(insights['optimization_history']) == 2


class TestOptimizeHyperparametersFunction:
    """Test the convenience function for hyperparameter optimization."""
    
    @patch('neurolite.training.optimizer.HyperparameterOptimizer')
    def test_optimize_hyperparameters_function(self, mock_optimizer_class, sample_data):
        """Test the optimize_hyperparameters convenience function."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock optimizer instance
        mock_optimizer = Mock()
        mock_result = OptimizationResult(
            best_params={'learning_rate': 0.001},
            best_value=0.5
        )
        mock_optimizer.optimize.return_value = mock_result
        mock_optimizer_class.return_value = mock_optimizer
        
        result = optimize_hyperparameters(
            model_factory=mock_model_factory,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert result == mock_result
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once()


class TestTrainingEngineIntegration:
    """Test integration with TrainingEngine."""
    
    @patch('neurolite.training.optimizer.HyperparameterOptimizer')
    def test_training_engine_optimize_hyperparameters(self, mock_optimizer_class, sample_data):
        """Test hyperparameter optimization through TrainingEngine."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_result = OptimizationResult(
            best_params={'learning_rate': 0.001},
            best_value=0.5
        )
        mock_optimizer.optimize.return_value = mock_result
        mock_optimizer_class.return_value = mock_optimizer
        
        # Create training engine
        engine = TrainingEngine()
        
        result = engine.optimize_hyperparameters(
            model_factory=mock_model_factory,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert result == mock_result
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once()
        
        # Verify the optimizer was called with correct parameters
        call_args = mock_optimizer.optimize.call_args
        assert call_args[1]['model_factory'] == mock_model_factory
        assert np.array_equal(call_args[1]['X_train'], X_train)
        assert np.array_equal(call_args[1]['y_train'], y_train)


class TestErrorHandling:
    """Test error handling in hyperparameter optimization."""
    
    def test_optimizer_without_optuna_import_error(self):
        """Test error when Optuna is not available."""
        with patch('neurolite.training.optimizer.OPTUNA_AVAILABLE', False):
            with pytest.raises(ImportError, match="Optuna is required"):
                HyperparameterOptimizer()
    
    @patch('neurolite.training.optimizer.optuna')
    def test_optimization_failure_handling(self, mock_optuna, sample_data, optimization_config):
        """Test handling of optimization failures."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock study that raises exception
        mock_study = Mock()
        mock_study.optimize.side_effect = Exception("Optimization failed")
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(optimization_config)
        
        with pytest.raises(TrainingError, match="Hyperparameter optimization failed"):
            optimizer.optimize(
                model_factory=mock_model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
    
    @patch('neurolite.training.optimizer.optuna')
    def test_trial_failure_handling(self, mock_optuna, sample_data, optimization_config):
        """Test handling of individual trial failures."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'learning_rate': 0.001}
        mock_study.best_value = float('inf')  # Failed trial value
        mock_study.best_trial = Mock()
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(optimization_config)
        
        # Mock objective function that raises exception
        def failing_objective(trial):
            raise Exception("Trial failed")
        
        with patch.object(optimizer, '_create_objective_function', return_value=failing_objective):
            result = optimizer.optimize(
                model_factory=mock_model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
        
        # Should still return a result even with failed trials
        assert isinstance(result, OptimizationResult)


class TestResourceConstraints:
    """Test resource constraint enforcement."""
    
    @patch('neurolite.training.optimizer.optuna')
    def test_max_trials_constraint(self, mock_optuna, sample_data):
        """Test maximum trials constraint."""
        X_train, y_train, X_val, y_val = sample_data
        
        config = OptimizationConfig(
            resource_constraints=ResourceConstraints(max_trials=3),
            verbose=False
        )
        
        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = 0.5
        mock_study.best_trial = Mock()
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(config)
        
        with patch.object(optimizer, '_create_objective_function') as mock_objective:
            mock_objective.return_value = Mock(return_value=0.5)
            
            optimizer.optimize(
                model_factory=mock_model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
        
        # Verify optimize was called with correct max_trials
        call_args = mock_study.optimize.call_args
        assert call_args[1]['n_trials'] == 3
    
    @patch('neurolite.training.optimizer.optuna')
    def test_timeout_constraint(self, mock_optuna, sample_data):
        """Test timeout constraint."""
        X_train, y_train, X_val, y_val = sample_data
        
        config = OptimizationConfig(
            resource_constraints=ResourceConstraints(
                max_trials=100,
                max_time_minutes=1  # 1 minute timeout
            ),
            verbose=False
        )
        
        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = 0.5
        mock_study.best_trial = Mock()
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(config)
        
        with patch.object(optimizer, '_create_objective_function') as mock_objective:
            mock_objective.return_value = Mock(return_value=0.5)
            
            optimizer.optimize(
                model_factory=mock_model_factory,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
        
        # Verify optimize was called with timeout
        call_args = mock_study.optimize.call_args
        assert call_args[1]['timeout'] == 60  # 1 minute in seconds


if __name__ == '__main__':
    pytest.main([__file__])