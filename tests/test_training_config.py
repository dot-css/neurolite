"""
Unit tests for training configuration management.
"""

import pytest
import numpy as np
from neurolite.training.config import (
    TrainingConfiguration, ConfigurationOptimizer, 
    OptimizerType, LossFunction, SchedulerType,
    get_default_training_config, optimize_training_config
)
from neurolite.models.base import TaskType
from neurolite.data.detector import DataType


class TestTrainingConfiguration:
    """Test training configuration class."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = TrainingConfiguration()
        
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.validation_split == 0.2
        assert config.test_split == 0.1
        assert config.random_seed == 42
        assert config.loss_function == LossFunction.AUTO
        assert config.optimizer.type == OptimizerType.ADAM
        assert config.optimizer.learning_rate == 0.001
        assert config.early_stopping.enabled is True
        assert config.checkpoint.enabled is True
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfiguration(batch_size=0)
        
        # Test invalid epochs
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            TrainingConfiguration(epochs=-1)
        
        # Test invalid validation split
        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            TrainingConfiguration(validation_split=1.5)
        
        # Test invalid combined splits
        with pytest.raises(ValueError, match="Validation and test splits combined must be less than 1"):
            TrainingConfiguration(validation_split=0.6, test_split=0.5)
    
    def test_configuration_update(self):
        """Test configuration update."""
        config = TrainingConfiguration()
        
        config.update(batch_size=64, epochs=50)
        assert config.batch_size == 64
        assert config.epochs == 50
        
        # Test extra parameters
        config.update(custom_param="test")
        assert config.extra_params["custom_param"] == "test"
    
    def test_configuration_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = TrainingConfiguration(batch_size=64, epochs=50)
        config_dict = config.to_dict()
        
        assert config_dict["batch_size"] == 64
        assert config_dict["epochs"] == 50
        assert config_dict["loss_function"] == "auto"
        assert isinstance(config_dict["optimizer"], dict)
        assert config_dict["optimizer"]["type"] == "adam"
    
    def test_default_metrics_assignment(self):
        """Test automatic default metrics assignment."""
        # Test classification metrics
        config = TrainingConfiguration(loss_function=LossFunction.CROSS_ENTROPY)
        assert "accuracy" in config.metrics
        
        # Test regression metrics
        config = TrainingConfiguration(loss_function=LossFunction.MSE)
        assert "mae" in config.metrics
        assert "mse" in config.metrics


class TestConfigurationOptimizer:
    """Test configuration optimizer."""
    
    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        # Small dataset
        batch_size = ConfigurationOptimizer._optimize_batch_size(500, DataType.IMAGE)
        assert batch_size == 16
        
        # Medium dataset
        batch_size = ConfigurationOptimizer._optimize_batch_size(5000, DataType.IMAGE)
        assert batch_size == 32
        
        # Large dataset
        batch_size = ConfigurationOptimizer._optimize_batch_size(50000, DataType.IMAGE)
        assert batch_size == 64
        
        # Text data (smaller batches)
        batch_size = ConfigurationOptimizer._optimize_batch_size(5000, DataType.TEXT)
        assert batch_size == 16
        
        # Tabular data (larger batches)
        batch_size = ConfigurationOptimizer._optimize_batch_size(5000, DataType.TABULAR)
        assert batch_size == 64
    
    def test_optimize_epochs(self):
        """Test epochs optimization."""
        # Small dataset - more epochs
        epochs = ConfigurationOptimizer._optimize_epochs(500, DataType.TABULAR, TaskType.CLASSIFICATION)
        assert epochs == 200
        
        # Large dataset - fewer epochs
        epochs = ConfigurationOptimizer._optimize_epochs(200000, DataType.TABULAR, TaskType.CLASSIFICATION)
        assert epochs == 50
        
        # Image data - ensure minimum epochs
        epochs = ConfigurationOptimizer._optimize_epochs(5000, DataType.IMAGE, TaskType.IMAGE_CLASSIFICATION)
        assert epochs >= 50
    
    def test_optimize_learning_rate(self):
        """Test learning rate optimization."""
        # Base learning rate for tabular data
        lr = ConfigurationOptimizer._optimize_learning_rate(TaskType.CLASSIFICATION, DataType.TABULAR, 5000)
        assert lr == 0.001
        
        # Lower learning rate for images
        lr = ConfigurationOptimizer._optimize_learning_rate(TaskType.IMAGE_CLASSIFICATION, DataType.IMAGE, 5000)
        assert lr == 0.0001
        
        # Even lower for text
        lr = ConfigurationOptimizer._optimize_learning_rate(TaskType.TEXT_CLASSIFICATION, DataType.TEXT, 5000)
        assert lr == 0.00005
        
        # Adjust for dataset size
        lr_small = ConfigurationOptimizer._optimize_learning_rate(TaskType.CLASSIFICATION, DataType.TABULAR, 500)
        lr_large = ConfigurationOptimizer._optimize_learning_rate(TaskType.CLASSIFICATION, DataType.TABULAR, 200000)
        assert lr_small < 0.001  # Lower for small datasets
        assert lr_large > 0.001  # Higher for large datasets
    
    def test_select_loss_function(self):
        """Test loss function selection."""
        # Binary classification
        loss = ConfigurationOptimizer._select_loss_function(TaskType.BINARY_CLASSIFICATION, 2)
        assert loss == LossFunction.BINARY_CROSS_ENTROPY
        
        # Multi-class classification
        loss = ConfigurationOptimizer._select_loss_function(TaskType.CLASSIFICATION, 5)
        assert loss == LossFunction.CROSS_ENTROPY
        
        # Regression
        loss = ConfigurationOptimizer._select_loss_function(TaskType.REGRESSION, None)
        assert loss == LossFunction.MSE
        
        # Auto for unknown tasks
        loss = ConfigurationOptimizer._select_loss_function(TaskType.AUTO, None)
        assert loss == LossFunction.AUTO
    
    def test_select_metrics(self):
        """Test metrics selection."""
        # Classification metrics
        metrics = ConfigurationOptimizer._select_metrics(TaskType.CLASSIFICATION)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Regression metrics
        metrics = ConfigurationOptimizer._select_metrics(TaskType.REGRESSION)
        assert "mae" in metrics
        assert "mse" in metrics
        assert "r2" in metrics
    
    def test_optimize_patience(self):
        """Test early stopping patience optimization."""
        # Small dataset - limited patience
        patience = ConfigurationOptimizer._optimize_patience(500, 100)
        assert patience <= 20
        
        # Large dataset - more patience
        patience = ConfigurationOptimizer._optimize_patience(50000, 100)
        assert patience >= 10
    
    def test_full_optimization(self):
        """Test complete configuration optimization."""
        config = ConfigurationOptimizer.optimize_for_task_and_data(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            data_type=DataType.IMAGE,
            num_samples=10000,
            num_classes=10
        )
        
        assert config.batch_size == 64  # Medium dataset, image data (corrected expected value)
        assert config.optimizer.learning_rate == 0.0001  # Lower for images
        assert config.loss_function == LossFunction.CROSS_ENTROPY
        assert "accuracy" in config.metrics
        assert config.early_stopping.patience >= 10


class TestConfigurationFunctions:
    """Test configuration utility functions."""
    
    def test_get_default_training_config(self):
        """Test default configuration getter."""
        config = get_default_training_config()
        assert isinstance(config, TrainingConfiguration)
        assert config.batch_size == 32
        assert config.epochs == 100
    
    def test_optimize_training_config(self):
        """Test configuration optimization function."""
        config = optimize_training_config(
            task_type=TaskType.TEXT_CLASSIFICATION,
            data_type=DataType.TEXT,
            num_samples=5000,
            num_classes=3,
            batch_size=16  # Override
        )
        
        assert isinstance(config, TrainingConfiguration)
        assert config.batch_size == 16  # Should use override
        assert config.optimizer.learning_rate == 0.00005  # Optimized for text
        assert config.loss_function == LossFunction.CROSS_ENTROPY
    
    def test_optimize_training_config_with_overrides(self):
        """Test configuration optimization with parameter overrides."""
        config = optimize_training_config(
            task_type=TaskType.REGRESSION,
            data_type=DataType.TABULAR,
            num_samples=1000,
            epochs=200,  # Override
            learning_rate=0.01  # Override
        )
        
        assert config.epochs == 200  # Should use override
        # Note: learning_rate override would be in extra_params since it's nested
        assert config.extra_params.get("learning_rate") == 0.01