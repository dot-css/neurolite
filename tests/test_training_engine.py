"""
Unit tests for training engine.
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from neurolite.training.trainer import TrainingEngine, TrainedModel, TrainingHistory
from neurolite.training.config import TrainingConfiguration
from neurolite.training.callbacks import EarlyStopping, ModelCheckpoint
from neurolite.models.base import BaseModel, SklearnModelAdapter, TaskType, ModelCapabilities
from neurolite.data.detector import DataType
from neurolite.core.exceptions import TrainingFailedError


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_called = False
        self.predict_called = False
        self._predictions = None
    
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="mock"
        )
    
    def fit(self, X, y, validation_data=None, **kwargs):
        self.fit_called = True
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        self.predict_called = True
        if self._predictions is not None:
            return self._predictions
        
        # Return mock predictions
        from neurolite.models.base import PredictionResult
        n_samples = len(X) if hasattr(X, '__len__') else 10
        predictions = np.random.rand(n_samples)
        return PredictionResult(predictions=predictions)
    
    def save(self, path):
        # Mock save
        with open(path, 'w') as f:
            f.write("mock_model")
    
    def load(self, path):
        # Mock load
        self.is_trained = True
        return self


class TestTrainingHistory:
    """Test training history class."""
    
    def test_training_history_initialization(self):
        """Test training history initialization."""
        history = TrainingHistory()
        
        assert history.train_loss == []
        assert history.val_loss == []
        assert history.metrics == {}
        assert history.epochs == []
        assert history.learning_rates == []
    
    def test_add_epoch(self):
        """Test adding epoch to history."""
        history = TrainingHistory()
        
        history.add_epoch(
            epoch=0,
            train_loss=0.5,
            val_loss=0.6,
            metrics={"accuracy": 0.8, "f1": 0.75},
            learning_rate=0.001
        )
        
        assert history.epochs == [0]
        assert history.train_loss == [0.5]
        assert history.val_loss == [0.6]
        assert history.learning_rates == [0.001]
        assert history.metrics["accuracy"] == [0.8]
        assert history.metrics["f1"] == [0.75]
        
        # Add another epoch
        history.add_epoch(
            epoch=1,
            train_loss=0.4,
            val_loss=0.5,
            metrics={"accuracy": 0.85, "f1": 0.8},
            learning_rate=0.0008
        )
        
        assert len(history.epochs) == 2
        assert history.train_loss == [0.5, 0.4]
        assert history.metrics["accuracy"] == [0.8, 0.85]
    
    def test_get_best_epoch(self):
        """Test getting best epoch."""
        history = TrainingHistory()
        
        # Add some epochs
        history.add_epoch(0, 0.5, 0.6, {"accuracy": 0.8})
        history.add_epoch(1, 0.4, 0.5, {"accuracy": 0.85})
        history.add_epoch(2, 0.3, 0.55, {"accuracy": 0.82})
        
        # Best validation loss (min mode)
        best_epoch = history.get_best_epoch("val_loss", "min")
        assert best_epoch == 1  # Epoch 1 has lowest val_loss (0.5)
        
        # Best accuracy (max mode)
        best_epoch = history.get_best_epoch("accuracy", "max")
        assert best_epoch == 1  # Epoch 1 has highest accuracy (0.85)
        
        # Test with missing metric
        with pytest.raises(ValueError, match="Metric 'missing' not found"):
            history.get_best_epoch("missing")


class TestTrainedModel:
    """Test trained model class."""
    
    def test_trained_model_initialization(self):
        """Test trained model initialization."""
        mock_model = MockModel()
        config = TrainingConfiguration()
        history = TrainingHistory()
        
        from neurolite.models.base import ModelMetadata
        metadata = ModelMetadata(
            name="MockModel",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR,
            framework="mock",
            training_time=10.0
        )
        
        trained_model = TrainedModel(
            model=mock_model,
            config=config,
            history=history,
            metadata=metadata,
            training_time=10.0,
            best_epoch=5
        )
        
        assert trained_model.model == mock_model
        assert trained_model.config == config
        assert trained_model.history == history
        assert trained_model.metadata == metadata
        assert trained_model.training_time == 10.0
        assert trained_model.best_epoch == 5
    
    def test_trained_model_predict(self):
        """Test trained model prediction."""
        mock_model = MockModel()
        trained_model = TrainedModel(
            model=mock_model,
            config=TrainingConfiguration(),
            history=TrainingHistory(),
            metadata=Mock(),
            training_time=0.0,
            best_epoch=0
        )
        
        X_test = np.random.rand(10, 5)
        result = trained_model.predict(X_test)
        
        assert mock_model.predict_called
        assert result is not None
    
    def test_trained_model_save_load(self):
        """Test trained model save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            
            mock_model = MockModel()
            config = TrainingConfiguration(batch_size=64)
            history = TrainingHistory()
            history.add_epoch(0, 0.5, 0.6, {"accuracy": 0.8})
            
            trained_model = TrainedModel(
                model=mock_model,
                config=config,
                history=history,
                metadata=Mock(),
                training_time=10.0,
                best_epoch=0
            )
            
            # Save
            trained_model.save(model_path)
            assert os.path.exists(model_path)
            assert os.path.exists(model_path.replace('.pt', '.json'))
            
            # Load
            new_trained_model = TrainedModel(
                model=MockModel(),
                config=TrainingConfiguration(),
                history=TrainingHistory(),
                metadata=Mock(),
                training_time=0.0,
                best_epoch=0
            )
            
            new_trained_model.load(model_path)
            assert new_trained_model.training_time == 10.0
            assert new_trained_model.best_epoch == 0
            assert len(new_trained_model.history.train_loss) == 1


class TestTrainingEngine:
    """Test training engine."""
    
    def test_training_engine_initialization(self):
        """Test training engine initialization."""
        engine = TrainingEngine()
        
        assert engine.config is not None
        assert isinstance(engine.config, TrainingConfiguration)
        assert engine.callback_manager is not None
        assert len(engine.callback_manager) >= 1  # Should have default callbacks
    
    def test_training_engine_custom_config(self):
        """Test training engine with custom configuration."""
        config = TrainingConfiguration(batch_size=64, epochs=50)
        engine = TrainingEngine(config)
        
        assert engine.config.batch_size == 64
        assert engine.config.epochs == 50
    
    def test_add_callback(self):
        """Test adding custom callback."""
        engine = TrainingEngine()
        initial_callback_count = len(engine.callback_manager)
        
        custom_callback = EarlyStopping(patience=5)
        engine.add_callback(custom_callback)
        
        assert len(engine.callback_manager) == initial_callback_count + 1
    
    def test_train_sklearn_model(self):
        """Test training sklearn model."""
        # Create sample data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.randint(0, 2, 20)
        
        # Create sklearn model adapter
        sklearn_model = RandomForestClassifier(n_estimators=10, random_state=42)
        model = SklearnModelAdapter(sklearn_model)
        
        # Create training engine
        config = TrainingConfiguration(epochs=1, verbose=False)
        engine = TrainingEngine(config)
        
        # Train model
        trained_model = engine.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert isinstance(trained_model, TrainedModel)
        assert trained_model.model.is_trained
        assert trained_model.training_time > 0
        assert len(trained_model.history.train_loss) > 0
    
    def test_train_mock_model(self):
        """Test training mock model."""
        # Create sample data
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 3)
        y_val = np.random.rand(10)
        
        # Create mock model
        model = MockModel()
        
        # Create training engine with minimal epochs for testing
        config = TrainingConfiguration(epochs=2, verbose=False)
        engine = TrainingEngine(config)
        
        # Train model
        trained_model = engine.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert isinstance(trained_model, TrainedModel)
        assert model.fit_called
        assert model.is_trained
        assert trained_model.training_time >= 0  # Allow 0 for very fast training
        assert len(trained_model.history.epochs) == 2
    
    def test_train_without_validation(self):
        """Test training without validation data."""
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        
        model = MockModel()
        config = TrainingConfiguration(epochs=1, verbose=False)
        engine = TrainingEngine(config)
        
        trained_model = engine.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert isinstance(trained_model, TrainedModel)
        assert model.is_trained
        # Validation loss should be None in history
        assert all(val_loss == 0.0 for val_loss in trained_model.history.val_loss)
    
    def test_train_with_early_stopping(self):
        """Test training with early stopping."""
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 3)
        y_val = np.random.rand(10)
        
        # Create model that returns consistent predictions for early stopping test
        model = MockModel()
        from neurolite.models.base import PredictionResult
        model._predictions = PredictionResult(predictions=np.ones(10) * 0.5)  # Consistent predictions
        
        # Configure early stopping with very low patience
        config = TrainingConfiguration(epochs=10, verbose=False)
        config.early_stopping.patience = 1
        config.early_stopping.min_delta = 0.0
        
        engine = TrainingEngine(config)
        
        trained_model = engine.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        # Should stop early due to no improvement
        assert len(trained_model.history.epochs) < 10
    
    def test_train_with_checkpointing(self):
        """Test training with checkpointing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            X_train = np.random.rand(50, 3)
            y_train = np.random.rand(50)
            
            model = MockModel()
            config = TrainingConfiguration(epochs=2, verbose=False)
            config.checkpoint.enabled = True
            
            engine = TrainingEngine(config)
            
            trained_model = engine.train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                task_type=TaskType.CLASSIFICATION,
                data_type=DataType.TABULAR
            )
            
            assert isinstance(trained_model, TrainedModel)
            # Checkpoints should be created (though we can't easily verify the exact files)
    
    def test_train_with_configuration_optimization(self):
        """Test training with automatic configuration optimization."""
        X_train = np.random.rand(1000, 10)  # Larger dataset
        y_train = np.random.randint(0, 3, 1000)
        
        model = MockModel()
        config = TrainingConfiguration(verbose=False)
        engine = TrainingEngine(config)
        
        trained_model = engine.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        # Configuration should be optimized based on data characteristics
        assert isinstance(trained_model, TrainedModel)
        # The exact optimizations depend on the ConfigurationOptimizer logic
    
    def test_train_failure_handling(self):
        """Test training failure handling."""
        # Create a model that will fail during training
        class FailingModel(MockModel):
            def fit(self, X, y, validation_data=None, **kwargs):
                raise RuntimeError("Training failed")
        
        X_train = np.random.rand(50, 3)
        y_train = np.random.rand(50)
        
        model = FailingModel()
        config = TrainingConfiguration(epochs=1, verbose=False)
        engine = TrainingEngine(config)
        
        with pytest.raises(TrainingFailedError):
            engine.train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                task_type=TaskType.CLASSIFICATION,
                data_type=DataType.TABULAR
            )
    
    def test_resume_training(self):
        """Test resuming training from checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            
            # Create initial model and save it
            model = MockModel()
            model.is_trained = True
            model.save(checkpoint_path)
            
            # Create training state file
            state_path = checkpoint_path.replace('.pt', '_state.json')
            state_data = {
                'epoch': 5,
                'batch': 0,
                'train_loss': 0.3,
                'val_loss': 0.4,
                'metrics': {'accuracy': 0.8},
                'learning_rate': 0.001,
                'best_metric': 0.4,
                'best_epoch': 3
            }
            
            import json
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
            
            # Resume training
            X_train = np.random.rand(50, 3)
            y_train = np.random.rand(50)
            
            new_model = MockModel()
            config = TrainingConfiguration(epochs=10, verbose=False)  # Total epochs
            engine = TrainingEngine(config)
            
            trained_model = engine.resume_training(
                checkpoint_path=checkpoint_path,
                model=new_model,
                X_train=X_train,
                y_train=y_train,
                task_type=TaskType.CLASSIFICATION,
                data_type=DataType.TABULAR
            )
            
            assert isinstance(trained_model, TrainedModel)
            assert new_model.is_trained
    
    def test_resume_training_completed_model(self):
        """Test resuming training from a completed model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            
            # Create model that has completed training
            model = MockModel()
            model.is_trained = True
            model.save(checkpoint_path)
            
            # Create state showing training is complete
            state_path = checkpoint_path.replace('.pt', '_state.json')
            state_data = {
                'epoch': 100,  # Completed all epochs (100 epochs means 0-99, so epoch 100 means done)
                'batch': 0,
                'train_loss': 0.1,
                'val_loss': 0.15,
                'metrics': {'accuracy': 0.95},
                'learning_rate': 0.0001,
                'best_metric': 0.15,
                'best_epoch': 95
            }
            
            import json
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
            
            X_train = np.random.rand(50, 3)
            y_train = np.random.rand(50)
            
            new_model = MockModel()
            config = TrainingConfiguration(epochs=100, verbose=False)
            engine = TrainingEngine(config)
            
            # Should return the loaded model without additional training
            trained_model = engine.resume_training(
                checkpoint_path=checkpoint_path,
                model=new_model,
                X_train=X_train,
                y_train=y_train
            )
            
            assert isinstance(trained_model, TrainedModel)
            assert trained_model.training_time == 0.0  # No additional training time