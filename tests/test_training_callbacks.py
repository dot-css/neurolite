"""
Unit tests for training callbacks.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, MagicMock
import numpy as np

from neurolite.training.callbacks import (
    BaseCallback, EarlyStopping, ModelCheckpoint, ProgressMonitor,
    CallbackManager, TrainingState
)


class TestTrainingState:
    """Test training state class."""
    
    def test_training_state_initialization(self):
        """Test training state initialization."""
        state = TrainingState()
        
        assert state.epoch == 0
        assert state.batch == 0
        assert state.total_batches == 0
        assert state.train_loss == 0.0
        assert state.val_loss is None
        assert state.metrics == {}
        assert state.learning_rate == 0.0
        assert state.best_metric is None
        assert state.best_epoch == 0
    
    def test_training_state_with_values(self):
        """Test training state with initial values."""
        metrics = {"accuracy": 0.85, "loss": 0.3}
        state = TrainingState(
            epoch=5,
            batch=100,
            train_loss=0.25,
            val_loss=0.35,
            metrics=metrics,
            learning_rate=0.001
        )
        
        assert state.epoch == 5
        assert state.batch == 100
        assert state.train_loss == 0.25
        assert state.val_loss == 0.35
        assert state.metrics == metrics
        assert state.learning_rate == 0.001


class TestBaseCallback:
    """Test base callback class."""
    
    def test_base_callback_initialization(self):
        """Test base callback initialization."""
        callback = BaseCallback()
        assert callback.enabled is True
    
    def test_base_callback_methods(self):
        """Test base callback methods do nothing by default."""
        callback = BaseCallback()
        state = TrainingState()
        logs = {}
        
        # Should not raise any exceptions
        callback.on_train_begin(state, logs)
        callback.on_train_end(state, logs)
        callback.on_epoch_begin(state, logs)
        callback.on_epoch_end(state, logs)
        callback.on_batch_begin(state, logs)
        callback.on_batch_end(state, logs)
        
        assert callback.should_stop_training() is False


class TestEarlyStopping:
    """Test early stopping callback."""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.01,
            mode="min",
            verbose=False
        )
        
        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.mode == "min"
        assert callback.wait == 0
        assert callback.best_metric == np.inf
    
    def test_early_stopping_invalid_mode(self):
        """Test early stopping with invalid mode."""
        with pytest.raises(ValueError, match="Mode must be 'min' or 'max'"):
            EarlyStopping(mode="invalid")
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        callback = EarlyStopping(
            monitor="val_loss",
            patience=2,
            min_delta=0.01,
            mode="min",
            verbose=False
        )
        
        state = TrainingState()
        
        # Initialize
        callback.on_train_begin(state, {})
        assert callback.best_metric == np.inf
        assert callback.wait == 0
        
        # First epoch - improvement
        state.epoch = 0
        state.val_loss = 0.5
        callback.on_epoch_end(state, {})
        assert callback.best_metric == 0.5
        assert callback.wait == 0
        assert not callback.should_stop_training()
        
        # Second epoch - improvement
        state.epoch = 1
        state.val_loss = 0.4
        callback.on_epoch_end(state, {})
        assert callback.best_metric == 0.4
        assert callback.wait == 0
        assert not callback.should_stop_training()
        
        # Third epoch - no improvement
        state.epoch = 2
        state.val_loss = 0.45
        callback.on_epoch_end(state, {})
        assert callback.best_metric == 0.4
        assert callback.wait == 1
        assert not callback.should_stop_training()
        
        # Fourth epoch - no improvement, should trigger stopping
        state.epoch = 3
        state.val_loss = 0.42
        callback.on_epoch_end(state, {})
        assert callback.wait == 2
        assert callback.should_stop_training()
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        callback = EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            mode="max",
            verbose=False
        )
        
        state = TrainingState()
        
        # Initialize
        callback.on_train_begin(state, {})
        assert callback.best_metric == -np.inf
        
        # First epoch - improvement
        state.epoch = 0
        state.metrics = {"val_accuracy": 0.8}
        callback.on_epoch_end(state, {})
        assert callback.best_metric == 0.8
        assert callback.wait == 0
        
        # Second epoch - no improvement
        state.epoch = 1
        state.metrics = {"val_accuracy": 0.75}
        callback.on_epoch_end(state, {})
        assert callback.wait == 1
        
        # Third epoch - no improvement, should trigger stopping
        state.epoch = 2
        state.metrics = {"val_accuracy": 0.78}
        callback.on_epoch_end(state, {})
        assert callback.should_stop_training()
    
    def test_early_stopping_missing_metric(self):
        """Test early stopping with missing metric."""
        callback = EarlyStopping(
            monitor="missing_metric",
            patience=2,
            verbose=False
        )
        
        state = TrainingState()
        callback.on_train_begin(state, {})
        
        # Epoch with missing metric
        state.epoch = 0
        state.val_loss = 0.5
        callback.on_epoch_end(state, {})
        
        # Should not trigger stopping due to missing metric
        assert not callback.should_stop_training()
    
    def test_early_stopping_restore_weights(self):
        """Test early stopping with weight restoration."""
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weight": "best_weights"}
        mock_model.load_state_dict = Mock()
        
        callback = EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
            verbose=False
        )
        
        state = TrainingState()
        logs = {"model": mock_model}
        
        callback.on_train_begin(state, logs)
        
        # First epoch - improvement, should save weights
        state.epoch = 0
        state.val_loss = 0.5
        callback.on_epoch_end(state, logs)
        
        # Second epoch - no improvement, should trigger stopping
        state.epoch = 1
        state.val_loss = 0.6
        callback.on_epoch_end(state, logs)
        
        assert callback.should_stop_training()
        
        # On training end, should restore weights
        callback.on_train_end(state, logs)
        mock_model.load_state_dict.assert_called_once()


class TestModelCheckpoint:
    """Test model checkpoint callback."""
    
    def test_model_checkpoint_initialization(self):
        """Test model checkpoint initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "checkpoint.pt")
            callback = ModelCheckpoint(
                filepath=filepath,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=False
            )
            
            assert callback.filepath == filepath
            assert callback.monitor == "val_loss"
            assert callback.save_best_only is True
            assert callback.mode == "min"
            assert callback.best_metric == np.inf
    
    def test_model_checkpoint_save_best_only(self):
        """Test model checkpoint saving only best models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "checkpoint_epoch_{epoch:03d}.pt")
            
            mock_model = Mock()
            mock_model.save = Mock()
            
            callback = ModelCheckpoint(
                filepath=filepath,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=False
            )
            
            state = TrainingState()
            logs = {"model": mock_model}
            
            callback.on_train_begin(state, logs)
            
            # First epoch - should save (first is always best)
            state.epoch = 0
            state.val_loss = 0.5
            callback.on_epoch_end(state, logs)
            mock_model.save.assert_called_once()
            
            mock_model.save.reset_mock()
            
            # Second epoch - worse, should not save
            state.epoch = 1
            state.val_loss = 0.6
            callback.on_epoch_end(state, logs)
            mock_model.save.assert_not_called()
            
            # Third epoch - better, should save
            state.epoch = 2
            state.val_loss = 0.4
            callback.on_epoch_end(state, logs)
            mock_model.save.assert_called_once()
    
    def test_model_checkpoint_save_all(self):
        """Test model checkpoint saving all epochs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "checkpoint_epoch_{epoch:03d}.pt")
            
            mock_model = Mock()
            mock_model.save = Mock()
            
            callback = ModelCheckpoint(
                filepath=filepath,
                save_best_only=False,
                save_frequency=1,
                verbose=False
            )
            
            state = TrainingState()
            logs = {"model": mock_model}
            
            callback.on_train_begin(state, logs)
            
            # Should save every epoch
            for epoch in range(3):
                state.epoch = epoch
                state.val_loss = 0.5
                callback.on_epoch_end(state, logs)
            
            assert mock_model.save.call_count == 3
    
    def test_model_checkpoint_save_frequency(self):
        """Test model checkpoint save frequency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "checkpoint_epoch_{epoch:03d}.pt")
            
            mock_model = Mock()
            mock_model.save = Mock()
            
            callback = ModelCheckpoint(
                filepath=filepath,
                save_best_only=False,
                save_frequency=2,  # Save every 2 epochs
                verbose=False
            )
            
            state = TrainingState()
            logs = {"model": mock_model}
            
            callback.on_train_begin(state, logs)
            
            # Epochs 0, 1, 2, 3, 4
            for epoch in range(5):
                state.epoch = epoch
                state.val_loss = 0.5
                callback.on_epoch_end(state, logs)
            
            # Should save on epochs 0, 2, 4 (every 2nd epoch)
            assert mock_model.save.call_count == 3


class TestProgressMonitor:
    """Test progress monitor callback."""
    
    def test_progress_monitor_initialization(self):
        """Test progress monitor initialization."""
        callback = ProgressMonitor(
            log_frequency=5,
            verbose=True,
            show_metrics=True,
            show_time=True
        )
        
        assert callback.log_frequency == 5
        assert callback.verbose is True
        assert callback.show_metrics is True
        assert callback.show_time is True
    
    def test_progress_monitor_timing(self):
        """Test progress monitor timing functionality."""
        callback = ProgressMonitor(verbose=False)  # Disable logging for test
        
        state = TrainingState()
        logs = {"total_epochs": 10}
        
        # Test timing tracking
        callback.on_train_begin(state, logs)
        assert callback.train_start_time is not None
        
        callback.on_epoch_begin(state, logs)
        assert callback.epoch_start_time is not None
        
        callback.on_batch_begin(state, logs)
        assert callback.batch_start_time is not None


class TestCallbackManager:
    """Test callback manager."""
    
    def test_callback_manager_initialization(self):
        """Test callback manager initialization."""
        manager = CallbackManager()
        assert len(manager) == 0
        
        callbacks = [BaseCallback(), BaseCallback()]
        manager = CallbackManager(callbacks)
        assert len(manager) == 2
    
    def test_callback_manager_add_remove(self):
        """Test adding and removing callbacks."""
        manager = CallbackManager()
        callback1 = BaseCallback()
        callback2 = BaseCallback()
        
        manager.add_callback(callback1)
        assert len(manager) == 1
        
        manager.add_callback(callback2)
        assert len(manager) == 2
        
        manager.remove_callback(callback1)
        assert len(manager) == 1
        assert callback2 in manager.callbacks
    
    def test_callback_manager_event_propagation(self):
        """Test callback manager event propagation."""
        callback1 = Mock(spec=BaseCallback)
        callback1.enabled = True
        callback2 = Mock(spec=BaseCallback)
        callback2.enabled = True
        
        manager = CallbackManager([callback1, callback2])
        state = TrainingState()
        logs = {}
        
        # Test all event methods
        manager.on_train_begin(state, logs)
        callback1.on_train_begin.assert_called_once_with(state, logs)
        callback2.on_train_begin.assert_called_once_with(state, logs)
        
        manager.on_epoch_end(state, logs)
        callback1.on_epoch_end.assert_called_once_with(state, logs)
        callback2.on_epoch_end.assert_called_once_with(state, logs)
    
    def test_callback_manager_disabled_callback(self):
        """Test callback manager with disabled callback."""
        callback1 = Mock(spec=BaseCallback)
        callback1.enabled = True
        callback2 = Mock(spec=BaseCallback)
        callback2.enabled = False  # Disabled
        
        manager = CallbackManager([callback1, callback2])
        state = TrainingState()
        logs = {}
        
        manager.on_train_begin(state, logs)
        callback1.on_train_begin.assert_called_once_with(state, logs)
        callback2.on_train_begin.assert_not_called()  # Should not be called
    
    def test_callback_manager_should_stop_training(self):
        """Test callback manager stop training check."""
        callback1 = Mock(spec=BaseCallback)
        callback1.enabled = True
        callback1.should_stop_training.return_value = False
        
        callback2 = Mock(spec=BaseCallback)
        callback2.enabled = True
        callback2.should_stop_training.return_value = True
        
        manager = CallbackManager([callback1, callback2])
        
        # Should return True if any callback requests stopping
        assert manager.should_stop_training() is True
        
        # Test with no stopping requests
        callback2.should_stop_training.return_value = False
        assert manager.should_stop_training() is False
    
    def test_callback_manager_iteration(self):
        """Test callback manager iteration."""
        callback1 = BaseCallback()
        callback2 = BaseCallback()
        manager = CallbackManager([callback1, callback2])
        
        callbacks = list(manager)
        assert len(callbacks) == 2
        assert callback1 in callbacks
        assert callback2 in callbacks