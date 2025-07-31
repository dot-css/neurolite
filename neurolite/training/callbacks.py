"""
Training callback system for monitoring and controlling training process.

Provides callbacks for early stopping, checkpointing, progress monitoring,
and other training lifecycle events.
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..core import get_logger
from ..core.exceptions import TrainingError

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Current state of training process."""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    metrics: Dict[str, float] = None
    learning_rate: float = 0.0
    best_metric: Optional[float] = None
    best_epoch: int = 0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class BaseCallback(ABC):
    """Base class for training callbacks."""
    
    def __init__(self):
        self.enabled = True
    
    def on_train_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Called at the end of each batch."""
        pass
    
    def should_stop_training(self) -> bool:
        """Return True if training should be stopped."""
        return False


class EarlyStopping(BaseCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor for early stopping
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" for metrics that should decrease, "max" for metrics that should increase
            restore_best_weights: Whether to restore model weights from best epoch
            verbose: Whether to print early stopping messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_metric = None
        self._stop_training = False
        
        if mode == "min":
            self.monitor_op = np.less
            self.best_metric = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best_metric = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_train_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Reset early stopping state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self._stop_training = False
        
        if self.mode == "min":
            self.best_metric = np.inf
        else:
            self.best_metric = -np.inf
    
    def on_epoch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Check for early stopping condition at epoch end."""
        if not self.enabled:
            return
        
        # Get current metric value
        current_metric = self._get_monitor_value(state, logs)
        if current_metric is None:
            if self.verbose:
                logger.warning(f"Early stopping metric '{self.monitor}' not found")
            return
        
        # Check if current metric is better than best
        if self.monitor_op(current_metric - self.min_delta, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
            state.best_metric = current_metric
            state.best_epoch = state.epoch
            
            # Store best weights if requested
            if self.restore_best_weights and logs and 'model' in logs:
                self.best_weights = self._get_model_weights(logs['model'])
            
            if self.verbose:
                logger.info(f"Epoch {state.epoch}: {self.monitor} improved to {current_metric:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                logger.info(f"Epoch {state.epoch}: {self.monitor} did not improve from {self.best_metric:.6f}")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = state.epoch
            self._stop_training = True
            
            if self.verbose:
                logger.info(f"Early stopping at epoch {state.epoch}")
                logger.info(f"Best {self.monitor}: {self.best_metric:.6f} at epoch {state.best_epoch}")
    
    def on_train_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Restore best weights if early stopping occurred."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
        
        if (self.restore_best_weights and self.best_weights is not None and 
            logs and 'model' in logs):
            if self.verbose:
                logger.info(f"Restoring model weights from epoch {state.best_epoch}")
            self._set_model_weights(logs['model'], self.best_weights)
    
    def should_stop_training(self) -> bool:
        """Return True if training should be stopped."""
        return self._stop_training
    
    def _get_monitor_value(self, state: TrainingState, logs: Dict[str, Any] = None) -> Optional[float]:
        """Get the value of the monitored metric."""
        # Check in state metrics first
        if self.monitor in state.metrics:
            return state.metrics[self.monitor]
        
        # Check in logs
        if logs and self.monitor in logs:
            return logs[self.monitor]
        
        # Check common metric names
        if self.monitor == "val_loss" and state.val_loss is not None:
            return state.val_loss
        elif self.monitor == "train_loss":
            return state.train_loss
        
        return None
    
    def _get_model_weights(self, model: Any) -> Any:
        """Get model weights for restoration."""
        # This is framework-specific and should be implemented by subclasses
        # or handled by the training engine
        if hasattr(model, 'state_dict'):  # PyTorch
            import copy
            return copy.deepcopy(model.state_dict())
        elif hasattr(model, 'get_weights'):  # TensorFlow/Keras
            return model.get_weights()
        else:
            logger.warning("Cannot save model weights - unsupported model type")
            return None
    
    def _set_model_weights(self, model: Any, weights: Any) -> None:
        """Restore model weights."""
        if hasattr(model, 'load_state_dict') and weights is not None:  # PyTorch
            model.load_state_dict(weights)
        elif hasattr(model, 'set_weights') and weights is not None:  # TensorFlow/Keras
            model.set_weights(weights)
        else:
            logger.warning("Cannot restore model weights - unsupported model type")


class ModelCheckpoint(BaseCallback):
    """Callback to save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        save_frequency: int = 1,
        max_checkpoints: int = 5,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path template for saving checkpoints (can include {epoch}, {metric})
            monitor: Metric to monitor for best model selection
            save_best_only: Whether to save only the best model
            mode: "min" for metrics that should decrease, "max" for metrics that should increase
            save_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
            verbose: Whether to print checkpoint messages
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose
        
        self.best_metric = None
        self.checkpoint_files = []
        
        if mode == "min":
            self.monitor_op = np.less
            self.best_metric = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best_metric = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # Ensure checkpoint directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Initialize checkpoint state at training start."""
        if self.mode == "min":
            self.best_metric = np.inf
        else:
            self.best_metric = -np.inf
        
        self.checkpoint_files = []
    
    def on_epoch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Save checkpoint at epoch end if conditions are met."""
        if not self.enabled:
            return
        
        # Check if we should save this epoch
        if state.epoch % self.save_frequency != 0:
            return
        
        # Get current metric value
        current_metric = self._get_monitor_value(state, logs)
        should_save = False
        
        if self.save_best_only:
            if current_metric is not None:
                if self.monitor_op(current_metric, self.best_metric):
                    self.best_metric = current_metric
                    should_save = True
            else:
                if self.verbose:
                    logger.warning(f"Checkpoint metric '{self.monitor}' not found")
        else:
            should_save = True
        
        if should_save:
            self._save_checkpoint(state, logs, current_metric)
    
    def _save_checkpoint(self, state: TrainingState, logs: Dict[str, Any], metric_value: Optional[float]) -> None:
        """Save model checkpoint."""
        # Format filepath with current values
        filepath = self.filepath.format(
            epoch=state.epoch,
            metric=metric_value if metric_value is not None else 0,
            **state.metrics
        )
        
        try:
            # Save model
            if logs and 'model' in logs:
                model = logs['model']
                if hasattr(model, 'save'):
                    model.save(filepath)
                else:
                    # Try framework-specific saving
                    self._save_model_framework_specific(model, filepath)
            
            # Save training state
            state_filepath = filepath.replace('.', '_state.')
            if not state_filepath.endswith('.json'):
                state_filepath += '.json'
            
            state_data = {
                'epoch': state.epoch,
                'batch': state.batch,
                'train_loss': state.train_loss,
                'val_loss': state.val_loss,
                'metrics': state.metrics,
                'learning_rate': state.learning_rate,
                'best_metric': state.best_metric,
                'best_epoch': state.best_epoch
            }
            
            with open(state_filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Track checkpoint files
            self.checkpoint_files.append((filepath, state_filepath))
            
            # Remove old checkpoints if we exceed max_checkpoints
            if len(self.checkpoint_files) > self.max_checkpoints:
                old_model_file, old_state_file = self.checkpoint_files.pop(0)
                try:
                    if os.path.exists(old_model_file):
                        os.remove(old_model_file)
                    if os.path.exists(old_state_file):
                        os.remove(old_state_file)
                except OSError as e:
                    logger.warning(f"Could not remove old checkpoint: {e}")
            
            if self.verbose:
                logger.info(f"Saved checkpoint to {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_model_framework_specific(self, model: Any, filepath: str) -> None:
        """Save model using framework-specific methods."""
        import torch
        
        if hasattr(model, 'state_dict'):  # PyTorch
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__
            }, filepath)
        elif hasattr(model, 'save_weights'):  # TensorFlow/Keras
            model.save_weights(filepath)
        else:
            raise TrainingError(f"Cannot save model of type {type(model)}")
    
    def _get_monitor_value(self, state: TrainingState, logs: Dict[str, Any] = None) -> Optional[float]:
        """Get the value of the monitored metric."""
        # Check in state metrics first
        if self.monitor in state.metrics:
            return state.metrics[self.monitor]
        
        # Check in logs
        if logs and self.monitor in logs:
            return logs[self.monitor]
        
        # Check common metric names
        if self.monitor == "val_loss" and state.val_loss is not None:
            return state.val_loss
        elif self.monitor == "train_loss":
            return state.train_loss
        
        return None


class ProgressMonitor(BaseCallback):
    """Callback for monitoring and logging training progress."""
    
    def __init__(
        self,
        log_frequency: int = 10,
        verbose: bool = True,
        show_metrics: bool = True,
        show_time: bool = True
    ):
        """
        Initialize progress monitor callback.
        
        Args:
            log_frequency: Log progress every N batches
            verbose: Whether to print progress messages
            show_metrics: Whether to show metrics in progress
            show_time: Whether to show timing information
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.show_metrics = show_metrics
        self.show_time = show_time
        
        self.epoch_start_time = None
        self.batch_start_time = None
        self.train_start_time = None
    
    def on_train_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log training start."""
        self.train_start_time = time.time()
        if self.verbose:
            logger.info("Starting training...")
    
    def on_train_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log training completion."""
        if self.verbose and self.train_start_time:
            total_time = time.time() - self.train_start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Final train loss: {state.train_loss:.6f}")
            if state.val_loss is not None:
                logger.info(f"Final validation loss: {state.val_loss:.6f}")
    
    def on_epoch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        if self.verbose:
            logger.info(f"Epoch {state.epoch + 1}/{logs.get('total_epochs', '?')}")
    
    def on_epoch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log epoch completion."""
        if self.verbose and self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            message = f"Epoch {state.epoch + 1} completed"
            if self.show_time:
                message += f" in {epoch_time:.2f}s"
            
            message += f" - train_loss: {state.train_loss:.6f}"
            
            if state.val_loss is not None:
                message += f" - val_loss: {state.val_loss:.6f}"
            
            if self.show_metrics and state.metrics:
                for metric_name, metric_value in state.metrics.items():
                    message += f" - {metric_name}: {metric_value:.6f}"
            
            logger.info(message)
    
    def on_batch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log batch start."""
        self.batch_start_time = time.time()
    
    def on_batch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Log batch completion."""
        if (self.verbose and 
            state.batch % self.log_frequency == 0 and 
            self.batch_start_time):
            
            batch_time = time.time() - self.batch_start_time
            
            message = f"Batch {state.batch}/{state.total_batches}"
            if self.show_time:
                message += f" ({batch_time:.3f}s)"
            
            message += f" - loss: {state.train_loss:.6f}"
            
            if self.show_metrics and state.metrics:
                for metric_name, metric_value in state.metrics.items():
                    if not metric_name.startswith('val_'):
                        message += f" - {metric_name}: {metric_value:.6f}"
            
            logger.debug(message)


class CallbackManager:
    """Manages multiple training callbacks."""
    
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: BaseCallback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: BaseCallback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_train_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_train_begin(state, logs)
    
    def on_train_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_train_end(state, logs)
    
    def on_epoch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_epoch_begin(state, logs)
    
    def on_epoch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_epoch_end(state, logs)
    
    def on_batch_begin(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_batch_begin(state, logs)
    
    def on_batch_end(self, state: TrainingState, logs: Dict[str, Any] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                callback.on_batch_end(state, logs)
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests training to stop."""
        return any(callback.should_stop_training() for callback in self.callbacks if callback.enabled)
    
    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)
    
    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)