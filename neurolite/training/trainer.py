"""
Main training orchestration engine for NeuroLite.

Coordinates the entire training process including data preparation,
model training, evaluation, and callback management.
"""

import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from ..models.base import BaseModel, TaskType, ModelMetadata, PredictionResult
from ..data.detector import DataType
from ..core import get_logger
from ..core.exceptions import TrainingError, TrainingConfigurationError, TrainingFailedError
from .config import TrainingConfiguration, optimize_training_config
from .callbacks import CallbackManager, TrainingState, EarlyStopping, ModelCheckpoint, ProgressMonitor

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Training history containing loss and metrics over time."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    epochs: List[int] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    def add_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.0
    ) -> None:
        """Add epoch results to history."""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss if val_loss is not None else 0.0)
        self.learning_rates.append(learning_rate)
        
        if metrics:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                self.metrics[metric_name].append(metric_value)
    
    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> int:
        """Get epoch with best metric value."""
        if metric == "val_loss":
            values = self.val_loss
        elif metric == "train_loss":
            values = self.train_loss
        elif metric in self.metrics:
            values = self.metrics[metric]
        else:
            raise ValueError(f"Metric '{metric}' not found in history")
        
        if not values:
            return 0
        
        if mode == "min":
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))


@dataclass
class TrainedModel:
    """Container for a trained model with metadata and history."""
    model: BaseModel
    config: TrainingConfiguration
    history: TrainingHistory
    metadata: ModelMetadata
    training_time: float
    best_epoch: int
    
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """Make predictions using the trained model."""
        return self.model.predict(X, **kwargs)
    
    def predict_proba(self, X: Any, **kwargs) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X, **kwargs)
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        self.model.save(path)
        
        # Save additional metadata
        metadata_path = str(Path(path).with_suffix('.json'))
        import json
        
        metadata_dict = {
            'config': self.config.to_dict(),
            'training_time': self.training_time,
            'best_epoch': self.best_epoch,
            'history': {
                'train_loss': self.history.train_loss,
                'val_loss': self.history.val_loss,
                'metrics': self.history.metrics,
                'epochs': self.history.epochs,
                'learning_rates': self.history.learning_rates
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def load(self, path: str) -> 'TrainedModel':
        """Load a trained model."""
        self.model.load(path)
        
        # Load additional metadata
        metadata_path = str(Path(path).with_suffix('.json'))
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            self.training_time = metadata_dict.get('training_time', 0.0)
            self.best_epoch = metadata_dict.get('best_epoch', 0)
            
            # Reconstruct history
            history_data = metadata_dict.get('history', {})
            self.history = TrainingHistory(
                train_loss=history_data.get('train_loss', []),
                val_loss=history_data.get('val_loss', []),
                metrics=history_data.get('metrics', {}),
                epochs=history_data.get('epochs', []),
                learning_rates=history_data.get('learning_rates', [])
            )
        
        return self


class TrainingEngine:
    """
    Main training orchestration engine.
    
    Coordinates the entire training process including data preparation,
    model training, evaluation, and callback management with intelligent defaults.
    """
    
    def __init__(self, config: Optional[TrainingConfiguration] = None):
        """
        Initialize training engine.
        
        Args:
            config: Training configuration (uses defaults if None)
        """
        self.config = config or TrainingConfiguration()
        self.callback_manager = CallbackManager()
        self._setup_default_callbacks()
        
        logger.debug("Initialized TrainingEngine")
    
    def _setup_default_callbacks(self) -> None:
        """Setup default callbacks based on configuration."""
        # Add early stopping if enabled
        if self.config.early_stopping.enabled:
            early_stopping = EarlyStopping(
                monitor=self.config.early_stopping.monitor,
                patience=self.config.early_stopping.patience,
                min_delta=self.config.early_stopping.min_delta,
                mode=self.config.early_stopping.mode,
                restore_best_weights=self.config.early_stopping.restore_best_weights,
                verbose=self.config.verbose
            )
            self.callback_manager.add_callback(early_stopping)
        
        # Add progress monitoring
        if self.config.verbose:
            progress_monitor = ProgressMonitor(
                log_frequency=self.config.log_frequency,
                verbose=self.config.verbose
            )
            self.callback_manager.add_callback(progress_monitor)
    
    def add_callback(self, callback) -> None:
        """Add a custom callback."""
        self.callback_manager.add_callback(callback)
    
    def optimize_hyperparameters(
        self,
        model_factory: Any,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        optimization_config: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Optimize hyperparameters for the given model and data.
        
        Args:
            model_factory: Function that creates model instances
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            task_type: Type of ML task
            data_type: Type of input data
            optimization_config: Hyperparameter optimization configuration
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult containing best parameters and model
        """
        # Import here to avoid circular imports
        from .optimizer import HyperparameterOptimizer, OptimizationConfig
        
        # Use provided config or create default
        config = optimization_config or OptimizationConfig()
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(config)
        
        # Run optimization
        result = optimizer.optimize(
            model_factory=model_factory,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            data_type=data_type,
            base_config=self.config,
            **kwargs
        )
        
        return result
    
    def train(
        self,
        model: BaseModel,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        **kwargs
    ) -> TrainedModel:
        """
        Train a model with the provided data.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            task_type: Type of ML task
            data_type: Type of input data
            **kwargs: Additional training parameters
            
        Returns:
            TrainedModel containing the trained model and metadata
            
        Raises:
            TrainingError: If training fails
        """
        logger.info("Starting model training")
        start_time = time.time()
        
        try:
            # Update configuration with any overrides
            if kwargs:
                self.config.update(**kwargs)
            
            # Optimize configuration if task and data types are provided
            if task_type and data_type:
                num_samples = len(X_train) if hasattr(X_train, '__len__') else 1000
                optimized_config = optimize_training_config(
                    task_type=task_type,
                    data_type=data_type,
                    num_samples=num_samples,
                    **kwargs
                )
                # Merge with current config
                optimized_dict = optimized_config.to_dict()
                for key, value in optimized_dict.items():
                    if hasattr(self.config, key):
                        current_value = getattr(self.config, key)
                        default_value = getattr(TrainingConfiguration(), key)
                        
                        # Only update if current value is still default
                        if isinstance(value, dict) and hasattr(current_value, '__dict__'):
                            # Handle nested config objects
                            for nested_key, nested_value in value.items():
                                if hasattr(current_value, nested_key):
                                    setattr(current_value, nested_key, nested_value)
                        elif current_value == default_value:
                            setattr(self.config, key, value)
            
            # Validate model compatibility
            if task_type and data_type:
                model.validate_data(X_train, y_train, task_type, data_type)
            
            # Setup checkpointing if enabled
            if self.config.checkpoint.enabled:
                checkpoint_dir = Path("checkpoints") / f"model_{int(time.time())}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_callback = ModelCheckpoint(
                    filepath=str(checkpoint_dir / "checkpoint_epoch_{epoch:03d}.pt"),
                    monitor=self.config.checkpoint.monitor,
                    save_best_only=self.config.checkpoint.save_best_only,
                    mode=self.config.checkpoint.mode,
                    save_frequency=self.config.checkpoint.save_frequency,
                    max_checkpoints=self.config.checkpoint.max_checkpoints,
                    verbose=self.config.verbose
                )
                self.callback_manager.add_callback(checkpoint_callback)
            
            # Initialize training state
            state = TrainingState()
            history = TrainingHistory()
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Initialize callbacks
            logs = {
                'model': model,
                'total_epochs': self.config.epochs,
                'config': self.config
            }
            self.callback_manager.on_train_begin(state, logs)
            
            # Training loop
            best_val_loss = float('inf')
            best_epoch = 0
            
            for epoch in range(self.config.epochs):
                state.epoch = epoch
                
                # Epoch begin callbacks
                self.callback_manager.on_epoch_begin(state, logs)
                
                # Train for one epoch
                epoch_train_loss, epoch_metrics = self._train_epoch(
                    model, X_train, y_train, state, logs
                )
                
                # Validation
                epoch_val_loss = None
                val_metrics = {}
                if validation_data is not None:
                    epoch_val_loss, val_metrics = self._validate_epoch(
                        model, validation_data[0], validation_data[1], state, logs
                    )
                
                # Update state
                state.train_loss = epoch_train_loss
                state.val_loss = epoch_val_loss
                state.metrics.update(epoch_metrics)
                state.metrics.update(val_metrics)
                
                # Track best model
                current_val_loss = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    state.best_metric = best_val_loss
                    state.best_epoch = best_epoch
                
                # Add to history
                history.add_epoch(
                    epoch=epoch,
                    train_loss=epoch_train_loss,
                    val_loss=epoch_val_loss,
                    metrics=state.metrics.copy(),
                    learning_rate=state.learning_rate
                )
                
                # Epoch end callbacks
                self.callback_manager.on_epoch_end(state, logs)
                
                # Check for early stopping
                if self.callback_manager.should_stop_training():
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break
            
            # Training end callbacks
            self.callback_manager.on_train_end(state, logs)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create model metadata
            metadata = ModelMetadata(
                name=model.__class__.__name__,
                task_type=task_type or TaskType.AUTO,
                data_type=data_type or DataType.UNKNOWN,
                framework=model.capabilities.framework,
                training_time=training_time,
                training_samples=len(X_train) if hasattr(X_train, '__len__') else None,
                validation_accuracy=state.metrics.get('val_accuracy'),
                hyperparameters=self.config.to_dict()
            )
            
            # Mark model as trained
            model.is_trained = True
            model.metadata = metadata
            
            # Create trained model container
            trained_model = TrainedModel(
                model=model,
                config=self.config,
                history=history,
                metadata=metadata,
                training_time=training_time,
                best_epoch=best_epoch
            )
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
            
            return trained_model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise TrainingFailedError(str(e)) from e
    
    def _train_epoch(
        self, 
        model: BaseModel, 
        X_train: Any, 
        y_train: Any, 
        state: TrainingState,
        logs: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train model for one epoch.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            state: Current training state
            logs: Training logs
            
        Returns:
            Tuple of (epoch_loss, epoch_metrics)
        """
        # This is a simplified implementation
        # In practice, this would handle batching, gradient updates, etc.
        
        # For sklearn models, training is done in one step
        if hasattr(model, 'sklearn_model'):
            if not model.is_trained:
                model.fit(X_train, y_train)
            
            # Calculate training loss (simplified)
            predictions = model.predict(X_train)
            train_loss = self._calculate_loss(y_train, predictions.predictions)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, predictions.predictions)
            
            return train_loss, metrics
        
        # For deep learning models, this would implement proper batch training
        else:
            # Placeholder for deep learning training loop
            # This would involve:
            # 1. Creating data loaders
            # 2. Iterating through batches
            # 3. Forward pass
            # 4. Loss calculation
            # 5. Backward pass
            # 6. Optimizer step
            
            # For now, just call the model's fit method
            validation_data = logs.get('validation_data')
            model.fit(X_train, y_train, validation_data=validation_data)
            
            # Calculate training loss and metrics
            predictions = model.predict(X_train)
            train_loss = self._calculate_loss(y_train, predictions.predictions)
            metrics = self._calculate_metrics(y_train, predictions.predictions)
            
            return train_loss, metrics
    
    def _validate_epoch(
        self,
        model: BaseModel,
        X_val: Any,
        y_val: Any,
        state: TrainingState,
        logs: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model for one epoch.
        
        Args:
            model: Model to validate
            X_val: Validation features
            y_val: Validation targets
            state: Current training state
            logs: Training logs
            
        Returns:
            Tuple of (val_loss, val_metrics)
        """
        try:
            # Make predictions
            predictions = model.predict(X_val)
            
            # Calculate validation loss
            val_loss = self._calculate_loss(y_val, predictions.predictions)
            
            # Calculate validation metrics
            val_metrics = self._calculate_metrics(y_val, predictions.predictions)
            
            # Add 'val_' prefix to metric names
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            
            return val_loss, val_metrics
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return float('inf'), {}
    
    def _calculate_loss(self, y_true: Any, y_pred: Any) -> float:
        """Calculate loss between true and predicted values."""
        try:
            # Convert to numpy arrays
            if not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Determine loss function based on configuration
            loss_func_value = self.config.loss_function.value if hasattr(self.config.loss_function, 'value') else self.config.loss_function
            if loss_func_value == "mse":
                return float(np.mean((y_true - y_pred) ** 2))
            elif loss_func_value == "mae":
                return float(np.mean(np.abs(y_true - y_pred)))
            elif loss_func_value in ["cross_entropy", "binary_cross_entropy"]:
                # Simplified cross-entropy calculation
                # In practice, this would be more sophisticated
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Multi-class
                    return float(-np.mean(np.log(np.maximum(y_pred[np.arange(len(y_true)), y_true], 1e-15))))
                else:
                    # Binary
                    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
            else:
                # Default to MSE
                return float(np.mean((y_true - y_pred) ** 2))
                
        except Exception as e:
            logger.warning(f"Loss calculation failed: {e}")
            return float('inf')
    
    def _calculate_metrics(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        try:
            # Convert to numpy arrays
            if not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Calculate metrics based on configuration
            for metric_name in self.config.metrics:
                if metric_name == "accuracy":
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        # Multi-class: take argmax
                        y_pred_classes = np.argmax(y_pred, axis=1)
                    else:
                        # Binary: threshold at 0.5
                        y_pred_classes = (y_pred > 0.5).astype(int)
                    
                    accuracy = np.mean(y_true == y_pred_classes)
                    metrics["accuracy"] = float(accuracy)
                
                elif metric_name == "mse":
                    mse = np.mean((y_true - y_pred) ** 2)
                    metrics["mse"] = float(mse)
                
                elif metric_name == "mae":
                    mae = np.mean(np.abs(y_true - y_pred))
                    metrics["mae"] = float(mae)
                
                elif metric_name == "r2":
                    # R-squared for regression
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-15))
                    metrics["r2"] = float(r2)
                
                # Additional metrics would be implemented here
                # (precision, recall, f1, etc.)
                
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
        
        return metrics
    
    def resume_training(
        self,
        checkpoint_path: str,
        model: BaseModel,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **kwargs
    ) -> TrainedModel:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to resume training
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            TrainedModel containing the resumed trained model
        """
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            # Load model state
            model.load(checkpoint_path)
            
            # Load training state if available
            state_path = checkpoint_path.rsplit('.', 1)[0] + '_state.json'
            
            start_epoch = 0
            if os.path.exists(state_path):
                import json
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                start_epoch = state_data.get('epoch', 0) + 1
                logger.info(f"Resuming from epoch {start_epoch}")
            
            # Update configuration to start from the correct epoch
            remaining_epochs = max(0, self.config.epochs - start_epoch)
            if remaining_epochs <= 0:
                logger.warning("Model has already completed training")
                # Return a TrainedModel with the loaded model
                return TrainedModel(
                    model=model,
                    config=self.config,
                    history=TrainingHistory(),
                    metadata=model.metadata or ModelMetadata(
                        name=model.__class__.__name__,
                        task_type=TaskType.AUTO,
                        data_type=DataType.UNKNOWN,
                        framework=model.capabilities.framework,
                        training_time=0.0
                    ),
                    training_time=0.0,
                    best_epoch=start_epoch
                )
            
            # Update epochs to continue training
            original_epochs = self.config.epochs
            self.config.epochs = remaining_epochs
            
            # Continue training
            trained_model = self.train(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **kwargs
            )
            
            # Restore original epoch count
            self.config.epochs = original_epochs
            
            return trained_model
            
        except Exception as e:
            logger.error(f"Failed to resume training: {str(e)}")
            raise TrainingFailedError(f"Resume training failed: {str(e)}") from e