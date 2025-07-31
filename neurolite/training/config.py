"""
Training configuration management with intelligent defaults.

Provides configuration classes and utilities for training parameters
with automatic optimization based on data characteristics and task type.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import numpy as np

from ..models.base import TaskType
from ..data.detector import DataType
from ..core import get_logger

logger = get_logger(__name__)


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class LossFunction(Enum):
    """Supported loss functions."""
    # Classification losses
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    SPARSE_CATEGORICAL_CROSS_ENTROPY = "sparse_categorical_cross_entropy"
    FOCAL_LOSS = "focal_loss"
    
    # Regression losses
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    
    # Auto selection
    AUTO = "auto"


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    NONE = "none"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9     # For Adam/AdamW
    beta2: float = 0.999   # For Adam/AdamW
    epsilon: float = 1e-8  # For Adam/AdamW


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    step_size: int = 30        # For StepLR
    gamma: float = 0.1         # Decay factor
    patience: int = 10         # For ReduceLROnPlateau
    factor: float = 0.5        # For ReduceLROnPlateau
    min_lr: float = 1e-7       # Minimum learning rate


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    restore_best_weights: bool = True


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""
    enabled: bool = True
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    save_frequency: int = 1  # Save every N epochs
    max_checkpoints: int = 5  # Maximum number of checkpoints to keep


@dataclass
class TrainingConfiguration:
    """Complete training configuration with intelligent defaults."""
    
    # Basic training parameters
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Loss and metrics
    loss_function: LossFunction = LossFunction.AUTO
    metrics: List[str] = field(default_factory=list)
    
    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Learning rate scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Callbacks
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Device and precision
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Regularization
    dropout_rate: float = 0.0
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    
    # Gradient clipping
    gradient_clip_value: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    
    # Logging and monitoring
    log_frequency: int = 10  # Log every N batches
    verbose: bool = True
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and adjustments."""
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate epochs
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        # Validate splits
        if not 0 <= self.validation_split <= 1:
            raise ValueError("Validation split must be between 0 and 1")
        if not 0 <= self.test_split <= 1:
            raise ValueError("Test split must be between 0 and 1")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("Validation and test splits combined must be less than 1")
        
        # Auto-configure metrics if empty
        if not self.metrics:
            self.metrics = self._get_default_metrics()
    
    def _get_default_metrics(self) -> List[str]:
        """Get default metrics based on loss function."""
        if self.loss_function in [LossFunction.CROSS_ENTROPY, 
                                  LossFunction.BINARY_CROSS_ENTROPY,
                                  LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY]:
            return ["accuracy"]
        elif self.loss_function in [LossFunction.MSE, LossFunction.MAE, LossFunction.HUBER]:
            return ["mae", "mse"]
        else:
            return ["accuracy"]  # Default fallback
    
    def update(self, **kwargs) -> 'TrainingConfiguration':
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_params[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, '__dict__'):
                # Handle nested dataclass objects
                nested_dict = {}
                for nested_key, nested_value in value.__dict__.items():
                    if isinstance(nested_value, Enum):
                        nested_dict[nested_key] = nested_value.value
                    else:
                        nested_dict[nested_key] = nested_value
                result[key] = nested_dict
            else:
                result[key] = value
        return result


class ConfigurationOptimizer:
    """Optimizes training configuration based on data characteristics."""
    
    @staticmethod
    def optimize_for_task_and_data(
        task_type: TaskType,
        data_type: DataType,
        num_samples: int,
        num_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        base_config: Optional[TrainingConfiguration] = None
    ) -> TrainingConfiguration:
        """
        Optimize training configuration for specific task and data characteristics.
        
        Args:
            task_type: Type of ML task
            data_type: Type of input data
            num_samples: Number of training samples
            num_features: Number of input features (for tabular data)
            num_classes: Number of classes (for classification)
            base_config: Base configuration to modify
            
        Returns:
            Optimized training configuration
        """
        config = base_config or TrainingConfiguration()
        
        # Optimize batch size based on data size
        config.batch_size = ConfigurationOptimizer._optimize_batch_size(
            num_samples, data_type
        )
        
        # Optimize epochs based on data size and type
        config.epochs = ConfigurationOptimizer._optimize_epochs(
            num_samples, data_type, task_type
        )
        
        # Optimize learning rate based on task and data type
        config.optimizer.learning_rate = ConfigurationOptimizer._optimize_learning_rate(
            task_type, data_type, num_samples
        )
        
        # Set appropriate loss function
        config.loss_function = ConfigurationOptimizer._select_loss_function(
            task_type, num_classes
        )
        
        # Optimize early stopping patience
        config.early_stopping.patience = ConfigurationOptimizer._optimize_patience(
            num_samples, config.epochs
        )
        
        # Set appropriate metrics
        config.metrics = ConfigurationOptimizer._select_metrics(task_type)
        
        logger.debug(f"Optimized configuration for {task_type.value} task with {data_type.value} data")
        return config
    
    @staticmethod
    def _optimize_batch_size(num_samples: int, data_type: DataType) -> int:
        """Optimize batch size based on data characteristics."""
        if data_type == DataType.IMAGE:
            # Images require more memory
            if num_samples < 1000:
                return min(16, num_samples)
            elif num_samples < 10000:
                return 32
            else:
                return 64
        elif data_type == DataType.TEXT:
            # Text can vary greatly in length
            if num_samples < 1000:
                return min(8, num_samples)
            elif num_samples < 10000:
                return 16
            else:
                return 32
        else:  # Tabular and others
            if num_samples < 1000:
                return min(32, num_samples)
            elif num_samples < 10000:
                return 64
            else:
                return 128
    
    @staticmethod
    def _optimize_epochs(num_samples: int, data_type: DataType, task_type: TaskType) -> int:
        """Optimize number of epochs based on data characteristics."""
        base_epochs = 100
        
        # Adjust based on data size
        if num_samples < 1000:
            base_epochs = 200  # More epochs for small datasets
        elif num_samples > 100000:
            base_epochs = 50   # Fewer epochs for large datasets
        
        # Adjust based on data type
        if data_type == DataType.IMAGE:
            base_epochs = max(50, base_epochs)  # Images often need more training
        elif data_type == DataType.TEXT:
            base_epochs = max(30, base_epochs)  # Text models can converge faster
        
        return base_epochs
    
    @staticmethod
    def _optimize_learning_rate(task_type: TaskType, data_type: DataType, num_samples: int) -> float:
        """Optimize learning rate based on task and data characteristics."""
        base_lr = 0.001
        
        # Adjust based on data type
        if data_type == DataType.IMAGE:
            base_lr = 0.0001  # Lower LR for image models
        elif data_type == DataType.TEXT:
            base_lr = 0.00005  # Even lower for text models
        
        # Adjust based on dataset size
        if num_samples < 1000:
            base_lr *= 0.5  # Lower LR for small datasets
        elif num_samples > 100000:
            base_lr *= 2.0  # Higher LR for large datasets
        
        return base_lr
    
    @staticmethod
    def _select_loss_function(task_type: TaskType, num_classes: Optional[int]) -> LossFunction:
        """Select appropriate loss function based on task type."""
        if task_type in [TaskType.BINARY_CLASSIFICATION]:
            return LossFunction.BINARY_CROSS_ENTROPY
        elif task_type in [TaskType.CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION,
                          TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_CLASSIFICATION]:
            return LossFunction.CROSS_ENTROPY
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION]:
            return LossFunction.MSE
        else:
            return LossFunction.AUTO
    
    @staticmethod
    def _optimize_patience(num_samples: int, epochs: int) -> int:
        """Optimize early stopping patience."""
        base_patience = max(10, epochs // 10)  # At least 10, or 10% of epochs
        
        if num_samples < 1000:
            return min(base_patience, 20)  # Cap patience for small datasets
        else:
            return base_patience
    
    @staticmethod
    def _select_metrics(task_type: TaskType) -> List[str]:
        """Select appropriate metrics based on task type."""
        if task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION,
                        TaskType.TEXT_CLASSIFICATION]:
            return ["accuracy", "precision", "recall", "f1"]
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION]:
            return ["mae", "mse", "r2"]
        else:
            return ["accuracy"]


def get_default_training_config() -> TrainingConfiguration:
    """Get default training configuration."""
    return TrainingConfiguration()


def optimize_training_config(
    task_type: TaskType,
    data_type: DataType,
    num_samples: int,
    num_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    **overrides
) -> TrainingConfiguration:
    """
    Get optimized training configuration for specific task and data.
    
    Args:
        task_type: Type of ML task
        data_type: Type of input data
        num_samples: Number of training samples
        num_features: Number of input features
        num_classes: Number of classes
        **overrides: Configuration overrides
        
    Returns:
        Optimized training configuration
    """
    config = ConfigurationOptimizer.optimize_for_task_and_data(
        task_type=task_type,
        data_type=data_type,
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes
    )
    
    # Apply any overrides
    if overrides:
        config.update(**overrides)
    
    return config