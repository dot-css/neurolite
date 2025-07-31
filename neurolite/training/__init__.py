"""
Training orchestration module for NeuroLite.

Provides training engine, callbacks, configuration management,
and hyperparameter optimization for automated model training with intelligent defaults.
"""

from .trainer import TrainingEngine, TrainedModel
from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, ProgressMonitor
from .config import TrainingConfiguration, get_default_training_config
from .optimizer import (
    HyperparameterOptimizer, 
    OptimizationConfig, 
    OptimizationResult,
    OptimizationBounds,
    ResourceConstraints,
    SearchStrategy,
    PruningStrategy,
    optimize_hyperparameters
)

__all__ = [
    'TrainingEngine',
    'TrainedModel', 
    'CallbackManager',
    'EarlyStopping',
    'ModelCheckpoint',
    'ProgressMonitor',
    'TrainingConfiguration',
    'get_default_training_config',
    'HyperparameterOptimizer',
    'OptimizationConfig',
    'OptimizationResult',
    'OptimizationBounds',
    'ResourceConstraints',
    'SearchStrategy',
    'PruningStrategy',
    'optimize_hyperparameters'
]