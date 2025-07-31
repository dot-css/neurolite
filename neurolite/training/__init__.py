"""
Training orchestration module for NeuroLite.

Provides training engine, callbacks, and configuration management
for automated model training with intelligent defaults.
"""

from .trainer import TrainingEngine, TrainedModel
from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, ProgressMonitor
from .config import TrainingConfiguration, get_default_training_config

__all__ = [
    'TrainingEngine',
    'TrainedModel', 
    'CallbackManager',
    'EarlyStopping',
    'ModelCheckpoint',
    'ProgressMonitor',
    'TrainingConfiguration',
    'get_default_training_config'
]