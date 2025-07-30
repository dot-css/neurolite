"""
Model zoo and registry system for NeuroLite.

Provides unified interfaces for different model types and automatic model selection.
"""

from .base import BaseModel, TaskType
from .registry import ModelRegistry, get_model_registry
from .factory import ModelFactory, create_model

# Import model registration functions
from .ml.sklearn_models import register_sklearn_models
from .ml.xgboost_models import register_xgboost_models
from .ml.ensemble_models import register_ensemble_models


def initialize_models():
    """Initialize and register all available models."""
    try:
        register_sklearn_models()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to register sklearn models: {e}")
    
    try:
        register_xgboost_models()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to register XGBoost models: {e}")
    
    try:
        register_ensemble_models()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to register ensemble models: {e}")


# Auto-initialize models when module is imported
initialize_models()


__all__ = [
    "BaseModel",
    "TaskType", 
    "ModelRegistry",
    "get_model_registry",
    "ModelFactory",
    "create_model",
    "initialize_models"
]