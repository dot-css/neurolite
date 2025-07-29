"""
XGBoost model implementations for NeuroLite.

Provides pre-configured XGBoost models with intelligent defaults.
"""

from typing import Any, Dict, Optional

from ...core import get_logger, safe_import, DependencyError
from ...data.detector import DataType
from ..base import BaseModel, TaskType, ModelCapabilities, SklearnModelAdapter
from ..registry import register_model


logger = get_logger(__name__)


def create_xgboost_classifier(**kwargs) -> BaseModel:
    """Create an XGBoost classifier with intelligent defaults."""
    xgb = safe_import('xgboost', 'xgboost')
    if xgb is None:
        raise DependencyError("xgboost", "XGBoost", "pip install xgboost")
    
    # Set intelligent defaults
    defaults = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    defaults.update(kwargs)
    
    xgb_model = xgb.XGBClassifier(**defaults)
    return SklearnModelAdapter(xgb_model)


def create_xgboost_regressor(**kwargs) -> BaseModel:
    """Create an XGBoost regressor with intelligent defaults."""
    xgb = safe_import('xgboost', 'xgboost')
    if xgb is None:
        raise DependencyError("xgboost", "XGBoost", "pip install xgboost")
    
    defaults = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse'
    }
    defaults.update(kwargs)
    
    xgb_model = xgb.XGBRegressor(**defaults)
    return SklearnModelAdapter(xgb_model)


def register_xgboost_models() -> None:
    """Register all XGBoost models in the global registry."""
    logger.debug("Registering XGBoost models")
    
    # Check if XGBoost is available
    xgb = safe_import('xgboost', 'xgboost')
    if xgb is None:
        logger.warning("XGBoost not available, skipping XGBoost model registration")
        return
    
    try:
        # Classification model
        register_model(
            name="xgboost",
            model_class=SklearnModelAdapter,
            factory_function=create_xgboost_classifier,
            priority=9,  # High priority for auto-selection
            description="XGBoost classifier with intelligent defaults and early stopping",
            tags=["boosting", "classification", "feature_importance", "fast"]
        )
        
        # Regression model
        register_model(
            name="xgboost_regressor",
            model_class=SklearnModelAdapter,
            factory_function=create_xgboost_regressor,
            priority=9,  # High priority for auto-selection
            description="XGBoost regressor with intelligent defaults and early stopping",
            tags=["boosting", "regression", "feature_importance", "fast"]
        )
        
        logger.debug("Successfully registered XGBoost models")
        
    except Exception as e:
        logger.error(f"Failed to register XGBoost models: {e}")
        raise