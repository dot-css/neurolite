"""
Traditional machine learning models for NeuroLite.

Provides scikit-learn and XGBoost model implementations.
"""

from .sklearn_models import register_sklearn_models
from .xgboost_models import register_xgboost_models

__all__ = [
    "register_sklearn_models",
    "register_xgboost_models"
]