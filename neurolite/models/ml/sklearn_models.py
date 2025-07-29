"""
Scikit-learn model implementations for NeuroLite.

Provides pre-configured sklearn models with intelligent defaults.
"""

from typing import Any, Dict, Optional
import warnings

from ...core import get_logger, safe_import, DependencyError
from ...data.detector import DataType
from ..base import BaseModel, TaskType, ModelCapabilities, SklearnModelAdapter
from ..registry import register_model


logger = get_logger(__name__)


def create_random_forest_classifier(**kwargs) -> BaseModel:
    """Create a Random Forest classifier with intelligent defaults."""
    sklearn = safe_import('sklearn.ensemble', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Random Forest", "pip install scikit-learn")
    
    # Set intelligent defaults
    defaults = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1  # Use all available cores
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.RandomForestClassifier(**defaults)
    return SklearnModelAdapter(sklearn_model)


def create_random_forest_regressor(**kwargs) -> BaseModel:
    """Create a Random Forest regressor with intelligent defaults."""
    sklearn = safe_import('sklearn.ensemble', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Random Forest", "pip install scikit-learn")
    
    defaults = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.RandomForestRegressor(**defaults)
    return SklearnModelAdapter(sklearn_model)


def create_logistic_regression(**kwargs) -> BaseModel:
    """Create a Logistic Regression model with intelligent defaults."""
    sklearn = safe_import('sklearn.linear_model', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Logistic Regression", "pip install scikit-learn")
    
    defaults = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.LogisticRegression(**defaults)
    return SklearnModelAdapter(sklearn_model)


def create_linear_regression(**kwargs) -> BaseModel:
    """Create a Linear Regression model."""
    sklearn = safe_import('sklearn.linear_model', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Linear Regression", "pip install scikit-learn")
    
    sklearn_model = sklearn.LinearRegression(**kwargs)
    return SklearnModelAdapter(sklearn_model)


def create_svm_classifier(**kwargs) -> BaseModel:
    """Create an SVM classifier with intelligent defaults."""
    sklearn = safe_import('sklearn.svm', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "SVM", "pip install scikit-learn")
    
    defaults = {
        'kernel': 'rbf',
        'random_state': 42,
        'probability': True  # Enable probability prediction
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.SVC(**defaults)
    return SklearnModelAdapter(sklearn_model)


def create_gradient_boosting_classifier(**kwargs) -> BaseModel:
    """Create a Gradient Boosting classifier with intelligent defaults."""
    sklearn = safe_import('sklearn.ensemble', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Gradient Boosting", "pip install scikit-learn")
    
    defaults = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.GradientBoostingClassifier(**defaults)
    return SklearnModelAdapter(sklearn_model)


def create_gradient_boosting_regressor(**kwargs) -> BaseModel:
    """Create a Gradient Boosting regressor with intelligent defaults."""
    sklearn = safe_import('sklearn.ensemble', 'scikit-learn')
    if sklearn is None:
        raise DependencyError("scikit-learn", "Gradient Boosting", "pip install scikit-learn")
    
    defaults = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    defaults.update(kwargs)
    
    sklearn_model = sklearn.GradientBoostingRegressor(**defaults)
    return SklearnModelAdapter(sklearn_model)


def register_sklearn_models() -> None:
    """Register all sklearn models in the global registry."""
    logger.debug("Registering sklearn models")
    
    # Check if sklearn is available
    sklearn = safe_import('sklearn', 'scikit-learn')
    if sklearn is None:
        logger.warning("scikit-learn not available, skipping sklearn model registration")
        return
    
    try:
        # Classification models
        register_model(
            name="random_forest",
            model_class=SklearnModelAdapter,
            factory_function=create_random_forest_classifier,
            priority=8,
            description="Random Forest classifier with intelligent defaults",
            tags=["ensemble", "classification", "feature_importance"]
        )
        
        register_model(
            name="logistic_regression",
            model_class=SklearnModelAdapter,
            factory_function=create_logistic_regression,
            priority=6,
            description="Logistic Regression with L2 regularization",
            tags=["linear", "classification", "interpretable"]
        )
        
        register_model(
            name="svm",
            model_class=SklearnModelAdapter,
            factory_function=create_svm_classifier,
            priority=7,
            description="Support Vector Machine with RBF kernel",
            tags=["svm", "classification", "kernel"]
        )
        
        register_model(
            name="gradient_boosting",
            model_class=SklearnModelAdapter,
            factory_function=create_gradient_boosting_classifier,
            priority=7,
            description="Gradient Boosting classifier",
            tags=["boosting", "classification", "feature_importance"]
        )
        
        # Regression models
        register_model(
            name="random_forest_regressor",
            model_class=SklearnModelAdapter,
            factory_function=create_random_forest_regressor,
            priority=8,
            description="Random Forest regressor with intelligent defaults",
            tags=["ensemble", "regression", "feature_importance"]
        )
        
        register_model(
            name="linear_regression",
            model_class=SklearnModelAdapter,
            factory_function=create_linear_regression,
            priority=5,
            description="Ordinary Least Squares Linear Regression",
            tags=["linear", "regression", "interpretable"]
        )
        
        register_model(
            name="gradient_boosting_regressor",
            model_class=SklearnModelAdapter,
            factory_function=create_gradient_boosting_regressor,
            priority=7,
            description="Gradient Boosting regressor",
            tags=["boosting", "regression", "feature_importance"]
        )
        
        logger.debug("Successfully registered sklearn models")
        
    except Exception as e:
        logger.error(f"Failed to register sklearn models: {e}")
        raise