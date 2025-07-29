"""
Model factory for NeuroLite.

Provides factory functions to create model instances based on specifications
and automatic model selection logic.
"""

from typing import Any, Dict, Optional, Union
import importlib

from ..core import get_logger, ModelError, ModelNotFoundError, ModelCompatibilityError, DependencyError
from ..data.detector import DataType
from .base import BaseModel, TaskType
from .registry import get_model_registry, auto_select_model


logger = get_logger(__name__)


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Handles model creation, automatic selection, and dependency management.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.registry = get_model_registry()
        logger.debug("Initialized ModelFactory")
    
    def create_model(
        self,
        model_name: str,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance by name.
        
        Args:
            model_name: Name of the model to create
            task_type: Type of ML task (for validation)
            data_type: Type of input data (for validation)
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model is not registered
            ModelError: If model creation fails
        """
        logger.debug(f"Creating model '{model_name}' for task={task_type}, data={data_type}")
        
        # Handle auto selection
        if model_name.lower() == "auto":
            if task_type is None or data_type is None:
                raise ModelError(
                    "task_type and data_type must be specified for automatic model selection"
                )
            
            model_name = self.auto_select(
                task_type=task_type,
                data_type=data_type,
                **kwargs
            )
            logger.debug(f"Auto-selected model: {model_name}")
        
        # Get model instance from registry
        try:
            model = self.registry.get_model(model_name, **kwargs)
        except (ModelNotFoundError, ModelCompatibilityError):
            # Re-raise these specific exceptions as-is
            raise
        except Exception as e:
            raise ModelError(f"Failed to create model '{model_name}': {e}")
        
        # Validate compatibility if task and data types are provided
        if task_type is not None and data_type is not None:
            try:
                # Create dummy data for validation
                dummy_X = self._create_dummy_data(data_type)
                dummy_y = self._create_dummy_targets(task_type)
                model.validate_data(dummy_X, dummy_y, task_type, data_type)
            except Exception as e:
                logger.warning(f"Model validation failed: {e}")
                # Don't raise here as validation might fail due to dummy data
        
        return model
    
    def auto_select(
        self,
        task_type: TaskType,
        data_type: DataType,
        num_samples: Optional[int] = None,
        prefer_framework: Optional[str] = None,
        require_gpu: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Automatically select the best model for given criteria.
        
        Args:
            task_type: Type of ML task
            data_type: Type of input data
            num_samples: Number of training samples
            prefer_framework: Preferred framework
            require_gpu: Whether GPU is required/available
            **kwargs: Additional selection criteria
            
        Returns:
            Name of the selected model
        """
        return auto_select_model(
            task_type=task_type,
            data_type=data_type,
            num_samples=num_samples,
            prefer_framework=prefer_framework,
            require_gpu=require_gpu
        )
    
    def create_sklearn_model(
        self,
        sklearn_model_name: str,
        **kwargs
    ) -> BaseModel:
        """
        Create a scikit-learn model wrapped in NeuroLite interface.
        
        Args:
            sklearn_model_name: Name of the sklearn model
            **kwargs: Arguments to pass to sklearn model constructor
            
        Returns:
            Wrapped sklearn model
            
        Raises:
            DependencyError: If scikit-learn is not available
            ModelError: If sklearn model creation fails
        """
        try:
            from sklearn import ensemble, linear_model, svm, tree, naive_bayes
            from .base import SklearnModelAdapter
        except ImportError:
            raise DependencyError(
                "Missing dependency 'scikit-learn' required for sklearn models",
                suggestions=[
                    "Install the missing dependency: pip install scikit-learn",
                    "Install NeuroLite with the appropriate extras: pip install neurolite[sklearn]",
                    "Check the installation documentation for complete setup instructions"
                ]
            )
        
        # Map of sklearn model names to classes
        sklearn_models = {
            # Ensemble methods
            'random_forest': ensemble.RandomForestClassifier,
            'random_forest_regressor': ensemble.RandomForestRegressor,
            'gradient_boosting': ensemble.GradientBoostingClassifier,
            'gradient_boosting_regressor': ensemble.GradientBoostingRegressor,
            'extra_trees': ensemble.ExtraTreesClassifier,
            'extra_trees_regressor': ensemble.ExtraTreesRegressor,
            'ada_boost': ensemble.AdaBoostClassifier,
            'ada_boost_regressor': ensemble.AdaBoostRegressor,
            
            # Linear models
            'logistic_regression': linear_model.LogisticRegression,
            'linear_regression': linear_model.LinearRegression,
            'ridge': linear_model.Ridge,
            'lasso': linear_model.Lasso,
            'elastic_net': linear_model.ElasticNet,
            'sgd_classifier': linear_model.SGDClassifier,
            'sgd_regressor': linear_model.SGDRegressor,
            
            # SVM
            'svc': svm.SVC,
            'svr': svm.SVR,
            'linear_svc': svm.LinearSVC,
            'linear_svr': svm.LinearSVR,
            
            # Tree-based
            'decision_tree': tree.DecisionTreeClassifier,
            'decision_tree_regressor': tree.DecisionTreeRegressor,
            
            # Naive Bayes
            'gaussian_nb': naive_bayes.GaussianNB,
            'multinomial_nb': naive_bayes.MultinomialNB,
            'bernoulli_nb': naive_bayes.BernoulliNB,
        }
        
        if sklearn_model_name not in sklearn_models:
            available_models = list(sklearn_models.keys())
            raise ModelNotFoundError(sklearn_model_name, available_models)
        
        try:
            # Create sklearn model instance
            sklearn_model_class = sklearn_models[sklearn_model_name]
            sklearn_model = sklearn_model_class(**kwargs)
            
            # Wrap in adapter
            model = SklearnModelAdapter(sklearn_model)
            
            logger.debug(f"Created sklearn model '{sklearn_model_name}'")
            return model
            
        except Exception as e:
            raise ModelError(f"Failed to create sklearn model '{sklearn_model_name}': {e}")
    
    def create_xgboost_model(
        self,
        task_type: TaskType,
        **kwargs
    ) -> BaseModel:
        """
        Create an XGBoost model for the specified task.
        
        Args:
            task_type: Type of ML task
            **kwargs: Arguments to pass to XGBoost constructor
            
        Returns:
            XGBoost model wrapped in NeuroLite interface
            
        Raises:
            DependencyError: If XGBoost is not available
            ModelError: If XGBoost model creation fails
        """
        try:
            import xgboost as xgb
            from .base import SklearnModelAdapter
        except ImportError:
            raise DependencyError(
                "Missing dependency 'xgboost' required for XGBoost models",
                suggestions=[
                    "Install the missing dependency: pip install xgboost",
                    "Install NeuroLite with the appropriate extras: pip install neurolite[xgboost]",
                    "Check the installation documentation for complete setup instructions"
                ]
            )
        
        try:
            # Select appropriate XGBoost model based on task type
            if task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION, 
                           TaskType.MULTICLASS_CLASSIFICATION]:
                xgb_model = xgb.XGBClassifier(**kwargs)
            elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION]:
                xgb_model = xgb.XGBRegressor(**kwargs)
            else:
                raise ModelError(f"XGBoost does not support task type: {task_type.value}")
            
            # Wrap in adapter
            model = SklearnModelAdapter(xgb_model)
            
            logger.debug(f"Created XGBoost model for task '{task_type.value}'")
            return model
            
        except Exception as e:
            raise ModelError(f"Failed to create XGBoost model: {e}")
    
    def _create_dummy_data(self, data_type: DataType) -> Any:
        """Create dummy data for validation purposes."""
        import numpy as np
        
        if data_type == DataType.TABULAR:
            return np.array([[1, 2, 3], [4, 5, 6]])
        elif data_type == DataType.IMAGE:
            return np.random.rand(2, 224, 224, 3)
        elif data_type == DataType.TEXT:
            return ["sample text 1", "sample text 2"]
        elif data_type == DataType.AUDIO:
            return np.random.rand(2, 16000)  # 1 second at 16kHz
        elif data_type == DataType.VIDEO:
            return np.random.rand(2, 10, 224, 224, 3)  # 10 frames
        else:
            return np.array([[1, 2], [3, 4]])
    
    def _create_dummy_targets(self, task_type: TaskType) -> Any:
        """Create dummy targets for validation purposes."""
        import numpy as np
        
        if task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION,
                        TaskType.TEXT_CLASSIFICATION]:
            return np.array([0, 1])
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION]:
            return np.array([1.5, 2.7])
        else:
            return np.array([0, 1])  # Default to classification-like targets
    
    def get_available_models(
        self,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        framework: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models.
        
        Args:
            task_type: Filter by task type
            data_type: Filter by data type
            framework: Filter by framework
            
        Returns:
            Dictionary mapping model names to their information
        """
        model_names = self.registry.list_models(task_type, data_type, framework)
        
        models_info = {}
        for name in model_names:
            try:
                registration = self.registry.get_model_info(name)
                models_info[name] = {
                    'description': registration.description,
                    'framework': registration.capabilities.framework,
                    'supported_tasks': [t.value for t in registration.capabilities.supported_tasks],
                    'supported_data_types': [d.value for d in registration.capabilities.supported_data_types],
                    'requires_gpu': registration.capabilities.requires_gpu,
                    'priority': registration.priority,
                    'tags': registration.tags
                }
            except Exception as e:
                logger.warning(f"Failed to get info for model '{name}': {e}")
        
        return models_info


# Global factory instance
_global_factory = ModelFactory()


def create_model(
    model_name: str,
    task_type: Optional[TaskType] = None,
    data_type: Optional[DataType] = None,
    **kwargs
) -> BaseModel:
    """
    Create a model instance using the global factory.
    
    Args:
        model_name: Name of the model to create
        task_type: Type of ML task
        data_type: Type of input data
        **kwargs: Additional arguments
        
    Returns:
        Model instance
    """
    return _global_factory.create_model(
        model_name=model_name,
        task_type=task_type,
        data_type=data_type,
        **kwargs
    )


def get_model_factory() -> ModelFactory:
    """
    Get the global model factory instance.
    
    Returns:
        Global ModelFactory instance
    """
    return _global_factory