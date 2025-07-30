"""
Base model interface and task type definitions for NeuroLite.

Provides abstract base classes that define the unified model interface
and task type enumeration.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from ..core import get_logger
from ..data.detector import DataType


logger = get_logger(__name__)


class TaskType(Enum):
    """Enumeration of supported machine learning task types."""
    
    # Classification tasks
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    
    # Regression tasks
    REGRESSION = "regression"
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    
    # Computer vision tasks
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    
    # NLP tasks
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_GENERATION = "text_generation"
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    
    # Clustering tasks
    CLUSTERING = "clustering"
    
    # Time series tasks
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    
    # Auto detection
    AUTO = "auto"


@dataclass
class ModelCapabilities:
    """Defines the capabilities and requirements of a model."""
    
    supported_tasks: List[TaskType]
    supported_data_types: List[DataType]
    framework: str  # "pytorch", "tensorflow", "sklearn", etc.
    requires_gpu: bool = False
    min_samples: int = 1
    max_samples: Optional[int] = None
    supports_incremental_learning: bool = False
    supports_feature_importance: bool = False
    supports_probability_prediction: bool = False


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    
    name: str
    task_type: TaskType
    data_type: DataType
    framework: str
    training_time: float
    num_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    training_samples: Optional[int] = None
    validation_accuracy: Optional[float] = None
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Result of model prediction."""
    
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all models in NeuroLite.
    
    Defines the unified interface that all models must implement,
    regardless of the underlying framework (PyTorch, TensorFlow, scikit-learn, etc.).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model.
        
        Args:
            **kwargs: Model-specific configuration parameters
        """
        self.is_trained = False
        self.metadata: Optional[ModelMetadata] = None
        self._config = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with config: {kwargs}")
    
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """
        Get model capabilities and requirements.
        
        Returns:
            ModelCapabilities object describing what this model can do
        """
        pass
    
    @abstractmethod
    def fit(
        self, 
        X: Union[np.ndarray, List, Any], 
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features/inputs
            y: Training targets/labels
            validation_data: Optional validation data as (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """
        Make predictions on new data.
        
        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            PredictionResult containing predictions and metadata
        """
        pass
    
    def predict_proba(self, X: Union[np.ndarray, List, Any], **kwargs) -> np.ndarray:
        """
        Predict class probabilities (for classification tasks).
        
        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of class probabilities
            
        Raises:
            NotImplementedError: If model doesn't support probability prediction
        """
        if not self.capabilities.supports_probability_prediction:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support probability prediction"
            )
        
        result = self.predict(X, **kwargs)
        if result.probabilities is not None:
            return result.probabilities
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} predict method doesn't return probabilities"
            )
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (if supported).
        
        Returns:
            Array of feature importance scores, or None if not supported
        """
        if not self.capabilities.supports_feature_importance:
            return None
        
        # Default implementation - subclasses should override
        return None
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        return self._config.copy()
    
    def set_config(self, **kwargs) -> None:
        """
        Update model configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self._config.update(kwargs)
        logger.debug(f"Updated {self.__class__.__name__} config: {kwargs}")
    
    def validate_data(
        self, 
        X: Any, 
        y: Any, 
        task_type: TaskType,
        data_type: DataType
    ) -> None:
        """
        Validate that the provided data is compatible with this model.
        
        Args:
            X: Input features
            y: Target values
            task_type: Type of ML task
            data_type: Type of input data
            
        Raises:
            ValueError: If data is not compatible with model
        """
        # Check if model supports the task type
        if task_type not in self.capabilities.supported_tasks:
            raise ValueError(
                f"Model {self.__class__.__name__} does not support task type {task_type.value}. "
                f"Supported tasks: {[t.value for t in self.capabilities.supported_tasks]}"
            )
        
        # Check if model supports the data type
        if data_type not in self.capabilities.supported_data_types:
            raise ValueError(
                f"Model {self.__class__.__name__} does not support data type {data_type.value}. "
                f"Supported data types: {[d.value for d in self.capabilities.supported_data_types]}"
            )
        
        # Check sample count requirements
        if hasattr(X, '__len__'):
            num_samples = len(X)
            if num_samples < self.capabilities.min_samples:
                raise ValueError(
                    f"Model {self.__class__.__name__} requires at least {self.capabilities.min_samples} "
                    f"samples, but got {num_samples}"
                )
            
            if (self.capabilities.max_samples is not None and 
                num_samples > self.capabilities.max_samples):
                raise ValueError(
                    f"Model {self.__class__.__name__} supports at most {self.capabilities.max_samples} "
                    f"samples, but got {num_samples}"
                )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}({status})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


class SklearnModelAdapter(BaseModel):
    """
    Adapter for scikit-learn models to conform to NeuroLite interface.
    
    This class wraps scikit-learn models and provides the unified interface
    expected by NeuroLite.
    """
    
    def __init__(self, sklearn_model: Any, **kwargs):
        """
        Initialize the sklearn adapter.
        
        Args:
            sklearn_model: Instance of a scikit-learn model
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.sklearn_model = sklearn_model
        self._sklearn_model_name = sklearn_model.__class__.__name__
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get capabilities based on the sklearn model type."""
        from sklearn.base import ClassifierMixin, RegressorMixin
        
        supported_tasks = []
        supports_proba = False
        supports_feature_importance = False
        
        if isinstance(self.sklearn_model, ClassifierMixin):
            supported_tasks.extend([
                TaskType.CLASSIFICATION,
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION
            ])
            supports_proba = hasattr(self.sklearn_model, 'predict_proba')
        
        if isinstance(self.sklearn_model, RegressorMixin):
            supported_tasks.extend([
                TaskType.REGRESSION,
                TaskType.LINEAR_REGRESSION
            ])
        
        # Check for feature importance based on model type
        model_class_name = self.sklearn_model.__class__.__name__
        
        # Models that have feature_importances_ after training
        tree_based_models = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor',
            'AdaBoostClassifier', 'AdaBoostRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor'
        ]
        
        # Models that have coef_ after training
        linear_models = [
            'LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso',
            'ElasticNet', 'SGDClassifier', 'SGDRegressor', 'SVC', 'SVR'
        ]
        
        if model_class_name in tree_based_models or model_class_name in linear_models:
            supports_feature_importance = True
        
        return ModelCapabilities(
            supported_tasks=supported_tasks,
            supported_data_types=[DataType.TABULAR],  # Most sklearn models work with tabular data
            framework="sklearn",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=supports_proba,
            supports_feature_importance=supports_feature_importance
        )
    
    def fit(
        self, 
        X: Union[np.ndarray, List, Any], 
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> 'SklearnModelAdapter':
        """Train the sklearn model."""
        logger.debug(f"Training {self._sklearn_model_name} with {len(X)} samples")
        
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Train the model
        self.sklearn_model.fit(X, y)
        self.is_trained = True
        
        logger.debug(f"Successfully trained {self._sklearn_model_name}")
        return self
    
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """Make predictions using the sklearn model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Make predictions
        predictions = self.sklearn_model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.sklearn_model, 'predict_proba'):
            try:
                probabilities = self.sklearn_model.predict_proba(X)
            except Exception as e:
                logger.debug(f"Could not get probabilities: {e}")
        
        # Get feature importance if available
        feature_importance = self.get_feature_importance()
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=feature_importance
        )
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from sklearn model."""
        if hasattr(self.sklearn_model, 'feature_importances_'):
            return self.sklearn_model.feature_importances_
        elif hasattr(self.sklearn_model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            coef = self.sklearn_model.coef_
            if coef.ndim > 1:
                # Multi-class case - take mean of absolute coefficients
                return np.mean(np.abs(coef), axis=0)
            else:
                return np.abs(coef)
        return None
    
    def save(self, path: str) -> None:
        """Save the sklearn model using joblib."""
        import joblib
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'sklearn_model': self.sklearn_model,
            'is_trained': self.is_trained,
            'config': self._config,
            'metadata': self.metadata
        }
        
        joblib.dump(model_data, path)
        logger.debug(f"Saved {self._sklearn_model_name} to {path}")
    
    def load(self, path: str) -> 'SklearnModelAdapter':
        """Load the sklearn model using joblib."""
        import joblib
        
        model_data = joblib.load(path)
        
        self.sklearn_model = model_data['sklearn_model']
        self.is_trained = model_data['is_trained']
        self._config = model_data['config']
        self.metadata = model_data.get('metadata')
        self._sklearn_model_name = self.sklearn_model.__class__.__name__
        
        logger.debug(f"Loaded {self._sklearn_model_name} from {path}")
        return self