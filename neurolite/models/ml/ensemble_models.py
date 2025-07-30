"""
Ensemble model implementations for NeuroLite.

Provides ensemble models that combine multiple algorithms for improved performance.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from unittest.mock import Mock

from ...core import get_logger, safe_import, DependencyError, ModelError
from ...data.detector import DataType
from ..base import BaseModel, TaskType, ModelCapabilities, PredictionResult, ModelMetadata
from ..registry import register_model


logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    
    models: List[str]  # List of model names to include in ensemble
    voting: str = "soft"  # "hard" or "soft" voting for classification
    weights: Optional[List[float]] = None  # Weights for each model
    n_jobs: int = -1  # Number of parallel jobs


class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.
    
    For classification, supports both hard and soft voting.
    For regression, uses averaging of predictions.
    """
    
    def __init__(self, models: List[BaseModel], voting: str = "soft", 
                 weights: Optional[List[float]] = None, **kwargs):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of base models to ensemble
            voting: Voting strategy ("hard" or "soft" for classification)
            weights: Optional weights for each model
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.models = models
        self.voting = voting
        self.weights = weights
        self._task_type = None
        self._data_type = None
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        logger.debug(f"Initialized VotingEnsemble with {len(models)} models")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get ensemble capabilities based on constituent models."""
        if not self.models:
            return ModelCapabilities(
                supported_tasks=[],
                supported_data_types=[],
                framework="ensemble"
            )
        
        # Find intersection of supported tasks and data types
        supported_tasks = set(self.models[0].capabilities.supported_tasks)
        supported_data_types = set(self.models[0].capabilities.supported_data_types)
        
        for model in self.models[1:]:
            supported_tasks &= set(model.capabilities.supported_tasks)
            supported_data_types &= set(model.capabilities.supported_data_types)
        
        # Check if any model supports probability prediction
        supports_proba = any(
            model.capabilities.supports_probability_prediction 
            for model in self.models
        )
        
        # Check if any model supports feature importance
        supports_feature_importance = any(
            model.capabilities.supports_feature_importance 
            for model in self.models
        )
        
        return ModelCapabilities(
            supported_tasks=list(supported_tasks),
            supported_data_types=list(supported_data_types),
            framework="ensemble",
            requires_gpu=False,
            min_samples=max(model.capabilities.min_samples for model in self.models),
            supports_probability_prediction=supports_proba,
            supports_feature_importance=supports_feature_importance
        )
    
    def fit(
        self, 
        X: Union[np.ndarray, List, Any], 
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> 'VotingEnsemble':
        """Train all models in the ensemble."""
        logger.debug(f"Training ensemble with {len(self.models)} models")
        
        # Train each model
        for i, model in enumerate(self.models):
            logger.debug(f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(X, y, validation_data=validation_data, **kwargs)
        
        self.is_trained = True
        logger.debug("Ensemble training completed")
        return self
    
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from all models
        predictions_list = []
        probabilities_list = []
        
        for model in self.models:
            result = model.predict(X, **kwargs)
            predictions_list.append(result.predictions)
            
            if result.probabilities is not None:
                probabilities_list.append(result.probabilities)
        
        # Combine predictions
        if self._is_classification_task():
            if self.voting == "hard" or not probabilities_list:
                # Hard voting - majority vote
                predictions = self._hard_voting(predictions_list)
                probabilities = None
            else:
                # Soft voting - average probabilities
                predictions, probabilities = self._soft_voting(probabilities_list)
        else:
            # Regression - average predictions
            predictions = self._average_predictions(predictions_list)
            probabilities = None
        
        # Get ensemble feature importance
        feature_importance = self.get_feature_importance()
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=feature_importance
        )
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get ensemble feature importance by averaging individual importances."""
        importances = []
        
        for model in self.models:
            importance = model.get_feature_importance()
            if importance is not None:
                importances.append(importance)
        
        if not importances:
            return None
        
        # Average importances with optional weights
        if self.weights is not None:
            # Use only weights for models that have feature importance
            valid_weights = self.weights[:len(importances)]
            weighted_importances = [
                imp * weight for imp, weight in zip(importances, valid_weights)
            ]
            return np.mean(weighted_importances, axis=0)
        else:
            return np.mean(importances, axis=0)
    
    def _is_classification_task(self) -> bool:
        """Check if this is a classification task."""
        # Simple heuristic - if any model supports probability prediction, assume classification
        return any(
            model.capabilities.supports_probability_prediction 
            for model in self.models
        )
    
    def _hard_voting(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Perform hard voting (majority vote)."""
        predictions_array = np.array(predictions_list)
        
        if self.weights is not None:
            # Weighted voting
            weighted_votes = np.zeros_like(predictions_array[0])
            for i, (preds, weight) in enumerate(zip(predictions_array, self.weights)):
                weighted_votes += preds * weight
            return np.round(weighted_votes).astype(int)
        else:
            # Simple majority vote
            from scipy import stats
            return stats.mode(predictions_array, axis=0)[0].flatten()
    
    def _soft_voting(self, probabilities_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform soft voting (average probabilities)."""
        if self.weights is not None:
            # Weighted average
            weighted_probs = np.zeros_like(probabilities_list[0])
            total_weight = sum(self.weights)
            for probs, weight in zip(probabilities_list, self.weights):
                weighted_probs += probs * (weight / total_weight)
        else:
            # Simple average
            weighted_probs = np.mean(probabilities_list, axis=0)
        
        # Get predictions from probabilities
        predictions = np.argmax(weighted_probs, axis=1)
        
        return predictions, weighted_probs
    
    def _average_predictions(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Average predictions for regression."""
        if self.weights is not None:
            # Weighted average
            weighted_preds = np.zeros_like(predictions_list[0])
            total_weight = sum(self.weights)
            for preds, weight in zip(predictions_list, self.weights):
                weighted_preds += preds * (weight / total_weight)
            return weighted_preds
        else:
            # Simple average
            return np.mean(predictions_list, axis=0)
    
    def save(self, path: str) -> None:
        """Save the ensemble model."""
        import joblib
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save ensemble configuration and individual models
        ensemble_data = {
            'models': self.models,
            'voting': self.voting,
            'weights': self.weights,
            'is_trained': self.is_trained,
            'config': self._config,
            'metadata': self.metadata
        }
        
        joblib.dump(ensemble_data, path)
        logger.debug(f"Saved VotingEnsemble to {path}")
    
    def load(self, path: str) -> 'VotingEnsemble':
        """Load the ensemble model."""
        import joblib
        
        ensemble_data = joblib.load(path)
        
        self.models = ensemble_data['models']
        self.voting = ensemble_data['voting']
        self.weights = ensemble_data['weights']
        self.is_trained = ensemble_data['is_trained']
        self._config = ensemble_data['config']
        self.metadata = ensemble_data.get('metadata')
        
        logger.debug(f"Loaded VotingEnsemble from {path}")
        return self


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble that uses a meta-learner to combine base model predictions.
    """
    
    def __init__(self, base_models: List[BaseModel], meta_model: BaseModel, 
                 cv_folds: int = 5, **kwargs):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner model
            cv_folds: Number of cross-validation folds for generating meta-features
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self._meta_features = None
        
        logger.debug(f"Initialized StackingEnsemble with {len(base_models)} base models")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get stacking ensemble capabilities."""
        if not self.base_models:
            return ModelCapabilities(
                supported_tasks=[],
                supported_data_types=[],
                framework="ensemble"
            )
        
        # Find intersection of supported tasks and data types
        supported_tasks = set(self.base_models[0].capabilities.supported_tasks)
        supported_data_types = set(self.base_models[0].capabilities.supported_data_types)
        
        for model in self.base_models[1:]:
            supported_tasks &= set(model.capabilities.supported_tasks)
            supported_data_types &= set(model.capabilities.supported_data_types)
        
        # Also consider meta-model capabilities
        supported_tasks &= set(self.meta_model.capabilities.supported_tasks)
        supported_data_types &= set(self.meta_model.capabilities.supported_data_types)
        
        return ModelCapabilities(
            supported_tasks=list(supported_tasks),
            supported_data_types=list(supported_data_types),
            framework="ensemble",
            requires_gpu=False,
            min_samples=max(
                max(model.capabilities.min_samples for model in self.base_models),
                self.meta_model.capabilities.min_samples
            ),
            supports_probability_prediction=self.meta_model.capabilities.supports_probability_prediction,
            supports_feature_importance=self.meta_model.capabilities.supports_feature_importance
        )
    
    def fit(
        self, 
        X: Union[np.ndarray, List, Any], 
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> 'StackingEnsemble':
        """Train the stacking ensemble."""
        logger.debug("Training stacking ensemble")
        
        # Convert to numpy arrays
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Generate meta-features using cross-validation
        self._meta_features = self._generate_meta_features(X, y)
        
        # Train base models on full dataset
        for i, model in enumerate(self.base_models):
            logger.debug(f"Training base model {i+1}/{len(self.base_models)}")
            model.fit(X, y, validation_data=validation_data, **kwargs)
        
        # Train meta-model on meta-features
        logger.debug("Training meta-model")
        self.meta_model.fit(self._meta_features, y, **kwargs)
        
        self.is_trained = True
        logger.debug("Stacking ensemble training completed")
        return self
    
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """Make stacking ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Stacking ensemble must be trained before making predictions")
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features_predict(X)
        
        # Get final prediction from meta-model
        result = self.meta_model.predict(meta_features, **kwargs)
        
        return result
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from meta-model."""
        return self.meta_model.get_feature_importance()
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Determine if this is classification (for stratified CV)
        is_classification = self._is_classification_task()
        
        if is_classification:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Initialize meta-features array
        if is_classification and hasattr(self.base_models[0], 'predict_proba'):
            # For classification with probability prediction
            n_classes = len(np.unique(y))
            meta_features = np.zeros((n_samples, n_models * n_classes))
        else:
            # For regression or classification without probabilities
            meta_features = np.zeros((n_samples, n_models))
        
        # Generate meta-features using cross-validation
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, model in enumerate(self.base_models):
                # Create a copy of the model for this fold
                model_copy = self._copy_model(model)
                model_copy.fit(X_train, y_train)
                
                # Generate predictions for validation set
                if is_classification and hasattr(model_copy, 'predict_proba'):
                    try:
                        probs = model_copy.predict_proba(X_val)
                        start_idx = i * n_classes
                        end_idx = (i + 1) * n_classes
                        meta_features[val_idx, start_idx:end_idx] = probs
                    except:
                        # Fallback to regular predictions
                        preds = model_copy.predict(X_val).predictions
                        meta_features[val_idx, i] = preds
                else:
                    preds = model_copy.predict(X_val).predictions
                    meta_features[val_idx, i] = preds
        
        return meta_features
    
    def _generate_meta_features_predict(self, X: Union[np.ndarray, List, Any]) -> np.ndarray:
        """Generate meta-features for prediction."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Determine feature dimensions based on training
        if self._meta_features is not None:
            n_features = self._meta_features.shape[1]
        else:
            n_features = n_models
        
        meta_features = np.zeros((n_samples, n_features))
        
        # Generate features from each base model
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba') and n_features > n_models:
                # Use probabilities if available and expected
                try:
                    probs = model.predict_proba(X)
                    n_classes = probs.shape[1]
                    start_idx = i * n_classes
                    end_idx = (i + 1) * n_classes
                    meta_features[:, start_idx:end_idx] = probs
                except:
                    # Fallback to regular predictions
                    preds = model.predict(X).predictions
                    meta_features[:, i] = preds
            else:
                preds = model.predict(X).predictions
                meta_features[:, i] = preds
        
        return meta_features
    
    def _copy_model(self, model: BaseModel) -> BaseModel:
        """Create a copy of a model for cross-validation."""
        # This is a simplified copy - in practice, you might need more sophisticated copying
        model_class = model.__class__
        config = model.get_config()
        return model_class(**config)
    
    def _is_classification_task(self) -> bool:
        """Check if this is a classification task."""
        return any(
            model.capabilities.supports_probability_prediction 
            for model in self.base_models
        )
    
    def save(self, path: str) -> None:
        """Save the stacking ensemble."""
        import joblib
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        ensemble_data = {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'cv_folds': self.cv_folds,
            'meta_features': self._meta_features,
            'is_trained': self.is_trained,
            'config': self._config,
            'metadata': self.metadata
        }
        
        joblib.dump(ensemble_data, path)
        logger.debug(f"Saved StackingEnsemble to {path}")
    
    def load(self, path: str) -> 'StackingEnsemble':
        """Load the stacking ensemble."""
        import joblib
        
        ensemble_data = joblib.load(path)
        
        self.base_models = ensemble_data['base_models']
        self.meta_model = ensemble_data['meta_model']
        self.cv_folds = ensemble_data['cv_folds']
        self._meta_features = ensemble_data['meta_features']
        self.is_trained = ensemble_data['is_trained']
        self._config = ensemble_data['config']
        self.metadata = ensemble_data.get('metadata')
        
        logger.debug(f"Loaded StackingEnsemble from {path}")
        return self


def create_voting_classifier(**kwargs) -> BaseModel:
    """Create a voting classifier ensemble with default models."""
    models = kwargs.pop('models', None)
    voting = kwargs.pop('voting', 'soft')
    weights = kwargs.pop('weights', None)
    
    # If no models provided, create default ones
    if models is None:
        try:
            from ..factory import get_model_factory
            factory = get_model_factory()
            
            # Try to create default models
            default_models = []
            for model_name in ["random_forest", "xgboost", "svm"]:
                try:
                    model = factory.create_model(model_name)
                    default_models.append(model)
                except Exception as e:
                    logger.warning(f"Could not create model {model_name} for ensemble: {e}")
            
            if not default_models:
                raise ModelError("No models available for voting classifier ensemble")
            
            models = default_models
        except Exception as e:
            raise ModelError(f"Failed to create voting classifier: {e}")
    
    return VotingEnsemble(models=models, voting=voting, weights=weights, **kwargs)


def create_voting_regressor(**kwargs) -> BaseModel:
    """Create a voting regressor ensemble with default models."""
    models = kwargs.pop('models', None)
    weights = kwargs.pop('weights', None)
    
    # If no models provided, create default ones
    if models is None:
        try:
            from ..factory import get_model_factory
            factory = get_model_factory()
            
            # Try to create default models
            default_models = []
            for model_name in ["random_forest_regressor", "xgboost_regressor", "linear_regression"]:
                try:
                    model = factory.create_model(model_name)
                    default_models.append(model)
                except Exception as e:
                    logger.warning(f"Could not create model {model_name} for ensemble: {e}")
            
            if not default_models:
                raise ModelError("No models available for voting regressor ensemble")
            
            models = default_models
        except Exception as e:
            raise ModelError(f"Failed to create voting regressor: {e}")
    
    return VotingEnsemble(models=models, voting="soft", weights=weights, **kwargs)


def create_stacking_classifier(**kwargs) -> BaseModel:
    """Create a stacking classifier ensemble with default models."""
    base_models = kwargs.pop('base_models', None)
    meta_model = kwargs.pop('meta_model', None)
    cv_folds = kwargs.pop('cv_folds', 5)
    
    # If no models provided, create default ones
    if base_models is None or meta_model is None:
        try:
            from ..factory import get_model_factory
            factory = get_model_factory()
            
            if base_models is None:
                # Try to create default base models
                default_base_models = []
                for model_name in ["random_forest", "xgboost", "svm"]:
                    try:
                        model = factory.create_model(model_name)
                        default_base_models.append(model)
                    except Exception as e:
                        logger.warning(f"Could not create model {model_name} for ensemble: {e}")
                
                if not default_base_models:
                    raise ModelError("No base models available for stacking classifier ensemble")
                
                base_models = default_base_models
            
            if meta_model is None:
                try:
                    meta_model = factory.create_model("logistic_regression")
                except Exception as e:
                    logger.warning(f"Could not create meta-model for ensemble: {e}")
                    raise ModelError("No meta-model available for stacking classifier ensemble")
        
        except Exception as e:
            raise ModelError(f"Failed to create stacking classifier: {e}")
    
    return StackingEnsemble(
        base_models=base_models, 
        meta_model=meta_model, 
        cv_folds=cv_folds, 
        **kwargs
    )


def create_stacking_regressor(**kwargs) -> BaseModel:
    """Create a stacking regressor ensemble with default models."""
    base_models = kwargs.pop('base_models', None)
    meta_model = kwargs.pop('meta_model', None)
    cv_folds = kwargs.pop('cv_folds', 5)
    
    # If no models provided, create default ones
    if base_models is None or meta_model is None:
        try:
            from ..factory import get_model_factory
            factory = get_model_factory()
            
            if base_models is None:
                # Try to create default base models
                default_base_models = []
                for model_name in ["random_forest_regressor", "xgboost_regressor", "gradient_boosting_regressor"]:
                    try:
                        model = factory.create_model(model_name)
                        default_base_models.append(model)
                    except Exception as e:
                        logger.warning(f"Could not create model {model_name} for ensemble: {e}")
                
                if not default_base_models:
                    raise ModelError("No base models available for stacking regressor ensemble")
                
                base_models = default_base_models
            
            if meta_model is None:
                try:
                    meta_model = factory.create_model("linear_regression")
                except Exception as e:
                    logger.warning(f"Could not create meta-model for ensemble: {e}")
                    raise ModelError("No meta-model available for stacking regressor ensemble")
        
        except Exception as e:
            raise ModelError(f"Failed to create stacking regressor: {e}")
    
    return StackingEnsemble(
        base_models=base_models, 
        meta_model=meta_model, 
        cv_folds=cv_folds, 
        **kwargs
    )


def register_ensemble_models() -> None:
    """Register all ensemble models in the global registry."""
    logger.debug("Registering ensemble models")
    
    # Check if required dependencies are available
    sklearn = safe_import('sklearn', 'ensemble models')
    if sklearn is None:
        logger.warning("scikit-learn not available, skipping ensemble model registration")
        return
    
    try:
        # Create simple ensemble instances for capability detection
        # Use mock models to avoid dependency issues during registration
        mock_models = []
        for i in range(2):  # Minimal ensemble
            mock_model = Mock()
            mock_model.capabilities = ModelCapabilities(
                supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
                supported_data_types=[DataType.TABULAR],
                framework="mock",
                requires_gpu=False,
                min_samples=1,
                supports_probability_prediction=True,
                supports_feature_importance=True
            )
            mock_models.append(mock_model)
        
        # Voting ensemble models
        register_model(
            name="voting_classifier",
            model_class=VotingEnsemble,
            factory_function=lambda **kwargs: VotingEnsemble(models=mock_models, **kwargs),
            priority=10,  # High priority for ensemble methods
            description="Voting classifier ensemble combining multiple algorithms",
            tags=["ensemble", "classification", "voting", "robust"]
        )
        
        register_model(
            name="voting_regressor",
            model_class=VotingEnsemble,
            factory_function=lambda **kwargs: VotingEnsemble(models=mock_models, **kwargs),
            priority=10,
            description="Voting regressor ensemble combining multiple algorithms",
            tags=["ensemble", "regression", "voting", "robust"]
        )
        
        # Stacking ensemble models
        meta_model = Mock()
        meta_model.capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
        
        register_model(
            name="stacking_classifier",
            model_class=StackingEnsemble,
            factory_function=lambda **kwargs: StackingEnsemble(base_models=mock_models, meta_model=meta_model, **kwargs),
            priority=11,  # Highest priority for advanced ensemble
            description="Stacking classifier ensemble with meta-learner",
            tags=["ensemble", "classification", "stacking", "advanced"]
        )
        
        register_model(
            name="stacking_regressor",
            model_class=StackingEnsemble,
            factory_function=lambda **kwargs: StackingEnsemble(base_models=mock_models, meta_model=meta_model, **kwargs),
            priority=11,
            description="Stacking regressor ensemble with meta-learner",
            tags=["ensemble", "regression", "stacking", "advanced"]
        )
        
        logger.debug("Successfully registered ensemble models")
        
    except Exception as e:
        logger.error(f"Failed to register ensemble models: {e}")
        raise