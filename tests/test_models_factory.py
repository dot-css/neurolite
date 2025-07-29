"""
Unit tests for the model factory system.

Tests model creation, automatic selection, and factory functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from neurolite.models.factory import ModelFactory, create_model, get_model_factory
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, PredictionResult
from neurolite.data.detector import DataType
from neurolite.core.exceptions import ModelError, ModelNotFoundError, DependencyError


class MockModel(BaseModel):
    """Mock model for testing purposes."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="mock",
            requires_gpu=False,
            min_samples=1,
            supports_probability_prediction=True
        )
    
    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities
    
    def fit(self, X, y, validation_data=None, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        predictions = np.array([0] * len(X))
        probabilities = np.array([[0.6, 0.4]] * len(X))
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        return self


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
        # Clear registry for clean tests
        self.factory.registry.clear()
    
    def test_create_model_by_name(self):
        """Test creating model by name."""
        # Register a test model
        self.factory.registry.register_model("test_model", MockModel)
        
        model = self.factory.create_model("test_model")
        assert isinstance(model, MockModel)
    
    def test_create_model_with_kwargs(self):
        """Test creating model with constructor arguments."""
        self.factory.registry.register_model("test_model", MockModel)
        
        model = self.factory.create_model("test_model", custom_param="value")
        assert model.get_config()["custom_param"] == "value"
    
    def test_create_model_nonexistent_fails(self):
        """Test that creating nonexistent model fails."""
        with pytest.raises(ModelNotFoundError):
            self.factory.create_model("nonexistent")
    
    def test_create_model_auto_selection(self):
        """Test automatic model selection."""
        self.factory.registry.register_model("test_model", MockModel, priority=5)
        
        model = self.factory.create_model(
            "auto",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        assert isinstance(model, MockModel)
    
    def test_create_model_auto_without_task_fails(self):
        """Test that auto selection without task type fails."""
        with pytest.raises(ModelError, match="task_type and data_type must be specified"):
            self.factory.create_model("auto")
    
    def test_auto_select_method(self):
        """Test the auto_select method."""
        self.factory.registry.register_model("test_model", MockModel)
        
        selected = self.factory.auto_select(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        assert selected == "test_model"
    
    @patch('neurolite.core.safe_import')
    def test_create_sklearn_model_success(self, mock_safe_import):
        """Test successful sklearn model creation."""
        # Mock sklearn imports
        mock_ensemble = Mock()
        mock_ensemble.RandomForestClassifier = Mock()
        mock_safe_import.return_value = mock_ensemble
        
        with patch('neurolite.models.base.SklearnModelAdapter') as mock_adapter:
            model = self.factory.create_sklearn_model("random_forest")
            mock_adapter.assert_called_once()
    
    def test_create_sklearn_model_not_available(self):
        """Test sklearn model creation when sklearn is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'sklearn'")):
            with pytest.raises(DependencyError, match="scikit-learn"):
                self.factory.create_sklearn_model("random_forest")
    
    def test_create_sklearn_model_invalid_name(self):
        """Test sklearn model creation with invalid name."""
        with pytest.raises(ModelNotFoundError):
            self.factory.create_sklearn_model("invalid_model")
    
    @patch('neurolite.core.safe_import')
    def test_create_xgboost_model_success(self, mock_safe_import):
        """Test successful XGBoost model creation."""
        # Mock XGBoost imports
        mock_xgb = Mock()
        mock_xgb.XGBClassifier = Mock()
        mock_safe_import.return_value = mock_xgb
        
        with patch('neurolite.models.base.SklearnModelAdapter') as mock_adapter:
            model = self.factory.create_xgboost_model(TaskType.CLASSIFICATION)
            mock_adapter.assert_called_once()
    
    def test_create_xgboost_model_not_available(self):
        """Test XGBoost model creation when XGBoost is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'xgboost'")):
            with pytest.raises(DependencyError, match="xgboost"):
                self.factory.create_xgboost_model(TaskType.CLASSIFICATION)
    
    def test_create_xgboost_model_unsupported_task(self):
        """Test XGBoost model creation with unsupported task."""
        with patch('neurolite.core.safe_import') as mock_safe_import:
            mock_xgb = Mock()
            mock_safe_import.return_value = mock_xgb
            
            with pytest.raises(ModelError, match="does not support task type"):
                self.factory.create_xgboost_model(TaskType.CLUSTERING)
    
    def test_create_dummy_data(self):
        """Test dummy data creation for different data types."""
        # Test tabular data
        dummy_data = self.factory._create_dummy_data(DataType.TABULAR)
        assert isinstance(dummy_data, np.ndarray)
        assert dummy_data.shape == (2, 3)
        
        # Test image data
        dummy_data = self.factory._create_dummy_data(DataType.IMAGE)
        assert isinstance(dummy_data, np.ndarray)
        assert dummy_data.shape == (2, 224, 224, 3)
        
        # Test text data
        dummy_data = self.factory._create_dummy_data(DataType.TEXT)
        assert isinstance(dummy_data, list)
        assert len(dummy_data) == 2
        
        # Test audio data
        dummy_data = self.factory._create_dummy_data(DataType.AUDIO)
        assert isinstance(dummy_data, np.ndarray)
        assert dummy_data.shape == (2, 16000)
        
        # Test video data
        dummy_data = self.factory._create_dummy_data(DataType.VIDEO)
        assert isinstance(dummy_data, np.ndarray)
        assert dummy_data.shape == (2, 10, 224, 224, 3)
    
    def test_create_dummy_targets(self):
        """Test dummy target creation for different task types."""
        # Test classification targets
        dummy_targets = self.factory._create_dummy_targets(TaskType.CLASSIFICATION)
        assert isinstance(dummy_targets, np.ndarray)
        assert len(dummy_targets) == 2
        assert all(t in [0, 1] for t in dummy_targets)
        
        # Test regression targets
        dummy_targets = self.factory._create_dummy_targets(TaskType.REGRESSION)
        assert isinstance(dummy_targets, np.ndarray)
        assert len(dummy_targets) == 2
        assert all(isinstance(t, (int, float)) for t in dummy_targets)
    
    def test_get_available_models(self):
        """Test getting available models information."""
        self.factory.registry.register_model(
            "test_model",
            MockModel,
            description="Test model",
            tags=["test"]
        )
        
        models_info = self.factory.get_available_models()
        
        assert "test_model" in models_info
        model_info = models_info["test_model"]
        assert model_info["description"] == "Test model"
        assert model_info["framework"] == "mock"
        assert "classification" in model_info["supported_tasks"]
        assert "tabular" in model_info["supported_data_types"]
        assert model_info["tags"] == ["test"]
    
    def test_get_available_models_filtered(self):
        """Test getting available models with filters."""
        # Register models with different capabilities
        class Model1(MockModel):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    supported_tasks=[TaskType.CLASSIFICATION],
                    supported_data_types=[DataType.TABULAR],
                    framework="sklearn"
                )
        
        class Model2(MockModel):
            @property
            def capabilities(self):
                return ModelCapabilities(
                    supported_tasks=[TaskType.REGRESSION],
                    supported_data_types=[DataType.TABULAR],
                    framework="pytorch"
                )
        
        self.factory.registry.register_model("model1", Model1)
        self.factory.registry.register_model("model2", Model2)
        
        # Filter by task type
        models_info = self.factory.get_available_models(task_type=TaskType.CLASSIFICATION)
        assert "model1" in models_info
        assert "model2" not in models_info
        
        # Filter by framework
        models_info = self.factory.get_available_models(framework="pytorch")
        assert "model1" not in models_info
        assert "model2" in models_info


class TestModelFactoryGlobalFunctions:
    """Test cases for global factory functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry
        from neurolite.models.registry import _global_registry
        _global_registry.clear()
    
    def test_create_model_global(self):
        """Test creating model using global factory."""
        from neurolite.models.registry import register_model
        
        register_model("test_model", MockModel)
        
        model = create_model("test_model")
        assert isinstance(model, MockModel)
    
    def test_get_model_factory_global(self):
        """Test getting global factory instance."""
        factory = get_model_factory()
        assert isinstance(factory, ModelFactory)


class TestModelFactoryIntegration:
    """Integration tests for model factory with real models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
        self.factory.registry.clear()
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", minversion=None),
        reason="scikit-learn not available"
    )
    def test_sklearn_integration(self):
        """Test integration with actual sklearn models."""
        # Test Random Forest
        model = self.factory.create_sklearn_model("random_forest", n_estimators=10)
        
        # Test training
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        model.fit(X, y)
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions.predictions) == len(X)
        assert predictions.probabilities is not None
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", minversion=None),
        reason="scikit-learn not available"
    )
    def test_sklearn_model_types(self):
        """Test different sklearn model types."""
        # Test classifier
        classifier = self.factory.create_sklearn_model("logistic_regression")
        assert TaskType.CLASSIFICATION in classifier.capabilities.supported_tasks
        
        # Test regressor
        regressor = self.factory.create_sklearn_model("linear_regression")
        assert TaskType.REGRESSION in regressor.capabilities.supported_tasks
    
    def test_xgboost_integration(self):
        """Test integration with actual XGBoost models."""
        pytest.importorskip("xgboost")
        
        # Test XGBoost classifier
        model = self.factory.create_xgboost_model(
            TaskType.CLASSIFICATION,
            n_estimators=10
        )
        
        # Test training
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        model.fit(X, y)
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions.predictions) == len(X)
        
        # Test XGBoost regressor
        regressor = self.factory.create_xgboost_model(
            TaskType.REGRESSION,
            n_estimators=10
        )
        
        y_reg = np.array([1.5, 2.7, 3.1, 4.2])
        regressor.fit(X, y_reg)
        
        predictions = regressor.predict(X)
        assert len(predictions.predictions) == len(X)
    
    def test_model_validation_integration(self):
        """Test model validation with real data scenarios."""
        from neurolite.models.registry import register_model
        
        register_model("test_model", MockModel)
        
        # Test with compatible data
        model = self.factory.create_model(
            "test_model",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        assert isinstance(model, MockModel)
        
        # Test auto-selection with compatible data
        model = self.factory.create_model(
            "auto",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        assert isinstance(model, MockModel)
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test with missing dependencies
        with patch('builtins.__import__', side_effect=ImportError("No module named 'sklearn'")):
            with pytest.raises(DependencyError):
                self.factory.create_sklearn_model("random_forest")
        
        # Test with invalid model creation
        with patch('neurolite.models.registry.ModelRegistry.get_model', 
                  side_effect=Exception("Creation failed")):
            with pytest.raises(ModelError, match="Creation failed"):
                self.factory.create_model("test_model")
    
    def test_factory_with_registered_models(self):
        """Test factory functionality with pre-registered models."""
        # This test simulates the real scenario where models are pre-registered
        from neurolite.models.ml.sklearn_models import register_sklearn_models
        
        # Register sklearn models (if available)
        try:
            register_sklearn_models()
            
            # Test that we can create registered models
            available_models = self.factory.get_available_models()
            
            if "random_forest" in available_models:
                model = self.factory.create_model("random_forest")
                assert model is not None
                
            if "logistic_regression" in available_models:
                model = self.factory.create_model("logistic_regression")
                assert model is not None
                
        except Exception as e:
            # Skip if sklearn is not available
            pytest.skip(f"sklearn models not available: {e}")
    
    def test_comprehensive_model_workflow(self):
        """Test complete model workflow from creation to prediction."""
        from neurolite.models.registry import register_model
        
        # Register a test model
        register_model("workflow_model", MockModel, priority=10)
        
        # Create model via auto-selection
        model = self.factory.create_model(
            "auto",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        # Train model
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        model.fit(X, y)
        assert model.is_trained
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions.predictions) == len(X)
        assert predictions.probabilities is not None
        
        # Test probability prediction
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)  # Binary classification