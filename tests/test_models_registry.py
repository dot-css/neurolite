"""
Unit tests for the model registry system.

Tests model registration, discovery, and automatic selection functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, PredictionResult
from neurolite.models.registry import ModelRegistry, ModelRegistration
from neurolite.data.detector import DataType
from neurolite.core.exceptions import ModelError, ModelNotFoundError, ModelCompatibilityError


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
            supports_probability_prediction=True,
            supports_feature_importance=True
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


class MockGPUModel(BaseModel):
    """Mock GPU model for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities = ModelCapabilities(
            supported_tasks=[TaskType.IMAGE_CLASSIFICATION],
            supported_data_types=[DataType.IMAGE],
            framework="pytorch",
            requires_gpu=True,
            min_samples=10
        )
    
    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities
    
    def fit(self, X, y, validation_data=None, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        return PredictionResult(predictions=np.array([0] * len(X)))
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        return self


class TestModelRegistry:
    """Test cases for ModelRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
    
    def test_register_model_success(self):
        """Test successful model registration."""
        self.registry.register_model(
            name="test_model",
            model_class=MockModel,
            priority=5,
            description="Test model",
            tags=["test", "mock"]
        )
        
        assert "test_model" in self.registry._models
        registration = self.registry._models["test_model"]
        assert registration.name == "test_model"
        assert registration.model_class == MockModel
        assert registration.priority == 5
        assert registration.description == "Test model"
        assert registration.tags == ["test", "mock"]
    
    def test_register_duplicate_model_fails(self):
        """Test that registering duplicate model names fails."""
        self.registry.register_model("test_model", MockModel)
        
        with pytest.raises(ModelError, match="already registered"):
            self.registry.register_model("test_model", MockModel)
    
    def test_register_invalid_model_class_fails(self):
        """Test that registering invalid model class fails."""
        class InvalidModel:
            pass
        
        with pytest.raises(ModelError, match="must inherit from BaseModel"):
            self.registry.register_model("invalid", InvalidModel)
    
    def test_unregister_model_success(self):
        """Test successful model unregistration."""
        self.registry.register_model("test_model", MockModel)
        assert "test_model" in self.registry._models
        
        self.registry.unregister_model("test_model")
        assert "test_model" not in self.registry._models
    
    def test_unregister_nonexistent_model_fails(self):
        """Test that unregistering nonexistent model fails."""
        with pytest.raises(ModelNotFoundError):
            self.registry.unregister_model("nonexistent")
    
    def test_get_model_success(self):
        """Test successful model retrieval."""
        self.registry.register_model("test_model", MockModel)
        
        model = self.registry.get_model("test_model")
        assert isinstance(model, MockModel)
    
    def test_get_model_with_kwargs(self):
        """Test model retrieval with constructor arguments."""
        self.registry.register_model("test_model", MockModel)
        
        model = self.registry.get_model("test_model", custom_param="value")
        assert model.get_config()["custom_param"] == "value"
    
    def test_get_nonexistent_model_fails(self):
        """Test that getting nonexistent model fails."""
        with pytest.raises(ModelNotFoundError):
            self.registry.get_model("nonexistent")
    
    def test_list_models_all(self):
        """Test listing all models."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        models = self.registry.list_models()
        assert sorted(models) == ["model1", "model2"]
    
    def test_list_models_by_task_type(self):
        """Test listing models filtered by task type."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        models = self.registry.list_models(task_type=TaskType.CLASSIFICATION)
        assert models == ["model1"]
        
        models = self.registry.list_models(task_type=TaskType.IMAGE_CLASSIFICATION)
        assert models == ["model2"]
    
    def test_list_models_by_data_type(self):
        """Test listing models filtered by data type."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        models = self.registry.list_models(data_type=DataType.TABULAR)
        assert models == ["model1"]
        
        models = self.registry.list_models(data_type=DataType.IMAGE)
        assert models == ["model2"]
    
    def test_list_models_by_framework(self):
        """Test listing models filtered by framework."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        models = self.registry.list_models(framework="mock")
        assert models == ["model1"]
        
        models = self.registry.list_models(framework="pytorch")
        assert models == ["model2"]
    
    def test_get_model_info(self):
        """Test getting model information."""
        self.registry.register_model(
            "test_model", 
            MockModel, 
            description="Test model",
            tags=["test"]
        )
        
        info = self.registry.get_model_info("test_model")
        assert info.name == "test_model"
        assert info.description == "Test model"
        assert info.tags == ["test"]
    
    def test_get_model_info_nonexistent_fails(self):
        """Test that getting info for nonexistent model fails."""
        with pytest.raises(ModelNotFoundError):
            self.registry.get_model_info("nonexistent")
    
    def test_auto_select_model_success(self):
        """Test successful automatic model selection."""
        self.registry.register_model("model1", MockModel, priority=5)
        self.registry.register_model("model2", MockModel, priority=8)
        
        selected = self.registry.auto_select_model(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        # Should select model2 due to higher priority
        assert selected == "model2"
    
    def test_auto_select_model_with_sample_count(self):
        """Test automatic model selection with sample count filtering."""
        self.registry.register_model("model1", MockModel)  # min_samples=1
        self.registry.register_model("model2", MockGPUModel)  # min_samples=10
        
        # With few samples, should select model1
        selected = self.registry.auto_select_model(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR,
            num_samples=5
        )
        assert selected == "model1"
    
    def test_auto_select_model_with_gpu_preference(self):
        """Test automatic model selection with GPU preferences."""
        self.registry.register_model("cpu_model", MockModel)
        self.registry.register_model("gpu_model", MockGPUModel)
        
        # Prefer GPU model when GPU is available
        selected = self.registry.auto_select_model(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            data_type=DataType.IMAGE,
            require_gpu=True
        )
        assert selected == "gpu_model"
        
        # Should not select GPU model when GPU is not available
        with pytest.raises(ModelCompatibilityError):
            self.registry.auto_select_model(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                data_type=DataType.IMAGE,
                require_gpu=False
            )
    
    def test_auto_select_model_no_compatible_fails(self):
        """Test that auto-selection fails when no compatible models exist."""
        self.registry.register_model("model1", MockModel)  # Only supports TABULAR
        
        with pytest.raises(ModelCompatibilityError):
            self.registry.auto_select_model(
                task_type=TaskType.CLASSIFICATION,
                data_type=DataType.IMAGE  # Not supported
            )
    
    def test_get_compatible_models(self):
        """Test getting compatible models."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        compatible = self.registry.get_compatible_models(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        
        assert len(compatible) == 1
        assert compatible[0][0] == "model1"
    
    def test_get_statistics(self):
        """Test getting registry statistics."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        stats = self.registry.get_statistics()
        
        assert stats["total_models"] == 2
        assert "mock" in stats["by_framework"]
        assert "pytorch" in stats["by_framework"]
        assert TaskType.CLASSIFICATION.value in stats["by_task_type"]
        assert DataType.TABULAR.value in stats["by_data_type"]
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        self.registry.register_model("model1", MockModel)
        self.registry.register_model("model2", MockGPUModel)
        
        assert len(self.registry._models) == 2
        
        self.registry.clear()
        
        assert len(self.registry._models) == 0
        assert len(self.registry._task_index) == 0
        assert len(self.registry._data_type_index) == 0
        assert len(self.registry._framework_index) == 0
    
    def test_factory_function_usage(self):
        """Test using factory function for model creation."""
        def custom_factory(**kwargs):
            model = MockModel(**kwargs)
            model.custom_attribute = "factory_created"
            return model
        
        self.registry.register_model(
            "factory_model",
            MockModel,
            factory_function=custom_factory
        )
        
        model = self.registry.get_model("factory_model")
        assert hasattr(model, "custom_attribute")
        assert model.custom_attribute == "factory_created"


class TestModelRegistryGlobalFunctions:
    """Test cases for global registry functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry
        from neurolite.models.registry import _global_registry
        _global_registry.clear()
    
    def test_register_model_global(self):
        """Test registering model in global registry."""
        from neurolite.models.registry import register_model, list_models
        
        register_model("test_model", MockModel)
        
        models = list_models()
        assert "test_model" in models
    
    def test_get_model_global(self):
        """Test getting model from global registry."""
        from neurolite.models.registry import register_model, get_model
        
        register_model("test_model", MockModel)
        
        model = get_model("test_model")
        assert isinstance(model, MockModel)
    
    def test_auto_select_model_global(self):
        """Test auto-selecting model from global registry."""
        from neurolite.models.registry import register_model, auto_select_model
        
        register_model("test_model", MockModel)
        
        selected = auto_select_model(
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR
        )
        assert selected == "test_model"


class TestModelRegistryIntegration:
    """Integration tests for model registry with real sklearn models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", minversion=None),
        reason="scikit-learn not available"
    )
    def test_sklearn_model_registration(self):
        """Test registering actual sklearn models."""
        from sklearn.ensemble import RandomForestClassifier
        from neurolite.models.base import SklearnModelAdapter
        
        def create_rf(**kwargs):
            sklearn_model = RandomForestClassifier(n_estimators=10, random_state=42, **kwargs)
            return SklearnModelAdapter(sklearn_model)
        
        self.registry.register_model(
            "random_forest",
            SklearnModelAdapter,
            factory_function=create_rf,
            priority=8
        )
        
        # Test model creation
        model = self.registry.get_model("random_forest")
        assert isinstance(model, SklearnModelAdapter)
        
        # Test training with dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        model.fit(X, y)
        assert model.is_trained
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions.predictions) == len(X)
    
    def test_model_validation(self):
        """Test model data validation."""
        self.registry.register_model("test_model", MockModel)
        model = self.registry.get_model("test_model")
        
        # Valid data should not raise
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.validate_data(X, y, TaskType.CLASSIFICATION, DataType.TABULAR)
        
        # Invalid task type should raise
        with pytest.raises(ValueError, match="does not support task type"):
            model.validate_data(X, y, TaskType.REGRESSION, DataType.TABULAR)
        
        # Invalid data type should raise
        with pytest.raises(ValueError, match="does not support data type"):
            model.validate_data(X, y, TaskType.CLASSIFICATION, DataType.IMAGE)
    
    def test_model_capabilities_inheritance(self):
        """Test that model capabilities are properly inherited."""
        class CustomModel(MockModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._capabilities = ModelCapabilities(
                    supported_tasks=[TaskType.REGRESSION],
                    supported_data_types=[DataType.TABULAR, DataType.TEXT],
                    framework="custom",
                    requires_gpu=True,
                    min_samples=5
                )
        
        self.registry.register_model("custom_model", CustomModel)
        
        # Test filtering by capabilities
        models = self.registry.list_models(task_type=TaskType.REGRESSION)
        assert "custom_model" in models
        
        models = self.registry.list_models(data_type=DataType.TEXT)
        assert "custom_model" in models
        
        models = self.registry.list_models(framework="custom")
        assert "custom_model" in models