"""
Unit tests for the NeuroLite plugin system.

Tests plugin registration, discovery, loading, validation, and integration
with the core NeuroLite systems.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Any, Union, Tuple
import numpy as np

from neurolite.core.plugins import (
    PluginRegistry, PluginLoader, ModelPlugin, PreprocessorPlugin,
    PluginMetadata, PluginError, PluginValidationError, PluginLoadError,
    get_plugin_registry, register_model_plugin, register_preprocessor_plugin
)
from neurolite.core.plugin_integration import PluginIntegration, get_plugin_integration
from neurolite.core.plugin_templates import PluginTemplateGenerator, create_plugin_template
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, PredictionResult
from neurolite.data.preprocessor import BasePreprocessor, PreprocessingConfig
from neurolite.data.loader import Dataset, DatasetInfo
from neurolite.data.detector import DataType


class TestModelPlugin(BaseModel):
    """Test model for plugin testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_data = None
    
    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION, TaskType.REGRESSION],
            supported_data_types=[DataType.TABULAR],
            framework="test",
            requires_gpu=False,
            supports_probability_prediction=True,
            supports_feature_importance=True
        )
    
    def fit(self, X, y, validation_data=None, **kwargs):
        self.test_data = (X, y)
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        if not self.is_trained:
            raise ValueError("Model not trained")
        predictions = np.zeros(len(X))
        probabilities = np.random.rand(len(X), 2)
        return PredictionResult(predictions=predictions, probabilities=probabilities)
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        return self


class TestPreprocessor(BasePreprocessor):
    """Test preprocessor for plugin testing."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        super().__init__(config)
        self.test_stats = {}
    
    def fit(self, dataset: Dataset):
        self.test_stats = {'fitted': True}
        self.is_fitted = True
        return self
    
    def transform(self, dataset: Dataset):
        if not self.is_fitted:
            raise ValueError("Not fitted")
        return dataset


class ValidTestModelPlugin(ModelPlugin):
    """Valid test model plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test-model",
            version="1.0.0",
            description="Test model plugin",
            author="Test Author",
            plugin_type="model",
            supported_tasks=["classification", "regression"],
            supported_data_types=["tabular"],
            framework="test"
        )
    
    @property
    def model_class(self):
        return TestModelPlugin


class ValidTestPreprocessorPlugin(PreprocessorPlugin):
    """Valid test preprocessor plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test-preprocessor",
            version="1.0.0",
            description="Test preprocessor plugin",
            author="Test Author",
            plugin_type="preprocessor",
            supported_data_types=["tabular"]
        )
    
    @property
    def preprocessor_class(self):
        return TestPreprocessor


class InvalidModelPlugin(ModelPlugin):
    """Invalid model plugin for testing validation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="",  # Invalid: empty name
            version="1.0.0",
            description="Invalid plugin",
            author="Test Author",
            plugin_type="model",
            supported_tasks=["invalid_task"],  # Invalid task
            supported_data_types=["tabular"],
            framework="test"
        )
    
    @property
    def model_class(self):
        return str  # Invalid: not a BaseModel subclass


class TestPluginRegistry:
    """Test cases for PluginRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = PluginRegistry()
    
    def test_register_valid_model_plugin(self):
        """Test registering a valid model plugin."""
        plugin = ValidTestModelPlugin()
        
        self.registry.register_model_plugin(plugin)
        
        assert "test-model" in self.registry._model_plugins
        plugin_info = self.registry._model_plugins["test-model"]
        assert plugin_info.metadata.name == "test-model"
        assert plugin_info.model_class == TestModelPlugin
    
    def test_register_valid_preprocessor_plugin(self):
        """Test registering a valid preprocessor plugin."""
        plugin = ValidTestPreprocessorPlugin()
        
        self.registry.register_preprocessor_plugin(plugin)
        
        assert "test-preprocessor" in self.registry._preprocessor_plugins
        plugin_info = self.registry._preprocessor_plugins["test-preprocessor"]
        assert plugin_info.metadata.name == "test-preprocessor"
        assert plugin_info.preprocessor_class == TestPreprocessor
    
    def test_register_invalid_model_plugin(self):
        """Test registering an invalid model plugin."""
        plugin = InvalidModelPlugin()
        
        with pytest.raises(PluginValidationError) as exc_info:
            self.registry.register_model_plugin(plugin)
        
        assert "test-model" not in self.registry._model_plugins
        assert "Plugin name is required" in str(exc_info.value)
    
    def test_get_model_plugin(self):
        """Test getting a model plugin."""
        plugin = ValidTestModelPlugin()
        self.registry.register_model_plugin(plugin)
        
        plugin_info = self.registry.get_model_plugin("test-model")
        assert plugin_info.metadata.name == "test-model"
    
    def test_get_nonexistent_plugin(self):
        """Test getting a non-existent plugin."""
        with pytest.raises(PluginError) as exc_info:
            self.registry.get_model_plugin("nonexistent")
        
        assert "not found" in str(exc_info.value)
    
    def test_list_model_plugins(self):
        """Test listing model plugins."""
        plugin = ValidTestModelPlugin()
        self.registry.register_model_plugin(plugin)
        
        plugins = self.registry.list_model_plugins()
        assert "test-model" in plugins
        
        # Test filtering
        plugins = self.registry.list_model_plugins(task_type=TaskType.CLASSIFICATION)
        assert "test-model" in plugins
        
        plugins = self.registry.list_model_plugins(data_type=DataType.IMAGE)
        assert "test-model" not in plugins
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        plugin = ValidTestModelPlugin()
        self.registry.register_model_plugin(plugin)
        
        assert "test-model" in self.registry._model_plugins
        
        self.registry.unregister_plugin("test-model")
        
        assert "test-model" not in self.registry._model_plugins
    
    def test_get_statistics(self):
        """Test getting registry statistics."""
        plugin1 = ValidTestModelPlugin()
        plugin2 = ValidTestPreprocessorPlugin()
        
        self.registry.register_model_plugin(plugin1)
        self.registry.register_preprocessor_plugin(plugin2)
        
        stats = self.registry.get_statistics()
        
        assert stats['total_plugins'] == 2
        assert stats['model_plugins'] == 1
        assert stats['preprocessor_plugins'] == 1
    
    def test_clear(self):
        """Test clearing the registry."""
        plugin = ValidTestModelPlugin()
        self.registry.register_model_plugin(plugin)
        
        assert len(self.registry._model_plugins) > 0
        
        self.registry.clear()
        
        assert len(self.registry._model_plugins) == 0
        assert len(self.registry._preprocessor_plugins) == 0


class TestPluginLoader:
    """Test cases for PluginLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_plugin_from_file(self):
        """Test loading a plugin from a file."""
        # Create a test plugin file
        plugin_code = '''
from neurolite.core.plugins import ModelPlugin, PluginMetadata
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities
from neurolite.data.detector import DataType
import numpy as np

class TestFileModel(BaseModel):
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="test"
        )
    
    def fit(self, X, y, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        from neurolite.models.base import PredictionResult
        return PredictionResult(predictions=np.zeros(len(X)))
    
    def save(self, path): pass
    def load(self, path): return self

class TestFilePlugin(ModelPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-file-plugin",
            version="1.0.0",
            description="Test plugin from file",
            author="Test",
            plugin_type="model",
            supported_tasks=["classification"],
            supported_data_types=["tabular"],
            framework="test"
        )
    
    @property
    def model_class(self):
        return TestFileModel

plugin = TestFilePlugin()
'''
        
        plugin_file = Path(self.temp_dir) / "test_plugin.py"
        with open(plugin_file, 'w') as f:
            f.write(plugin_code)
        
        # Load the plugin
        self.loader.load_plugin_from_file(plugin_file)
        
        # Verify it was loaded
        assert "test-file-plugin" in self.registry._model_plugins
    
    def test_load_plugin_from_nonexistent_file(self):
        """Test loading from a non-existent file."""
        with pytest.raises(PluginLoadError) as exc_info:
            self.loader.load_plugin_from_file("nonexistent.py")
        
        assert "not found" in str(exc_info.value)
    
    def test_load_plugins_from_directory(self):
        """Test loading plugins from a directory."""
        # Create multiple plugin files
        for i in range(2):
            plugin_code = f'''
from neurolite.core.plugins import ModelPlugin, PluginMetadata
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities
from neurolite.data.detector import DataType
import numpy as np

class TestDirModel{i}(BaseModel):
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="test"
        )
    
    def fit(self, X, y, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        from neurolite.models.base import PredictionResult
        return PredictionResult(predictions=np.zeros(len(X)))
    
    def save(self, path): pass
    def load(self, path): return self

class TestDirPlugin{i}(ModelPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-dir-plugin-{i}",
            version="1.0.0",
            description="Test plugin {i}",
            author="Test",
            plugin_type="model",
            supported_tasks=["classification"],
            supported_data_types=["tabular"],
            framework="test"
        )
    
    @property
    def model_class(self):
        return TestDirModel{i}

plugin = TestDirPlugin{i}()
'''
            
            plugin_file = Path(self.temp_dir) / f"plugin_{i}.py"
            with open(plugin_file, 'w') as f:
                f.write(plugin_code)
        
        # Load plugins from directory
        self.loader.load_plugins_from_directory(self.temp_dir)
        
        # Verify plugins were loaded
        assert "test-dir-plugin-0" in self.registry._model_plugins
        assert "test-dir-plugin-1" in self.registry._model_plugins
    
    def test_load_plugins_from_config(self):
        """Test loading plugins from a configuration file."""
        # Create a plugin file
        plugin_code = '''
from neurolite.core.plugins import ModelPlugin, PluginMetadata
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities
from neurolite.data.detector import DataType
import numpy as np

class TestConfigModel(BaseModel):
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="test"
        )
    
    def fit(self, X, y, **kwargs):
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        from neurolite.models.base import PredictionResult
        return PredictionResult(predictions=np.zeros(len(X)))
    
    def save(self, path): pass
    def load(self, path): return self

class TestConfigPlugin(ModelPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-config-plugin",
            version="1.0.0",
            description="Test plugin from config",
            author="Test",
            plugin_type="model",
            supported_tasks=["classification"],
            supported_data_types=["tabular"],
            framework="test"
        )
    
    @property
    def model_class(self):
        return TestConfigModel

plugin = TestConfigPlugin()
'''
        
        plugin_file = Path(self.temp_dir) / "config_plugin.py"
        with open(plugin_file, 'w') as f:
            f.write(plugin_code)
        
        # Create config file
        config = f'''
plugins:
  - type: file
    path: "{str(plugin_file).replace(chr(92), '/')}"
'''
        
        config_file = Path(self.temp_dir) / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config)
        
        # Load from config
        self.loader.load_plugin_from_config(config_file)
        
        # Verify plugin was loaded
        assert "test-config-plugin" in self.registry._model_plugins


class TestPluginIntegration:
    """Test cases for PluginIntegration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.integration = PluginIntegration()
        # Clear registries
        self.integration.plugin_registry.clear()
        self.integration.model_registry.clear()
    
    def test_integrate_model_plugins(self):
        """Test integrating model plugins with model registry."""
        # Register a plugin
        plugin = ValidTestModelPlugin()
        self.integration.plugin_registry.register_model_plugin(plugin)
        
        # Integrate
        self.integration.integrate_model_plugins()
        
        # Verify integration
        models = self.integration.model_registry.list_models()
        assert "test-model" in models
    
    def test_get_integration_status(self):
        """Test getting integration status."""
        # Register plugins
        model_plugin = ValidTestModelPlugin()
        preprocessor_plugin = ValidTestPreprocessorPlugin()
        
        self.integration.plugin_registry.register_model_plugin(model_plugin)
        self.integration.plugin_registry.register_preprocessor_plugin(preprocessor_plugin)
        
        # Integrate
        self.integration.integrate_all_plugins()
        
        # Get status
        status = self.integration.get_integration_status()
        
        assert status['total_plugins'] == 2
        assert status['integrated_models'] == 1
        assert "test-model" in status['model_plugins']


class TestPluginTemplateGenerator:
    """Test cases for PluginTemplateGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_model_plugin_template(self):
        """Test generating a model plugin template."""
        generator = PluginTemplateGenerator()
        
        template = generator.generate_model_plugin_template(
            plugin_name="test-model",
            author="Test Author",
            description="Test model plugin",
            framework="sklearn",
            output_dir=self.temp_dir
        )
        
        # Check that template was generated
        assert "class TestModelModel(BaseModel)" in template
        assert "class TestModelPlugin(ModelPlugin)" in template
        
        # Check that file was created
        output_file = Path(self.temp_dir) / "test_model_plugin.py"
        assert output_file.exists()
    
    def test_generate_preprocessor_plugin_template(self):
        """Test generating a preprocessor plugin template."""
        generator = PluginTemplateGenerator()
        
        template = generator.generate_preprocessor_plugin_template(
            plugin_name="test-preprocessor",
            author="Test Author",
            description="Test preprocessor plugin",
            data_type="tabular",
            output_dir=self.temp_dir
        )
        
        # Check that template was generated
        assert "class TestPreprocessorPreprocessor(BasePreprocessor)" in template
        assert "class TestPreprocessorPlugin(PreprocessorPlugin)" in template
        
        # Check that file was created
        output_file = Path(self.temp_dir) / "test_preprocessor_preprocessor_plugin.py"
        assert output_file.exists()
    
    def test_generate_plugin_config_template(self):
        """Test generating a plugin configuration template."""
        generator = PluginTemplateGenerator()
        
        config = generator.generate_plugin_config_template(self.temp_dir)
        
        # Check content
        assert "plugins:" in config
        assert "type: file" in config
        assert "type: directory" in config
        
        # Check that file was created
        config_file = Path(self.temp_dir) / "plugin_config.yaml"
        assert config_file.exists()
    
    def test_generate_plugin_documentation(self):
        """Test generating plugin documentation."""
        generator = PluginTemplateGenerator()
        
        docs = generator.generate_plugin_documentation(self.temp_dir)
        
        # Check content
        assert "# NeuroLite Plugin Development Guide" in docs
        assert "Model Plugins" in docs
        assert "Preprocessor Plugins" in docs
        
        # Check that file was created
        docs_file = Path(self.temp_dir) / "PLUGIN_DEVELOPMENT.md"
        assert docs_file.exists()
    
    def test_create_plugin_template(self):
        """Test creating a complete plugin template."""
        create_plugin_template(
            plugin_type="model",
            plugin_name="test-complete",
            author="Test Author",
            description="Complete test plugin",
            output_dir=self.temp_dir,
            framework="sklearn"
        )
        
        # Check that all files were created
        plugin_file = Path(self.temp_dir) / "test_complete_plugin.py"
        config_file = Path(self.temp_dir) / "plugin_config.yaml"
        docs_file = Path(self.temp_dir) / "PLUGIN_DEVELOPMENT.md"
        
        assert plugin_file.exists()
        assert config_file.exists()
        assert docs_file.exists()


class TestGlobalPluginFunctions:
    """Test cases for global plugin functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry
        get_plugin_registry().clear()
    
    def test_register_model_plugin_global(self):
        """Test registering a model plugin using global function."""
        plugin = ValidTestModelPlugin()
        
        register_model_plugin(plugin)
        
        registry = get_plugin_registry()
        assert "test-model" in registry._model_plugins
    
    def test_register_preprocessor_plugin_global(self):
        """Test registering a preprocessor plugin using global function."""
        plugin = ValidTestPreprocessorPlugin()
        
        register_preprocessor_plugin(plugin)
        
        registry = get_plugin_registry()
        assert "test-preprocessor" in registry._preprocessor_plugins


class TestPluginValidation:
    """Test cases for plugin validation."""
    
    def test_model_plugin_validation_success(self):
        """Test successful model plugin validation."""
        plugin = ValidTestModelPlugin()
        errors = plugin.validate()
        assert len(errors) == 0
    
    def test_model_plugin_validation_failure(self):
        """Test model plugin validation failure."""
        plugin = InvalidModelPlugin()
        errors = plugin.validate()
        assert len(errors) > 0
        assert any("Plugin name is required" in error for error in errors)
    
    def test_preprocessor_plugin_validation_success(self):
        """Test successful preprocessor plugin validation."""
        plugin = ValidTestPreprocessorPlugin()
        errors = plugin.validate()
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])