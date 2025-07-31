"""
Templates and examples for NeuroLite plugin development.

Provides code templates, documentation, and examples to help developers
create custom plugins for models and preprocessors.
"""

from pathlib import Path
from typing import Optional, Dict, Any


class PluginTemplateGenerator:
    """Generates plugin templates and documentation."""
    
    @staticmethod
    def generate_model_plugin_template(
        plugin_name: str,
        author: str,
        description: str,
        framework: str = "sklearn",
        output_dir: Optional[str] = None
    ) -> str:
        """
        Generate a model plugin template.
        
        Args:
            plugin_name: Name of the plugin
            author: Author name
            description: Plugin description
            framework: ML framework to use
            output_dir: Directory to save the template (optional)
            
        Returns:
            Generated template code as string
        """
        template = f'''"""
{plugin_name} - NeuroLite Model Plugin

{description}

Author: {author}
"""

from typing import List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass

from neurolite.core.plugins import ModelPlugin, PluginMetadata
from neurolite.models.base import BaseModel, TaskType, ModelCapabilities, PredictionResult
from neurolite.data.detector import DataType


class {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Model(BaseModel):
    """
    Custom model implementation for {plugin_name}.
    
    This model implements the NeuroLite BaseModel interface and provides
    {description.lower()}.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model.
        
        Args:
            **kwargs: Model configuration parameters
        """
        super().__init__(**kwargs)
        self.model = None  # Your actual model instance
        
        # Set default configuration
        self._config.setdefault('param1', 'default_value')
        self._config.setdefault('param2', 42)
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        return ModelCapabilities(
            supported_tasks=[
                TaskType.CLASSIFICATION,  # Modify based on your model
                TaskType.REGRESSION
            ],
            supported_data_types=[
                DataType.TABULAR  # Modify based on your model
            ],
            framework="{framework}",
            requires_gpu=False,  # Set to True if GPU is required
            min_samples=10,
            supports_probability_prediction=True,  # Set based on your model
            supports_feature_importance=True  # Set based on your model
        )
    
    def fit(
        self, 
        X: Union[np.ndarray, List, Any], 
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> '{''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Model':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        # TODO: Implement your model training logic here
        # Example for sklearn-style models:
        # from sklearn.ensemble import RandomForestClassifier
        # self.model = RandomForestClassifier(**self._config)
        # self.model.fit(X, y)
        
        self.is_trained = True
        return self
    
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """
        Make predictions.
        
        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # TODO: Implement your prediction logic here
        # Example:
        # predictions = self.model.predict(X)
        # probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        predictions = np.zeros(len(X))  # Placeholder
        probabilities = None
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            feature_importance=self.get_feature_importance()
        )
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        # TODO: Implement feature importance extraction
        # Example for tree-based models:
        # if hasattr(self.model, 'feature_importances_'):
        #     return self.model.feature_importances_
        
        return None
    
    def save(self, path: str) -> None:
        """Save the model."""
        # TODO: Implement model saving
        # Example for sklearn models:
        # import joblib
        # joblib.dump(self.model, path)
        pass
    
    def load(self, path: str) -> '{''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Model':
        """Load the model."""
        # TODO: Implement model loading
        # Example for sklearn models:
        # import joblib
        # self.model = joblib.load(path)
        # self.is_trained = True
        return self


class {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Plugin(ModelPlugin):
    """Plugin class for {plugin_name}."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="{plugin_name}",
            version="1.0.0",
            description="{description}",
            author="{author}",
            plugin_type="model",
            dependencies=[
                # List your dependencies here, e.g.:
                # "scikit-learn>=1.0.0",
                # "numpy>=1.20.0"
            ],
            supported_tasks=[
                "classification",  # Modify based on your model
                "regression"
            ],
            supported_data_types=[
                "tabular"  # Modify based on your model
            ],
            framework="{framework}",
            tags=["custom", "{framework}"]
        )
    
    @property
    def model_class(self):
        """Get the model class."""
        return {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Model
    
    @property
    def priority(self) -> int:
        """Get plugin priority for auto-selection."""
        return 5  # Adjust based on how much you want this model to be preferred


# Create plugin instance (this will be automatically discovered)
plugin = {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Plugin()
'''
        
        if output_dir:
            output_path = Path(output_dir) / f"{plugin_name.replace('-', '_')}_plugin.py"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template)
        
        return template
    
    @staticmethod
    def generate_preprocessor_plugin_template(
        plugin_name: str,
        author: str,
        description: str,
        data_type: str = "tabular",
        output_dir: Optional[str] = None
    ) -> str:
        """
        Generate a preprocessor plugin template.
        
        Args:
            plugin_name: Name of the plugin
            author: Author name
            description: Plugin description
            data_type: Data type to support
            output_dir: Directory to save the template (optional)
            
        Returns:
            Generated template code as string
        """
        template = f'''"""
{plugin_name} - NeuroLite Preprocessor Plugin

{description}

Author: {author}
"""

from typing import Dict, Any, Optional, List
import numpy as np

from neurolite.core.plugins import PreprocessorPlugin, PluginMetadata
from neurolite.data.preprocessor import BasePreprocessor, PreprocessingConfig
from neurolite.data.loader import Dataset
from neurolite.data.detector import DataType


class {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Preprocessor(BasePreprocessor):
    """
    Custom preprocessor implementation for {plugin_name}.
    
    This preprocessor implements the NeuroLite BasePreprocessor interface
    and provides {description.lower()}.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        super().__init__(config)
        
        # Add custom configuration parameters
        self.custom_param1 = getattr(config, 'custom_param1', 'default_value') if config else 'default_value'
        self.custom_param2 = getattr(config, 'custom_param2', 42) if config else 42
        
        # Internal state
        self._fitted_params = {{}}
    
    def fit(self, dataset: Dataset) -> '{''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Preprocessor':
        """
        Fit the preprocessor to the dataset.
        
        Args:
            dataset: Dataset to fit on
            
        Returns:
            Self for method chaining
        """
        if dataset.info.data_type != DataType.{data_type.upper()}:
            raise ValueError(f"This preprocessor expects {data_type} data, got {{dataset.info.data_type.value}}")
        
        # TODO: Implement your fitting logic here
        # Example: calculate statistics, build vocabularies, etc.
        
        # Store fitted parameters
        self._fitted_params = {{
            'param1': 'fitted_value',
            'param2': 123
        }}
        
        self.is_fitted = True
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset.
        
        Args:
            dataset: Dataset to transform
            
        Returns:
            Transformed dataset
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # TODO: Implement your transformation logic here
        transformed_data = []
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            # Apply your custom transformation
            transformed_sample = self._transform_sample(sample)
            transformed_data.append((transformed_sample, target))
        
        # Create new dataset with transformed data
        data = [item[0] for item in transformed_data]
        targets = [item[1] for item in transformed_data] if transformed_data[0][1] is not None else None
        
        # Update dataset info
        new_info = dataset.info
        new_info.metadata = {{**(new_info.metadata or {{}}), 'preprocessed_by': '{plugin_name}'}}
        
        return Dataset(data, targets=targets, info=new_info)
    
    def _transform_sample(self, sample: Any) -> Any:
        """
        Transform a single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Transformed sample
        """
        # TODO: Implement your sample transformation logic
        # This is where you apply your custom preprocessing
        
        return sample  # Placeholder - replace with actual transformation


class {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Plugin(PreprocessorPlugin):
    """Plugin class for {plugin_name}."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="{plugin_name}",
            version="1.0.0",
            description="{description}",
            author="{author}",
            plugin_type="preprocessor",
            dependencies=[
                # List your dependencies here, e.g.:
                # "numpy>=1.20.0",
                # "pandas>=1.3.0"
            ],
            supported_data_types=[
                "{data_type}"
            ],
            tags=["custom", "preprocessing"]
        )
    
    @property
    def preprocessor_class(self):
        """Get the preprocessor class."""
        return {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Preprocessor
    
    @property
    def config_schema(self) -> Optional[Dict[str, Any]]:
        """Get configuration schema for the preprocessor."""
        return {{
            "custom_param1": {{
                "type": "string",
                "default": "default_value",
                "description": "Description of custom parameter 1"
            }},
            "custom_param2": {{
                "type": "integer",
                "default": 42,
                "description": "Description of custom parameter 2"
            }}
        }}


# Create plugin instance (this will be automatically discovered)
plugin = {''.join(word.capitalize() for word in plugin_name.replace('-', '_').split('_'))}Plugin()
'''
        
        if output_dir:
            output_path = Path(output_dir) / f"{plugin_name.replace('-', '_')}_preprocessor_plugin.py"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template)
        
        return template
    
    @staticmethod
    def generate_plugin_config_template(output_dir: Optional[str] = None) -> str:
        """
        Generate a plugin configuration file template.
        
        Args:
            output_dir: Directory to save the template (optional)
            
        Returns:
            Generated configuration as string
        """
        config = '''# NeuroLite Plugin Configuration
# This file defines how to load plugins into NeuroLite

plugins:
  # Load plugins from individual files
  - type: file
    path: "path/to/your/plugin.py"
  
  # Load all plugins from a directory
  - type: directory
    path: "path/to/plugin/directory"
    recursive: true  # Search subdirectories
    pattern: "*.py"  # File pattern to match
  
  # Load plugins from an installed Python module
  - type: module
    module: "your_plugin_package"

# Plugin settings (optional)
settings:
  auto_integrate: true  # Automatically integrate plugins with core systems
  validate_on_load: true  # Validate plugins when loading
  log_level: "INFO"  # Logging level for plugin system
'''
        
        if output_dir:
            output_path = Path(output_dir) / "plugin_config.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(config)
        
        return config
    
    @staticmethod
    def generate_plugin_documentation(output_dir: Optional[str] = None) -> str:
        """
        Generate plugin development documentation.
        
        Args:
            output_dir: Directory to save the documentation (optional)
            
        Returns:
            Generated documentation as string
        """
        docs = '''# NeuroLite Plugin Development Guide

## Overview

NeuroLite supports a plugin system that allows you to extend the library with custom models and preprocessors. This guide will help you create your own plugins.

## Plugin Types

### Model Plugins

Model plugins allow you to add custom machine learning models to NeuroLite. Your model must inherit from `BaseModel` and implement the required interface.

#### Required Methods

- `capabilities`: Property that returns `ModelCapabilities`
- `fit(X, y, validation_data=None, **kwargs)`: Train the model
- `predict(X, **kwargs)`: Make predictions
- `save(path)`: Save the model to disk
- `load(path)`: Load the model from disk

#### Optional Methods

- `predict_proba(X, **kwargs)`: Predict class probabilities
- `get_feature_importance()`: Get feature importance scores

### Preprocessor Plugins

Preprocessor plugins allow you to add custom data preprocessing steps. Your preprocessor must inherit from `BasePreprocessor`.

#### Required Methods

- `fit(dataset)`: Fit the preprocessor to data
- `transform(dataset)`: Transform the data
- `fit_transform(dataset)`: Fit and transform in one step

## Creating a Plugin

### Step 1: Create Your Plugin Class

Create a Python file with your plugin implementation. Your plugin class should inherit from either `ModelPlugin` or `PreprocessorPlugin`.

### Step 2: Implement Required Methods

Implement all required methods for your plugin type. See the templates for examples.

### Step 3: Define Plugin Metadata

Your plugin must provide metadata including name, version, description, dependencies, and supported data types/tasks.

### Step 4: Test Your Plugin

Test your plugin thoroughly before deployment:

```python
from neurolite.core.plugins import register_model_plugin
from your_plugin import YourPlugin

# Register and test
plugin = YourPlugin()
register_model_plugin(plugin)

# Test with NeuroLite
import neurolite as nl
result = nl.train(data="your_data.csv", model="your-plugin-name")
```

## Plugin Configuration

You can configure plugin loading using a YAML configuration file:

```yaml
plugins:
  - type: file
    path: "path/to/plugin.py"
  - type: directory
    path: "plugins/"
    recursive: true
```

## Best Practices

1. **Follow Naming Conventions**: Use descriptive names for your plugins
2. **Document Your Code**: Provide clear docstrings and comments
3. **Handle Errors Gracefully**: Use appropriate exception handling
4. **Validate Inputs**: Check input data types and shapes
5. **Test Thoroughly**: Test with different data types and edge cases
6. **Declare Dependencies**: List all required packages in metadata
7. **Version Your Plugins**: Use semantic versioning

## Example: Simple Model Plugin

```python
from neurolite.core.plugins import ModelPlugin, PluginMetadata
from neurolite.models.base import BaseModel, ModelCapabilities
from sklearn.ensemble import RandomForestClassifier

class SimpleRFModel(BaseModel):
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="sklearn"
        )
    
    def fit(self, X, y, **kwargs):
        self.model = RandomForestClassifier()
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X, **kwargs):
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return PredictionResult(predictions=predictions, probabilities=probabilities)
    
    def save(self, path):
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path):
        import joblib
        self.model = joblib.load(path)
        self.is_trained = True
        return self

class SimpleRFPlugin(ModelPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="simple-rf",
            version="1.0.0",
            description="Simple Random Forest classifier",
            author="Your Name",
            plugin_type="model",
            dependencies=["scikit-learn>=1.0.0"],
            supported_tasks=["classification"],
            supported_data_types=["tabular"],
            framework="sklearn"
        )
    
    @property
    def model_class(self):
        return SimpleRFModel

plugin = SimpleRFPlugin()
```

## Troubleshooting

### Common Issues

1. **Plugin Not Found**: Check that your plugin file is in the search path
2. **Validation Errors**: Ensure your plugin implements all required methods
3. **Import Errors**: Check that all dependencies are installed
4. **Registration Fails**: Verify your plugin metadata is correct

### Debugging Tips

1. Enable debug logging: `logging.getLogger('neurolite').setLevel(logging.DEBUG)`
2. Test plugin validation: `plugin.validate()`
3. Check plugin registry: `get_plugin_registry().get_statistics()`

## Support

For help with plugin development:
- Check the NeuroLite documentation
- Look at example plugins in the repository
- Open an issue on GitHub for bugs or feature requests
'''
        
        if output_dir:
            output_path = Path(output_dir) / "PLUGIN_DEVELOPMENT.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(docs)
        
        return docs


def create_plugin_template(
    plugin_type: str,
    plugin_name: str,
    author: str,
    description: str,
    output_dir: str = ".",
    **kwargs
) -> None:
    """
    Create a plugin template with all necessary files.
    
    Args:
        plugin_type: Type of plugin ("model" or "preprocessor")
        plugin_name: Name of the plugin
        author: Author name
        description: Plugin description
        output_dir: Output directory
        **kwargs: Additional template parameters
    """
    generator = PluginTemplateGenerator()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate appropriate template
    if plugin_type.lower() == "model":
        framework = kwargs.get('framework', 'sklearn')
        generator.generate_model_plugin_template(
            plugin_name, author, description, framework, str(output_path)
        )
    elif plugin_type.lower() == "preprocessor":
        data_type = kwargs.get('data_type', 'tabular')
        generator.generate_preprocessor_plugin_template(
            plugin_name, author, description, data_type, str(output_path)
        )
    else:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    
    # Generate configuration template
    generator.generate_plugin_config_template(str(output_path))
    
    # Generate documentation
    generator.generate_plugin_documentation(str(output_path))
    
    print(f"Plugin template created in: {output_path}")
    print(f"Files generated:")
    print(f"  - {plugin_name.replace('-', '_')}_plugin.py (or *_preprocessor_plugin.py)")
    print(f"  - plugin_config.yaml")
    print(f"  - PLUGIN_DEVELOPMENT.md")