"""
Plugin system for NeuroLite.

Provides extensible plugin architecture for custom models and preprocessors
with registration, discovery, validation, and loading capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any, Callable, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.util
import inspect
import sys
import json
import yaml
from collections import defaultdict

from . import get_logger
from .exceptions import NeuroLiteError, ConfigurationError, DependencyError
from ..data.detector import DataType
from ..models.base import BaseModel, TaskType, ModelCapabilities
from ..data.preprocessor import BasePreprocessor


logger = get_logger(__name__)


class PluginError(NeuroLiteError):
    """Plugin-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'PLUGIN_ERROR')
        super().__init__(message, **kwargs)


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""
    
    def __init__(self, plugin_name: str, validation_errors: List[str], **kwargs):
        message = f"Plugin '{plugin_name}' validation failed"
        details = "Validation errors:\n" + "\n".join(f"  - {error}" for error in validation_errors)
        suggestions = [
            "Check that your plugin class inherits from the correct base class",
            "Ensure all required methods are implemented",
            "Verify plugin metadata is correctly specified",
            "Check that all dependencies are properly declared"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="PLUGIN_VALIDATION_ERROR",
            context={"plugin_name": plugin_name, "validation_errors": validation_errors},
            **kwargs
        )


class PluginLoadError(PluginError):
    """Raised when plugin fails to load."""
    
    def __init__(self, plugin_name: str, reason: str, **kwargs):
        message = f"Failed to load plugin '{plugin_name}': {reason}"
        suggestions = [
            "Check that the plugin file exists and is accessible",
            "Ensure all plugin dependencies are installed",
            "Verify the plugin code has no syntax errors",
            "Check that the plugin follows the correct structure"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="PLUGIN_LOAD_ERROR",
            context={"plugin_name": plugin_name, "reason": reason},
            **kwargs
        )


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # "model" or "preprocessor"
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_version: Optional[str] = None
    neurolite_version: Optional[str] = None
    
    # Plugin-specific metadata
    supported_tasks: List[str] = field(default_factory=list)
    supported_data_types: List[str] = field(default_factory=list)
    framework: Optional[str] = None
    
    # Optional metadata
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Internal metadata
    file_path: Optional[str] = None
    loaded: bool = False
    validated: bool = False


@dataclass
class ModelPluginInfo:
    """Information about a model plugin."""
    
    metadata: PluginMetadata
    model_class: Type[BaseModel]
    capabilities: ModelCapabilities
    factory_function: Optional[Callable[..., BaseModel]] = None
    priority: int = 0


@dataclass
class PreprocessorPluginInfo:
    """Information about a preprocessor plugin."""
    
    metadata: PluginMetadata
    preprocessor_class: Type[BasePreprocessor]
    supported_data_types: List[DataType]
    config_schema: Optional[Dict[str, Any]] = None


class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """
        Validate plugin implementation.
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass


class ModelPlugin(PluginInterface):
    """Base class for model plugins."""
    
    @property
    @abstractmethod
    def model_class(self) -> Type[BaseModel]:
        """Get the model class provided by this plugin."""
        pass
    
    @property
    def factory_function(self) -> Optional[Callable[..., BaseModel]]:
        """Optional factory function for creating model instances."""
        return None
    
    @property
    def priority(self) -> int:
        """Priority for auto-selection (higher = preferred)."""
        return 0
    
    def validate(self) -> List[str]:
        """Validate model plugin."""
        errors = []
        
        # Check model class
        model_class = self.model_class
        if not inspect.isclass(model_class):
            errors.append("model_class must be a class")
        elif not issubclass(model_class, BaseModel):
            errors.append("model_class must inherit from BaseModel")
        
        # Check metadata
        metadata = self.metadata
        if not metadata.name:
            errors.append("Plugin name is required")
        if not metadata.version:
            errors.append("Plugin version is required")
        if metadata.plugin_type != "model":
            errors.append("Plugin type must be 'model' for model plugins")
        
        # Check capabilities
        try:
            if hasattr(model_class, 'capabilities'):
                # Try to get capabilities from class
                temp_instance = model_class()
                capabilities = temp_instance.capabilities
                
                # Validate task types
                for task_str in metadata.supported_tasks:
                    try:
                        TaskType(task_str)
                    except ValueError:
                        errors.append(f"Invalid task type: {task_str}")
                
                # Validate data types
                for data_type_str in metadata.supported_data_types:
                    try:
                        DataType(data_type_str)
                    except ValueError:
                        errors.append(f"Invalid data type: {data_type_str}")
            else:
                errors.append("Model class must have capabilities property")
        except Exception as e:
            errors.append(f"Failed to validate model capabilities: {e}")
        
        return errors


class PreprocessorPlugin(PluginInterface):
    """Base class for preprocessor plugins."""
    
    @property
    @abstractmethod
    def preprocessor_class(self) -> Type[BasePreprocessor]:
        """Get the preprocessor class provided by this plugin."""
        pass
    
    @property
    def config_schema(self) -> Optional[Dict[str, Any]]:
        """Optional configuration schema for the preprocessor."""
        return None
    
    def validate(self) -> List[str]:
        """Validate preprocessor plugin."""
        errors = []
        
        # Check preprocessor class
        preprocessor_class = self.preprocessor_class
        if not inspect.isclass(preprocessor_class):
            errors.append("preprocessor_class must be a class")
        elif not issubclass(preprocessor_class, BasePreprocessor):
            errors.append("preprocessor_class must inherit from BasePreprocessor")
        
        # Check metadata
        metadata = self.metadata
        if not metadata.name:
            errors.append("Plugin name is required")
        if not metadata.version:
            errors.append("Plugin version is required")
        if metadata.plugin_type != "preprocessor":
            errors.append("Plugin type must be 'preprocessor' for preprocessor plugins")
        
        # Validate data types
        for data_type_str in metadata.supported_data_types:
            try:
                DataType(data_type_str)
            except ValueError:
                errors.append(f"Invalid data type: {data_type_str}")
        
        return errors


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self._model_plugins: Dict[str, ModelPluginInfo] = {}
        self._preprocessor_plugins: Dict[str, PreprocessorPluginInfo] = {}
        self._plugin_paths: Set[str] = set()
        self._loaded_modules: Dict[str, Any] = {}
        logger.debug("Initialized PluginRegistry")
    
    def register_model_plugin(
        self,
        plugin: ModelPlugin,
        validate: bool = True
    ) -> None:
        """
        Register a model plugin.
        
        Args:
            plugin: Model plugin instance
            validate: Whether to validate the plugin
            
        Raises:
            PluginValidationError: If plugin validation fails
        """
        metadata = plugin.metadata
        
        if validate:
            errors = plugin.validate()
            if errors:
                raise PluginValidationError(metadata.name, errors)
        
        # Create capabilities from model class
        try:
            temp_instance = plugin.model_class()
            capabilities = temp_instance.capabilities
        except Exception as e:
            raise PluginError(f"Failed to get capabilities for plugin '{metadata.name}': {e}")
        
        # Create plugin info
        plugin_info = ModelPluginInfo(
            metadata=metadata,
            model_class=plugin.model_class,
            capabilities=capabilities,
            factory_function=plugin.factory_function,
            priority=plugin.priority
        )
        
        self._model_plugins[metadata.name] = plugin_info
        metadata.validated = True
        
        logger.info(f"Registered model plugin: {metadata.name} v{metadata.version}")
    
    def register_preprocessor_plugin(
        self,
        plugin: PreprocessorPlugin,
        validate: bool = True
    ) -> None:
        """
        Register a preprocessor plugin.
        
        Args:
            plugin: Preprocessor plugin instance
            validate: Whether to validate the plugin
            
        Raises:
            PluginValidationError: If plugin validation fails
        """
        metadata = plugin.metadata
        
        if validate:
            errors = plugin.validate()
            if errors:
                raise PluginValidationError(metadata.name, errors)
        
        # Convert string data types to enum
        supported_data_types = []
        for data_type_str in metadata.supported_data_types:
            try:
                supported_data_types.append(DataType(data_type_str))
            except ValueError:
                raise PluginError(f"Invalid data type '{data_type_str}' in plugin '{metadata.name}'")
        
        # Create plugin info
        plugin_info = PreprocessorPluginInfo(
            metadata=metadata,
            preprocessor_class=plugin.preprocessor_class,
            supported_data_types=supported_data_types,
            config_schema=plugin.config_schema
        )
        
        self._preprocessor_plugins[metadata.name] = plugin_info
        metadata.validated = True
        
        logger.info(f"Registered preprocessor plugin: {metadata.name} v{metadata.version}")
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Name of the plugin to unregister
        """
        if name in self._model_plugins:
            del self._model_plugins[name]
            logger.info(f"Unregistered model plugin: {name}")
        elif name in self._preprocessor_plugins:
            del self._preprocessor_plugins[name]
            logger.info(f"Unregistered preprocessor plugin: {name}")
        else:
            logger.warning(f"Plugin '{name}' not found for unregistration")
    
    def get_model_plugin(self, name: str) -> ModelPluginInfo:
        """
        Get model plugin by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Model plugin info
            
        Raises:
            PluginError: If plugin not found
        """
        if name not in self._model_plugins:
            available = list(self._model_plugins.keys())
            raise PluginError(
                f"Model plugin '{name}' not found",
                details=f"Available model plugins: {', '.join(available)}",
                suggestions=["Check the plugin name spelling", "Ensure the plugin is properly registered"]
            )
        
        return self._model_plugins[name]
    
    def get_preprocessor_plugin(self, name: str) -> PreprocessorPluginInfo:
        """
        Get preprocessor plugin by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Preprocessor plugin info
            
        Raises:
            PluginError: If plugin not found
        """
        if name not in self._preprocessor_plugins:
            available = list(self._preprocessor_plugins.keys())
            raise PluginError(
                f"Preprocessor plugin '{name}' not found",
                details=f"Available preprocessor plugins: {', '.join(available)}",
                suggestions=["Check the plugin name spelling", "Ensure the plugin is properly registered"]
            )
        
        return self._preprocessor_plugins[name]
    
    def list_model_plugins(
        self,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        framework: Optional[str] = None
    ) -> List[str]:
        """
        List available model plugins.
        
        Args:
            task_type: Filter by task type
            data_type: Filter by data type
            framework: Filter by framework
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for name, plugin_info in self._model_plugins.items():
            capabilities = plugin_info.capabilities
            
            # Apply filters
            if task_type and task_type not in capabilities.supported_tasks:
                continue
            if data_type and data_type not in capabilities.supported_data_types:
                continue
            if framework and capabilities.framework != framework:
                continue
            
            plugins.append(name)
        
        return sorted(plugins)
    
    def list_preprocessor_plugins(
        self,
        data_type: Optional[DataType] = None
    ) -> List[str]:
        """
        List available preprocessor plugins.
        
        Args:
            data_type: Filter by data type
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for name, plugin_info in self._preprocessor_plugins.items():
            # Apply filters
            if data_type and data_type not in plugin_info.supported_data_types:
                continue
            
            plugins.append(name)
        
        return sorted(plugins)
    
    def get_plugin_info(self, name: str) -> Union[ModelPluginInfo, PreprocessorPluginInfo]:
        """
        Get plugin information by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Plugin information
            
        Raises:
            PluginError: If plugin not found
        """
        if name in self._model_plugins:
            return self._model_plugins[name]
        elif name in self._preprocessor_plugins:
            return self._preprocessor_plugins[name]
        else:
            all_plugins = list(self._model_plugins.keys()) + list(self._preprocessor_plugins.keys())
            raise PluginError(
                f"Plugin '{name}' not found",
                details=f"Available plugins: {', '.join(all_plugins)}",
                suggestions=["Check the plugin name spelling", "Ensure the plugin is properly registered"]
            )
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._model_plugins.clear()
        self._preprocessor_plugins.clear()
        self._plugin_paths.clear()
        self._loaded_modules.clear()
        logger.info("Cleared all registered plugins")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get plugin registry statistics.
        
        Returns:
            Dictionary with statistics
        """
        model_frameworks = defaultdict(int)
        preprocessor_data_types = defaultdict(int)
        
        for plugin_info in self._model_plugins.values():
            model_frameworks[plugin_info.capabilities.framework] += 1
        
        for plugin_info in self._preprocessor_plugins.values():
            for data_type in plugin_info.supported_data_types:
                preprocessor_data_types[data_type.value] += 1
        
        return {
            'total_plugins': len(self._model_plugins) + len(self._preprocessor_plugins),
            'model_plugins': len(self._model_plugins),
            'preprocessor_plugins': len(self._preprocessor_plugins),
            'model_frameworks': dict(model_frameworks),
            'preprocessor_data_types': dict(preprocessor_data_types),
            'loaded_paths': list(self._plugin_paths)
        }


# Global plugin registry instance
_global_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        Global PluginRegistry instance
    """
    return _global_plugin_registry


def register_model_plugin(plugin: ModelPlugin, validate: bool = True) -> None:
    """
    Register a model plugin in the global registry.
    
    Args:
        plugin: Model plugin instance
        validate: Whether to validate the plugin
    """
    _global_plugin_registry.register_model_plugin(plugin, validate)


def register_preprocessor_plugin(plugin: PreprocessorPlugin, validate: bool = True) -> None:
    """
    Register a preprocessor plugin in the global registry.
    
    Args:
        plugin: Preprocessor plugin instance
        validate: Whether to validate the plugin
    """
    _global_plugin_registry.register_preprocessor_plugin(plugin, validate)


class PluginLoader:
    """Handles plugin discovery and loading."""
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize plugin loader.
        
        Args:
            registry: Plugin registry to use (defaults to global registry)
        """
        self.registry = registry or get_plugin_registry()
        self._dependency_checker = DependencyChecker()
    
    def load_plugin_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load a plugin from a Python file.
        
        Args:
            file_path: Path to the plugin file
            
        Raises:
            PluginLoadError: If plugin fails to load
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PluginLoadError("file_not_found", f"Plugin file not found: {file_path}")
        
        if not file_path.suffix == '.py':
            raise PluginLoadError("invalid_file", f"Plugin file must be a Python file: {file_path}")
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError("import_error", f"Failed to create module spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in module
            plugins_found = self._discover_plugins_in_module(module)
            
            if not plugins_found:
                raise PluginLoadError("no_plugins", f"No valid plugins found in {file_path}")
            
            # Register found plugins
            for plugin in plugins_found:
                self._register_plugin_with_validation(plugin, str(file_path))
            
            self.registry._plugin_paths.add(str(file_path))
            self.registry._loaded_modules[str(file_path)] = module
            
            logger.info(f"Loaded {len(plugins_found)} plugin(s) from {file_path}")
            
        except Exception as e:
            if isinstance(e, PluginLoadError):
                raise
            raise PluginLoadError("load_error", f"Failed to load plugin from {file_path}: {e}")
    
    def load_plugins_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*.py"
    ) -> None:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory to search for plugins
            recursive: Whether to search recursively
            pattern: File pattern to match
            
        Raises:
            PluginLoadError: If directory doesn't exist
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise PluginLoadError("directory_not_found", f"Plugin directory not found: {directory}")
        
        if not directory.is_dir():
            raise PluginLoadError("not_directory", f"Path is not a directory: {directory}")
        
        # Find plugin files
        if recursive:
            plugin_files = list(directory.rglob(pattern))
        else:
            plugin_files = list(directory.glob(pattern))
        
        # Filter out __init__.py and other special files
        plugin_files = [f for f in plugin_files if f.name not in ['__init__.py', '__pycache__']]
        
        loaded_count = 0
        errors = []
        
        for plugin_file in plugin_files:
            try:
                self.load_plugin_from_file(plugin_file)
                loaded_count += 1
            except PluginLoadError as e:
                errors.append(f"{plugin_file}: {e.message}")
                logger.warning(f"Failed to load plugin from {plugin_file}: {e.message}")
        
        if errors and loaded_count == 0:
            # If no plugins loaded and there were errors, raise an error
            raise PluginLoadError(
                "no_plugins_loaded",
                f"Failed to load any plugins from {directory}. Errors: {'; '.join(errors)}"
            )
        
        logger.info(f"Loaded {loaded_count} plugin(s) from {directory}")
        if errors:
            logger.warning(f"Failed to load {len(errors)} plugin file(s)")
    
    def load_plugin_from_config(self, config_path: Union[str, Path]) -> None:
        """
        Load plugins from a configuration file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Raises:
            PluginLoadError: If config file is invalid or plugins fail to load
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise PluginLoadError("config_not_found", f"Plugin config file not found: {config_path}")
        
        try:
            # Load configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise PluginLoadError("invalid_config", f"Unsupported config file format: {config_path.suffix}")
            
            if not isinstance(config, dict):
                raise PluginLoadError("invalid_config", "Config file must contain a dictionary")
            
            plugins_config = config.get('plugins', [])
            if not isinstance(plugins_config, list):
                raise PluginLoadError("invalid_config", "Plugins configuration must be a list")
            
            # Load each plugin
            for plugin_config in plugins_config:
                self._load_plugin_from_config_entry(plugin_config)
            
            logger.info(f"Loaded plugins from config: {config_path}")
            
        except Exception as e:
            if isinstance(e, PluginLoadError):
                raise
            raise PluginLoadError("config_error", f"Failed to load plugins from config {config_path}: {e}")
    
    def _discover_plugins_in_module(self, module: Any) -> List[PluginInterface]:
        """
        Discover plugin classes in a module.
        
        Args:
            module: Python module to search
            
        Returns:
            List of plugin instances found in the module
        """
        plugins = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, PluginInterface) and 
                obj is not PluginInterface and
                obj is not ModelPlugin and
                obj is not PreprocessorPlugin):
                
                try:
                    # Try to instantiate the plugin
                    plugin_instance = obj()
                    plugins.append(plugin_instance)
                    logger.debug(f"Found plugin class: {name}")
                except Exception as e:
                    logger.warning(f"Failed to instantiate plugin class {name}: {e}")
        
        return plugins
    
    def _register_plugin_with_validation(self, plugin: PluginInterface, file_path: str) -> None:
        """
        Register a plugin with validation and dependency checking.
        
        Args:
            plugin: Plugin instance to register
            file_path: Path to the plugin file
            
        Raises:
            PluginLoadError: If plugin validation or dependency check fails
        """
        metadata = plugin.metadata
        metadata.file_path = file_path
        
        # Check dependencies
        missing_deps = self._dependency_checker.check_dependencies(metadata.dependencies)
        if missing_deps:
            raise PluginLoadError(
                "missing_dependencies",
                f"Plugin '{metadata.name}' has missing dependencies: {', '.join(missing_deps)}"
            )
        
        # Register based on plugin type
        try:
            if isinstance(plugin, ModelPlugin):
                self.registry.register_model_plugin(plugin, validate=True)
            elif isinstance(plugin, PreprocessorPlugin):
                self.registry.register_preprocessor_plugin(plugin, validate=True)
            else:
                raise PluginLoadError("invalid_plugin", f"Unknown plugin type: {type(plugin)}")
        except PluginValidationError as e:
            raise PluginLoadError("validation_failed", str(e))
    
    def _load_plugin_from_config_entry(self, plugin_config: Dict[str, Any]) -> None:
        """
        Load a plugin from a configuration entry.
        
        Args:
            plugin_config: Plugin configuration dictionary
            
        Raises:
            PluginLoadError: If plugin configuration is invalid
        """
        if not isinstance(plugin_config, dict):
            raise PluginLoadError("invalid_config", "Plugin configuration must be a dictionary")
        
        plugin_type = plugin_config.get('type')
        if plugin_type == 'file':
            file_path = plugin_config.get('path')
            if not file_path:
                raise PluginLoadError("invalid_config", "File plugin must specify 'path'")
            self.load_plugin_from_file(file_path)
        
        elif plugin_type == 'directory':
            directory = plugin_config.get('path')
            if not directory:
                raise PluginLoadError("invalid_config", "Directory plugin must specify 'path'")
            recursive = plugin_config.get('recursive', True)
            pattern = plugin_config.get('pattern', '*.py')
            self.load_plugins_from_directory(directory, recursive, pattern)
        
        elif plugin_type == 'module':
            module_name = plugin_config.get('module')
            if not module_name:
                raise PluginLoadError("invalid_config", "Module plugin must specify 'module'")
            self._load_plugin_from_module(module_name)
        
        else:
            raise PluginLoadError("invalid_config", f"Unknown plugin type: {plugin_type}")
    
    def _load_plugin_from_module(self, module_name: str) -> None:
        """
        Load plugins from an installed Python module.
        
        Args:
            module_name: Name of the module to load
            
        Raises:
            PluginLoadError: If module cannot be loaded
        """
        try:
            module = importlib.import_module(module_name)
            plugins_found = self._discover_plugins_in_module(module)
            
            if not plugins_found:
                raise PluginLoadError("no_plugins", f"No valid plugins found in module {module_name}")
            
            # Register found plugins
            for plugin in plugins_found:
                self._register_plugin_with_validation(plugin, f"module:{module_name}")
            
            self.registry._loaded_modules[module_name] = module
            
            logger.info(f"Loaded {len(plugins_found)} plugin(s) from module {module_name}")
            
        except ImportError as e:
            raise PluginLoadError("import_error", f"Failed to import module {module_name}: {e}")
        except Exception as e:
            raise PluginLoadError("load_error", f"Failed to load plugins from module {module_name}: {e}")


class DependencyChecker:
    """Checks plugin dependencies."""
    
    def check_dependencies(self, dependencies: List[str]) -> List[str]:
        """
        Check if all dependencies are available.
        
        Args:
            dependencies: List of dependency specifications
            
        Returns:
            List of missing dependencies
        """
        missing = []
        
        for dep in dependencies:
            if not self._is_dependency_available(dep):
                missing.append(dep)
        
        return missing
    
    def _is_dependency_available(self, dependency: str) -> bool:
        """
        Check if a single dependency is available.
        
        Args:
            dependency: Dependency specification (e.g., "numpy>=1.20.0")
            
        Returns:
            True if dependency is available
        """
        try:
            # Simple check - just try to import the package name
            # In a more sophisticated implementation, you would parse version requirements
            package_name = dependency.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False


# Convenience functions for plugin loading
def load_plugin_from_file(file_path: Union[str, Path]) -> None:
    """
    Load a plugin from a file using the global registry.
    
    Args:
        file_path: Path to the plugin file
    """
    loader = PluginLoader()
    loader.load_plugin_from_file(file_path)


def load_plugins_from_directory(
    directory: Union[str, Path],
    recursive: bool = True,
    pattern: str = "*.py"
) -> None:
    """
    Load plugins from a directory using the global registry.
    
    Args:
        directory: Directory to search for plugins
        recursive: Whether to search recursively
        pattern: File pattern to match
    """
    loader = PluginLoader()
    loader.load_plugins_from_directory(directory, recursive, pattern)


def load_plugins_from_config(config_path: Union[str, Path]) -> None:
    """
    Load plugins from a configuration file using the global registry.
    
    Args:
        config_path: Path to the configuration file
    """
    loader = PluginLoader()
    loader.load_plugin_from_config(config_path)