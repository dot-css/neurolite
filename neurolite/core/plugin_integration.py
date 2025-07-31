"""
Integration between plugin system and existing NeuroLite components.

Provides seamless integration of plugins with the model registry and
preprocessor factory systems.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from . import get_logger
from .plugins import (
    get_plugin_registry, PluginLoader, ModelPluginInfo, PreprocessorPluginInfo
)
from ..models.registry import get_model_registry, register_model
from ..data.preprocessor import PreprocessorFactory, BasePreprocessor
from ..data.detector import DataType


logger = get_logger(__name__)


class PluginIntegration:
    """Handles integration between plugins and core NeuroLite systems."""
    
    def __init__(self):
        """Initialize plugin integration."""
        self.plugin_registry = get_plugin_registry()
        self.model_registry = get_model_registry()
        self._integrated_models = set()
        self._integrated_preprocessors = set()
    
    def integrate_all_plugins(self) -> None:
        """Integrate all registered plugins with core systems."""
        self.integrate_model_plugins()
        self.integrate_preprocessor_plugins()
    
    def integrate_model_plugins(self) -> None:
        """Integrate model plugins with the model registry."""
        for name, plugin_info in self.plugin_registry._model_plugins.items():
            if name not in self._integrated_models:
                self._integrate_model_plugin(name, plugin_info)
                self._integrated_models.add(name)
    
    def integrate_preprocessor_plugins(self) -> None:
        """Integrate preprocessor plugins with the preprocessor factory."""
        for name, plugin_info in self.plugin_registry._preprocessor_plugins.items():
            if name not in self._integrated_preprocessors:
                self._integrate_preprocessor_plugin(name, plugin_info)
                self._integrated_preprocessors.add(name)
    
    def _integrate_model_plugin(self, name: str, plugin_info: ModelPluginInfo) -> None:
        """
        Integrate a single model plugin with the model registry.
        
        Args:
            name: Plugin name
            plugin_info: Model plugin information
        """
        try:
            # Register the model with the model registry
            register_model(
                name=name,
                model_class=plugin_info.model_class,
                factory_function=plugin_info.factory_function,
                priority=plugin_info.priority,
                description=plugin_info.metadata.description,
                tags=plugin_info.metadata.tags
            )
            
            logger.debug(f"Integrated model plugin '{name}' with model registry")
            
        except Exception as e:
            logger.error(f"Failed to integrate model plugin '{name}': {e}")
    
    def _integrate_preprocessor_plugin(self, name: str, plugin_info: PreprocessorPluginInfo) -> None:
        """
        Integrate a single preprocessor plugin with the preprocessor factory.
        
        Args:
            name: Plugin name
            plugin_info: Preprocessor plugin information
        """
        try:
            # Register the preprocessor with the factory
            # Note: This would require extending PreprocessorFactory to support plugins
            # For now, we'll just log the integration
            logger.debug(f"Integrated preprocessor plugin '{name}' (placeholder)")
            
        except Exception as e:
            logger.error(f"Failed to integrate preprocessor plugin '{name}': {e}")
    
    def load_and_integrate_plugins(
        self,
        plugin_paths: Optional[List[str]] = None,
        plugin_directories: Optional[List[str]] = None,
        config_file: Optional[str] = None
    ) -> None:
        """
        Load plugins from various sources and integrate them.
        
        Args:
            plugin_paths: List of plugin file paths to load
            plugin_directories: List of directories to search for plugins
            config_file: Path to plugin configuration file
        """
        loader = PluginLoader(self.plugin_registry)
        
        # Load from individual files
        if plugin_paths:
            for path in plugin_paths:
                try:
                    loader.load_plugin_from_file(path)
                except Exception as e:
                    logger.error(f"Failed to load plugin from {path}: {e}")
        
        # Load from directories
        if plugin_directories:
            for directory in plugin_directories:
                try:
                    loader.load_plugins_from_directory(directory)
                except Exception as e:
                    logger.error(f"Failed to load plugins from {directory}: {e}")
        
        # Load from config file
        if config_file:
            try:
                loader.load_plugins_from_config(config_file)
            except Exception as e:
                logger.error(f"Failed to load plugins from config {config_file}: {e}")
        
        # Integrate all loaded plugins
        self.integrate_all_plugins()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of plugin integration.
        
        Returns:
            Dictionary with integration status information
        """
        plugin_stats = self.plugin_registry.get_statistics()
        
        return {
            'total_plugins': plugin_stats['total_plugins'],
            'integrated_models': len(self._integrated_models),
            'integrated_preprocessors': len(self._integrated_preprocessors),
            'model_plugins': list(self._integrated_models),
            'preprocessor_plugins': list(self._integrated_preprocessors),
            'plugin_registry_stats': plugin_stats
        }


# Global integration instance
_global_integration = PluginIntegration()


def get_plugin_integration() -> PluginIntegration:
    """
    Get the global plugin integration instance.
    
    Returns:
        Global PluginIntegration instance
    """
    return _global_integration


def auto_discover_and_load_plugins(
    search_paths: Optional[List[str]] = None,
    config_file: Optional[str] = None
) -> None:
    """
    Automatically discover and load plugins from common locations.
    
    Args:
        search_paths: Additional paths to search for plugins
        config_file: Path to plugin configuration file
    """
    integration = get_plugin_integration()
    
    # Default search paths
    default_paths = [
        "plugins",
        "neurolite_plugins",
        Path.home() / ".neurolite" / "plugins",
        Path.cwd() / "plugins"
    ]
    
    # Add user-specified paths
    if search_paths:
        default_paths.extend(search_paths)
    
    # Filter to existing directories
    existing_dirs = [str(path) for path in default_paths if Path(path).is_dir()]
    
    # Load plugins
    integration.load_and_integrate_plugins(
        plugin_directories=existing_dirs,
        config_file=config_file
    )
    
    if existing_dirs:
        logger.info(f"Auto-discovered plugins from: {', '.join(existing_dirs)}")
    else:
        logger.debug("No plugin directories found for auto-discovery")


def initialize_plugin_system(
    auto_discover: bool = True,
    search_paths: Optional[List[str]] = None,
    config_file: Optional[str] = None
) -> None:
    """
    Initialize the plugin system.
    
    Args:
        auto_discover: Whether to automatically discover plugins
        search_paths: Additional paths to search for plugins
        config_file: Path to plugin configuration file
    """
    logger.info("Initializing NeuroLite plugin system...")
    
    if auto_discover:
        auto_discover_and_load_plugins(search_paths, config_file)
    
    # Get integration status
    integration = get_plugin_integration()
    status = integration.get_integration_status()
    
    logger.info(
        f"Plugin system initialized: {status['total_plugins']} plugins loaded, "
        f"{status['integrated_models']} models integrated, "
        f"{status['integrated_preprocessors']} preprocessors integrated"
    )