"""
Model registry system for NeuroLite.

Provides centralized registration and discovery of available models
with automatic model selection capabilities.
"""

from typing import Dict, List, Optional, Type, Any, Callable, Tuple
from dataclasses import dataclass
import importlib
from collections import defaultdict

from ..core import get_logger, ModelError, ModelNotFoundError, ModelCompatibilityError
from ..data.detector import DataType
from .base import BaseModel, TaskType, ModelCapabilities


logger = get_logger(__name__)


@dataclass
class ModelRegistration:
    """Information about a registered model."""
    
    name: str
    model_class: Type[BaseModel]
    capabilities: ModelCapabilities
    factory_function: Optional[Callable[..., BaseModel]] = None
    priority: int = 0  # Higher priority models are preferred for auto-selection
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelRegistry:
    """
    Central registry for all available models in NeuroLite.
    
    Manages model registration, discovery, and automatic selection
    based on data characteristics and task requirements.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, ModelRegistration] = {}
        self._task_index: Dict[TaskType, List[str]] = defaultdict(list)
        self._data_type_index: Dict[DataType, List[str]] = defaultdict(list)
        self._framework_index: Dict[str, List[str]] = defaultdict(list)
        logger.debug("Initialized ModelRegistry")
    
    def register_model(
        self,
        name: str,
        model_class: Type[BaseModel],
        factory_function: Optional[Callable[..., BaseModel]] = None,
        priority: int = 0,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            name: Unique name for the model
            model_class: Model class that inherits from BaseModel
            factory_function: Optional factory function to create model instances
            priority: Priority for auto-selection (higher = preferred)
            description: Human-readable description of the model
            tags: Optional tags for categorization
            
        Raises:
            ModelError: If model name already exists or model class is invalid
        """
        if name in self._models:
            raise ModelError(f"Model '{name}' is already registered")
        
        if not issubclass(model_class, BaseModel):
            raise ModelError(f"Model class must inherit from BaseModel, got {model_class}")
        
        # Get capabilities from a temporary instance or factory function
        try:
            if factory_function:
                # Use factory function to create temporary instance
                temp_instance = factory_function()
            else:
                # Try to create instance directly
                temp_instance = model_class()
            capabilities = temp_instance.capabilities
        except Exception as e:
            raise ModelError(f"Failed to get capabilities for model '{name}': {e}")
        
        # Create registration
        registration = ModelRegistration(
            name=name,
            model_class=model_class,
            capabilities=capabilities,
            factory_function=factory_function,
            priority=priority,
            description=description,
            tags=tags or []
        )
        
        # Store registration
        self._models[name] = registration
        
        # Update indices
        for task_type in capabilities.supported_tasks:
            self._task_index[task_type].append(name)
        
        for data_type in capabilities.supported_data_types:
            self._data_type_index[data_type].append(name)
        
        self._framework_index[capabilities.framework].append(name)
        
        logger.debug(f"Registered model '{name}' with capabilities: {capabilities}")
    
    def unregister_model(self, name: str) -> None:
        """
        Unregister a model from the registry.
        
        Args:
            name: Name of the model to unregister
            
        Raises:
            ModelNotFoundError: If model is not registered
        """
        if name not in self._models:
            raise ModelNotFoundError(name, list(self._models.keys()))
        
        registration = self._models[name]
        
        # Remove from indices
        for task_type in registration.capabilities.supported_tasks:
            if name in self._task_index[task_type]:
                self._task_index[task_type].remove(name)
        
        for data_type in registration.capabilities.supported_data_types:
            if name in self._data_type_index[data_type]:
                self._data_type_index[data_type].remove(name)
        
        if name in self._framework_index[registration.capabilities.framework]:
            self._framework_index[registration.capabilities.framework].remove(name)
        
        # Remove registration
        del self._models[name]
        
        logger.debug(f"Unregistered model '{name}'")
    
    def get_model(self, name: str, **kwargs) -> BaseModel:
        """
        Get a model instance by name.
        
        Args:
            name: Name of the model
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model is not registered
        """
        if name not in self._models:
            raise ModelNotFoundError(name, list(self._models.keys()))
        
        registration = self._models[name]
        
        try:
            if registration.factory_function:
                model = registration.factory_function(**kwargs)
            else:
                model = registration.model_class(**kwargs)
            
            logger.debug(f"Created instance of model '{name}'")
            return model
            
        except Exception as e:
            raise ModelError(f"Failed to create model '{name}': {e}")
    
    def list_models(
        self,
        task_type: Optional[TaskType] = None,
        data_type: Optional[DataType] = None,
        framework: Optional[str] = None
    ) -> List[str]:
        """
        List available models, optionally filtered by criteria.
        
        Args:
            task_type: Filter by task type
            data_type: Filter by data type
            framework: Filter by framework
            
        Returns:
            List of model names matching the criteria
        """
        models = set(self._models.keys())
        
        if task_type is not None:
            models &= set(self._task_index[task_type])
        
        if data_type is not None:
            models &= set(self._data_type_index[data_type])
        
        if framework is not None:
            models &= set(self._framework_index[framework])
        
        return sorted(list(models))
    
    def get_model_info(self, name: str) -> ModelRegistration:
        """
        Get detailed information about a registered model.
        
        Args:
            name: Name of the model
            
        Returns:
            ModelRegistration with model information
            
        Raises:
            ModelNotFoundError: If model is not registered
        """
        if name not in self._models:
            raise ModelNotFoundError(name, list(self._models.keys()))
        
        return self._models[name]
    
    def auto_select_model(
        self,
        task_type: TaskType,
        data_type: DataType,
        num_samples: Optional[int] = None,
        prefer_framework: Optional[str] = None,
        require_gpu: Optional[bool] = None
    ) -> str:
        """
        Automatically select the best model for given criteria.
        
        Args:
            task_type: Type of ML task
            data_type: Type of input data
            num_samples: Number of training samples (for model selection)
            prefer_framework: Preferred framework (if available)
            require_gpu: Whether GPU is required/available
            
        Returns:
            Name of the selected model
            
        Raises:
            ModelCompatibilityError: If no compatible model is found
        """
        logger.debug(f"Auto-selecting model for task={task_type.value}, data={data_type.value}")
        
        # Get models that support the task and data type
        compatible_models = []
        
        for name, registration in self._models.items():
            capabilities = registration.capabilities
            
            # Check task compatibility
            if task_type not in capabilities.supported_tasks:
                continue
            
            # Check data type compatibility
            if data_type not in capabilities.supported_data_types:
                continue
            
            # Check sample count requirements
            if num_samples is not None:
                if num_samples < capabilities.min_samples:
                    continue
                if (capabilities.max_samples is not None and 
                    num_samples > capabilities.max_samples):
                    continue
            
            # Check GPU requirements
            if require_gpu is not None:
                if require_gpu and not capabilities.requires_gpu:
                    continue
                if not require_gpu and capabilities.requires_gpu:
                    continue
            
            compatible_models.append((name, registration))
        
        if not compatible_models:
            raise ModelCompatibilityError(
                "auto",
                task_type.value,
                data_type.value,
                details=f"No models found that support task '{task_type.value}' with data type '{data_type.value}'"
            )
        
        # Score models based on preferences and priority
        scored_models = []
        
        for name, registration in compatible_models:
            score = registration.priority
            
            # Prefer specific framework if requested
            if prefer_framework and registration.capabilities.framework == prefer_framework:
                score += 10
            
            # Prefer models that don't require GPU if GPU is not available
            if require_gpu is False and not registration.capabilities.requires_gpu:
                score += 5
            
            scored_models.append((score, name, registration))
        
        # Sort by score (descending) and select the best
        scored_models.sort(key=lambda x: x[0], reverse=True)
        selected_name = scored_models[0][1]
        
        logger.debug(f"Auto-selected model: {selected_name}")
        return selected_name
    
    def get_compatible_models(
        self,
        task_type: TaskType,
        data_type: DataType
    ) -> List[Tuple[str, ModelRegistration]]:
        """
        Get all models compatible with given task and data type.
        
        Args:
            task_type: Type of ML task
            data_type: Type of input data
            
        Returns:
            List of (name, registration) tuples for compatible models
        """
        compatible = []
        
        for name, registration in self._models.items():
            capabilities = registration.capabilities
            
            if (task_type in capabilities.supported_tasks and
                data_type in capabilities.supported_data_types):
                compatible.append((name, registration))
        
        return compatible
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        stats = {
            'total_models': len(self._models),
            'by_framework': dict(self._framework_index),
            'by_task_type': {k.value: len(v) for k, v in self._task_index.items()},
            'by_data_type': {k.value: len(v) for k, v in self._data_type_index.items()},
        }
        
        return stats
    
    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        self._task_index.clear()
        self._data_type_index.clear()
        self._framework_index.clear()
        logger.debug("Cleared all registered models")


# Global registry instance
_global_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        Global ModelRegistry instance
    """
    return _global_registry


def register_model(
    name: str,
    model_class: Type[BaseModel],
    factory_function: Optional[Callable[..., BaseModel]] = None,
    priority: int = 0,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Register a model in the global registry.
    
    Args:
        name: Unique name for the model
        model_class: Model class that inherits from BaseModel
        factory_function: Optional factory function to create model instances
        priority: Priority for auto-selection (higher = preferred)
        description: Human-readable description of the model
        tags: Optional tags for categorization
    """
    _global_registry.register_model(
        name=name,
        model_class=model_class,
        factory_function=factory_function,
        priority=priority,
        description=description,
        tags=tags
    )


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Get a model instance from the global registry.
    
    Args:
        name: Name of the model
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
    """
    return _global_registry.get_model(name, **kwargs)


def list_models(
    task_type: Optional[TaskType] = None,
    data_type: Optional[DataType] = None,
    framework: Optional[str] = None
) -> List[str]:
    """
    List available models from the global registry.
    
    Args:
        task_type: Filter by task type
        data_type: Filter by data type
        framework: Filter by framework
        
    Returns:
        List of model names matching the criteria
    """
    return _global_registry.list_models(task_type, data_type, framework)


def auto_select_model(
    task_type: TaskType,
    data_type: DataType,
    num_samples: Optional[int] = None,
    prefer_framework: Optional[str] = None,
    require_gpu: Optional[bool] = None
) -> str:
    """
    Automatically select the best model from the global registry.
    
    Args:
        task_type: Type of ML task
        data_type: Type of input data
        num_samples: Number of training samples
        prefer_framework: Preferred framework
        require_gpu: Whether GPU is required/available
        
    Returns:
        Name of the selected model
    """
    return _global_registry.auto_select_model(
        task_type=task_type,
        data_type=data_type,
        num_samples=num_samples,
        prefer_framework=prefer_framework,
        require_gpu=require_gpu
    )