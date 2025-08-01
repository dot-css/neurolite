"""
Workflow factory for NeuroLite.

Provides factory functions to create workflow instances based on data type
and task specifications for domain-specific workflow coordination.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

from ..core import get_logger, ConfigurationError
from ..data import DataType, detect_data_type
from ..models import TaskType
from .base import BaseWorkflow, WorkflowConfig
from .vision import VisionWorkflow
from .nlp import NLPWorkflow
from .tabular import TabularWorkflow


logger = get_logger(__name__)


class WorkflowFactory:
    """
    Factory class for creating domain-specific workflow instances.
    
    Handles workflow selection based on data type and task type,
    and provides unified interface for workflow creation.
    """
    
    def __init__(self):
        """Initialize the workflow factory."""
        self._workflows = {
            DataType.IMAGE: VisionWorkflow,
            DataType.TEXT: NLPWorkflow,
            DataType.TABULAR: TabularWorkflow
        }
        logger.debug("Initialized WorkflowFactory")
    
    def create_workflow(
        self,
        data_path: Union[str, Path],
        model: str = "auto",
        task: str = "auto",
        target: Optional[str] = None,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        optimize: bool = True,
        deploy: bool = False,
        domain_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseWorkflow:
        """
        Create appropriate workflow based on data type.
        
        Args:
            data_path: Path to data file or directory
            model: Model type to use ('auto' for automatic selection)
            task: Task type ('auto', 'classification', 'regression', etc.)
            target: Target column name for tabular data
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            optimize: Whether to perform hyperparameter optimization
            deploy: Whether to create deployment artifacts
            domain_config: Domain-specific configuration parameters
            **kwargs: Additional configuration options
            
        Returns:
            Domain-specific workflow instance
            
        Raises:
            ConfigurationError: If workflow cannot be created
        """
        logger.debug(f"Creating workflow for data: {data_path}")
        
        # Detect data type
        try:
            data_type = detect_data_type(data_path)
            logger.info(f"Detected data type: {data_type.value}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to detect data type for '{data_path}': {e}\n"
                f"Please ensure the data path exists and contains valid data files."
            )
        
        # Get appropriate workflow class
        workflow_class = self._workflows.get(data_type)
        if workflow_class is None:
            supported_types = list(self._workflows.keys())
            raise ConfigurationError(
                f"No workflow available for data type: {data_type.value}\n"
                f"Supported data types: {[dt.value for dt in supported_types]}"
            )
        
        # Create workflow configuration
        config = WorkflowConfig(
            data_path=data_path,
            target=target,
            validation_split=validation_split,
            test_split=test_split,
            model=model,
            task=task,
            optimize=optimize,
            deploy=deploy,
            domain_config=domain_config or {},
            kwargs=kwargs
        )
        
        # Create and validate workflow
        try:
            workflow = workflow_class(config)
            workflow.validate_config()
            
            logger.info(f"Created {workflow_class.__name__} for {data_type.value} data")
            return workflow
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create {workflow_class.__name__}: {e}"
            )
    
    def create_vision_workflow(
        self,
        data_path: Union[str, Path],
        task: str = "auto",
        model: str = "auto",
        image_size: Optional[tuple] = None,
        augmentation: bool = True,
        **kwargs
    ) -> VisionWorkflow:
        """
        Create a computer vision workflow with vision-specific parameters.
        
        Args:
            data_path: Path to image data
            task: Vision task ('auto', 'image_classification', 'object_detection', etc.)
            model: Vision model ('auto', 'resnet18', 'yolo', etc.)
            image_size: Target image size as (height, width)
            augmentation: Whether to apply data augmentation
            **kwargs: Additional configuration options
            
        Returns:
            VisionWorkflow instance
        """
        domain_config = {
            'augmentation': augmentation
        }
        
        if image_size is not None:
            domain_config['image_size'] = image_size
        
        # Add any additional domain config from kwargs
        vision_keys = ['confidence_threshold', 'nms_threshold', 'augmentation_params']
        for key in vision_keys:
            if key in kwargs:
                domain_config[key] = kwargs.pop(key)
        
        return self.create_workflow(
            data_path=data_path,
            model=model,
            task=task,
            domain_config=domain_config,
            **kwargs
        )
    
    def create_nlp_workflow(
        self,
        data_path: Union[str, Path],
        task: str = "auto",
        model: str = "auto",
        max_length: Optional[int] = None,
        tokenizer: Optional[str] = None,
        **kwargs
    ) -> NLPWorkflow:
        """
        Create an NLP workflow with NLP-specific parameters.
        
        Args:
            data_path: Path to text data
            task: NLP task ('auto', 'text_classification', 'sentiment_analysis', etc.)
            model: NLP model ('auto', 'bert', 'gpt2', etc.)
            max_length: Maximum sequence length
            tokenizer: Tokenizer to use
            **kwargs: Additional configuration options
            
        Returns:
            NLPWorkflow instance
        """
        domain_config = {}
        
        if max_length is not None:
            domain_config['max_length'] = max_length
        
        if tokenizer is not None:
            domain_config['tokenizer'] = tokenizer
        
        # Add any additional domain config from kwargs
        nlp_keys = ['remove_stopwords', 'temperature', 'top_p', 'max_source_length', 'max_target_length']
        for key in nlp_keys:
            if key in kwargs:
                domain_config[key] = kwargs.pop(key)
        
        return self.create_workflow(
            data_path=data_path,
            model=model,
            task=task,
            domain_config=domain_config,
            **kwargs
        )
    
    def create_tabular_workflow(
        self,
        data_path: Union[str, Path],
        target: str,
        task: str = "auto",
        model: str = "auto",
        feature_engineering: bool = True,
        scaling: str = "standard",
        categorical_encoding: str = "onehot",
        **kwargs
    ) -> TabularWorkflow:
        """
        Create a tabular data workflow with tabular-specific parameters.
        
        Args:
            data_path: Path to tabular data (CSV, etc.)
            target: Target column name
            task: Tabular task ('auto', 'classification', 'regression', etc.)
            model: Tabular model ('auto', 'random_forest', 'xgboost', etc.)
            feature_engineering: Whether to apply feature engineering
            scaling: Scaling method ('standard', 'minmax', 'robust', 'none')
            categorical_encoding: Categorical encoding method ('onehot', 'label', 'target')
            **kwargs: Additional configuration options
            
        Returns:
            TabularWorkflow instance
        """
        domain_config = {
            'feature_engineering': feature_engineering,
            'scaling': scaling,
            'categorical_encoding': categorical_encoding
        }
        
        # Add any additional domain config from kwargs
        tabular_keys = [
            'missing_value_strategy', 'remove_outliers', 'feature_selection',
            'balance_classes', 'n_clusters', 'sequence_length', 'forecast_horizon'
        ]
        for key in tabular_keys:
            if key in kwargs:
                domain_config[key] = kwargs.pop(key)
        
        return self.create_workflow(
            data_path=data_path,
            target=target,
            model=model,
            task=task,
            domain_config=domain_config,
            **kwargs
        )
    
    def get_supported_data_types(self) -> list[DataType]:
        """
        Get list of supported data types.
        
        Returns:
            List of supported DataType enums
        """
        return list(self._workflows.keys())
    
    def get_workflow_info(self, data_type: DataType) -> Dict[str, Any]:
        """
        Get information about workflow for a specific data type.
        
        Args:
            data_type: Data type to get workflow info for
            
        Returns:
            Dictionary with workflow information
            
        Raises:
            ConfigurationError: If data type is not supported
        """
        workflow_class = self._workflows.get(data_type)
        if workflow_class is None:
            raise ConfigurationError(f"No workflow available for data type: {data_type.value}")
        
        # Create a temporary instance to get information
        temp_config = WorkflowConfig(data_path="dummy")
        temp_workflow = workflow_class(temp_config)
        
        return {
            'class_name': workflow_class.__name__,
            'supported_data_types': [dt.value for dt in temp_workflow.supported_data_types],
            'supported_tasks': [tt.value for tt in temp_workflow.supported_tasks],
            'default_models': {tt.value: model for tt, model in temp_workflow.default_models.items()}
        }


# Global factory instance
_global_factory = WorkflowFactory()


def create_workflow(
    data_path: Union[str, Path],
    model: str = "auto",
    task: str = "auto",
    target: Optional[str] = None,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    optimize: bool = True,
    deploy: bool = False,
    domain_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseWorkflow:
    """
    Create appropriate workflow using the global factory.
    
    Args:
        data_path: Path to data file or directory
        model: Model type to use ('auto' for automatic selection)
        task: Task type ('auto', 'classification', 'regression', etc.)
        target: Target column name for tabular data
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        optimize: Whether to perform hyperparameter optimization
        deploy: Whether to create deployment artifacts
        domain_config: Domain-specific configuration parameters
        **kwargs: Additional configuration options
        
    Returns:
        Domain-specific workflow instance
    """
    return _global_factory.create_workflow(
        data_path=data_path,
        model=model,
        task=task,
        target=target,
        validation_split=validation_split,
        test_split=test_split,
        optimize=optimize,
        deploy=deploy,
        domain_config=domain_config,
        **kwargs
    )


def get_workflow_factory() -> WorkflowFactory:
    """
    Get the global workflow factory instance.
    
    Returns:
        Global WorkflowFactory instance
    """
    return _global_factory