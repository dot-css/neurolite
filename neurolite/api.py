"""
Main API interface for NeuroLite.

Provides the primary user-facing functions for training and deploying models
with minimal code requirements.
"""

import os
import time
from typing import Union, Optional, Dict, Any, List
from pathlib import Path

from .core import (
    get_logger, NeuroLiteError, DataError, ModelError, TrainingError,
    ConfigurationError, validate_input, timer, get_config
)
from .data import (
    detect_data_type, load_data, validate_data, preprocess_data, 
    clean_data, split_data, DataType
)
from .models import get_model_registry, create_model, TaskType
from .training import TrainingEngine, TrainedModel, get_default_training_config
from .evaluation import evaluate_model
from .deployment import create_api_server, ModelExporter, ExportedModel
from .workflows import create_workflow, WorkflowResult

logger = get_logger(__name__)


def _validate_parameters(
    data: Union[str, Path],
    model: str,
    task: str,
    target: Optional[str],
    validation_split: float,
    test_split: float,
    optimize: bool,
    deploy: bool,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate and normalize input parameters.
    
    Args:
        data: Path to data file or directory
        model: Model type to use
        task: Task type
        target: Target column name for tabular data
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        optimize: Whether to perform hyperparameter optimization
        deploy: Whether to create deployment artifacts
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ConfigurationError: If parameters are invalid
    """
    logger.debug("Validating input parameters")
    
    # Validate data path
    data_path = Path(data)
    if not data_path.exists():
        raise ConfigurationError(
            f"Data path does not exist: {data_path}\n"
            f"Please check that the file or directory exists and is accessible."
        )
    
    # Validate splits
    if not (0.0 <= validation_split <= 1.0):
        raise ConfigurationError(
            f"validation_split must be between 0.0 and 1.0, got {validation_split}\n"
            f"Suggested values: 0.2 (20% for validation) or 0.15 (15% for validation)"
        )
    
    if not (0.0 <= test_split <= 1.0):
        raise ConfigurationError(
            f"test_split must be between 0.0 and 1.0, got {test_split}\n"
            f"Suggested values: 0.1 (10% for testing) or 0.2 (20% for testing)"
        )
    
    if validation_split + test_split >= 1.0:
        raise ConfigurationError(
            f"validation_split ({validation_split}) + test_split ({test_split}) "
            f"must be less than 1.0\n"
            f"Suggested: reduce splits so training data is at least 60% of total"
        )
    
    # Validate model parameter
    if model != "auto":
        registry = get_model_registry()
        available_models = registry.list_models()
        if model not in available_models:
            raise ConfigurationError(
                f"Unknown model type: {model}\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Use 'auto' for automatic model selection"
            )
    
    # Validate task parameter
    if task != "auto":
        valid_tasks = [t.value for t in TaskType]
        if task not in valid_tasks:
            raise ConfigurationError(
                f"Unknown task type: {task}\n"
                f"Available tasks: {', '.join(valid_tasks)}\n"
                f"Use 'auto' for automatic task detection"
            )
    
    return {
        'data_path': data_path,
        'model': model,
        'task': task,
        'target': target,
        'validation_split': validation_split,
        'test_split': test_split,
        'optimize': optimize,
        'deploy': deploy,
        'kwargs': kwargs
    }


def _detect_task_from_data(data_type: DataType, dataset_info: Any, task: str) -> TaskType:
    """
    Automatically detect task type from data characteristics.
    
    Args:
        data_type: Detected data type
        dataset_info: Information about the dataset
        task: User-specified task (may be 'auto')
        
    Returns:
        Detected TaskType
        
    Raises:
        ConfigurationError: If task cannot be determined
    """
    logger.debug(f"Detecting task type for data_type={data_type}, task={task}")
    
    if task != "auto":
        try:
            return TaskType(task)
        except ValueError:
            raise ConfigurationError(f"Invalid task type: {task}")
    
    # Auto-detect based on data type and characteristics
    if data_type == DataType.IMAGE:
        return TaskType.IMAGE_CLASSIFICATION
    elif data_type == DataType.TEXT:
        return TaskType.TEXT_CLASSIFICATION
    elif data_type == DataType.TABULAR:
        # Check if target is numeric or categorical
        if hasattr(dataset_info, 'target_type'):
            if dataset_info.target_type == 'numeric':
                return TaskType.REGRESSION
            else:
                return TaskType.CLASSIFICATION
        else:
            # Default to classification for tabular data
            return TaskType.CLASSIFICATION
    else:
        raise ConfigurationError(
            f"Cannot auto-detect task for data type: {data_type}\n"
            f"Please specify task explicitly using the 'task' parameter"
        )


def _select_model(model: str, data_type: DataType, task_type: TaskType) -> str:
    """
    Select appropriate model based on data type and task.
    
    Args:
        model: User-specified model (may be 'auto')
        data_type: Detected data type
        task_type: Detected task type
        
    Returns:
        Selected model name
        
    Raises:
        ModelError: If no suitable model is found
    """
    logger.debug(f"Selecting model for data_type={data_type}, task_type={task_type}")
    
    if model != "auto":
        return model
    
    registry = get_model_registry()
    
    # Auto-select based on data type and task
    if data_type == DataType.IMAGE:
        if task_type in [TaskType.IMAGE_CLASSIFICATION, TaskType.CLASSIFICATION]:
            return "resnet18"  # Default CNN for image classification
        elif task_type == TaskType.OBJECT_DETECTION:
            return "yolo"
    elif data_type == DataType.TEXT:
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.SENTIMENT_ANALYSIS]:
            return "bert"  # Default transformer for text classification
    elif data_type == DataType.TABULAR:
        if task_type == TaskType.REGRESSION:
            return "random_forest_regressor"
        elif task_type == TaskType.CLASSIFICATION:
            return "random_forest_classifier"
    
    # Fallback to first available model for the task
    available_models = registry.list_models(task_type)
    if available_models:
        selected = available_models[0]
        logger.warning(f"Using fallback model: {selected}")
        return selected
    
    raise ModelError(
        f"No suitable model found for data_type={data_type}, task_type={task_type}\n"
        f"Please specify a model explicitly or check available models"
    )


def train(
    data: Union[str, Path],
    model: str = "auto",
    task: str = "auto",
    target: Optional[str] = None,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    optimize: bool = True,
    deploy: bool = False,
    **kwargs
) -> TrainedModel:
    """
    Train a machine learning model with minimal configuration using domain-specific workflows.
    
    This is the main entry point for NeuroLite. It automatically detects the data type
    and uses the appropriate domain-specific workflow (Vision, NLP, or Tabular) to handle
    data loading, preprocessing, model selection, training, and evaluation.
    
    Args:
        data: Path to data file or directory
        model: Model type to use ('auto' for automatic selection)
        task: Task type ('auto', 'classification', 'regression', etc.)
        target: Target column name for tabular data
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        optimize: Whether to perform hyperparameter optimization
        deploy: Whether to create deployment artifacts
        **kwargs: Additional configuration options including domain-specific parameters
        
    Returns:
        TrainedModel instance ready for inference and deployment
        
    Raises:
        NeuroLiteError: If training fails
        
    Example:
        >>> # Computer Vision
        >>> model = train('data/images/', model='resnet18', task='image_classification')
        
        >>> # NLP
        >>> model = train('data/text.csv', model='bert', task='text_classification', target='label')
        
        >>> # Tabular Data
        >>> model = train('data/tabular.csv', model='random_forest', task='classification', target='target')
    """
    logger.info("Starting NeuroLite multi-domain training workflow")
    start_time = time.time()
    
    try:
        # Extract domain-specific configuration from kwargs
        domain_config = {}
        
        # Vision-specific parameters
        vision_params = ['image_size', 'augmentation', 'confidence_threshold', 'nms_threshold', 'augmentation_params']
        for param in vision_params:
            if param in kwargs:
                domain_config[param] = kwargs.pop(param)
        
        # NLP-specific parameters
        nlp_params = ['max_length', 'tokenizer', 'remove_stopwords', 'temperature', 'top_p', 
                     'max_source_length', 'max_target_length']
        for param in nlp_params:
            if param in kwargs:
                domain_config[param] = kwargs.pop(param)
        
        # Tabular-specific parameters
        tabular_params = ['feature_engineering', 'scaling', 'categorical_encoding', 'missing_value_strategy',
                         'remove_outliers', 'feature_selection', 'balance_classes', 'n_clusters',
                         'sequence_length', 'forecast_horizon']
        for param in tabular_params:
            if param in kwargs:
                domain_config[param] = kwargs.pop(param)
        
        # Create appropriate workflow based on data type
        logger.info("Creating domain-specific workflow")
        workflow = create_workflow(
            data_path=data,
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
        
        logger.info(f"Using {workflow.__class__.__name__} for data processing and training")
        
        # Execute the workflow
        result = workflow.execute()
        
        # Log workflow completion
        total_time = time.time() - start_time
        logger.info(f"Multi-domain training workflow completed successfully in {total_time:.2f} seconds")
        logger.info(f"Data type: {result.data_type.value}, Task type: {result.task_type.value}")
        logger.info(f"Training time: {result.training_info.get('training_time', 0.0):.2f}s")
        
        if result.evaluation_info.get('primary_metric') is not None:
            logger.info(f"Primary metric: {result.evaluation_info['primary_metric']:.4f}")
        
        return result.trained_model
        
    except Exception as e:
        logger.error(f"Multi-domain training workflow failed: {str(e)}")
        
        # Provide helpful error messages and suggestions
        if isinstance(e, (ConfigurationError, DataError, ModelError, TrainingError)):
            raise e
        else:
            # Wrap unexpected errors with helpful context
            raise NeuroLiteError(
                f"Unexpected error during multi-domain training: {str(e)}\n"
                f"This might be due to:\n"
                f"1. Incompatible data format or corrupted files\n"
                f"2. Insufficient system resources (memory/disk space)\n"
                f"3. Missing dependencies for the selected model or domain\n"
                f"4. Network issues when downloading model weights\n"
                f"5. Unsupported combination of data type, task, and model\n"
                f"Please check the logs for more details and try again."
            ) from e


def deploy(
    model: TrainedModel,
    format: str = "api",
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs
) -> Union[ExportedModel, Any]:
    """
    Deploy a trained model for inference.
    
    Args:
        model: Trained model to deploy
        format: Deployment format ('api', 'onnx', 'tflite', etc.)
        host: Host address for API deployment
        port: Port for API deployment
        **kwargs: Additional deployment options
        
    Returns:
        Deployed model instance (API server or exported model)
        
    Raises:
        NeuroLiteError: If deployment fails
        
    Example:
        >>> deployed = deploy(model, format='api', port=8080)
        >>> # Model now available at http://localhost:8080
    """
    logger.info(f"Deploying model in format: {format}")
    
    try:
        if format == "api":
            # Create API server
            api_server = create_api_server(
                model, 
                host=host, 
                port=port, 
                **kwargs
            )
            logger.info(f"API server created at http://{host}:{port}")
            return api_server
            
        else:
            # Export model to specified format
            exporter = ModelExporter()
            exported_model = exporter.export(model, format, **kwargs)
            logger.info(f"Model exported to {format} format")
            return exported_model
            
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise NeuroLiteError(
            f"Failed to deploy model in {format} format: {str(e)}\n"
            f"This might be due to:\n"
            f"1. Unsupported export format for this model type\n"
            f"2. Missing dependencies for the target format\n"
            f"3. Insufficient permissions to create server/files\n"
            f"4. Port already in use (for API deployment)\n"
            f"Please check the logs for more details."
        ) from e