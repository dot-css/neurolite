"""
Computer vision workflow coordination for NeuroLite.

Implements task-specific workflow coordination for computer vision tasks
including image classification, object detection, and segmentation.
"""

from typing import List, Dict, Tuple, Any
from pathlib import Path

from .base import BaseWorkflow, WorkflowConfig
from ..core import get_logger, ConfigurationError
from ..data import (
    DataType, Dataset, detect_data_type, load_data, validate_data, 
    preprocess_data, clean_data, split_data
)
from ..models import TaskType, BaseModel, create_model
from ..training import TrainingEngine, get_default_training_config


logger = get_logger(__name__)


class VisionWorkflow(BaseWorkflow):
    """
    Workflow coordination for computer vision tasks.
    
    Handles image classification, object detection, and segmentation tasks
    with appropriate preprocessing, model selection, and training configuration.
    """
    
    @property
    def supported_data_types(self) -> List[DataType]:
        """Computer vision workflows support image data."""
        return [DataType.IMAGE]
    
    @property
    def supported_tasks(self) -> List[TaskType]:
        """Supported computer vision tasks."""
        return [
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.OBJECT_DETECTION,
            TaskType.SEMANTIC_SEGMENTATION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.CLASSIFICATION,  # Generic classification for images
        ]
    
    @property
    def default_models(self) -> Dict[TaskType, str]:
        """Default model mappings for vision tasks."""
        return {
            TaskType.IMAGE_CLASSIFICATION: "resnet18",
            TaskType.CLASSIFICATION: "resnet18",
            TaskType.OBJECT_DETECTION: "yolo",
            TaskType.SEMANTIC_SEGMENTATION: "unet",
            TaskType.INSTANCE_SEGMENTATION: "mask_rcnn"
        }
    
    def _load_and_validate_data(self) -> Tuple[Dataset, DataType]:
        """Load and validate image data."""
        self.logger.debug(f"Loading image data from {self.config.data_path}")
        
        # Detect data type
        data_type = detect_data_type(self.config.data_path)
        
        if data_type not in self.supported_data_types:
            raise ConfigurationError(
                f"VisionWorkflow does not support data type: {data_type.value}. "
                f"Supported types: {[dt.value for dt in self.supported_data_types]}"
            )
        
        # Load data
        dataset = load_data(self.config.data_path, data_type, target=self.config.target)
        self.logger.info(f"Loaded {len(dataset)} image samples")
        
        # Validate data quality
        validation_result = validate_data(dataset)
        if not validation_result.is_valid:
            self.logger.warning(f"Data validation issues: {validation_result.issues}")
            dataset = clean_data(dataset)
            self.logger.info("Applied automatic data cleaning")
        
        return dataset, data_type
    
    def _detect_task_type(self, dataset: Dataset, data_type: DataType) -> TaskType:
        """Detect or validate task type for vision data."""
        if self.config.task != "auto":
            task_type = TaskType(self.config.task)
            if task_type not in self.supported_tasks:
                raise ConfigurationError(
                    f"Task {self.config.task} not supported for computer vision. "
                    f"Supported tasks: {[t.value for t in self.supported_tasks]}"
                )
            return task_type
        
        # Auto-detect based on data characteristics
        self.logger.debug("Auto-detecting vision task type")
        
        # Check for object detection annotations
        if hasattr(dataset, 'has_bounding_boxes') and dataset.has_bounding_boxes:
            return TaskType.OBJECT_DETECTION
        
        # Check for segmentation masks
        if hasattr(dataset, 'has_segmentation_masks') and dataset.has_segmentation_masks:
            return TaskType.SEMANTIC_SEGMENTATION
        
        # Default to image classification
        return TaskType.IMAGE_CLASSIFICATION
    
    def _preprocess_data(self, dataset: Dataset, task_type: TaskType) -> Tuple[Dataset, Dict[str, Any]]:
        """Apply vision-specific preprocessing."""
        self.logger.debug(f"Applying vision preprocessing for task: {task_type.value}")
        
        # Get vision-specific preprocessing configuration
        preprocessing_config = self._get_vision_preprocessing_config(task_type)
        
        # Apply preprocessing
        processed_dataset = preprocess_data(dataset, task_type, config=preprocessing_config)
        
        # Split data
        data_splits = split_data(
            processed_dataset,
            train_ratio=1.0 - self.config.validation_split - self.config.test_split,
            val_ratio=self.config.validation_split,
            test_ratio=self.config.test_split,
            stratify=True  # Stratify for classification tasks
        )
        
        # Update dataset with splits
        processed_dataset.train = data_splits.train
        processed_dataset.validation = data_splits.validation
        processed_dataset.test = data_splits.test
        
        preprocessing_info = {
            'config': preprocessing_config,
            'train_samples': len(data_splits.train),
            'validation_samples': len(data_splits.validation),
            'test_samples': len(data_splits.test),
            'image_size': getattr(preprocessing_config, 'image_size', (224, 224)),
            'augmentation_applied': getattr(preprocessing_config, 'augmentation', True)
        }
        
        self.logger.info(f"Vision preprocessing completed: {preprocessing_info}")
        return processed_dataset, preprocessing_info
    
    def _get_vision_preprocessing_config(self, task_type: TaskType) -> Any:
        """Get vision-specific preprocessing configuration."""
        from ..data.preprocessor import PreprocessingConfig
        
        # Base configuration for vision tasks
        config = PreprocessingConfig()
        
        # Task-specific configurations
        if task_type in [TaskType.IMAGE_CLASSIFICATION, TaskType.CLASSIFICATION]:
            config.image_size = self.config.domain_config.get('image_size', (224, 224))
            config.normalize = True
            config.augmentation = self.config.domain_config.get('augmentation', True)
            config.augmentation_params = {
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'horizontal_flip': True,
                'zoom_range': 0.1
            }
        
        elif task_type == TaskType.OBJECT_DETECTION:
            config.image_size = self.config.domain_config.get('image_size', (416, 416))
            config.normalize = True
            config.augmentation = self.config.domain_config.get('augmentation', True)
            config.augmentation_params = {
                'rotation_range': 5,  # Less rotation for detection
                'horizontal_flip': True,
                'brightness_range': (0.8, 1.2),
                'preserve_bboxes': True
            }
        
        elif task_type in [TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION]:
            config.image_size = self.config.domain_config.get('image_size', (256, 256))
            config.normalize = True
            config.augmentation = self.config.domain_config.get('augmentation', True)
            config.augmentation_params = {
                'horizontal_flip': True,
                'rotation_range': 10,
                'preserve_masks': True
            }
        
        # Override with user-provided domain config
        for key, value in self.config.domain_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _select_and_create_model(self, task_type: TaskType, dataset: Dataset) -> BaseModel:
        """Select and create vision model."""
        if self.config.model == "auto":
            model_name = self.default_models.get(task_type)
            if not model_name:
                raise ConfigurationError(
                    f"No default model available for task: {task_type.value}"
                )
        else:
            model_name = self.config.model
        
        self.logger.info(f"Creating vision model: {model_name} for task: {task_type.value}")
        
        # Vision-specific model parameters
        model_params = {
            'num_classes': getattr(dataset.info, 'num_classes', 2),
            'input_shape': getattr(dataset.info, 'input_shape', (3, 224, 224)),
            **self.config.kwargs
        }
        
        # Task-specific model parameters
        if task_type == TaskType.OBJECT_DETECTION:
            model_params.update({
                'num_classes': getattr(dataset.info, 'num_classes', 80),  # COCO default
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4
            })
        
        elif task_type in [TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION]:
            model_params.update({
                'num_classes': getattr(dataset.info, 'num_classes', 21),  # Pascal VOC default
                'output_stride': 16
            })
        
        model = create_model(model_name, task_type, **model_params)
        return model
    
    def _train_model(
        self, 
        model: BaseModel, 
        dataset: Dataset, 
        task_type: TaskType
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train vision model with domain-specific configuration."""
        from ..training import TrainingEngine
        
        # Get vision-specific training configuration
        training_config = self._get_vision_training_config(task_type, len(dataset.train))
        
        # Override with user-provided parameters
        if self.config.epochs is not None:
            training_config.epochs = self.config.epochs
        if self.config.batch_size is not None:
            training_config.batch_size = self.config.batch_size
        if self.config.learning_rate is not None:
            training_config.learning_rate = self.config.learning_rate
        
        self.logger.info(f"Training vision model with config: {training_config}")
        
        # Create training engine and train
        training_engine = TrainingEngine()
        trained_model = training_engine.train(
            model=model,
            train_data=dataset.train,
            val_data=dataset.validation,
            config=training_config
        )
        
        training_info = {
            'config': training_config,
            'epochs_completed': len(trained_model.training_history.get('loss', [])),
            'best_validation_metric': max(trained_model.training_history.get('val_accuracy', [0])),
            'training_time': trained_model.metadata.training_time if trained_model.metadata else 0.0
        }
        
        return trained_model, training_info
    
    def _get_vision_training_config(self, task_type: TaskType, train_size: int) -> Any:
        """Get vision-specific training configuration."""
        from ..training.config import optimize_training_config
        
        config = optimize_training_config(
            task_type=task_type,
            data_type=DataType.IMAGE,
            num_samples=train_size
        )
        
        # Vision-specific training parameters
        if task_type in [TaskType.IMAGE_CLASSIFICATION, TaskType.CLASSIFICATION]:
            config.epochs = min(100, max(10, train_size // 100))
            config.batch_size = min(32, max(8, train_size // 100))
            config.learning_rate = 0.001
            config.optimizer = "adam"
            config.loss_function = "categorical_crossentropy"
            config.metrics = ["accuracy", "top_5_accuracy"]
            config.early_stopping = True
            config.patience = 10
        
        elif task_type == TaskType.OBJECT_DETECTION:
            config.epochs = min(200, max(20, train_size // 50))
            config.batch_size = min(16, max(4, train_size // 200))
            config.learning_rate = 0.0001
            config.optimizer = "adam"
            config.loss_function = "yolo_loss"
            config.metrics = ["map", "precision", "recall"]
            config.early_stopping = True
            config.patience = 15
        
        elif task_type in [TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION]:
            config.epochs = min(150, max(15, train_size // 75))
            config.batch_size = min(8, max(2, train_size // 300))
            config.learning_rate = 0.0001
            config.optimizer = "adam"
            config.loss_function = "dice_loss"
            config.metrics = ["dice_coefficient", "iou"]
            config.early_stopping = True
            config.patience = 12
        
        return config
    
    def _validate_domain_config(self) -> None:
        """Validate vision-specific configuration."""
        domain_config = self.config.domain_config
        
        # Validate image size
        if 'image_size' in domain_config:
            image_size = domain_config['image_size']
            if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
                raise ValueError("image_size must be a tuple/list of 2 integers")
            if not all(isinstance(x, int) and x > 0 for x in image_size):
                raise ValueError("image_size values must be positive integers")
        
        # Validate augmentation parameters
        if 'augmentation_params' in domain_config:
            aug_params = domain_config['augmentation_params']
            if not isinstance(aug_params, dict):
                raise ValueError("augmentation_params must be a dictionary")
        
        # Task-specific validation
        if self.config.task == TaskType.OBJECT_DETECTION.value:
            if 'confidence_threshold' in domain_config:
                conf_thresh = domain_config['confidence_threshold']
                if not (0.0 <= conf_thresh <= 1.0):
                    raise ValueError("confidence_threshold must be between 0.0 and 1.0")
            
            if 'nms_threshold' in domain_config:
                nms_thresh = domain_config['nms_threshold']
                if not (0.0 <= nms_thresh <= 1.0):
                    raise ValueError("nms_threshold must be between 0.0 and 1.0")