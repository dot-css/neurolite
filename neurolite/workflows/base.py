"""
Base workflow interface for domain-specific task coordination.

Provides abstract base classes and common data structures for implementing
domain-specific workflows in NeuroLite.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import time

from ..core import get_logger
from ..data import DataType, Dataset
from ..models import TaskType, BaseModel
from ..training import TrainedModel


logger = get_logger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for domain-specific workflows."""
    
    # Data configuration
    data_path: Union[str, Path]
    target: Optional[str] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Model configuration
    model: str = "auto"
    task: str = "auto"
    
    # Training configuration
    optimize: bool = True
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    
    # Deployment configuration
    deploy: bool = False
    export_format: str = "api"
    
    # Domain-specific configuration
    domain_config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional parameters
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of a domain-specific workflow execution."""
    
    trained_model: TrainedModel
    data_type: DataType
    task_type: TaskType
    execution_time: float
    preprocessing_info: Dict[str, Any]
    training_info: Dict[str, Any]
    evaluation_info: Dict[str, Any]
    deployment_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWorkflow(ABC):
    """
    Abstract base class for domain-specific workflows.
    
    Each domain (vision, NLP, tabular) implements this interface to provide
    specialized preprocessing, model selection, and training coordination
    while maintaining consistent API patterns.
    """
    
    def __init__(self, config: WorkflowConfig):
        """
        Initialize the workflow with configuration.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        
    @property
    @abstractmethod
    def supported_data_types(self) -> List[DataType]:
        """Get list of data types supported by this workflow."""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[TaskType]:
        """Get list of task types supported by this workflow."""
        pass
    
    @property
    @abstractmethod
    def default_models(self) -> Dict[TaskType, str]:
        """Get default model mappings for each supported task."""
        pass
    
    def execute(self) -> WorkflowResult:
        """
        Execute the complete workflow.
        
        Returns:
            WorkflowResult containing all execution information
        """
        self.logger.info(f"Starting {self.__class__.__name__} execution")
        start_time = time.time()
        
        try:
            # Step 1: Load and validate data
            self.logger.info("Step 1/6: Loading and validating data")
            dataset, data_type = self._load_and_validate_data()
            
            # Step 2: Detect task type
            self.logger.info("Step 2/6: Detecting task type")
            task_type = self._detect_task_type(dataset, data_type)
            
            # Step 3: Domain-specific preprocessing
            self.logger.info("Step 3/6: Applying domain-specific preprocessing")
            processed_dataset, preprocessing_info = self._preprocess_data(dataset, task_type)
            
            # Step 4: Model selection and creation
            self.logger.info("Step 4/6: Selecting and creating model")
            model = self._select_and_create_model(task_type, processed_dataset)
            
            # Step 5: Training with domain-specific configuration
            self.logger.info("Step 5/6: Training model")
            trained_model, training_info = self._train_model(model, processed_dataset, task_type)
            
            # Step 6: Evaluation and optional deployment
            self.logger.info("Step 6/6: Evaluating model and creating deployment artifacts")
            evaluation_info = self._evaluate_model(trained_model, processed_dataset)
            deployment_info = self._handle_deployment(trained_model) if self.config.deploy else None
            
            execution_time = time.time() - start_time
            
            result = WorkflowResult(
                trained_model=trained_model,
                data_type=data_type,
                task_type=task_type,
                execution_time=execution_time,
                preprocessing_info=preprocessing_info,
                training_info=training_info,
                evaluation_info=evaluation_info,
                deployment_info=deployment_info,
                metadata={
                    'workflow_class': self.__class__.__name__,
                    'config': self.config,
                    'timestamp': time.time()
                }
            )
            
            self.logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    @abstractmethod
    def _load_and_validate_data(self) -> Tuple[Dataset, DataType]:
        """
        Load and validate data for this domain.
        
        Returns:
            Tuple of (dataset, data_type)
        """
        pass
    
    @abstractmethod
    def _detect_task_type(self, dataset: Dataset, data_type: DataType) -> TaskType:
        """
        Detect or validate task type for this domain.
        
        Args:
            dataset: Loaded dataset
            data_type: Detected data type
            
        Returns:
            Detected or validated task type
        """
        pass
    
    @abstractmethod
    def _preprocess_data(self, dataset: Dataset, task_type: TaskType) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Apply domain-specific preprocessing.
        
        Args:
            dataset: Raw dataset
            task_type: Detected task type
            
        Returns:
            Tuple of (processed_dataset, preprocessing_info)
        """
        pass
    
    @abstractmethod
    def _select_and_create_model(self, task_type: TaskType, dataset: Dataset) -> BaseModel:
        """
        Select and create appropriate model for this domain and task.
        
        Args:
            task_type: Task type
            dataset: Processed dataset
            
        Returns:
            Created model instance
        """
        pass
    
    @abstractmethod
    def _train_model(
        self, 
        model: BaseModel, 
        dataset: Dataset, 
        task_type: TaskType
    ) -> Tuple[TrainedModel, Dict[str, Any]]:
        """
        Train model with domain-specific configuration.
        
        Args:
            model: Model instance to train
            dataset: Processed dataset
            task_type: Task type
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        pass
    
    def _evaluate_model(self, trained_model: TrainedModel, dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate trained model (common implementation).
        
        Args:
            trained_model: Trained model
            dataset: Dataset with test split
            
        Returns:
            Evaluation information
        """
        from ..evaluation import evaluate_model
        
        try:
            if hasattr(dataset, 'test') and len(dataset.test) > 0:
                evaluation_results = evaluate_model(trained_model, dataset.test)
                return {
                    'metrics': evaluation_results.metrics,
                    'primary_metric': evaluation_results.primary_metric,
                    'execution_time': evaluation_results.execution_time
                }
            else:
                self.logger.warning("No test data available for evaluation")
                return {'metrics': {}, 'primary_metric': None, 'execution_time': 0.0}
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            return {'metrics': {}, 'primary_metric': None, 'execution_time': 0.0, 'error': str(e)}
    
    def _handle_deployment(self, trained_model: TrainedModel) -> Dict[str, Any]:
        """
        Handle model deployment (common implementation).
        
        Args:
            trained_model: Trained model to deploy
            
        Returns:
            Deployment information
        """
        from ..deployment import create_api_server, ModelExporter
        
        try:
            deployment_info = {'format': self.config.export_format}
            
            if self.config.export_format == "api":
                api_server = create_api_server(trained_model)
                deployment_info.update({
                    'api_server': api_server,
                    'endpoint': f"http://localhost:8000",
                    'status': 'ready'
                })
            else:
                exporter = ModelExporter()
                exported_model = exporter.export(trained_model, self.config.export_format)
                deployment_info.update({
                    'exported_model': exported_model,
                    'export_path': exported_model.path if hasattr(exported_model, 'path') else None,
                    'status': 'exported'
                })
            
            return deployment_info
            
        except Exception as e:
            self.logger.warning(f"Deployment failed: {e}")
            return {'format': self.config.export_format, 'status': 'failed', 'error': str(e)}
    
    def validate_config(self) -> None:
        """
        Validate workflow configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate data path
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Validate splits
        if not (0.0 <= self.config.validation_split <= 1.0):
            raise ValueError(f"validation_split must be between 0.0 and 1.0")
        
        if not (0.0 <= self.config.test_split <= 1.0):
            raise ValueError(f"test_split must be between 0.0 and 1.0")
        
        if self.config.validation_split + self.config.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be less than 1.0")
        
        # Validate task type
        if self.config.task != "auto":
            try:
                task_type = TaskType(self.config.task)
                if task_type not in self.supported_tasks:
                    raise ValueError(
                        f"Task {self.config.task} not supported by {self.__class__.__name__}. "
                        f"Supported tasks: {[t.value for t in self.supported_tasks]}"
                    )
            except ValueError as e:
                if "not supported by" not in str(e):
                    raise ValueError(f"Invalid task type: {self.config.task}")
                raise
        
        # Domain-specific validation
        self._validate_domain_config()
    
    def _validate_domain_config(self) -> None:
        """
        Validate domain-specific configuration.
        
        Subclasses can override this method to add domain-specific validation.
        """
        pass