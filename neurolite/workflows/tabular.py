"""
Tabular data workflow coordination for NeuroLite.

Implements task-specific workflow coordination for tabular data tasks
including classification, regression, and clustering with feature engineering
and appropriate model selection.
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


class TabularWorkflow(BaseWorkflow):
    """
    Workflow coordination for tabular data tasks.
    
    Handles classification, regression, and clustering tasks with appropriate
    feature engineering, model selection, and training configuration.
    """
    
    @property
    def supported_data_types(self) -> List[DataType]:
        """Tabular workflows support tabular data."""
        return [DataType.TABULAR]
    
    @property
    def supported_tasks(self) -> List[TaskType]:
        """Supported tabular data tasks."""
        return [
            TaskType.CLASSIFICATION,
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.LINEAR_REGRESSION,
            TaskType.POLYNOMIAL_REGRESSION,
            TaskType.CLUSTERING,
            TaskType.TIME_SERIES_FORECASTING
        ]
    
    @property
    def default_models(self) -> Dict[TaskType, str]:
        """Default model mappings for tabular tasks."""
        return {
            TaskType.CLASSIFICATION: "random_forest_classifier",
            TaskType.BINARY_CLASSIFICATION: "random_forest_classifier",
            TaskType.MULTICLASS_CLASSIFICATION: "random_forest_classifier",
            TaskType.MULTILABEL_CLASSIFICATION: "random_forest_classifier",
            TaskType.REGRESSION: "random_forest_regressor",
            TaskType.LINEAR_REGRESSION: "linear_regression",
            TaskType.POLYNOMIAL_REGRESSION: "polynomial_regression",
            TaskType.CLUSTERING: "kmeans",
            TaskType.TIME_SERIES_FORECASTING: "lstm_forecaster"
        }
    
    def _load_and_validate_data(self) -> Tuple[Dataset, DataType]:
        """Load and validate tabular data."""
        self.logger.debug(f"Loading tabular data from {self.config.data_path}")
        
        # Detect data type
        data_type = detect_data_type(self.config.data_path)
        
        if data_type not in self.supported_data_types:
            raise ConfigurationError(
                f"TabularWorkflow does not support data type: {data_type.value}. "
                f"Supported types: {[dt.value for dt in self.supported_data_types]}"
            )
        
        # Load data
        dataset = load_data(self.config.data_path, data_type, target=self.config.target)
        self.logger.info(f"Loaded {len(dataset)} tabular samples with {dataset.info.num_features} features")
        
        # Validate data quality
        validation_result = validate_data(dataset)
        if not validation_result.is_valid:
            self.logger.warning(f"Data validation issues: {validation_result.issues}")
            dataset = clean_data(dataset)
            self.logger.info("Applied automatic data cleaning")
        
        return dataset, data_type
    
    def _detect_task_type(self, dataset: Dataset, data_type: DataType) -> TaskType:
        """Detect or validate task type for tabular data."""
        if self.config.task != "auto":
            task_type = TaskType(self.config.task)
            if task_type not in self.supported_tasks:
                raise ConfigurationError(
                    f"Task {self.config.task} not supported for tabular data. "
                    f"Supported tasks: {[t.value for t in self.supported_tasks]}"
                )
            return task_type
        
        # Auto-detect based on data characteristics
        self.logger.debug("Auto-detecting tabular task type")
        
        # Check if target column exists
        if not hasattr(dataset, 'target') or dataset.target is None:
            return TaskType.CLUSTERING
        
        # Check target type for supervised tasks
        if hasattr(dataset.info, 'target_type'):
            if dataset.info.target_type == 'numeric':
                return TaskType.REGRESSION
            elif dataset.info.target_type == 'categorical':
                num_classes = getattr(dataset.info, 'num_classes', 2)
                if num_classes == 2:
                    return TaskType.BINARY_CLASSIFICATION
                else:
                    return TaskType.MULTICLASS_CLASSIFICATION
        
        # Check for time series data
        if hasattr(dataset.info, 'has_time_column') and dataset.info.has_time_column:
            return TaskType.TIME_SERIES_FORECASTING
        
        # Default to classification
        return TaskType.CLASSIFICATION
    
    def _preprocess_data(self, dataset: Dataset, task_type: TaskType) -> Tuple[Dataset, Dict[str, Any]]:
        """Apply tabular-specific preprocessing with feature engineering."""
        self.logger.debug(f"Applying tabular preprocessing for task: {task_type.value}")
        
        # Get tabular-specific preprocessing configuration
        preprocessing_config = self._get_tabular_preprocessing_config(task_type)
        
        # Apply preprocessing with feature engineering
        processed_dataset = preprocess_data(dataset, task_type, config=preprocessing_config)
        
        # Split data (no stratification for clustering)
        stratify = task_type not in [TaskType.CLUSTERING, TaskType.TIME_SERIES_FORECASTING]
        
        data_splits = split_data(
            processed_dataset,
            train_ratio=1.0 - self.config.validation_split - self.config.test_split,
            val_ratio=self.config.validation_split,
            test_ratio=self.config.test_split,
            stratify=stratify
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
            'num_features': getattr(processed_dataset.info, 'num_features', 0),
            'feature_engineering_applied': getattr(preprocessing_config, 'feature_engineering', True),
            'scaling_method': getattr(preprocessing_config, 'scaling', 'standard'),
            'categorical_encoding': getattr(preprocessing_config, 'categorical_encoding', 'onehot')
        }
        
        self.logger.info(f"Tabular preprocessing completed: {preprocessing_info}")
        return processed_dataset, preprocessing_info
    
    def _get_tabular_preprocessing_config(self, task_type: TaskType) -> Any:
        """Get tabular-specific preprocessing configuration."""
        from ..data.preprocessor import PreprocessingConfig
        
        # Base configuration for tabular tasks
        config = PreprocessingConfig()
        
        # Common tabular preprocessing
        config.handle_missing_values = True
        config.missing_value_strategy = self.config.domain_config.get('missing_value_strategy', 'median')
        config.scaling = self.config.domain_config.get('scaling', 'standard')
        config.categorical_encoding = self.config.domain_config.get('categorical_encoding', 'onehot')
        config.feature_engineering = self.config.domain_config.get('feature_engineering', True)
        config.remove_outliers = self.config.domain_config.get('remove_outliers', False)
        config.feature_selection = self.config.domain_config.get('feature_selection', True)
        
        # Task-specific configurations
        if task_type in [
            TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION, 
            TaskType.MULTICLASS_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION
        ]:
            config.balance_classes = self.config.domain_config.get('balance_classes', False)
            config.feature_selection_method = 'chi2'
            config.max_features = self.config.domain_config.get('max_features', 100)
            
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION, TaskType.POLYNOMIAL_REGRESSION]:
            config.target_transformation = self.config.domain_config.get('target_transformation', None)
            config.feature_selection_method = 'f_regression'
            config.max_features = self.config.domain_config.get('max_features', 50)
            config.polynomial_features = task_type == TaskType.POLYNOMIAL_REGRESSION
            
        elif task_type == TaskType.CLUSTERING:
            config.scaling = 'standard'  # Important for clustering
            config.dimensionality_reduction = self.config.domain_config.get('dimensionality_reduction', 'pca')
            config.n_components = self.config.domain_config.get('n_components', 10)
            
        elif task_type == TaskType.TIME_SERIES_FORECASTING:
            config.create_time_features = True
            config.lag_features = self.config.domain_config.get('lag_features', [1, 2, 3, 7, 14])
            config.rolling_features = self.config.domain_config.get('rolling_features', [7, 14, 30])
            config.seasonal_decomposition = self.config.domain_config.get('seasonal_decomposition', True)
        
        # Override with user-provided domain config
        for key, value in self.config.domain_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _select_and_create_model(self, task_type: TaskType, dataset: Dataset) -> BaseModel:
        """Select and create tabular model."""
        if self.config.model == "auto":
            model_name = self.default_models.get(task_type)
            if not model_name:
                raise ConfigurationError(
                    f"No default model available for task: {task_type.value}"
                )
        else:
            model_name = self.config.model
        
        self.logger.info(f"Creating tabular model: {model_name} for task: {task_type.value}")
        
        # Tabular-specific model parameters
        model_params = {
            'n_features': getattr(dataset.info, 'num_features', 10),
            **self.config.kwargs
        }
        
        # Task-specific model parameters
        if task_type in [
            TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION, 
            TaskType.MULTICLASS_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION
        ]:
            model_params.update({
                'n_classes': getattr(dataset.info, 'num_classes', 2),
                'class_weight': 'balanced' if self.config.domain_config.get('balance_classes', False) else None,
                'random_state': 42
            })
            
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION, TaskType.POLYNOMIAL_REGRESSION]:
            model_params.update({
                'normalize': True,
                'random_state': 42
            })
            
        elif task_type == TaskType.CLUSTERING:
            model_params.update({
                'n_clusters': self.config.domain_config.get('n_clusters', 'auto'),
                'random_state': 42
            })
            
        elif task_type == TaskType.TIME_SERIES_FORECASTING:
            model_params.update({
                'sequence_length': self.config.domain_config.get('sequence_length', 30),
                'forecast_horizon': self.config.domain_config.get('forecast_horizon', 1),
                'hidden_size': 64,
                'num_layers': 2
            })
        
        model = create_model(model_name, task_type, **model_params)
        return model
    
    def _train_model(
        self, 
        model: BaseModel, 
        dataset: Dataset, 
        task_type: TaskType
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train tabular model with domain-specific configuration."""
        from ..training import TrainingEngine
        
        # Get tabular-specific training configuration
        training_config = self._get_tabular_training_config(task_type, len(dataset.train))
        
        # Override with user-provided parameters
        if self.config.epochs is not None:
            training_config.epochs = self.config.epochs
        if self.config.batch_size is not None:
            training_config.batch_size = self.config.batch_size
        if self.config.learning_rate is not None:
            training_config.learning_rate = self.config.learning_rate
        
        self.logger.info(f"Training tabular model with config: {training_config}")
        
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
            'training_time': trained_model.metadata.training_time if trained_model.metadata else 0.0,
            'feature_importance': getattr(trained_model.model, 'feature_importances_', None)
        }
        
        return trained_model, training_info
    
    def _get_tabular_training_config(self, task_type: TaskType, train_size: int) -> Any:
        """Get tabular-specific training configuration."""
        from ..training.config import optimize_training_config
        
        config = optimize_training_config(
            task_type=task_type,
            data_type=DataType.TABULAR,
            num_samples=train_size
        )
        
        # Tabular-specific training parameters
        if task_type in [
            TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION, 
            TaskType.MULTICLASS_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION
        ]:
            # Most tabular models don't need many epochs
            config.epochs = 1  # For tree-based models
            config.batch_size = min(1000, train_size)  # Process all data at once for sklearn
            config.learning_rate = 0.1
            config.optimizer = "auto"  # Let sklearn handle optimization
            config.loss_function = "auto"
            config.metrics = ["accuracy", "precision", "recall", "f1_score"]
            config.early_stopping = False  # Not applicable for most sklearn models
            config.cross_validation = True
            config.cv_folds = 5
            
        elif task_type in [TaskType.REGRESSION, TaskType.LINEAR_REGRESSION, TaskType.POLYNOMIAL_REGRESSION]:
            config.epochs = 1
            config.batch_size = min(1000, train_size)
            config.learning_rate = 0.1
            config.optimizer = "auto"
            config.loss_function = "auto"
            config.metrics = ["mae", "mse", "r2_score"]
            config.early_stopping = False
            config.cross_validation = True
            config.cv_folds = 5
            
        elif task_type == TaskType.CLUSTERING:
            config.epochs = 1
            config.batch_size = train_size  # Process all data
            config.optimizer = "auto"
            config.metrics = ["silhouette_score", "calinski_harabasz_score"]
            config.early_stopping = False
            config.cross_validation = False  # Not applicable for clustering
            
        elif task_type == TaskType.TIME_SERIES_FORECASTING:
            config.epochs = min(100, max(20, train_size // 100))
            config.batch_size = min(64, max(16, train_size // 50))
            config.learning_rate = 0.001
            config.optimizer = "adam"
            config.loss_function = "mse"
            config.metrics = ["mae", "mape", "rmse"]
            config.early_stopping = True
            config.patience = 10
        
        return config
    
    def _validate_domain_config(self) -> None:
        """Validate tabular-specific configuration."""
        domain_config = self.config.domain_config
        
        # Validate missing value strategy
        if 'missing_value_strategy' in domain_config:
            strategy = domain_config['missing_value_strategy']
            valid_strategies = ['mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill']
            if strategy not in valid_strategies:
                raise ValueError(f"missing_value_strategy must be one of {valid_strategies}")
        
        # Validate scaling method
        if 'scaling' in domain_config:
            scaling = domain_config['scaling']
            valid_scaling = ['standard', 'minmax', 'robust', 'none']
            if scaling not in valid_scaling:
                raise ValueError(f"scaling must be one of {valid_scaling}")
        
        # Validate categorical encoding
        if 'categorical_encoding' in domain_config:
            encoding = domain_config['categorical_encoding']
            valid_encodings = ['onehot', 'label', 'target', 'binary']
            if encoding not in valid_encodings:
                raise ValueError(f"categorical_encoding must be one of {valid_encodings}")
        
        # Task-specific validation
        if self.config.task == TaskType.CLUSTERING.value:
            if 'n_clusters' in domain_config:
                n_clusters = domain_config['n_clusters']
                if n_clusters != 'auto' and (not isinstance(n_clusters, int) or n_clusters < 2):
                    raise ValueError("n_clusters must be 'auto' or an integer >= 2")
        
        if self.config.task == TaskType.TIME_SERIES_FORECASTING.value:
            if 'sequence_length' in domain_config:
                seq_len = domain_config['sequence_length']
                if not isinstance(seq_len, int) or seq_len < 1:
                    raise ValueError("sequence_length must be a positive integer")
            
            if 'forecast_horizon' in domain_config:
                horizon = domain_config['forecast_horizon']
                if not isinstance(horizon, int) or horizon < 1:
                    raise ValueError("forecast_horizon must be a positive integer")
        
        # Validate feature selection parameters
        if 'max_features' in domain_config:
            max_features = domain_config['max_features']
            if not isinstance(max_features, int) or max_features < 1:
                raise ValueError("max_features must be a positive integer")