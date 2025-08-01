"""
NLP workflow coordination for NeuroLite.

Implements task-specific workflow coordination for natural language processing tasks
including text classification, sentiment analysis, and text generation.
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


class NLPWorkflow(BaseWorkflow):
    """
    Workflow coordination for natural language processing tasks.
    
    Handles text classification, sentiment analysis, text generation, and other
    NLP tasks with appropriate preprocessing, model selection, and training configuration.
    """
    
    @property
    def supported_data_types(self) -> List[DataType]:
        """NLP workflows support text data."""
        return [DataType.TEXT]
    
    @property
    def supported_tasks(self) -> List[TaskType]:
        """Supported NLP tasks."""
        return [
            TaskType.TEXT_CLASSIFICATION,
            TaskType.SENTIMENT_ANALYSIS,
            TaskType.TEXT_GENERATION,
            TaskType.LANGUAGE_MODELING,
            TaskType.SEQUENCE_TO_SEQUENCE,
            TaskType.NAMED_ENTITY_RECOGNITION,
            TaskType.CLASSIFICATION,  # Generic classification for text
        ]
    
    @property
    def default_models(self) -> Dict[TaskType, str]:
        """Default model mappings for NLP tasks."""
        return {
            TaskType.TEXT_CLASSIFICATION: "bert",
            TaskType.CLASSIFICATION: "bert",
            TaskType.SENTIMENT_ANALYSIS: "bert",
            TaskType.TEXT_GENERATION: "gpt2",
            TaskType.LANGUAGE_MODELING: "gpt2",
            TaskType.SEQUENCE_TO_SEQUENCE: "t5",
            TaskType.NAMED_ENTITY_RECOGNITION: "bert_ner"
        }
    
    def _load_and_validate_data(self) -> Tuple[Dataset, DataType]:
        """Load and validate text data."""
        self.logger.debug(f"Loading text data from {self.config.data_path}")
        
        # Detect data type
        data_type = detect_data_type(self.config.data_path)
        
        if data_type not in self.supported_data_types:
            raise ConfigurationError(
                f"NLPWorkflow does not support data type: {data_type.value}. "
                f"Supported types: {[dt.value for dt in self.supported_data_types]}"
            )
        
        # Load data
        dataset = load_data(self.config.data_path, data_type, target=self.config.target)
        self.logger.info(f"Loaded {len(dataset)} text samples")
        
        # Validate data quality
        validation_result = validate_data(dataset)
        if not validation_result.is_valid:
            self.logger.warning(f"Data validation issues: {validation_result.issues}")
            dataset = clean_data(dataset)
            self.logger.info("Applied automatic data cleaning")
        
        return dataset, data_type
    
    def _detect_task_type(self, dataset: Dataset, data_type: DataType) -> TaskType:
        """Detect or validate task type for NLP data."""
        if self.config.task != "auto":
            task_type = TaskType(self.config.task)
            if task_type not in self.supported_tasks:
                raise ConfigurationError(
                    f"Task {self.config.task} not supported for NLP. "
                    f"Supported tasks: {[t.value for t in self.supported_tasks]}"
                )
            return task_type
        
        # Auto-detect based on data characteristics
        self.logger.debug("Auto-detecting NLP task type")
        
        # Check for sentiment labels
        if hasattr(dataset, 'has_sentiment_labels') and dataset.has_sentiment_labels:
            return TaskType.SENTIMENT_ANALYSIS
        
        # Check for sequence-to-sequence structure
        if hasattr(dataset, 'has_input_output_pairs') and dataset.has_input_output_pairs:
            return TaskType.SEQUENCE_TO_SEQUENCE
        
        # Check for generation task (no labels, just text)
        if not hasattr(dataset, 'labels') or dataset.labels is None:
            return TaskType.TEXT_GENERATION
        
        # Default to text classification
        return TaskType.TEXT_CLASSIFICATION
    
    def _preprocess_data(self, dataset: Dataset, task_type: TaskType) -> Tuple[Dataset, Dict[str, Any]]:
        """Apply NLP-specific preprocessing."""
        self.logger.debug(f"Applying NLP preprocessing for task: {task_type.value}")
        
        # Get NLP-specific preprocessing configuration
        preprocessing_config = self._get_nlp_preprocessing_config(task_type)
        
        # Apply preprocessing
        processed_dataset = preprocess_data(dataset, task_type, config=preprocessing_config)
        
        # Split data
        data_splits = split_data(
            processed_dataset,
            train_ratio=1.0 - self.config.validation_split - self.config.test_split,
            val_ratio=self.config.validation_split,
            test_ratio=self.config.test_split,
            stratify=task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.SENTIMENT_ANALYSIS]
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
            'max_sequence_length': getattr(preprocessing_config, 'max_length', 512),
            'tokenizer': getattr(preprocessing_config, 'tokenizer', 'bert-base-uncased'),
            'vocabulary_size': getattr(processed_dataset.info, 'vocab_size', None)
        }
        
        self.logger.info(f"NLP preprocessing completed: {preprocessing_info}")
        return processed_dataset, preprocessing_info
    
    def _get_nlp_preprocessing_config(self, task_type: TaskType) -> Any:
        """Get NLP-specific preprocessing configuration."""
        from ..data.preprocessor import PreprocessingConfig
        
        # Base configuration for NLP tasks
        config = PreprocessingConfig()
        
        # Task-specific configurations
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.SENTIMENT_ANALYSIS, TaskType.CLASSIFICATION]:
            config.max_length = self.config.domain_config.get('max_length', 512)
            config.tokenizer = self.config.domain_config.get('tokenizer', 'bert-base-uncased')
            config.lowercase = True
            config.remove_stopwords = self.config.domain_config.get('remove_stopwords', False)
            config.remove_punctuation = False
            config.padding = 'max_length'
            config.truncation = True
        
        elif task_type in [TaskType.TEXT_GENERATION, TaskType.LANGUAGE_MODELING]:
            config.max_length = self.config.domain_config.get('max_length', 1024)
            config.tokenizer = self.config.domain_config.get('tokenizer', 'gpt2')
            config.lowercase = False
            config.remove_stopwords = False
            config.remove_punctuation = False
            config.padding = False
            config.truncation = True
        
        elif task_type == TaskType.SEQUENCE_TO_SEQUENCE:
            config.max_length = self.config.domain_config.get('max_length', 512)
            config.tokenizer = self.config.domain_config.get('tokenizer', 't5-base')
            config.lowercase = False
            config.remove_stopwords = False
            config.remove_punctuation = False
            config.padding = 'max_length'
            config.truncation = True
            config.add_special_tokens = True
        
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            config.max_length = self.config.domain_config.get('max_length', 256)
            config.tokenizer = self.config.domain_config.get('tokenizer', 'bert-base-uncased')
            config.lowercase = False
            config.remove_stopwords = False
            config.remove_punctuation = False
            config.padding = 'max_length'
            config.truncation = True
            config.word_level_tokenization = True
        
        # Override with user-provided domain config
        for key, value in self.config.domain_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _select_and_create_model(self, task_type: TaskType, dataset: Dataset) -> BaseModel:
        """Select and create NLP model."""
        if self.config.model == "auto":
            model_name = self.default_models.get(task_type)
            if not model_name:
                raise ConfigurationError(
                    f"No default model available for task: {task_type.value}"
                )
        else:
            model_name = self.config.model
        
        self.logger.info(f"Creating NLP model: {model_name} for task: {task_type.value}")
        
        # NLP-specific model parameters
        model_params = {
            'num_classes': getattr(dataset.info, 'num_classes', 2),
            'vocab_size': getattr(dataset.info, 'vocab_size', 30522),  # BERT default
            'max_length': self.config.domain_config.get('max_length', 512),
            **self.config.kwargs
        }
        
        # Task-specific model parameters
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.SENTIMENT_ANALYSIS]:
            model_params.update({
                'dropout_rate': 0.1,
                'hidden_size': 768,
                'num_attention_heads': 12
            })
        
        elif task_type in [TaskType.TEXT_GENERATION, TaskType.LANGUAGE_MODELING]:
            model_params.update({
                'vocab_size': getattr(dataset.info, 'vocab_size', 50257),  # GPT-2 default
                'max_length': self.config.domain_config.get('max_length', 1024),
                'temperature': 0.8,
                'top_p': 0.9
            })
        
        elif task_type == TaskType.SEQUENCE_TO_SEQUENCE:
            model_params.update({
                'encoder_layers': 6,
                'decoder_layers': 6,
                'max_source_length': self.config.domain_config.get('max_source_length', 512),
                'max_target_length': self.config.domain_config.get('max_target_length', 512)
            })
        
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            model_params.update({
                'num_labels': getattr(dataset.info, 'num_entity_types', 9),  # CoNLL default
                'label_scheme': 'BIO'
            })
        
        model = create_model(model_name, task_type, **model_params)
        return model
    
    def _train_model(
        self, 
        model: BaseModel, 
        dataset: Dataset, 
        task_type: TaskType
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train NLP model with domain-specific configuration."""
        from ..training import TrainingEngine
        
        # Get NLP-specific training configuration
        training_config = self._get_nlp_training_config(task_type, len(dataset.train))
        
        # Override with user-provided parameters
        if self.config.epochs is not None:
            training_config.epochs = self.config.epochs
        if self.config.batch_size is not None:
            training_config.batch_size = self.config.batch_size
        if self.config.learning_rate is not None:
            training_config.learning_rate = self.config.learning_rate
        
        self.logger.info(f"Training NLP model with config: {training_config}")
        
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
            'perplexity': trained_model.training_history.get('perplexity', [])[-1] if 'perplexity' in trained_model.training_history else None
        }
        
        return trained_model, training_info
    
    def _get_nlp_training_config(self, task_type: TaskType, train_size: int) -> Any:
        """Get NLP-specific training configuration."""
        from ..training.config import optimize_training_config
        
        config = optimize_training_config(
            task_type=task_type,
            data_type=DataType.TEXT,
            num_samples=train_size
        )
        
        # NLP-specific training parameters
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.SENTIMENT_ANALYSIS, TaskType.CLASSIFICATION]:
            config.epochs = min(10, max(3, train_size // 1000))
            config.batch_size = min(16, max(4, train_size // 500))
            config.learning_rate = 2e-5  # Common for BERT fine-tuning
            config.optimizer = "adamw"
            config.loss_function = "categorical_crossentropy"
            config.metrics = ["accuracy", "f1_score"]
            config.early_stopping = True
            config.patience = 3
            config.warmup_steps = train_size // 10
        
        elif task_type in [TaskType.TEXT_GENERATION, TaskType.LANGUAGE_MODELING]:
            config.epochs = min(20, max(5, train_size // 500))
            config.batch_size = min(8, max(2, train_size // 1000))
            config.learning_rate = 5e-5
            config.optimizer = "adamw"
            config.loss_function = "cross_entropy"
            config.metrics = ["perplexity", "bleu"]
            config.early_stopping = True
            config.patience = 5
            config.gradient_accumulation_steps = 4
        
        elif task_type == TaskType.SEQUENCE_TO_SEQUENCE:
            config.epochs = min(15, max(3, train_size // 750))
            config.batch_size = min(8, max(2, train_size // 1000))
            config.learning_rate = 1e-4
            config.optimizer = "adamw"
            config.loss_function = "cross_entropy"
            config.metrics = ["bleu", "rouge"]
            config.early_stopping = True
            config.patience = 4
            config.beam_size = 4
        
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            config.epochs = min(8, max(3, train_size // 1000))
            config.batch_size = min(16, max(4, train_size // 500))
            config.learning_rate = 3e-5
            config.optimizer = "adamw"
            config.loss_function = "crf_loss"
            config.metrics = ["entity_f1", "precision", "recall"]
            config.early_stopping = True
            config.patience = 3
        
        return config
    
    def _validate_domain_config(self) -> None:
        """Validate NLP-specific configuration."""
        domain_config = self.config.domain_config
        
        # Validate max_length
        if 'max_length' in domain_config:
            max_length = domain_config['max_length']
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("max_length must be a positive integer")
            if max_length > 2048:
                self.logger.warning(f"max_length {max_length} is very large, may cause memory issues")
        
        # Validate tokenizer
        if 'tokenizer' in domain_config:
            tokenizer = domain_config['tokenizer']
            if not isinstance(tokenizer, str):
                raise ValueError("tokenizer must be a string")
        
        # Task-specific validation
        if self.config.task in [TaskType.TEXT_GENERATION.value, TaskType.LANGUAGE_MODELING.value]:
            if 'temperature' in domain_config:
                temp = domain_config['temperature']
                if not (0.0 < temp <= 2.0):
                    raise ValueError("temperature must be between 0.0 and 2.0")
            
            if 'top_p' in domain_config:
                top_p = domain_config['top_p']
                if not (0.0 < top_p <= 1.0):
                    raise ValueError("top_p must be between 0.0 and 1.0")
        
        if self.config.task == TaskType.SEQUENCE_TO_SEQUENCE.value:
            if 'max_source_length' in domain_config:
                max_src_len = domain_config['max_source_length']
                if not isinstance(max_src_len, int) or max_src_len <= 0:
                    raise ValueError("max_source_length must be a positive integer")
            
            if 'max_target_length' in domain_config:
                max_tgt_len = domain_config['max_target_length']
                if not isinstance(max_tgt_len, int) or max_tgt_len <= 0:
                    raise ValueError("max_target_length must be a positive integer")