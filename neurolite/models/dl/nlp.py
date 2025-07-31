"""
Natural Language Processing deep learning models for NeuroLite.

Implements transformer-based models for text classification, sentiment analysis,
text generation, and sequence-to-sequence tasks using Hugging Face transformers.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ...core import get_logger, DependencyError, ModelError
from ...data.detector import DataType
from ..base import BaseModel, TaskType, ModelCapabilities, PredictionResult, ModelMetadata
from ..registry import register_model


logger = get_logger(__name__)


class HuggingFaceTransformerModel(BaseModel):
    """
    Base class for Hugging Face transformer-based NLP models.
    
    Provides common functionality for text classification, generation,
    and sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "text-classification",
        num_labels: Optional[int] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Hugging Face transformer model.
        
        Args:
            model_name: Name of the pre-trained model (e.g., 'bert-base-uncased')
            task_type: Type of NLP task ('text-classification', 'text-generation', 'text2text-generation')
            num_labels: Number of labels for classification tasks
            max_length: Maximum sequence length
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.training_history = {}
        
        # Import Hugging Face dependencies
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                Trainer, TrainingArguments, pipeline
            )
            import torch
            
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
            self.Trainer = Trainer
            self.TrainingArguments = TrainingArguments
            self.pipeline = pipeline
            self.torch = torch
            
        except ImportError as e:
            raise DependencyError(
                f"Missing Hugging Face transformers dependencies required for NLP models: {e}",
                suggestions=[
                    "Install transformers: pip install transformers",
                    "Install PyTorch: pip install torch",
                    "Install NeuroLite with NLP extras: pip install neurolite[nlp]"
                ]
            )
        
        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.debug(f"Initialized {model_name} model for {task_type} on device: {self.device}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        supported_tasks = []
        
        if self.task_type == "text-classification":
            supported_tasks.extend([
                TaskType.TEXT_CLASSIFICATION,
                TaskType.SENTIMENT_ANALYSIS,
                TaskType.CLASSIFICATION
            ])
        elif self.task_type == "text-generation":
            supported_tasks.extend([
                TaskType.TEXT_GENERATION,
                TaskType.LANGUAGE_MODELING
            ])
        elif self.task_type == "text2text-generation":
            supported_tasks.extend([
                TaskType.SEQUENCE_TO_SEQUENCE,
                TaskType.TEXT_GENERATION
            ])
        
        return ModelCapabilities(
            supported_tasks=supported_tasks,
            supported_data_types=[DataType.TEXT],
            framework="transformers",
            requires_gpu=False,  # Can run on CPU but GPU is preferred
            min_samples=10,
            supports_probability_prediction=True if self.task_type == "text-classification" else False,
            supports_feature_importance=False
        )
    
    def _load_model_and_tokenizer(self) -> None:
        """Load the pre-trained model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return
        
        logger.debug(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on task type
        if self.task_type == "text-classification":
            self.model = self.AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels if self.num_labels else 2
            )
        elif self.task_type == "text-generation":
            self.model = self.AutoModelForCausalLM.from_pretrained(self.model_name)
        elif self.task_type == "text2text-generation":
            self.model = self.AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Move model to device
        self.model.to(self.device)
        
        logger.debug(f"Successfully loaded {self.model_name}")
    
    def _prepare_data(self, texts: List[str], labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Prepare text data for training or inference."""
        if self.tokenizer is None:
            self._load_model_and_tokenizer()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        
        if labels is not None:
            encodings["labels"] = self.torch.tensor(labels, dtype=self.torch.long).to(self.device)
        
        return encodings
    
    def fit(
        self,
        X: Union[List[str], np.ndarray, Any],
        y: Union[List[int], np.ndarray, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        **kwargs
    ) -> 'HuggingFaceTransformerModel':
        """Train the transformer model."""
        logger.debug(f"Training {self.model_name} for {epochs} epochs")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Convert inputs to lists if needed
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        
        # Ensure we have string inputs
        if not all(isinstance(text, str) for text in X):
            raise ValueError("All inputs must be strings for NLP models")
        
        # For text generation tasks, we don't need labels during training
        if self.task_type in ["text-generation", "text2text-generation"]:
            # For generation tasks, X contains the input texts and y contains target texts
            if self.task_type == "text2text-generation":
                # Prepare input-target pairs
                train_encodings = self._prepare_seq2seq_data(X, y)
            else:
                # For causal language modeling, concatenate input and target
                combined_texts = [f"{input_text} {target_text}" for input_text, target_text in zip(X, y)]
                train_encodings = self._prepare_data(combined_texts)
        else:
            # Classification task
            train_encodings = self._prepare_data(X, y)
        
        # Create dataset
        train_dataset = self._create_dataset(train_encodings)
        
        # Prepare validation data if provided
        eval_dataset = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(X_val, np.ndarray):
                X_val = X_val.tolist()
            if isinstance(y_val, np.ndarray):
                y_val = y_val.tolist()
            
            if self.task_type == "text-classification":
                val_encodings = self._prepare_data(X_val, y_val)
            else:
                val_encodings = self._prepare_data(X_val)
            
            eval_dataset = self._create_dataset(val_encodings)
        
        # Setup training arguments
        training_args = self.TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )
        
        # Create trainer
        trainer = self.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Store training history
        self.training_history = trainer.state.log_history
        
        self.is_trained = True
        logger.debug(f"Training completed for {self.model_name}")
        return self
    
    def _prepare_seq2seq_data(self, inputs: List[str], targets: List[str]) -> Dict[str, Any]:
        """Prepare data for sequence-to-sequence tasks."""
        # Tokenize inputs
        input_encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            targets,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encodings = {
            "input_ids": input_encodings["input_ids"].to(self.device),
            "attention_mask": input_encodings["attention_mask"].to(self.device),
            "labels": target_encodings["input_ids"].to(self.device)
        }
        
        return encodings
    
    def _create_dataset(self, encodings: Dict[str, Any]) -> Any:
        """Create a PyTorch dataset from encodings."""
        class NLPDataset(self.torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings["input_ids"])
        
        return NLPDataset(encodings)
    
    def predict(self, X: Union[List[str], np.ndarray, Any], **kwargs) -> PredictionResult:
        """Make predictions using the transformer model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to list if needed
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Ensure we have string inputs
        if not all(isinstance(text, str) for text in X):
            raise ValueError("All inputs must be strings for NLP models")
        
        logger.debug(f"Making predictions for {len(X)} samples")
        
        if self.task_type == "text-classification":
            return self._predict_classification(X, **kwargs)
        elif self.task_type == "text-generation":
            return self._predict_generation(X, **kwargs)
        elif self.task_type == "text2text-generation":
            return self._predict_seq2seq(X, **kwargs)
        else:
            raise ValueError(f"Unsupported task type for prediction: {self.task_type}")
    
    def _predict_classification(self, texts: List[str], **kwargs) -> PredictionResult:
        """Make classification predictions."""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        with self.torch.no_grad():
            for text in texts:
                # Tokenize single text
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get predictions and probabilities
                probs = self.torch.softmax(logits, dim=-1)
                pred = self.torch.argmax(logits, dim=-1)
                
                predictions.append(pred.cpu().item())
                probabilities.append(probs.cpu().numpy()[0])
        
        return PredictionResult(
            predictions=np.array(predictions),
            probabilities=np.array(probabilities)
        )
    
    def _predict_generation(self, texts: List[str], max_new_tokens: int = 50, **kwargs) -> PredictionResult:
        """Make text generation predictions."""
        self.model.eval()
        
        generated_texts = []
        
        with self.torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove input text from generated text
                if generated_text.startswith(text):
                    generated_text = generated_text[len(text):].strip()
                
                generated_texts.append(generated_text)
        
        return PredictionResult(
            predictions=np.array(generated_texts, dtype=object)
        )
    
    def _predict_seq2seq(self, texts: List[str], max_length: int = 128, **kwargs) -> PredictionResult:
        """Make sequence-to-sequence predictions."""
        self.model.eval()
        
        generated_texts = []
        
        with self.torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate text
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    **kwargs
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return PredictionResult(
            predictions=np.array(generated_texts, dtype=object)
        )
    
    def save(self, path: str) -> None:
        """Save the transformer model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'training_history': self.training_history,
            'config': self._config,
            'is_trained': self.is_trained
        }
        
        import json
        with open(os.path.join(path, 'neurolite_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved {self.model_name} to {path}")
    
    def load(self, path: str) -> 'HuggingFaceTransformerModel':
        """Load the transformer model."""
        # Load metadata
        import json
        with open(os.path.join(path, 'neurolite_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        self.model_name = metadata['model_name']
        self.task_type = metadata['task_type']
        self.num_labels = metadata['num_labels']
        self.max_length = metadata['max_length']
        self.training_history = metadata.get('training_history', {})
        self._config = metadata.get('config', {})
        self.is_trained = metadata.get('is_trained', True)
        
        # Load tokenizer
        self.tokenizer = self.AutoTokenizer.from_pretrained(path)
        
        # Load model based on task type
        if self.task_type == "text-classification":
            self.model = self.AutoModelForSequenceClassification.from_pretrained(path)
        elif self.task_type == "text-generation":
            self.model = self.AutoModelForCausalLM.from_pretrained(path)
        elif self.task_type == "text2text-generation":
            self.model = self.AutoModelForSeq2SeqLM.from_pretrained(path)
        
        # Move to device
        self.model.to(self.device)
        
        logger.debug(f"Loaded {self.model_name} from {path}")
        return self


class TextClassificationModel(HuggingFaceTransformerModel):
    """Transformer-based text classification model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", **kwargs):
        """
        Initialize text classification model.
        
        Args:
            model_name: Pre-trained model name (default: distilbert-base-uncased)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            task_type="text-classification",
            **kwargs
        )


class SentimentAnalysisModel(HuggingFaceTransformerModel):
    """Transformer-based sentiment analysis model."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", **kwargs):
        """
        Initialize sentiment analysis model.
        
        Args:
            model_name: Pre-trained model name (default: cardiffnlp/twitter-roberta-base-sentiment-latest)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            task_type="text-classification",
            num_labels=3,  # negative, neutral, positive
            **kwargs
        )


class TextGenerationModel(HuggingFaceTransformerModel):
    """Transformer-based text generation model."""
    
    def __init__(self, model_name: str = "gpt2", **kwargs):
        """
        Initialize text generation model.
        
        Args:
            model_name: Pre-trained model name (default: gpt2)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            task_type="text-generation",
            **kwargs
        )


class Seq2SeqModel(HuggingFaceTransformerModel):
    """Transformer-based sequence-to-sequence model for translation and other tasks."""
    
    def __init__(self, model_name: str = "t5-small", **kwargs):
        """
        Initialize sequence-to-sequence model.
        
        Args:
            model_name: Pre-trained model name (default: t5-small)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            task_type="text2text-generation",
            **kwargs
        )


class TranslationModel(Seq2SeqModel):
    """Specialized translation model using sequence-to-sequence architecture."""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "fr", **kwargs):
        """
        Initialize translation model.
        
        Args:
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'fr')
            **kwargs: Additional arguments passed to parent
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Use language-specific model if available
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        super().__init__(model_name=model_name, **kwargs)
    
    def fit(self, X: Union[List[str], np.ndarray, Any], y: Union[List[str], np.ndarray, Any], **kwargs):
        """
        Train translation model with source-target pairs.
        
        Args:
            X: Source language texts
            y: Target language texts
            **kwargs: Additional training arguments
        """
        # For translation, we need to add task prefix for T5-based models
        if "t5" in self.model_name.lower():
            X = [f"translate {self.source_lang} to {self.target_lang}: {text}" for text in X]
        
        return super().fit(X, y, **kwargs)
    
    def predict(self, X: Union[List[str], np.ndarray, Any], **kwargs) -> PredictionResult:
        """
        Translate texts from source to target language.
        
        Args:
            X: Source language texts
            **kwargs: Additional prediction arguments
            
        Returns:
            PredictionResult with translated texts
        """
        # Add task prefix for T5-based models
        if "t5" in self.model_name.lower():
            X = [f"translate {self.source_lang} to {self.target_lang}: {text}" for text in X]
        
        return super().predict(X, **kwargs)


def register_nlp_models():
    """Register all NLP models in the global registry."""
    logger.debug("Registering NLP models")
    
    # Text Classification Models
    classification_models = [
        ("bert_base", "bert-base-uncased", "BERT base model for text classification"),
        ("distilbert_base", "distilbert-base-uncased", "DistilBERT base model for text classification"),
        ("roberta_base", "roberta-base", "RoBERTa base model for text classification"),
        ("albert_base", "albert-base-v2", "ALBERT base model for text classification"),
    ]
    
    for name, model_name, description in classification_models:
        register_model(
            name=f"text_classification_{name}",
            model_class=TextClassificationModel,
            factory_function=lambda mn=model_name, **kwargs: TextClassificationModel(model_name=mn, **kwargs),
            priority=8,
            description=description,
            tags=["transformer", "text_classification", "pretrained", "nlp"]
        )
    
    # Sentiment Analysis Models
    sentiment_models = [
        ("roberta_twitter", "cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa model fine-tuned for Twitter sentiment"),
        ("bert_sentiment", "nlptown/bert-base-multilingual-uncased-sentiment", "Multilingual BERT for sentiment analysis"),
        ("distilbert_sentiment", "distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT fine-tuned on SST-2"),
    ]
    
    for name, model_name, description in sentiment_models:
        register_model(
            name=f"sentiment_analysis_{name}",
            model_class=SentimentAnalysisModel,
            factory_function=lambda mn=model_name, **kwargs: SentimentAnalysisModel(model_name=mn, **kwargs),
            priority=9,
            description=description,
            tags=["transformer", "sentiment_analysis", "pretrained", "nlp"]
        )
    
    # Text Generation Models
    generation_models = [
        ("gpt2", "gpt2", "GPT-2 model for text generation"),
        ("gpt2_medium", "gpt2-medium", "GPT-2 medium model for text generation"),
        ("distilgpt2", "distilgpt2", "DistilGPT-2 model for text generation"),
    ]
    
    for name, model_name, description in generation_models:
        register_model(
            name=f"text_generation_{name}",
            model_class=TextGenerationModel,
            factory_function=lambda mn=model_name, **kwargs: TextGenerationModel(model_name=mn, **kwargs),
            priority=7,
            description=description,
            tags=["transformer", "text_generation", "pretrained", "nlp"]
        )
    
    # Sequence-to-Sequence Models
    seq2seq_models = [
        ("t5_small", "t5-small", "T5 small model for sequence-to-sequence tasks"),
        ("t5_base", "t5-base", "T5 base model for sequence-to-sequence tasks"),
        ("bart_base", "facebook/bart-base", "BART base model for sequence-to-sequence tasks"),
    ]
    
    for name, model_name, description in seq2seq_models:
        register_model(
            name=f"seq2seq_{name}",
            model_class=Seq2SeqModel,
            factory_function=lambda mn=model_name, **kwargs: Seq2SeqModel(model_name=mn, **kwargs),
            priority=7,
            description=description,
            tags=["transformer", "seq2seq", "pretrained", "nlp"]
        )
    
    # Translation Models
    translation_pairs = [
        ("en_fr", "en", "fr", "English to French translation"),
        ("en_de", "en", "de", "English to German translation"),
        ("en_es", "en", "es", "English to Spanish translation"),
        ("fr_en", "fr", "en", "French to English translation"),
        ("de_en", "de", "en", "German to English translation"),
        ("es_en", "es", "en", "Spanish to English translation"),
    ]
    
    for name, source_lang, target_lang, description in translation_pairs:
        register_model(
            name=f"translation_{name}",
            model_class=TranslationModel,
            factory_function=lambda sl=source_lang, tl=target_lang, **kwargs: TranslationModel(source_lang=sl, target_lang=tl, **kwargs),
            priority=8,
            description=description,
            tags=["transformer", "translation", "seq2seq", "pretrained", "nlp"]
        )
    
    logger.debug("Successfully registered NLP models")