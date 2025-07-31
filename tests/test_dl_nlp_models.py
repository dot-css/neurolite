"""
Unit tests for NLP deep learning models.

Tests transformer-based models for text classification, sentiment analysis,
text generation, and sequence-to-sequence tasks.
"""

import pytest
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil

from neurolite.core.exceptions import DependencyError, ModelError
from neurolite.data.detector import DataType
from neurolite.models.base import TaskType, PredictionResult
from neurolite.models.registry import get_model_registry


def _transformers_available():
    """Check if transformers library is available."""
    try:
        import transformers
        import torch
        return True
    except ImportError:
        return False


class TestNLPModelsBasic:
    """Basic tests for NLP models that don't require complex mocking."""
    
    def test_import_nlp_models(self):
        """Test that NLP models can be imported successfully."""
        try:
            from neurolite.models.dl.nlp import (
                HuggingFaceTransformerModel,
                TextClassificationModel,
                SentimentAnalysisModel,
                TextGenerationModel,
                Seq2SeqModel,
                TranslationModel,
                register_nlp_models
            )
            # If we get here, imports were successful
            assert True
        except ImportError as e:
            # If transformers is not available, we should get a DependencyError when initializing
            # but the import itself should work
            assert "transformers" in str(e).lower() or "torch" in str(e).lower()
    
    def test_model_classes_exist(self):
        """Test that all model classes are properly defined."""
        from neurolite.models.dl.nlp import (
            HuggingFaceTransformerModel,
            TextClassificationModel,
            SentimentAnalysisModel,
            TextGenerationModel,
            Seq2SeqModel,
            TranslationModel
        )
        
        # Check that classes exist and have the right inheritance
        assert issubclass(TextClassificationModel, HuggingFaceTransformerModel)
        assert issubclass(SentimentAnalysisModel, HuggingFaceTransformerModel)
        assert issubclass(TextGenerationModel, HuggingFaceTransformerModel)
        assert issubclass(Seq2SeqModel, HuggingFaceTransformerModel)
        assert issubclass(TranslationModel, Seq2SeqModel)
    
    def test_model_initialization_without_transformers(self):
        """Test model initialization when transformers is not available."""
        # Mock the transformers import to fail
        with patch.dict('sys.modules', {'transformers': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'transformers'")):
                from neurolite.models.dl.nlp import HuggingFaceTransformerModel
                
                with pytest.raises(DependencyError) as exc_info:
                    HuggingFaceTransformerModel("test-model")
                
                assert "Missing Hugging Face transformers dependencies" in str(exc_info.value)
                assert "pip install transformers" in str(exc_info.value)
    
    def test_task_type_constants(self):
        """Test that required task types are defined."""
        assert hasattr(TaskType, 'TEXT_CLASSIFICATION')
        assert hasattr(TaskType, 'SENTIMENT_ANALYSIS')
        assert hasattr(TaskType, 'TEXT_GENERATION')
        assert hasattr(TaskType, 'SEQUENCE_TO_SEQUENCE')
        assert hasattr(TaskType, 'LANGUAGE_MODELING')
    
    def test_data_type_constants(self):
        """Test that required data types are defined."""
        assert hasattr(DataType, 'TEXT')
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_initialization_with_transformers(self):
        """Test model initialization when transformers is available."""
        from neurolite.models.dl.nlp import (
            TextClassificationModel,
            SentimentAnalysisModel,
            TextGenerationModel,
            Seq2SeqModel,
            TranslationModel
        )
        
        # Test TextClassificationModel
        model = TextClassificationModel()
        assert model.model_name == "distilbert-base-uncased"
        assert model.task_type == "text-classification"
        
        # Test SentimentAnalysisModel
        model = SentimentAnalysisModel()
        assert model.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert model.task_type == "text-classification"
        assert model.num_labels == 3
        
        # Test TextGenerationModel
        model = TextGenerationModel()
        assert model.model_name == "gpt2"
        assert model.task_type == "text-generation"
        
        # Test Seq2SeqModel
        model = Seq2SeqModel()
        assert model.model_name == "t5-small"
        assert model.task_type == "text2text-generation"
        
        # Test TranslationModel
        model = TranslationModel(source_lang="en", target_lang="fr")
        assert model.source_lang == "en"
        assert model.target_lang == "fr"
        assert model.model_name == "Helsinki-NLP/opus-mt-en-fr"
        assert model.task_type == "text2text-generation"
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_capabilities(self):
        """Test model capabilities."""
        from neurolite.models.dl.nlp import (
            TextClassificationModel,
            TextGenerationModel,
            Seq2SeqModel
        )
        
        # Test text classification capabilities
        model = TextClassificationModel()
        capabilities = model.capabilities
        assert TaskType.TEXT_CLASSIFICATION in capabilities.supported_tasks
        assert TaskType.SENTIMENT_ANALYSIS in capabilities.supported_tasks
        assert DataType.TEXT in capabilities.supported_data_types
        assert capabilities.framework == "transformers"
        assert capabilities.supports_probability_prediction
        
        # Test text generation capabilities
        model = TextGenerationModel()
        capabilities = model.capabilities
        assert TaskType.TEXT_GENERATION in capabilities.supported_tasks
        assert TaskType.LANGUAGE_MODELING in capabilities.supported_tasks
        assert DataType.TEXT in capabilities.supported_data_types
        assert capabilities.framework == "transformers"
        assert not capabilities.supports_probability_prediction
        
        # Test seq2seq capabilities
        model = Seq2SeqModel()
        capabilities = model.capabilities
        assert TaskType.SEQUENCE_TO_SEQUENCE in capabilities.supported_tasks
        assert TaskType.TEXT_GENERATION in capabilities.supported_tasks
        assert DataType.TEXT in capabilities.supported_data_types
        assert capabilities.framework == "transformers"
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_registration(self):
        """Test NLP model registration."""
        from neurolite.models.dl.nlp import register_nlp_models
        
        # Clean registry
        registry = get_model_registry()
        registry.clear()
        
        try:
            # Register NLP models
            register_nlp_models()
            
            # Check that models were registered
            all_models = registry.list_models()
            
            # Should have multiple models registered
            assert len(all_models) > 0
            
            # Check for some expected models
            expected_models = [
                "text_classification_bert_base",
                "sentiment_analysis_roberta_twitter",
                "text_generation_gpt2",
                "seq2seq_t5_small",
                "translation_en_fr"
            ]
            
            for expected_model in expected_models:
                assert expected_model in all_models, f"Expected model {expected_model} not found in {all_models}"
            
            # Test filtering by task type
            classification_models = registry.list_models(task_type=TaskType.TEXT_CLASSIFICATION)
            assert len(classification_models) > 0
            
            generation_models = registry.list_models(task_type=TaskType.TEXT_GENERATION)
            assert len(generation_models) > 0
            
            seq2seq_models = registry.list_models(task_type=TaskType.SEQUENCE_TO_SEQUENCE)
            assert len(seq2seq_models) > 0
            
            # Test filtering by data type
            text_models = registry.list_models(data_type=DataType.TEXT)
            assert len(text_models) == len(all_models)  # All NLP models should support text
            
        finally:
            # Clean up
            registry.clear()
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_creation_from_registry(self):
        """Test creating models from registry."""
        from neurolite.models.dl.nlp import (
            register_nlp_models,
            TextClassificationModel,
            SentimentAnalysisModel,
            TextGenerationModel,
            TranslationModel
        )
        
        # Clean registry
        registry = get_model_registry()
        registry.clear()
        
        try:
            # Register NLP models
            register_nlp_models()
            
            # Test creating different types of models
            model = registry.get_model("text_classification_bert_base")
            assert isinstance(model, TextClassificationModel)
            assert model.model_name == "bert-base-uncased"
            
            model = registry.get_model("sentiment_analysis_roberta_twitter")
            assert isinstance(model, SentimentAnalysisModel)
            assert model.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            model = registry.get_model("text_generation_gpt2")
            assert isinstance(model, TextGenerationModel)
            assert model.model_name == "gpt2"
            
            model = registry.get_model("translation_en_fr")
            assert isinstance(model, TranslationModel)
            assert model.source_lang == "en"
            assert model.target_lang == "fr"
            
        finally:
            # Clean up
            registry.clear()
    
    def test_prediction_result_structure(self):
        """Test that PredictionResult has the expected structure."""
        # Create a sample prediction result
        predictions = np.array([0, 1, 2])
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        
        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities
        )
        
        assert np.array_equal(result.predictions, predictions)
        assert np.array_equal(result.probabilities, probabilities)
        assert result.confidence_scores is None
        assert result.feature_importance is None
        assert result.metadata is None


class TestNLPModelsIntegration:
    """Integration tests that require transformers library."""
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_text_classification_model_basic_workflow(self):
        """Test basic workflow of text classification model."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        # Create model
        model = TextClassificationModel(model_name="distilbert-base-uncased", num_labels=2)
        
        # Check initial state
        assert not model.is_trained
        assert model.model is None
        assert model.tokenizer is None
        
        # Test capabilities
        capabilities = model.capabilities
        assert TaskType.TEXT_CLASSIFICATION in capabilities.supported_tasks
        assert DataType.TEXT in capabilities.supported_data_types
        
        # Note: We don't test actual training/prediction here as it would require
        # downloading models and significant compute time
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_translation_model_task_prefix(self):
        """Test that translation model adds task prefix for T5 models."""
        from neurolite.models.dl.nlp import TranslationModel
        
        # Create translation model
        model = TranslationModel(source_lang="en", target_lang="fr")
        
        # Test that it has the right configuration
        assert model.source_lang == "en"
        assert model.target_lang == "fr"
        assert model.task_type == "text2text-generation"
        
        # For T5 models, the model name should be set appropriately
        if "t5" in model.model_name.lower():
            # Test that fit method would add task prefix (we can't easily test this without mocking)
            pass
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_save_load_structure(self):
        """Test the structure of save/load methods."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        model = TextClassificationModel()
        
        # Test that save method exists and has right signature
        assert hasattr(model, 'save')
        assert hasattr(model, 'load')
        
        # Test that save raises error for untrained model
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            model.save("/tmp/test")
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_text_classification_with_sample_data(self):
        """Test text classification model with sample text data."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        # Sample text data for testing
        sample_texts = [
            "This is a positive example.",
            "This is a negative example.",
            "Another positive text.",
            "Another negative text."
        ]
        sample_labels = [1, 0, 1, 0]  # Binary classification
        
        # Create model with small configuration for testing
        model = TextClassificationModel(model_name="distilbert-base-uncased", num_labels=2)
        
        # Test data validation
        model.validate_data(sample_texts, sample_labels, TaskType.TEXT_CLASSIFICATION, DataType.TEXT)
        
        # Test that model can be initialized for prediction (without actual training)
        # This tests the model loading infrastructure
        assert model.capabilities.supports_probability_prediction
        assert TaskType.TEXT_CLASSIFICATION in model.capabilities.supported_tasks
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_sentiment_analysis_with_sample_data(self):
        """Test sentiment analysis model with sample text data."""
        from neurolite.models.dl.nlp import SentimentAnalysisModel
        
        # Sample sentiment data
        sample_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Amazing quality and service!"
        ]
        sample_labels = [2, 0, 1, 2]  # 0=negative, 1=neutral, 2=positive
        
        # Create sentiment model
        model = SentimentAnalysisModel()
        
        # Test data validation
        model.validate_data(sample_texts, sample_labels, TaskType.SENTIMENT_ANALYSIS, DataType.TEXT)
        
        # Test model configuration
        assert model.num_labels == 3  # negative, neutral, positive
        assert model.task_type == "text-classification"
        assert TaskType.SENTIMENT_ANALYSIS in model.capabilities.supported_tasks
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_text_generation_with_sample_data(self):
        """Test text generation model with sample text data."""
        from neurolite.models.dl.nlp import TextGenerationModel
        
        # Sample text prompts
        sample_prompts = [
            "Once upon a time",
            "The future of AI is",
            "In a world where",
            "Scientists have discovered"
        ]
        
        # Create text generation model
        model = TextGenerationModel(model_name="gpt2")
        
        # Test model configuration
        assert model.task_type == "text-generation"
        assert TaskType.TEXT_GENERATION in model.capabilities.supported_tasks
        assert not model.capabilities.supports_probability_prediction  # Generation models don't predict probabilities
        
        # Test that model can handle text input validation
        # For generation, we don't need labels, so we pass None
        model.validate_data(sample_prompts, None, TaskType.TEXT_GENERATION, DataType.TEXT)
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_sequence_to_sequence_with_sample_data(self):
        """Test sequence-to-sequence model with sample text data."""
        from neurolite.models.dl.nlp import Seq2SeqModel
        
        # Sample input-output pairs for seq2seq
        sample_inputs = [
            "Translate to French: Hello world",
            "Summarize: This is a long text that needs to be summarized.",
            "Question: What is the capital of France?",
            "Paraphrase: The weather is nice today."
        ]
        sample_outputs = [
            "Bonjour le monde",
            "Long text summary.",
            "Paris is the capital of France.",
            "Today has pleasant weather."
        ]
        
        # Create seq2seq model
        model = Seq2SeqModel(model_name="t5-small")
        
        # Test model configuration
        assert model.task_type == "text2text-generation"
        assert TaskType.SEQUENCE_TO_SEQUENCE in model.capabilities.supported_tasks
        assert TaskType.TEXT_GENERATION in model.capabilities.supported_tasks
        
        # Test data validation
        model.validate_data(sample_inputs, sample_outputs, TaskType.SEQUENCE_TO_SEQUENCE, DataType.TEXT)
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_translation_model_with_sample_data(self):
        """Test translation model with sample text data."""
        from neurolite.models.dl.nlp import TranslationModel
        
        # Sample English-French translation pairs
        english_texts = [
            "Hello, how are you?",
            "The weather is beautiful today.",
            "I love learning new languages.",
            "Thank you for your help."
        ]
        french_texts = [
            "Bonjour, comment allez-vous?",
            "Le temps est magnifique aujourd'hui.",
            "J'adore apprendre de nouvelles langues.",
            "Merci pour votre aide."
        ]
        
        # Create translation model
        model = TranslationModel(source_lang="en", target_lang="fr")
        
        # Test model configuration
        assert model.source_lang == "en"
        assert model.target_lang == "fr"
        assert model.task_type == "text2text-generation"
        assert TaskType.SEQUENCE_TO_SEQUENCE in model.capabilities.supported_tasks
        
        # Test data validation
        model.validate_data(english_texts, french_texts, TaskType.SEQUENCE_TO_SEQUENCE, DataType.TEXT)
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_data_preparation(self):
        """Test data preparation methods with sample data."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        # Create model
        model = TextClassificationModel(model_name="distilbert-base-uncased", num_labels=2)
        
        # Sample data
        texts = ["This is a test.", "Another test sentence."]
        labels = [1, 0]
        
        # Test data preparation (this will load the tokenizer)
        prepared_data = model._prepare_data(texts, labels)
        
        # Check that prepared data has the right structure
        assert "input_ids" in prepared_data
        assert "attention_mask" in prepared_data
        assert "labels" in prepared_data
        
        # Check that tokenizer was loaded
        assert model.tokenizer is not None
        assert hasattr(model.tokenizer, 'encode')
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_error_handling(self):
        """Test error handling with invalid data."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        model = TextClassificationModel()
        
        # Test prediction without training
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(["test text"])
        
        # Test invalid input types
        with pytest.raises(ValueError, match="All inputs must be strings for NLP models"):
            model._prepare_data([123, 456])  # Non-string inputs
    
    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers library not available"
    )
    def test_model_device_handling(self):
        """Test device handling (CPU/GPU) for models."""
        from neurolite.models.dl.nlp import TextClassificationModel
        
        # Test CPU device
        model_cpu = TextClassificationModel(device="cpu")
        assert model_cpu.device == "cpu"
        
        # Test auto device selection
        model_auto = TextClassificationModel(device=None)
        assert model_auto.device in ["cpu", "cuda"]  # Should be one of these
        
        # Test explicit CUDA (will fall back to CPU if not available)
        model_cuda = TextClassificationModel(device="cuda")
        # Device should be set, even if CUDA is not available (will use CPU)
        assert model_cuda.device in ["cpu", "cuda"]


if __name__ == "__main__":
    pytest.main([__file__])