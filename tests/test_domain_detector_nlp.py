"""
Unit tests for DomainDetector NLP functionality.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from neurolite.detectors.domain_detector import DomainDetector
from neurolite.core.data_models import NLPTaskAnalysis
from neurolite.core.exceptions import NeuroLiteException


class TestDomainDetectorNLP:
    """Test cases for NLP detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DomainDetector()
    
    @patch('neurolite.detectors.domain_detector.Path.is_file')
    def test_detect_nlp_task_file(self, mock_is_file):
        """Test NLP task detection from file."""
        mock_is_file.return_value = True
        
        with patch.object(self.detector, '_analyze_text_file') as mock_analyze:
            expected_result = NLPTaskAnalysis(
                task_type='sentiment',
                task_subtype='sentiment_analysis',
                confidence=0.8
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_nlp_task('/fake/file.json')
            
            assert result == expected_result
            mock_analyze.assert_called_once()
    
    def test_detect_nlp_task_dataframe(self):
        """Test NLP task detection from DataFrame."""
        df = pd.DataFrame({'text': ['sample text'], 'sentiment': ['positive']})
        
        with patch.object(self.detector, '_analyze_text_dataframe') as mock_analyze:
            expected_result = NLPTaskAnalysis(
                task_type='sentiment',
                task_subtype='sentiment_analysis',
                confidence=0.85
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_nlp_task(df)
            
            assert result == expected_result
            mock_analyze.assert_called_once_with(df)
    
    def test_detect_nlp_task_text_list(self):
        """Test NLP task detection from text list."""
        text_list = ['This is great!', 'This is terrible.']
        
        with patch.object(self.detector, '_analyze_text_list') as mock_analyze:
            expected_result = NLPTaskAnalysis(
                task_type='classification',
                task_subtype='text_classification',
                confidence=0.5
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_nlp_task(text_list)
            
            assert result == expected_result
            mock_analyze.assert_called_once_with(text_list)
    
    @patch('neurolite.detectors.domain_detector.Path.is_file')
    def test_detect_nlp_task_nonexistent_file(self, mock_is_file):
        """Test NLP task detection with nonexistent file."""
        mock_is_file.return_value = False
        
        with pytest.raises(NeuroLiteException, match="File does not exist"):
            self.detector.detect_nlp_task('/nonexistent/file.txt')
    
    def test_detect_nlp_task_unsupported_type(self):
        """Test NLP task detection with unsupported data type."""
        with pytest.raises(NeuroLiteException, match="Unsupported data source type"):
            self.detector.detect_nlp_task(123)
    
    def test_analyze_json_text_data_qa_format(self):
        """Test JSON Q&A format detection."""
        qa_data = {
            "question": "What is the capital of France?",
            "answer": "Paris"
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(qa_data))):
            result = self.detector._analyze_json_text_data(Path('fake.json'))
            
            assert result.task_type == 'qa'
            assert result.task_subtype == 'question_answering'
            assert result.confidence == 0.8
    
    def test_analyze_json_text_data_sentiment_list(self):
        """Test JSON sentiment analysis format detection."""
        sentiment_data = [
            {"text": "I love this product!", "sentiment": "positive"},
            {"text": "This is terrible.", "sentiment": "negative"}
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(sentiment_data))):
            result = self.detector._analyze_json_text_data(Path('fake.json'))
            
            assert result.task_type == 'sentiment'
            assert result.task_subtype == 'sentiment_analysis'
            assert result.confidence == 0.8
    
    def test_analyze_json_text_data_ner_format(self):
        """Test JSON NER format detection."""
        ner_data = [
            {
                "text": "John works at Google",
                "entities": [
                    {"start": 0, "end": 4, "label": "PERSON"},
                    {"start": 14, "end": 20, "label": "ORG"}
                ]
            }
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(ner_data))):
            result = self.detector._analyze_json_text_data(Path('fake.json'))
            
            assert result.task_type == 'ner'
            assert result.task_subtype == 'named_entity_recognition'
            assert result.confidence == 0.8
    
    def test_analyze_json_text_data_conversation_format(self):
        """Test JSON conversation format detection."""
        conversation_data = {
            "conversation": [
                {"speaker": "user", "message": "Hello"},
                {"speaker": "bot", "message": "Hi there!"}
            ]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(conversation_data))):
            result = self.detector._analyze_json_text_data(Path('fake.json'))
            
            assert result.task_type == 'conversation'
            assert result.task_subtype == 'dialogue'
            assert result.confidence == 0.8
    
    def test_analyze_json_text_data_invalid_json(self):
        """Test handling of invalid JSON."""
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with pytest.raises(NeuroLiteException, match="Invalid JSON text data file"):
                self.detector._analyze_json_text_data(Path('fake.json'))
    
    def test_analyze_jsonl_text_data(self):
        """Test JSONL format detection."""
        jsonl_content = '''{"text": "Great product!", "label": "positive"}
{"text": "Poor quality", "label": "negative"}
{"text": "Average item", "label": "neutral"}'''
        
        with patch('builtins.open', mock_open(read_data=jsonl_content)):
            result = self.detector._analyze_jsonl_text_data(Path('fake.jsonl'))
            
            assert result.task_type == 'classification'
            assert result.task_subtype == 'text_classification'
            assert result.confidence == 0.7
    
    def test_analyze_jsonl_text_data_no_valid_samples(self):
        """Test JSONL with no valid samples."""
        jsonl_content = '''invalid json line
another invalid line'''
        
        with patch('builtins.open', mock_open(read_data=jsonl_content)):
            result = self.detector._analyze_jsonl_text_data(Path('fake.jsonl'))
            
            assert result.task_type == 'unknown'
            assert result.task_subtype == 'no_valid_samples'
            assert result.confidence == 0.3
    
    def test_analyze_plain_text_file(self):
        """Test plain text file analysis."""
        text_content = "This is a sample document for classification."
        
        with patch('builtins.open', mock_open(read_data=text_content)):
            with patch.object(self.detector, '_analyze_text_content') as mock_analyze:
                mock_analyze.return_value = {'avg_length': 45}
                
                result = self.detector._analyze_plain_text_file(Path('fake.txt'))
                
                assert result.task_type == 'classification'
                assert result.task_subtype == 'document_classification'
                assert result.confidence == 0.6
    
    def test_analyze_text_dataframe_sentiment(self):
        """Test DataFrame sentiment analysis detection."""
        df = pd.DataFrame({
            'text': ['I love this!', 'This is bad.', 'It\'s okay.'],
            'sentiment': ['positive', 'negative', 'neutral']
        })
        
        with patch.object(self.detector, '_find_text_columns') as mock_find:
            mock_find.return_value = ['text']
            
            result = self.detector._analyze_text_dataframe(df)
            
            assert result.task_type == 'sentiment'
            assert result.task_subtype == 'sentiment_analysis'
            assert result.confidence == 0.85
    
    def test_analyze_text_dataframe_ner(self):
        """Test DataFrame NER detection."""
        df = pd.DataFrame({
            'text': ['John works at Google', 'Mary lives in Paris'],
            'entities': ['PERSON,ORG', 'PERSON,LOC']
        })
        
        with patch.object(self.detector, '_find_text_columns') as mock_find:
            mock_find.return_value = ['text']
            
            result = self.detector._analyze_text_dataframe(df)
            
            assert result.task_type == 'ner'
            assert result.task_subtype == 'sequence_labeling'
            assert result.confidence == 0.8
    
    def test_analyze_text_dataframe_qa(self):
        """Test DataFrame Q&A detection."""
        df = pd.DataFrame({
            'question': ['What is AI?', 'How does ML work?'],
            'answer': ['Artificial Intelligence...', 'Machine Learning...']
        })
        
        with patch.object(self.detector, '_find_text_columns') as mock_find:
            mock_find.return_value = ['question', 'answer']
            
            result = self.detector._analyze_text_dataframe(df)
            
            assert result.task_type == 'qa'
            assert result.task_subtype == 'question_answering'
            assert result.confidence == 0.8
    
    def test_analyze_text_dataframe_conversation(self):
        """Test DataFrame conversation detection."""
        df = pd.DataFrame({
            'message': ['Hello there', 'Hi, how are you?', 'I\'m doing well'],
            'speaker': ['user', 'bot', 'user']
        })
        
        with patch.object(self.detector, '_find_text_columns') as mock_find:
            mock_find.return_value = ['message']
            
            result = self.detector._analyze_text_dataframe(df)
            
            assert result.task_type == 'conversation'
            assert result.task_subtype == 'dialogue'
            assert result.confidence == 0.8
    
    def test_analyze_text_dataframe_empty(self):
        """Test empty DataFrame analysis."""
        df = pd.DataFrame()
        
        result = self.detector._analyze_text_dataframe(df)
        
        assert result.task_type == 'unknown'
        assert result.task_subtype == 'empty_dataframe'
        assert result.confidence == 0.3
    
    def test_analyze_text_dataframe_no_text_columns(self):
        """Test DataFrame with no text columns."""
        df = pd.DataFrame({
            'number': [1, 2, 3],
            'value': [10.5, 20.3, 30.1]
        })
        
        with patch.object(self.detector, '_find_text_columns') as mock_find:
            mock_find.return_value = []
            
            result = self.detector._analyze_text_dataframe(df)
            
            assert result.task_type == 'unknown'
            assert result.task_subtype == 'no_text_columns'
            assert result.confidence == 0.3
    
    def test_analyze_text_list_empty(self):
        """Test empty text list analysis."""
        result = self.detector._analyze_text_list([])
        
        assert result.task_type == 'unknown'
        assert result.task_subtype == 'empty_list'
        assert result.confidence == 0.3
    
    def test_analyze_text_list_valid(self):
        """Test valid text list analysis."""
        text_list = ['This is great!', 'Not so good.', 'Average quality.']
        
        with patch.object(self.detector, '_analyze_text_content') as mock_analyze:
            mock_analyze.return_value = {'total_texts': 3, 'avg_length': 15}
            
            result = self.detector._analyze_text_list(text_list)
            
            assert result.task_type == 'classification'
            assert result.task_subtype == 'text_classification'
            assert result.confidence == 0.5
    
    def test_find_text_columns(self):
        """Test finding text columns in DataFrame."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['This is a long text sample', 'Another text sample here', 'Third text sample'],
            'short': ['a', 'b', 'c'],  # Too short to be considered text
            'number': [10, 20, 30],
            'description': ['Long description text here', 'Another description', 'Third description']
        })
        
        result = self.detector._find_text_columns(df)
        
        # Should find 'text' and 'description' columns
        assert 'text' in result
        assert 'description' in result
        assert 'short' not in result  # Too short
        assert 'id' not in result  # Not string type
        assert 'number' not in result  # Not string type
    
    def test_analyze_text_content(self):
        """Test text content analysis."""
        texts = [
            'This is a great product! I love it.',
            'Not so good quality. Poor design.',
            'Average item with decent features.'
        ]
        
        result = self.detector._analyze_text_content(texts)
        
        assert result['total_texts'] == 3
        assert result['avg_length'] > 0
        assert result['max_length'] > 0
        assert result['min_length'] > 0
        assert result['has_punctuation'] is True
        assert 'estimated_language' in result
    
    def test_analyze_text_content_empty(self):
        """Test text content analysis with empty list."""
        result = self.detector._analyze_text_content([])
        
        assert result == {}
    
    def test_detect_nlp_task_from_dict_qa(self):
        """Test NLP task detection from Q&A dictionary."""
        qa_dict = {
            'question': 'What is machine learning?',
            'answer': 'Machine learning is a subset of AI...'
        }
        
        result = self.detector._detect_nlp_task_from_dict(qa_dict)
        
        assert result.task_type == 'qa'
        assert result.task_subtype == 'question_answering'
        assert result.confidence == 0.8
    
    def test_detect_nlp_task_from_dict_conversation(self):
        """Test NLP task detection from conversation dictionary."""
        conv_dict = {
            'conversation': [
                {'speaker': 'user', 'message': 'Hello'},
                {'speaker': 'bot', 'message': 'Hi there!'}
            ]
        }
        
        result = self.detector._detect_nlp_task_from_dict(conv_dict)
        
        assert result.task_type == 'conversation'
        assert result.task_subtype == 'dialogue'
        assert result.confidence == 0.8
    
    def test_detect_nlp_task_from_dict_classification(self):
        """Test NLP task detection from classification dictionary."""
        class_dict = {
            'text': 'This is a sample document',
            'label': 'positive'
        }
        
        result = self.detector._detect_nlp_task_from_dict(class_dict)
        
        assert result.task_type == 'classification'
        assert result.task_subtype == 'text_classification'
        assert result.confidence == 0.7


if __name__ == '__main__':
    pytest.main([__file__])