"""
Integration tests for multi-domain task support in NeuroLite.

Tests the complete workflow coordination for computer vision, NLP, and tabular data tasks
with appropriate preprocessing, model selection, and consistent API patterns.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurolite.workflows import (
    create_workflow, get_workflow_factory, WorkflowFactory,
    VisionWorkflow, NLPWorkflow, TabularWorkflow,
    WorkflowConfig, WorkflowResult
)
from neurolite.data import DataType
from neurolite.models import TaskType
from neurolite.core import ConfigurationError
from neurolite.api import train


class TestWorkflowFactory:
    """Test the workflow factory functionality."""
    
    def test_factory_initialization(self):
        """Test that the workflow factory initializes correctly."""
        factory = WorkflowFactory()
        
        supported_types = factory.get_supported_data_types()
        assert DataType.IMAGE in supported_types
        assert DataType.TEXT in supported_types
        assert DataType.TABULAR in supported_types
    
    def test_get_workflow_info(self):
        """Test getting workflow information for different data types."""
        factory = WorkflowFactory()
        
        # Test vision workflow info
        vision_info = factory.get_workflow_info(DataType.IMAGE)
        assert vision_info['class_name'] == 'VisionWorkflow'
        assert 'image_classification' in vision_info['supported_tasks']
        assert 'resnet18' in vision_info['default_models'].values()
        
        # Test NLP workflow info
        nlp_info = factory.get_workflow_info(DataType.TEXT)
        assert nlp_info['class_name'] == 'NLPWorkflow'
        assert 'text_classification' in nlp_info['supported_tasks']
        assert 'bert' in nlp_info['default_models'].values()
        
        # Test tabular workflow info
        tabular_info = factory.get_workflow_info(DataType.TABULAR)
        assert tabular_info['class_name'] == 'TabularWorkflow'
        assert 'classification' in tabular_info['supported_tasks']
        assert 'random_forest_classifier' in tabular_info['default_models'].values()
    
    def test_unsupported_data_type(self):
        """Test error handling for unsupported data types."""
        factory = WorkflowFactory()
        
        with pytest.raises(ConfigurationError, match="No workflow available"):
            factory.get_workflow_info(DataType.AUDIO)


class TestVisionWorkflowIntegration:
    """Test computer vision workflow integration."""
    
    @pytest.fixture
    def mock_image_data(self):
        """Create mock image data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock image files
            for i in range(10):
                class_dir = Path(temp_dir) / f"class_{i % 2}"
                class_dir.mkdir(exist_ok=True)
                
                # Create dummy image file
                image_file = class_dir / f"image_{i}.jpg"
                image_file.write_bytes(b"fake_image_data")
            
            yield temp_dir
    
    @patch('neurolite.workflows.vision.detect_data_type')
    @patch('neurolite.workflows.vision.load_data')
    @patch('neurolite.workflows.vision.validate_data')
    @patch('neurolite.workflows.vision.preprocess_data')
    @patch('neurolite.workflows.vision.split_data')
    @patch('neurolite.workflows.vision.create_model')
    @patch('neurolite.workflows.vision.TrainingEngine')
    def test_vision_workflow_execution(
        self, mock_training_engine, mock_create_model, mock_split_data,
        mock_preprocess_data, mock_validate_data, mock_load_data,
        mock_detect_data_type, mock_image_data
    ):
        """Test complete vision workflow execution."""
        # Setup mocks
        mock_detect_data_type.return_value = DataType.IMAGE
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)  # Add len() support
        mock_dataset.info = Mock()
        mock_dataset.info.num_classes = 2
        mock_dataset.info.input_shape = (3, 224, 224)
        mock_load_data.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validate_data.return_value = mock_validation_result
        
        mock_processed_dataset = Mock()
        mock_preprocess_data.return_value = mock_processed_dataset
        
        mock_data_splits = Mock()
        mock_data_splits.train = [1, 2, 3]
        mock_data_splits.validation = [4, 5]
        mock_data_splits.test = [6]
        mock_split_data.return_value = mock_data_splits
        
        mock_processed_dataset.train = mock_data_splits.train
        mock_processed_dataset.validation = mock_data_splits.validation
        mock_processed_dataset.test = mock_data_splits.test
        
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        mock_trained_model = Mock()
        mock_trained_model.training_history = {'loss': [0.5, 0.3, 0.1]}
        mock_trained_model.metadata = Mock()
        mock_trained_model.metadata.training_time = 10.0
        
        mock_engine = Mock()
        mock_engine.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine
        
        # Create workflow
        config = WorkflowConfig(
            data_path=mock_image_data,
            task="image_classification",
            model="resnet18",
            domain_config={'image_size': (224, 224), 'augmentation': True}
        )
        
        workflow = VisionWorkflow(config)
        result = workflow.execute()
        
        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.data_type == DataType.IMAGE
        assert result.task_type == TaskType.IMAGE_CLASSIFICATION
        assert result.trained_model == mock_trained_model
        assert 'train_samples' in result.preprocessing_info
        assert 'config' in result.training_info
    
    def test_vision_workflow_config_validation(self, mock_image_data):
        """Test vision workflow configuration validation."""
        # Test valid configuration
        config = WorkflowConfig(
            data_path=mock_image_data,
            domain_config={'image_size': (224, 224)}
        )
        workflow = VisionWorkflow(config)
        workflow.validate_config()  # Should not raise
        
        # Test invalid image size
        config = WorkflowConfig(
            data_path=mock_image_data,
            domain_config={'image_size': (224,)}  # Invalid: only one dimension
        )
        workflow = VisionWorkflow(config)
        
        with pytest.raises(ValueError, match="image_size must be a tuple/list of 2 integers"):
            workflow.validate_config()


class TestNLPWorkflowIntegration:
    """Test NLP workflow integration."""
    
    @pytest.fixture
    def mock_text_data(self):
        """Create mock text data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create mock CSV with text data
            df = pd.DataFrame({
                'text': [f'Sample text {i}' for i in range(100)],
                'label': [i % 3 for i in range(100)]  # 3 classes
            })
            df.to_csv(f.name, index=False)
            yield f.name
        
        os.unlink(f.name)
    
    @patch('neurolite.workflows.nlp.detect_data_type')
    @patch('neurolite.workflows.nlp.load_data')
    @patch('neurolite.workflows.nlp.validate_data')
    @patch('neurolite.workflows.nlp.preprocess_data')
    @patch('neurolite.workflows.nlp.split_data')
    @patch('neurolite.workflows.nlp.create_model')
    @patch('neurolite.workflows.nlp.TrainingEngine')
    def test_nlp_workflow_execution(
        self, mock_training_engine, mock_create_model, mock_split_data,
        mock_preprocess_data, mock_validate_data, mock_load_data,
        mock_detect_data_type, mock_text_data
    ):
        """Test complete NLP workflow execution."""
        # Setup mocks
        mock_detect_data_type.return_value = DataType.TEXT
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)  # Add len() support
        mock_dataset.info = Mock()
        mock_dataset.info.num_classes = 3
        mock_dataset.info.vocab_size = 30522
        mock_load_data.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validate_data.return_value = mock_validation_result
        
        mock_processed_dataset = Mock()
        mock_preprocess_data.return_value = mock_processed_dataset
        
        mock_data_splits = Mock()
        mock_data_splits.train = list(range(70))
        mock_data_splits.validation = list(range(70, 85))
        mock_data_splits.test = list(range(85, 100))
        mock_split_data.return_value = mock_data_splits
        
        mock_processed_dataset.train = mock_data_splits.train
        mock_processed_dataset.validation = mock_data_splits.validation
        mock_processed_dataset.test = mock_data_splits.test
        
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        mock_trained_model = Mock()
        mock_trained_model.training_history = {'loss': [1.2, 0.8, 0.4], 'val_accuracy': [0.6, 0.7, 0.8]}
        mock_trained_model.metadata = Mock()
        mock_trained_model.metadata.training_time = 15.0
        
        mock_engine = Mock()
        mock_engine.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine
        
        # Create workflow
        config = WorkflowConfig(
            data_path=mock_text_data,
            task="text_classification",
            model="bert",
            target="label",
            domain_config={'max_length': 512, 'tokenizer': 'bert-base-uncased'}
        )
        
        workflow = NLPWorkflow(config)
        result = workflow.execute()
        
        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.data_type == DataType.TEXT
        assert result.task_type == TaskType.TEXT_CLASSIFICATION
        assert result.trained_model == mock_trained_model
        assert 'max_sequence_length' in result.preprocessing_info
        assert 'tokenizer' in result.preprocessing_info
    
    def test_nlp_workflow_config_validation(self, mock_text_data):
        """Test NLP workflow configuration validation."""
        # Test valid configuration
        config = WorkflowConfig(
            data_path=mock_text_data,
            domain_config={'max_length': 512}
        )
        workflow = NLPWorkflow(config)
        workflow.validate_config()  # Should not raise
        
        # Test invalid max_length
        config = WorkflowConfig(
            data_path=mock_text_data,
            domain_config={'max_length': -1}  # Invalid: negative
        )
        workflow = NLPWorkflow(config)
        
        with pytest.raises(ValueError, match="max_length must be a positive integer"):
            workflow.validate_config()


class TestTabularWorkflowIntegration:
    """Test tabular data workflow integration."""
    
    @pytest.fixture
    def mock_tabular_data(self):
        """Create mock tabular data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create mock CSV with tabular data
            np.random.seed(42)
            df = pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'feature3': np.random.randint(0, 5, 1000),
                'feature4': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.randint(0, 2, 1000)
            })
            df.to_csv(f.name, index=False)
            yield f.name
        
        os.unlink(f.name)
    
    @patch('neurolite.workflows.tabular.detect_data_type')
    @patch('neurolite.workflows.tabular.load_data')
    @patch('neurolite.workflows.tabular.validate_data')
    @patch('neurolite.workflows.tabular.preprocess_data')
    @patch('neurolite.workflows.tabular.split_data')
    @patch('neurolite.workflows.tabular.create_model')
    @patch('neurolite.workflows.tabular.TrainingEngine')
    def test_tabular_workflow_execution(
        self, mock_training_engine, mock_create_model, mock_split_data,
        mock_preprocess_data, mock_validate_data, mock_load_data,
        mock_detect_data_type, mock_tabular_data
    ):
        """Test complete tabular workflow execution."""
        # Setup mocks
        mock_detect_data_type.return_value = DataType.TABULAR
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)  # Add len() support
        mock_dataset.info = Mock()
        mock_dataset.info.num_classes = 2
        mock_dataset.info.num_features = 4
        mock_dataset.info.target_type = 'categorical'
        mock_load_data.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validate_data.return_value = mock_validation_result
        
        mock_processed_dataset = Mock()
        mock_processed_dataset.info = Mock()
        mock_processed_dataset.info.num_features = 6  # After preprocessing
        mock_preprocess_data.return_value = mock_processed_dataset
        
        mock_data_splits = Mock()
        mock_data_splits.train = list(range(700))
        mock_data_splits.validation = list(range(700, 850))
        mock_data_splits.test = list(range(850, 1000))
        mock_split_data.return_value = mock_data_splits
        
        mock_processed_dataset.train = mock_data_splits.train
        mock_processed_dataset.validation = mock_data_splits.validation
        mock_processed_dataset.test = mock_data_splits.test
        
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        mock_trained_model = Mock()
        mock_trained_model.training_history = {'loss': [0.7], 'val_accuracy': [0.85]}
        mock_trained_model.metadata = Mock()
        mock_trained_model.metadata.training_time = 5.0
        mock_trained_model.model = Mock()
        mock_trained_model.model.feature_importances_ = [0.3, 0.2, 0.25, 0.15, 0.05, 0.05]
        
        mock_engine = Mock()
        mock_engine.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine
        
        # Create workflow
        config = WorkflowConfig(
            data_path=mock_tabular_data,
            task="classification",
            model="random_forest_classifier",
            target="target",
            domain_config={
                'feature_engineering': True,
                'scaling': 'standard',
                'categorical_encoding': 'onehot'
            }
        )
        
        workflow = TabularWorkflow(config)
        result = workflow.execute()
        
        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.data_type == DataType.TABULAR
        assert result.task_type in [TaskType.CLASSIFICATION, TaskType.BINARY_CLASSIFICATION]
        assert result.trained_model == mock_trained_model
        assert 'feature_engineering_applied' in result.preprocessing_info
        assert 'scaling_method' in result.preprocessing_info
        assert 'feature_importance' in result.training_info
    
    def test_tabular_workflow_config_validation(self, mock_tabular_data):
        """Test tabular workflow configuration validation."""
        # Test valid configuration
        config = WorkflowConfig(
            data_path=mock_tabular_data,
            domain_config={'scaling': 'standard'}
        )
        workflow = TabularWorkflow(config)
        workflow.validate_config()  # Should not raise
        
        # Test invalid scaling method
        config = WorkflowConfig(
            data_path=mock_tabular_data,
            domain_config={'scaling': 'invalid_method'}
        )
        workflow = TabularWorkflow(config)
        
        with pytest.raises(ValueError, match="scaling must be one of"):
            workflow.validate_config()


class TestMultiDomainAPIIntegration:
    """Test the main API integration with multi-domain workflows."""
    
    @pytest.fixture
    def mock_workflow_result(self):
        """Create a mock workflow result."""
        mock_trained_model = Mock()
        mock_trained_model.training_history = {'loss': [0.5, 0.3, 0.1]}
        
        return WorkflowResult(
            trained_model=mock_trained_model,
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            execution_time=10.0,
            preprocessing_info={'train_samples': 100, 'validation_samples': 20, 'test_samples': 10},
            training_info={'training_time': 8.0, 'epochs_completed': 3},
            evaluation_info={'primary_metric': 0.95, 'metrics': {'accuracy': 0.95}},
            deployment_info=None
        )
    
    @patch('neurolite.api.create_workflow')
    def test_train_with_vision_parameters(self, mock_create_workflow, mock_workflow_result):
        """Test training with vision-specific parameters."""
        mock_workflow = Mock()
        mock_workflow.execute.return_value = mock_workflow_result
        mock_workflow.__class__.__name__ = 'VisionWorkflow'
        mock_create_workflow.return_value = mock_workflow
        
        # Train with vision parameters
        result = train(
            data='path/to/images',
            model='resnet18',
            task='image_classification',
            image_size=(224, 224),
            augmentation=True,
            confidence_threshold=0.5
        )
        
        # Verify workflow creation was called with correct parameters
        mock_create_workflow.assert_called_once()
        call_args = mock_create_workflow.call_args
        
        assert call_args[1]['data_path'] == 'path/to/images'
        assert call_args[1]['model'] == 'resnet18'
        assert call_args[1]['task'] == 'image_classification'
        assert 'image_size' in call_args[1]['domain_config']
        assert 'augmentation' in call_args[1]['domain_config']
        assert 'confidence_threshold' in call_args[1]['domain_config']
        
        assert result == mock_workflow_result.trained_model
    
    @patch('neurolite.api.create_workflow')
    def test_train_with_nlp_parameters(self, mock_create_workflow, mock_workflow_result):
        """Test training with NLP-specific parameters."""
        mock_workflow_result.data_type = DataType.TEXT
        mock_workflow_result.task_type = TaskType.TEXT_CLASSIFICATION
        
        mock_workflow = Mock()
        mock_workflow.execute.return_value = mock_workflow_result
        mock_workflow.__class__.__name__ = 'NLPWorkflow'
        mock_create_workflow.return_value = mock_workflow
        
        # Train with NLP parameters
        result = train(
            data='path/to/text.csv',
            model='bert',
            task='text_classification',
            target='label',
            max_length=512,
            tokenizer='bert-base-uncased',
            temperature=0.8
        )
        
        # Verify workflow creation was called with correct parameters
        mock_create_workflow.assert_called_once()
        call_args = mock_create_workflow.call_args
        
        assert call_args[1]['data_path'] == 'path/to/text.csv'
        assert call_args[1]['model'] == 'bert'
        assert call_args[1]['task'] == 'text_classification'
        assert call_args[1]['target'] == 'label'
        assert 'max_length' in call_args[1]['domain_config']
        assert 'tokenizer' in call_args[1]['domain_config']
        assert 'temperature' in call_args[1]['domain_config']
        
        assert result == mock_workflow_result.trained_model
    
    @patch('neurolite.api.create_workflow')
    def test_train_with_tabular_parameters(self, mock_create_workflow, mock_workflow_result):
        """Test training with tabular-specific parameters."""
        mock_workflow_result.data_type = DataType.TABULAR
        mock_workflow_result.task_type = TaskType.CLASSIFICATION
        
        mock_workflow = Mock()
        mock_workflow.execute.return_value = mock_workflow_result
        mock_workflow.__class__.__name__ = 'TabularWorkflow'
        mock_create_workflow.return_value = mock_workflow
        
        # Train with tabular parameters
        result = train(
            data='path/to/data.csv',
            model='random_forest',
            task='classification',
            target='target_column',
            feature_engineering=True,
            scaling='standard',
            categorical_encoding='onehot',
            balance_classes=True
        )
        
        # Verify workflow creation was called with correct parameters
        mock_create_workflow.assert_called_once()
        call_args = mock_create_workflow.call_args
        
        assert call_args[1]['data_path'] == 'path/to/data.csv'
        assert call_args[1]['model'] == 'random_forest'
        assert call_args[1]['task'] == 'classification'
        assert call_args[1]['target'] == 'target_column'
        assert 'feature_engineering' in call_args[1]['domain_config']
        assert 'scaling' in call_args[1]['domain_config']
        assert 'categorical_encoding' in call_args[1]['domain_config']
        assert 'balance_classes' in call_args[1]['domain_config']
        
        assert result == mock_workflow_result.trained_model


class TestConsistentAPIPatterns:
    """Test that all workflows maintain consistent API patterns."""
    
    def test_workflow_base_interface_consistency(self):
        """Test that all workflows implement the base interface consistently."""
        workflows = [VisionWorkflow, NLPWorkflow, TabularWorkflow]
        
        for workflow_class in workflows:
            # Create dummy config
            config = WorkflowConfig(data_path="dummy")
            workflow = workflow_class(config)
            
            # Check required properties exist
            assert hasattr(workflow, 'supported_data_types')
            assert hasattr(workflow, 'supported_tasks')
            assert hasattr(workflow, 'default_models')
            
            # Check properties return correct types
            assert isinstance(workflow.supported_data_types, list)
            assert isinstance(workflow.supported_tasks, list)
            assert isinstance(workflow.default_models, dict)
            
            # Check required methods exist
            assert hasattr(workflow, 'execute')
            assert hasattr(workflow, 'validate_config')
            assert callable(workflow.execute)
            assert callable(workflow.validate_config)
    
    def test_workflow_config_parameter_consistency(self):
        """Test that all workflows handle configuration parameters consistently."""
        base_config = WorkflowConfig(
            data_path="dummy",
            model="auto",
            task="auto",
            validation_split=0.2,
            test_split=0.1,
            optimize=True,
            deploy=False
        )
        
        workflows = [VisionWorkflow, NLPWorkflow, TabularWorkflow]
        
        for workflow_class in workflows:
            workflow = workflow_class(base_config)
            
            # Check that config is stored
            assert workflow.config == base_config
            
            # Check that validation doesn't raise for base config
            # (Note: may raise for missing data path, but not for parameter structure)
            try:
                workflow.validate_config()
            except (FileNotFoundError, ValueError) as e:
                # Expected for dummy path, check it's not a config structure error
                assert "does not exist" in str(e) or "Data path" in str(e)
    
    def test_workflow_result_structure_consistency(self):
        """Test that all workflows return results with consistent structure."""
        # This test would need to be run with actual workflow execution
        # For now, we test the WorkflowResult structure
        
        from neurolite.training import TrainedModel
        
        mock_trained_model = Mock(spec=TrainedModel)
        
        result = WorkflowResult(
            trained_model=mock_trained_model,
            data_type=DataType.IMAGE,
            task_type=TaskType.IMAGE_CLASSIFICATION,
            execution_time=10.0,
            preprocessing_info={'key': 'value'},
            training_info={'key': 'value'},
            evaluation_info={'key': 'value'}
        )
        
        # Check required fields exist
        assert hasattr(result, 'trained_model')
        assert hasattr(result, 'data_type')
        assert hasattr(result, 'task_type')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'preprocessing_info')
        assert hasattr(result, 'training_info')
        assert hasattr(result, 'evaluation_info')
        assert hasattr(result, 'deployment_info')
        assert hasattr(result, 'metadata')
        
        # Check types
        assert isinstance(result.preprocessing_info, dict)
        assert isinstance(result.training_info, dict)
        assert isinstance(result.evaluation_info, dict)
        assert isinstance(result.metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__])