"""
Integration tests for the main NeuroLite API.

Tests the primary train() function and deploy() function with various
configurations and data types to ensure proper workflow coordination.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurolite.api import train, deploy
from neurolite.core import (
    NeuroLiteError, DataError, ModelError, TrainingError, ConfigurationError
)
from neurolite.data import DataType
from neurolite.models import TaskType
from neurolite.training import TrainedModel


class TestAPIParameterValidation:
    """Test parameter validation in the main API."""
    
    def test_invalid_data_path(self):
        """Test error handling for non-existent data path."""
        with pytest.raises(ConfigurationError) as exc_info:
            train("non_existent_path.csv")
        
        assert "Data path does not exist" in str(exc_info.value)
        assert "Please check that the file or directory exists" in str(exc_info.value)
    
    def test_invalid_validation_split(self):
        """Test error handling for invalid validation split values."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test negative validation split
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, validation_split=-0.1)
            assert "validation_split must be between 0.0 and 1.0" in str(exc_info.value)
            
            # Test validation split > 1.0
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, validation_split=1.5)
            assert "validation_split must be between 0.0 and 1.0" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_test_split(self):
        """Test error handling for invalid test split values."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test negative test split
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, test_split=-0.1)
            assert "test_split must be between 0.0 and 1.0" in str(exc_info.value)
            
            # Test test split > 1.0
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, test_split=1.5)
            assert "test_split must be between 0.0 and 1.0" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_splits_sum_too_large(self):
        """Test error handling when validation + test splits >= 1.0."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, validation_split=0.6, test_split=0.5)
            assert "must be less than 1.0" in str(exc_info.value)
            assert "reduce splits so training data is at least 60%" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    @patch('neurolite.api.get_model_registry')
    def test_invalid_model_type(self, mock_registry):
        """Test error handling for invalid model type."""
        mock_registry.return_value.list_models.return_value = ['model1', 'model2']
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, model='invalid_model')
            assert "Unknown model type: invalid_model" in str(exc_info.value)
            assert "Available models:" in str(exc_info.value)
            assert "Use 'auto' for automatic model selection" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_task_type(self):
        """Test error handling for invalid task type."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                train(temp_path, task='invalid_task')
            assert "Unknown task type: invalid_task" in str(exc_info.value)
            assert "Available tasks:" in str(exc_info.value)
            assert "Use 'auto' for automatic task detection" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestAPIWorkflowIntegration:
    """Test the complete training workflow integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample CSV data
        self.csv_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.csv_data.to_csv(self.csv_path, index=False)
        
        # Create sample image directory structure
        self.image_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(os.path.join(self.image_dir, 'class1'))
        os.makedirs(os.path.join(self.image_dir, 'class2'))
        
        # Create dummy image files
        for i in range(10):
            Path(os.path.join(self.image_dir, 'class1', f'img_{i}.jpg')).touch()
            Path(os.path.join(self.image_dir, 'class2', f'img_{i}.jpg')).touch()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('neurolite.api.detect_data_type')
    @patch('neurolite.api.load_data')
    @patch('neurolite.api.validate_data')
    @patch('neurolite.api.preprocess_data')
    @patch('neurolite.api.split_data')
    @patch('neurolite.api.create_model')
    @patch('neurolite.api.TrainingEngine')
    @patch('neurolite.api.evaluate_model')
    def test_complete_training_workflow_tabular(
        self, mock_evaluate, mock_training_engine, mock_create_model,
        mock_split_data, mock_preprocess, mock_validate, mock_load, mock_detect
    ):
        """Test complete training workflow for tabular data."""
        # Mock the workflow components
        mock_detect.return_value = DataType.TABULAR
        
        mock_dataset = Mock()
        mock_dataset.info.target_type = 'categorical'
        mock_load.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validate.return_value = mock_validation_result
        
        mock_processed_dataset = Mock()
        mock_preprocess.return_value = mock_processed_dataset
        
        mock_splits = Mock()
        mock_splits.train = [1, 2, 3, 4, 5]  # Mock training data
        mock_splits.validation = [6, 7]      # Mock validation data
        mock_splits.test = [8, 9]            # Mock test data
        mock_split_data.return_value = mock_splits
        
        mock_model_instance = Mock()
        mock_create_model.return_value = mock_model_instance
        
        mock_trained_model = Mock(spec=TrainedModel)
        mock_engine_instance = Mock()
        mock_engine_instance.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine_instance
        
        mock_evaluation_results = Mock()
        mock_evaluation_results.primary_metric = 0.85
        mock_evaluate.return_value = mock_evaluation_results
        
        # Execute training
        result = train(
            self.csv_path,
            model='random_forest_classifier',
            task='classification',
            target='target'
        )
        
        # Verify workflow steps were called
        mock_detect.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_split_data.assert_called_once()
        mock_create_model.assert_called_once()
        mock_engine_instance.train.assert_called_once()
        mock_evaluate.assert_called_once()
        
        # Verify result
        assert result == mock_trained_model
        assert result.evaluation_results == mock_evaluation_results
    
    @patch('neurolite.api.detect_data_type')
    @patch('neurolite.api.load_data')
    @patch('neurolite.api.validate_data')
    @patch('neurolite.api.clean_data')
    @patch('neurolite.api.preprocess_data')
    @patch('neurolite.api.split_data')
    @patch('neurolite.api.create_model')
    @patch('neurolite.api.TrainingEngine')
    @patch('neurolite.api.evaluate_model')
    def test_workflow_with_data_cleaning(
        self, mock_evaluate, mock_training_engine, mock_create_model,
        mock_split_data, mock_preprocess, mock_clean, mock_validate, 
        mock_load, mock_detect
    ):
        """Test workflow when data cleaning is required."""
        # Mock data validation failure
        mock_detect.return_value = DataType.TABULAR
        
        mock_dataset = Mock()
        mock_load.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.issues = ['missing_values', 'outliers']
        mock_validate.return_value = mock_validation_result
        
        mock_cleaned_dataset = Mock()
        mock_clean.return_value = mock_cleaned_dataset
        
        mock_processed_dataset = Mock()
        mock_preprocess.return_value = mock_processed_dataset
        
        mock_splits = Mock()
        mock_splits.train = [1, 2, 3]
        mock_splits.validation = [4]
        mock_splits.test = [5]
        mock_split_data.return_value = mock_splits
        
        mock_model_instance = Mock()
        mock_create_model.return_value = mock_model_instance
        
        mock_trained_model = Mock(spec=TrainedModel)
        mock_engine_instance = Mock()
        mock_engine_instance.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine_instance
        
        mock_evaluation_results = Mock()
        mock_evaluation_results.primary_metric = 0.75
        mock_evaluate.return_value = mock_evaluation_results
        
        # Execute training
        result = train(self.csv_path)
        
        # Verify data cleaning was called
        mock_validate.assert_called_once()
        mock_clean.assert_called_once_with(mock_dataset)
        
        # Verify preprocessing used cleaned data
        mock_preprocess.assert_called_once_with(mock_cleaned_dataset, TaskType.CLASSIFICATION)
    
    @patch('neurolite.api.detect_data_type')
    @patch('neurolite.api.load_data')
    @patch('neurolite.api.validate_data')
    @patch('neurolite.api.preprocess_data')
    @patch('neurolite.api.split_data')
    @patch('neurolite.api.create_model')
    @patch('neurolite.api.TrainingEngine')
    @patch('neurolite.api.evaluate_model')
    @patch('neurolite.api.create_api_server')
    def test_workflow_with_deployment(
        self, mock_create_api, mock_evaluate, mock_training_engine, 
        mock_create_model, mock_split_data, mock_preprocess, 
        mock_validate, mock_load, mock_detect
    ):
        """Test workflow with deployment artifacts creation."""
        # Mock successful training workflow
        mock_detect.return_value = DataType.TABULAR
        
        mock_dataset = Mock()
        mock_load.return_value = mock_dataset
        
        mock_validation_result = Mock()
        mock_validation_result.is_valid = True
        mock_validate.return_value = mock_validation_result
        
        mock_processed_dataset = Mock()
        mock_preprocess.return_value = mock_processed_dataset
        
        mock_splits = Mock()
        mock_splits.train = [1, 2, 3]
        mock_splits.validation = [4]
        mock_splits.test = [5]
        mock_split_data.return_value = mock_splits
        
        mock_model_instance = Mock()
        mock_create_model.return_value = mock_model_instance
        
        mock_trained_model = Mock(spec=TrainedModel)
        mock_engine_instance = Mock()
        mock_engine_instance.train.return_value = mock_trained_model
        mock_training_engine.return_value = mock_engine_instance
        
        mock_evaluation_results = Mock()
        mock_evaluation_results.primary_metric = 0.80
        mock_evaluate.return_value = mock_evaluation_results
        
        mock_api_server = Mock()
        mock_create_api.return_value = mock_api_server
        
        # Execute training with deployment
        result = train(self.csv_path, deploy=True)
        
        # Verify deployment artifacts were created
        mock_create_api.assert_called_once_with(mock_trained_model)
        assert hasattr(result, 'deployment_info')
        assert result.deployment_info['api_server'] == mock_api_server


class TestAPITaskDetection:
    """Test automatic task detection logic."""
    
    def test_detect_task_image_data(self):
        """Test task detection for image data."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        task_type = _detect_task_from_data(DataType.IMAGE, mock_dataset_info, "auto")
        assert task_type == TaskType.IMAGE_CLASSIFICATION
    
    def test_detect_task_text_data(self):
        """Test task detection for text data."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        task_type = _detect_task_from_data(DataType.TEXT, mock_dataset_info, "auto")
        assert task_type == TaskType.TEXT_CLASSIFICATION
    
    def test_detect_task_tabular_numeric(self):
        """Test task detection for tabular data with numeric target."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        mock_dataset_info.target_type = 'numeric'
        task_type = _detect_task_from_data(DataType.TABULAR, mock_dataset_info, "auto")
        assert task_type == TaskType.REGRESSION
    
    def test_detect_task_tabular_categorical(self):
        """Test task detection for tabular data with categorical target."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        mock_dataset_info.target_type = 'categorical'
        task_type = _detect_task_from_data(DataType.TABULAR, mock_dataset_info, "auto")
        assert task_type == TaskType.CLASSIFICATION
    
    def test_detect_task_explicit_task(self):
        """Test explicit task specification overrides auto-detection."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        task_type = _detect_task_from_data(DataType.IMAGE, mock_dataset_info, "regression")
        assert task_type == TaskType.REGRESSION
    
    def test_detect_task_unsupported_data_type(self):
        """Test error handling for unsupported data types."""
        from neurolite.api import _detect_task_from_data
        
        mock_dataset_info = Mock()
        with pytest.raises(ConfigurationError) as exc_info:
            _detect_task_from_data(DataType.AUDIO, mock_dataset_info, "auto")
        
        assert "Cannot auto-detect task for data type: DataType.AUDIO" in str(exc_info.value)
        assert "Please specify task explicitly" in str(exc_info.value)


class TestAPIModelSelection:
    """Test automatic model selection logic."""
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_image_classification(self, mock_registry):
        """Test model selection for image classification."""
        from neurolite.api import _select_model
        
        mock_registry.return_value.list_models.return_value = ['resnet18', 'vgg16']
        
        model = _select_model("auto", DataType.IMAGE, TaskType.IMAGE_CLASSIFICATION)
        assert model == "resnet18"
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_text_classification(self, mock_registry):
        """Test model selection for text classification."""
        from neurolite.api import _select_model
        
        mock_registry.return_value.list_models.return_value = ['bert', 'roberta']
        
        model = _select_model("auto", DataType.TEXT, TaskType.TEXT_CLASSIFICATION)
        assert model == "bert"
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_tabular_regression(self, mock_registry):
        """Test model selection for tabular regression."""
        from neurolite.api import _select_model
        
        mock_registry.return_value.list_models.return_value = ['random_forest_regressor', 'xgboost_regressor']
        
        model = _select_model("auto", DataType.TABULAR, TaskType.REGRESSION)
        assert model == "random_forest_regressor"
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_explicit_model(self, mock_registry):
        """Test explicit model specification overrides auto-selection."""
        from neurolite.api import _select_model
        
        model = _select_model("custom_model", DataType.IMAGE, TaskType.IMAGE_CLASSIFICATION)
        assert model == "custom_model"
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_fallback(self, mock_registry):
        """Test fallback model selection when no specific match."""
        from neurolite.api import _select_model
        
        mock_registry.return_value.list_models.return_value = ['fallback_model']
        
        model = _select_model("auto", DataType.AUDIO, TaskType.CLASSIFICATION)
        assert model == "fallback_model"
    
    @patch('neurolite.api.get_model_registry')
    def test_select_model_no_available_models(self, mock_registry):
        """Test error handling when no models are available."""
        from neurolite.api import _select_model
        
        mock_registry.return_value.list_models.return_value = []
        
        with pytest.raises(ModelError) as exc_info:
            _select_model("auto", DataType.AUDIO, TaskType.CLASSIFICATION)
        
        assert "No suitable model found" in str(exc_info.value)
        assert "Please specify a model explicitly" in str(exc_info.value)


class TestAPIErrorHandling:
    """Test error handling and informative error messages."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create minimal CSV file
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(self.csv_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('neurolite.api.detect_data_type')
    def test_data_error_propagation(self, mock_detect):
        """Test that DataError is properly propagated."""
        mock_detect.side_effect = DataError("Test data error")
        
        with pytest.raises(DataError) as exc_info:
            train(self.csv_path)
        
        assert "Test data error" in str(exc_info.value)
    
    @patch('neurolite.api.detect_data_type')
    @patch('neurolite.api.load_data')
    def test_model_error_propagation(self, mock_load, mock_detect):
        """Test that ModelError is properly propagated."""
        mock_detect.return_value = DataType.TABULAR
        mock_load.side_effect = ModelError("Test model error")
        
        with pytest.raises(ModelError) as exc_info:
            train(self.csv_path)
        
        assert "Test model error" in str(exc_info.value)
    
    @patch('neurolite.api.detect_data_type')
    @patch('neurolite.api.load_data')
    def test_unexpected_error_wrapping(self, mock_load, mock_detect):
        """Test that unexpected errors are wrapped with helpful context."""
        mock_detect.return_value = DataType.TABULAR
        mock_load.side_effect = RuntimeError("Unexpected runtime error")
        
        with pytest.raises(NeuroLiteError) as exc_info:
            train(self.csv_path)
        
        error_msg = str(exc_info.value)
        assert "Unexpected error during training" in error_msg
        assert "Incompatible data format or corrupted files" in error_msg
        assert "Insufficient system resources" in error_msg
        assert "Missing dependencies" in error_msg
        assert "Network issues" in error_msg


class TestDeployFunction:
    """Test the deploy() function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_trained_model = Mock(spec=TrainedModel)
    
    @patch('neurolite.api.create_api_server')
    def test_deploy_api_format(self, mock_create_api):
        """Test deploying model as API server."""
        mock_api_server = Mock()
        mock_create_api.return_value = mock_api_server
        
        result = deploy(self.mock_trained_model, format="api", host="localhost", port=8080)
        
        mock_create_api.assert_called_once_with(
            self.mock_trained_model, 
            host="localhost", 
            port=8080
        )
        assert result == mock_api_server
    
    @patch('neurolite.api.ModelExporter')
    def test_deploy_export_format(self, mock_exporter_class):
        """Test deploying model by exporting to specific format."""
        mock_exporter = Mock()
        mock_exported_model = Mock()
        mock_exporter.export.return_value = mock_exported_model
        mock_exporter_class.return_value = mock_exporter
        
        result = deploy(self.mock_trained_model, format="onnx")
        
        mock_exporter.export.assert_called_once_with(self.mock_trained_model, "onnx")
        assert result == mock_exported_model
    
    @patch('neurolite.api.create_api_server')
    def test_deploy_error_handling(self, mock_create_api):
        """Test error handling in deploy function."""
        mock_create_api.side_effect = RuntimeError("Server creation failed")
        
        with pytest.raises(NeuroLiteError) as exc_info:
            deploy(self.mock_trained_model, format="api")
        
        error_msg = str(exc_info.value)
        assert "Failed to deploy model in api format" in error_msg
        assert "Unsupported export format" in error_msg
        assert "Missing dependencies" in error_msg
        assert "Port already in use" in error_msg


if __name__ == "__main__":
    pytest.main([__file__])