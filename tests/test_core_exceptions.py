"""
Tests for core exception handling.
"""

import pytest

from neurolite.core.exceptions import (
    NeuroLiteError,
    DataError,
    DataNotFoundError,
    DataFormatError,
    DataValidationError,
    ModelError,
    ModelNotFoundError,
    ModelCompatibilityError,
    ModelLoadError,
    TrainingError,
    TrainingConfigurationError,
    TrainingFailedError,
    EvaluationError,
    MetricError,
    DeploymentError,
    ExportError,
    APIServerError,
    ConfigurationError,
    InvalidConfigurationError,
    DependencyError,
    MissingDependencyError,
    ResourceError,
    InsufficientMemoryError,
    GPUError
)


class TestNeuroLiteError:
    """Test base NeuroLiteError class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = NeuroLiteError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details is None
        assert error.suggestions == []
        assert error.error_code is None
        assert error.context == {}
    
    def test_error_with_details(self):
        """Test error with additional details."""
        error = NeuroLiteError(
            "Test error",
            details="Additional details",
            suggestions=["Try this", "Or this"],
            error_code="TEST_ERROR",
            context={"key": "value"}
        )
        
        assert "Test error" in str(error)
        assert "Additional details" in str(error)
        assert "Try this" in str(error)
        assert "Or this" in str(error)
        
        assert error.message == "Test error"
        assert error.details == "Additional details"
        assert error.suggestions == ["Try this", "Or this"]
        assert error.error_code == "TEST_ERROR"
        assert error.context == {"key": "value"}


class TestDataErrors:
    """Test data-related errors."""
    
    def test_data_error(self):
        """Test basic DataError."""
        error = DataError("Data processing failed")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "DATA_ERROR"
    
    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        error = DataNotFoundError("/path/to/data")
        
        assert isinstance(error, DataError)
        assert error.error_code == "DATA_NOT_FOUND"
        assert "/path/to/data" in str(error)
        assert "Check if the file or directory path is correct" in str(error)
        assert error.context["data_path"] == "/path/to/data"
    
    def test_data_format_error(self):
        """Test DataFormatError."""
        error = DataFormatError("xyz", ["csv", "json", "parquet"])
        
        assert isinstance(error, DataError)
        assert error.error_code == "DATA_FORMAT_ERROR"
        assert "xyz" in str(error)
        assert "csv, json, parquet" in str(error)
        assert error.context["format_type"] == "xyz"
        assert error.context["supported_formats"] == ["csv", "json", "parquet"]
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        validation_errors = ["Missing values in column A", "Invalid data type in column B"]
        error = DataValidationError(validation_errors)
        
        assert isinstance(error, DataError)
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert "Missing values in column A" in str(error)
        assert "Invalid data type in column B" in str(error)
        assert error.context["validation_errors"] == validation_errors


class TestModelErrors:
    """Test model-related errors."""
    
    def test_model_error(self):
        """Test basic ModelError."""
        error = ModelError("Model failed")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "MODEL_ERROR"
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        available_models = ["resnet", "vgg", "mobilenet"]
        error = ModelNotFoundError("alexnet", available_models)
        
        assert isinstance(error, ModelError)
        assert error.error_code == "MODEL_NOT_FOUND"
        assert "alexnet" in str(error)
        assert "resnet, vgg, mobilenet" in str(error)
        assert error.context["model_name"] == "alexnet"
        assert error.context["available_models"] == available_models
    
    def test_model_compatibility_error(self):
        """Test ModelCompatibilityError."""
        error = ModelCompatibilityError("bert", "classification", "image")
        
        assert isinstance(error, ModelError)
        assert error.error_code == "MODEL_COMPATIBILITY_ERROR"
        assert "bert" in str(error)
        assert "classification" in str(error)
        assert "image" in str(error)
        assert error.context["model_name"] == "bert"
        assert error.context["task_type"] == "classification"
        assert error.context["data_type"] == "image"
    
    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError("resnet50", "Insufficient memory")
        
        assert isinstance(error, ModelError)
        assert error.error_code == "MODEL_LOAD_ERROR"
        assert "resnet50" in str(error)
        assert "Insufficient memory" in str(error)
        assert error.context["model_name"] == "resnet50"
        assert error.context["reason"] == "Insufficient memory"


class TestTrainingErrors:
    """Test training-related errors."""
    
    def test_training_error(self):
        """Test basic TrainingError."""
        error = TrainingError("Training failed")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "TRAINING_ERROR"
    
    def test_training_configuration_error(self):
        """Test TrainingConfigurationError."""
        config_errors = ["Invalid batch size", "Learning rate too high"]
        error = TrainingConfigurationError(config_errors)
        
        assert isinstance(error, TrainingError)
        assert error.error_code == "TRAINING_CONFIG_ERROR"
        assert "Invalid batch size" in str(error)
        assert "Learning rate too high" in str(error)
        assert error.context["config_errors"] == config_errors
    
    def test_training_failed_error(self):
        """Test TrainingFailedError."""
        error = TrainingFailedError("NaN loss detected", epoch=15)
        
        assert isinstance(error, TrainingError)
        assert error.error_code == "TRAINING_FAILED"
        assert "NaN loss detected" in str(error)
        assert "epoch 15" in str(error)
        assert error.context["reason"] == "NaN loss detected"
        assert error.context["epoch"] == 15
    
    def test_training_failed_error_no_epoch(self):
        """Test TrainingFailedError without epoch."""
        error = TrainingFailedError("Data loading failed")
        
        assert isinstance(error, TrainingError)
        assert error.error_code == "TRAINING_FAILED"
        assert "Data loading failed" in str(error)
        assert "epoch" not in error.context


class TestEvaluationErrors:
    """Test evaluation-related errors."""
    
    def test_evaluation_error(self):
        """Test basic EvaluationError."""
        error = EvaluationError("Evaluation failed")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "EVALUATION_ERROR"
    
    def test_metric_error(self):
        """Test MetricError."""
        error = MetricError("accuracy", "Mismatched array shapes")
        
        assert isinstance(error, EvaluationError)
        assert error.error_code == "METRIC_ERROR"
        assert "accuracy" in str(error)
        assert "Mismatched array shapes" in str(error)
        assert error.context["metric_name"] == "accuracy"
        assert error.context["reason"] == "Mismatched array shapes"


class TestDeploymentErrors:
    """Test deployment-related errors."""
    
    def test_deployment_error(self):
        """Test basic DeploymentError."""
        error = DeploymentError("Deployment failed")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "DEPLOYMENT_ERROR"
    
    def test_export_error(self):
        """Test ExportError."""
        error = ExportError("onnx", "Unsupported operation")
        
        assert isinstance(error, DeploymentError)
        assert error.error_code == "EXPORT_ERROR"
        assert "onnx" in str(error)
        assert "Unsupported operation" in str(error)
        assert error.context["format_type"] == "onnx"
        assert error.context["reason"] == "Unsupported operation"
    
    def test_api_server_error(self):
        """Test APIServerError."""
        error = APIServerError("Port already in use")
        
        assert isinstance(error, DeploymentError)
        assert error.error_code == "API_SERVER_ERROR"
        assert "Port already in use" in str(error)
        assert error.context["reason"] == "Port already in use"


class TestConfigurationErrors:
    """Test configuration-related errors."""
    
    def test_configuration_error(self):
        """Test basic ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "CONFIGURATION_ERROR"
    
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        errors = ["Missing required field", "Invalid value type"]
        error = InvalidConfigurationError("config.yaml", errors)
        
        assert isinstance(error, ConfigurationError)
        assert error.error_code == "INVALID_CONFIG"
        assert "config.yaml" in str(error)
        assert "Missing required field" in str(error)
        assert "Invalid value type" in str(error)
        assert error.context["config_path"] == "config.yaml"
        assert error.context["errors"] == errors


class TestDependencyErrors:
    """Test dependency-related errors."""
    
    def test_dependency_error(self):
        """Test basic DependencyError."""
        error = DependencyError("Dependency issue")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "DEPENDENCY_ERROR"
    
    def test_missing_dependency_error(self):
        """Test MissingDependencyError."""
        error = MissingDependencyError("torch", "deep learning", "pip install torch")
        
        assert isinstance(error, DependencyError)
        assert error.error_code == "MISSING_DEPENDENCY"
        assert "torch" in str(error)
        assert "deep learning" in str(error)
        assert "pip install torch" in str(error)
        assert error.context["dependency"] == "torch"
        assert error.context["feature"] == "deep learning"
        assert error.context["install_command"] == "pip install torch"


class TestResourceErrors:
    """Test resource-related errors."""
    
    def test_resource_error(self):
        """Test basic ResourceError."""
        error = ResourceError("Resource issue")
        
        assert isinstance(error, NeuroLiteError)
        assert error.error_code == "RESOURCE_ERROR"
    
    def test_insufficient_memory_error(self):
        """Test InsufficientMemoryError."""
        error = InsufficientMemoryError("8GB", "4GB")
        
        assert isinstance(error, ResourceError)
        assert error.error_code == "INSUFFICIENT_MEMORY"
        assert "8GB" in str(error)
        assert "4GB" in str(error)
        assert error.context["required_memory"] == "8GB"
        assert error.context["available_memory"] == "4GB"
    
    def test_gpu_error(self):
        """Test GPUError."""
        error = GPUError("CUDA out of memory")
        
        assert isinstance(error, ResourceError)
        assert error.error_code == "GPU_ERROR"
        assert "CUDA out of memory" in str(error)
        assert error.context["reason"] == "CUDA out of memory"


if __name__ == '__main__':
    pytest.main([__file__])