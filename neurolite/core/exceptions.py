"""
Custom exception hierarchy for NeuroLite.

Provides structured error handling with informative messages
and suggested solutions for common issues.
"""

from typing import Optional, Dict, Any, List


class NeuroLiteError(Exception):
    """
    Base exception for all NeuroLite errors.
    
    Provides structured error information with context and suggestions.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NeuroLite error.
        
        Args:
            message: Primary error message
            details: Additional error details
            suggestions: List of suggested solutions
            error_code: Unique error code for programmatic handling
            context: Additional context information
        """
        self.message = message
        self.details = details
        self.suggestions = suggestions or []
        self.error_code = error_code
        self.context = context or {}
        
        # Build full error message
        full_message = message
        if details:
            full_message += f"\nDetails: {details}"
        if suggestions:
            full_message += f"\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
        
        super().__init__(full_message)


class DataError(NeuroLiteError):
    """Data-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'DATA_ERROR')
        super().__init__(message, **kwargs)


class DataNotFoundError(DataError):
    """Raised when specified data cannot be found."""
    
    def __init__(self, data_path: str, **kwargs):
        message = f"Data not found at path: {data_path}"
        suggestions = [
            "Check if the file or directory path is correct",
            "Ensure the file exists and is accessible",
            "Verify file permissions allow reading"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="DATA_NOT_FOUND",
            context={"data_path": data_path},
            **kwargs
        )


class DataFormatError(DataError):
    """Raised when data format is not supported or invalid."""
    
    def __init__(self, format_type: str, supported_formats: List[str], **kwargs):
        message = f"Unsupported data format: {format_type}"
        details = f"Supported formats: {', '.join(supported_formats)}"
        suggestions = [
            f"Convert your data to one of the supported formats: {', '.join(supported_formats)}",
            "Check if the file extension matches the actual file format",
            "Ensure the data file is not corrupted"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="DATA_FORMAT_ERROR",
            context={"format_type": format_type, "supported_formats": supported_formats},
            **kwargs
        )


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(self, validation_errors: List[str], **kwargs):
        message = "Data validation failed"
        details = "Validation errors:\n" + "\n".join(f"  - {error}" for error in validation_errors)
        suggestions = [
            "Check your data for missing values, incorrect types, or invalid ranges",
            "Review the data preprocessing requirements for your task",
            "Consider using data cleaning utilities to fix common issues"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="DATA_VALIDATION_ERROR",
            context={"validation_errors": validation_errors},
            **kwargs
        )


class ModelError(NeuroLiteError):
    """Model-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'MODEL_ERROR')
        super().__init__(message, **kwargs)


class ModelNotFoundError(ModelError):
    """Raised when specified model cannot be found."""
    
    def __init__(self, model_name: str, available_models: List[str], **kwargs):
        message = f"Model not found: {model_name}"
        details = f"Available models: {', '.join(available_models)}"
        suggestions = [
            f"Use one of the available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}",
            "Check if the model name is spelled correctly",
            "Use 'auto' to let NeuroLite select the best model automatically"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="MODEL_NOT_FOUND",
            context={"model_name": model_name, "available_models": available_models},
            **kwargs
        )


class ModelCompatibilityError(ModelError):
    """Raised when model is not compatible with the data or task."""
    
    def __init__(self, model_name: str, task_type: str, data_type: str, **kwargs):
        message = f"Model '{model_name}' is not compatible with task '{task_type}' and data type '{data_type}'"
        suggestions = [
            "Use 'auto' model selection to find a compatible model",
            "Check the model documentation for supported tasks and data types",
            "Consider using a different model that supports your task and data type"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="MODEL_COMPATIBILITY_ERROR",
            context={"model_name": model_name, "task_type": task_type, "data_type": data_type},
            **kwargs
        )


class ModelLoadError(ModelError):
    """Raised when model fails to load."""
    
    def __init__(self, model_name: str, reason: str, **kwargs):
        message = f"Failed to load model '{model_name}': {reason}"
        suggestions = [
            "Check if all required dependencies are installed",
            "Ensure sufficient memory is available",
            "Try clearing the model cache and downloading again",
            "Check your internet connection if the model needs to be downloaded"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="MODEL_LOAD_ERROR",
            context={"model_name": model_name, "reason": reason},
            **kwargs
        )


class TrainingError(NeuroLiteError):
    """Training-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'TRAINING_ERROR')
        super().__init__(message, **kwargs)


class TrainingConfigurationError(TrainingError):
    """Raised when training configuration is invalid."""
    
    def __init__(self, config_errors: List[str], **kwargs):
        message = "Invalid training configuration"
        details = "Configuration errors:\n" + "\n".join(f"  - {error}" for error in config_errors)
        suggestions = [
            "Check the training parameters for valid ranges and types",
            "Ensure batch size is appropriate for your data and memory",
            "Verify learning rate and other hyperparameters are reasonable"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="TRAINING_CONFIG_ERROR",
            context={"config_errors": config_errors},
            **kwargs
        )


class TrainingFailedError(TrainingError):
    """Raised when training process fails."""
    
    def __init__(self, reason: str, epoch: Optional[int] = None, **kwargs):
        message = f"Training failed: {reason}"
        context = {"reason": reason}
        if epoch is not None:
            message += f" at epoch {epoch}"
            context["epoch"] = epoch
        
        suggestions = [
            "Check if your data is properly formatted and not corrupted",
            "Try reducing the learning rate or batch size",
            "Ensure sufficient memory and computational resources",
            "Check for NaN or infinite values in your data"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="TRAINING_FAILED",
            context=context,
            **kwargs
        )


class EvaluationError(NeuroLiteError):
    """Evaluation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'EVALUATION_ERROR')
        super().__init__(message, **kwargs)


class MetricError(EvaluationError):
    """Raised when metric calculation fails."""
    
    def __init__(self, metric_name: str, reason: str, **kwargs):
        message = f"Failed to calculate metric '{metric_name}': {reason}"
        suggestions = [
            "Check if the metric is appropriate for your task type",
            "Ensure predictions and ground truth have compatible shapes",
            "Verify that class labels are properly encoded"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="METRIC_ERROR",
            context={"metric_name": metric_name, "reason": reason},
            **kwargs
        )


class DeploymentError(NeuroLiteError):
    """Deployment-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'DEPLOYMENT_ERROR')
        super().__init__(message, **kwargs)


class ExportError(DeploymentError):
    """Raised when model export fails."""
    
    def __init__(self, format_type: str, reason: str, **kwargs):
        message = f"Failed to export model to {format_type}: {reason}"
        suggestions = [
            "Check if the export format is supported for your model type",
            "Ensure all required dependencies for the export format are installed",
            "Try exporting to a different format",
            "Check if the model is properly trained and saved"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="EXPORT_ERROR",
            context={"format_type": format_type, "reason": reason},
            **kwargs
        )


class APIServerError(DeploymentError):
    """Raised when API server fails to start or operate."""
    
    def __init__(self, reason: str, **kwargs):
        message = f"API server error: {reason}"
        suggestions = [
            "Check if the specified port is available",
            "Ensure the model is properly loaded and functional",
            "Verify server configuration parameters",
            "Check system resources and permissions"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="API_SERVER_ERROR",
            context={"reason": reason},
            **kwargs
        )


class ConfigurationError(NeuroLiteError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'CONFIGURATION_ERROR')
        super().__init__(message, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_path: str, errors: List[str], **kwargs):
        message = f"Invalid configuration in {config_path}"
        details = "Configuration errors:\n" + "\n".join(f"  - {error}" for error in errors)
        suggestions = [
            "Check the configuration file syntax (YAML/JSON)",
            "Verify all required configuration fields are present",
            "Ensure configuration values are within valid ranges",
            "Refer to the documentation for configuration examples"
        ]
        super().__init__(
            message,
            details=details,
            suggestions=suggestions,
            error_code="INVALID_CONFIG",
            context={"config_path": config_path, "errors": errors},
            **kwargs
        )


class DependencyError(NeuroLiteError):
    """Dependency-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'DEPENDENCY_ERROR')
        super().__init__(message, **kwargs)


class MissingDependencyError(DependencyError):
    """Raised when required dependency is missing."""
    
    def __init__(self, dependency: str, feature: str, install_command: str, **kwargs):
        message = f"Missing dependency '{dependency}' required for {feature}"
        suggestions = [
            f"Install the missing dependency: {install_command}",
            f"Install NeuroLite with the appropriate extras: pip install neurolite[{feature}]",
            "Check the installation documentation for complete setup instructions"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="MISSING_DEPENDENCY",
            context={"dependency": dependency, "feature": feature, "install_command": install_command},
            **kwargs
        )


class ResourceError(NeuroLiteError):
    """Resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'RESOURCE_ERROR')
        super().__init__(message, **kwargs)


class InsufficientMemoryError(ResourceError):
    """Raised when insufficient memory is available."""
    
    def __init__(self, required_memory: str, available_memory: str, **kwargs):
        message = f"Insufficient memory: required {required_memory}, available {available_memory}"
        suggestions = [
            "Reduce batch size to use less memory",
            "Use a smaller model or enable model optimization",
            "Close other applications to free up memory",
            "Consider using gradient checkpointing or mixed precision training"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="INSUFFICIENT_MEMORY",
            context={"required_memory": required_memory, "available_memory": available_memory},
            **kwargs
        )


class GPUError(ResourceError):
    """Raised when GPU-related issues occur."""
    
    def __init__(self, reason: str, **kwargs):
        message = f"GPU error: {reason}"
        suggestions = [
            "Check if CUDA is properly installed and compatible",
            "Verify GPU drivers are up to date",
            "Try running on CPU by setting device='cpu'",
            "Check GPU memory usage and free up memory if needed"
        ]
        super().__init__(
            message,
            suggestions=suggestions,
            error_code="GPU_ERROR",
            context={"reason": reason},
            **kwargs
        )