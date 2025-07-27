"""
Exception classes for NeuroLite library.

This module defines all custom exceptions used throughout the NeuroLite library
for proper error handling and user feedback.
"""

from typing import Optional, Any


class NeuroLiteException(Exception):
    """
    Base exception class for all NeuroLite-specific errors.
    
    All custom exceptions in the NeuroLite library should inherit from this class
    to provide consistent error handling and identification.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        """
        Initialize NeuroLiteException.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class UnsupportedFormatError(NeuroLiteException):
    """
    Raised when a file format is not supported by NeuroLite.
    
    This exception is thrown when the library encounters a file format
    that it cannot process or analyze.
    """
    
    def __init__(self, format_type: str, supported_formats: Optional[list] = None):
        """
        Initialize UnsupportedFormatError.
        
        Args:
            format_type: The unsupported format that was encountered
            supported_formats: Optional list of supported formats
        """
        message = f"Unsupported file format: {format_type}"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
        
        details = {
            "format_type": format_type,
            "supported_formats": supported_formats
        }
        
        super().__init__(message, "UNSUPPORTED_FORMAT", details)
        self.format_type = format_type
        self.supported_formats = supported_formats


class InsufficientDataError(NeuroLiteException):
    """
    Raised when the dataset is too small or insufficient for analysis.
    
    This exception is thrown when the provided dataset doesn't meet
    the minimum requirements for meaningful analysis.
    """
    
    def __init__(self, current_size: int, minimum_required: int, data_type: str = "dataset"):
        """
        Initialize InsufficientDataError.
        
        Args:
            current_size: Current size of the dataset
            minimum_required: Minimum required size for analysis
            data_type: Type of data (e.g., "dataset", "column", "samples")
        """
        message = (f"Insufficient {data_type} size: {current_size}. "
                  f"Minimum required: {minimum_required}")
        
        details = {
            "current_size": current_size,
            "minimum_required": minimum_required,
            "data_type": data_type
        }
        
        super().__init__(message, "INSUFFICIENT_DATA", details)
        self.current_size = current_size
        self.minimum_required = minimum_required
        self.data_type = data_type


class ResourceLimitError(NeuroLiteException):
    """
    Raised when system resource limits are exceeded during analysis.
    
    This exception is thrown when the analysis cannot proceed due to
    memory, processing time, or other resource constraints.
    """
    
    def __init__(self, resource_type: str, limit_exceeded: str, current_usage: Optional[Any] = None):
        """
        Initialize ResourceLimitError.
        
        Args:
            resource_type: Type of resource that exceeded limits (e.g., "memory", "time")
            limit_exceeded: Description of the limit that was exceeded
            current_usage: Optional current usage information
        """
        message = f"Resource limit exceeded for {resource_type}: {limit_exceeded}"
        if current_usage is not None:
            message += f" (current usage: {current_usage})"
        
        details = {
            "resource_type": resource_type,
            "limit_exceeded": limit_exceeded,
            "current_usage": current_usage
        }
        
        super().__init__(message, "RESOURCE_LIMIT", details)
        self.resource_type = resource_type
        self.limit_exceeded = limit_exceeded
        self.current_usage = current_usage


class ValidationError(NeuroLiteException):
    """
    Raised when data validation fails.
    
    This exception is thrown when input data doesn't meet validation
    requirements or contains invalid values.
    """
    
    def __init__(self, field_name: str, invalid_value: Any, validation_rule: str):
        """
        Initialize ValidationError.
        
        Args:
            field_name: Name of the field that failed validation
            invalid_value: The invalid value that caused the error
            validation_rule: Description of the validation rule that was violated
        """
        message = f"Validation failed for field '{field_name}': {validation_rule}"
        
        details = {
            "field_name": field_name,
            "invalid_value": invalid_value,
            "validation_rule": validation_rule
        }
        
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_rule = validation_rule


class ProcessingError(NeuroLiteException):
    """
    Raised when an error occurs during data processing or analysis.
    
    This exception is thrown when the analysis pipeline encounters
    an error that prevents it from completing successfully.
    """
    
    def __init__(self, stage: str, original_error: Optional[Exception] = None):
        """
        Initialize ProcessingError.
        
        Args:
            stage: The processing stage where the error occurred
            original_error: Optional original exception that caused this error
        """
        message = f"Processing error in stage '{stage}'"
        if original_error:
            message += f": {str(original_error)}"
        
        details = {
            "stage": stage,
            "original_error": str(original_error) if original_error else None,
            "original_error_type": type(original_error).__name__ if original_error else None
        }
        
        super().__init__(message, "PROCESSING_ERROR", details)
        self.stage = stage
        self.original_error = original_error


class ConfigurationError(NeuroLiteException):
    """
    Raised when there's an error in library configuration.
    
    This exception is thrown when the library is misconfigured or
    required dependencies are missing.
    """
    
    def __init__(self, config_item: str, issue: str, suggestion: Optional[str] = None):
        """
        Initialize ConfigurationError.
        
        Args:
            config_item: The configuration item that has an issue
            issue: Description of the configuration issue
            suggestion: Optional suggestion for fixing the issue
        """
        message = f"Configuration error for '{config_item}': {issue}"
        if suggestion:
            message += f". Suggestion: {suggestion}"
        
        details = {
            "config_item": config_item,
            "issue": issue,
            "suggestion": suggestion
        }
        
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_item = config_item
        self.issue = issue
        self.suggestion = suggestion


# Utility functions for error handling

def handle_graceful_degradation(error: Exception, component: str, fallback_result: Any = None) -> Any:
    """
    Handle graceful degradation when a component fails.
    
    Args:
        error: The exception that occurred
        component: Name of the component that failed
        fallback_result: Optional fallback result to return
        
    Returns:
        Fallback result or None if no fallback provided
    """
    # Log the error (in a real implementation, this would use proper logging)
    print(f"Warning: Component '{component}' failed with error: {error}")
    print(f"Continuing with graceful degradation...")
    
    return fallback_result


def validate_input_data(data: Any, data_type: str, min_size: Optional[int] = None) -> None:
    """
    Validate input data and raise appropriate exceptions if invalid.
    
    Args:
        data: The data to validate
        data_type: Type description for error messages
        min_size: Optional minimum size requirement
        
    Raises:
        ValidationError: If data validation fails
        InsufficientDataError: If data size is insufficient
    """
    if data is None:
        raise ValidationError("data", data, f"{data_type} cannot be None")
    
    if min_size is not None:
        try:
            current_size = len(data)
            if current_size < min_size:
                raise InsufficientDataError(current_size, min_size, data_type)
        except TypeError:
            # Data doesn't support len(), skip size validation
            pass