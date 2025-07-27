"""
Unit tests for exception classes.

This module contains comprehensive tests for all custom exception classes
and error handling utilities.
"""

import pytest
from neurolite.core.exceptions import (
    NeuroLiteException,
    UnsupportedFormatError,
    InsufficientDataError,
    ResourceLimitError,
    ValidationError,
    ProcessingError,
    ConfigurationError,
    handle_graceful_degradation,
    validate_input_data
)


class TestNeuroLiteException:
    """Test cases for base NeuroLiteException class."""
    
    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = NeuroLiteException("Test error message")
        
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.error_code is None
        assert exc.details == {}
    
    def test_exception_with_error_code(self):
        """Test exception with error code."""
        exc = NeuroLiteException("Test error", "TEST_ERROR")
        
        assert str(exc) == "[TEST_ERROR] Test error"
        assert exc.error_code == "TEST_ERROR"
    
    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"field": "value", "count": 42}
        exc = NeuroLiteException("Test error", "TEST_ERROR", details)
        
        assert exc.details == details
        assert exc.details["field"] == "value"
        assert exc.details["count"] == 42


class TestUnsupportedFormatError:
    """Test cases for UnsupportedFormatError class."""
    
    def test_basic_unsupported_format(self):
        """Test basic unsupported format error."""
        exc = UnsupportedFormatError("XYZ")
        
        assert "Unsupported file format: XYZ" in str(exc)
        assert exc.format_type == "XYZ"
        assert exc.error_code == "UNSUPPORTED_FORMAT"
        assert exc.details["format_type"] == "XYZ"
    
    def test_unsupported_format_with_supported_list(self):
        """Test unsupported format error with supported formats list."""
        supported = ["CSV", "JSON", "XML"]
        exc = UnsupportedFormatError("XYZ", supported)
        
        error_msg = str(exc)
        assert "Unsupported file format: XYZ" in error_msg
        assert "Supported formats: CSV, JSON, XML" in error_msg
        assert exc.supported_formats == supported
        assert exc.details["supported_formats"] == supported


class TestInsufficientDataError:
    """Test cases for InsufficientDataError class."""
    
    def test_basic_insufficient_data(self):
        """Test basic insufficient data error."""
        exc = InsufficientDataError(5, 10)
        
        error_msg = str(exc)
        assert "Insufficient dataset size: 5" in error_msg
        assert "Minimum required: 10" in error_msg
        assert exc.current_size == 5
        assert exc.minimum_required == 10
        assert exc.data_type == "dataset"
        assert exc.error_code == "INSUFFICIENT_DATA"
    
    def test_insufficient_data_with_custom_type(self):
        """Test insufficient data error with custom data type."""
        exc = InsufficientDataError(3, 5, "column")
        
        error_msg = str(exc)
        assert "Insufficient column size: 3" in error_msg
        assert exc.data_type == "column"
        assert exc.details["data_type"] == "column"


class TestResourceLimitError:
    """Test cases for ResourceLimitError class."""
    
    def test_basic_resource_limit(self):
        """Test basic resource limit error."""
        exc = ResourceLimitError("memory", "8GB limit exceeded")
        
        error_msg = str(exc)
        assert "Resource limit exceeded for memory" in error_msg
        assert "8GB limit exceeded" in error_msg
        assert exc.resource_type == "memory"
        assert exc.limit_exceeded == "8GB limit exceeded"
        assert exc.error_code == "RESOURCE_LIMIT"
    
    def test_resource_limit_with_current_usage(self):
        """Test resource limit error with current usage information."""
        exc = ResourceLimitError("memory", "8GB limit exceeded", "10GB")
        
        error_msg = str(exc)
        assert "current usage: 10GB" in error_msg
        assert exc.current_usage == "10GB"
        assert exc.details["current_usage"] == "10GB"


class TestValidationError:
    """Test cases for ValidationError class."""
    
    def test_basic_validation_error(self):
        """Test basic validation error."""
        exc = ValidationError("confidence", 1.5, "must be between 0.0 and 1.0")
        
        error_msg = str(exc)
        assert "Validation failed for field 'confidence'" in error_msg
        assert "must be between 0.0 and 1.0" in error_msg
        assert exc.field_name == "confidence"
        assert exc.invalid_value == 1.5
        assert exc.validation_rule == "must be between 0.0 and 1.0"
        assert exc.error_code == "VALIDATION_ERROR"


class TestProcessingError:
    """Test cases for ProcessingError class."""
    
    def test_basic_processing_error(self):
        """Test basic processing error."""
        exc = ProcessingError("data_analysis")
        
        error_msg = str(exc)
        assert "Processing error in stage 'data_analysis'" in error_msg
        assert exc.stage == "data_analysis"
        assert exc.original_error is None
        assert exc.error_code == "PROCESSING_ERROR"
    
    def test_processing_error_with_original(self):
        """Test processing error with original exception."""
        original = ValueError("Invalid input")
        exc = ProcessingError("data_analysis", original)
        
        error_msg = str(exc)
        assert "Processing error in stage 'data_analysis'" in error_msg
        assert "Invalid input" in error_msg
        assert exc.original_error == original
        assert exc.details["original_error_type"] == "ValueError"


class TestConfigurationError:
    """Test cases for ConfigurationError class."""
    
    def test_basic_configuration_error(self):
        """Test basic configuration error."""
        exc = ConfigurationError("database_url", "URL is malformed")
        
        error_msg = str(exc)
        assert "Configuration error for 'database_url'" in error_msg
        assert "URL is malformed" in error_msg
        assert exc.config_item == "database_url"
        assert exc.issue == "URL is malformed"
        assert exc.error_code == "CONFIGURATION_ERROR"
    
    def test_configuration_error_with_suggestion(self):
        """Test configuration error with suggestion."""
        exc = ConfigurationError(
            "database_url", 
            "URL is malformed", 
            "Use format: postgresql://user:pass@host:port/db"
        )
        
        error_msg = str(exc)
        assert "Suggestion: Use format: postgresql://user:pass@host:port/db" in error_msg
        assert exc.suggestion == "Use format: postgresql://user:pass@host:port/db"


class TestErrorHandlingUtilities:
    """Test cases for error handling utility functions."""
    
    def test_handle_graceful_degradation(self):
        """Test graceful degradation handling."""
        error = ValueError("Test error")
        result = handle_graceful_degradation(error, "test_component", "fallback_value")
        
        assert result == "fallback_value"
    
    def test_handle_graceful_degradation_no_fallback(self):
        """Test graceful degradation without fallback."""
        error = ValueError("Test error")
        result = handle_graceful_degradation(error, "test_component")
        
        assert result is None
    
    def test_validate_input_data_valid(self):
        """Test input data validation with valid data."""
        data = [1, 2, 3, 4, 5]
        
        # Should not raise any exception
        validate_input_data(data, "test_data", min_size=3)
    
    def test_validate_input_data_none(self):
        """Test input data validation with None data."""
        with pytest.raises(ValidationError) as exc_info:
            validate_input_data(None, "test_data")
        
        assert exc_info.value.field_name == "data"
        assert "test_data cannot be None" in exc_info.value.validation_rule
    
    def test_validate_input_data_insufficient_size(self):
        """Test input data validation with insufficient size."""
        data = [1, 2]
        
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_input_data(data, "test_data", min_size=5)
        
        assert exc_info.value.current_size == 2
        assert exc_info.value.minimum_required == 5
        assert exc_info.value.data_type == "test_data"
    
    def test_validate_input_data_no_len_support(self):
        """Test input data validation with data that doesn't support len()."""
        data = 42  # Integer doesn't support len()
        
        # Should not raise exception even with min_size specified
        validate_input_data(data, "test_data", min_size=5)


class TestExceptionInheritance:
    """Test cases for exception inheritance hierarchy."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from NeuroLiteException."""
        exceptions = [
            UnsupportedFormatError("test"),
            InsufficientDataError(1, 2),
            ResourceLimitError("memory", "limit"),
            ValidationError("field", "value", "rule"),
            ProcessingError("stage"),
            ConfigurationError("item", "issue")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, NeuroLiteException)
            assert isinstance(exc, Exception)
    
    def test_exception_error_codes(self):
        """Test that exceptions have appropriate error codes."""
        test_cases = [
            (UnsupportedFormatError("test"), "UNSUPPORTED_FORMAT"),
            (InsufficientDataError(1, 2), "INSUFFICIENT_DATA"),
            (ResourceLimitError("memory", "limit"), "RESOURCE_LIMIT"),
            (ValidationError("field", "value", "rule"), "VALIDATION_ERROR"),
            (ProcessingError("stage"), "PROCESSING_ERROR"),
            (ConfigurationError("item", "issue"), "CONFIGURATION_ERROR")
        ]
        
        for exc, expected_code in test_cases:
            assert exc.error_code == expected_code