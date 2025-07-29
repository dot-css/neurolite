"""
Unit tests for data validation functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from neurolite.data.validator import (
    DataValidator, ValidationIssue, ValidationResult, validate_data
)
from neurolite.data.loader import Dataset, DatasetInfo
from neurolite.data.detector import DataType


class TestValidationIssue:
    """Test cases for ValidationIssue class."""
    
    def test_init(self):
        """Test ValidationIssue initialization."""
        issue = ValidationIssue(
            severity='error',
            category='missing_data',
            message='Test message',
            details={'key': 'value'},
            suggestions=['Fix this', 'Try that']
        )
        
        assert issue.severity == 'error'
        assert issue.category == 'missing_data'
        assert issue.message == 'Test message'
        assert issue.details == {'key': 'value'}
        assert issue.suggestions == ['Fix this', 'Try that']
    
    def test_init_minimal(self):
        """Test ValidationIssue initialization with minimal parameters."""
        issue = ValidationIssue(
            severity='warning',
            category='format',
            message='Test warning'
        )
        
        assert issue.severity == 'warning'
        assert issue.category == 'format'
        assert issue.message == 'Test warning'
        assert issue.details is None
        assert issue.suggestions is None


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_init(self):
        """Test ValidationResult initialization."""
        issues = [
            ValidationIssue('error', 'test', 'Error message'),
            ValidationIssue('warning', 'test', 'Warning message'),
            ValidationIssue('info', 'test', 'Info message')
        ]
        summary = {'total_issues': 3}
        
        result = ValidationResult(is_valid=False, issues=issues, summary=summary)
        
        assert result.is_valid is False
        assert len(result.issues) == 3
        assert result.summary == summary
    
    def test_get_errors(self):
        """Test getting error-level issues."""
        issues = [
            ValidationIssue('error', 'test', 'Error 1'),
            ValidationIssue('warning', 'test', 'Warning 1'),
            ValidationIssue('error', 'test', 'Error 2'),
            ValidationIssue('info', 'test', 'Info 1')
        ]
        
        result = ValidationResult(is_valid=False, issues=issues, summary={})
        errors = result.get_errors()
        
        assert len(errors) == 2
        assert all(issue.severity == 'error' for issue in errors)
    
    def test_get_warnings(self):
        """Test getting warning-level issues."""
        issues = [
            ValidationIssue('error', 'test', 'Error 1'),
            ValidationIssue('warning', 'test', 'Warning 1'),
            ValidationIssue('warning', 'test', 'Warning 2'),
            ValidationIssue('info', 'test', 'Info 1')
        ]
        
        result = ValidationResult(is_valid=False, issues=issues, summary={})
        warnings = result.get_warnings()
        
        assert len(warnings) == 2
        assert all(issue.severity == 'warning' for issue in warnings)
    
    def test_has_errors(self):
        """Test checking for errors."""
        # With errors
        issues_with_errors = [
            ValidationIssue('error', 'test', 'Error'),
            ValidationIssue('warning', 'test', 'Warning')
        ]
        result_with_errors = ValidationResult(is_valid=False, issues=issues_with_errors, summary={})
        assert result_with_errors.has_errors() is True
        
        # Without errors
        issues_without_errors = [
            ValidationIssue('warning', 'test', 'Warning'),
            ValidationIssue('info', 'test', 'Info')
        ]
        result_without_errors = ValidationResult(is_valid=True, issues=issues_without_errors, summary={})
        assert result_without_errors.has_errors() is False
    
    def test_has_warnings(self):
        """Test checking for warnings."""
        # With warnings
        issues_with_warnings = [
            ValidationIssue('warning', 'test', 'Warning'),
            ValidationIssue('info', 'test', 'Info')
        ]
        result_with_warnings = ValidationResult(is_valid=True, issues=issues_with_warnings, summary={})
        assert result_with_warnings.has_warnings() is True
        
        # Without warnings
        issues_without_warnings = [
            ValidationIssue('error', 'test', 'Error'),
            ValidationIssue('info', 'test', 'Info')
        ]
        result_without_warnings = ValidationResult(is_valid=False, issues=issues_without_warnings, summary={})
        assert result_without_warnings.has_warnings() is False


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_init(self):
        """Test validator initialization."""
        assert self.validator is not None
        assert hasattr(self.validator, '_validators')
        assert DataType.IMAGE in self.validator._validators
        assert DataType.TEXT in self.validator._validators
        assert DataType.TABULAR in self.validator._validators
        assert DataType.AUDIO in self.validator._validators
        assert DataType.VIDEO in self.validator._validators
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        info = DatasetInfo(DataType.TEXT, 0)
        dataset = Dataset([], info=info)
        
        result = self.validator.validate(dataset)
        
        assert result.is_valid is False
        assert result.has_errors() is True
        
        errors = result.get_errors()
        assert len(errors) >= 1
        assert any('empty' in error.message.lower() for error in errors)
    
    def test_validate_basic_structure_success(self):
        """Test successful basic structure validation."""
        data = ['text1', 'text2', 'text3']
        info = DatasetInfo(DataType.TEXT, 3)
        dataset = Dataset(data, info=info)
        
        issues = self.validator._validate_basic_structure(dataset)
        
        # Should have no structural issues
        structural_errors = [issue for issue in issues if issue.severity == 'error' and issue.category == 'structure']
        assert len(structural_errors) == 0
    
    def test_validate_basic_structure_inconsistent_types(self):
        """Test basic structure validation with inconsistent sample types."""
        data = ['text1', 123, 'text3']  # Mixed types
        info = DatasetInfo(DataType.TEXT, 3)
        dataset = Dataset(data, info=info)
        
        issues = self.validator._validate_basic_structure(dataset)
        
        # Should detect inconsistent types
        consistency_issues = [issue for issue in issues if issue.category == 'consistency']
        assert len(consistency_issues) >= 1
    
    def test_validate_image_data_valid(self):
        """Test validation of valid image data."""
        # Create mock image data
        image_data = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ]
        info = DatasetInfo(DataType.IMAGE, 3, shape=(224, 224, 3))
        dataset = Dataset(image_data, info=info)
        
        issues = self.validator._validate_image_data(dataset)
        
        # Should have minimal issues for valid data
        errors = [issue for issue in issues if issue.severity == 'error']
        assert len(errors) == 0
    
    def test_validate_image_data_invalid_dimensions(self):
        """Test validation of image data with invalid dimensions."""
        # Create image data with invalid dimensions
        image_data = [
            np.random.randint(0, 255, (10000, 10000, 3), dtype=np.uint8),  # Too large
            np.random.randint(0, 255, (0, 0, 3), dtype=np.uint8),  # Too small
        ]
        info = DatasetInfo(DataType.IMAGE, 2)
        dataset = Dataset(image_data, info=info)
        
        issues = self.validator._validate_image_data(dataset)
        
        # Should detect dimension issues
        dimension_issues = [issue for issue in issues if 'dimensions' in issue.message.lower()]
        assert len(dimension_issues) >= 1
    
    def test_validate_image_data_inconsistent_shapes(self):
        """Test validation of image data with inconsistent shapes."""
        # Create images with different shapes
        image_data = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ]
        info = DatasetInfo(DataType.IMAGE, 3)
        dataset = Dataset(image_data, info=info)
        
        issues = self.validator._validate_image_data(dataset)
        
        # Should detect shape inconsistency
        consistency_issues = [issue for issue in issues if 'inconsistent shapes' in issue.message.lower()]
        assert len(consistency_issues) >= 1
    
    def test_validate_text_data_valid(self):
        """Test validation of valid text data."""
        text_data = [
            "This is a valid text sample.",
            "Another good text example.",
            "Third text sample for testing."
        ]
        info = DatasetInfo(DataType.TEXT, 3)
        dataset = Dataset(text_data, info=info)
        
        issues = self.validator._validate_text_data(dataset)
        
        # Should have minimal issues for valid data
        errors = [issue for issue in issues if issue.severity == 'error']
        assert len(errors) == 0
        
        # Should have info about text statistics
        info_issues = [issue for issue in issues if issue.severity == 'info' and 'statistics' in issue.category]
        assert len(info_issues) >= 1
    
    def test_validate_text_data_empty_texts(self):
        """Test validation of text data with empty strings."""
        text_data = [
            "Valid text",
            "",  # Empty string
            "   ",  # Whitespace only
            "Another valid text"
        ]
        info = DatasetInfo(DataType.TEXT, 4)
        dataset = Dataset(text_data, info=info)
        
        issues = self.validator._validate_text_data(dataset)
        
        # Should detect empty texts
        empty_text_issues = [issue for issue in issues if 'empty' in issue.message.lower()]
        assert len(empty_text_issues) >= 1
    
    def test_validate_text_data_non_strings(self):
        """Test validation of text data with non-string samples."""
        text_data = [
            "Valid text",
            123,  # Non-string
            "Another valid text",
            None  # Non-string
        ]
        info = DatasetInfo(DataType.TEXT, 4)
        dataset = Dataset(text_data, info=info)
        
        issues = self.validator._validate_text_data(dataset)
        
        # Should detect non-string samples
        non_string_issues = [issue for issue in issues if 'non-string' in issue.message.lower()]
        assert len(non_string_issues) >= 1
        assert any(issue.severity == 'error' for issue in non_string_issues)
    
    def test_validate_tabular_data_valid(self):
        """Test validation of valid tabular data."""
        # Create valid tabular data
        data = np.random.randn(100, 5).astype(np.float32)
        targets = np.random.randint(0, 3, 100)
        info = DatasetInfo(DataType.TABULAR, 100, shape=(100, 5))
        dataset = Dataset(data, targets=targets, info=info)
        
        issues = self.validator._validate_tabular_data(dataset)
        
        # Should have minimal issues for valid data
        errors = [issue for issue in issues if issue.severity == 'error']
        assert len(errors) == 0
    
    def test_validate_tabular_data_with_nan(self):
        """Test validation of tabular data with NaN values."""
        # Create data with NaN values
        data = np.random.randn(100, 5).astype(np.float32)
        data[10:20, 2] = np.nan  # Add some NaN values
        info = DatasetInfo(DataType.TABULAR, 100, shape=(100, 5))
        dataset = Dataset(data, info=info)
        
        issues = self.validator._validate_tabular_data(dataset)
        
        # Should detect NaN values
        nan_issues = [issue for issue in issues if 'nan' in issue.message.lower()]
        assert len(nan_issues) >= 1
    
    def test_validate_tabular_data_with_inf(self):
        """Test validation of tabular data with infinite values."""
        # Create data with infinite values
        data = np.random.randn(100, 5).astype(np.float32)
        data[5:10, 1] = np.inf  # Add some infinite values
        info = DatasetInfo(DataType.TABULAR, 100, shape=(100, 5))
        dataset = Dataset(data, info=info)
        
        issues = self.validator._validate_tabular_data(dataset)
        
        # Should detect infinite values
        inf_issues = [issue for issue in issues if 'infinite' in issue.message.lower()]
        assert len(inf_issues) >= 1
        assert any(issue.severity == 'error' for issue in inf_issues)
    
    def test_validate_tabular_data_constant_features(self):
        """Test validation of tabular data with constant features."""
        # Create data with constant features
        data = np.random.randn(100, 5).astype(np.float32)
        data[:, 2] = 5.0  # Make one feature constant
        info = DatasetInfo(DataType.TABULAR, 100, shape=(100, 5))
        dataset = Dataset(data, info=info)
        
        issues = self.validator._validate_tabular_data(dataset)
        
        # Should detect constant features
        constant_issues = [issue for issue in issues if 'constant' in issue.message.lower()]
        assert len(constant_issues) >= 1
    
    def test_validate_audio_data_valid(self):
        """Test validation of valid audio data."""
        # Create mock audio data
        audio_data = [
            np.random.randn(44100).astype(np.float32),  # 1 second at 44.1kHz
            np.random.randn(88200).astype(np.float32),  # 2 seconds at 44.1kHz
        ]
        info = DatasetInfo(
            DataType.AUDIO, 
            2, 
            metadata={'sample_rates': [44100, 44100]}
        )
        dataset = Dataset(audio_data, info=info)
        
        issues = self.validator._validate_audio_data(dataset)
        
        # Should have minimal issues for valid data
        errors = [issue for issue in issues if issue.severity == 'error']
        assert len(errors) == 0
        
        # Should have info about audio statistics
        info_issues = [issue for issue in issues if issue.severity == 'info' and 'statistics' in issue.category]
        assert len(info_issues) >= 1
    
    def test_validate_audio_data_invalid(self):
        """Test validation of invalid audio data."""
        # Create invalid audio data
        audio_data = [
            "not_audio_data",  # Invalid type
            np.random.randn(44100).astype(np.float32),
        ]
        info = DatasetInfo(DataType.AUDIO, 2)
        dataset = Dataset(audio_data, info=info)
        
        issues = self.validator._validate_audio_data(dataset)
        
        # Should detect invalid audio data
        invalid_issues = [issue for issue in issues if 'invalid' in issue.message.lower()]
        assert len(invalid_issues) >= 1
        assert any(issue.severity == 'error' for issue in invalid_issues)
    
    def test_validate_video_data(self):
        """Test validation of video data."""
        # Video validation is not fully implemented
        video_data = ["video1.mp4", "video2.mp4"]
        info = DatasetInfo(DataType.VIDEO, 2)
        dataset = Dataset(video_data, info=info)
        
        issues = self.validator._validate_video_data(dataset)
        
        # Should indicate that video validation is not implemented
        implementation_issues = [issue for issue in issues if 'implementation' in issue.category]
        assert len(implementation_issues) >= 1
    
    def test_validate_full_dataset(self):
        """Test full dataset validation."""
        # Create a valid text dataset
        text_data = ["Sample text 1", "Sample text 2", "Sample text 3"]
        info = DatasetInfo(DataType.TEXT, 3)
        dataset = Dataset(text_data, info=info)
        
        result = self.validator.validate(dataset)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True  # Should be valid for good data
        assert 'data_type' in result.summary
        assert 'num_samples' in result.summary
        assert 'is_valid' in result.summary
        assert 'num_errors' in result.summary
        assert 'num_warnings' in result.summary
        assert 'num_info' in result.summary
    
    def test_validate_dataset_with_validation_error(self):
        """Test validation when type-specific validation fails."""
        # Create a dataset that will cause validation to fail
        data = ["text1", "text2"]
        info = DatasetInfo(DataType.TEXT, 2)
        dataset = Dataset(data, info=info)
        
        # Create a custom validator that will raise an exception
        class FailingValidator(DataValidator):
            def _validate_text_data(self, dataset):
                raise Exception("Validation failed")
        
        failing_validator = FailingValidator()
        result = failing_validator.validate(dataset)
        
        # Should handle the exception gracefully
        assert isinstance(result, ValidationResult)
        
        validation_error_issues = [issue for issue in result.issues if issue.category == 'validation_error']
        assert len(validation_error_issues) >= 1


class TestValidateDataFunction:
    """Test cases for the validate_data function."""
    
    def test_validate_data_function(self):
        """Test the validate_data convenience function."""
        text_data = ["Sample text 1", "Sample text 2"]
        info = DatasetInfo(DataType.TEXT, 2)
        dataset = Dataset(text_data, info=info)
        
        result = validate_data(dataset)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True