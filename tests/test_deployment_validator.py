"""
Unit tests for deployment validation functionality.

Tests the DeploymentValidator class and validation checks.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurolite.deployment import (
    DeploymentValidator,
    ValidationResult,
    ValidationCheck,
    ValidationStatus
)
from neurolite.models.base import BaseModel, ModelCapabilities, TaskType
from neurolite.data.detector import DataType


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, framework="pytorch", **kwargs):
        super().__init__(**kwargs)
        self.is_trained = True
        self._framework = framework
        self.model = Mock()  # Mock underlying model
    
    @property
    def capabilities(self):
        return ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework=self._framework
        )
    
    def fit(self, X, y, validation_data=None, **kwargs):
        return self
    
    def predict(self, X, **kwargs):
        from neurolite.models.base import PredictionResult
        # Return consistent predictions for testing
        predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])[:len(X)]
        return PredictionResult(predictions=predictions)
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self


class TestValidationCheck:
    """Test ValidationCheck class."""
    
    def test_initialization(self):
        """Test ValidationCheck initialization."""
        check = ValidationCheck(
            name="test_check",
            status=ValidationStatus.PASSED,
            message="Test passed",
            expected_value=1.0,
            actual_value=1.0,
            tolerance=0.01
        )
        
        assert check.name == "test_check"
        assert check.status == ValidationStatus.PASSED
        assert check.message == "Test passed"
        assert check.expected_value == 1.0
        assert check.actual_value == 1.0
        assert check.tolerance == 0.01


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_initialization_with_passed_checks(self):
        """Test ValidationResult with all passed checks."""
        checks = [
            ValidationCheck("check1", ValidationStatus.PASSED, "Passed"),
            ValidationCheck("check2", ValidationStatus.PASSED, "Passed")
        ]
        
        result = ValidationResult(
            overall_status=ValidationStatus.PASSED,
            checks=checks,
            model_path="/test/model.onnx",
            validation_time_seconds=1.5
        )
        
        assert result.overall_status == ValidationStatus.PASSED
        assert result.num_passed == 2
        assert result.num_failed == 0
        assert result.num_warnings == 0
        assert result.num_skipped == 0
    
    def test_initialization_with_failed_checks(self):
        """Test ValidationResult with failed checks."""
        checks = [
            ValidationCheck("check1", ValidationStatus.PASSED, "Passed"),
            ValidationCheck("check2", ValidationStatus.FAILED, "Failed"),
            ValidationCheck("check3", ValidationStatus.WARNING, "Warning")
        ]
        
        result = ValidationResult(
            overall_status=ValidationStatus.FAILED,  # Will be overridden
            checks=checks,
            model_path="/test/model.onnx",
            validation_time_seconds=2.0
        )
        
        # Should be updated to FAILED due to failed check
        assert result.overall_status == ValidationStatus.FAILED
        assert result.num_passed == 1
        assert result.num_failed == 1
        assert result.num_warnings == 1
        assert result.num_skipped == 0
    
    def test_initialization_with_warnings_only(self):
        """Test ValidationResult with warnings only."""
        checks = [
            ValidationCheck("check1", ValidationStatus.PASSED, "Passed"),
            ValidationCheck("check2", ValidationStatus.WARNING, "Warning")
        ]
        
        result = ValidationResult(
            overall_status=ValidationStatus.PASSED,  # Will be overridden
            checks=checks,
            model_path="/test/model.onnx",
            validation_time_seconds=1.0
        )
        
        # Should be updated to WARNING due to warning check
        assert result.overall_status == ValidationStatus.WARNING
        assert result.num_passed == 1
        assert result.num_failed == 0
        assert result.num_warnings == 1


class TestDeploymentValidator:
    """Test DeploymentValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DeploymentValidator()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel()
        self.test_data = (
            np.random.rand(10, 5),
            np.random.randint(0, 2, 10)
        )
        self.sample_input = np.random.rand(1, 5)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test DeploymentValidator initialization."""
        assert self.validator.accuracy_tolerance == 1e-5
        assert self.validator.performance_tolerance == 0.1
        assert self.validator.enable_performance_checks is True
    
    def test_custom_initialization(self):
        """Test DeploymentValidator with custom parameters."""
        validator = DeploymentValidator(
            accuracy_tolerance=1e-3,
            performance_tolerance=0.2,
            enable_performance_checks=False
        )
        
        assert validator.accuracy_tolerance == 1e-3
        assert validator.performance_tolerance == 0.2
        assert validator.enable_performance_checks is False
    
    def test_check_file_exists_success(self):
        """Test file existence check when file exists."""
        # Create a test file
        test_file = self.temp_dir / "model.onnx"
        test_file.touch()
        
        check = self.validator._check_file_exists(test_file)
        
        assert check.name == "file_exists"
        assert check.status == ValidationStatus.PASSED
        assert str(test_file) in check.message
    
    def test_check_file_exists_failure(self):
        """Test file existence check when file doesn't exist."""
        test_file = self.temp_dir / "nonexistent.onnx"
        
        check = self.validator._check_file_exists(test_file)
        
        assert check.name == "file_exists"
        assert check.status == ValidationStatus.FAILED
        assert "does not exist" in check.message
    
    def test_check_file_size_success(self):
        """Test file size check with valid file."""
        # Create a test file with some content
        test_file = self.temp_dir / "model.onnx"
        test_file.write_text("test content")
        
        check = self.validator._check_file_size(test_file)
        
        assert check.name == "file_size"
        assert check.status == ValidationStatus.PASSED
        assert check.actual_value > 0
    
    def test_check_file_size_empty_file(self):
        """Test file size check with empty file."""
        # Create an empty test file
        test_file = self.temp_dir / "model.onnx"
        test_file.touch()
        
        check = self.validator._check_file_size(test_file)
        
        assert check.name == "file_size"
        assert check.status == ValidationStatus.FAILED
        assert "empty" in check.message
    
    def test_check_file_size_nonexistent(self):
        """Test file size check with nonexistent file."""
        test_file = self.temp_dir / "nonexistent.onnx"
        
        check = self.validator._check_file_size(test_file)
        
        assert check.name == "file_size"
        assert check.status == ValidationStatus.SKIPPED
        assert "does not exist" in check.message
    
    @patch('neurolite.deployment.validator.DeploymentValidator._load_exported_model')
    def test_check_model_loading_success(self, mock_load):
        """Test model loading check when successful."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        check, loaded_model = self.validator._check_model_loading(
            self.temp_dir / "model.onnx", "onnx"
        )
        
        assert check.name == "model_loading"
        assert check.status == ValidationStatus.PASSED
        assert "Successfully loaded" in check.message
        assert loaded_model == mock_model
    
    @patch('neurolite.deployment.validator.DeploymentValidator._load_exported_model')
    def test_check_model_loading_failure(self, mock_load):
        """Test model loading check when it fails."""
        mock_load.side_effect = Exception("Loading failed")
        
        check, loaded_model = self.validator._check_model_loading(
            self.temp_dir / "model.onnx", "onnx"
        )
        
        assert check.name == "model_loading"
        assert check.status == ValidationStatus.FAILED
        assert "Failed to load" in check.message
        assert loaded_model is None
    
    @patch('neurolite.deployment.validator.DeploymentValidator._predict_with_exported_model')
    def test_check_accuracy_consistency_success(self, mock_predict):
        """Test accuracy consistency check when predictions match."""
        # Mock exported model predictions to match original
        original_predictions = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_predict.return_value = original_predictions
        
        # Mock the original model to return the same predictions
        self.mock_model.predict = Mock(return_value=Mock(predictions=original_predictions))
        
        mock_exported_model = Mock()
        test_data = (np.random.rand(2, 5), np.array([0, 1]))
        
        check = self.validator._check_accuracy_consistency(
            self.mock_model, mock_exported_model, test_data, "onnx"
        )
        
        assert check.name == "accuracy_consistency"
        assert check.status == ValidationStatus.PASSED
        assert "consistent" in check.message
    
    @patch('neurolite.deployment.validator.DeploymentValidator._predict_with_exported_model')
    def test_check_accuracy_consistency_failure(self, mock_predict):
        """Test accuracy consistency check when predictions don't match."""
        # Mock different predictions
        original_predictions = np.array([[0.8, 0.2], [0.3, 0.7]])
        exported_predictions = np.array([[0.5, 0.5], [0.6, 0.4]])
        
        self.mock_model.predict = Mock(return_value=Mock(predictions=original_predictions))
        mock_predict.return_value = exported_predictions
        
        mock_exported_model = Mock()
        test_data = (np.random.rand(2, 5), np.array([0, 1]))
        
        check = self.validator._check_accuracy_consistency(
            self.mock_model, mock_exported_model, test_data, "onnx"
        )
        
        assert check.name == "accuracy_consistency"
        assert check.status == ValidationStatus.FAILED
        assert "differ beyond tolerance" in check.message
    
    def test_compare_predictions_identical(self):
        """Test comparing identical predictions."""
        pred1 = np.array([1.0, 2.0, 3.0])
        pred2 = np.array([1.0, 2.0, 3.0])
        
        result = self.validator._compare_predictions(pred1, pred2)
        assert result is True
    
    def test_compare_predictions_within_tolerance(self):
        """Test comparing predictions within tolerance."""
        pred1 = np.array([1.0, 2.0, 3.0])
        pred2 = np.array([1.00001, 2.00001, 3.00001])
        
        result = self.validator._compare_predictions(pred1, pred2)
        assert result is True
    
    def test_compare_predictions_beyond_tolerance(self):
        """Test comparing predictions beyond tolerance."""
        pred1 = np.array([1.0, 2.0, 3.0])
        pred2 = np.array([1.1, 2.1, 3.1])
        
        result = self.validator._compare_predictions(pred1, pred2)
        assert result is False
    
    def test_measure_inference_time(self):
        """Test measuring inference time."""
        def mock_predict_fn(x):
            import time
            time.sleep(0.001)  # 1ms delay
            return np.array([0.5])
        
        inference_time = self.validator._measure_inference_time(
            mock_predict_fn, self.sample_input, num_runs=10
        )
        
        assert inference_time is not None
        assert inference_time > 0
        # Should be around 1ms, but allow for variance
        assert 0.5 < inference_time < 10
    
    @patch('neurolite.deployment.validator.DeploymentValidator._load_exported_model')
    @patch('neurolite.deployment.validator.DeploymentValidator._predict_with_exported_model')
    def test_validate_export_complete(self, mock_predict, mock_load):
        """Test complete export validation."""
        # Create a test file
        test_file = self.temp_dir / "model.onnx"
        test_file.write_text("test model content")
        
        # Setup mocks
        mock_exported_model = Mock()
        mock_load.return_value = mock_exported_model
        
        # Mock predictions to be identical
        original_predictions = np.array([[0.8, 0.2], [0.3, 0.7]])
        self.mock_model.predict = Mock(return_value=Mock(predictions=original_predictions))
        mock_predict.return_value = original_predictions
        
        result = self.validator.validate_export(
            self.mock_model,
            test_file,
            self.test_data,
            "onnx",
            sample_input=self.sample_input
        )
        
        assert isinstance(result, ValidationResult)
        assert result.model_path == str(test_file)
        assert result.validation_time_seconds > 0
        assert len(result.checks) > 0
        
        # Check that various validation checks were performed
        check_names = [check.name for check in result.checks]
        assert "file_exists" in check_names
        assert "file_size" in check_names
        assert "model_loading" in check_names


if __name__ == "__main__":
    pytest.main([__file__])