"""
Unit tests for model export functionality.

Tests the ModelExporter class and format-specific exporters.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurolite.deployment import (
    ModelExporter,
    ExportedModel,
    ONNXExporter,
    TensorFlowLiteExporter,
    PyTorchExporter,
    ExportFormat
)
from neurolite.models.base import BaseModel, ModelCapabilities, TaskType
from neurolite.data.detector import DataType
from neurolite.core.exceptions import ExportError


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
        # Return mock predictions
        predictions = np.random.rand(len(X) if hasattr(X, '__len__') else 1, 2)
        return PredictionResult(predictions=predictions)
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self


class TestModelExporter:
    """Test ModelExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = ModelExporter()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel()
        self.sample_input = np.random.rand(1, 10)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelExporter initialization."""
        assert isinstance(self.exporter, ModelExporter)
        assert ExportFormat.ONNX in self.exporter._exporters
        assert ExportFormat.TENSORFLOW_LITE in self.exporter._exporters
        assert ExportFormat.PYTORCH in self.exporter._exporters
        assert ExportFormat.TORCHSCRIPT in self.exporter._exporters
    
    def test_get_supported_formats(self):
        """Test getting supported formats for a model."""
        # Mock the can_export method to return True for all formats
        for exporter in self.exporter._exporters.values():
            exporter.can_export = Mock(return_value=True)
        
        supported_formats = self.exporter.get_supported_formats(self.mock_model)
        
        assert len(supported_formats) > 0
        assert all(isinstance(fmt, ExportFormat) for fmt in supported_formats)
    
    def test_export_model_untrained(self):
        """Test exporting untrained model raises error."""
        untrained_model = MockModel()
        untrained_model.is_trained = False
        
        with pytest.raises(ExportError) as exc_info:
            self.exporter.export_model(
                untrained_model,
                self.temp_dir / "model.onnx",
                sample_input=self.sample_input
            )
        
        assert "must be trained" in str(exc_info.value)
    
    @patch('neurolite.deployment.formats.onnx.ONNXExporter.export')
    @patch('neurolite.deployment.formats.onnx.ONNXExporter.can_export')
    @patch('neurolite.deployment.formats.onnx.ONNXExporter.validate_export')
    def test_export_model_success(self, mock_validate, mock_can_export, mock_export):
        """Test successful model export."""
        # Setup mocks
        mock_can_export.return_value = True
        mock_export.return_value = {
            "format": "onnx",
            "output_path": str(self.temp_dir / "model.onnx"),
            "file_size_mb": 1.5
        }
        mock_validate.return_value = True
        
        # Create a dummy file to simulate export
        output_path = self.temp_dir / "model.onnx"
        output_path.touch()
        
        result = self.exporter.export_model(
            self.mock_model,
            output_path,
            format="onnx",
            sample_input=self.sample_input
        )
        
        assert isinstance(result, ExportedModel)
        assert result.format == "onnx"
        assert result.validation_passed is True
        assert result.file_size_mb == 1.5
        
        mock_export.assert_called_once()
        mock_validate.assert_called_once()
    
    def test_export_multiple_formats(self):
        """Test exporting to multiple formats."""
        # Mock all exporters
        for exporter in self.exporter._exporters.values():
            exporter.can_export = Mock(return_value=True)
            exporter.export = Mock(return_value={
                "format": "test",
                "output_path": "test_path",
                "file_size_mb": 1.0
            })
            exporter.validate_export = Mock(return_value=True)
        
        formats = ["onnx", "pytorch"]
        results = self.exporter.export_multiple_formats(
            self.mock_model,
            self.temp_dir,
            formats,
            "test_model",
            sample_input=self.sample_input
        )
        
        assert len(results) == len(formats)
        for format_name in formats:
            assert format_name in results
            assert isinstance(results[format_name], ExportedModel)
    
    def test_auto_format_selection_pytorch(self):
        """Test automatic format selection for PyTorch models."""
        pytorch_model = MockModel(framework="pytorch")
        
        # Mock ONNX exporter to be available
        self.exporter._exporters[ExportFormat.ONNX].can_export = Mock(return_value=True)
        
        format_enum = self.exporter._auto_select_format(pytorch_model)
        assert format_enum == ExportFormat.ONNX
    
    def test_auto_format_selection_tensorflow(self):
        """Test automatic format selection for TensorFlow models."""
        tf_model = MockModel(framework="tensorflow")
        
        # Mock TFLite exporter to be available
        self.exporter._exporters[ExportFormat.TENSORFLOW_LITE].can_export = Mock(return_value=True)
        
        format_enum = self.exporter._auto_select_format(tf_model)
        assert format_enum == ExportFormat.TENSORFLOW_LITE
    
    def test_get_export_info(self):
        """Test getting export information for a model."""
        # Mock some exporters as available
        self.exporter._exporters[ExportFormat.ONNX].can_export = Mock(return_value=True)
        self.exporter._exporters[ExportFormat.PYTORCH].can_export = Mock(return_value=True)
        self.exporter._exporters[ExportFormat.TENSORFLOW_LITE].can_export = Mock(return_value=False)
        
        info = self.exporter.get_export_info(self.mock_model)
        
        assert "model_framework" in info
        assert "supported_formats" in info
        assert "auto_selected_format" in info
        assert "recommendations" in info
        
        assert info["model_framework"] == "pytorch"
        assert "onnx" in info["supported_formats"]
        assert "pytorch" in info["supported_formats"]
        assert "tflite" not in info["supported_formats"]


class TestONNXExporter:
    """Test ONNXExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = ONNXExporter()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel(framework="pytorch")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ONNXExporter initialization."""
        assert self.exporter.format == ExportFormat.ONNX
        assert self.exporter.file_extension == ".onnx"
        assert self.exporter.opset_version == 11
    
    def test_can_export_trained_pytorch(self):
        """Test can_export for trained PyTorch model."""
        assert self.exporter.can_export(self.mock_model) is True
    
    def test_can_export_untrained(self):
        """Test can_export for untrained model."""
        untrained_model = MockModel(framework="pytorch")
        untrained_model.is_trained = False
        
        assert self.exporter.can_export(untrained_model) is False
    
    def test_can_export_unsupported_framework(self):
        """Test can_export for unsupported framework."""
        sklearn_model = MockModel(framework="sklearn")
        
        assert self.exporter.can_export(sklearn_model) is False
    
    def test_prepare_output_path(self):
        """Test output path preparation."""
        # Test path without extension
        path = self.exporter.prepare_output_path(self.temp_dir / "model")
        assert path.suffix == ".onnx"
        
        # Test path with correct extension
        path = self.exporter.prepare_output_path(self.temp_dir / "model.onnx")
        assert path.suffix == ".onnx"
        
        # Test path with wrong extension
        path = self.exporter.prepare_output_path(self.temp_dir / "model.pt")
        assert path.suffix == ".onnx"
    
    def test_compare_outputs(self):
        """Test output comparison."""
        output1 = np.array([1.0, 2.0, 3.0])
        output2 = np.array([1.0, 2.0, 3.0])
        output3 = np.array([1.1, 2.1, 3.1])
        
        # Identical outputs
        assert self.exporter.compare_outputs(output1, output2) is True
        
        # Similar outputs within tolerance
        assert self.exporter.compare_outputs(output1, output3, tolerance=0.2) is True
        
        # Different outputs beyond tolerance (use smaller tolerance)
        assert self.exporter.compare_outputs(output1, output3, tolerance=0.01) is False
    
    @patch('torch.onnx.export')
    @patch('neurolite.deployment.formats.onnx.ONNXExporter._get_pytorch_model')
    def test_export_pytorch_model(self, mock_get_model, mock_torch_export):
        """Test exporting PyTorch model."""
        # Setup mocks
        mock_pytorch_model = Mock()
        mock_pytorch_model.eval = Mock()
        mock_get_model.return_value = mock_pytorch_model
        
        sample_input = np.random.rand(1, 10)
        output_path = self.temp_dir / "model.onnx"
        
        # Create dummy file to simulate export
        output_path.touch()
        
        result = self.exporter.export(
            self.mock_model,
            output_path,
            sample_input=sample_input
        )
        
        assert result["format"] == "onnx"
        assert result["output_path"] == str(output_path)
        assert "opset_version" in result
        
        mock_torch_export.assert_called_once()


class TestTensorFlowLiteExporter:
    """Test TensorFlowLiteExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = TensorFlowLiteExporter()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel(framework="tensorflow")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test TensorFlowLiteExporter initialization."""
        assert self.exporter.format == ExportFormat.TENSORFLOW_LITE
        assert self.exporter.file_extension == ".tflite"
        assert self.exporter.quantize is False
    
    def test_can_export_tensorflow(self):
        """Test can_export for TensorFlow model."""
        assert self.exporter.can_export(self.mock_model) is True
    
    def test_can_export_non_tensorflow(self):
        """Test can_export for non-TensorFlow model."""
        pytorch_model = MockModel(framework="pytorch")
        assert self.exporter.can_export(pytorch_model) is False


class TestPyTorchExporter:
    """Test PyTorchExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = PyTorchExporter()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel(framework="pytorch")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PyTorchExporter initialization."""
        assert self.exporter.format == ExportFormat.TORCHSCRIPT
        assert self.exporter.file_extension == ".pt"
        assert self.exporter.export_type == "torchscript"
    
    def test_initialization_state_dict(self):
        """Test PyTorchExporter initialization for state dict."""
        exporter = PyTorchExporter(export_type="state_dict")
        assert exporter.format == ExportFormat.PYTORCH
        assert exporter.file_extension == ".pth"
    
    def test_can_export_pytorch(self):
        """Test can_export for PyTorch model."""
        assert self.exporter.can_export(self.mock_model) is True
    
    def test_can_export_non_pytorch(self):
        """Test can_export for non-PyTorch model."""
        tf_model = MockModel(framework="tensorflow")
        assert self.exporter.can_export(tf_model) is False


if __name__ == "__main__":
    pytest.main([__file__])