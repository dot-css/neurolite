"""
Unit tests for model optimization functionality.

Tests the ModelOptimizer class and optimization configurations.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from neurolite.deployment import (
    ModelOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationType,
    QuantizationType
)
from neurolite.models.base import BaseModel, ModelCapabilities, TaskType, ModelMetadata
from neurolite.data.detector import DataType
from neurolite.core.exceptions import ExportError


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, framework="pytorch", **kwargs):
        super().__init__(**kwargs)
        self.is_trained = True
        self._framework = framework
        self.model = Mock()  # Mock underlying model
        self.metadata = ModelMetadata(
            name="test_model",
            task_type=TaskType.CLASSIFICATION,
            data_type=DataType.TABULAR,
            framework=framework,
            training_time=10.0,
            model_size_mb=5.0
        )
    
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


class TestOptimizationConfig:
    """Test OptimizationConfig class."""
    
    def test_default_initialization(self):
        """Test default OptimizationConfig initialization."""
        config = OptimizationConfig()
        
        assert config.quantization_enabled is False
        assert config.quantization_type == QuantizationType.DYNAMIC
        assert config.pruning_enabled is False
        assert config.pruning_sparsity == 0.5
        assert config.optimize_for_inference is True
        assert config.target_device == "cpu"
        assert config.max_accuracy_drop == 0.05
        assert config.additional_optimizations == {}
    
    def test_custom_initialization(self):
        """Test custom OptimizationConfig initialization."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type=QuantizationType.INT8,
            pruning_enabled=True,
            pruning_sparsity=0.7,
            target_device="gpu",
            max_accuracy_drop=0.1
        )
        
        assert config.quantization_enabled is True
        assert config.quantization_type == QuantizationType.INT8
        assert config.pruning_enabled is True
        assert config.pruning_sparsity == 0.7
        assert config.target_device == "gpu"
        assert config.max_accuracy_drop == 0.1


class TestOptimizationResult:
    """Test OptimizationResult class."""
    
    def test_initialization(self):
        """Test OptimizationResult initialization."""
        result = OptimizationResult(
            original_size_mb=10.0,
            optimized_size_mb=5.0,
            compression_ratio=2.0,
            optimization_time_seconds=30.0,
            original_inference_time_ms=100.0,
            optimized_inference_time_ms=50.0,
            original_accuracy=0.95,
            optimized_accuracy=0.93
        )
        
        assert result.original_size_mb == 10.0
        assert result.optimized_size_mb == 5.0
        assert result.compression_ratio == 2.0
        assert result.optimization_time_seconds == 30.0
        assert result.speedup_ratio == 2.0  # 100/50
        assert abs(result.accuracy_drop - 0.02) < 1e-10  # 0.95 - 0.93 (floating point precision)
    
    def test_post_init_calculations(self):
        """Test post-init calculations."""
        result = OptimizationResult(
            original_size_mb=8.0,
            optimized_size_mb=4.0,
            compression_ratio=2.0,
            optimization_time_seconds=20.0
        )
        
        # Should have empty applied_optimizations list
        assert result.applied_optimizations == []
        
        # Should not calculate speedup without inference times
        assert result.speedup_ratio is None
        
        # Should not calculate accuracy drop without accuracy values
        assert result.accuracy_drop is None


class TestModelOptimizer:
    """Test ModelOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ModelOptimizer()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_model = MockModel()
        self.validation_data = (
            np.random.rand(100, 10),
            np.random.randint(0, 2, 100)
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelOptimizer initialization."""
        assert isinstance(self.optimizer, ModelOptimizer)
        assert "pytorch" in self.optimizer._optimizers
        assert "tensorflow" in self.optimizer._optimizers
        assert "onnx" in self.optimizer._optimizers
    
    def test_optimize_untrained_model(self):
        """Test optimizing untrained model raises error."""
        untrained_model = MockModel()
        untrained_model.is_trained = False
        
        config = OptimizationConfig()
        
        with pytest.raises(ExportError) as exc_info:
            self.optimizer.optimize_model(untrained_model, config)
        
        assert "must be trained" in str(exc_info.value)
    
    def test_optimize_unsupported_framework(self):
        """Test optimizing model with unsupported framework."""
        unsupported_model = MockModel(framework="unsupported")
        config = OptimizationConfig()
        
        with pytest.raises(ExportError) as exc_info:
            self.optimizer.optimize_model(unsupported_model, config)
        
        assert "No optimizer available" in str(exc_info.value)
    
    @patch('time.time')
    @patch('neurolite.deployment.optimizer.PyTorchOptimizer.optimize')
    def test_optimize_pytorch_model(self, mock_pytorch_optimize, mock_time):
        """Test optimizing PyTorch model."""
        # Setup time mock to simulate elapsed time (provide extra values for any additional calls)
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 1.5]  # start_time, various intermediate calls, end_time
        
        # Setup mock
        optimized_model = MockModel()
        optimized_model.metadata.model_size_mb = 2.5  # Smaller than original
        mock_pytorch_optimize.return_value = optimized_model
        
        config = OptimizationConfig(quantization_enabled=True)
        
        result_model, result = self.optimizer.optimize_model(
            self.mock_model,
            config,
            validation_data=self.validation_data
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.original_size_mb == 5.0  # From mock model metadata
        assert result.optimized_size_mb == 2.5
        assert result.compression_ratio == 2.0  # 5.0 / 2.5
        assert result.optimization_time_seconds == 1.5  # Mocked time difference
        
        mock_pytorch_optimize.assert_called_once_with(
            self.mock_model, config, self.validation_data
        )
    
    def test_get_model_size_with_metadata(self):
        """Test getting model size from metadata."""
        size = self.optimizer._get_model_size(self.mock_model)
        assert size == 5.0  # From mock model metadata
    
    def test_get_model_size_without_metadata(self):
        """Test getting model size without metadata."""
        model_without_metadata = MockModel()
        model_without_metadata.metadata = None
        
        with patch.object(self.optimizer, '_get_pytorch_model_size', return_value=3.0):
            size = self.optimizer._get_model_size(model_without_metadata)
            assert size == 3.0
    
    @patch('torch.nn.Module.parameters')
    @patch('torch.nn.Module.buffers')
    def test_get_pytorch_model_size(self, mock_buffers, mock_parameters):
        """Test calculating PyTorch model size."""
        # Mock parameters and buffers
        mock_param = Mock()
        mock_param.nelement.return_value = 1000
        mock_param.element_size.return_value = 4  # float32
        mock_parameters.return_value = [mock_param]
        
        mock_buffer = Mock()
        mock_buffer.nelement.return_value = 100
        mock_buffer.element_size.return_value = 4
        mock_buffers.return_value = [mock_buffer]
        
        # Mock the model attribute
        self.mock_model.model = Mock()
        self.mock_model.model.parameters = mock_parameters
        self.mock_model.model.buffers = mock_buffers
        
        size = self.optimizer._get_pytorch_model_size(self.mock_model)
        
        # (1000 + 100) * 4 bytes = 4400 bytes = 4400 / (1024 * 1024) MB
        expected_size = 4400 / (1024 * 1024)
        assert abs(size - expected_size) < 1e-6
    
    def test_evaluate_model_accuracy(self):
        """Test evaluating model accuracy."""
        # Mock model to return predictable results
        mock_predictions = np.array([0, 1, 0, 1, 1])
        self.mock_model.predict = Mock(return_value=Mock(predictions=mock_predictions))
        
        y_true = np.array([0, 1, 0, 1, 0])  # 4/5 correct = 0.8 accuracy
        validation_data = (np.random.rand(5, 10), y_true)
        
        accuracy = self.optimizer._evaluate_model_accuracy(self.mock_model, validation_data)
        assert accuracy == 0.8
    
    def test_measure_inference_time(self):
        """Test measuring inference time."""
        sample_input = np.random.rand(1, 10)
        
        # Mock predict method with a small delay
        def mock_predict(x):
            import time
            time.sleep(0.001)  # 1ms delay
            return Mock(predictions=np.array([0.5]))
        
        self.mock_model.predict = mock_predict
        
        inference_time = self.optimizer._measure_inference_time(
            self.mock_model, sample_input, num_runs=10
        )
        
        assert inference_time is not None
        assert inference_time > 0
        # Should be around 1ms, but allow for some variance
        assert 0.5 < inference_time < 10


class TestPyTorchOptimizer:
    """Test PyTorchOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from neurolite.deployment.optimizer import PyTorchOptimizer
        self.optimizer = PyTorchOptimizer()
        self.mock_model = MockModel(framework="pytorch")
    
    def test_initialization(self):
        """Test PyTorchOptimizer initialization."""
        assert self.optimizer.applied_optimizations == []
    
    def test_optimize_with_quantization(self):
        """Test optimization with quantization enabled."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type=QuantizationType.DYNAMIC
        )
        
        result = self.optimizer.optimize(self.mock_model, config)
        
        assert result == self.mock_model  # Should return the same model for now
        assert "quantization_dynamic" in self.optimizer.get_applied_optimizations()
    
    def test_optimize_with_pruning(self):
        """Test optimization with pruning enabled."""
        config = OptimizationConfig(
            pruning_enabled=True,
            pruning_sparsity=0.6
        )
        
        result = self.optimizer.optimize(self.mock_model, config)
        
        assert result == self.mock_model  # Should return the same model for now
        assert "pruning_0.6" in self.optimizer.get_applied_optimizations()
    
    def test_optimize_with_both(self):
        """Test optimization with both quantization and pruning."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type=QuantizationType.INT8,
            pruning_enabled=True,
            pruning_sparsity=0.7
        )
        
        result = self.optimizer.optimize(self.mock_model, config)
        
        applied = self.optimizer.get_applied_optimizations()
        assert "quantization_int8" in applied
        assert "pruning_0.7" in applied


class TestTensorFlowOptimizer:
    """Test TensorFlowOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from neurolite.deployment.optimizer import TensorFlowOptimizer
        self.optimizer = TensorFlowOptimizer()
        self.mock_model = MockModel(framework="tensorflow")
    
    def test_initialization(self):
        """Test TensorFlowOptimizer initialization."""
        assert self.optimizer.applied_optimizations == []
    
    @pytest.mark.skip(reason="TensorFlow DLL issues on this system")
    def test_optimize_with_quantization(self):
        """Test optimization with quantization enabled."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type=QuantizationType.FLOAT16
        )
        
        result = self.optimizer.optimize(self.mock_model, config)
        
        assert result == self.mock_model  # Should return the same model for now
        assert "quantization_float16" in self.optimizer.get_applied_optimizations()


if __name__ == "__main__":
    pytest.main([__file__])