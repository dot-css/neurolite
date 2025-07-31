"""
Model optimization utilities for NeuroLite.

Provides functionality for model optimization including quantization,
pruning, and other compression techniques.
"""

from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core import get_logger, ExportError, DependencyError
from ..models.base import BaseModel


logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of model optimization."""
    
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    COMPRESSION = "compression"


class QuantizationType(Enum):
    """Types of quantization."""
    
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    FLOAT16 = "float16"


@dataclass
class OptimizationConfig:
    """
    Configuration for model optimization.
    
    Defines the optimization techniques to apply and their parameters.
    """
    
    # Quantization settings
    quantization_enabled: bool = False
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    quantization_calibration_data: Optional[Any] = None
    
    # Pruning settings
    pruning_enabled: bool = False
    pruning_sparsity: float = 0.5
    pruning_structured: bool = False
    
    # General optimization settings
    optimize_for_inference: bool = True
    target_device: str = "cpu"  # "cpu", "gpu", "mobile", "edge"
    
    # Performance targets
    target_latency_ms: Optional[float] = None
    target_memory_mb: Optional[float] = None
    max_accuracy_drop: float = 0.05  # Maximum acceptable accuracy drop
    
    # Additional options
    additional_optimizations: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize additional optimizations if None."""
        if self.additional_optimizations is None:
            self.additional_optimizations = {}


@dataclass
class OptimizationResult:
    """
    Result of model optimization.
    
    Contains information about the optimization process and results.
    """
    
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    optimization_time_seconds: float
    
    # Performance metrics
    original_inference_time_ms: Optional[float] = None
    optimized_inference_time_ms: Optional[float] = None
    speedup_ratio: Optional[float] = None
    
    # Accuracy metrics
    original_accuracy: Optional[float] = None
    optimized_accuracy: Optional[float] = None
    accuracy_drop: Optional[float] = None
    
    # Applied optimizations
    applied_optimizations: List[str] = None
    optimization_config: Optional[OptimizationConfig] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.applied_optimizations is None:
            self.applied_optimizations = []
        
        # Calculate speedup ratio
        if (self.original_inference_time_ms is not None and 
            self.optimized_inference_time_ms is not None and
            self.optimized_inference_time_ms > 0):
            self.speedup_ratio = self.original_inference_time_ms / self.optimized_inference_time_ms
        
        # Calculate accuracy drop
        if (self.original_accuracy is not None and 
            self.optimized_accuracy is not None):
            self.accuracy_drop = self.original_accuracy - self.optimized_accuracy


class ModelOptimizer:
    """
    Model optimizer for NeuroLite.
    
    Provides various optimization techniques including quantization,
    pruning, and other compression methods.
    """
    
    def __init__(self):
        """Initialize the model optimizer."""
        self._optimizers = {
            "pytorch": PyTorchOptimizer(),
            "tensorflow": TensorFlowOptimizer(),
            "onnx": ONNXOptimizer()
        }
    
    def optimize_model(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[BaseModel, OptimizationResult]:
        """
        Optimize a model according to the given configuration.
        
        Args:
            model: Model to optimize
            config: Optimization configuration
            validation_data: Data for validation and calibration
            output_path: Optional path to save optimized model
            
        Returns:
            Tuple of (optimized_model, optimization_result)
            
        Raises:
            ExportError: If optimization fails
        """
        import time
        
        if not model.is_trained:
            raise ExportError("optimization", "Model must be trained before optimization")
        
        framework = model.capabilities.framework.lower()
        optimizer = self._optimizers.get(framework)
        
        if not optimizer:
            raise ExportError("optimization", f"No optimizer available for framework: {framework}")
        
        logger.info(f"Starting model optimization for {framework} model")
        start_time = time.time()
        
        # Get original model metrics
        original_size = self._get_model_size(model)
        original_accuracy = None
        original_inference_time = None
        
        if validation_data is not None:
            original_accuracy = self._evaluate_model_accuracy(model, validation_data)
            original_inference_time = self._measure_inference_time(model, validation_data[0][:1])
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, config, validation_data)
        
        # Get optimized model metrics
        optimized_size = self._get_model_size(optimized_model)
        optimized_accuracy = None
        optimized_inference_time = None
        
        if validation_data is not None:
            optimized_accuracy = self._evaluate_model_accuracy(optimized_model, validation_data)
            optimized_inference_time = self._measure_inference_time(optimized_model, validation_data[0][:1])
        
        optimization_time = time.time() - start_time
        
        # Create optimization result
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=original_size / optimized_size if optimized_size > 0 else 1.0,
            optimization_time_seconds=optimization_time,
            original_inference_time_ms=original_inference_time,
            optimized_inference_time_ms=optimized_inference_time,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            applied_optimizations=optimizer.get_applied_optimizations(),
            optimization_config=config
        )
        
        # Save optimized model if path provided
        if output_path is not None:
            optimized_model.save(str(output_path))
            logger.info(f"Saved optimized model to: {output_path}")
        
        logger.info(
            f"Optimization completed in {optimization_time:.2f}s. "
            f"Size reduction: {result.compression_ratio:.2f}x "
            f"({original_size:.2f}MB -> {optimized_size:.2f}MB)"
        )
        
        if result.accuracy_drop is not None:
            logger.info(f"Accuracy change: {result.accuracy_drop:.4f}")
        
        return optimized_model, result
    
    def _get_model_size(self, model: BaseModel) -> float:
        """Get model size in MB."""
        if hasattr(model, 'metadata') and model.metadata and model.metadata.model_size_mb:
            return model.metadata.model_size_mb
        
        # Estimate size based on framework
        framework = model.capabilities.framework.lower()
        
        if framework == "pytorch":
            return self._get_pytorch_model_size(model)
        elif framework == "tensorflow":
            return self._get_tensorflow_model_size(model)
        else:
            return 0.0  # Unknown size
    
    def _get_pytorch_model_size(self, model: BaseModel) -> float:
        """Get PyTorch model size."""
        try:
            import torch
            
            pytorch_model = getattr(model, 'model', None) or getattr(model, 'pytorch_model', None)
            if pytorch_model is None:
                return 0.0
            
            # Calculate parameter size
            param_size = 0
            for param in pytorch_model.parameters():
                param_size += param.nelement() * param.element_size()
            
            # Calculate buffer size
            buffer_size = 0
            for buffer in pytorch_model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not calculate PyTorch model size: {e}")
            return 0.0
    
    def _get_tensorflow_model_size(self, model: BaseModel) -> float:
        """Get TensorFlow model size."""
        try:
            import tensorflow as tf
            
            tf_model = getattr(model, 'model', None) or getattr(model, 'tf_model', None)
            if tf_model is None:
                return 0.0
            
            # Count parameters
            total_params = tf_model.count_params()
            # Assume float32 parameters (4 bytes each)
            return (total_params * 4) / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not calculate TensorFlow model size: {e}")
            return 0.0
    
    def _evaluate_model_accuracy(self, model: BaseModel, validation_data: Tuple[Any, Any]) -> float:
        """Evaluate model accuracy on validation data."""
        try:
            X_val, y_val = validation_data
            predictions = model.predict(X_val)
            
            if hasattr(predictions, 'predictions'):
                predictions = predictions.predictions
            
            # Calculate accuracy (assuming classification)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class classification
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                # Binary classification or regression
                predicted_classes = (predictions > 0.5).astype(int) if predictions.max() <= 1.0 else predictions
            
            if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                y_val = np.argmax(y_val, axis=1)
            
            accuracy = np.mean(predicted_classes == y_val)
            return float(accuracy)
            
        except Exception as e:
            logger.warning(f"Could not evaluate model accuracy: {e}")
            return None
    
    def _measure_inference_time(self, model: BaseModel, sample_input: Any, num_runs: int = 100) -> float:
        """Measure model inference time."""
        try:
            import time
            
            # Warm up
            for _ in range(10):
                model.predict(sample_input)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(num_runs):
                model.predict(sample_input)
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / num_runs) * 1000
            return avg_time_ms
            
        except Exception as e:
            logger.warning(f"Could not measure inference time: {e}")
            return None


class BaseFrameworkOptimizer:
    """Base class for framework-specific optimizers."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.applied_optimizations = []
    
    def optimize(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """
        Optimize the model.
        
        Args:
            model: Model to optimize
            config: Optimization configuration
            validation_data: Validation data for calibration
            
        Returns:
            Optimized model
        """
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def get_applied_optimizations(self) -> List[str]:
        """Get list of applied optimizations."""
        return self.applied_optimizations.copy()


class PyTorchOptimizer(BaseFrameworkOptimizer):
    """PyTorch-specific model optimizer."""
    
    def optimize(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """Optimize PyTorch model."""
        try:
            import torch
            import torch.quantization as quant
        except ImportError:
            raise DependencyError("PyTorch is required for PyTorch model optimization")
        
        self.applied_optimizations = []
        optimized_model = model  # Start with original model
        
        # Apply quantization
        if config.quantization_enabled:
            optimized_model = self._apply_pytorch_quantization(
                optimized_model, config, validation_data
            )
            self.applied_optimizations.append(f"quantization_{config.quantization_type.value}")
        
        # Apply pruning
        if config.pruning_enabled:
            optimized_model = self._apply_pytorch_pruning(
                optimized_model, config
            )
            self.applied_optimizations.append(f"pruning_{config.pruning_sparsity}")
        
        return optimized_model
    
    def _apply_pytorch_quantization(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """Apply PyTorch quantization."""
        # This is a simplified implementation
        # In practice, you would need more sophisticated quantization logic
        logger.info(f"Applying PyTorch {config.quantization_type.value} quantization")
        
        # For now, return the original model
        # TODO: Implement actual PyTorch quantization
        return model
    
    def _apply_pytorch_pruning(
        self,
        model: BaseModel,
        config: OptimizationConfig
    ) -> BaseModel:
        """Apply PyTorch pruning."""
        logger.info(f"Applying PyTorch pruning with sparsity {config.pruning_sparsity}")
        
        # For now, return the original model
        # TODO: Implement actual PyTorch pruning
        return model


class TensorFlowOptimizer(BaseFrameworkOptimizer):
    """TensorFlow-specific model optimizer."""
    
    def optimize(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """Optimize TensorFlow model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError("TensorFlow is required for TensorFlow model optimization")
        
        self.applied_optimizations = []
        optimized_model = model  # Start with original model
        
        # Apply quantization
        if config.quantization_enabled:
            optimized_model = self._apply_tensorflow_quantization(
                optimized_model, config, validation_data
            )
            self.applied_optimizations.append(f"quantization_{config.quantization_type.value}")
        
        # Apply pruning
        if config.pruning_enabled:
            optimized_model = self._apply_tensorflow_pruning(
                optimized_model, config
            )
            self.applied_optimizations.append(f"pruning_{config.pruning_sparsity}")
        
        return optimized_model
    
    def _apply_tensorflow_quantization(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """Apply TensorFlow quantization."""
        logger.info(f"Applying TensorFlow {config.quantization_type.value} quantization")
        
        # For now, return the original model
        # TODO: Implement actual TensorFlow quantization
        return model
    
    def _apply_tensorflow_pruning(
        self,
        model: BaseModel,
        config: OptimizationConfig
    ) -> BaseModel:
        """Apply TensorFlow pruning."""
        logger.info(f"Applying TensorFlow pruning with sparsity {config.pruning_sparsity}")
        
        # For now, return the original model
        # TODO: Implement actual TensorFlow pruning
        return model


class ONNXOptimizer(BaseFrameworkOptimizer):
    """ONNX-specific model optimizer."""
    
    def optimize(
        self,
        model: BaseModel,
        config: OptimizationConfig,
        validation_data: Optional[Tuple[Any, Any]] = None
    ) -> BaseModel:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic
        except ImportError:
            raise DependencyError("ONNX and ONNXRuntime are required for ONNX model optimization")
        
        self.applied_optimizations = []
        
        # For ONNX optimization, we would typically work with ONNX model files
        # This is a placeholder implementation
        logger.info("ONNX optimization not fully implemented yet")
        
        return model