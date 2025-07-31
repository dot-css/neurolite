"""
TensorFlow Lite format exporter for NeuroLite models.

Provides functionality to export TensorFlow models to TensorFlow Lite format
with quantization and optimization support.
"""

from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import numpy as np

from ...core import get_logger, ExportError, DependencyError
from ...models.base import BaseModel
from .base import BaseFormatExporter, ExportFormat


logger = get_logger(__name__)


class TensorFlowLiteExporter(BaseFormatExporter):
    """
    Exporter for TensorFlow Lite format.
    
    Supports exporting TensorFlow models to TensorFlow Lite format
    with quantization and optimization options.
    """
    
    def __init__(
        self,
        quantize: bool = False,
        quantization_type: str = "dynamic",
        representative_dataset: Optional[callable] = None,
        **kwargs
    ):
        """
        Initialize TensorFlow Lite exporter.
        
        Args:
            quantize: Whether to apply quantization
            quantization_type: Type of quantization ('dynamic', 'int8', 'float16')
            representative_dataset: Representative dataset for quantization
            **kwargs: Additional TFLite conversion parameters
        """
        super().__init__(**kwargs)
        self.quantize = quantize
        self.quantization_type = quantization_type
        self.representative_dataset = representative_dataset
    
    @property
    def format(self) -> ExportFormat:
        """Get the export format."""
        return ExportFormat.TENSORFLOW_LITE
    
    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        return ".tflite"
    
    def can_export(self, model: BaseModel) -> bool:
        """
        Check if model can be exported to TensorFlow Lite.
        
        Args:
            model: Model to check
            
        Returns:
            True if model can be exported to TensorFlow Lite
        """
        # Check if model is trained
        if not model.is_trained:
            logger.warning("Model must be trained before TensorFlow Lite export")
            return False
        
        # Check framework support
        framework = model.capabilities.framework.lower()
        supported_frameworks = ["tensorflow"]
        
        if framework not in supported_frameworks:
            logger.warning(f"TensorFlow Lite export only supported for TensorFlow models, got: {framework}")
            return False
        
        return True
    
    def export(
        self,
        model: BaseModel,
        output_path: Union[str, Path],
        sample_input: Optional[Any] = None,
        optimize: bool = True,
        target_spec: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export model to TensorFlow Lite format.
        
        Args:
            model: Trained model to export
            output_path: Path to save TFLite model
            sample_input: Sample input for shape inference
            optimize: Whether to apply optimizations
            target_spec: Target specification for optimization
            **kwargs: Additional export parameters
            
        Returns:
            Export metadata dictionary
            
        Raises:
            ExportError: If export fails
        """
        if not self.can_export(model):
            raise ExportError("tflite", "Model cannot be exported to TensorFlow Lite format")
        
        output_path = self.prepare_output_path(output_path)
        
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError("TensorFlow is required for TensorFlow Lite export")
        
        try:
            # Get the underlying TensorFlow model
            tf_model = self._get_tensorflow_model(model)
            
            logger.info(f"Exporting TensorFlow model to TFLite: {output_path}")
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            
            # Configure optimization
            if optimize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Configure quantization
            if self.quantize:
                self._configure_quantization(converter, target_spec)
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model info
            model_info = self.get_model_info(model)
            
            # Get model size
            file_size_mb = len(tflite_model) / (1024 * 1024)
            
            return {
                "format": "tflite",
                "output_path": str(output_path),
                "quantized": self.quantize,
                "quantization_type": self.quantization_type if self.quantize else None,
                "optimized": optimize,
                "model_info": model_info,
                "file_size_mb": file_size_mb
            }
            
        except Exception as e:
            raise ExportError("tflite", f"Export failed: {str(e)}")
    
    def _configure_quantization(
        self,
        converter: Any,
        target_spec: Optional[List[str]] = None
    ) -> None:
        """
        Configure quantization settings for TFLite converter.
        
        Args:
            converter: TFLite converter instance
            target_spec: Target specification for optimization
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError("TensorFlow is required for quantization configuration")
        
        if self.quantization_type == "dynamic":
            # Dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif self.quantization_type == "int8":
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            if self.representative_dataset:
                converter.representative_dataset = self.representative_dataset
            
        elif self.quantization_type == "float16":
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        # Set target specification if provided
        if target_spec:
            if "gpu" in target_spec:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
            elif "edge_tpu" in target_spec:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
    
    def validate_export(
        self,
        original_model: BaseModel,
        exported_path: Union[str, Path],
        test_input: Any,
        tolerance: float = 1e-3  # Higher tolerance for quantized models
    ) -> bool:
        """
        Validate TensorFlow Lite export by comparing outputs.
        
        Args:
            original_model: Original model
            exported_path: Path to exported TFLite model
            test_input: Test input for validation
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if validation passes
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError("TensorFlow is required for TFLite model validation")
        
        try:
            # Get original model output
            original_output = original_model.predict(test_input)
            if hasattr(original_output, 'predictions'):
                original_output = original_output.predictions
            
            # Load and run TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(exported_path))
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input for TFLite interpreter
            if not isinstance(test_input, np.ndarray):
                test_input = np.array(test_input, dtype=np.float32)
            
            # Ensure input matches expected shape and type
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            if test_input.shape != tuple(input_shape):
                # Try to reshape if possible
                if test_input.size == np.prod(input_shape):
                    test_input = test_input.reshape(input_shape)
                else:
                    logger.error(f"Input shape mismatch: {test_input.shape} vs {input_shape}")
                    return False
            
            test_input = test_input.astype(input_dtype)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Compare outputs
            is_valid = self.compare_outputs(original_output, tflite_output, tolerance)
            
            if is_valid:
                logger.info("TensorFlow Lite export validation passed")
            else:
                logger.error("TensorFlow Lite export validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"TFLite validation failed: {e}")
            return False
    
    def _get_tensorflow_model(self, model: BaseModel) -> Any:
        """Extract TensorFlow model from NeuroLite model wrapper."""
        # Handle different TensorFlow model wrapper types
        if hasattr(model, 'model'):
            return model.model
        elif hasattr(model, 'tf_model'):
            return model.tf_model
        elif hasattr(model, '_model'):
            return model._model
        else:
            raise ExportError("tflite", "Could not extract TensorFlow model from wrapper")
    
    def get_model_size_info(self, exported_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed size information about the exported TFLite model.
        
        Args:
            exported_path: Path to exported TFLite model
            
        Returns:
            Dictionary with size information
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError("TensorFlow is required for model size analysis")
        
        try:
            # Load model
            interpreter = tf.lite.Interpreter(model_path=str(exported_path))
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Calculate file size
            file_size = Path(exported_path).stat().st_size
            
            return {
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "num_inputs": len(input_details),
                "num_outputs": len(output_details),
                "input_shapes": [detail['shape'].tolist() for detail in input_details],
                "output_shapes": [detail['shape'].tolist() for detail in output_details],
                "input_types": [str(detail['dtype']) for detail in input_details],
                "output_types": [str(detail['dtype']) for detail in output_details]
            }
            
        except Exception as e:
            logger.error(f"Failed to get model size info: {e}")
            return {"error": str(e)}