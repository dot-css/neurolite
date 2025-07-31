"""
ONNX format exporter for NeuroLite models.

Provides functionality to export PyTorch and TensorFlow models to ONNX format
with validation and optimization support.
"""

from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from ...core import get_logger, ExportError, DependencyError
from ...models.base import BaseModel
from .base import BaseFormatExporter, ExportFormat


logger = get_logger(__name__)


class ONNXExporter(BaseFormatExporter):
    """
    Exporter for ONNX (Open Neural Network Exchange) format.
    
    Supports exporting PyTorch and TensorFlow models to ONNX format
    with automatic input shape inference and validation.
    """
    
    def __init__(
        self,
        opset_version: int = 11,
        do_constant_folding: bool = True,
        export_params: bool = True,
        **kwargs
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            opset_version: ONNX opset version to use
            do_constant_folding: Whether to apply constant folding optimization
            export_params: Whether to export model parameters
            **kwargs: Additional ONNX export parameters
        """
        super().__init__(**kwargs)
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.export_params = export_params
    
    @property
    def format(self) -> ExportFormat:
        """Get the export format."""
        return ExportFormat.ONNX
    
    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        return ".onnx"
    
    def can_export(self, model: BaseModel) -> bool:
        """
        Check if model can be exported to ONNX.
        
        Args:
            model: Model to check
            
        Returns:
            True if model can be exported to ONNX
        """
        # Check if model is trained
        if not model.is_trained:
            logger.warning("Model must be trained before ONNX export")
            return False
        
        # Check framework support
        framework = model.capabilities.framework.lower()
        supported_frameworks = ["pytorch", "tensorflow"]
        
        if framework not in supported_frameworks:
            logger.warning(f"ONNX export not supported for framework: {framework}")
            return False
        
        return True
    
    def export(
        self,
        model: BaseModel,
        output_path: Union[str, Path],
        sample_input: Optional[Any] = None,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export model to ONNX format.
        
        Args:
            model: Trained model to export
            output_path: Path to save ONNX model
            sample_input: Sample input for tracing (required for PyTorch)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification for variable input sizes
            **kwargs: Additional export parameters
            
        Returns:
            Export metadata dictionary
            
        Raises:
            ExportError: If export fails
        """
        if not self.can_export(model):
            raise ExportError("onnx", "Model cannot be exported to ONNX format")
        
        output_path = self.prepare_output_path(output_path)
        framework = model.capabilities.framework.lower()
        
        try:
            if framework == "pytorch":
                return self._export_pytorch_model(
                    model, output_path, sample_input, input_names, 
                    output_names, dynamic_axes, **kwargs
                )
            elif framework == "tensorflow":
                return self._export_tensorflow_model(
                    model, output_path, sample_input, input_names,
                    output_names, **kwargs
                )
            else:
                raise ExportError("onnx", f"Unsupported framework: {framework}")
                
        except Exception as e:
            raise ExportError("onnx", f"Export failed: {str(e)}")
    
    def _export_pytorch_model(
        self,
        model: BaseModel,
        output_path: Path,
        sample_input: Any,
        input_names: Optional[list],
        output_names: Optional[list],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]],
        **kwargs
    ) -> Dict[str, Any]:
        """Export PyTorch model to ONNX."""
        try:
            import torch
            import torch.onnx
        except ImportError:
            raise DependencyError("PyTorch is required for PyTorch model ONNX export")
        
        if sample_input is None:
            raise ExportError("onnx", "sample_input is required for PyTorch model export")
        
        # Get the underlying PyTorch model
        pytorch_model = self._get_pytorch_model(model)
        pytorch_model.eval()
        
        # Convert sample input to tensor if needed
        if not isinstance(sample_input, torch.Tensor):
            sample_input = torch.tensor(sample_input, dtype=torch.float32)
        
        # Set default input/output names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        
        logger.info(f"Exporting PyTorch model to ONNX: {output_path}")
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            sample_input,
            str(output_path),
            export_params=self.export_params,
            opset_version=self.opset_version,
            do_constant_folding=self.do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs
        )
        
        # Get model info
        model_info = self.get_model_info(model)
        
        return {
            "format": "onnx",
            "output_path": str(output_path),
            "opset_version": self.opset_version,
            "input_names": input_names,
            "output_names": output_names,
            "model_info": model_info,
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
    
    def _export_tensorflow_model(
        self,
        model: BaseModel,
        output_path: Path,
        sample_input: Any,
        input_names: Optional[list],
        output_names: Optional[list],
        **kwargs
    ) -> Dict[str, Any]:
        """Export TensorFlow model to ONNX."""
        try:
            import tf2onnx
            import tensorflow as tf
        except ImportError:
            raise DependencyError("tf2onnx and TensorFlow are required for TensorFlow model ONNX export")
        
        # Get the underlying TensorFlow model
        tf_model = self._get_tensorflow_model(model)
        
        logger.info(f"Exporting TensorFlow model to ONNX: {output_path}")
        
        # Convert TensorFlow model to ONNX
        spec = (tf.TensorSpec(sample_input.shape, tf.float32, name="input"),)
        
        onnx_model, _ = tf2onnx.convert.from_keras(
            tf_model,
            input_signature=spec,
            opset=self.opset_version,
            **kwargs
        )
        
        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Get model info
        model_info = self.get_model_info(model)
        
        return {
            "format": "onnx",
            "output_path": str(output_path),
            "opset_version": self.opset_version,
            "model_info": model_info,
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
    
    def validate_export(
        self,
        original_model: BaseModel,
        exported_path: Union[str, Path],
        test_input: Any,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Validate ONNX export by comparing outputs.
        
        Args:
            original_model: Original model
            exported_path: Path to exported ONNX model
            test_input: Test input for validation
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if validation passes
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise DependencyError("onnxruntime is required for ONNX model validation")
        
        try:
            # Get original model output
            original_output = original_model.predict(test_input)
            if hasattr(original_output, 'predictions'):
                original_output = original_output.predictions
            
            # Load and run ONNX model
            ort_session = ort.InferenceSession(str(exported_path))
            input_name = ort_session.get_inputs()[0].name
            
            # Prepare input for ONNX runtime
            if not isinstance(test_input, np.ndarray):
                test_input = np.array(test_input, dtype=np.float32)
            
            onnx_output = ort_session.run(None, {input_name: test_input})[0]
            
            # Compare outputs
            is_valid = self.compare_outputs(original_output, onnx_output, tolerance)
            
            if is_valid:
                logger.info("ONNX export validation passed")
            else:
                logger.error("ONNX export validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False
    
    def _get_pytorch_model(self, model: BaseModel) -> Any:
        """Extract PyTorch model from NeuroLite model wrapper."""
        # Handle different PyTorch model wrapper types
        if hasattr(model, 'model'):
            return model.model
        elif hasattr(model, 'pytorch_model'):
            return model.pytorch_model
        elif hasattr(model, '_model'):
            return model._model
        else:
            raise ExportError("onnx", "Could not extract PyTorch model from wrapper")
    
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
            raise ExportError("onnx", "Could not extract TensorFlow model from wrapper")