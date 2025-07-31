"""
PyTorch format exporter for NeuroLite models.

Provides functionality to export PyTorch models to various PyTorch formats
including TorchScript and standard PyTorch model files.
"""

from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from ...core import get_logger, ExportError, DependencyError
from ...models.base import BaseModel
from .base import BaseFormatExporter, ExportFormat


logger = get_logger(__name__)


class PyTorchExporter(BaseFormatExporter):
    """
    Exporter for PyTorch formats.
    
    Supports exporting PyTorch models to TorchScript and standard PyTorch formats
    with optimization and tracing support.
    """
    
    def __init__(
        self,
        export_type: str = "torchscript",
        trace_mode: bool = True,
        optimize_for_inference: bool = True,
        **kwargs
    ):
        """
        Initialize PyTorch exporter.
        
        Args:
            export_type: Type of export ('torchscript', 'state_dict', 'full_model')
            trace_mode: Whether to use tracing (True) or scripting (False) for TorchScript
            optimize_for_inference: Whether to optimize model for inference
            **kwargs: Additional export parameters
        """
        super().__init__(**kwargs)
        self.export_type = export_type
        self.trace_mode = trace_mode
        self.optimize_for_inference = optimize_for_inference
    
    @property
    def format(self) -> ExportFormat:
        """Get the export format."""
        if self.export_type == "torchscript":
            return ExportFormat.TORCHSCRIPT
        else:
            return ExportFormat.PYTORCH
    
    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        if self.export_type == "torchscript":
            return ".pt"
        else:
            return ".pth"
    
    def can_export(self, model: BaseModel) -> bool:
        """
        Check if model can be exported to PyTorch format.
        
        Args:
            model: Model to check
            
        Returns:
            True if model can be exported to PyTorch format
        """
        # Check if model is trained
        if not model.is_trained:
            logger.warning("Model must be trained before PyTorch export")
            return False
        
        # Check framework support
        framework = model.capabilities.framework.lower()
        supported_frameworks = ["pytorch"]
        
        if framework not in supported_frameworks:
            logger.warning(f"PyTorch export only supported for PyTorch models, got: {framework}")
            return False
        
        return True
    
    def export(
        self,
        model: BaseModel,
        output_path: Union[str, Path],
        sample_input: Optional[Any] = None,
        strict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export model to PyTorch format.
        
        Args:
            model: Trained model to export
            output_path: Path to save PyTorch model
            sample_input: Sample input for tracing (required for TorchScript tracing)
            strict: Whether to use strict mode for TorchScript
            **kwargs: Additional export parameters
            
        Returns:
            Export metadata dictionary
            
        Raises:
            ExportError: If export fails
        """
        if not self.can_export(model):
            raise ExportError("pytorch", "Model cannot be exported to PyTorch format")
        
        output_path = self.prepare_output_path(output_path)
        
        try:
            import torch
        except ImportError:
            raise DependencyError("PyTorch is required for PyTorch model export")
        
        try:
            # Get the underlying PyTorch model
            pytorch_model = self._get_pytorch_model(model)
            pytorch_model.eval()
            
            logger.info(f"Exporting PyTorch model to {self.export_type}: {output_path}")
            
            if self.export_type == "torchscript":
                return self._export_torchscript(
                    pytorch_model, output_path, sample_input, strict, **kwargs
                )
            elif self.export_type == "state_dict":
                return self._export_state_dict(
                    pytorch_model, output_path, model, **kwargs
                )
            elif self.export_type == "full_model":
                return self._export_full_model(
                    pytorch_model, output_path, **kwargs
                )
            else:
                raise ExportError("pytorch", f"Unsupported export type: {self.export_type}")
                
        except Exception as e:
            raise ExportError("pytorch", f"Export failed: {str(e)}")
    
    def _export_torchscript(
        self,
        pytorch_model: Any,
        output_path: Path,
        sample_input: Any,
        strict: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Export model to TorchScript format."""
        import torch
        
        if self.trace_mode:
            if sample_input is None:
                raise ExportError("pytorch", "sample_input is required for TorchScript tracing")
            
            # Convert sample input to tensor if needed
            if not isinstance(sample_input, torch.Tensor):
                sample_input = torch.tensor(sample_input, dtype=torch.float32)
            
            # Trace the model
            traced_model = torch.jit.trace(pytorch_model, sample_input, strict=strict)
        else:
            # Script the model
            traced_model = torch.jit.script(pytorch_model)
        
        # Optimize for inference if requested
        if self.optimize_for_inference:
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save the model
        traced_model.save(str(output_path))
        
        # Get model info
        model_info = {
            "export_type": "torchscript",
            "trace_mode": self.trace_mode,
            "optimized": self.optimize_for_inference,
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
        
        return {
            "format": "torchscript",
            "output_path": str(output_path),
            "model_info": model_info
        }
    
    def _export_state_dict(
        self,
        pytorch_model: Any,
        output_path: Path,
        original_model: BaseModel,
        **kwargs
    ) -> Dict[str, Any]:
        """Export model state dictionary."""
        import torch
        
        # Save state dict along with model metadata
        save_dict = {
            "state_dict": pytorch_model.state_dict(),
            "model_class": pytorch_model.__class__.__name__,
            "model_config": original_model.get_config(),
            "metadata": original_model.metadata.__dict__ if original_model.metadata else None
        }
        
        torch.save(save_dict, str(output_path))
        
        model_info = {
            "export_type": "state_dict",
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
        
        return {
            "format": "pytorch",
            "output_path": str(output_path),
            "model_info": model_info
        }
    
    def _export_full_model(
        self,
        pytorch_model: Any,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Export full PyTorch model."""
        import torch
        
        # Save the entire model
        torch.save(pytorch_model, str(output_path))
        
        model_info = {
            "export_type": "full_model",
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
        
        return {
            "format": "pytorch",
            "output_path": str(output_path),
            "model_info": model_info
        }
    
    def validate_export(
        self,
        original_model: BaseModel,
        exported_path: Union[str, Path],
        test_input: Any,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Validate PyTorch export by comparing outputs.
        
        Args:
            original_model: Original model
            exported_path: Path to exported PyTorch model
            test_input: Test input for validation
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if validation passes
        """
        try:
            import torch
        except ImportError:
            raise DependencyError("PyTorch is required for PyTorch model validation")
        
        try:
            # Get original model output
            original_output = original_model.predict(test_input)
            if hasattr(original_output, 'predictions'):
                original_output = original_output.predictions
            
            # Load exported model
            if self.export_type == "torchscript":
                exported_model = torch.jit.load(str(exported_path))
            else:
                exported_model = torch.load(str(exported_path))
                if isinstance(exported_model, dict) and "state_dict" in exported_model:
                    # For state dict exports, we need the original model architecture
                    logger.warning("Cannot validate state_dict export without model architecture")
                    return True  # Skip validation for state dict
            
            exported_model.eval()
            
            # Prepare input for PyTorch model
            if not isinstance(test_input, torch.Tensor):
                test_input = torch.tensor(test_input, dtype=torch.float32)
            
            # Run inference
            with torch.no_grad():
                exported_output = exported_model(test_input)
                if isinstance(exported_output, torch.Tensor):
                    exported_output = exported_output.numpy()
            
            # Compare outputs
            is_valid = self.compare_outputs(original_output, exported_output, tolerance)
            
            if is_valid:
                logger.info("PyTorch export validation passed")
            else:
                logger.error("PyTorch export validation failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"PyTorch validation failed: {e}")
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
            raise ExportError("pytorch", "Could not extract PyTorch model from wrapper")
    
    def get_model_complexity_info(self, model: BaseModel, input_size: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Get model complexity information (parameters, FLOPs, etc.).
        
        Args:
            model: Model to analyze
            input_size: Input size for FLOP calculation
            
        Returns:
            Dictionary with complexity information
        """
        try:
            import torch
            from torchinfo import summary
        except ImportError:
            logger.warning("torchinfo not available for model complexity analysis")
            return {}
        
        try:
            pytorch_model = self._get_pytorch_model(model)
            
            # Get model summary
            model_summary = summary(
                pytorch_model,
                input_size=input_size,
                verbose=0
            )
            
            return {
                "total_params": model_summary.total_params,
                "trainable_params": model_summary.trainable_params,
                "model_size_mb": model_summary.total_param_bytes / (1024 * 1024),
                "input_size": input_size,
                "total_mult_adds": getattr(model_summary, 'total_mult_adds', None)
            }
            
        except Exception as e:
            logger.error(f"Failed to get model complexity info: {e}")
            return {"error": str(e)}