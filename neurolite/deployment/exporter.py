"""
Main model exporter for NeuroLite.

Provides a unified interface for exporting models to various formats
with automatic format selection and validation.
"""

from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from ..core import get_logger, ExportError
from ..models.base import BaseModel
from .formats import (
    BaseFormatExporter,
    ExportFormat,
    ONNXExporter,
    TensorFlowLiteExporter,
    PyTorchExporter
)


logger = get_logger(__name__)


@dataclass
class ExportedModel:
    """
    Container for exported model information.
    
    Contains metadata about the exported model including format,
    file path, validation results, and performance metrics.
    """
    
    format: str
    output_path: str
    file_size_mb: float
    validation_passed: bool
    export_metadata: Dict[str, Any]
    original_model_info: Dict[str, Any]
    export_time_seconds: Optional[float] = None
    validation_tolerance: Optional[float] = None


class ModelExporter:
    """
    Main model exporter for NeuroLite.
    
    Provides a unified interface for exporting trained models to various formats
    including ONNX, TensorFlow Lite, and PyTorch with automatic validation.
    """
    
    def __init__(self):
        """Initialize the model exporter."""
        self._exporters: Dict[ExportFormat, BaseFormatExporter] = {}
        self._register_default_exporters()
    
    def _register_default_exporters(self) -> None:
        """Register default format exporters."""
        self._exporters[ExportFormat.ONNX] = ONNXExporter()
        self._exporters[ExportFormat.TENSORFLOW_LITE] = TensorFlowLiteExporter()
        self._exporters[ExportFormat.PYTORCH] = PyTorchExporter(export_type="state_dict")
        self._exporters[ExportFormat.TORCHSCRIPT] = PyTorchExporter(export_type="torchscript")
    
    def register_exporter(
        self,
        format_type: ExportFormat,
        exporter: BaseFormatExporter
    ) -> None:
        """
        Register a custom format exporter.
        
        Args:
            format_type: Export format type
            exporter: Format exporter instance
        """
        self._exporters[format_type] = exporter
        logger.debug(f"Registered custom exporter for format: {format_type.value}")
    
    def get_supported_formats(self, model: BaseModel) -> List[ExportFormat]:
        """
        Get list of supported export formats for a model.
        
        Args:
            model: Model to check export compatibility
            
        Returns:
            List of supported export formats
        """
        supported_formats = []
        
        for format_type, exporter in self._exporters.items():
            if exporter.can_export(model):
                supported_formats.append(format_type)
        
        return supported_formats
    
    def export_model(
        self,
        model: BaseModel,
        output_path: Union[str, Path],
        format: Union[str, ExportFormat] = "auto",
        sample_input: Optional[Any] = None,
        validate: bool = True,
        validation_tolerance: float = 1e-5,
        **kwargs
    ) -> ExportedModel:
        """
        Export a trained model to the specified format.
        
        Args:
            model: Trained model to export
            output_path: Path where to save the exported model
            format: Export format ('auto', 'onnx', 'tflite', 'pytorch', 'torchscript')
            sample_input: Sample input for tracing/shape inference
            validate: Whether to validate the exported model
            validation_tolerance: Numerical tolerance for validation
            **kwargs: Additional format-specific export parameters
            
        Returns:
            ExportedModel containing export information
            
        Raises:
            ExportError: If export fails
        """
        import time
        
        # Validate model is trained
        if not model.is_trained:
            raise ExportError("general", "Model must be trained before export")
        
        # Determine export format
        export_format = self._determine_export_format(model, format)
        
        # Get the appropriate exporter
        exporter = self._exporters.get(export_format)
        if not exporter:
            raise ExportError("general", f"No exporter available for format: {export_format.value}")
        
        # Check if exporter can handle the model
        if not exporter.can_export(model):
            raise ExportError(
                export_format.value,
                f"Model cannot be exported to {export_format.value} format"
            )
        
        logger.info(f"Exporting model to {export_format.value} format: {output_path}")
        
        # Perform export
        start_time = time.time()
        try:
            export_metadata = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                **kwargs
            )
        except Exception as e:
            raise ExportError(export_format.value, f"Export failed: {str(e)}")
        
        export_time = time.time() - start_time
        
        # Validate export if requested
        validation_passed = True
        if validate and sample_input is not None:
            try:
                validation_passed = exporter.validate_export(
                    original_model=model,
                    exported_path=output_path,
                    test_input=sample_input,
                    tolerance=validation_tolerance
                )
            except Exception as e:
                logger.warning(f"Export validation failed: {e}")
                validation_passed = False
        elif validate and sample_input is None:
            logger.warning("Cannot validate export without sample_input")
            validation_passed = False
        
        # Get original model info
        original_model_info = exporter.get_model_info(model)
        
        # Create exported model object
        exported_model = ExportedModel(
            format=export_format.value,
            output_path=str(output_path),
            file_size_mb=export_metadata.get("file_size_mb", 0.0),
            validation_passed=validation_passed,
            export_metadata=export_metadata,
            original_model_info=original_model_info,
            export_time_seconds=export_time,
            validation_tolerance=validation_tolerance if validate else None
        )
        
        logger.info(
            f"Export completed in {export_time:.2f}s. "
            f"File size: {exported_model.file_size_mb:.2f}MB. "
            f"Validation: {'PASSED' if validation_passed else 'FAILED' if validate else 'SKIPPED'}"
        )
        
        return exported_model
    
    def export_multiple_formats(
        self,
        model: BaseModel,
        output_dir: Union[str, Path],
        formats: List[Union[str, ExportFormat]],
        base_name: str,
        sample_input: Optional[Any] = None,
        validate: bool = True,
        **kwargs
    ) -> Dict[str, ExportedModel]:
        """
        Export model to multiple formats.
        
        Args:
            model: Trained model to export
            output_dir: Directory to save exported models
            formats: List of export formats
            base_name: Base name for exported files
            sample_input: Sample input for tracing/validation
            validate: Whether to validate exported models
            **kwargs: Additional export parameters
            
        Returns:
            Dictionary mapping format names to ExportedModel objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_models = {}
        
        for format_str in formats:
            try:
                # Determine format
                if isinstance(format_str, str):
                    format_enum = ExportFormat(format_str)
                else:
                    format_enum = format_str
                
                # Get exporter and file extension
                exporter = self._exporters.get(format_enum)
                if not exporter:
                    logger.warning(f"No exporter available for format: {format_enum.value}")
                    continue
                
                # Create output path
                output_path = output_dir / f"{base_name}{exporter.file_extension}"
                
                # Export model
                exported_model = self.export_model(
                    model=model,
                    output_path=output_path,
                    format=format_enum,
                    sample_input=sample_input,
                    validate=validate,
                    **kwargs
                )
                
                exported_models[format_enum.value] = exported_model
                
            except Exception as e:
                logger.error(f"Failed to export to {format_str}: {e}")
                continue
        
        logger.info(f"Exported model to {len(exported_models)} formats")
        return exported_models
    
    def _determine_export_format(
        self,
        model: BaseModel,
        format: Union[str, ExportFormat]
    ) -> ExportFormat:
        """
        Determine the appropriate export format.
        
        Args:
            model: Model to export
            format: Requested format or 'auto'
            
        Returns:
            Determined export format
            
        Raises:
            ExportError: If format cannot be determined
        """
        if isinstance(format, ExportFormat):
            return format
        
        if isinstance(format, str):
            if format == "auto":
                return self._auto_select_format(model)
            else:
                try:
                    return ExportFormat(format)
                except ValueError:
                    raise ExportError("general", f"Unsupported export format: {format}")
        
        raise ExportError("general", f"Invalid format type: {type(format)}")
    
    def _auto_select_format(self, model: BaseModel) -> ExportFormat:
        """
        Automatically select the best export format for a model.
        
        Args:
            model: Model to select format for
            
        Returns:
            Selected export format
        """
        framework = model.capabilities.framework.lower()
        
        # Priority order for format selection
        if framework == "pytorch":
            # For PyTorch models, prefer ONNX for interoperability
            if ExportFormat.ONNX in self._exporters and self._exporters[ExportFormat.ONNX].can_export(model):
                return ExportFormat.ONNX
            elif ExportFormat.TORCHSCRIPT in self._exporters and self._exporters[ExportFormat.TORCHSCRIPT].can_export(model):
                return ExportFormat.TORCHSCRIPT
            else:
                return ExportFormat.PYTORCH
                
        elif framework == "tensorflow":
            # For TensorFlow models, prefer TFLite for mobile deployment
            if ExportFormat.TENSORFLOW_LITE in self._exporters and self._exporters[ExportFormat.TENSORFLOW_LITE].can_export(model):
                return ExportFormat.TENSORFLOW_LITE
            elif ExportFormat.ONNX in self._exporters and self._exporters[ExportFormat.ONNX].can_export(model):
                return ExportFormat.ONNX
            else:
                raise ExportError("auto", f"No suitable export format for TensorFlow model")
                
        elif framework == "sklearn":
            # For sklearn models, ONNX is the best option for deployment
            if ExportFormat.ONNX in self._exporters and self._exporters[ExportFormat.ONNX].can_export(model):
                return ExportFormat.ONNX
            else:
                raise ExportError("auto", f"No suitable export format for sklearn model")
        
        else:
            raise ExportError("auto", f"Auto format selection not supported for framework: {framework}")
    
    def get_export_info(self, model: BaseModel) -> Dict[str, Any]:
        """
        Get information about export options for a model.
        
        Args:
            model: Model to get export info for
            
        Returns:
            Dictionary with export information
        """
        supported_formats = self.get_supported_formats(model)
        auto_format = None
        
        try:
            auto_format = self._auto_select_format(model)
        except ExportError:
            pass
        
        return {
            "model_framework": model.capabilities.framework,
            "supported_formats": [fmt.value for fmt in supported_formats],
            "auto_selected_format": auto_format.value if auto_format else None,
            "requires_sample_input": model.capabilities.framework.lower() in ["pytorch"],
            "recommendations": self._get_format_recommendations(model)
        }
    
    def _get_format_recommendations(self, model: BaseModel) -> Dict[str, str]:
        """
        Get format recommendations based on use case.
        
        Args:
            model: Model to get recommendations for
            
        Returns:
            Dictionary mapping use cases to recommended formats
        """
        framework = model.capabilities.framework.lower()
        
        recommendations = {}
        
        if framework == "pytorch":
            recommendations.update({
                "cross_platform": "onnx",
                "mobile_deployment": "onnx",
                "production_serving": "torchscript",
                "model_sharing": "pytorch"
            })
        elif framework == "tensorflow":
            recommendations.update({
                "mobile_deployment": "tflite",
                "edge_deployment": "tflite",
                "cross_platform": "onnx",
                "production_serving": "tf_savedmodel"
            })
        elif framework == "sklearn":
            recommendations.update({
                "production_serving": "onnx",
                "cross_platform": "onnx"
            })
        
        return recommendations