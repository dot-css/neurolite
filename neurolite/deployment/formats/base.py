"""
Base format exporter interface and export format definitions.

Defines the abstract interface that all format exporters must implement
and the enumeration of supported export formats.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union
from pathlib import Path
import numpy as np

from ...core import get_logger
from ...models.base import BaseModel


logger = get_logger(__name__)


class ExportFormat(Enum):
    """Enumeration of supported model export formats."""
    
    ONNX = "onnx"
    TENSORFLOW_LITE = "tflite"
    PYTORCH = "pytorch"
    TENSORFLOW_SAVEDMODEL = "tf_savedmodel"
    TORCHSCRIPT = "torchscript"


class BaseFormatExporter(ABC):
    """
    Abstract base class for model format exporters.
    
    Defines the interface that all format-specific exporters must implement.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the format exporter.
        
        Args:
            **kwargs: Format-specific configuration options
        """
        self.config = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with config: {kwargs}")
    
    @property
    @abstractmethod
    def format(self) -> ExportFormat:
        """
        Get the export format this exporter handles.
        
        Returns:
            ExportFormat enum value
        """
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        Get the file extension for this format.
        
        Returns:
            File extension string (e.g., '.onnx', '.tflite')
        """
        pass
    
    @abstractmethod
    def can_export(self, model: BaseModel) -> bool:
        """
        Check if this exporter can handle the given model.
        
        Args:
            model: Model to check for export compatibility
            
        Returns:
            True if model can be exported, False otherwise
        """
        pass
    
    @abstractmethod
    def export(
        self,
        model: BaseModel,
        output_path: Union[str, Path],
        sample_input: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export the model to the target format.
        
        Args:
            model: Trained model to export
            output_path: Path where to save the exported model
            sample_input: Sample input for tracing/shape inference
            **kwargs: Additional export parameters
            
        Returns:
            Dictionary containing export metadata and information
            
        Raises:
            ExportError: If export fails
        """
        pass
    
    @abstractmethod
    def validate_export(
        self,
        original_model: BaseModel,
        exported_path: Union[str, Path],
        test_input: Any,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Validate that the exported model produces similar outputs to the original.
        
        Args:
            original_model: Original trained model
            exported_path: Path to the exported model
            test_input: Test input for validation
            tolerance: Numerical tolerance for output comparison
            
        Returns:
            True if validation passes, False otherwise
        """
        pass
    
    def get_model_info(self, model: BaseModel) -> Dict[str, Any]:
        """
        Extract relevant information from the model for export.
        
        Args:
            model: Model to extract information from
            
        Returns:
            Dictionary containing model information
        """
        info = {
            "model_class": model.__class__.__name__,
            "framework": model.capabilities.framework,
            "is_trained": model.is_trained,
            "task_types": [task.value for task in model.capabilities.supported_tasks],
            "data_types": [dtype.value for dtype in model.capabilities.supported_data_types]
        }
        
        if hasattr(model, 'metadata') and model.metadata:
            info.update({
                "model_name": model.metadata.name,
                "num_parameters": model.metadata.num_parameters,
                "model_size_mb": model.metadata.model_size_mb
            })
        
        return info
    
    def prepare_output_path(self, output_path: Union[str, Path]) -> Path:
        """
        Prepare and validate the output path for export.
        
        Args:
            output_path: Desired output path
            
        Returns:
            Validated Path object with correct extension
        """
        path = Path(output_path)
        
        # Add correct extension if not present
        if not path.suffix:
            path = path.with_suffix(self.file_extension)
        elif path.suffix != self.file_extension:
            logger.warning(
                f"Output path has extension {path.suffix}, but expected {self.file_extension}. "
                f"Using {self.file_extension}"
            )
            path = path.with_suffix(self.file_extension)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def compare_outputs(
        self,
        output1: np.ndarray,
        output2: np.ndarray,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Compare two model outputs for validation.
        
        Args:
            output1: First output array
            output2: Second output array  
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if outputs are similar within tolerance
        """
        try:
            # Ensure both outputs are numpy arrays
            if not isinstance(output1, np.ndarray):
                output1 = np.array(output1)
            if not isinstance(output2, np.ndarray):
                output2 = np.array(output2)
            
            # Check shapes match
            if output1.shape != output2.shape:
                logger.error(f"Output shapes don't match: {output1.shape} vs {output2.shape}")
                return False
            
            # Check if outputs are close within tolerance
            # Use both relative and absolute tolerance
            return np.allclose(output1, output2, rtol=tolerance, atol=tolerance)
            
        except Exception as e:
            logger.error(f"Error comparing outputs: {e}")
            return False