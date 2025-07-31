"""
Deployment validation utilities for NeuroLite.

Provides functionality to validate exported models and ensure they
maintain accuracy and performance characteristics.
"""

from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core import get_logger, ExportError
from ..models.base import BaseModel


logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Status of validation checks."""
    
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationCheck:
    """
    Individual validation check result.
    
    Represents the result of a single validation check.
    """
    
    name: str
    status: ValidationStatus
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    tolerance: Optional[float] = None


@dataclass
class ValidationResult:
    """
    Complete validation result for an exported model.
    
    Contains all validation checks and overall status.
    """
    
    overall_status: ValidationStatus
    checks: List[ValidationCheck]
    model_path: str
    validation_time_seconds: float
    
    # Summary statistics
    num_passed: int = 0
    num_failed: int = 0
    num_warnings: int = 0
    num_skipped: int = 0
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.num_passed = sum(1 for check in self.checks if check.status == ValidationStatus.PASSED)
        self.num_failed = sum(1 for check in self.checks if check.status == ValidationStatus.FAILED)
        self.num_warnings = sum(1 for check in self.checks if check.status == ValidationStatus.WARNING)
        self.num_skipped = sum(1 for check in self.checks if check.status == ValidationStatus.SKIPPED)
        
        # Determine overall status
        if self.num_failed > 0:
            self.overall_status = ValidationStatus.FAILED
        elif self.num_warnings > 0:
            self.overall_status = ValidationStatus.WARNING
        elif self.num_passed > 0:
            self.overall_status = ValidationStatus.PASSED
        else:
            self.overall_status = ValidationStatus.SKIPPED


class DeploymentValidator:
    """
    Validator for exported models.
    
    Provides comprehensive validation of exported models including
    accuracy, performance, and compatibility checks.
    """
    
    def __init__(
        self,
        accuracy_tolerance: float = 1e-5,
        performance_tolerance: float = 0.1,
        enable_performance_checks: bool = True
    ):
        """
        Initialize the deployment validator.
        
        Args:
            accuracy_tolerance: Tolerance for accuracy comparisons
            performance_tolerance: Tolerance for performance comparisons (as ratio)
            enable_performance_checks: Whether to run performance checks
        """
        self.accuracy_tolerance = accuracy_tolerance
        self.performance_tolerance = performance_tolerance
        self.enable_performance_checks = enable_performance_checks
    
    def validate_export(
        self,
        original_model: BaseModel,
        exported_model_path: Union[str, Path],
        test_data: Tuple[Any, Any],
        export_format: str,
        sample_input: Optional[Any] = None
    ) -> ValidationResult:
        """
        Validate an exported model against the original.
        
        Args:
            original_model: Original trained model
            exported_model_path: Path to exported model
            test_data: Test data as (X_test, y_test)
            export_format: Format of exported model
            sample_input: Sample input for inference testing
            
        Returns:
            ValidationResult with all check results
        """
        import time
        
        start_time = time.time()
        checks = []
        
        logger.info(f"Starting validation of exported {export_format} model: {exported_model_path}")
        
        # File existence check
        checks.append(self._check_file_exists(exported_model_path))
        
        # File size check
        checks.append(self._check_file_size(exported_model_path))
        
        # Model loading check
        load_check, loaded_model = self._check_model_loading(exported_model_path, export_format)
        checks.append(load_check)
        
        if loaded_model is not None:
            # Accuracy validation
            if test_data is not None:
                checks.append(self._check_accuracy_consistency(
                    original_model, loaded_model, test_data, export_format
                ))
            
            # Performance validation
            if self.enable_performance_checks and sample_input is not None:
                checks.append(self._check_inference_performance(
                    original_model, loaded_model, sample_input, export_format
                ))
            
            # Output shape validation
            if sample_input is not None:
                checks.append(self._check_output_shapes(
                    original_model, loaded_model, sample_input, export_format
                ))
        
        validation_time = time.time() - start_time
        
        # Create validation result
        result = ValidationResult(
            overall_status=ValidationStatus.PASSED,  # Will be updated in __post_init__
            checks=checks,
            model_path=str(exported_model_path),
            validation_time_seconds=validation_time
        )
        
        logger.info(
            f"Validation completed in {validation_time:.2f}s. "
            f"Status: {result.overall_status.value.upper()}. "
            f"Passed: {result.num_passed}, Failed: {result.num_failed}, "
            f"Warnings: {result.num_warnings}, Skipped: {result.num_skipped}"
        )
        
        return result
    
    def _check_file_exists(self, model_path: Union[str, Path]) -> ValidationCheck:
        """Check if exported model file exists."""
        path = Path(model_path)
        
        if path.exists():
            return ValidationCheck(
                name="file_exists",
                status=ValidationStatus.PASSED,
                message=f"Model file exists: {path}"
            )
        else:
            return ValidationCheck(
                name="file_exists",
                status=ValidationStatus.FAILED,
                message=f"Model file does not exist: {path}"
            )
    
    def _check_file_size(self, model_path: Union[str, Path]) -> ValidationCheck:
        """Check exported model file size."""
        try:
            path = Path(model_path)
            if not path.exists():
                return ValidationCheck(
                    name="file_size",
                    status=ValidationStatus.SKIPPED,
                    message="File does not exist, skipping size check"
                )
            
            file_size_mb = path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 0:
                return ValidationCheck(
                    name="file_size",
                    status=ValidationStatus.PASSED,
                    message=f"Model file size: {file_size_mb:.2f} MB",
                    actual_value=file_size_mb
                )
            else:
                return ValidationCheck(
                    name="file_size",
                    status=ValidationStatus.FAILED,
                    message="Model file is empty",
                    actual_value=file_size_mb
                )
                
        except Exception as e:
            return ValidationCheck(
                name="file_size",
                status=ValidationStatus.FAILED,
                message=f"Error checking file size: {e}"
            )
    
    def _check_model_loading(
        self,
        model_path: Union[str, Path],
        export_format: str
    ) -> Tuple[ValidationCheck, Optional[Any]]:
        """Check if exported model can be loaded."""
        try:
            loaded_model = self._load_exported_model(model_path, export_format)
            
            return (
                ValidationCheck(
                    name="model_loading",
                    status=ValidationStatus.PASSED,
                    message=f"Successfully loaded {export_format} model"
                ),
                loaded_model
            )
            
        except Exception as e:
            return (
                ValidationCheck(
                    name="model_loading",
                    status=ValidationStatus.FAILED,
                    message=f"Failed to load {export_format} model: {e}"
                ),
                None
            )
    
    def _check_accuracy_consistency(
        self,
        original_model: BaseModel,
        loaded_model: Any,
        test_data: Tuple[Any, Any],
        export_format: str
    ) -> ValidationCheck:
        """Check accuracy consistency between original and exported models."""
        try:
            X_test, y_test = test_data
            
            # Get predictions from original model
            original_predictions = original_model.predict(X_test)
            if hasattr(original_predictions, 'predictions'):
                original_predictions = original_predictions.predictions
            
            # Get predictions from exported model
            exported_predictions = self._predict_with_exported_model(
                loaded_model, X_test, export_format
            )
            
            # Compare predictions
            if self._compare_predictions(original_predictions, exported_predictions):
                return ValidationCheck(
                    name="accuracy_consistency",
                    status=ValidationStatus.PASSED,
                    message="Predictions are consistent within tolerance",
                    tolerance=self.accuracy_tolerance
                )
            else:
                # Calculate actual difference
                diff = np.mean(np.abs(original_predictions - exported_predictions))
                return ValidationCheck(
                    name="accuracy_consistency",
                    status=ValidationStatus.FAILED,
                    message=f"Predictions differ beyond tolerance. Mean absolute difference: {diff:.6f}",
                    expected_value=self.accuracy_tolerance,
                    actual_value=diff,
                    tolerance=self.accuracy_tolerance
                )
                
        except Exception as e:
            return ValidationCheck(
                name="accuracy_consistency",
                status=ValidationStatus.FAILED,
                message=f"Error checking accuracy consistency: {e}"
            )
    
    def _check_inference_performance(
        self,
        original_model: BaseModel,
        loaded_model: Any,
        sample_input: Any,
        export_format: str
    ) -> ValidationCheck:
        """Check inference performance comparison."""
        try:
            # Measure original model inference time
            original_time = self._measure_inference_time(
                lambda x: original_model.predict(x), sample_input
            )
            
            # Measure exported model inference time
            exported_time = self._measure_inference_time(
                lambda x: self._predict_with_exported_model(loaded_model, x, export_format),
                sample_input
            )
            
            if original_time is None or exported_time is None:
                return ValidationCheck(
                    name="inference_performance",
                    status=ValidationStatus.SKIPPED,
                    message="Could not measure inference times"
                )
            
            # Calculate performance ratio
            performance_ratio = exported_time / original_time
            
            if performance_ratio <= (1 + self.performance_tolerance):
                status = ValidationStatus.PASSED
                message = f"Performance acceptable. Exported model: {exported_time:.2f}ms, Original: {original_time:.2f}ms"
            else:
                status = ValidationStatus.WARNING
                message = f"Performance degraded. Exported model: {exported_time:.2f}ms, Original: {original_time:.2f}ms"
            
            return ValidationCheck(
                name="inference_performance",
                status=status,
                message=message,
                expected_value=original_time,
                actual_value=exported_time,
                tolerance=self.performance_tolerance
            )
            
        except Exception as e:
            return ValidationCheck(
                name="inference_performance",
                status=ValidationStatus.FAILED,
                message=f"Error checking inference performance: {e}"
            )
    
    def _check_output_shapes(
        self,
        original_model: BaseModel,
        loaded_model: Any,
        sample_input: Any,
        export_format: str
    ) -> ValidationCheck:
        """Check output shape consistency."""
        try:
            # Get output from original model
            original_output = original_model.predict(sample_input)
            if hasattr(original_output, 'predictions'):
                original_output = original_output.predictions
            
            # Get output from exported model
            exported_output = self._predict_with_exported_model(
                loaded_model, sample_input, export_format
            )
            
            # Compare shapes
            if original_output.shape == exported_output.shape:
                return ValidationCheck(
                    name="output_shapes",
                    status=ValidationStatus.PASSED,
                    message=f"Output shapes match: {original_output.shape}",
                    expected_value=original_output.shape,
                    actual_value=exported_output.shape
                )
            else:
                return ValidationCheck(
                    name="output_shapes",
                    status=ValidationStatus.FAILED,
                    message=f"Output shapes don't match. Original: {original_output.shape}, Exported: {exported_output.shape}",
                    expected_value=original_output.shape,
                    actual_value=exported_output.shape
                )
                
        except Exception as e:
            return ValidationCheck(
                name="output_shapes",
                status=ValidationStatus.FAILED,
                message=f"Error checking output shapes: {e}"
            )
    
    def _load_exported_model(self, model_path: Union[str, Path], export_format: str) -> Any:
        """Load exported model based on format."""
        if export_format.lower() == "onnx":
            import onnxruntime as ort
            return ort.InferenceSession(str(model_path))
        
        elif export_format.lower() == "tflite":
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            return interpreter
        
        elif export_format.lower() in ["pytorch", "torchscript"]:
            import torch
            return torch.jit.load(str(model_path)) if export_format.lower() == "torchscript" else torch.load(str(model_path))
        
        else:
            raise ValueError(f"Unsupported export format for loading: {export_format}")
    
    def _predict_with_exported_model(self, loaded_model: Any, input_data: Any, export_format: str) -> np.ndarray:
        """Make predictions with exported model."""
        if export_format.lower() == "onnx":
            input_name = loaded_model.get_inputs()[0].name
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
            return loaded_model.run(None, {input_name: input_data})[0]
        
        elif export_format.lower() == "tflite":
            input_details = loaded_model.get_input_details()
            output_details = loaded_model.get_output_details()
            
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
            
            loaded_model.set_tensor(input_details[0]['index'], input_data)
            loaded_model.invoke()
            return loaded_model.get_tensor(output_details[0]['index'])
        
        elif export_format.lower() in ["pytorch", "torchscript"]:
            import torch
            
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            
            with torch.no_grad():
                output = loaded_model(input_data)
                if isinstance(output, torch.Tensor):
                    return output.numpy()
                return output
        
        else:
            raise ValueError(f"Unsupported export format for prediction: {export_format}")
    
    def _compare_predictions(self, pred1: np.ndarray, pred2: np.ndarray) -> bool:
        """Compare two prediction arrays within tolerance."""
        try:
            return np.allclose(pred1, pred2, rtol=self.accuracy_tolerance, atol=self.accuracy_tolerance)
        except Exception:
            return False
    
    def _measure_inference_time(self, predict_fn: callable, sample_input: Any, num_runs: int = 100) -> Optional[float]:
        """Measure inference time for a prediction function."""
        try:
            import time
            
            # Warm up
            for _ in range(10):
                predict_fn(sample_input)
            
            # Measure
            start_time = time.time()
            for _ in range(num_runs):
                predict_fn(sample_input)
            end_time = time.time()
            
            return ((end_time - start_time) / num_runs) * 1000  # Convert to milliseconds
            
        except Exception as e:
            logger.warning(f"Could not measure inference time: {e}")
            return None