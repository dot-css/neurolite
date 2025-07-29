"""
Data validation utilities for NeuroLite.

Provides comprehensive data quality checks and validation for different data types
including missing values, format consistency, and common data issues.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..core import get_logger, DataValidationError, safe_import
from .detector import DataType
from .loader import Dataset


logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'missing_data', 'format', 'consistency', etc.
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


@dataclass
class ValidationResult:
    """Results of data validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == 'error']
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == 'warning']
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return len(self.get_errors()) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return len(self.get_warnings()) > 0


class DataValidator:
    """
    Comprehensive data validation system.
    
    Performs data quality checks including missing values, format consistency,
    outlier detection, and data type specific validations.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self._validators = {
            DataType.IMAGE: self._validate_image_data,
            DataType.TEXT: self._validate_text_data,
            DataType.TABULAR: self._validate_tabular_data,
            DataType.AUDIO: self._validate_audio_data,
            DataType.VIDEO: self._validate_video_data,
        }
    
    def validate(self, dataset: Dataset) -> ValidationResult:
        """
        Validate a dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating {dataset.info.data_type.value} dataset with {len(dataset)} samples")
        
        issues = []
        summary = {
            'data_type': dataset.info.data_type.value,
            'num_samples': len(dataset),
            'validation_timestamp': None
        }
        
        # Basic dataset validation
        issues.extend(self._validate_basic_structure(dataset))
        
        # Data type specific validation
        if dataset.info.data_type in self._validators:
            try:
                type_issues = self._validators[dataset.info.data_type](dataset)
                issues.extend(type_issues)
            except Exception as e:
                logger.warning(f"Type-specific validation failed: {e}")
                issues.append(ValidationIssue(
                    severity='warning',
                    category='validation_error',
                    message=f"Type-specific validation failed: {e}",
                    suggestions=['Check data format and try again']
                ))
        
        # Determine overall validity
        is_valid = not any(issue.severity == 'error' for issue in issues)
        
        # Update summary
        summary.update({
            'is_valid': is_valid,
            'num_errors': len([i for i in issues if i.severity == 'error']),
            'num_warnings': len([i for i in issues if i.severity == 'warning']),
            'num_info': len([i for i in issues if i.severity == 'info']),
        })
        
        result = ValidationResult(is_valid=is_valid, issues=issues, summary=summary)
        
        logger.info(f"Validation complete: {len(result.get_errors())} errors, {len(result.get_warnings())} warnings")
        
        return result
    
    def _validate_basic_structure(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate basic dataset structure."""
        issues = []
        
        # Check if dataset is empty
        if len(dataset) == 0:
            issues.append(ValidationIssue(
                severity='error',
                category='structure',
                message='Dataset is empty',
                suggestions=['Check data path and ensure files exist']
            ))
            return issues
        
        # Check for consistent data structure
        try:
            first_sample, first_target = dataset[0]
            sample_type = type(first_sample)
            
            # Check a few samples for consistency
            check_count = min(10, len(dataset))
            inconsistent_samples = []
            
            for i in range(1, check_count):
                sample, target = dataset[i]
                if type(sample) != sample_type:
                    inconsistent_samples.append(i)
            
            if inconsistent_samples:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Inconsistent sample types found in samples: {inconsistent_samples}',
                    details={'expected_type': str(sample_type), 'inconsistent_indices': inconsistent_samples},
                    suggestions=['Check data loading process', 'Ensure all files are of the same type']
                ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='structure',
                message=f'Failed to access dataset samples: {e}',
                suggestions=['Check dataset implementation', 'Verify data loading was successful']
            ))
        
        return issues
    
    def _validate_image_data(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate image dataset."""
        issues = []
        
        # Check sample of images
        check_count = min(20, len(dataset))
        shapes = []
        invalid_images = []
        
        for i in range(check_count):
            try:
                sample, _ = dataset[i]
                
                if isinstance(sample, np.ndarray):
                    shape = sample.shape
                    shapes.append(shape)
                    
                    # Check if image has valid dimensions
                    if len(shape) < 2 or len(shape) > 4:
                        invalid_images.append(i)
                    
                    # Check for reasonable image size
                    if len(shape) >= 2:
                        height, width = shape[:2]
                        if height < 1 or width < 1 or height > 10000 or width > 10000:
                            issues.append(ValidationIssue(
                                severity='warning',
                                category='format',
                                message=f'Image {i} has unusual dimensions: {height}x{width}',
                                suggestions=['Check if image dimensions are correct']
                            ))
                    
                    # Check for valid pixel values
                    if sample.dtype in [np.uint8]:
                        if sample.min() < 0 or sample.max() > 255:
                            issues.append(ValidationIssue(
                                severity='warning',
                                category='format',
                                message=f'Image {i} has pixel values outside valid range [0, 255]',
                                suggestions=['Check image preprocessing', 'Ensure proper normalization']
                            ))
                    elif sample.dtype in [np.float32, np.float64]:
                        if sample.min() < 0 or sample.max() > 1:
                            issues.append(ValidationIssue(
                                severity='info',
                                category='format',
                                message=f'Image {i} has float values outside [0, 1] range',
                                details={'min_value': float(sample.min()), 'max_value': float(sample.max())},
                                suggestions=['Consider normalizing to [0, 1] range if needed']
                            ))
                
                else:
                    invalid_images.append(i)
                    
            except Exception as e:
                invalid_images.append(i)
                logger.debug(f"Failed to validate image {i}: {e}")
        
        if invalid_images:
            issues.append(ValidationIssue(
                severity='error',
                category='format',
                message=f'Invalid image data found in samples: {invalid_images[:5]}{"..." if len(invalid_images) > 5 else ""}',
                details={'invalid_count': len(invalid_images), 'total_checked': check_count},
                suggestions=['Check image loading process', 'Verify image files are not corrupted']
            ))
        
        # Check shape consistency
        if shapes:
            unique_shapes = list(set(shapes))
            if len(unique_shapes) > 1:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Images have inconsistent shapes: {unique_shapes[:3]}{"..." if len(unique_shapes) > 3 else ""}',
                    details={'unique_shapes': unique_shapes},
                    suggestions=['Consider resizing images to consistent dimensions', 'Use image preprocessing']
                ))
        
        return issues
    
    def _validate_text_data(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate text dataset."""
        issues = []
        
        # Check sample of texts
        check_count = min(50, len(dataset))
        text_lengths = []
        empty_texts = []
        non_string_samples = []
        
        for i in range(check_count):
            try:
                sample, _ = dataset[i]
                
                if isinstance(sample, str):
                    text_lengths.append(len(sample))
                    
                    if len(sample.strip()) == 0:
                        empty_texts.append(i)
                    
                    # Check for very long texts that might cause issues
                    if len(sample) > 100000:  # 100k characters
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='format',
                            message=f'Text sample {i} is very long ({len(sample)} characters)',
                            suggestions=['Consider text truncation or chunking for processing']
                        ))
                
                else:
                    non_string_samples.append(i)
                    
            except Exception as e:
                logger.debug(f"Failed to validate text sample {i}: {e}")
        
        if empty_texts:
            issues.append(ValidationIssue(
                severity='warning',
                category='missing_data',
                message=f'Found {len(empty_texts)} empty text samples',
                details={'empty_indices': empty_texts[:10]},
                suggestions=['Remove empty samples', 'Check data preprocessing']
            ))
        
        if non_string_samples:
            issues.append(ValidationIssue(
                severity='error',
                category='format',
                message=f'Found {len(non_string_samples)} non-string samples in text dataset',
                details={'non_string_indices': non_string_samples[:10]},
                suggestions=['Ensure all samples are strings', 'Check text loading process']
            ))
        
        # Analyze text length distribution
        if text_lengths:
            mean_length = np.mean(text_lengths)
            std_length = np.std(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)
            
            issues.append(ValidationIssue(
                severity='info',
                category='statistics',
                message=f'Text length statistics: mean={mean_length:.1f}, std={std_length:.1f}, range=[{min_length}, {max_length}]',
                details={
                    'mean_length': mean_length,
                    'std_length': std_length,
                    'min_length': min_length,
                    'max_length': max_length
                }
            ))
            
            # Check for very short texts
            very_short = sum(1 for length in text_lengths if length < 10)
            if very_short > len(text_lengths) * 0.1:  # More than 10% very short
                issues.append(ValidationIssue(
                    severity='warning',
                    category='format',
                    message=f'{very_short} texts are very short (< 10 characters)',
                    suggestions=['Check if short texts are meaningful', 'Consider filtering or preprocessing']
                ))
        
        return issues
    
    def _validate_tabular_data(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate tabular dataset."""
        issues = []
        
        try:
            # Get a sample of data to analyze
            sample_size = min(1000, len(dataset))
            samples = []
            targets = []
            
            for i in range(sample_size):
                sample, target = dataset[i]
                samples.append(sample)
                if target is not None:
                    targets.append(target)
            
            if not samples:
                return issues
            
            # Convert to numpy array for analysis
            data_array = np.array(samples)
            
            # Check for missing values (NaN, None, inf)
            if data_array.dtype in [np.float32, np.float64]:
                nan_mask = np.isnan(data_array)
                inf_mask = np.isinf(data_array)
                
                if np.any(nan_mask):
                    nan_count = np.sum(nan_mask)
                    nan_percentage = (nan_count / data_array.size) * 100
                    
                    issues.append(ValidationIssue(
                        severity='warning' if nan_percentage < 10 else 'error',
                        category='missing_data',
                        message=f'Found {nan_count} NaN values ({nan_percentage:.1f}% of data)',
                        details={'nan_count': int(nan_count), 'percentage': nan_percentage},
                        suggestions=['Handle missing values with imputation or removal', 'Check data source quality']
                    ))
                
                if np.any(inf_mask):
                    inf_count = np.sum(inf_mask)
                    issues.append(ValidationIssue(
                        severity='error',
                        category='format',
                        message=f'Found {inf_count} infinite values',
                        details={'inf_count': int(inf_count)},
                        suggestions=['Replace infinite values', 'Check for division by zero in preprocessing']
                    ))
            
            # Check for constant features
            if len(data_array.shape) == 2:
                constant_features = []
                for col in range(data_array.shape[1]):
                    column_data = data_array[:, col]
                    if np.all(column_data == column_data[0]):
                        constant_features.append(col)
                
                if constant_features:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='feature_quality',
                        message=f'Found {len(constant_features)} constant features',
                        details={'constant_features': constant_features[:10]},
                        suggestions=['Remove constant features', 'Check feature engineering process']
                    ))
            
            # Check data types and ranges
            if data_array.dtype in [np.float32, np.float64]:
                # Check for outliers using IQR method
                if len(data_array.shape) == 2:
                    outlier_features = []
                    for col in range(data_array.shape[1]):
                        column_data = data_array[:, col]
                        if not np.all(np.isnan(column_data)):
                            q1, q3 = np.nanpercentile(column_data, [25, 75])
                            iqr = q3 - q1
                            if iqr > 0:
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outliers = np.sum((column_data < lower_bound) | (column_data > upper_bound))
                                outlier_percentage = (outliers / len(column_data)) * 100
                                
                                if outlier_percentage > 5:  # More than 5% outliers
                                    outlier_features.append((col, outlier_percentage))
                    
                    if outlier_features:
                        issues.append(ValidationIssue(
                            severity='info',
                            category='data_quality',
                            message=f'Found potential outliers in {len(outlier_features)} features',
                            details={'outlier_features': outlier_features[:5]},
                            suggestions=['Consider outlier detection and treatment', 'Verify if outliers are valid']
                        ))
            
            # Validate targets if present
            if targets:
                target_array = np.array(targets)
                
                # Check for missing targets
                if target_array.dtype == object:
                    none_targets = sum(1 for t in targets if t is None)
                    if none_targets > 0:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='missing_data',
                            message=f'Found {none_targets} missing target values',
                            suggestions=['Remove samples with missing targets', 'Impute missing targets']
                        ))
                
                # Check target distribution for classification
                unique_targets = np.unique(target_array)
                if len(unique_targets) < len(target_array) * 0.5:  # Likely classification
                    target_counts = {str(t): np.sum(target_array == t) for t in unique_targets}
                    min_count = min(target_counts.values())
                    max_count = max(target_counts.values())
                    
                    if max_count / min_count > 10:  # Highly imbalanced
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='data_quality',
                            message='Target classes are highly imbalanced',
                            details={'class_counts': target_counts},
                            suggestions=['Consider class balancing techniques', 'Use stratified sampling']
                        ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity='warning',
                category='validation_error',
                message=f'Tabular validation failed: {e}',
                suggestions=['Check data format and structure']
            ))
        
        return issues
    
    def _validate_audio_data(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate audio dataset."""
        issues = []
        
        # Check sample of audio data
        check_count = min(10, len(dataset))
        durations = []
        invalid_audio = []
        
        for i in range(check_count):
            try:
                sample, _ = dataset[i]
                
                if isinstance(sample, np.ndarray):
                    # Check audio properties
                    if len(sample.shape) > 2:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='format',
                            message=f'Audio sample {i} has unexpected dimensions: {sample.shape}',
                            suggestions=['Check audio loading process']
                        ))
                    
                    # Estimate duration (assuming sample rate info is available)
                    if hasattr(dataset.info, 'metadata') and dataset.info.metadata:
                        sample_rates = dataset.info.metadata.get('sample_rates', [])
                        if i < len(sample_rates):
                            duration = len(sample) / sample_rates[i]
                            durations.append(duration)
                            
                            if duration < 0.1:  # Very short audio
                                issues.append(ValidationIssue(
                                    severity='warning',
                                    category='format',
                                    message=f'Audio sample {i} is very short ({duration:.2f}s)',
                                    suggestions=['Check if short audio clips are meaningful']
                                ))
                            elif duration > 300:  # Very long audio (5 minutes)
                                issues.append(ValidationIssue(
                                    severity='info',
                                    category='format',
                                    message=f'Audio sample {i} is very long ({duration:.1f}s)',
                                    suggestions=['Consider audio segmentation for processing']
                                ))
                
                else:
                    invalid_audio.append(i)
                    
            except Exception as e:
                invalid_audio.append(i)
                logger.debug(f"Failed to validate audio sample {i}: {e}")
        
        if invalid_audio:
            issues.append(ValidationIssue(
                severity='error',
                category='format',
                message=f'Invalid audio data found in samples: {invalid_audio}',
                suggestions=['Check audio loading process', 'Verify audio files are not corrupted']
            ))
        
        if durations:
            mean_duration = np.mean(durations)
            issues.append(ValidationIssue(
                severity='info',
                category='statistics',
                message=f'Audio duration statistics: mean={mean_duration:.2f}s, range=[{min(durations):.2f}, {max(durations):.2f}]s',
                details={'mean_duration': mean_duration, 'min_duration': min(durations), 'max_duration': max(durations)}
            ))
        
        return issues
    
    def _validate_video_data(self, dataset: Dataset) -> List[ValidationIssue]:
        """Validate video dataset."""
        issues = []
        
        # Video validation is complex and depends on the specific implementation
        # For now, just check basic structure
        issues.append(ValidationIssue(
            severity='info',
            category='implementation',
            message='Video validation is not fully implemented',
            suggestions=['Video processing requires specialized libraries and validation']
        ))
        
        return issues


# Global validator instance
_validator = DataValidator()


def validate_data(dataset: Dataset) -> ValidationResult:
    """
    Validate a dataset for common data quality issues.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Validation results with issues and suggestions
    """
    return _validator.validate(dataset)