"""
Automatic data cleaning utilities for NeuroLite.

Provides comprehensive data cleaning functionality including missing value handling,
outlier detection and removal, duplicate detection, and data consistency checks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..core import get_logger, DataError, safe_import
from .detector import DataType
from .loader import Dataset, DatasetInfo


logger = get_logger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    # Missing value handling
    missing_threshold: float = 0.5  # Drop features/samples with >50% missing
    imputation_strategy: str = "auto"  # "auto", "mean", "median", "mode", "drop"
    
    # Outlier handling
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest", "none"
    outlier_threshold: float = 1.5  # IQR multiplier or z-score threshold
    outlier_action: str = "clip"  # "clip", "remove", "flag"
    
    # Duplicate handling
    remove_duplicates: bool = True
    duplicate_threshold: float = 0.95  # Similarity threshold for near-duplicates
    
    # Data consistency
    fix_encoding: bool = True
    normalize_whitespace: bool = True
    fix_data_types: bool = True
    
    # Quality thresholds
    min_samples: int = 10  # Minimum samples required
    min_features: int = 1  # Minimum features required
    
    # General
    random_seed: int = 42


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    original_samples: int
    final_samples: int
    original_features: int
    final_features: int
    
    missing_values_handled: int
    outliers_handled: int
    duplicates_removed: int
    
    operations_performed: List[str]
    warnings: List[str]
    
    def samples_removed(self) -> int:
        """Number of samples removed."""
        return self.original_samples - self.final_samples
    
    def features_removed(self) -> int:
        """Number of features removed."""
        return self.original_features - self.final_features
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Data Cleaning Summary:\n"
            f"  Samples: {self.original_samples} → {self.final_samples} "
            f"({self.samples_removed()} removed)\n"
            f"  Features: {self.original_features} → {self.final_features} "
            f"({self.features_removed()} removed)\n"
            f"  Missing values handled: {self.missing_values_handled}\n"
            f"  Outliers handled: {self.outliers_handled}\n"
            f"  Duplicates removed: {self.duplicates_removed}\n"
            f"  Operations: {', '.join(self.operations_performed)}"
        )


class DataCleaner:
    """
    Comprehensive data cleaning system.
    
    Automatically detects and fixes common data quality issues including
    missing values, outliers, duplicates, and inconsistencies.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize data cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        self._cleaning_stats = {}
    
    def clean(self, dataset: Dataset) -> Tuple[Dataset, CleaningReport]:
        """
        Clean dataset automatically.
        
        Args:
            dataset: Dataset to clean
            
        Returns:
            Tuple of (cleaned_dataset, cleaning_report)
        """
        logger.info(f"Starting data cleaning for {dataset.info.data_type.value} dataset")
        
        # Initialize report
        report = CleaningReport(
            original_samples=len(dataset),
            final_samples=len(dataset),
            original_features=self._get_feature_count(dataset),
            final_features=self._get_feature_count(dataset),
            missing_values_handled=0,
            outliers_handled=0,
            duplicates_removed=0,
            operations_performed=[],
            warnings=[]
        )
        
        # Apply cleaning based on data type
        cleaned_dataset = dataset
        
        if dataset.info.data_type == DataType.TABULAR:
            cleaned_dataset, report = self._clean_tabular_data(cleaned_dataset, report)
        elif dataset.info.data_type == DataType.TEXT:
            cleaned_dataset, report = self._clean_text_data(cleaned_dataset, report)
        elif dataset.info.data_type == DataType.IMAGE:
            cleaned_dataset, report = self._clean_image_data(cleaned_dataset, report)
        else:
            logger.warning(f"No specific cleaning implemented for {dataset.info.data_type.value}")
        
        # General cleaning operations
        cleaned_dataset, report = self._remove_duplicates(cleaned_dataset, report)
        
        # Final validation
        if len(cleaned_dataset) < self.config.min_samples:
            report.warnings.append(f"Dataset has only {len(cleaned_dataset)} samples (minimum: {self.config.min_samples})")
        
        logger.info(f"Data cleaning completed: {report.summary()}")
        return cleaned_dataset, report
    
    def _get_feature_count(self, dataset: Dataset) -> int:
        """Get number of features in dataset."""
        if len(dataset) == 0:
            return 0
        
        sample, _ = dataset[0]
        if isinstance(sample, np.ndarray):
            if len(sample.shape) == 1:
                return sample.shape[0]
            elif len(sample.shape) == 2:
                return sample.shape[1]
        
        return 1  # Single feature for non-array data
    
    def _clean_tabular_data(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Clean tabular dataset."""
        logger.debug("Cleaning tabular data...")
        
        # Convert to array for processing
        data_array, targets = self._dataset_to_arrays(dataset)
        
        if data_array is None:
            return dataset, report
        
        original_shape = data_array.shape
        
        # Handle missing values
        data_array, targets, missing_handled = self._handle_missing_values_tabular(data_array, targets)
        report.missing_values_handled = missing_handled
        if missing_handled > 0:
            report.operations_performed.append("missing_value_handling")
        
        # Handle outliers
        data_array, outliers_handled = self._handle_outliers_tabular(data_array)
        report.outliers_handled = outliers_handled
        if outliers_handled > 0:
            report.operations_performed.append("outlier_handling")
        
        # Update report
        report.final_samples = data_array.shape[0]
        report.final_features = data_array.shape[1] if len(data_array.shape) > 1 else 1
        
        # Convert back to dataset
        cleaned_dataset = self._arrays_to_dataset(data_array, targets, dataset)
        
        return cleaned_dataset, report
    
    def _clean_text_data(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Clean text dataset."""
        logger.debug("Cleaning text data...")
        
        cleaned_data = []
        cleaned_targets = []
        empty_removed = 0
        encoding_fixed = 0
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            if not isinstance(sample, str):
                continue
            
            # Fix encoding issues
            if self.config.fix_encoding:
                original_sample = sample
                sample = self._fix_text_encoding(sample)
                if sample != original_sample:
                    encoding_fixed += 1
            
            # Normalize whitespace
            if self.config.normalize_whitespace:
                sample = self._normalize_whitespace(sample)
            
            # Remove empty or very short texts
            if len(sample.strip()) < 3:  # Less than 3 characters
                empty_removed += 1
                continue
            
            cleaned_data.append(sample)
            cleaned_targets.append(target)
        
        # Update report
        report.final_samples = len(cleaned_data)
        if empty_removed > 0:
            report.operations_performed.append("empty_text_removal")
        if encoding_fixed > 0:
            report.operations_performed.append("encoding_fix")
        if self.config.normalize_whitespace:
            report.operations_performed.append("whitespace_normalization")
        
        # Create cleaned dataset
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(cleaned_data),
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'cleaned': True}
        )
        
        cleaned_dataset = Dataset(
            cleaned_data,
            targets=cleaned_targets if cleaned_targets[0] is not None else None,
            info=new_info
        )
        
        return cleaned_dataset, report
    
    def _clean_image_data(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Clean image dataset."""
        logger.debug("Cleaning image data...")
        
        cleaned_data = []
        cleaned_targets = []
        invalid_removed = 0
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            if not isinstance(sample, np.ndarray):
                invalid_removed += 1
                continue
            
            # Check for valid image dimensions
            if len(sample.shape) < 2 or len(sample.shape) > 4:
                invalid_removed += 1
                continue
            
            # Check for reasonable image size
            if len(sample.shape) >= 2:
                height, width = sample.shape[:2]
                if height < 1 or width < 1 or height > 10000 or width > 10000:
                    invalid_removed += 1
                    continue
            
            # Check for valid pixel values
            if sample.dtype == np.uint8:
                if sample.min() < 0 or sample.max() > 255:
                    # Clip to valid range
                    sample = np.clip(sample, 0, 255)
            elif sample.dtype in [np.float32, np.float64]:
                if sample.min() < 0 or sample.max() > 1:
                    # Normalize to [0, 1] range
                    sample = (sample - sample.min()) / (sample.max() - sample.min())
            
            cleaned_data.append(sample)
            cleaned_targets.append(target)
        
        # Update report
        report.final_samples = len(cleaned_data)
        if invalid_removed > 0:
            report.operations_performed.append("invalid_image_removal")
        
        # Create cleaned dataset
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(cleaned_data),
            shape=cleaned_data[0].shape if cleaned_data else None,
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'cleaned': True}
        )
        
        cleaned_dataset = Dataset(
            cleaned_data,
            targets=cleaned_targets if cleaned_targets[0] is not None else None,
            info=new_info
        )
        
        return cleaned_dataset, report
    
    def _dataset_to_arrays(self, dataset: Dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert dataset to numpy arrays."""
        try:
            data_list = []
            target_list = []
            
            for i in range(len(dataset)):
                sample, target = dataset[i]
                data_list.append(sample)
                target_list.append(target)
            
            data_array = np.array(data_list)
            targets = np.array(target_list) if target_list[0] is not None else None
            
            return data_array, targets
            
        except Exception as e:
            logger.warning(f"Failed to convert dataset to arrays: {e}")
            return None, None
    
    def _arrays_to_dataset(self, data_array: np.ndarray, targets: Optional[np.ndarray], original_dataset: Dataset) -> Dataset:
        """Convert arrays back to dataset."""
        # Convert to list format
        data_list = [data_array[i] for i in range(len(data_array))]
        target_list = [targets[i] for i in range(len(targets))] if targets is not None else None
        
        # Create new dataset info
        new_info = DatasetInfo(
            data_type=original_dataset.info.data_type,
            num_samples=len(data_list),
            shape=data_array.shape,
            num_classes=original_dataset.info.num_classes,
            class_names=original_dataset.info.class_names,
            feature_names=original_dataset.info.feature_names,
            target_column=original_dataset.info.target_column,
            file_paths=original_dataset.info.file_paths,
            metadata={**(original_dataset.info.metadata or {}), 'cleaned': True}
        )
        
        return Dataset(data_list, targets=target_list, info=new_info)
    
    def _handle_missing_values_tabular(self, data: np.ndarray, targets: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        """Handle missing values in tabular data."""
        if not np.issubdtype(data.dtype, np.number):
            return data, targets, 0
        
        missing_count = 0
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            missing_count = np.sum(np.isnan(data))
            
            if self.config.imputation_strategy == "drop":
                # Remove rows with any missing values
                valid_rows = ~np.any(np.isnan(data), axis=1)
                data = data[valid_rows]
                if targets is not None:
                    targets = targets[valid_rows]
            
            elif self.config.imputation_strategy in ["mean", "auto"]:
                # Impute with column means
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    if np.any(np.isnan(col_data)):
                        mean_val = np.nanmean(col_data)
                        data[np.isnan(col_data), col] = mean_val
            
            elif self.config.imputation_strategy == "median":
                # Impute with column medians
                for col in range(data.shape[1]):
                    col_data = data[:, col]
                    if np.any(np.isnan(col_data)):
                        median_val = np.nanmedian(col_data)
                        data[np.isnan(col_data), col] = median_val
        
        return data, targets, missing_count
    
    def _handle_outliers_tabular(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Handle outliers in tabular data."""
        if not np.issubdtype(data.dtype, np.number):
            return data, 0
        
        outliers_handled = 0
        
        if self.config.outlier_method == "iqr":
            for col in range(data.shape[1]):
                col_data = data[:, col]
                q1, q3 = np.percentile(col_data, [25, 75])
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - self.config.outlier_threshold * iqr
                    upper_bound = q3 + self.config.outlier_threshold * iqr
                    
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = np.sum(outlier_mask)
                    
                    if outlier_count > 0:
                        outliers_handled += outlier_count
                        
                        if self.config.outlier_action == "clip":
                            data[:, col] = np.clip(col_data, lower_bound, upper_bound)
        
        elif self.config.outlier_method == "zscore":
            for col in range(data.shape[1]):
                col_data = data[:, col]
                z_scores = np.abs((col_data - np.mean(col_data)) / np.std(col_data))
                
                outlier_mask = z_scores > self.config.outlier_threshold
                outlier_count = np.sum(outlier_mask)
                
                if outlier_count > 0:
                    outliers_handled += outlier_count
                    
                    if self.config.outlier_action == "clip":
                        # Clip to mean ± threshold * std
                        mean_val = np.mean(col_data)
                        std_val = np.std(col_data)
                        lower_bound = mean_val - self.config.outlier_threshold * std_val
                        upper_bound = mean_val + self.config.outlier_threshold * std_val
                        data[:, col] = np.clip(col_data, lower_bound, upper_bound)
        
        return data, outliers_handled
    
    def _remove_duplicates(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Remove duplicate samples from dataset."""
        if not self.config.remove_duplicates:
            return dataset, report
        
        logger.debug("Removing duplicates...")
        
        # For different data types, use different duplicate detection methods
        if dataset.info.data_type == DataType.TABULAR:
            return self._remove_duplicates_tabular(dataset, report)
        elif dataset.info.data_type == DataType.TEXT:
            return self._remove_duplicates_text(dataset, report)
        else:
            # For other types, use simple equality check
            return self._remove_duplicates_simple(dataset, report)
    
    def _remove_duplicates_tabular(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Remove duplicates from tabular dataset."""
        try:
            # Convert to pandas for efficient duplicate removal
            pd = safe_import('pandas', 'duplicate removal')
            
            # Convert dataset to DataFrame
            data_list = []
            target_list = []
            
            for i in range(len(dataset)):
                sample, target = dataset[i]
                data_list.append(sample)
                target_list.append(target)
            
            df = pd.DataFrame(data_list)
            
            # Remove duplicates
            original_len = len(df)
            df_clean = df.drop_duplicates()
            duplicates_removed = original_len - len(df_clean)
            
            if duplicates_removed > 0:
                # Get indices of kept rows
                kept_indices = df_clean.index.tolist()
                
                # Filter targets accordingly
                if target_list[0] is not None:
                    target_list = [target_list[i] for i in kept_indices]
                
                # Convert back to dataset
                data_list = df_clean.values.tolist()
                
                new_info = DatasetInfo(
                    data_type=dataset.info.data_type,
                    num_samples=len(data_list),
                    shape=(len(data_list), len(data_list[0])) if data_list else None,
                    num_classes=dataset.info.num_classes,
                    class_names=dataset.info.class_names,
                    feature_names=dataset.info.feature_names,
                    target_column=dataset.info.target_column,
                    file_paths=dataset.info.file_paths,
                    metadata={**(dataset.info.metadata or {}), 'cleaned': True}
                )
                
                cleaned_dataset = Dataset(
                    data_list,
                    targets=target_list if target_list[0] is not None else None,
                    info=new_info
                )
                
                report.duplicates_removed = duplicates_removed
                report.final_samples = len(data_list)
                report.operations_performed.append("duplicate_removal")
                
                return cleaned_dataset, report
        
        except Exception as e:
            logger.warning(f"Failed to remove duplicates from tabular data: {e}")
        
        return dataset, report
    
    def _remove_duplicates_text(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Remove duplicates from text dataset."""
        seen_texts = set()
        cleaned_data = []
        cleaned_targets = []
        duplicates_removed = 0
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            if isinstance(sample, str):
                # Normalize text for comparison
                normalized = sample.strip().lower()
                
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    cleaned_data.append(sample)
                    cleaned_targets.append(target)
                else:
                    duplicates_removed += 1
        
        if duplicates_removed > 0:
            new_info = DatasetInfo(
                data_type=dataset.info.data_type,
                num_samples=len(cleaned_data),
                num_classes=dataset.info.num_classes,
                class_names=dataset.info.class_names,
                feature_names=dataset.info.feature_names,
                target_column=dataset.info.target_column,
                file_paths=dataset.info.file_paths,
                metadata={**(dataset.info.metadata or {}), 'cleaned': True}
            )
            
            cleaned_dataset = Dataset(
                cleaned_data,
                targets=cleaned_targets if cleaned_targets[0] is not None else None,
                info=new_info
            )
            
            report.duplicates_removed = duplicates_removed
            report.final_samples = len(cleaned_data)
            report.operations_performed.append("duplicate_removal")
            
            return cleaned_dataset, report
        
        return dataset, report
    
    def _remove_duplicates_simple(self, dataset: Dataset, report: CleaningReport) -> Tuple[Dataset, CleaningReport]:
        """Simple duplicate removal using equality check."""
        seen_samples = []
        cleaned_data = []
        cleaned_targets = []
        duplicates_removed = 0
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            # Simple equality check
            is_duplicate = False
            for seen_sample in seen_samples:
                try:
                    if np.array_equal(sample, seen_sample):
                        is_duplicate = True
                        break
                except:
                    if sample == seen_sample:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_samples.append(sample)
                cleaned_data.append(sample)
                cleaned_targets.append(target)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            new_info = DatasetInfo(
                data_type=dataset.info.data_type,
                num_samples=len(cleaned_data),
                shape=cleaned_data[0].shape if cleaned_data and hasattr(cleaned_data[0], 'shape') else None,
                num_classes=dataset.info.num_classes,
                class_names=dataset.info.class_names,
                feature_names=dataset.info.feature_names,
                target_column=dataset.info.target_column,
                file_paths=dataset.info.file_paths,
                metadata={**(dataset.info.metadata or {}), 'cleaned': True}
            )
            
            cleaned_dataset = Dataset(
                cleaned_data,
                targets=cleaned_targets if cleaned_targets[0] is not None else None,
                info=new_info
            )
            
            report.duplicates_removed = duplicates_removed
            report.final_samples = len(cleaned_data)
            report.operations_performed.append("duplicate_removal")
            
            return cleaned_dataset, report
        
        return dataset, report
    
    def _fix_text_encoding(self, text: str) -> str:
        """Fix common text encoding issues."""
        # Common encoding fixes
        fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        import re
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


def clean_data(
    dataset: Dataset,
    config: Optional[CleaningConfig] = None
) -> Tuple[Dataset, CleaningReport]:
    """
    Clean dataset automatically.
    
    Args:
        dataset: Dataset to clean
        config: Cleaning configuration
        
    Returns:
        Tuple of (cleaned_dataset, cleaning_report)
    """
    cleaner = DataCleaner(config)
    return cleaner.clean(dataset)