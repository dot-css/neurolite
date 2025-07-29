"""
Data preprocessing pipelines for NeuroLite.

Provides domain-specific preprocessing pipelines for different data types
including automatic data cleaning, normalization, and transformation utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..core import get_logger, DataError, safe_import
from .detector import DataType
from .loader import Dataset, DatasetInfo


logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations."""
    # Image preprocessing
    image_size: Optional[Tuple[int, int]] = None
    normalize_images: bool = True
    image_augmentation: bool = False
    
    # Text preprocessing
    max_text_length: Optional[int] = None
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    tokenize: bool = True
    
    # Tabular preprocessing
    handle_missing: str = "auto"  # "auto", "drop", "impute", "fill"
    normalize_features: bool = True
    encode_categorical: bool = True
    remove_outliers: bool = False
    
    # General
    random_seed: int = 42


class BasePreprocessor(ABC):
    """Base class for data preprocessors."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.is_fitted = False
        self._preprocessing_stats = {}
    
    @abstractmethod
    def fit(self, dataset: Dataset) -> 'BasePreprocessor':
        """
        Fit preprocessor to dataset.
        
        Args:
            dataset: Dataset to fit on
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform dataset using fitted preprocessor.
        
        Args:
            dataset: Dataset to transform
            
        Returns:
            Transformed dataset
        """
        pass
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit preprocessor and transform dataset.
        
        Args:
            dataset: Dataset to fit and transform
            
        Returns:
            Transformed dataset
        """
        return self.fit(dataset).transform(dataset)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self._preprocessing_stats.copy()


class ImagePreprocessor(BasePreprocessor):
    """Preprocessor for image data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        super().__init__(config)
        self._mean = None
        self._std = None
    
    def fit(self, dataset: Dataset) -> 'ImagePreprocessor':
        """Fit image preprocessor to dataset."""
        if dataset.info.data_type != DataType.IMAGE:
            raise DataError(f"ImagePreprocessor expects image data, got {dataset.info.data_type.value}")
        
        logger.info("Fitting image preprocessor...")
        
        # Calculate normalization statistics if needed
        if self.config.normalize_images:
            self._calculate_normalization_stats(dataset)
        
        self.is_fitted = True
        logger.info("Image preprocessor fitted successfully")
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """Transform image dataset."""
        if not self.is_fitted:
            raise DataError("Preprocessor must be fitted before transform")
        
        logger.info(f"Transforming {len(dataset)} images...")
        
        transformed_data = []
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            # Apply transformations
            transformed_sample = self._transform_image(sample)
            transformed_data.append((transformed_sample, target))
        
        # Create new dataset with transformed data
        data = [item[0] for item in transformed_data]
        targets = [item[1] for item in transformed_data] if transformed_data[0][1] is not None else None
        
        # Update dataset info
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(data),
            shape=data[0].shape if data else None,
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'preprocessed': True}
        )
        
        return Dataset(data, targets=targets, info=new_info)
    
    def _calculate_normalization_stats(self, dataset: Dataset):
        """Calculate mean and std for normalization."""
        logger.debug("Calculating normalization statistics...")
        
        # Sample a subset for efficiency
        sample_size = min(1000, len(dataset))
        samples = []
        
        for i in range(0, len(dataset), max(1, len(dataset) // sample_size)):
            sample, _ = dataset[i]
            if isinstance(sample, np.ndarray):
                samples.append(sample)
        
        if samples:
            # Stack samples and calculate statistics
            stacked = np.stack(samples)
            self._mean = np.mean(stacked, axis=(0, 1, 2))
            self._std = np.std(stacked, axis=(0, 1, 2))
            
            # Avoid division by zero
            self._std = np.maximum(self._std, 1e-8)
            
            self._preprocessing_stats['normalization_mean'] = self._mean.tolist()
            self._preprocessing_stats['normalization_std'] = self._std.tolist()
    
    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """Transform a single image."""
        transformed = image.copy()
        
        # Resize if specified
        if self.config.image_size:
            transformed = self._resize_image(transformed, self.config.image_size)
        
        # Normalize
        if self.config.normalize_images and self._mean is not None:
            transformed = (transformed - self._mean) / self._std
        elif self.config.normalize_images:
            # Simple normalization to [0, 1]
            if transformed.dtype == np.uint8:
                transformed = transformed.astype(np.float32) / 255.0
        
        return transformed
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        try:
            PIL = safe_import('PIL', 'image processing')
            Image = PIL.Image
            
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                # Convert to uint8 if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(resized)
            
        except Exception as e:
            logger.warning(f"Failed to resize image: {e}")
            return image


class TextPreprocessor(BasePreprocessor):
    """Preprocessor for text data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        super().__init__(config)
        self._vocab = None
        self._tokenizer = None
    
    def fit(self, dataset: Dataset) -> 'TextPreprocessor':
        """Fit text preprocessor to dataset."""
        if dataset.info.data_type != DataType.TEXT:
            raise DataError(f"TextPreprocessor expects text data, got {dataset.info.data_type.value}")
        
        logger.info("Fitting text preprocessor...")
        
        # Build vocabulary if tokenizing
        if self.config.tokenize:
            self._build_vocabulary(dataset)
        
        self.is_fitted = True
        logger.info("Text preprocessor fitted successfully")
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """Transform text dataset."""
        if not self.is_fitted:
            raise DataError("Preprocessor must be fitted before transform")
        
        logger.info(f"Transforming {len(dataset)} text samples...")
        
        transformed_data = []
        for i in range(len(dataset)):
            sample, target = dataset[i]
            
            # Apply transformations
            transformed_sample = self._transform_text(sample)
            transformed_data.append((transformed_sample, target))
        
        # Create new dataset
        data = [item[0] for item in transformed_data]
        targets = [item[1] for item in transformed_data] if transformed_data[0][1] is not None else None
        
        # Update dataset info
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(data),
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'preprocessed': True, 'vocab_size': len(self._vocab) if self._vocab else None}
        )
        
        return Dataset(data, targets=targets, info=new_info)
    
    def _build_vocabulary(self, dataset: Dataset):
        """Build vocabulary from dataset."""
        logger.debug("Building vocabulary...")
        
        vocab = set()
        for i in range(len(dataset)):
            sample, _ = dataset[i]
            if isinstance(sample, str):
                tokens = self._tokenize_text(sample)
                vocab.update(tokens)
        
        # Convert to sorted list for consistency
        self._vocab = sorted(list(vocab))
        self._preprocessing_stats['vocab_size'] = len(self._vocab)
        
        logger.debug(f"Built vocabulary with {len(self._vocab)} tokens")
    
    def _transform_text(self, text: str) -> Union[str, List[str], List[int]]:
        """Transform a single text sample."""
        if not isinstance(text, str):
            return text
        
        transformed = text
        
        # Lowercase
        if self.config.lowercase:
            transformed = transformed.lower()
        
        # Remove punctuation
        if self.config.remove_punctuation:
            transformed = self._remove_punctuation(transformed)
        
        # Remove stopwords
        if self.config.remove_stopwords:
            transformed = self._remove_stopwords(transformed)
        
        # Truncate if max length specified
        if self.config.max_text_length:
            transformed = transformed[:self.config.max_text_length]
        
        # Tokenize
        if self.config.tokenize:
            tokens = self._tokenize_text(transformed)
            if self._vocab:
                # Convert to indices
                vocab_dict = {token: i for i, token in enumerate(self._vocab)}
                return [vocab_dict.get(token, 0) for token in tokens]  # 0 for unknown tokens
            return tokens
        
        return transformed
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        import re
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        import string
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove common stopwords."""
        # Simple English stopwords list
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return ' '.join(filtered_words)


class TabularPreprocessor(BasePreprocessor):
    """Preprocessor for tabular data."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        super().__init__(config)
        self._feature_stats = {}
        self._categorical_encoders = {}
        self._scaler = None
    
    def fit(self, dataset: Dataset) -> 'TabularPreprocessor':
        """Fit tabular preprocessor to dataset."""
        if dataset.info.data_type != DataType.TABULAR:
            raise DataError(f"TabularPreprocessor expects tabular data, got {dataset.info.data_type.value}")
        
        logger.info("Fitting tabular preprocessor...")
        
        # Get data as numpy array
        data_array = self._get_data_array(dataset)
        
        # Calculate feature statistics
        self._calculate_feature_stats(data_array)
        
        # Fit categorical encoders if needed
        if self.config.encode_categorical:
            self._fit_categorical_encoders(data_array)
        
        # Fit scaler if needed
        if self.config.normalize_features:
            self._fit_scaler(data_array)
        
        self.is_fitted = True
        logger.info("Tabular preprocessor fitted successfully")
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """Transform tabular dataset."""
        if not self.is_fitted:
            raise DataError("Preprocessor must be fitted before transform")
        
        logger.info(f"Transforming tabular dataset with {len(dataset)} samples...")
        
        # Get data as numpy array
        data_array = self._get_data_array(dataset)
        
        # Apply transformations
        transformed_data = self._transform_tabular_data(data_array)
        
        # Create new dataset
        targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
        
        targets = targets if targets[0] is not None else None
        
        # Update dataset info
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(transformed_data),
            shape=transformed_data.shape,
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'preprocessed': True}
        )
        
        return Dataset(transformed_data, targets=targets, info=new_info)
    
    def _get_data_array(self, dataset: Dataset) -> np.ndarray:
        """Extract data as numpy array."""
        data_list = []
        for i in range(len(dataset)):
            sample, _ = dataset[i]
            data_list.append(sample)
        
        return np.array(data_list)
    
    def _calculate_feature_stats(self, data: np.ndarray):
        """Calculate statistics for each feature."""
        logger.debug("Calculating feature statistics...")
        
        self._feature_stats = {}
        
        for col in range(data.shape[1]):
            column_data = data[:, col]
            
            # Handle different data types
            if np.issubdtype(column_data.dtype, np.number):
                # Numeric feature
                self._feature_stats[col] = {
                    'type': 'numeric',
                    'mean': np.nanmean(column_data),
                    'std': np.nanstd(column_data),
                    'min': np.nanmin(column_data),
                    'max': np.nanmax(column_data),
                    'missing_count': np.sum(np.isnan(column_data))
                }
            else:
                # Categorical feature
                unique_values = np.unique(column_data)
                self._feature_stats[col] = {
                    'type': 'categorical',
                    'unique_values': unique_values.tolist(),
                    'unique_count': len(unique_values),
                    'missing_count': np.sum(pd.isna(column_data)) if 'pd' in globals() else 0
                }
        
        self._preprocessing_stats['feature_stats'] = self._feature_stats
    
    def _fit_categorical_encoders(self, data: np.ndarray):
        """Fit categorical encoders."""
        logger.debug("Fitting categorical encoders...")
        
        for col, stats in self._feature_stats.items():
            if stats['type'] == 'categorical':
                # Simple label encoding
                unique_values = stats['unique_values']
                self._categorical_encoders[col] = {val: i for i, val in enumerate(unique_values)}
    
    def _fit_scaler(self, data: np.ndarray):
        """Fit feature scaler."""
        logger.debug("Fitting feature scaler...")
        
        # Simple standardization
        numeric_cols = [col for col, stats in self._feature_stats.items() if stats['type'] == 'numeric']
        
        if numeric_cols:
            numeric_data = data[:, numeric_cols]
            self._scaler = {
                'mean': np.nanmean(numeric_data, axis=0),
                'std': np.nanstd(numeric_data, axis=0),
                'numeric_cols': numeric_cols
            }
            
            # Avoid division by zero
            self._scaler['std'] = np.maximum(self._scaler['std'], 1e-8)
    
    def _transform_tabular_data(self, data: np.ndarray) -> np.ndarray:
        """Transform tabular data."""
        transformed = data.copy()
        
        # Handle missing values
        if self.config.handle_missing != "drop":
            transformed = self._handle_missing_values(transformed)
        
        # Encode categorical features
        if self.config.encode_categorical and self._categorical_encoders:
            transformed = self._encode_categorical_features(transformed)
        
        # Normalize features
        if self.config.normalize_features and self._scaler:
            transformed = self._normalize_features(transformed)
        
        # Remove outliers if specified
        if self.config.remove_outliers:
            transformed = self._remove_outliers(transformed)
        
        return transformed
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in data."""
        if self.config.handle_missing == "impute":
            # Simple imputation
            for col, stats in self._feature_stats.items():
                if stats['missing_count'] > 0:
                    if stats['type'] == 'numeric':
                        # Fill with mean
                        mask = np.isnan(data[:, col])
                        data[mask, col] = stats['mean']
                    else:
                        # Fill with most common value
                        # This is a simplified approach
                        pass
        
        elif self.config.handle_missing == "fill":
            # Fill with default values
            for col, stats in self._feature_stats.items():
                if stats['missing_count'] > 0:
                    if stats['type'] == 'numeric':
                        mask = np.isnan(data[:, col])
                        data[mask, col] = 0
        
        return data
    
    def _encode_categorical_features(self, data: np.ndarray) -> np.ndarray:
        """Encode categorical features."""
        for col, encoder in self._categorical_encoders.items():
            for i in range(len(data)):
                value = data[i, col]
                data[i, col] = encoder.get(value, -1)  # -1 for unknown values
        
        return data
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize numeric features."""
        if self._scaler:
            numeric_cols = self._scaler['numeric_cols']
            mean = self._scaler['mean']
            std = self._scaler['std']
            
            data[:, numeric_cols] = (data[:, numeric_cols] - mean) / std
        
        return data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated outlier detection
        for col, stats in self._feature_stats.items():
            if stats['type'] == 'numeric':
                column_data = data[:, col]
                q1, q3 = np.nanpercentile(column_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Clip outliers
                data[:, col] = np.clip(column_data, lower_bound, upper_bound)
        
        return data


class PreprocessorFactory:
    """Factory for creating appropriate preprocessors."""
    
    @staticmethod
    def create_preprocessor(
        data_type: DataType,
        config: Optional[PreprocessingConfig] = None
    ) -> BasePreprocessor:
        """
        Create appropriate preprocessor for data type.
        
        Args:
            data_type: Type of data to preprocess
            config: Preprocessing configuration
            
        Returns:
            Appropriate preprocessor instance
            
        Raises:
            DataError: If data type is not supported
        """
        preprocessors = {
            DataType.IMAGE: ImagePreprocessor,
            DataType.TEXT: TextPreprocessor,
            DataType.TABULAR: TabularPreprocessor,
        }
        
        if data_type not in preprocessors:
            supported_types = list(preprocessors.keys())
            raise DataError(f"No preprocessor available for data type: {data_type.value}. Supported types: {[t.value for t in supported_types]}")
        
        return preprocessors[data_type](config)


def preprocess_data(
    dataset: Dataset,
    config: Optional[PreprocessingConfig] = None,
    fit: bool = True
) -> Dataset:
    """
    Preprocess dataset using appropriate preprocessor.
    
    Args:
        dataset: Dataset to preprocess
        config: Preprocessing configuration
        fit: Whether to fit the preprocessor (True for training data)
        
    Returns:
        Preprocessed dataset
    """
    preprocessor = PreprocessorFactory.create_preprocessor(dataset.info.data_type, config)
    
    if fit:
        return preprocessor.fit_transform(dataset)
    else:
        # For inference, you would need a pre-fitted preprocessor
        # This is a simplified version
        logger.warning("Preprocessing without fitting - using default transformations")
        return preprocessor.fit_transform(dataset)