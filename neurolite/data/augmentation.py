"""
Data augmentation modules for NeuroLite.

Provides domain-specific data augmentation techniques to improve model
generalization including image transformations, text augmentation, and
synthetic data generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import random

from ..core import get_logger, DataError, safe_import
from .detector import DataType
from .loader import Dataset, DatasetInfo


logger = get_logger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Image augmentation
    rotation_range: float = 15.0  # degrees
    width_shift_range: float = 0.1  # fraction of width
    height_shift_range: float = 0.1  # fraction of height
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    zoom_range: float = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False
    
    # Text augmentation
    synonym_replacement: bool = True
    random_insertion: bool = True
    random_swap: bool = True
    random_deletion: bool = True
    augmentation_probability: float = 0.1  # per operation
    
    # Tabular augmentation
    noise_factor: float = 0.05  # Gaussian noise std as fraction of feature std
    feature_dropout: float = 0.1  # Probability of dropping features
    
    # General
    augmentation_factor: float = 2.0  # Multiply dataset size by this factor
    random_seed: int = 42


class BaseAugmentor(ABC):
    """Base class for data augmentors."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentor.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
    
    @abstractmethod
    def augment(self, dataset: Dataset) -> Dataset:
        """
        Augment dataset.
        
        Args:
            dataset: Dataset to augment
            
        Returns:
            Augmented dataset
        """
        pass
    
    def _calculate_target_size(self, original_size: int) -> int:
        """Calculate target dataset size after augmentation."""
        return int(original_size * self.config.augmentation_factor)


class ImageAugmentor(BaseAugmentor):
    """Augmentor for image data."""
    
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment image dataset."""
        if dataset.info.data_type != DataType.IMAGE:
            raise DataError(f"ImageAugmentor expects image data, got {dataset.info.data_type.value}")
        
        logger.info(f"Augmenting image dataset from {len(dataset)} to {self._calculate_target_size(len(dataset))} samples")
        
        original_data = []
        original_targets = []
        
        # Collect original data
        for i in range(len(dataset)):
            sample, target = dataset[i]
            original_data.append(sample)
            original_targets.append(target)
        
        # Generate augmented data
        augmented_data = original_data.copy()
        augmented_targets = original_targets.copy()
        
        target_size = self._calculate_target_size(len(dataset))
        augmentations_needed = target_size - len(dataset)
        
        for _ in range(augmentations_needed):
            # Randomly select an original sample to augment
            idx = np.random.randint(0, len(original_data))
            original_sample = original_data[idx]
            original_target = original_targets[idx]
            
            # Apply random augmentations
            augmented_sample = self._augment_image(original_sample)
            
            augmented_data.append(augmented_sample)
            augmented_targets.append(original_target)
        
        # Create new dataset
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(augmented_data),
            shape=augmented_data[0].shape if augmented_data else None,
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'augmented': True}
        )
        
        return Dataset(
            augmented_data,
            targets=augmented_targets if augmented_targets[0] is not None else None,
            info=new_info
        )
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single image."""
        augmented = image.copy()
        
        # Rotation
        if self.config.rotation_range > 0:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            augmented = self._rotate_image(augmented, angle)
        
        # Translation
        if self.config.width_shift_range > 0 or self.config.height_shift_range > 0:
            augmented = self._translate_image(augmented)
        
        # Brightness adjustment
        if self.config.brightness_range != (1.0, 1.0):
            brightness_factor = np.random.uniform(*self.config.brightness_range)
            augmented = self._adjust_brightness(augmented, brightness_factor)
        
        # Zoom
        if self.config.zoom_range > 0:
            zoom_factor = np.random.uniform(1 - self.config.zoom_range, 1 + self.config.zoom_range)
            augmented = self._zoom_image(augmented, zoom_factor)
        
        # Horizontal flip
        if self.config.horizontal_flip and np.random.random() < 0.5:
            augmented = np.fliplr(augmented)
        
        # Vertical flip
        if self.config.vertical_flip and np.random.random() < 0.5:
            augmented = np.flipud(augmented)
        
        return augmented
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        try:
            from scipy.ndimage import rotate
            return rotate(image, angle, reshape=False, mode='nearest')
        except ImportError:
            logger.warning("scipy not available for image rotation")
            return image
    
    def _translate_image(self, image: np.ndarray) -> np.ndarray:
        """Translate image randomly."""
        try:
            from scipy.ndimage import shift
            
            height, width = image.shape[:2]
            
            # Calculate shift amounts
            dx = np.random.uniform(-self.config.width_shift_range, self.config.width_shift_range) * width
            dy = np.random.uniform(-self.config.height_shift_range, self.config.height_shift_range) * height
            
            # Apply shift
            if len(image.shape) == 3:
                shift_vector = [dy, dx, 0]
            else:
                shift_vector = [dy, dx]
            
            return shift(image, shift_vector, mode='nearest')
        except ImportError:
            logger.warning("scipy not available for image translation")
            return image
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        adjusted = image * factor
        
        # Clip to valid range
        if image.dtype == np.uint8:
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        else:
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def _zoom_image(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom image by given factor."""
        try:
            from scipy.ndimage import zoom
            
            if len(image.shape) == 3:
                zoom_factors = [zoom_factor, zoom_factor, 1]
            else:
                zoom_factors = [zoom_factor, zoom_factor]
            
            zoomed = zoom(image, zoom_factors, mode='nearest')
            
            # Crop or pad to original size
            original_shape = image.shape
            zoomed_shape = zoomed.shape
            
            if zoom_factor > 1:
                # Crop center
                start_h = (zoomed_shape[0] - original_shape[0]) // 2
                start_w = (zoomed_shape[1] - original_shape[1]) // 2
                
                if len(image.shape) == 3:
                    zoomed = zoomed[start_h:start_h + original_shape[0],
                                   start_w:start_w + original_shape[1], :]
                else:
                    zoomed = zoomed[start_h:start_h + original_shape[0],
                                   start_w:start_w + original_shape[1]]
            else:
                # Pad to original size
                pad_h = (original_shape[0] - zoomed_shape[0]) // 2
                pad_w = (original_shape[1] - zoomed_shape[1]) // 2
                
                if len(image.shape) == 3:
                    pad_width = [(pad_h, original_shape[0] - zoomed_shape[0] - pad_h),
                                (pad_w, original_shape[1] - zoomed_shape[1] - pad_w),
                                (0, 0)]
                else:
                    pad_width = [(pad_h, original_shape[0] - zoomed_shape[0] - pad_h),
                                (pad_w, original_shape[1] - zoomed_shape[1] - pad_w)]
                
                zoomed = np.pad(zoomed, pad_width, mode='edge')
            
            return zoomed
            
        except ImportError:
            logger.warning("scipy not available for image zoom")
            return image


class TextAugmentor(BaseAugmentor):
    """Augmentor for text data."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        super().__init__(config)
        self._stopwords = self._get_stopwords()
    
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment text dataset."""
        if dataset.info.data_type != DataType.TEXT:
            raise DataError(f"TextAugmentor expects text data, got {dataset.info.data_type.value}")
        
        logger.info(f"Augmenting text dataset from {len(dataset)} to {self._calculate_target_size(len(dataset))} samples")
        
        original_data = []
        original_targets = []
        
        # Collect original data
        for i in range(len(dataset)):
            sample, target = dataset[i]
            original_data.append(sample)
            original_targets.append(target)
        
        # Generate augmented data
        augmented_data = original_data.copy()
        augmented_targets = original_targets.copy()
        
        target_size = self._calculate_target_size(len(dataset))
        augmentations_needed = target_size - len(dataset)
        
        for _ in range(augmentations_needed):
            # Randomly select an original sample to augment
            idx = np.random.randint(0, len(original_data))
            original_sample = original_data[idx]
            original_target = original_targets[idx]
            
            # Apply random augmentations
            if isinstance(original_sample, str):
                augmented_sample = self._augment_text(original_sample)
                augmented_data.append(augmented_sample)
                augmented_targets.append(original_target)
        
        # Create new dataset
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(augmented_data),
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'augmented': True}
        )
        
        return Dataset(
            augmented_data,
            targets=augmented_targets if augmented_targets[0] is not None else None,
            info=new_info
        )
    
    def _augment_text(self, text: str) -> str:
        """Apply random augmentations to text."""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        # Apply augmentations with probability
        if self.config.synonym_replacement and np.random.random() < self.config.augmentation_probability:
            words = self._synonym_replacement(words)
        
        if self.config.random_insertion and np.random.random() < self.config.augmentation_probability:
            words = self._random_insertion(words)
        
        if self.config.random_swap and np.random.random() < self.config.augmentation_probability:
            words = self._random_swap(words)
        
        if self.config.random_deletion and np.random.random() < self.config.augmentation_probability:
            words = self._random_deletion(words)
        
        return ' '.join(words)
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace random words with synonyms."""
        # Simple synonym replacement (in practice, you'd use a thesaurus or word embeddings)
        synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed'],
            'happy': ['joyful', 'cheerful', 'glad', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'melancholy'],
        }
        
        new_words = words.copy()
        
        # Replace one random word
        if len(words) > 0:
            idx = np.random.randint(0, len(words))
            word = words[idx].lower()
            
            if word in synonyms:
                synonym = np.random.choice(synonyms[word])
                new_words[idx] = synonym
        
        return new_words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """Insert random words."""
        if len(words) == 0:
            return words
        
        # Simple insertion of common words
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'very', 'really', 'quite']
        
        # Insert at random position
        insert_pos = np.random.randint(0, len(words) + 1)
        insert_word = np.random.choice(common_words)
        
        new_words = words[:insert_pos] + [insert_word] + words[insert_pos:]
        return new_words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """Randomly swap two words."""
        if len(words) < 2:
            return words
        
        new_words = words.copy()
        
        # Swap two random positions
        idx1, idx2 = np.random.choice(len(words), 2, replace=False)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """Randomly delete words."""
        if len(words) <= 2:
            return words
        
        # Delete non-stopwords preferentially
        candidates = []
        for i, word in enumerate(words):
            if word.lower() not in self._stopwords:
                candidates.append(i)
        
        if not candidates:
            candidates = list(range(len(words)))
        
        # Delete one random word
        delete_idx = np.random.choice(candidates)
        new_words = words[:delete_idx] + words[delete_idx + 1:]
        
        return new_words
    
    def _get_stopwords(self) -> set:
        """Get set of common stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some'
        }


class TabularAugmentor(BaseAugmentor):
    """Augmentor for tabular data."""
    
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment tabular dataset."""
        if dataset.info.data_type != DataType.TABULAR:
            raise DataError(f"TabularAugmentor expects tabular data, got {dataset.info.data_type.value}")
        
        logger.info(f"Augmenting tabular dataset from {len(dataset)} to {self._calculate_target_size(len(dataset))} samples")
        
        # Convert to array for processing
        original_data = []
        original_targets = []
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            original_data.append(sample)
            original_targets.append(target)
        
        data_array = np.array(original_data)
        
        # Calculate feature statistics for noise generation
        feature_stats = {}
        for col in range(data_array.shape[1]):
            col_data = data_array[:, col]
            if np.issubdtype(col_data.dtype, np.number):
                feature_stats[col] = {
                    'mean': np.mean(col_data),
                    'std': np.std(col_data),
                    'is_numeric': True
                }
            else:
                feature_stats[col] = {'is_numeric': False}
        
        # Generate augmented data
        augmented_data = original_data.copy()
        augmented_targets = original_targets.copy()
        
        target_size = self._calculate_target_size(len(dataset))
        augmentations_needed = target_size - len(dataset)
        
        for _ in range(augmentations_needed):
            # Randomly select an original sample to augment
            idx = np.random.randint(0, len(original_data))
            original_sample = original_data[idx].copy()
            original_target = original_targets[idx]
            
            # Apply augmentations
            augmented_sample = self._augment_tabular_sample(original_sample, feature_stats)
            
            augmented_data.append(augmented_sample)
            augmented_targets.append(original_target)
        
        # Create new dataset
        new_info = DatasetInfo(
            data_type=dataset.info.data_type,
            num_samples=len(augmented_data),
            shape=(len(augmented_data), len(augmented_data[0])) if augmented_data else None,
            num_classes=dataset.info.num_classes,
            class_names=dataset.info.class_names,
            feature_names=dataset.info.feature_names,
            target_column=dataset.info.target_column,
            file_paths=dataset.info.file_paths,
            metadata={**(dataset.info.metadata or {}), 'augmented': True}
        )
        
        return Dataset(
            augmented_data,
            targets=augmented_targets if augmented_targets[0] is not None else None,
            info=new_info
        )
    
    def _augment_tabular_sample(self, sample: np.ndarray, feature_stats: Dict) -> np.ndarray:
        """Augment a single tabular sample."""
        augmented = sample.copy()
        
        # Add Gaussian noise to numeric features
        for col, stats in feature_stats.items():
            if stats['is_numeric'] and np.random.random() < 0.5:  # 50% chance to add noise
                noise_std = stats['std'] * self.config.noise_factor
                noise = np.random.normal(0, noise_std)
                augmented[col] += noise
        
        # Feature dropout (set random features to mean/mode)
        if self.config.feature_dropout > 0:
            n_features = len(augmented)
            n_dropout = int(n_features * self.config.feature_dropout)
            
            if n_dropout > 0:
                dropout_indices = np.random.choice(n_features, n_dropout, replace=False)
                
                for idx in dropout_indices:
                    if feature_stats[idx]['is_numeric']:
                        augmented[idx] = feature_stats[idx]['mean']
                    # For categorical features, we'd need mode, but this is simplified
        
        return augmented


class AugmentorFactory:
    """Factory for creating appropriate augmentors."""
    
    @staticmethod
    def create_augmentor(
        data_type: DataType,
        config: Optional[AugmentationConfig] = None
    ) -> BaseAugmentor:
        """
        Create appropriate augmentor for data type.
        
        Args:
            data_type: Type of data to augment
            config: Augmentation configuration
            
        Returns:
            Appropriate augmentor instance
            
        Raises:
            DataError: If data type is not supported
        """
        augmentors = {
            DataType.IMAGE: ImageAugmentor,
            DataType.TEXT: TextAugmentor,
            DataType.TABULAR: TabularAugmentor,
        }
        
        if data_type not in augmentors:
            supported_types = list(augmentors.keys())
            raise DataError(f"No augmentor available for data type: {data_type.value}. Supported types: {[t.value for t in supported_types]}")
        
        return augmentors[data_type](config)


def augment_data(
    dataset: Dataset,
    config: Optional[AugmentationConfig] = None
) -> Dataset:
    """
    Augment dataset using appropriate augmentor.
    
    Args:
        dataset: Dataset to augment
        config: Augmentation configuration
        
    Returns:
        Augmented dataset
    """
    augmentor = AugmentorFactory.create_augmentor(dataset.info.data_type, config)
    return augmentor.augment(dataset)