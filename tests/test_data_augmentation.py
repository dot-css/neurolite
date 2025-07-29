"""
Unit tests for data augmentation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from neurolite.data import (
    Dataset, DatasetInfo, DataType,
    BaseAugmentor, ImageAugmentor, TextAugmentor, TabularAugmentor,
    AugmentorFactory, AugmentationConfig, augment_data
)
from neurolite.core import DataError


class TestAugmentationConfig:
    """Test augmentation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        
        assert config.rotation_range == 15.0
        assert config.width_shift_range == 0.1
        assert config.height_shift_range == 0.1
        assert config.brightness_range == (0.8, 1.2)
        assert config.zoom_range == 0.1
        assert config.horizontal_flip is True
        assert config.vertical_flip is False
        assert config.synonym_replacement is True
        assert config.random_insertion is True
        assert config.random_swap is True
        assert config.random_deletion is True
        assert config.augmentation_probability == 0.1
        assert config.noise_factor == 0.05
        assert config.feature_dropout == 0.1
        assert config.augmentation_factor == 2.0
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AugmentationConfig(
            rotation_range=30.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.5, 1.5),
            zoom_range=0.2,
            horizontal_flip=False,
            vertical_flip=True,
            synonym_replacement=False,
            random_insertion=False,
            random_swap=False,
            random_deletion=False,
            augmentation_probability=0.2,
            noise_factor=0.1,
            feature_dropout=0.2,
            augmentation_factor=3.0,
            random_seed=123
        )
        
        assert config.rotation_range == 30.0
        assert config.width_shift_range == 0.2
        assert config.height_shift_range == 0.2
        assert config.brightness_range == (0.5, 1.5)
        assert config.zoom_range == 0.2
        assert config.horizontal_flip is False
        assert config.vertical_flip is True
        assert config.synonym_replacement is False
        assert config.random_insertion is False
        assert config.random_swap is False
        assert config.random_deletion is False
        assert config.augmentation_probability == 0.2
        assert config.noise_factor == 0.1
        assert config.feature_dropout == 0.2
        assert config.augmentation_factor == 3.0
        assert config.random_seed == 123


class TestBaseAugmentor:
    """Test base augmentor functionality."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        # Create a concrete implementation for testing
        class TestAugmentor(BaseAugmentor):
            def augment(self, dataset):
                return dataset
        
        augmentor = TestAugmentor()
        assert isinstance(augmentor.config, AugmentationConfig)
        assert augmentor.config.augmentation_factor == 2.0
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        class TestAugmentor(BaseAugmentor):
            def augment(self, dataset):
                return dataset
        
        config = AugmentationConfig(augmentation_factor=3.0)
        augmentor = TestAugmentor(config)
        assert augmentor.config.augmentation_factor == 3.0
    
    def test_calculate_target_size(self):
        """Test target size calculation."""
        class TestAugmentor(BaseAugmentor):
            def augment(self, dataset):
                return dataset
        
        config = AugmentationConfig(augmentation_factor=2.5)
        augmentor = TestAugmentor(config)
        
        target_size = augmentor._calculate_target_size(100)
        assert target_size == 250


class TestImageAugmentor:
    """Test image augmentor functionality."""
    
    def test_init(self):
        """Test image augmentor initialization."""
        augmentor = ImageAugmentor()
        assert isinstance(augmentor.config, AugmentationConfig)
    
    def test_augment_wrong_data_type(self):
        """Test augmentation with wrong data type."""
        data = ["text1", "text2"]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.TEXT,  # Wrong type
            num_samples=2
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        augmentor = ImageAugmentor()
        with pytest.raises(DataError):
            augmentor.augment(dataset)
    
    def test_augment_images(self):
        """Test image augmentation."""
        # Create sample images
        images = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.IMAGE,
            num_samples=2,
            shape=(32, 32, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(images, targets=targets, info=info)
        
        config = AugmentationConfig(augmentation_factor=2.0)
        augmentor = ImageAugmentor(config)
        augmented_dataset = augmentor.augment(dataset)
        
        assert len(augmented_dataset) == 4  # 2 * 2.0
        assert augmented_dataset.info.data_type == DataType.IMAGE
        assert augmented_dataset.info.metadata['augmented'] is True
    
    def test_augment_image_transformations(self):
        """Test individual image transformations."""
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        config = AugmentationConfig(
            rotation_range=15.0,
            brightness_range=(0.8, 1.2),
            horizontal_flip=True
        )
        augmentor = ImageAugmentor(config)
        
        # Test that augmentation produces different image
        augmented = augmentor._augment_image(image)
        assert augmented.shape == image.shape
        assert augmented.dtype == image.dtype
    
    def test_adjust_brightness(self):
        """Test brightness adjustment."""
        image = np.ones((10, 10, 3), dtype=np.uint8) * 100
        
        augmentor = ImageAugmentor()
        
        # Test brightness increase
        brighter = augmentor._adjust_brightness(image, 1.5)
        assert np.all(brighter >= image)
        
        # Test brightness decrease
        darker = augmentor._adjust_brightness(image, 0.5)
        assert np.all(darker <= image)
    
    @patch('scipy.ndimage.rotate')
    def test_rotate_image_with_scipy(self, mock_rotate):
        """Test image rotation with scipy available."""
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        mock_rotate.return_value = image
        
        augmentor = ImageAugmentor()
        rotated = augmentor._rotate_image(image, 15.0)
        
        mock_rotate.assert_called_once()
        assert rotated.shape == image.shape
    
    def test_rotate_image_without_scipy(self):
        """Test image rotation without scipy."""
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        augmentor = ImageAugmentor()
        
        # Mock ImportError for scipy
        with patch('scipy.ndimage.rotate', side_effect=ImportError):
            rotated = augmentor._rotate_image(image, 15.0)
            # Should return original image when scipy is not available
            assert np.array_equal(rotated, image)


class TestTextAugmentor:
    """Test text augmentor functionality."""
    
    def test_init(self):
        """Test text augmentor initialization."""
        augmentor = TextAugmentor()
        assert isinstance(augmentor.config, AugmentationConfig)
        assert isinstance(augmentor._stopwords, set)
    
    def test_augment_wrong_data_type(self):
        """Test augmentation with wrong data type."""
        data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.TABULAR,  # Wrong type
            num_samples=2
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        augmentor = TextAugmentor()
        with pytest.raises(DataError):
            augmentor.augment(dataset)
    
    def test_augment_text(self):
        """Test text augmentation."""
        texts = [
            "This is a sample text for testing.",
            "Another text sample for augmentation."
        ]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.TEXT,
            num_samples=2,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(texts, targets=targets, info=info)
        
        config = AugmentationConfig(augmentation_factor=2.0)
        augmentor = TextAugmentor(config)
        augmented_dataset = augmentor.augment(dataset)
        
        assert len(augmented_dataset) == 4  # 2 * 2.0
        assert augmented_dataset.info.data_type == DataType.TEXT
        assert augmented_dataset.info.metadata['augmented'] is True
    
    def test_augment_text_transformations(self):
        """Test individual text transformations."""
        text = "This is a good example of text augmentation."
        
        config = AugmentationConfig(
            synonym_replacement=True,
            random_insertion=True,
            random_swap=True,
            random_deletion=True,
            augmentation_probability=1.0  # Always apply
        )
        augmentor = TextAugmentor(config)
        
        # Test that augmentation can produce different text
        augmented = augmentor._augment_text(text)
        assert isinstance(augmented, str)
        assert len(augmented) > 0
    
    def test_synonym_replacement(self):
        """Test synonym replacement."""
        words = ["This", "is", "good", "text"]
        
        augmentor = TextAugmentor()
        replaced = augmentor._synonym_replacement(words)
        
        assert len(replaced) == len(words)
        assert isinstance(replaced, list)
    
    def test_random_insertion(self):
        """Test random word insertion."""
        words = ["This", "is", "text"]
        
        augmentor = TextAugmentor()
        inserted = augmentor._random_insertion(words)
        
        assert len(inserted) == len(words) + 1
    
    def test_random_swap(self):
        """Test random word swapping."""
        words = ["This", "is", "good", "text"]
        
        augmentor = TextAugmentor()
        swapped = augmentor._random_swap(words)
        
        assert len(swapped) == len(words)
        assert set(swapped) == set(words)  # Same words, different order
    
    def test_random_deletion(self):
        """Test random word deletion."""
        words = ["This", "is", "good", "text", "example"]
        
        augmentor = TextAugmentor()
        deleted = augmentor._random_deletion(words)
        
        assert len(deleted) == len(words) - 1
    
    def test_random_deletion_short_text(self):
        """Test random deletion with short text."""
        words = ["Short", "text"]
        
        augmentor = TextAugmentor()
        deleted = augmentor._random_deletion(words)
        
        # Should not delete from very short text
        assert len(deleted) == len(words)
    
    def test_get_stopwords(self):
        """Test stopwords retrieval."""
        augmentor = TextAugmentor()
        stopwords = augmentor._get_stopwords()
        
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert "the" in stopwords
        assert "and" in stopwords


class TestTabularAugmentor:
    """Test tabular augmentor functionality."""
    
    def test_init(self):
        """Test tabular augmentor initialization."""
        augmentor = TabularAugmentor()
        assert isinstance(augmentor.config, AugmentationConfig)
    
    def test_augment_wrong_data_type(self):
        """Test augmentation with wrong data type."""
        data = ["text1", "text2"]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.TEXT,  # Wrong type
            num_samples=2
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        augmentor = TabularAugmentor()
        with pytest.raises(DataError):
            augmentor.augment(dataset)
    
    def test_augment_tabular(self):
        """Test tabular data augmentation."""
        data = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        targets = [0, 1, 0]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=3,
            shape=(3, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        config = AugmentationConfig(augmentation_factor=2.0)
        augmentor = TabularAugmentor(config)
        augmented_dataset = augmentor.augment(dataset)
        
        assert len(augmented_dataset) == 6  # 3 * 2.0
        assert augmented_dataset.info.data_type == DataType.TABULAR
        assert augmented_dataset.info.metadata['augmented'] is True
    
    def test_augment_tabular_sample(self):
        """Test individual tabular sample augmentation."""
        sample = np.array([1.0, 2.0, 3.0])
        feature_stats = {
            0: {'mean': 1.0, 'std': 0.5, 'is_numeric': True},
            1: {'mean': 2.0, 'std': 0.3, 'is_numeric': True},
            2: {'mean': 3.0, 'std': 0.7, 'is_numeric': True}
        }
        
        config = AugmentationConfig(noise_factor=0.1, feature_dropout=0.0)
        augmentor = TabularAugmentor(config)
        
        augmented = augmentor._augment_tabular_sample(sample, feature_stats)
        
        assert augmented.shape == sample.shape
        assert isinstance(augmented, np.ndarray)
        # Values should be close but not identical due to noise
        assert not np.array_equal(augmented, sample)


class TestAugmentorFactory:
    """Test augmentor factory functionality."""
    
    def test_create_image_augmentor(self):
        """Test creating image augmentor."""
        augmentor = AugmentorFactory.create_augmentor(DataType.IMAGE)
        assert isinstance(augmentor, ImageAugmentor)
    
    def test_create_text_augmentor(self):
        """Test creating text augmentor."""
        augmentor = AugmentorFactory.create_augmentor(DataType.TEXT)
        assert isinstance(augmentor, TextAugmentor)
    
    def test_create_tabular_augmentor(self):
        """Test creating tabular augmentor."""
        augmentor = AugmentorFactory.create_augmentor(DataType.TABULAR)
        assert isinstance(augmentor, TabularAugmentor)
    
    def test_create_augmentor_with_config(self):
        """Test creating augmentor with custom config."""
        config = AugmentationConfig(augmentation_factor=3.0)
        augmentor = AugmentorFactory.create_augmentor(DataType.IMAGE, config)
        
        assert isinstance(augmentor, ImageAugmentor)
        assert augmentor.config.augmentation_factor == 3.0
    
    def test_create_augmentor_unsupported_type(self):
        """Test creating augmentor for unsupported data type."""
        with pytest.raises(DataError):
            AugmentorFactory.create_augmentor(DataType.AUDIO)


def test_augment_data_function():
    """Test the augment_data convenience function."""
    images = [
        np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8),
        np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    ]
    targets = [0, 1]
    info = DatasetInfo(
        data_type=DataType.IMAGE,
        num_samples=2,
        shape=(16, 16, 3),
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(images, targets=targets, info=info)
    
    augmented_dataset = augment_data(dataset)
    
    assert isinstance(augmented_dataset, Dataset)
    assert len(augmented_dataset) == 4  # 2 * 2.0 (default factor)


def test_augment_data_with_custom_config():
    """Test augment_data function with custom configuration."""
    texts = ["Sample text one", "Sample text two"]
    targets = [0, 1]
    info = DatasetInfo(
        data_type=DataType.TEXT,
        num_samples=2,
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(texts, targets=targets, info=info)
    
    config = AugmentationConfig(augmentation_factor=3.0)
    augmented_dataset = augment_data(dataset, config)
    
    assert isinstance(augmented_dataset, Dataset)
    assert len(augmented_dataset) == 6  # 2 * 3.0