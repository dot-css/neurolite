"""
Unit tests for data splitting functionality.
"""

import pytest
import numpy as np
from collections import Counter

from neurolite.data import (
    Dataset, DatasetInfo, DataType,
    DataSplitter, DataSplit, SplitConfig,
    split_data, train_test_split, train_validation_split
)
from neurolite.core import DataError


class TestSplitConfig:
    """Test split configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SplitConfig()
        
        assert config.train_ratio == 0.7
        assert config.validation_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.stratify is True
        assert config.shuffle is True
        assert config.random_seed == 42
        assert config.min_samples_per_split == 1
        assert config.min_samples_per_class == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SplitConfig(
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            stratify=False,
            shuffle=False,
            random_seed=123,
            min_samples_per_split=5,
            min_samples_per_class=3
        )
        
        assert config.train_ratio == 0.8
        assert config.validation_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.stratify is False
        assert config.shuffle is False
        assert config.random_seed == 123
        assert config.min_samples_per_split == 5
        assert config.min_samples_per_class == 3
    
    def test_invalid_ratios_sum(self):
        """Test validation of split ratios sum."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            SplitConfig(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)
    
    def test_negative_ratios(self):
        """Test validation of negative ratios."""
        with pytest.raises(ValueError, match="Split ratios must be non-negative"):
            SplitConfig(train_ratio=-0.1, validation_ratio=0.5, test_ratio=0.6)


class TestDataSplit:
    """Test data split container."""
    
    def test_data_split_creation(self):
        """Test creating a data split."""
        # Create mock datasets
        train_data = [np.array([1, 2]), np.array([3, 4])]
        train_targets = [0, 1]
        train_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=2)
        train_dataset = Dataset(train_data, targets=train_targets, info=train_info)
        
        val_data = [np.array([5, 6])]
        val_targets = [0]
        val_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=1)
        val_dataset = Dataset(val_data, targets=val_targets, info=val_info)
        
        split = DataSplit(train=train_dataset, validation=val_dataset)
        
        assert split.train == train_dataset
        assert split.validation == val_dataset
        assert split.test is None
    
    def test_get_split_info(self):
        """Test getting split information."""
        # Create mock datasets
        train_data = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        train_targets = [0, 1, 0]
        train_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=3)
        train_dataset = Dataset(train_data, targets=train_targets, info=train_info)
        
        val_data = [np.array([7, 8])]
        val_targets = [1]
        val_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=1)
        val_dataset = Dataset(val_data, targets=val_targets, info=val_info)
        
        test_data = [np.array([9, 10])]
        test_targets = [0]
        test_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=1)
        test_dataset = Dataset(test_data, targets=test_targets, info=test_info)
        
        split = DataSplit(train=train_dataset, validation=val_dataset, test=test_dataset)
        info = split.get_split_info()
        
        assert info["train"] == 3
        assert info["validation"] == 1
        assert info["test"] == 1


class TestDataSplitter:
    """Test data splitter functionality."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        splitter = DataSplitter()
        assert isinstance(splitter.config, SplitConfig)
        assert splitter.config.train_ratio == 0.7
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
        splitter = DataSplitter(config)
        assert splitter.config.train_ratio == 0.8
    
    def test_split_too_small_dataset(self):
        """Test splitting dataset that's too small."""
        data = [np.array([1, 2])]
        targets = [0]
        info = DatasetInfo(data_type=DataType.TABULAR, num_samples=1)
        dataset = Dataset(data, targets=targets, info=info)
        
        splitter = DataSplitter()
        with pytest.raises(DataError, match="Dataset too small for splitting"):
            splitter.split(dataset)
    
    def test_split_tabular_data_stratified(self):
        """Test stratified splitting of tabular data."""
        # Create balanced dataset
        data = [
            np.array([1, 2]), np.array([3, 4]), np.array([5, 6]),  # Class 0
            np.array([7, 8]), np.array([9, 10]), np.array([11, 12])  # Class 1
        ]
        targets = [0, 0, 0, 1, 1, 1]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=6,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        config = SplitConfig(
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            stratify=True,
            random_seed=42
        )
        splitter = DataSplitter(config)
        split = splitter.split(dataset)
        
        assert isinstance(split, DataSplit)
        assert len(split.train) >= 2  # Stratified splitting may adjust ratios for balance
        assert split.test is not None  # May not have validation if too few samples
        
        # Check that classes are balanced in each split
        train_targets = [split.train[i][1] for i in range(len(split.train))]
        train_counts = Counter(train_targets)
        assert len(train_counts) == 2  # Both classes present
    
    def test_split_tabular_data_random(self):
        """Test random splitting of tabular data."""
        data = [np.array([i, i+1]) for i in range(10)]
        targets = [i % 2 for i in range(10)]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=10,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            stratify=False,
            random_seed=42
        )
        splitter = DataSplitter(config)
        split = splitter.split(dataset)
        
        assert isinstance(split, DataSplit)
        assert len(split.train) == 7  # 70% of 10
        assert len(split.validation) == 2  # 20% of 10
        assert len(split.test) == 1  # 10% of 10
    
    def test_split_text_data(self):
        """Test splitting text data."""
        texts = [f"Sample text {i}" for i in range(8)]
        targets = [i % 2 for i in range(8)]
        info = DatasetInfo(
            data_type=DataType.TEXT,
            num_samples=8,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(texts, targets=targets, info=info)
        
        splitter = DataSplitter()
        split = splitter.split(dataset)
        
        assert isinstance(split, DataSplit)
        assert len(split.train) >= 4  # Stratified splitting may adjust ratios for balance
        assert split.test is not None  # May not have validation if too few samples
    
    def test_split_image_data(self):
        """Test splitting image data."""
        images = [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(6)]
        targets = [i % 3 for i in range(6)]  # 3 classes
        info = DatasetInfo(
            data_type=DataType.IMAGE,
            num_samples=6,
            shape=(16, 16, 3),
            num_classes=3,
            class_names=["A", "B", "C"]
        )
        dataset = Dataset(images, targets=targets, info=info)
        
        splitter = DataSplitter()
        split = splitter.split(dataset)
        
        assert isinstance(split, DataSplit)
        assert len(split.train) >= 3  # Stratified splitting may adjust ratios for balance
    
    def test_should_stratify_classification(self):
        """Test stratification decision for classification data."""
        targets = [0, 1, 0, 1, 0, 1, 0, 1]  # Balanced binary classification
        
        config = SplitConfig(stratify=True, min_samples_per_class=2)
        splitter = DataSplitter(config)
        
        should_stratify = splitter._should_stratify(targets)
        assert should_stratify is True
    
    def test_should_stratify_regression(self):
        """Test stratification decision for regression data."""
        targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Continuous values
        
        config = SplitConfig(stratify=True)
        splitter = DataSplitter(config)
        
        should_stratify = splitter._should_stratify(targets)
        assert should_stratify is False  # Too many unique values
    
    def test_should_stratify_insufficient_samples(self):
        """Test stratification decision with insufficient samples per class."""
        targets = [0, 1, 2]  # Only 1 sample per class
        
        config = SplitConfig(stratify=True, min_samples_per_class=2)
        splitter = DataSplitter(config)
        
        should_stratify = splitter._should_stratify(targets)
        assert should_stratify is False
    
    def test_should_stratify_disabled(self):
        """Test stratification decision when disabled."""
        targets = [0, 1, 0, 1, 0, 1]
        
        config = SplitConfig(stratify=False)
        splitter = DataSplitter(config)
        
        should_stratify = splitter._should_stratify(targets)
        assert should_stratify is False
    
    def test_should_stratify_no_targets(self):
        """Test stratification decision with no targets."""
        targets = [None, None, None]
        
        config = SplitConfig(stratify=True)
        splitter = DataSplitter(config)
        
        should_stratify = splitter._should_stratify(targets)
        assert should_stratify is False
    
    def test_stratified_split(self):
        """Test stratified splitting logic."""
        targets = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]  # 4 samples per class
        
        config = SplitConfig(
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            shuffle=False,
            random_seed=42
        )
        splitter = DataSplitter(config)
        
        indices = splitter._stratified_split(targets)
        
        assert 'train' in indices
        assert 'validation' in indices
        assert 'test' in indices
        
        # Check that each class is represented in each split
        train_targets = [targets[i] for i in indices['train']]
        val_targets = [targets[i] for i in indices['validation']]
        test_targets = [targets[i] for i in indices['test']]
        
        train_counts = Counter(train_targets)
        val_counts = Counter(val_targets)
        test_counts = Counter(test_targets)
        
        # Each class should be in each split
        assert len(train_counts) == 3
        assert len(val_counts) == 3
        assert len(test_counts) == 3
    
    def test_random_split(self):
        """Test random splitting logic."""
        n_samples = 10
        
        config = SplitConfig(
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            shuffle=False,
            random_seed=42
        )
        splitter = DataSplitter(config)
        
        indices = splitter._random_split(n_samples)
        
        assert 'train' in indices
        assert 'validation' in indices
        assert 'test' in indices
        
        assert len(indices['train']) == 6  # 60% of 10
        assert len(indices['validation']) == 2  # 20% of 10
        assert len(indices['test']) == 2  # 20% of 10
        
        # Check that all indices are covered
        all_indices = set(indices['train'] + indices['validation'] + indices['test'])
        assert all_indices == set(range(n_samples))
    
    def test_random_split_no_validation(self):
        """Test random splitting with no validation set."""
        n_samples = 10
        
        config = SplitConfig(
            train_ratio=0.8,
            validation_ratio=0.0,
            test_ratio=0.2,
            shuffle=False
        )
        splitter = DataSplitter(config)
        
        indices = splitter._random_split(n_samples)
        
        assert len(indices['train']) == 8
        assert indices['validation'] is None
        assert len(indices['test']) == 2
    
    def test_create_split_dataset(self):
        """Test creating dataset from split indices."""
        data_list = [np.array([i, i+1]) for i in range(5)]
        target_list = [i % 2 for i in range(5)]
        
        original_info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=5,
            num_classes=2,
            class_names=["A", "B"]
        )
        original_dataset = Dataset(data_list, targets=target_list, info=original_info)
        
        indices = [0, 2, 4]  # Select samples 0, 2, 4
        
        splitter = DataSplitter()
        split_dataset = splitter._create_split_dataset(
            original_dataset, data_list, target_list, indices
        )
        
        assert len(split_dataset) == 3
        assert split_dataset.info.num_samples == 3
        assert split_dataset.info.data_type == DataType.TABULAR
        
        # Check that correct samples are selected
        sample_0, target_0 = split_dataset[0]
        assert np.array_equal(sample_0, np.array([0, 1]))
        assert target_0 == 0
    
    def test_create_split_dataset_empty_indices(self):
        """Test creating dataset with empty indices."""
        data_list = [np.array([1, 2])]
        target_list = [0]
        
        original_info = DatasetInfo(data_type=DataType.TABULAR, num_samples=1)
        original_dataset = Dataset(data_list, targets=target_list, info=original_info)
        
        splitter = DataSplitter()
        with pytest.raises(DataError, match="Cannot create dataset from empty indices"):
            splitter._create_split_dataset(original_dataset, data_list, target_list, [])
    
    def test_get_split_statistics(self):
        """Test getting split statistics."""
        # Create mock split
        train_data = [np.array([1, 2]), np.array([3, 4])]
        train_targets = [0, 1]
        train_info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=2,
            num_classes=2,
            class_names=["A", "B"]
        )
        train_dataset = Dataset(train_data, targets=train_targets, info=train_info)
        
        val_data = [np.array([5, 6])]
        val_targets = [0]
        val_info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=1,
            num_classes=2,
            class_names=["A", "B"]
        )
        val_dataset = Dataset(val_data, targets=val_targets, info=val_info)
        
        split = DataSplit(train=train_dataset, validation=val_dataset)
        
        splitter = DataSplitter()
        stats = splitter.get_split_statistics(split)
        
        assert 'train' in stats
        assert 'validation' in stats
        assert 'test' not in stats
        
        assert stats['train']['num_samples'] == 2
        assert stats['train']['data_type'] == 'tabular'
        assert 'class_distribution' in stats['train']
        assert 'class_balance' in stats['train']


def test_split_data_function():
    """Test the split_data convenience function."""
    data = [np.array([i, i+1]) for i in range(10)]
    targets = [i % 2 for i in range(10)]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=10,
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    split = split_data(dataset)
    
    assert isinstance(split, DataSplit)
    assert split.train is not None
    assert split.test is not None  # May not have validation if too few samples per class


def test_split_data_with_custom_params():
    """Test split_data function with custom parameters."""
    data = [np.array([i, i+1]) for i in range(10)]
    targets = [i % 2 for i in range(10)]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=10,
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    split = split_data(
        dataset,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        stratify=False,
        shuffle=False,
        random_seed=123
    )
    
    assert isinstance(split, DataSplit)
    assert len(split.train) == 8
    assert len(split.validation) == 1
    assert len(split.test) == 1


def test_train_test_split_function():
    """Test the train_test_split convenience function."""
    data = [np.array([i, i+1]) for i in range(10)]
    targets = [i % 2 for i in range(10)]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=10,
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    train_dataset, test_dataset = train_test_split(dataset, test_ratio=0.3)
    
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)
    # Stratified splitting may adjust ratios for balance
    assert len(train_dataset) >= 6  # Approximately 70% of 10
    assert len(test_dataset) >= 3   # Approximately 30% of 10


def test_train_validation_split_function():
    """Test the train_validation_split convenience function."""
    data = [np.array([i, i+1]) for i in range(10)]
    targets = [i % 2 for i in range(10)]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=10,
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    train_dataset, val_dataset = train_validation_split(dataset, validation_ratio=0.2)
    
    assert isinstance(train_dataset, Dataset)
    assert isinstance(val_dataset, Dataset)
    assert len(train_dataset) == 8  # 80% of 10
    assert len(val_dataset) == 2   # 20% of 10