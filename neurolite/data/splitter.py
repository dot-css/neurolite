"""
Intelligent data splitting utilities for NeuroLite.

Provides data splitting functionality with stratification support,
ensuring balanced distribution of classes across train/validation/test sets.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import Counter

from ..core import get_logger, DataError
from .detector import DataType
from .loader import Dataset, DatasetInfo


logger = get_logger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    stratify: bool = True  # Use stratified splitting for classification
    shuffle: bool = True
    random_seed: int = 42
    
    # Minimum samples per split
    min_samples_per_split: int = 1
    min_samples_per_class: int = 2  # For stratification
    
    def __post_init__(self):
        """Validate split ratios."""
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        if any(ratio < 0 for ratio in [self.train_ratio, self.validation_ratio, self.test_ratio]):
            raise ValueError("Split ratios must be non-negative")


@dataclass
class DataSplit:
    """Container for split datasets."""
    train: Dataset
    validation: Optional[Dataset] = None
    test: Optional[Dataset] = None
    
    def get_split_info(self) -> Dict[str, int]:
        """Get information about split sizes."""
        info = {"train": len(self.train)}
        
        if self.validation:
            info["validation"] = len(self.validation)
        
        if self.test:
            info["test"] = len(self.test)
        
        return info


class DataSplitter:
    """
    Intelligent data splitting system with stratification support.
    
    Handles splitting datasets while maintaining class balance and
    ensuring minimum sample requirements are met.
    """
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize data splitter.
        
        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
        np.random.seed(self.config.random_seed)
    
    def split(self, dataset: Dataset) -> DataSplit:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            DataSplit containing the split datasets
            
        Raises:
            DataError: If dataset is too small or splitting fails
        """
        logger.info(f"Splitting dataset of {len(dataset)} samples")
        
        # Validate dataset size
        if len(dataset) < self.config.min_samples_per_split * 2:
            raise DataError(f"Dataset too small for splitting: {len(dataset)} samples (minimum: {self.config.min_samples_per_split * 2})")
        
        # Extract data and targets
        data_list = []
        target_list = []
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            data_list.append(sample)
            target_list.append(target)
        
        # Determine if we should stratify
        should_stratify = self._should_stratify(target_list)
        
        if should_stratify:
            logger.debug("Using stratified splitting")
            indices = self._stratified_split(target_list)
        else:
            logger.debug("Using random splitting")
            indices = self._random_split(len(dataset))
        
        # Create split datasets
        train_dataset = self._create_split_dataset(dataset, data_list, target_list, indices['train'])
        
        validation_dataset = None
        if indices['validation']:
            validation_dataset = self._create_split_dataset(dataset, data_list, target_list, indices['validation'])
        
        test_dataset = None
        if indices['test']:
            test_dataset = self._create_split_dataset(dataset, data_list, target_list, indices['test'])
        
        split = DataSplit(train_dataset, validation_dataset, test_dataset)
        
        # Log split information
        split_info = split.get_split_info()
        logger.info(f"Split completed: {split_info}")
        
        return split
    
    def _should_stratify(self, targets: List) -> bool:
        """Determine if stratified splitting should be used."""
        if not self.config.stratify:
            return False
        
        # Check if targets exist and are suitable for stratification
        if not targets or targets[0] is None:
            return False
        
        # Count unique targets
        unique_targets = set(targets)
        
        # Don't stratify if too many unique values (likely regression)
        if len(unique_targets) > len(targets) * 0.5:
            logger.debug("Too many unique targets for stratification (likely regression)")
            return False
        
        # Check minimum samples per class
        target_counts = Counter(targets)
        min_count = min(target_counts.values())
        
        if min_count < self.config.min_samples_per_class:
            logger.warning(f"Some classes have fewer than {self.config.min_samples_per_class} samples, skipping stratification")
            return False
        
        return True
    
    def _stratified_split(self, targets: List) -> Dict[str, List[int]]:
        """Perform stratified splitting."""
        # Group indices by target class
        class_indices = {}
        for i, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(i)
        
        # Shuffle indices within each class
        if self.config.shuffle:
            for indices in class_indices.values():
                np.random.shuffle(indices)
        
        # Split each class proportionally
        train_indices = []
        validation_indices = []
        test_indices = []
        
        for target_class, indices in class_indices.items():
            n_samples = len(indices)
            
            # Calculate split sizes
            n_train = max(1, int(n_samples * self.config.train_ratio))
            n_validation = int(n_samples * self.config.validation_ratio)
            n_test = n_samples - n_train - n_validation
            
            # Ensure minimum samples
            if n_validation > 0 and n_validation < 1:
                n_validation = 1
                n_test = max(0, n_samples - n_train - n_validation)
            
            if n_test > 0 and n_test < 1:
                n_test = 1
                n_validation = max(0, n_samples - n_train - n_test)
            
            # Split indices
            train_indices.extend(indices[:n_train])
            
            if n_validation > 0:
                validation_indices.extend(indices[n_train:n_train + n_validation])
            
            if n_test > 0:
                test_indices.extend(indices[n_train + n_validation:])
        
        # Shuffle final indices
        if self.config.shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(validation_indices)
            np.random.shuffle(test_indices)
        
        return {
            'train': train_indices,
            'validation': validation_indices if validation_indices else None,
            'test': test_indices if test_indices else None
        }
    
    def _random_split(self, n_samples: int) -> Dict[str, List[int]]:
        """Perform random splitting."""
        # Create shuffled indices
        indices = list(range(n_samples))
        if self.config.shuffle:
            np.random.shuffle(indices)
        
        # Calculate split sizes
        n_train = int(n_samples * self.config.train_ratio)
        n_validation = int(n_samples * self.config.validation_ratio)
        n_test = n_samples - n_train - n_validation
        
        # Ensure minimum samples
        if n_validation > 0 and n_validation < self.config.min_samples_per_split:
            n_validation = 0
            n_test = n_samples - n_train
        
        if n_test > 0 and n_test < self.config.min_samples_per_split:
            n_test = 0
        
        # Split indices
        train_indices = indices[:n_train]
        validation_indices = indices[n_train:n_train + n_validation] if n_validation > 0 else None
        test_indices = indices[n_train + n_validation:] if n_test > 0 else None
        
        return {
            'train': train_indices,
            'validation': validation_indices,
            'test': test_indices
        }
    
    def _create_split_dataset(
        self,
        original_dataset: Dataset,
        data_list: List,
        target_list: List,
        indices: List[int]
    ) -> Dataset:
        """Create a dataset from split indices."""
        if not indices:
            raise DataError("Cannot create dataset from empty indices")
        
        # Extract split data
        split_data = [data_list[i] for i in indices]
        split_targets = [target_list[i] for i in indices] if target_list[0] is not None else None
        
        # Calculate class distribution for info
        class_names = original_dataset.info.class_names
        num_classes = original_dataset.info.num_classes
        
        if split_targets and class_names:
            # Update class distribution
            target_counts = Counter(split_targets)
            logger.debug(f"Split class distribution: {dict(target_counts)}")
        
        # Create new dataset info
        # Handle file_paths properly - for tabular data, all samples come from the same file
        split_file_paths = None
        if original_dataset.info.file_paths:
            if len(original_dataset.info.file_paths) == 1:
                # Single file (e.g., CSV) - all splits reference the same file
                split_file_paths = original_dataset.info.file_paths
            else:
                # Multiple files - map indices to corresponding files
                split_file_paths = [original_dataset.info.file_paths[i] for i in indices]
        
        new_info = DatasetInfo(
            data_type=original_dataset.info.data_type,
            num_samples=len(split_data),
            shape=split_data[0].shape if split_data and hasattr(split_data[0], 'shape') else None,
            num_classes=num_classes,
            class_names=class_names,
            feature_names=original_dataset.info.feature_names,
            target_column=original_dataset.info.target_column,
            file_paths=split_file_paths,
            metadata={**(original_dataset.info.metadata or {}), 'split': True}
        )
        
        return Dataset(split_data, targets=split_targets, info=new_info)
    
    def get_split_statistics(self, split: DataSplit) -> Dict[str, Dict]:
        """Get detailed statistics about the split."""
        stats = {}
        
        for split_name, dataset in [('train', split.train), ('validation', split.validation), ('test', split.test)]:
            if dataset is None:
                continue
            
            split_stats = {
                'num_samples': len(dataset),
                'data_type': dataset.info.data_type.value
            }
            
            # Add class distribution if available
            if dataset.info.class_names:
                targets = []
                for i in range(len(dataset)):
                    _, target = dataset[i]
                    if target is not None:
                        targets.append(target)
                
                if targets:
                    class_counts = Counter(targets)
                    split_stats['class_distribution'] = dict(class_counts)
                    split_stats['class_balance'] = {
                        class_name: class_counts.get(i, 0) / len(targets)
                        for i, class_name in enumerate(dataset.info.class_names)
                    }
            
            stats[split_name] = split_stats
        
        return stats


def split_data(
    dataset: Dataset,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    shuffle: bool = True,
    random_seed: int = 42
) -> DataSplit:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Proportion of data for training
        validation_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        stratify: Whether to use stratified splitting
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        DataSplit containing the split datasets
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        shuffle=shuffle,
        random_seed=random_seed
    )
    
    splitter = DataSplitter(config)
    return splitter.split(dataset)


def train_test_split(
    dataset: Dataset,
    test_ratio: float = 0.2,
    stratify: bool = True,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and test sets only.
    
    Args:
        dataset: Dataset to split
        test_ratio: Proportion of data for testing
        stratify: Whether to use stratified splitting
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    config = SplitConfig(
        train_ratio=1.0 - test_ratio,
        validation_ratio=0.0,
        test_ratio=test_ratio,
        stratify=stratify,
        shuffle=shuffle,
        random_seed=random_seed
    )
    
    splitter = DataSplitter(config)
    split = splitter.split(dataset)
    
    return split.train, split.test


def train_validation_split(
    dataset: Dataset,
    validation_ratio: float = 0.2,
    stratify: bool = True,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets only.
    
    Args:
        dataset: Dataset to split
        validation_ratio: Proportion of data for validation
        stratify: Whether to use stratified splitting
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    config = SplitConfig(
        train_ratio=1.0 - validation_ratio,
        validation_ratio=validation_ratio,
        test_ratio=0.0,
        stratify=stratify,
        shuffle=shuffle,
        random_seed=random_seed
    )
    
    splitter = DataSplitter(config)
    split = splitter.split(dataset)
    
    return split.train, split.validation