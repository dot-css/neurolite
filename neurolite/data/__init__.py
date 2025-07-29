"""
Data handling and preprocessing module for NeuroLite.

This module provides automatic data type detection, unified data loading interfaces,
data validation utilities, preprocessing pipelines, data cleaning, augmentation,
and intelligent data splitting for various data formats including images, text, CSV,
audio, and video files.
"""

from .detector import DataTypeDetector, DataType, detect_data_type
from .loader import DataLoader, Dataset, DatasetInfo, load_data
from .validator import DataValidator, ValidationResult, validate_data
from .preprocessor import (
    BasePreprocessor, ImagePreprocessor, TextPreprocessor, TabularPreprocessor,
    PreprocessorFactory, PreprocessingConfig, preprocess_data
)
from .cleaner import DataCleaner, CleaningConfig, CleaningReport, clean_data
from .augmentation import (
    BaseAugmentor, ImageAugmentor, TextAugmentor, TabularAugmentor,
    AugmentorFactory, AugmentationConfig, augment_data
)
from .splitter import (
    DataSplitter, DataSplit, SplitConfig, split_data,
    train_test_split, train_validation_split
)

__all__ = [
    # Detection and loading
    'DataTypeDetector',
    'DataType', 
    'detect_data_type',
    'DataLoader',
    'Dataset',
    'DatasetInfo',
    'load_data',
    
    # Validation
    'DataValidator',
    'ValidationResult',
    'validate_data',
    
    # Preprocessing
    'BasePreprocessor',
    'ImagePreprocessor',
    'TextPreprocessor', 
    'TabularPreprocessor',
    'PreprocessorFactory',
    'PreprocessingConfig',
    'preprocess_data',
    
    # Cleaning
    'DataCleaner',
    'CleaningConfig',
    'CleaningReport',
    'clean_data',
    
    # Augmentation
    'BaseAugmentor',
    'ImageAugmentor',
    'TextAugmentor',
    'TabularAugmentor',
    'AugmentorFactory',
    'AugmentationConfig',
    'augment_data',
    
    # Splitting
    'DataSplitter',
    'DataSplit',
    'SplitConfig',
    'split_data',
    'train_test_split',
    'train_validation_split'
]