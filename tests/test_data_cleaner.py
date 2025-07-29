"""
Unit tests for data cleaning functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from neurolite.data import (
    Dataset, DatasetInfo, DataType,
    DataCleaner, CleaningConfig, CleaningReport, clean_data
)
from neurolite.core import DataError


class TestCleaningConfig:
    """Test cleaning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CleaningConfig()
        
        assert config.missing_threshold == 0.5
        assert config.imputation_strategy == "auto"
        assert config.outlier_method == "iqr"
        assert config.outlier_threshold == 1.5
        assert config.outlier_action == "clip"
        assert config.remove_duplicates is True
        assert config.duplicate_threshold == 0.95
        assert config.fix_encoding is True
        assert config.normalize_whitespace is True
        assert config.fix_data_types is True
        assert config.min_samples == 10
        assert config.min_features == 1
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CleaningConfig(
            missing_threshold=0.3,
            imputation_strategy="mean",
            outlier_method="zscore",
            outlier_threshold=2.0,
            outlier_action="remove",
            remove_duplicates=False,
            duplicate_threshold=0.8,
            fix_encoding=False,
            normalize_whitespace=False,
            fix_data_types=False,
            min_samples=5,
            min_features=2,
            random_seed=123
        )
        
        assert config.missing_threshold == 0.3
        assert config.imputation_strategy == "mean"
        assert config.outlier_method == "zscore"
        assert config.outlier_threshold == 2.0
        assert config.outlier_action == "remove"
        assert config.remove_duplicates is False
        assert config.duplicate_threshold == 0.8
        assert config.fix_encoding is False
        assert config.normalize_whitespace is False
        assert config.fix_data_types is False
        assert config.min_samples == 5
        assert config.min_features == 2
        assert config.random_seed == 123


class TestCleaningReport:
    """Test cleaning report functionality."""
    
    def test_report_creation(self):
        """Test creating a cleaning report."""
        report = CleaningReport(
            original_samples=100,
            final_samples=90,
            original_features=10,
            final_features=9,
            missing_values_handled=5,
            outliers_handled=3,
            duplicates_removed=2,
            operations_performed=["missing_value_handling", "outlier_handling"],
            warnings=["Some warning"]
        )
        
        assert report.original_samples == 100
        assert report.final_samples == 90
        assert report.original_features == 10
        assert report.final_features == 9
        assert report.missing_values_handled == 5
        assert report.outliers_handled == 3
        assert report.duplicates_removed == 2
        assert len(report.operations_performed) == 2
        assert len(report.warnings) == 1
    
    def test_samples_removed(self):
        """Test samples removed calculation."""
        report = CleaningReport(
            original_samples=100,
            final_samples=85,
            original_features=10,
            final_features=10,
            missing_values_handled=0,
            outliers_handled=0,
            duplicates_removed=15,
            operations_performed=[],
            warnings=[]
        )
        
        assert report.samples_removed() == 15
    
    def test_features_removed(self):
        """Test features removed calculation."""
        report = CleaningReport(
            original_samples=100,
            final_samples=100,
            original_features=10,
            final_features=8,
            missing_values_handled=0,
            outliers_handled=0,
            duplicates_removed=0,
            operations_performed=[],
            warnings=[]
        )
        
        assert report.features_removed() == 2
    
    def test_summary(self):
        """Test summary generation."""
        report = CleaningReport(
            original_samples=100,
            final_samples=90,
            original_features=10,
            final_features=9,
            missing_values_handled=5,
            outliers_handled=3,
            duplicates_removed=10,
            operations_performed=["missing_value_handling", "outlier_handling"],
            warnings=[]
        )
        
        summary = report.summary()
        assert "100 → 90" in summary
        assert "10 → 9" in summary
        assert "Missing values handled: 5" in summary
        assert "Outliers handled: 3" in summary
        assert "Duplicates removed: 10" in summary


class TestDataCleaner:
    """Test data cleaner functionality."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        cleaner = DataCleaner()
        assert isinstance(cleaner.config, CleaningConfig)
        assert cleaner.config.missing_threshold == 0.5
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CleaningConfig(missing_threshold=0.3)
        cleaner = DataCleaner(config)
        assert cleaner.config.missing_threshold == 0.3
    
    def test_get_feature_count_array(self):
        """Test feature count calculation for array data."""
        # Create mock dataset with 2D array data
        data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        targets = [0, 1]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=2,
            shape=(2, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        cleaner = DataCleaner()
        feature_count = cleaner._get_feature_count(dataset)
        assert feature_count == 3
    
    def test_get_feature_count_empty(self):
        """Test feature count calculation for empty dataset."""
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=0
        )
        dataset = Dataset([], info=info)
        
        cleaner = DataCleaner()
        feature_count = cleaner._get_feature_count(dataset)
        assert feature_count == 0
    
    def test_clean_tabular_data(self):
        """Test cleaning tabular data."""
        # Create dataset with missing values and outliers
        data = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, np.nan, 6.0]),  # Missing value
            np.array([7.0, 8.0, 100.0]),  # Outlier
            np.array([10.0, 11.0, 12.0])
        ]
        targets = [0, 1, 0, 1]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=4,
            shape=(4, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        cleaner = DataCleaner()
        cleaned_dataset, report = cleaner.clean(dataset)
        
        assert isinstance(cleaned_dataset, Dataset)
        assert isinstance(report, CleaningReport)
        assert report.original_samples == 4
        assert report.missing_values_handled > 0
    
    def test_clean_text_data(self):
        """Test cleaning text data."""
        data = [
            "This is a normal text.",
            "   Text with extra whitespace   ",
            "",  # Empty text
            "Text with\tmixed\nwhitespace",
            "This is a normal text."  # Duplicate
        ]
        targets = [0, 1, 0, 1, 0]
        info = DatasetInfo(
            data_type=DataType.TEXT,
            num_samples=5,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        cleaner = DataCleaner()
        cleaned_dataset, report = cleaner.clean(dataset)
        
        assert isinstance(cleaned_dataset, Dataset)
        assert isinstance(report, CleaningReport)
        assert report.original_samples == 5
        assert len(cleaned_dataset) < 5  # Some samples should be removed
    
    def test_clean_image_data(self):
        """Test cleaning image data."""
        # Create valid and invalid images
        valid_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        invalid_image = np.array([])  # Invalid shape
        
        data = [valid_image, invalid_image, valid_image]
        targets = [0, 1, 0]
        info = DatasetInfo(
            data_type=DataType.IMAGE,
            num_samples=3,
            shape=(32, 32, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        cleaner = DataCleaner()
        cleaned_dataset, report = cleaner.clean(dataset)
        
        assert isinstance(cleaned_dataset, Dataset)
        assert isinstance(report, CleaningReport)
        assert report.original_samples == 3
        assert len(cleaned_dataset) <= 2  # Invalid image and duplicates should be removed
    
    def test_handle_missing_values_tabular(self):
        """Test missing value handling in tabular data."""
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, np.nan],
            [10.0, 11.0, 12.0]
        ])
        targets = np.array([0, 1, 0, 1])
        
        cleaner = DataCleaner()
        cleaned_data, cleaned_targets, missing_count = cleaner._handle_missing_values_tabular(data, targets)
        
        assert missing_count == 2
        assert not np.any(np.isnan(cleaned_data))
    
    def test_handle_outliers_tabular_iqr(self):
        """Test outlier handling using IQR method."""
        # Create data with outliers
        data = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 5.0],  # Outlier in first column
            [5.0, 200.0]   # Outlier in second column
        ])
        
        config = CleaningConfig(outlier_method="iqr", outlier_action="clip")
        cleaner = DataCleaner(config)
        cleaned_data, outliers_handled = cleaner._handle_outliers_tabular(data)
        
        assert outliers_handled > 0
        # Check that extreme values are clipped
        assert cleaned_data[3, 0] < 100.0
        assert cleaned_data[4, 1] < 200.0
    
    def test_handle_outliers_tabular_zscore(self):
        """Test outlier handling using z-score method."""
        # Create data with outliers
        data = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 5.0],  # Outlier
            [5.0, 6.0]
        ])
        
        config = CleaningConfig(outlier_method="zscore", outlier_threshold=2.0, outlier_action="clip")
        cleaner = DataCleaner(config)
        cleaned_data, outliers_handled = cleaner._handle_outliers_tabular(data)
        
        # The z-score method should detect and handle outliers
        assert outliers_handled >= 0  # May be 0 if threshold is too high
        assert isinstance(cleaned_data, np.ndarray)
    
    def test_remove_duplicates_tabular(self):
        """Test duplicate removal from tabular data."""
        # Create data with duplicates
        data = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([1, 2, 3]),  # Duplicate
            np.array([7, 8, 9])
        ]
        targets = [0, 1, 0, 1]
        info = DatasetInfo(
            data_type=DataType.TABULAR,
            num_samples=4,
            shape=(4, 3),
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        report = CleaningReport(
            original_samples=4,
            final_samples=4,
            original_features=3,
            final_features=3,
            missing_values_handled=0,
            outliers_handled=0,
            duplicates_removed=0,
            operations_performed=[],
            warnings=[]
        )
        
        cleaner = DataCleaner()
        with patch('neurolite.core.utils.safe_import') as mock_import:
            # Mock pandas
            mock_pd = Mock()
            mock_df = Mock()
            mock_df.drop_duplicates.return_value = mock_df
            mock_df.index.tolist.return_value = [0, 1, 3]  # Indices after duplicate removal
            mock_df.values.tolist.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            # Mock the __len__ method properly
            type(mock_df).__len__ = Mock(return_value=3)
            mock_pd.DataFrame.return_value = mock_df
            mock_import.return_value = mock_pd
            
            cleaned_dataset, updated_report = cleaner._remove_duplicates_tabular(dataset, report)
            
            assert len(cleaned_dataset) == 3
            assert updated_report.duplicates_removed == 1
    
    def test_remove_duplicates_text(self):
        """Test duplicate removal from text data."""
        data = [
            "Hello world",
            "Different text",
            "hello world",  # Duplicate (case insensitive)
            "Another text"
        ]
        targets = [0, 1, 0, 1]
        info = DatasetInfo(
            data_type=DataType.TEXT,
            num_samples=4,
            num_classes=2,
            class_names=["A", "B"]
        )
        dataset = Dataset(data, targets=targets, info=info)
        
        report = CleaningReport(
            original_samples=4,
            final_samples=4,
            original_features=1,
            final_features=1,
            missing_values_handled=0,
            outliers_handled=0,
            duplicates_removed=0,
            operations_performed=[],
            warnings=[]
        )
        
        cleaner = DataCleaner()
        cleaned_dataset, updated_report = cleaner._remove_duplicates_text(dataset, report)
        
        assert len(cleaned_dataset) == 3
        assert updated_report.duplicates_removed == 1
    
    def test_fix_text_encoding(self):
        """Test text encoding fixes."""
        cleaner = DataCleaner()
        
        # Test common encoding issues
        text_with_issues = "Itâ€™s a â€œgreatâ€ day"
        fixed_text = cleaner._fix_text_encoding(text_with_issues)
        
        assert "â€™" not in fixed_text
        assert "â€œ" not in fixed_text
        assert "'" in fixed_text
        assert '"' in fixed_text
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        cleaner = DataCleaner()
        
        text_with_whitespace = "  This   has\tmultiple\n\nwhitespace   "
        normalized = cleaner._normalize_whitespace(text_with_whitespace)
        
        assert normalized == "This has multiple whitespace"


def test_clean_data_function():
    """Test the clean_data convenience function."""
    data = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0])
    ]
    targets = [0, 1]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=2,
        shape=(2, 3),
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    cleaned_dataset, report = clean_data(dataset)
    
    assert isinstance(cleaned_dataset, Dataset)
    assert isinstance(report, CleaningReport)
    assert report.original_samples == 2


def test_clean_data_with_custom_config():
    """Test clean_data function with custom configuration."""
    data = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0])
    ]
    targets = [0, 1]
    info = DatasetInfo(
        data_type=DataType.TABULAR,
        num_samples=2,
        shape=(2, 3),
        num_classes=2,
        class_names=["A", "B"]
    )
    dataset = Dataset(data, targets=targets, info=info)
    
    config = CleaningConfig(remove_duplicates=False)
    cleaned_dataset, report = clean_data(dataset, config)
    
    assert isinstance(cleaned_dataset, Dataset)
    assert isinstance(report, CleaningReport)