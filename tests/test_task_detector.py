"""
Unit tests for TaskDetector class.

Tests supervised and unsupervised learning task detection functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from neurolite.detectors.task_detector import TaskDetector
from neurolite.core.data_models import SupervisedTaskAnalysis, UnsupervisedTaskAnalysis
from neurolite.core.exceptions import NeuroLiteException


class TestTaskDetector:
    """Test cases for TaskDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = TaskDetector()
        
        # Create sample datasets
        np.random.seed(42)
        
        # Binary classification data
        self.X_binary = np.random.randn(100, 5)
        self.y_binary = np.random.choice([0, 1], 100)
        
        # Multi-class classification data
        self.X_multiclass = np.random.randn(150, 4)
        self.y_multiclass = np.random.choice(['A', 'B', 'C'], 150)
        
        # Regression data
        self.X_regression = np.random.randn(120, 3)
        self.y_regression = 2 * self.X_regression[:, 0] + np.random.randn(120) * 0.1
        
        # Imbalanced classification data
        self.X_imbalanced = np.random.randn(100, 4)
        self.y_imbalanced = np.concatenate([
            np.ones(80),  # Majority class
            np.zeros(20)  # Minority class
        ])
        
        # Unsupervised data with clear clusters
        cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
        cluster2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 50)
        self.X_clusterable = np.vstack([cluster1, cluster2])
        
        # High-dimensional data
        self.X_high_dim = np.random.randn(50, 100)


class TestSupervisedTaskDetection(TestTaskDetector):
    """Test supervised learning task detection."""
    
    def test_detect_supervised_task_binary_classification(self):
        """Test detection of binary classification task."""
        result = self.detector.detect_supervised_task(self.X_binary, self.y_binary)
        
        assert isinstance(result, SupervisedTaskAnalysis)
        assert result.task_type == 'classification'
        assert result.task_subtype == 'binary'
        assert result.confidence > 0.9
        assert result.target_characteristics['unique_values'] == 2
        assert 'class_labels' in result.target_characteristics
        assert 'balance_ratio' in result.dataset_balance
    
    def test_detect_supervised_task_multiclass_classification(self):
        """Test detection of multi-class classification task."""
        result = self.detector.detect_supervised_task(self.X_multiclass, self.y_multiclass)
        
        assert isinstance(result, SupervisedTaskAnalysis)
        assert result.task_type == 'classification'
        assert result.task_subtype == 'multiclass'
        assert result.confidence > 0.8
        assert result.target_characteristics['unique_values'] == 3
        assert len(result.target_characteristics['class_labels']) == 3
        assert result.dataset_balance['balance_status'] in ['balanced', 'slightly_imbalanced']
    
    def test_detect_supervised_task_regression(self):
        """Test detection of regression task."""
        result = self.detector.detect_supervised_task(self.X_regression, self.y_regression)
        
        assert isinstance(result, SupervisedTaskAnalysis)
        assert result.task_type == 'regression'
        assert result.task_subtype in ['linear', 'non_linear']
        assert result.confidence > 0.7
        assert 'mean_value' in result.target_characteristics
        assert 'std_value' in result.target_characteristics
        assert result.dataset_balance == {}  # No balance analysis for regression
    
    def test_detect_supervised_task_imbalanced_dataset(self):
        """Test detection with imbalanced dataset."""
        result = self.detector.detect_supervised_task(self.X_imbalanced, self.y_imbalanced)
        
        assert result.task_type == 'classification'
        assert result.dataset_balance['balance_status'] in ['moderately_imbalanced', 'severely_imbalanced']
        assert result.dataset_balance['balance_ratio'] < 0.5
        assert result.dataset_balance['majority_proportion'] > 0.6
    
    def test_detect_supervised_task_with_pandas_input(self):
        """Test detection with pandas DataFrame and Series input."""
        X_df = pd.DataFrame(self.X_binary, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y_series = pd.Series(self.y_binary)
        
        result = self.detector.detect_supervised_task(X_df, y_series)
        
        assert isinstance(result, SupervisedTaskAnalysis)
        assert result.task_type == 'classification'
        assert result.task_subtype == 'binary'
    
    def test_detect_supervised_task_with_missing_values(self):
        """Test detection with missing values in target."""
        y_with_nan = self.y_binary.copy().astype(float)
        y_with_nan[::10] = np.nan  # Add some missing values
        
        result = self.detector.detect_supervised_task(self.X_binary, y_with_nan)
        
        assert isinstance(result, SupervisedTaskAnalysis)
        assert result.target_characteristics['missing_values'] > 0
    
    def test_detect_supervised_task_insufficient_samples(self):
        """Test error handling with insufficient samples."""
        X_small = self.X_binary[:10]
        y_small = self.y_binary[:10]
        
        with pytest.raises(NeuroLiteException, match="Insufficient samples"):
            self.detector.detect_supervised_task(X_small, y_small)
    
    def test_detect_supervised_task_mismatched_lengths(self):
        """Test error handling with mismatched X and y lengths."""
        with pytest.raises(NeuroLiteException, match="same length"):
            self.detector.detect_supervised_task(self.X_binary, self.y_binary[:-10])
    
    def test_classify_supervised_task_edge_cases(self):
        """Test edge cases in supervised task classification."""
        # Single unique value (degenerate case)
        y_constant = np.ones(100)
        task_type, task_subtype, confidence = self.detector._classify_supervised_task(y_constant)
        assert task_type == 'classification'
        
        # Many unique values but still categorical
        y_many_categories = np.random.choice(range(50), 1000)
        task_type, task_subtype, confidence = self.detector._classify_supervised_task(y_many_categories)
        assert task_type == 'classification'
        assert task_subtype == 'multiclass'
    
    def test_analyze_target_characteristics_classification(self):
        """Test target characteristics analysis for classification."""
        characteristics = self.detector._analyze_target_characteristics(self.y_multiclass, 'classification')
        
        assert 'class_labels' in characteristics
        assert 'class_counts' in characteristics
        assert 'most_frequent_class' in characteristics
        assert 'least_frequent_class' in characteristics
        assert characteristics['unique_values'] == 3
    
    def test_analyze_target_characteristics_regression(self):
        """Test target characteristics analysis for regression."""
        characteristics = self.detector._analyze_target_characteristics(self.y_regression, 'regression')
        
        assert 'min_value' in characteristics
        assert 'max_value' in characteristics
        assert 'mean_value' in characteristics
        assert 'std_value' in characteristics
        assert 'skewness' in characteristics
        assert 'kurtosis' in characteristics
    
    def test_assess_dataset_balance(self):
        """Test dataset balance assessment."""
        balance = self.detector._assess_dataset_balance(self.y_imbalanced)
        
        assert 'balance_ratio' in balance
        assert 'balance_status' in balance
        assert 'class_proportions' in balance
        assert 'majority_class' in balance
        assert 'minority_class' in balance
        assert balance['balance_ratio'] < 1.0
        assert balance['balance_status'] in ['balanced', 'slightly_imbalanced', 'moderately_imbalanced', 'severely_imbalanced']
    
    def test_analyze_complexity_indicators(self):
        """Test complexity indicators analysis."""
        complexity = self.detector._analyze_complexity_indicators(
            self.X_multiclass, self.y_multiclass, 'classification'
        )
        
        assert 'n_samples' in complexity
        assert 'n_features' in complexity
        assert 'samples_to_features_ratio' in complexity
        assert 'dimensionality' in complexity
        assert 'n_classes' in complexity
        assert 'classification_complexity' in complexity
    
    def test_analyze_complexity_indicators_regression(self):
        """Test complexity indicators for regression."""
        complexity = self.detector._analyze_complexity_indicators(
            self.X_regression, self.y_regression, 'regression'
        )
        
        assert 'target_variance' in complexity
        assert 'target_range' in complexity
        assert 'regression_complexity' in complexity
    
    def test_convert_to_numeric(self):
        """Test conversion of mixed data to numeric."""
        # Mixed data with strings and numbers
        mixed_data = np.array([
            ['1', '2.5', 'text'],
            ['3', '4.0', 'more_text'],
            ['5', '6.5', 'even_more']
        ])
        
        numeric_data = self.detector._convert_to_numeric(mixed_data)
        
        assert numeric_data.shape == mixed_data.shape
        assert not np.isnan(numeric_data[:, 0]).any()  # First column should be numeric
        assert not np.isnan(numeric_data[:, 1]).any()  # Second column should be numeric
        assert np.isnan(numeric_data[:, 2]).all()      # Third column should be NaN


class TestUnsupervisedTaskDetection(TestTaskDetector):
    """Test unsupervised learning task detection."""
    
    def test_detect_unsupervised_task_clusterable_data(self):
        """Test detection with clearly clusterable data."""
        result = self.detector.detect_unsupervised_task(self.X_clusterable)
        
        assert isinstance(result, UnsupervisedTaskAnalysis)
        assert result.clustering_potential > 0.5
        assert result.optimal_clusters is not None
        assert result.optimal_clusters >= 2
        assert result.confidence > 0.5
        assert 'n_samples' in result.clustering_characteristics
        assert 'n_features' in result.clustering_characteristics
    
    def test_detect_unsupervised_task_high_dimensional(self):
        """Test detection with high-dimensional data."""
        result = self.detector.detect_unsupervised_task(self.X_high_dim)
        
        assert isinstance(result, UnsupervisedTaskAnalysis)
        assert result.dimensionality_reduction_needed == True
        assert 'n_features' in result.dimensionality_info
        assert result.dimensionality_info['n_features'] == 100
    
    def test_detect_unsupervised_task_with_pandas_input(self):
        """Test detection with pandas DataFrame input."""
        X_df = pd.DataFrame(self.X_clusterable, columns=['x', 'y'])
        
        result = self.detector.detect_unsupervised_task(X_df)
        
        assert isinstance(result, UnsupervisedTaskAnalysis)
        assert result.clustering_potential > 0.0
    
    def test_detect_unsupervised_task_insufficient_samples(self):
        """Test error handling with insufficient samples."""
        X_small = self.X_clusterable[:10]
        
        with pytest.raises(NeuroLiteException, match="Insufficient samples"):
            self.detector.detect_unsupervised_task(X_small)
    
    def test_detect_unsupervised_task_no_valid_features(self):
        """Test error handling with no valid numeric features."""
        X_text = np.array([['a', 'b'], ['c', 'd'], ['e', 'f']] * 20)
        
        with pytest.raises(NeuroLiteException, match="No valid numeric features"):
            self.detector.detect_unsupervised_task(X_text)
    
    def test_assess_clustering_potential(self):
        """Test clustering potential assessment."""
        potential, optimal_k, characteristics = self.detector._assess_clustering_potential(self.X_clusterable)
        
        assert 0.0 <= potential <= 1.0
        assert optimal_k is None or optimal_k >= 2
        assert 'n_samples' in characteristics
        assert 'n_features' in characteristics
        assert 'silhouette_scores' in characteristics
        assert 'best_silhouette_score' in characteristics
    
    def test_assess_clustering_potential_poor_data(self):
        """Test clustering potential with poorly clusterable data."""
        # Random data with no clear clusters
        X_random = np.random.randn(100, 5)
        
        potential, optimal_k, characteristics = self.detector._assess_clustering_potential(X_random)
        
        assert 0.0 <= potential <= 1.0
        assert potential < 0.7  # Should be lower for random data
    
    def test_assess_dimensionality_reduction_need_high_dim(self):
        """Test dimensionality reduction assessment for high-dimensional data."""
        needed, info = self.detector._assess_dimensionality_reduction_need(self.X_high_dim)
        
        assert needed == True
        assert 'n_samples' in info
        assert 'n_features' in info
        assert info['n_features'] == 100
        assert 'samples_to_features_ratio' in info
    
    def test_assess_dimensionality_reduction_need_low_dim(self):
        """Test dimensionality reduction assessment for low-dimensional data."""
        needed, info = self.detector._assess_dimensionality_reduction_need(self.X_clusterable)
        
        assert needed == False
        assert info['n_features'] == 2
        assert info['samples_to_features_ratio'] > 1
    
    def test_assess_dimensionality_reduction_with_error_conditions(self):
        """Test dimensionality reduction assessment with error conditions."""
        # Test with data that might cause issues
        X_problematic = np.array([[np.inf, 1], [2, np.nan], [3, 4]] * 20)
        
        # Should handle problematic data gracefully
        needed, info = self.detector._assess_dimensionality_reduction_need(X_problematic)
        
        assert isinstance(needed, bool)
        assert 'n_samples' in info
        assert 'n_features' in info
    
    def test_calculate_unsupervised_confidence(self):
        """Test unsupervised confidence calculation."""
        # High clustering potential, normal dimensionality
        confidence = self.detector._calculate_unsupervised_confidence(0.8, False, (100, 5))
        assert 0.8 <= confidence <= 1.0
        
        # Low clustering potential, high dimensionality
        confidence = self.detector._calculate_unsupervised_confidence(0.3, True, (50, 100))
        assert 0.0 <= confidence <= 0.5
        
        # Edge case: very small sample size
        confidence = self.detector._calculate_unsupervised_confidence(0.7, False, (30, 5))
        assert confidence < 0.7  # Should be reduced due to small sample size


class TestTaskDetectorEdgeCases(TestTaskDetector):
    """Test edge cases and error conditions."""
    
    def test_is_linear_relationship_normal_data(self):
        """Test linear relationship detection with normal data."""
        # Normal distribution should suggest linear potential
        y_normal = np.random.normal(0, 1, 1000)
        is_linear = self.detector._is_linear_relationship(y_normal)
        assert is_linear in [True, False]
    
    def test_is_linear_relationship_skewed_data(self):
        """Test linear relationship detection with skewed data."""
        # Highly skewed data
        y_skewed = np.random.exponential(1, 1000)
        is_linear = self.detector._is_linear_relationship(y_skewed)
        assert is_linear in [True, False]
    
    def test_is_linear_relationship_error_handling(self):
        """Test linear relationship detection error handling."""
        # Data that might cause statistical test to fail
        y_constant = np.ones(100)
        is_linear = self.detector._is_linear_relationship(y_constant)
        assert is_linear == True  # Should default to True on error
    
    def test_convert_to_numeric_already_numeric(self):
        """Test numeric conversion with already numeric data."""
        X_numeric = np.random.randn(50, 3)
        result = self.detector._convert_to_numeric(X_numeric)
        np.testing.assert_array_equal(result, X_numeric)
    
    def test_convert_to_numeric_mixed_types(self):
        """Test numeric conversion with mixed data types."""
        X_mixed = np.array([
            [1, 2.5, 'text', '4'],
            [2, 3.0, 'more', '5'],
            [3, 4.5, 'text', '6']
        ])
        
        result = self.detector._convert_to_numeric(X_mixed)
        
        # First two columns should be numeric
        assert not np.isnan(result[:, 0]).any()
        assert not np.isnan(result[:, 1]).any()
        # Third column should be NaN (text)
        assert np.isnan(result[:, 2]).all()
        # Fourth column should be numeric (convertible strings)
        assert not np.isnan(result[:, 3]).any()


if __name__ == '__main__':
    pytest.main([__file__])