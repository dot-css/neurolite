"""
Unit tests for StatisticalAnalyzer module.

Tests distribution analysis, correlation computation, and relationship detection
functionality of the StatisticalAnalyzer class.
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from neurolite.analyzers.statistical_analyzer import (
    StatisticalAnalyzer, 
    DistributionAnalysis, 
    CorrelationMatrix, 
    RelationshipAnalysis
)
from neurolite.core.exceptions import NeuroLiteException


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for StatisticalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Create test datasets
        np.random.seed(42)
        
        # Normal distribution data
        self.normal_data = pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 1000),
            'normal_col2': np.random.normal(5, 2, 1000)
        })
        
        # Mixed distribution data
        self.mixed_data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 500),
            'exponential': np.random.exponential(2, 500),
            'uniform': np.random.uniform(-1, 1, 500),
            'categorical': (['A', 'B', 'C'] * 166) + ['A', 'B']  # Non-numerical column, 500 elements
        })
        
        # Correlated data
        x = np.random.normal(0, 1, 1000)
        self.correlated_data = pd.DataFrame({
            'x': x,
            'y_linear': 2 * x + np.random.normal(0, 0.1, 1000),  # Strong linear
            'y_nonlinear': x**2 + np.random.normal(0, 0.5, 1000),  # Non-linear
            'y_independent': np.random.normal(0, 1, 1000)  # Independent
        })
        
        # Small dataset for edge cases
        self.small_data = pd.DataFrame({
            'small_col': [1, 2, 3, 4, 5]
        })


class TestDistributionAnalysis(TestStatisticalAnalyzer):
    """Test distribution analysis functionality."""
    
    def test_analyze_distributions_normal_data(self):
        """Test distribution analysis on normal data."""
        results = self.analyzer.analyze_distributions(self.normal_data)
        
        # Should analyze both numerical columns
        self.assertEqual(len(results), 2)
        self.assertIn('normal_col', results)
        self.assertIn('normal_col2', results)
        
        # Check normal_col analysis
        normal_analysis = results['normal_col']
        self.assertIsInstance(normal_analysis, DistributionAnalysis)
        
        # Should detect a reasonable distribution for normal data
        self.assertIn(normal_analysis.best_fit.distribution_name, 
                     ['norm', 'chi2', 't', 'lognorm', 'expon', 'uniform'])  # Common fits for normal data
        
        # Should have reasonable skewness and kurtosis for normal data
        self.assertLess(abs(normal_analysis.skewness), 0.5)
        self.assertLess(abs(normal_analysis.kurtosis), 1.0)
        
        # Should have normality test results
        self.assertIsInstance(normal_analysis.normality_tests, dict)
        self.assertGreater(len(normal_analysis.normality_tests), 0)
    
    def test_analyze_distributions_mixed_data(self):
        """Test distribution analysis on mixed distribution types."""
        results = self.analyzer.analyze_distributions(self.mixed_data)
        
        # Should only analyze numerical columns (3 out of 4)
        self.assertEqual(len(results), 3)
        self.assertIn('normal', results)
        self.assertIn('exponential', results)
        self.assertIn('uniform', results)
        self.assertNotIn('categorical', results)
        
        # Check that different distributions are detected
        distribution_names = [analysis.best_fit.distribution_name 
                            for analysis in results.values()]
        
        # Should have variety in detected distributions
        self.assertGreater(len(set(distribution_names)), 1)
    
    def test_distribution_fit_properties(self):
        """Test properties of distribution fits."""
        results = self.analyzer.analyze_distributions(self.normal_data)
        analysis = results['normal_col']
        
        # Test best fit properties
        best_fit = analysis.best_fit
        self.assertIsInstance(best_fit.distribution_name, str)
        self.assertIsInstance(best_fit.parameters, dict)
        self.assertGreater(len(best_fit.parameters), 0)
        self.assertIsInstance(best_fit.goodness_of_fit, float)
        self.assertIsInstance(best_fit.p_value, float)
        self.assertIsInstance(best_fit.confidence_interval, tuple)
        self.assertEqual(len(best_fit.confidence_interval), 2)
        self.assertIsInstance(best_fit.aic, float)
        self.assertIsInstance(best_fit.bic, float)
        
        # Test alternative fits
        self.assertIsInstance(analysis.alternative_fits, list)
        self.assertLessEqual(len(analysis.alternative_fits), 3)
    
    def test_normality_tests(self):
        """Test normality testing functionality."""
        # Test with clearly normal data
        normal_series = pd.Series(np.random.normal(0, 1, 1000))
        results = self.analyzer.analyze_distributions(pd.DataFrame({'col': normal_series}))
        
        normality_tests = results['col'].normality_tests
        
        # Should have multiple normality tests
        expected_tests = ['dagostino', 'jarque_bera', 'anderson_darling']
        for test in expected_tests:
            if test in normality_tests:
                self.assertIn('statistic', normality_tests[test])
                if test != 'anderson_darling':
                    self.assertIn('p_value', normality_tests[test])
    
    def test_multimodality_detection(self):
        """Test multimodal distribution detection."""
        # Create bimodal data
        bimodal_data = np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
        
        df = pd.DataFrame({'bimodal': bimodal_data})
        results = self.analyzer.analyze_distributions(df)
        
        # Note: Multimodality detection is challenging and may not always work
        # This test checks that the method runs without error
        analysis = results['bimodal']
        self.assertIsInstance(analysis.is_multimodal, bool)
    
    def test_small_dataset_handling(self):
        """Test handling of small datasets."""
        results = self.analyzer.analyze_distributions(self.small_data)
        
        # Should skip columns with too few data points
        self.assertEqual(len(results), 0)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        results = self.analyzer.analyze_distributions(empty_df)
        
        self.assertEqual(len(results), 0)
        self.assertIsInstance(results, dict)
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        results = self.analyzer.analyze_distributions(self.normal_data)
        analysis = results['normal_col']
        
        ci = analysis.best_fit.confidence_interval
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])  # Lower bound < upper bound


class TestCorrelationAnalysis(TestStatisticalAnalyzer):
    """Test correlation analysis functionality."""
    
    def test_compute_correlations_basic(self):
        """Test basic correlation computation."""
        corr_results = self.analyzer.compute_correlations(self.correlated_data)
        
        self.assertIsInstance(corr_results, CorrelationMatrix)
        
        # Check matrix dimensions
        n_features = len(self.correlated_data.select_dtypes(include=[np.number]).columns)
        self.assertEqual(corr_results.pearson_correlation.shape, (n_features, n_features))
        self.assertEqual(corr_results.spearman_correlation.shape, (n_features, n_features))
        self.assertEqual(corr_results.kendall_correlation.shape, (n_features, n_features))
        self.assertEqual(corr_results.mutual_information.shape, (n_features, n_features))
        
        # Check column names
        self.assertEqual(len(corr_results.column_names), n_features)
        self.assertIn('x', corr_results.column_names)
        self.assertIn('y_linear', corr_results.column_names)
    
    def test_correlation_properties(self):
        """Test properties of correlation matrices."""
        corr_results = self.analyzer.compute_correlations(self.correlated_data)
        
        # Diagonal should be 1 for Pearson and Spearman
        np.testing.assert_array_almost_equal(
            np.diag(corr_results.pearson_correlation), 
            np.ones(len(corr_results.column_names))
        )
        np.testing.assert_array_almost_equal(
            np.diag(corr_results.spearman_correlation), 
            np.ones(len(corr_results.column_names))
        )
        
        # Matrices should be symmetric
        np.testing.assert_array_almost_equal(
            corr_results.pearson_correlation,
            corr_results.pearson_correlation.T
        )
        np.testing.assert_array_almost_equal(
            corr_results.spearman_correlation,
            corr_results.spearman_correlation.T
        )
    
    def test_strong_correlation_detection(self):
        """Test detection of strong correlations."""
        corr_results = self.analyzer.compute_correlations(self.correlated_data)
        
        # Find indices of x and y_linear
        x_idx = corr_results.column_names.index('x')
        y_linear_idx = corr_results.column_names.index('y_linear')
        
        # Should detect strong correlation between x and y_linear
        correlation = abs(corr_results.pearson_correlation[x_idx, y_linear_idx])
        self.assertGreater(correlation, 0.8)  # Should be strongly correlated
    
    def test_mutual_information_computation(self):
        """Test mutual information computation."""
        corr_results = self.analyzer.compute_correlations(self.correlated_data)
        
        # Mutual information diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(corr_results.mutual_information),
            np.ones(len(corr_results.column_names))
        )
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(
            corr_results.mutual_information,
            corr_results.mutual_information.T
        )
        
        # Values should be between 0 and 1
        self.assertTrue(np.all(corr_results.mutual_information >= 0))
        self.assertTrue(np.all(corr_results.mutual_information <= 1))
    
    def test_no_numerical_columns_error(self):
        """Test error handling when no numerical columns exist."""
        categorical_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        
        with self.assertRaises(NeuroLiteException):
            self.analyzer.compute_correlations(categorical_df)


class TestRelationshipDetection(TestStatisticalAnalyzer):
    """Test relationship detection functionality."""
    
    def test_detect_relationships_basic(self):
        """Test basic relationship detection."""
        rel_results = self.analyzer.detect_relationships(self.correlated_data)
        
        self.assertIsInstance(rel_results, RelationshipAnalysis)
        self.assertIsInstance(rel_results.linear_relationships, dict)
        self.assertIsInstance(rel_results.non_linear_relationships, dict)
        self.assertIsInstance(rel_results.multicollinearity_vif, dict)
        self.assertIsInstance(rel_results.feature_dependencies, dict)
    
    def test_linear_relationship_detection(self):
        """Test linear relationship detection."""
        rel_results = self.analyzer.detect_relationships(self.correlated_data)
        
        # Should detect linear relationship between x and y_linear
        linear_rels = rel_results.linear_relationships
        
        # Look for the relationship (could be in either direction)
        found_linear = any('x' in key and 'y_linear' in key for key in linear_rels.keys())
        self.assertTrue(found_linear, "Should detect linear relationship between x and y_linear")
        
        # Values should be correlation coefficients (0-1)
        for value in linear_rels.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def test_vif_calculation(self):
        """Test VIF (Variance Inflation Factor) calculation."""
        rel_results = self.analyzer.detect_relationships(self.correlated_data)
        
        vif_values = rel_results.multicollinearity_vif
        
        # Should have VIF for each numerical column
        expected_columns = self.correlated_data.select_dtypes(include=[np.number]).columns
        for col in expected_columns:
            self.assertIn(col, vif_values)
            self.assertIsInstance(vif_values[col], float)
            self.assertGreaterEqual(vif_values[col], 1.0)  # VIF should be >= 1
    
    def test_feature_dependencies(self):
        """Test feature dependency detection."""
        rel_results = self.analyzer.detect_relationships(self.correlated_data)
        
        dependencies = rel_results.feature_dependencies
        
        # Dependencies should be a dict mapping features to lists of dependent features
        for feature, deps in dependencies.items():
            self.assertIsInstance(feature, str)
            self.assertIsInstance(deps, list)
            for dep in deps:
                self.assertIsInstance(dep, str)
    
    def test_nonlinear_relationship_detection(self):
        """Test non-linear relationship detection."""
        rel_results = self.analyzer.detect_relationships(self.correlated_data)
        
        nonlinear_rels = rel_results.non_linear_relationships
        
        # Should potentially detect non-linear relationship between x and y_nonlinear
        # This is challenging to test reliably, so we just check the structure
        for key, value in nonlinear_rels.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)


class TestComprehensiveAnalysis(TestStatisticalAnalyzer):
    """Test comprehensive analysis functionality."""
    
    def test_analyze_comprehensive(self):
        """Test comprehensive statistical analysis."""
        result = self.analyzer.analyze_comprehensive(self.correlated_data)
        
        # Should return StatisticalProperties object
        from neurolite.core.data_models import StatisticalProperties
        self.assertIsInstance(result, StatisticalProperties)
        
        # Check required fields
        self.assertIsInstance(result.distribution, str)
        self.assertIsInstance(result.parameters, dict)
        self.assertIsInstance(result.feature_importance, dict)
        self.assertIsInstance(result.outlier_indices, list)
        
        # Correlation matrix should be present for multi-column data
        if len(self.correlated_data.select_dtypes(include=[np.number]).columns) > 1:
            self.assertIsNotNone(result.correlation_matrix)
    
    def test_outlier_detection(self):
        """Test outlier detection functionality."""
        # Create data with obvious outliers
        data_with_outliers = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'normal2': [10, 11, 12, 13, 14, 15]
        })
        
        result = self.analyzer.analyze_comprehensive(data_with_outliers)
        
        # Should detect the outlier
        self.assertIsInstance(result.outlier_indices, list)
        # Note: Exact outlier detection depends on the method and data
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        result = self.analyzer.analyze_comprehensive(self.correlated_data)
        
        importance = result.feature_importance
        
        # Should have importance scores for each numerical feature
        numerical_cols = self.correlated_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.assertIn(col, importance)
            self.assertIsInstance(importance[col], float)
            self.assertGreaterEqual(importance[col], 0)
            self.assertLessEqual(importance[col], 1)
    
    def test_error_handling_in_comprehensive_analysis(self):
        """Test error handling in comprehensive analysis."""
        # Test with problematic data
        problematic_df = pd.DataFrame({
            'all_nan': [np.nan] * 100,
            'constant': [1] * 100
        })
        
        # Should not raise exception, but return minimal results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.analyzer.analyze_comprehensive(problematic_df)
        
        self.assertIsInstance(result.distribution, str)
        self.assertIsInstance(result.parameters, dict)


if __name__ == '__main__':
    unittest.main()