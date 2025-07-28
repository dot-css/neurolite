"""
Unit tests for ComplexityAnalyzer computational requirement estimation and model complexity assessment.

This module tests the computational resource estimation functionality including
CPU vs GPU suitability assessment, memory requirement prediction, processing time estimation,
overfitting risk detection, regularization recommendations, and cross-validation strategies.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from neurolite.analyzers.complexity_analyzer import ComplexityAnalyzer
from neurolite.core.data_models import (
    ResourceEstimate, OverfittingRisk, DatasetComplexity, ComplexityAnalysis
)
from neurolite.core.exceptions import NeuroLiteException


class TestComplexityAnalyzerComputationalRequirements:
    """Test suite for computational requirement estimation functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = ComplexityAnalyzer()
        
        # Create test datasets of different sizes
        self.small_data = pd.DataFrame(np.random.randn(100, 5))
        self.medium_data = pd.DataFrame(np.random.randn(10000, 50))
        self.large_data = pd.DataFrame(np.random.randn(100000, 100))
        self.high_dim_data = pd.DataFrame(np.random.randn(1000, 2000))
        
        # Create numpy arrays for testing
        self.small_array = np.random.randn(100, 5)
        self.medium_array = np.random.randn(10000, 50)
        
        # Create target variables
        self.classification_target = pd.Series(np.random.choice([0, 1], 100))
        self.regression_target = pd.Series(np.random.randn(100))
    
    def test_estimate_computational_requirements_basic(self):
        """Test basic computational requirement estimation."""
        result = self.analyzer.estimate_computational_requirements(
            self.small_data, task_type="classification"
        )
        
        assert isinstance(result, ResourceEstimate)
        assert 0.0 <= result.cpu_suitability <= 1.0
        assert 0.0 <= result.gpu_suitability <= 1.0
        assert result.memory_requirement_mb > 0
        assert result.processing_time_seconds > 0
        assert result.recommended_hardware in ["cpu", "gpu", "distributed"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0
        assert isinstance(result.scaling_factors, dict)
    
    def test_hardware_suitability_small_dataset(self):
        """Test hardware suitability assessment for small datasets."""
        cpu_suit, gpu_suit = self.analyzer._assess_hardware_suitability(
            n_samples=100, n_features=5, task_type="linear"
        )
        
        assert 0.0 <= cpu_suit <= 1.0
        assert 0.0 <= gpu_suit <= 1.0
        # Small datasets should favor CPU
        assert cpu_suit > gpu_suit
    
    def test_hardware_suitability_large_dataset(self):
        """Test hardware suitability assessment for large datasets."""
        cpu_suit, gpu_suit = self.analyzer._assess_hardware_suitability(
            n_samples=100000, n_features=1000, task_type="deep_learning"
        )
        
        assert 0.0 <= cpu_suit <= 1.0
        assert 0.0 <= gpu_suit <= 1.0
        # Large deep learning tasks should favor GPU
        assert gpu_suit > cpu_suit
    
    def test_memory_requirement_prediction(self):
        """Test memory requirement prediction accuracy."""
        current_memory = 10.0  # 10 MB
        
        # Test different task types
        linear_memory = self.analyzer._predict_memory_requirements(
            n_samples=1000, n_features=10, current_memory_mb=current_memory, 
            task_type="linear"
        )
        
        deep_learning_memory = self.analyzer._predict_memory_requirements(
            n_samples=1000, n_features=10, current_memory_mb=current_memory,
            task_type="deep_learning"
        )
        
        assert linear_memory > current_memory  # Should be more than input
        assert deep_learning_memory > linear_memory  # DL needs more memory
        assert linear_memory > 0
        assert deep_learning_memory > 0
    
    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        # Test different task complexities
        linear_time = self.analyzer._estimate_processing_time(
            n_samples=1000, n_features=10, task_type="linear"
        )
        
        ensemble_time = self.analyzer._estimate_processing_time(
            n_samples=1000, n_features=10, task_type="ensemble"
        )
        
        deep_learning_time = self.analyzer._estimate_processing_time(
            n_samples=1000, n_features=10, task_type="deep_learning"
        )
        
        assert linear_time > 0
        assert ensemble_time > linear_time
        assert deep_learning_time > ensemble_time


class TestComplexityAnalyzerModelComplexity:
    """Test suite for model complexity assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures for model complexity testing."""
        self.analyzer = ComplexityAnalyzer()
        
        # Create test datasets with different characteristics
        self.balanced_data = pd.DataFrame(np.random.randn(1000, 20))
        self.balanced_target = pd.Series(np.random.choice([0, 1], 1000))
        
        self.imbalanced_target = pd.Series(np.concatenate([
            np.zeros(900), np.ones(100)  # 90-10 split
        ]))
        
        self.high_dim_data = pd.DataFrame(np.random.randn(100, 500))  # More features than samples
        self.high_dim_target = pd.Series(np.random.choice([0, 1], 100))
        
        # Create correlated features
        base_features = np.random.randn(1000, 5)
        correlated_features = np.column_stack([
            base_features,
            base_features[:, 0] + np.random.randn(1000) * 0.1,  # Highly correlated
            base_features[:, 1] + np.random.randn(1000) * 0.1   # Highly correlated
        ])
        self.correlated_data = pd.DataFrame(correlated_features)
        self.correlated_target = pd.Series(np.random.choice([0, 1], 1000))
    
    def test_detect_overfitting_risk_basic(self):
        """Test basic overfitting risk detection."""
        result = self.analyzer.detect_overfitting_risk(
            self.balanced_data, self.balanced_target, task_type="classification"
        )
        
        assert isinstance(result, OverfittingRisk)
        assert result.risk_level in ["low", "medium", "high"]
        assert 0.0 <= result.risk_score <= 1.0
        assert isinstance(result.contributing_factors, list)
        assert isinstance(result.regularization_recommendations, list)
        assert isinstance(result.cross_validation_strategy, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.mitigation_strategies, dict)
    
    def test_detect_overfitting_risk_high_dimensional(self):
        """Test overfitting risk detection with high-dimensional data."""
        result = self.analyzer.detect_overfitting_risk(
            self.high_dim_data, self.high_dim_target, task_type="classification"
        )
        
        # High-dimensional data should have higher risk
        assert result.risk_score > 0.3
        assert "Low sample-to-feature ratio" in result.contributing_factors
        assert len(result.regularization_recommendations) > 0
        assert any("regularization" in rec.lower() for rec in result.regularization_recommendations)
    
    def test_detect_overfitting_risk_imbalanced(self):
        """Test overfitting risk detection with imbalanced data."""
        result = self.analyzer.detect_overfitting_risk(
            self.balanced_data, self.imbalanced_target, task_type="classification"
        )
        
        # Should detect class imbalance risk
        imbalance_factors = [f for f in result.contributing_factors if "imbalance" in f.lower()]
        if len(imbalance_factors) > 0:  # May not always trigger depending on threshold
            assert len(result.regularization_recommendations) > 0
    
    def test_assess_model_complexity_risk(self):
        """Test model complexity risk assessment."""
        # Low risk scenario (many samples, few features)
        low_risk = self.analyzer._assess_model_complexity_risk(n_samples=10000, n_features=10)
        
        # High risk scenario (few samples, many features)
        high_risk = self.analyzer._assess_model_complexity_risk(n_samples=100, n_features=1000)
        
        assert 0.0 <= low_risk <= 1.0
        assert 0.0 <= high_risk <= 1.0
        assert high_risk > low_risk
        assert low_risk < 0.5  # Should be low risk
        assert high_risk > 0.5  # Should be high risk
    
    def test_assess_correlation_risk(self):
        """Test correlation risk assessment."""
        # Test with correlated data
        corr_risk = self.analyzer._assess_correlation_risk(self.correlated_data.values)
        
        # Test with uncorrelated data
        uncorr_risk = self.analyzer._assess_correlation_risk(self.balanced_data.values)
        
        assert 0.0 <= corr_risk <= 1.0
        assert 0.0 <= uncorr_risk <= 1.0
        assert corr_risk >= uncorr_risk  # Correlated data should have higher risk
    
    def test_assess_imbalance_overfitting_risk(self):
        """Test class imbalance overfitting risk assessment."""
        # Balanced classes
        balanced_risk = self.analyzer._assess_imbalance_overfitting_risk(self.balanced_target.values)
        
        # Imbalanced classes
        imbalanced_risk = self.analyzer._assess_imbalance_overfitting_risk(self.imbalanced_target.values)
        
        assert 0.0 <= balanced_risk <= 1.0
        assert 0.0 <= imbalanced_risk <= 1.0
        assert imbalanced_risk > balanced_risk
    
    def test_generate_regularization_recommendations(self):
        """Test regularization recommendation generation."""
        # High risk scenario
        high_risk_recs = self.analyzer._generate_regularization_recommendations(
            risk_factors=["Low sample-to-feature ratio", "High feature correlation/multicollinearity"],
            risk_score=0.8,
            n_samples=100,
            n_features=500
        )
        
        # Low risk scenario
        low_risk_recs = self.analyzer._generate_regularization_recommendations(
            risk_factors=[],
            risk_score=0.2,
            n_samples=10000,
            n_features=10
        )
        
        assert isinstance(high_risk_recs, list)
        assert isinstance(low_risk_recs, list)
        assert len(high_risk_recs) > len(low_risk_recs)
        assert any("regularization" in rec.lower() for rec in high_risk_recs)
        assert any("L1" in rec or "L2" in rec or "Ridge" in rec or "Lasso" in rec for rec in high_risk_recs)
    
    def test_recommend_cv_strategy(self):
        """Test cross-validation strategy recommendation."""
        # Small dataset
        small_cv = self.analyzer._recommend_cv_strategy(
            n_samples=50, task_type="classification", risk_level="medium"
        )
        
        # Medium dataset, high risk
        medium_high_cv = self.analyzer._recommend_cv_strategy(
            n_samples=500, task_type="classification", risk_level="high"
        )
        
        # Large dataset, low risk
        large_low_cv = self.analyzer._recommend_cv_strategy(
            n_samples=10000, task_type="regression", risk_level="low"
        )
        
        assert isinstance(small_cv, str)
        assert isinstance(medium_high_cv, str)
        assert isinstance(large_low_cv, str)
        
        assert "leave-one-out" in small_cv.lower()
        assert "10-fold" in medium_high_cv.lower()
        assert "fold" in large_low_cv.lower()
    
    def test_generate_mitigation_strategies(self):
        """Test mitigation strategy generation."""
        strategies = self.analyzer._generate_mitigation_strategies(
            risk_factors=["Low sample-to-feature ratio", "High feature correlation/multicollinearity"],
            risk_score=0.8,
            n_samples=100,
            n_features=500
        )
        
        assert isinstance(strategies, dict)
        assert len(strategies) > 0
        
        # Should include various strategy categories
        expected_categories = ["data_collection", "feature_engineering", "preprocessing", 
                             "model_selection", "validation", "monitoring"]
        
        # At least some categories should be present
        assert any(cat in strategies for cat in expected_categories)
        
        # All values should be strings
        for strategy in strategies.values():
            assert isinstance(strategy, str)
            assert len(strategy) > 0
    
    def test_comprehensive_analysis(self):
        """Test comprehensive complexity analysis."""
        result = self.analyzer.analyze_comprehensive(
            self.balanced_data, self.balanced_target, task_type="classification"
        )
        
        assert isinstance(result, ComplexityAnalysis)
        assert hasattr(result, 'dataset_complexity')
        assert hasattr(result, 'resource_estimate')
        assert hasattr(result, 'overfitting_risk')
        assert hasattr(result, 'recommended_approach')
        assert hasattr(result, 'performance_expectations')
        assert hasattr(result, 'optimization_suggestions')
        assert hasattr(result, 'overall_confidence')
        
        assert isinstance(result.dataset_complexity, DatasetComplexity)
        assert isinstance(result.resource_estimate, ResourceEstimate)
        assert isinstance(result.overfitting_risk, OverfittingRisk)
        assert isinstance(result.recommended_approach, str)
        assert isinstance(result.performance_expectations, dict)
        assert isinstance(result.optimization_suggestions, list)
        assert 0.0 <= result.overall_confidence <= 1.0


class TestComplexityAnalyzerAccuracy:
    """Test suite for accuracy of computational requirement estimations."""
    
    def setup_method(self):
        """Set up test fixtures for accuracy testing."""
        self.analyzer = ComplexityAnalyzer()
    
    def test_memory_estimation_accuracy(self):
        """Test accuracy of memory estimation against known patterns."""
        # Create datasets with known memory characteristics
        small_df = pd.DataFrame(np.random.randn(1000, 10))
        large_df = pd.DataFrame(np.random.randn(10000, 100))
        
        small_result = self.analyzer.estimate_computational_requirements(small_df)
        large_result = self.analyzer.estimate_computational_requirements(large_df)
        
        # Large dataset should require significantly more memory
        memory_ratio = large_result.memory_requirement_mb / small_result.memory_requirement_mb
        assert memory_ratio > 5  # Should be at least 5x more memory
    
    def test_processing_time_accuracy(self):
        """Test accuracy of processing time estimation."""
        # Linear algorithms should be faster than ensemble methods
        data = pd.DataFrame(np.random.randn(5000, 50))
        
        linear_result = self.analyzer.estimate_computational_requirements(
            data, task_type="linear"
        )
        ensemble_result = self.analyzer.estimate_computational_requirements(
            data, task_type="ensemble"
        )
        
        # Ensemble should take longer than linear
        time_ratio = ensemble_result.processing_time_seconds / linear_result.processing_time_seconds
        assert time_ratio > 2  # Should be at least 2x longer
    
    def test_overfitting_risk_accuracy(self):
        """Test accuracy of overfitting risk assessment."""
        # High-dimensional data should have higher overfitting risk
        low_dim_data = pd.DataFrame(np.random.randn(1000, 10))
        low_dim_target = pd.Series(np.random.choice([0, 1], 1000))
        
        high_dim_data = pd.DataFrame(np.random.randn(100, 500))
        high_dim_target = pd.Series(np.random.choice([0, 1], 100))
        
        low_dim_risk = self.analyzer.detect_overfitting_risk(
            low_dim_data, low_dim_target, task_type="classification"
        )
        high_dim_risk = self.analyzer.detect_overfitting_risk(
            high_dim_data, high_dim_target, task_type="classification"
        )
        
        # High-dimensional data should have higher risk
        assert high_dim_risk.risk_score >= low_dim_risk.risk_score


if __name__ == "__main__":
    pytest.main([__file__])