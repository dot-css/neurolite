"""
Unit tests for PreprocessingRecommender functionality.

Tests the preprocessing recommendation system including scaling, encoding,
and feature engineering recommendations based on data characteristics.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from neurolite.recommenders.preprocessing_recommender import PreprocessingRecommender
from neurolite.core.data_models import (
    ColumnType, TaskIdentification, QualityMetrics,
    ScalingRecommendation, EncodingRecommendation, FeatureEngineeringRecommendation,
    PreprocessingPipeline
)
from neurolite.core.exceptions import NeuroLiteException, InsufficientDataError


class TestPreprocessingRecommender:
    """Test cases for PreprocessingRecommender class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recommender = PreprocessingRecommender()
        
        # Create sample data for testing
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'numeric_normal': np.random.normal(0, 1, 100),
            'numeric_large_range': np.random.uniform(0, 10000, 100),
            'numeric_skewed': np.random.exponential(2, 100),
            'numeric_outliers': np.concatenate([np.random.normal(0, 1, 95), [100, -100, 200, -200, 300]]),
            'categorical_low': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_high': np.random.choice([f'cat_{i}' for i in range(60)], 100),
            'categorical_binary': np.random.choice(['Yes', 'No'], 100),
            'categorical_ordinal': np.random.choice(['Low', 'Medium', 'High'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Create column types
        self.column_types = {
            'numeric_normal': ColumnType('numerical', 'float', 0.9),
            'numeric_large_range': ColumnType('numerical', 'float', 0.9),
            'numeric_skewed': ColumnType('numerical', 'float', 0.9),
            'numeric_outliers': ColumnType('numerical', 'float', 0.9),
            'categorical_low': ColumnType('categorical', 'nominal', 0.9),
            'categorical_high': ColumnType('categorical', 'nominal', 0.9),
            'categorical_binary': ColumnType('categorical', 'nominal', 0.9),
            'categorical_ordinal': ColumnType('categorical', 'ordinal', 0.9),
            'target': ColumnType('categorical', 'nominal', 0.9)
        }
        
        # Create task identification
        self.classification_task = TaskIdentification(
            task_type='classification',
            task_subtype='binary',
            complexity='medium',
            confidence=0.9
        )
        
        self.regression_task = TaskIdentification(
            task_type='regression',
            task_subtype='linear',
            complexity='medium',
            confidence=0.9
        )
        
        # Create quality metrics
        self.quality_metrics = QualityMetrics(
            completeness=0.95,
            consistency=0.90,
            validity=0.85,
            uniqueness=0.80,
            missing_pattern='MCAR',
            duplicate_count=5
        )

    def test_initialization(self):
        """Test PreprocessingRecommender initialization."""
        recommender = PreprocessingRecommender()
        assert recommender.high_cardinality_threshold == 50
        assert recommender.low_cardinality_threshold == 10
        assert recommender.outlier_threshold == 0.05
        assert recommender.skewness_threshold == 1.0
        assert recommender.correlation_threshold == 0.8
        assert recommender.min_samples_for_analysis == 30

    def test_recommend_preprocessing_pipeline_empty_dataframe(self):
        """Test preprocessing pipeline recommendation with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(NeuroLiteException, match="Cannot recommend preprocessing for empty DataFrame"):
            self.recommender.recommend_preprocessing_pipeline(empty_df, {})

    def test_recommend_preprocessing_pipeline_insufficient_data(self):
        """Test preprocessing pipeline recommendation with insufficient data."""
        small_df = pd.DataFrame({'col1': [1, 2, 3]})  # Less than min_samples_for_analysis
        column_types = {'col1': ColumnType('numerical', 'integer', 0.9)}
        
        with pytest.raises(InsufficientDataError):
            self.recommender.recommend_preprocessing_pipeline(small_df, column_types)

    def test_recommend_preprocessing_pipeline_complete(self):
        """Test complete preprocessing pipeline recommendation."""
        pipeline = self.recommender.recommend_preprocessing_pipeline(
            self.sample_df,
            self.column_types,
            self.classification_task,
            self.quality_metrics,
            'target'
        )
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.scaling_recommendations) > 0
        assert len(pipeline.encoding_recommendations) > 0
        assert len(pipeline.pipeline_order) > 0
        assert 0.0 <= pipeline.overall_confidence <= 1.0
        assert pipeline.estimated_processing_time > 0

    def test_recommend_scaling_standardization(self):
        """Test scaling recommendation for normal data."""
        # Test with normal data that should get standardization
        normal_data = pd.DataFrame({'normal_col': np.random.normal(5, 2, 100)})
        column_types = {'normal_col': ColumnType('numerical', 'float', 0.9)}
        
        recommendations = self.recommender._recommend_scaling(normal_data, column_types, self.classification_task)
        
        assert len(recommendations) == 1
        assert recommendations[0].scaling_type == 'standardization'
        assert recommendations[0].confidence > 0.5
        assert 'normal_col' in recommendations[0].affected_columns

    def test_recommend_scaling_normalization(self):
        """Test scaling recommendation for large range data."""
        # Test with large range data that should get normalization
        large_range_data = pd.DataFrame({'large_col': np.random.uniform(0, 10000, 100)})
        column_types = {'large_col': ColumnType('numerical', 'float', 0.9)}
        
        recommendations = self.recommender._recommend_scaling(large_range_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].scaling_type == 'normalization'
        assert recommendations[0].confidence > 0.5

    def test_recommend_scaling_robust(self):
        """Test scaling recommendation for data with outliers."""
        # Test with outlier data that should get robust scaling
        outlier_data = pd.DataFrame({
            'outlier_col': np.concatenate([np.random.normal(0, 1, 95), [100, -100, 200, -200, 300]])
        })
        column_types = {'outlier_col': ColumnType('numerical', 'float', 0.9)}
        
        recommendations = self.recommender._recommend_scaling(outlier_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].scaling_type == 'robust_scaling'
        assert recommendations[0].confidence > 0.5

    def test_recommend_scaling_none_for_well_scaled(self):
        """Test no scaling recommendation for already well-scaled data."""
        # Test with already well-scaled data (mean ~0, std ~1)
        well_scaled_data = pd.DataFrame({'scaled_col': np.random.normal(0, 1, 100)})
        column_types = {'scaled_col': ColumnType('numerical', 'float', 0.9)}
        
        recommendations = self.recommender._recommend_scaling(well_scaled_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].scaling_type == 'none'

    def test_recommend_encoding_binary(self):
        """Test encoding recommendation for binary categorical data."""
        binary_data = pd.DataFrame({'binary_col': np.random.choice(['Yes', 'No'], 100)})
        column_types = {'binary_col': ColumnType('categorical', 'nominal', 0.9)}
        
        recommendations = self.recommender._recommend_encoding(binary_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].encoding_type == 'label_encoding'
        assert recommendations[0].confidence > 0.8

    def test_recommend_encoding_low_cardinality(self):
        """Test encoding recommendation for low cardinality categorical data."""
        low_card_data = pd.DataFrame({'low_card_col': np.random.choice(['A', 'B', 'C', 'D'], 100)})
        column_types = {'low_card_col': ColumnType('categorical', 'nominal', 0.9)}
        
        recommendations = self.recommender._recommend_encoding(low_card_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].encoding_type == 'one_hot'
        assert recommendations[0].confidence > 0.7

    def test_recommend_encoding_high_cardinality(self):
        """Test encoding recommendation for high cardinality categorical data."""
        high_card_data = pd.DataFrame({
            'high_card_col': np.random.choice([f'cat_{i}' for i in range(60)], 100)
        })
        column_types = {'high_card_col': ColumnType('categorical', 'nominal', 0.9)}
        
        recommendations = self.recommender._recommend_encoding(high_card_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].encoding_type in ['binary_encoding', 'target_encoding']

    def test_recommend_encoding_ordinal(self):
        """Test encoding recommendation for ordinal categorical data."""
        ordinal_data = pd.DataFrame({'ordinal_col': np.random.choice(['Low', 'Medium', 'High'], 100)})
        column_types = {'ordinal_col': ColumnType('categorical', 'ordinal', 0.9)}
        
        recommendations = self.recommender._recommend_encoding(ordinal_data, column_types)
        
        assert len(recommendations) == 1
        assert recommendations[0].encoding_type == 'label_encoding'

    def test_detect_ordinal_nature(self):
        """Test ordinal nature detection."""
        # Test common ordinal patterns
        assert self.recommender._detect_ordinal_nature(np.array(['Low', 'Medium', 'High']))
        assert self.recommender._detect_ordinal_nature(np.array(['Small', 'Medium', 'Large']))
        assert self.recommender._detect_ordinal_nature(np.array(['1', '2', '3', '4', '5']))
        assert self.recommender._detect_ordinal_nature(np.array(['S', 'M', 'L', 'XL']))
        
        # Test non-ordinal patterns
        assert not self.recommender._detect_ordinal_nature(np.array(['Red', 'Blue', 'Green']))
        assert not self.recommender._detect_ordinal_nature(np.array(['Cat', 'Dog', 'Bird']))

    def test_recommend_feature_engineering_polynomial(self):
        """Test polynomial feature engineering recommendation."""
        # Create data suitable for polynomial features
        poly_data = pd.DataFrame({
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        column_types = {
            'x1': ColumnType('numerical', 'float', 0.9),
            'x2': ColumnType('numerical', 'float', 0.9),
            'target': ColumnType('numerical', 'float', 0.9)
        }
        
        recommendations = self.recommender._recommend_feature_engineering(
            poly_data, column_types, self.regression_task, self.quality_metrics, 'target'
        )
        
        # Should recommend polynomial features for regression with few numerical features
        poly_recs = [rec for rec in recommendations if rec.technique == 'polynomial_features']
        assert len(poly_recs) > 0
        assert poly_recs[0].confidence > 0.5

    def test_recommend_feature_selection_correlation(self):
        """Test feature selection recommendation based on correlation."""
        # Create correlated features
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.1, 100)  # Highly correlated with x1
        x3 = np.random.normal(0, 1, 100)  # Independent
        
        corr_data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'target': np.random.normal(0, 1, 100)
        })
        column_types = {
            'x1': ColumnType('numerical', 'float', 0.9),
            'x2': ColumnType('numerical', 'float', 0.9),
            'x3': ColumnType('numerical', 'float', 0.9),
            'target': ColumnType('numerical', 'float', 0.9)
        }
        
        recommendations = self.recommender._recommend_feature_engineering(
            corr_data, column_types, self.regression_task, self.quality_metrics, 'target'
        )
        
        # Should recommend feature selection due to high correlation
        selection_recs = [rec for rec in recommendations if rec.technique == 'feature_selection']
        assert len(selection_recs) > 0

    def test_recommend_dimensionality_reduction(self):
        """Test dimensionality reduction recommendation."""
        # Create high-dimensional data
        high_dim_data = pd.DataFrame(np.random.normal(0, 1, (100, 60)))
        high_dim_data.columns = [f'feature_{i}' for i in range(60)]
        
        column_types = {f'feature_{i}': ColumnType('numerical', 'float', 0.9) for i in range(60)}
        
        recommendations = self.recommender._recommend_feature_engineering(
            high_dim_data, column_types, self.classification_task, self.quality_metrics
        )
        
        # Should recommend dimensionality reduction for high-dimensional data
        dim_red_recs = [rec for rec in recommendations if rec.technique == 'pca']
        assert len(dim_red_recs) > 0
        assert dim_red_recs[0].confidence > 0.5

    def test_determine_pipeline_order(self):
        """Test pipeline order determination."""
        scaling_recs = [ScalingRecommendation('standardization', 'test', 0.8, ['col1'])]
        encoding_recs = [EncodingRecommendation('one_hot', 'test', 0.8, ['col2'])]
        feature_eng_recs = [
            FeatureEngineeringRecommendation('polynomial_features', 'test', 0.7, ['col1']),
            FeatureEngineeringRecommendation('feature_selection', 'test', 0.7, ['col1'])
        ]
        
        pipeline_order = self.recommender._determine_pipeline_order(
            scaling_recs, encoding_recs, feature_eng_recs
        )
        
        # Check that encoding comes before scaling
        encoding_idx = pipeline_order.index('encoding')
        scaling_idx = pipeline_order.index('scaling')
        assert encoding_idx < scaling_idx
        
        # Check that feature creation comes before scaling
        if 'feature_creation' in pipeline_order:
            creation_idx = pipeline_order.index('feature_creation')
            assert creation_idx < scaling_idx
        
        # Check that feature selection comes after scaling
        if 'feature_selection' in pipeline_order:
            selection_idx = pipeline_order.index('feature_selection')
            assert selection_idx > scaling_idx

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        # Test with different data sizes
        small_df = pd.DataFrame(np.random.normal(0, 1, (100, 5)))
        large_df = pd.DataFrame(np.random.normal(0, 1, (10000, 20)))
        
        small_time = self.recommender._estimate_processing_time(small_df, 3)
        large_time = self.recommender._estimate_processing_time(large_df, 3)
        
        assert small_time > 0
        assert large_time > small_time  # Larger data should take more time

    def test_scaling_recommendation_validation(self):
        """Test scaling recommendation data validation."""
        # Test valid scaling recommendation
        valid_rec = ScalingRecommendation(
            scaling_type='standardization',
            rationale='Test rationale',
            confidence=0.8,
            affected_columns=['col1']
        )
        assert valid_rec.confidence == 0.8
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ScalingRecommendation(
                scaling_type='standardization',
                rationale='Test rationale',
                confidence=1.5,
                affected_columns=['col1']
            )

    def test_encoding_recommendation_validation(self):
        """Test encoding recommendation data validation."""
        # Test valid encoding recommendation
        valid_rec = EncodingRecommendation(
            encoding_type='one_hot',
            rationale='Test rationale',
            confidence=0.7,
            affected_columns=['col1']
        )
        assert valid_rec.confidence == 0.7
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            EncodingRecommendation(
                encoding_type='one_hot',
                rationale='Test rationale',
                confidence=-0.1,
                affected_columns=['col1']
            )

    def test_feature_engineering_recommendation_validation(self):
        """Test feature engineering recommendation data validation."""
        # Test valid feature engineering recommendation
        valid_rec = FeatureEngineeringRecommendation(
            technique='pca',
            rationale='Test rationale',
            confidence=0.6,
            affected_columns=['col1', 'col2']
        )
        assert valid_rec.technique == 'pca'
        
        # Test empty technique
        with pytest.raises(ValueError, match="Technique cannot be empty"):
            FeatureEngineeringRecommendation(
                technique='',
                rationale='Test rationale',
                confidence=0.6,
                affected_columns=['col1']
            )

    def test_preprocessing_pipeline_validation(self):
        """Test preprocessing pipeline data validation."""
        # Test valid pipeline
        valid_pipeline = PreprocessingPipeline(
            overall_confidence=0.8,
            estimated_processing_time=1.5
        )
        assert valid_pipeline.overall_confidence == 0.8
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Overall confidence must be between 0.0 and 1.0"):
            PreprocessingPipeline(
                overall_confidence=1.2,
                estimated_processing_time=1.5
            )
        
        # Test negative processing time
        with pytest.raises(ValueError, match="Estimated processing time cannot be negative"):
            PreprocessingPipeline(
                overall_confidence=0.8,
                estimated_processing_time=-1.0
            )

    def test_no_recommendations_for_non_numerical_scaling(self):
        """Test that scaling recommendations are not made for non-numerical columns."""
        text_data = pd.DataFrame({'text_col': ['hello', 'world', 'test'] * 34})
        column_types = {'text_col': ColumnType('text', 'natural_language', 0.9)}
        
        recommendations = self.recommender._recommend_scaling(text_data, column_types)
        
        assert len(recommendations) == 0

    def test_no_recommendations_for_non_categorical_encoding(self):
        """Test that encoding recommendations are not made for non-categorical columns."""
        numeric_data = pd.DataFrame({'numeric_col': np.random.normal(0, 1, 100)})
        column_types = {'numeric_col': ColumnType('numerical', 'float', 0.9)}
        
        recommendations = self.recommender._recommend_encoding(numeric_data, column_types)
        
        assert len(recommendations) == 0

    def test_skip_target_column_in_encoding(self):
        """Test that target column is skipped in encoding recommendations."""
        data_with_target = pd.DataFrame({
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice(['X', 'Y'], 100)
        })
        column_types = {
            'categorical_col': ColumnType('categorical', 'nominal', 0.9),
            'target': ColumnType('categorical', 'nominal', 0.9)
        }
        
        recommendations = self.recommender._recommend_encoding(
            data_with_target, column_types, target_column='target'
        )
        
        # Should only recommend encoding for categorical_col, not target
        assert len(recommendations) == 1
        assert recommendations[0].affected_columns == ['categorical_col']