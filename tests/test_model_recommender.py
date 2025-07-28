"""
Unit tests for ModelRecommender functionality.

Tests the traditional ML model recommendation system including suitability scoring,
hyperparameter suggestions, and rationale generation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from neurolite.recommenders.model_recommender import ModelRecommender, ModelSuitabilityScore
from neurolite.core.data_models import (
    SupervisedTaskAnalysis, StatisticalProperties, ModelRecommendation
)


class TestModelRecommender:
    """Test cases for ModelRecommender class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.recommender = ModelRecommender()
        
        # Create sample task analysis for classification
        self.classification_task = SupervisedTaskAnalysis(
            task_type='classification',
            task_subtype='binary',
            confidence=0.9,
            target_characteristics={
                'total_samples': 1000,
                'unique_values': 2,
                'class_labels': ['A', 'B'],
                'class_counts': [500, 500]
            },
            dataset_balance={
                'balance_ratio': 1.0,
                'balance_status': 'balanced',
                'class_proportions': {'A': 0.5, 'B': 0.5}
            },
            complexity_indicators={
                'n_samples': 1000,
                'n_features': 10,
                'samples_to_features_ratio': 100.0,
                'dimensionality': 'normal',
                'multicollinearity_risk': 'low'
            }
        )
        
        # Create sample task analysis for regression
        self.regression_task = SupervisedTaskAnalysis(
            task_type='regression',
            task_subtype='linear',
            confidence=0.85,
            target_characteristics={
                'total_samples': 500,
                'min_value': 0.0,
                'max_value': 100.0,
                'mean_value': 50.0,
                'std_value': 15.0
            },
            dataset_balance={},
            complexity_indicators={
                'n_samples': 500,
                'n_features': 5,
                'samples_to_features_ratio': 100.0,
                'dimensionality': 'normal',
                'multicollinearity_risk': 'low'
            }
        )
        
        # Create sample statistical properties
        self.statistical_properties = StatisticalProperties(
            distribution='normal',
            parameters={'mean': 0.0, 'std': 1.0},
            correlation_matrix=np.eye(3),
            feature_importance={'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2},
            outlier_indices=[10, 25, 100]
        )
    
    def test_initialization(self):
        """Test ModelRecommender initialization."""
        assert self.recommender.confidence_threshold == 0.6
        assert self.recommender.min_samples_for_evaluation == 50
        assert self.recommender.cross_validation_folds == 3
        
        # Check that model dictionaries are properly initialized
        assert 'decision_tree' in self.recommender.traditional_ml_models
        assert 'random_forest' in self.recommender.traditional_ml_models
        assert 'svm' in self.recommender.traditional_ml_models
        assert 'cnn' in self.recommender.deep_learning_models
    
    def test_filter_models_by_task_classification(self):
        """Test filtering models for classification tasks."""
        suitable_models = self.recommender._filter_models_by_task('classification', 'traditional')
        
        # Should include classification-suitable models
        assert 'decision_tree' in suitable_models
        assert 'random_forest' in suitable_models
        assert 'logistic_regression' in suitable_models
        assert 'svm' in suitable_models
        
        # Should not include regression-only models
        assert 'linear_regression' not in suitable_models
    
    def test_filter_models_by_task_regression(self):
        """Test filtering models for regression tasks."""
        suitable_models = self.recommender._filter_models_by_task('regression', 'traditional')
        
        # Should include regression-suitable models
        assert 'decision_tree' in suitable_models
        assert 'random_forest' in suitable_models
        assert 'linear_regression' in suitable_models
        assert 'svm' in suitable_models
        
        # Should not include classification-only models
        assert 'logistic_regression' not in suitable_models
        assert 'naive_bayes' not in suitable_models
    
    def test_evaluate_size_suitability(self):
        """Test dataset size suitability evaluation."""
        # Test with small dataset
        score_small = self.recommender._evaluate_size_suitability('decision_tree', 10, 5)
        assert 0.0 <= score_small <= 1.0
        assert score_small < 0.5  # Should be low for very small datasets
        
        # Test with optimal dataset size
        score_optimal = self.recommender._evaluate_size_suitability('decision_tree', 1000, 10)
        assert score_optimal > 0.8  # Should be high for optimal size
        
        # Test with high-dimensional data
        score_high_dim = self.recommender._evaluate_size_suitability('knn', 100, 200)
        assert score_high_dim < 0.5  # KNN struggles with high dimensions
        
        # Test with model that handles high dimensions well
        score_rf_high_dim = self.recommender._evaluate_size_suitability('random_forest', 100, 200)
        assert score_rf_high_dim > score_high_dim  # RF should handle better
    
    def test_evaluate_subtype_suitability(self):
        """Test task subtype suitability evaluation."""
        # Test binary classification
        score_binary = self.recommender._evaluate_subtype_suitability('logistic_regression', 'binary')
        assert score_binary >= 0.8  # Logistic regression is great for binary
        
        # Test linear regression with linear task
        score_linear = self.recommender._evaluate_subtype_suitability('linear_regression', 'linear')
        assert score_linear >= 0.8  # Perfect match
        
        # Test linear regression with non-linear task
        score_nonlinear = self.recommender._evaluate_subtype_suitability('linear_regression', 'non_linear')
        assert score_nonlinear < 0.5  # Poor match
        
        # Test tree-based model with non-linear task
        score_tree_nonlinear = self.recommender._evaluate_subtype_suitability('decision_tree', 'non_linear')
        assert score_tree_nonlinear >= 0.8  # Good match
    
    def test_evaluate_balance_suitability(self):
        """Test dataset balance suitability evaluation."""
        # Test balanced dataset
        balanced_info = {'balance_status': 'balanced'}
        score_balanced = self.recommender._evaluate_balance_suitability('random_forest', balanced_info)
        assert score_balanced >= 0.8
        
        # Test severely imbalanced dataset
        imbalanced_info = {'balance_status': 'severely_imbalanced'}
        score_imbalanced = self.recommender._evaluate_balance_suitability('svm', imbalanced_info)
        assert score_imbalanced < 0.5  # SVM struggles with severe imbalance
        
        # Test with empty balance info
        score_empty = self.recommender._evaluate_balance_suitability('decision_tree', {})
        assert score_empty == 0.5  # Default score
    
    def test_evaluate_complexity_suitability(self):
        """Test data complexity suitability evaluation."""
        complexity_indicators = {
            'multicollinearity_risk': 'high',
            'dimensionality': 'high'
        }
        data_characteristics = {
            'interpretability_required': True
        }
        
        # Test model that handles complexity well
        score_rf = self.recommender._evaluate_complexity_suitability(
            'random_forest', complexity_indicators, data_characteristics
        )
        assert score_rf > 0.5  # Should handle complexity well
        
        # Test interpretable model with interpretability requirement
        score_dt = self.recommender._evaluate_complexity_suitability(
            'decision_tree', complexity_indicators, data_characteristics
        )
        assert score_dt > 0.5  # Should get bonus for interpretability
        
        # Test linear model with high multicollinearity
        score_lr = self.recommender._evaluate_complexity_suitability(
            'linear_regression', complexity_indicators, data_characteristics
        )
        # Should be penalized for multicollinearity but get bonus for interpretability
        assert 0.0 <= score_lr <= 1.0
    
    def test_suggest_hyperparameters(self):
        """Test hyperparameter suggestion functionality."""
        # Test decision tree hyperparameters
        dt_params = self.recommender._suggest_hyperparameters(
            'decision_tree', self.classification_task, {}
        )
        assert 'max_depth' in dt_params
        assert 'min_samples_split' in dt_params
        assert 'random_state' in dt_params
        assert dt_params['random_state'] == 42
        
        # Test random forest hyperparameters
        rf_params = self.recommender._suggest_hyperparameters(
            'random_forest', self.classification_task, {}
        )
        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        assert rf_params['n_estimators'] >= 50
        
        # Test SVM hyperparameters
        svm_params = self.recommender._suggest_hyperparameters(
            'svm', self.classification_task, {}
        )
        assert 'C' in svm_params
        assert 'kernel' in svm_params
        assert 'gamma' in svm_params
        
        # Test logistic regression hyperparameters
        lr_params = self.recommender._suggest_hyperparameters(
            'logistic_regression', self.classification_task, {}
        )
        assert 'C' in lr_params
        assert 'solver' in lr_params
        assert 'max_iter' in lr_params
    
    def test_estimate_performance(self):
        """Test performance estimation functionality."""
        # Test classification performance estimation
        performance = self.recommender._estimate_performance(
            'random_forest', self.classification_task, 0.85
        )
        
        assert 'accuracy' in performance
        assert 'precision' in performance
        assert 'recall' in performance
        assert 'f1' in performance
        
        # Check that all metrics are in valid range
        for metric, value in performance.items():
            if not metric.endswith('_confidence_lower') and not metric.endswith('_confidence_upper'):
                assert 0.0 <= value <= 1.0
        
        # Check confidence intervals exist
        assert 'accuracy_confidence_lower' in performance
        assert 'accuracy_confidence_upper' in performance
        
        # Test regression performance estimation
        regression_performance = self.recommender._estimate_performance(
            'linear_regression', self.regression_task, 0.75
        )
        
        assert 'r2' in regression_performance
        assert 'mse' in regression_performance
        assert 'mae' in regression_performance
    
    def test_score_traditional_model(self):
        """Test traditional model scoring functionality."""
        model_info = self.recommender.traditional_ml_models['random_forest']
        
        score = self.recommender._score_traditional_model(
            'random_forest', model_info, self.classification_task, 
            self.statistical_properties, {}
        )
        
        assert isinstance(score, ModelSuitabilityScore)
        assert score.model_name == 'Random Forest'
        assert 0.0 <= score.suitability_score <= 1.0
        assert isinstance(score.rationale_factors, dict)
        assert isinstance(score.performance_estimate, dict)
        assert isinstance(score.hyperparameter_suggestions, dict)
        
        # Check that rationale factors are present
        assert 'task_compatibility' in score.rationale_factors
        assert 'dataset_size' in score.rationale_factors
        assert 'task_subtype' in score.rationale_factors
    
    def test_generate_rationale(self):
        """Test rationale generation functionality."""
        # Create a mock score object
        score = ModelSuitabilityScore(
            model_name='Random Forest',
            suitability_score=0.85,
            rationale_factors={
                'task_compatibility': 0.2,
                'dataset_size': 0.15,
                'task_subtype': 0.1,
                'dataset_balance': 0.05
            },
            performance_estimate={'accuracy': 0.87},
            hyperparameter_suggestions={}
        )
        
        rationale = self.recommender._generate_rationale(score)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert 'Random Forest' in rationale
        assert rationale.endswith('.')
        
        # Should mention high suitability
        assert 'highly suitable' in rationale.lower()
        
        # Should mention performance expectations
        assert 'performance' in rationale.lower()
    
    def test_recommend_traditional_ml_classification(self):
        """Test traditional ML recommendations for classification."""
        recommendations = self.recommender.recommend_traditional_ml(
            self.classification_task, self.statistical_properties, {}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that all recommendations are valid
        for rec in recommendations:
            assert isinstance(rec, ModelRecommendation)
            assert rec.model_type == 'traditional'
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.confidence >= self.recommender.confidence_threshold
            assert len(rec.rationale) > 0
            assert isinstance(rec.hyperparameters, dict)
            assert isinstance(rec.expected_performance, dict)
        
        # Should be sorted by confidence (highest first)
        confidences = [rec.confidence for rec in recommendations]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_recommend_traditional_ml_regression(self):
        """Test traditional ML recommendations for regression."""
        recommendations = self.recommender.recommend_traditional_ml(
            self.regression_task, self.statistical_properties, {}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include regression-appropriate models
        model_names = [rec.model_name for rec in recommendations]
        assert any('Linear Regression' in name for name in model_names)
        assert any('Random Forest' in name for name in model_names)
        
        # Should not include classification-only models
        assert not any('Logistic Regression' in name for name in model_names)
        assert not any('Naive Bayes' in name for name in model_names)
    
    def test_recommend_traditional_ml_with_constraints(self):
        """Test traditional ML recommendations with specific constraints."""
        data_characteristics = {
            'interpretability_required': True,
            'fast_training_required': True,
            'memory_constrained': True
        }
        
        recommendations = self.recommender.recommend_traditional_ml(
            self.classification_task, None, data_characteristics
        )
        
        assert len(recommendations) > 0
        
        # Should favor interpretable, fast, memory-efficient models
        model_names = [rec.model_name for rec in recommendations]
        
        # Decision tree and logistic regression should be highly ranked
        # due to interpretability and efficiency
        top_models = model_names[:3]  # Top 3 recommendations
        assert any('Decision Tree' in name for name in top_models) or \
               any('Logistic Regression' in name for name in top_models)
    
    def test_recommend_traditional_ml_imbalanced_data(self):
        """Test recommendations for imbalanced datasets."""
        # Create imbalanced task analysis
        imbalanced_task = SupervisedTaskAnalysis(
            task_type='classification',
            task_subtype='binary',
            confidence=0.9,
            target_characteristics={
                'total_samples': 1000,
                'unique_values': 2,
                'class_labels': ['A', 'B'],
                'class_counts': [900, 100]
            },
            dataset_balance={
                'balance_ratio': 0.11,
                'balance_status': 'severely_imbalanced',
                'class_proportions': {'A': 0.9, 'B': 0.1}
            },
            complexity_indicators={
                'n_samples': 1000,
                'n_features': 10,
                'samples_to_features_ratio': 100.0,
                'dimensionality': 'normal',
                'multicollinearity_risk': 'low'
            }
        )
        
        recommendations = self.recommender.recommend_traditional_ml(imbalanced_task)
        
        assert len(recommendations) > 0
        
        # Models that handle imbalance better should be ranked higher
        model_names = [rec.model_name for rec in recommendations]
        
        # Random Forest and Gradient Boosting should be preferred over SVM
        rf_index = next((i for i, name in enumerate(model_names) if 'Random Forest' in name), None)
        svm_index = next((i for i, name in enumerate(model_names) if 'Support Vector Machine' in name), None)
        
        if rf_index is not None and svm_index is not None:
            assert rf_index < svm_index  # RF should be ranked higher than SVM
    
    def test_recommend_traditional_ml_high_dimensional(self):
        """Test recommendations for high-dimensional datasets."""
        # Create high-dimensional task analysis
        high_dim_task = SupervisedTaskAnalysis(
            task_type='classification',
            task_subtype='multiclass',
            confidence=0.8,
            target_characteristics={
                'total_samples': 500,
                'unique_values': 5,
                'class_labels': ['A', 'B', 'C', 'D', 'E']
            },
            dataset_balance={
                'balance_ratio': 0.8,
                'balance_status': 'balanced'
            },
            complexity_indicators={
                'n_samples': 500,
                'n_features': 1000,  # High dimensionality
                'samples_to_features_ratio': 0.5,
                'dimensionality': 'high',
                'multicollinearity_risk': 'high'
            }
        )
        
        recommendations = self.recommender.recommend_traditional_ml(high_dim_task)
        
        assert len(recommendations) > 0
        
        # Models that handle high dimensions well should be preferred
        model_names = [rec.model_name for rec in recommendations]
        
        # SVM and Random Forest should be preferred over KNN
        top_models = model_names[:3]
        assert any('Support Vector Machine' in name or 'Random Forest' in name for name in top_models)
        
        # KNN should be ranked lower or not recommended
        knn_index = next((i for i, name in enumerate(model_names) if 'K-Nearest Neighbors' in name), None)
        if knn_index is not None:
            assert knn_index >= 3  # Should not be in top 3


if __name__ == '__main__':
    pytest.main([__file__])    

    def test_filter_deep_learning_models_cv(self):
        """Test filtering deep learning models for computer vision tasks."""
        # Mock CV analysis
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        cv_analysis.num_classes = 10
        
        domain_analysis = {'cv_analysis': cv_analysis}
        
        suitable_models = self.recommender._filter_deep_learning_models(
            self.classification_task, domain_analysis
        )
        
        assert 'cnn' in suitable_models
        assert suitable_models['cnn']['name'] == 'Convolutional Neural Network'
    
    def test_filter_deep_learning_models_nlp(self):
        """Test filtering deep learning models for NLP tasks."""
        # Mock NLP analysis
        nlp_analysis = Mock()
        nlp_analysis.task_type = 'sentiment'
        
        domain_analysis = {'nlp_analysis': nlp_analysis}
        
        suitable_models = self.recommender._filter_deep_learning_models(
            self.classification_task, domain_analysis
        )
        
        assert 'transformer' in suitable_models
        assert suitable_models['transformer']['name'] == 'Transformer'
    
    def test_filter_deep_learning_models_timeseries(self):
        """Test filtering deep learning models for time series tasks."""
        # Mock time series analysis
        ts_analysis = Mock()
        ts_analysis.series_type = 'univariate'
        
        domain_analysis = {'timeseries_analysis': ts_analysis}
        
        suitable_models = self.recommender._filter_deep_learning_models(
            self.classification_task, domain_analysis
        )
        
        assert 'rnn_lstm' in suitable_models
        assert 'transformer' in suitable_models
    
    def test_filter_deep_learning_models_unsupervised(self):
        """Test filtering deep learning models for unsupervised tasks."""
        # Create unsupervised task analysis
        unsupervised_task = UnsupervisedTaskAnalysis(
            clustering_potential=0.8,
            optimal_clusters=5,
            dimensionality_reduction_needed=True,
            confidence=0.85,
            clustering_characteristics={},
            dimensionality_info={}
        )
        
        suitable_models = self.recommender._filter_deep_learning_models(
            unsupervised_task, {}
        )
        
        assert 'autoencoder' in suitable_models
    
    def test_evaluate_domain_compatibility(self):
        """Test domain compatibility evaluation for deep learning models."""
        # Test CNN with CV domain
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        domain_analysis = {'cv_analysis': cv_analysis}
        
        score = self.recommender._evaluate_domain_compatibility('cnn', domain_analysis)
        assert score >= 0.9  # Should be very high for CNN + CV
        
        # Test Transformer with NLP domain
        nlp_analysis = Mock()
        nlp_analysis.task_type = 'sentiment'
        domain_analysis = {'nlp_analysis': nlp_analysis}
        
        score = self.recommender._evaluate_domain_compatibility('transformer', domain_analysis)
        assert score >= 0.9  # Should be very high for Transformer + NLP
        
        # Test mismatched domain
        score = self.recommender._evaluate_domain_compatibility('cnn', {'nlp_analysis': nlp_analysis})
        assert score < 0.5  # Should be low for CNN + NLP
    
    def test_evaluate_deep_learning_size_requirements(self):
        """Test dataset size evaluation for deep learning models."""
        # Test with insufficient data
        score_small = self.recommender._evaluate_deep_learning_size_requirements('cnn', 100)
        assert score_small <= 0.3  # Should be very low
        
        # Test with optimal data
        score_optimal = self.recommender._evaluate_deep_learning_size_requirements('cnn', 10000)
        assert score_optimal >= 0.9  # Should be high
        
        # Test transformer requirements (higher than CNN)
        score_transformer_small = self.recommender._evaluate_deep_learning_size_requirements('transformer', 1000)
        score_cnn_small = self.recommender._evaluate_deep_learning_size_requirements('cnn', 1000)
        assert score_transformer_small <= score_cnn_small  # Transformer needs more data
    
    def test_evaluate_computational_requirements(self):
        """Test computational requirements evaluation."""
        # Test with GPU available
        data_chars_gpu = {'gpu_available': True}
        score_gpu = self.recommender._evaluate_computational_requirements('cnn', data_chars_gpu)
        
        # Test without GPU
        data_chars_no_gpu = {'gpu_available': False}
        score_no_gpu = self.recommender._evaluate_computational_requirements('cnn', data_chars_no_gpu)
        
        assert score_gpu > score_no_gpu  # Should be higher with GPU
        
        # Test with compute constraints
        data_chars_constrained = {'compute_constrained': True}
        score_constrained = self.recommender._evaluate_computational_requirements('transformer', data_chars_constrained)
        assert score_constrained < 0.5  # Should be penalized
    
    def test_suggest_deep_learning_hyperparameters_cnn(self):
        """Test CNN hyperparameter suggestions."""
        cv_analysis = Mock()
        cv_analysis.num_classes = 10
        domain_analysis = {'cv_analysis': cv_analysis}
        
        hyperparams = self.recommender._suggest_deep_learning_hyperparameters(
            'cnn', self.classification_task, domain_analysis, {}
        )
        
        assert 'conv_layers' in hyperparams
        assert 'optimizer' in hyperparams
        assert 'learning_rate' in hyperparams
        assert 'batch_size' in hyperparams
        assert 'epochs' in hyperparams
        assert hyperparams['output_units'] == 10
        assert hyperparams['output_activation'] == 'softmax'
    
    def test_suggest_deep_learning_hyperparameters_transformer(self):
        """Test Transformer hyperparameter suggestions."""
        nlp_analysis = Mock()
        nlp_analysis.task_type = 'sentiment'
        domain_analysis = {'nlp_analysis': nlp_analysis}
        
        hyperparams = self.recommender._suggest_deep_learning_hyperparameters(
            'transformer', self.classification_task, domain_analysis, {}
        )
        
        assert 'num_layers' in hyperparams
        assert 'num_heads' in hyperparams
        assert 'hidden_size' in hyperparams
        assert 'learning_rate' in hyperparams
        assert 'warmup_steps' in hyperparams
        assert hyperparams['output_units'] == 2  # Binary sentiment
    
    def test_suggest_deep_learning_hyperparameters_autoencoder(self):
        """Test AutoEncoder hyperparameter suggestions."""
        data_characteristics = {'n_features': 100}
        
        hyperparams = self.recommender._suggest_deep_learning_hyperparameters(
            'autoencoder', self.classification_task, {}, data_characteristics
        )
        
        assert 'encoder_dims' in hyperparams
        assert 'decoder_dims' in hyperparams
        assert 'optimizer' in hyperparams
        assert 'loss' in hyperparams
        assert hyperparams['loss'] == 'mse'
        
        # Check that decoder mirrors encoder
        encoder_dims = hyperparams['encoder_dims']
        decoder_dims = hyperparams['decoder_dims']
        assert len(encoder_dims) > 0
        assert len(decoder_dims) > 0
        assert decoder_dims[-1] == 100  # Should reconstruct original dimension
    
    def test_estimate_deep_learning_performance(self):
        """Test deep learning performance estimation."""
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        domain_analysis = {'cv_analysis': cv_analysis}
        
        performance = self.recommender._estimate_deep_learning_performance(
            'cnn', self.classification_task, 0.9, domain_analysis
        )
        
        assert 'accuracy' in performance
        assert 'precision' in performance
        assert 'recall' in performance
        assert 'f1' in performance
        assert 'estimated_training_time_minutes' in performance
        
        # Check confidence intervals
        assert 'accuracy_confidence_lower' in performance
        assert 'accuracy_confidence_upper' in performance
        
        # Performance should be adjusted upward for good suitability + domain match
        assert performance['accuracy'] >= 0.8
    
    def test_score_deep_learning_model(self):
        """Test deep learning model scoring."""
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        cv_analysis.num_classes = 10
        domain_analysis = {'cv_analysis': cv_analysis}
        
        model_info = self.recommender.deep_learning_models['cnn']
        data_characteristics = {'gpu_available': True, 'n_samples': 5000}
        
        score = self.recommender._score_deep_learning_model(
            'cnn', model_info, self.classification_task, domain_analysis, data_characteristics
        )
        
        assert isinstance(score, ModelSuitabilityScore)
        assert score.model_name == 'Convolutional Neural Network'
        assert 0.0 <= score.suitability_score <= 1.0
        assert 'domain_compatibility' in score.rationale_factors
        assert 'dataset_size' in score.rationale_factors
        assert 'computational_feasibility' in score.rationale_factors
    
    def test_generate_deep_learning_rationale(self):
        """Test deep learning rationale generation."""
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        domain_analysis = {'cv_analysis': cv_analysis}
        
        score = ModelSuitabilityScore(
            model_name='Convolutional Neural Network',
            suitability_score=0.85,
            rationale_factors={
                'domain_compatibility': 0.25,
                'dataset_size': 0.2,
                'computational_feasibility': 0.15
            },
            performance_estimate={'accuracy': 0.9, 'estimated_training_time_minutes': 45},
            hyperparameter_suggestions={}
        )
        
        rationale = self.recommender._generate_deep_learning_rationale(score, domain_analysis)
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert 'Convolutional Neural Network' in rationale
        assert 'computer vision' in rationale.lower()
        assert 'performance' in rationale.lower()
        assert rationale.endswith('.')
    
    def test_recommend_deep_learning_cv_task(self):
        """Test deep learning recommendations for computer vision tasks."""
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        cv_analysis.num_classes = 10
        domain_analysis = {'cv_analysis': cv_analysis}
        
        data_characteristics = {
            'gpu_available': True,
            'data_structure': 'image',
            'n_samples': 5000
        }
        
        recommendations = self.recommender.recommend_deep_learning(
            self.classification_task, domain_analysis, data_characteristics
        )
        
        assert len(recommendations) > 0
        
        # Should recommend CNN for CV tasks
        model_names = [rec.model_name for rec in recommendations]
        assert any('Convolutional Neural Network' in name for name in model_names)
        
        # Check recommendation properties
        for rec in recommendations:
            assert isinstance(rec, ModelRecommendation)
            assert rec.model_type == 'deep_learning'
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.rationale) > 0
    
    def test_recommend_deep_learning_nlp_task(self):
        """Test deep learning recommendations for NLP tasks."""
        nlp_analysis = Mock()
        nlp_analysis.task_type = 'sentiment'
        nlp_analysis.text_characteristics = {'avg_length': 150, 'vocabulary_size': 5000}
        domain_analysis = {'nlp_analysis': nlp_analysis}
        
        data_characteristics = {
            'gpu_available': True,
            'data_structure': 'text',
            'n_samples': 10000
        }
        
        recommendations = self.recommender.recommend_deep_learning(
            self.classification_task, domain_analysis, data_characteristics
        )
        
        assert len(recommendations) > 0
        
        # Should recommend Transformer for NLP tasks
        model_names = [rec.model_name for rec in recommendations]
        assert any('Transformer' in name for name in model_names)
    
    def test_recommend_deep_learning_insufficient_data(self):
        """Test deep learning recommendations with insufficient data."""
        cv_analysis = Mock()
        cv_analysis.task_type = 'classification'
        domain_analysis = {'cv_analysis': cv_analysis}
        
        # Very small dataset
        small_task = SupervisedTaskAnalysis(
            task_type='classification',
            task_subtype='binary',
            confidence=0.9,
            target_characteristics={'total_samples': 100},
            dataset_balance={'balance_status': 'balanced'},
            complexity_indicators={
                'n_samples': 100,  # Too small for deep learning
                'n_features': 10,
                'dimensionality': 'normal'
            }
        )
        
        recommendations = self.recommender.recommend_deep_learning(
            small_task, domain_analysis, {}
        )
        
        # Should have few or no recommendations due to insufficient data
        assert len(recommendations) == 0 or all(rec.confidence < 0.7 for rec in recommendations)
    
    def test_recommend_deep_learning_unsupervised(self):
        """Test deep learning recommendations for unsupervised tasks."""
        unsupervised_task = UnsupervisedTaskAnalysis(
            clustering_potential=0.8,
            optimal_clusters=5,
            dimensionality_reduction_needed=True,
            confidence=0.85,
            clustering_characteristics={},
            dimensionality_info={}
        )
        
        data_characteristics = {
            'n_samples': 5000,
            'n_features': 100,
            'data_structure': 'tabular'
        }
        
        recommendations = self.recommender.recommend_deep_learning(
            unsupervised_task, {}, data_characteristics
        )
        
        if len(recommendations) > 0:
            # Should recommend AutoEncoder for unsupervised tasks
            model_names = [rec.model_name for rec in recommendations]
            assert any('AutoEncoder' in name for name in model_names)