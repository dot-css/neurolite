"""
Model recommender for ML algorithm suggestions.

This module provides functionality to recommend appropriate ML models and algorithms
based on data characteristics, task type, and performance requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

from ..core.data_models import (
    ModelRecommendation, TaskIdentification, SupervisedTaskAnalysis, 
    UnsupervisedTaskAnalysis, StatisticalProperties
)
from ..core.exceptions import NeuroLiteException


@dataclass
class ModelSuitabilityScore:
    """Represents suitability score for a specific model."""
    model_name: str
    suitability_score: float
    rationale_factors: Dict[str, float]
    performance_estimate: Dict[str, float]
    hyperparameter_suggestions: Dict[str, Any]


class ModelRecommender:
    """Recommender for ML models and algorithms."""
    
    def __init__(self):
        """Initialize the ModelRecommender."""
        self.confidence_threshold = 0.6
        self.min_samples_for_evaluation = 50
        self.cross_validation_folds = 3   
     
        # Model categories and their characteristics
        self.traditional_ml_models = {
            'decision_tree': {
                'name': 'Decision Tree',
                'type': 'traditional',
                'suitable_for': ['classification', 'regression'],
                'strengths': ['interpretable', 'handles_mixed_types', 'no_scaling_needed'],
                'weaknesses': ['overfitting_prone', 'unstable'],
                'complexity': 'low',
                'training_speed': 'fast',
                'prediction_speed': 'fast'
            },
            'random_forest': {
                'name': 'Random Forest',
                'type': 'traditional',
                'suitable_for': ['classification', 'regression'],
                'strengths': ['robust', 'handles_missing_values', 'feature_importance'],
                'weaknesses': ['less_interpretable', 'memory_intensive'],
                'complexity': 'medium',
                'training_speed': 'medium',
                'prediction_speed': 'fast'
            },
            'svm': {
                'name': 'Support Vector Machine',
                'type': 'traditional',
                'suitable_for': ['classification', 'regression'],
                'strengths': ['effective_high_dim', 'memory_efficient', 'versatile_kernels'],
                'weaknesses': ['slow_large_datasets', 'requires_scaling', 'no_probability_estimates'],
                'complexity': 'medium',
                'training_speed': 'slow',
                'prediction_speed': 'fast'
            },
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'traditional',
                'suitable_for': ['regression'],
                'strengths': ['interpretable', 'fast', 'simple', 'probabilistic'],
                'weaknesses': ['assumes_linearity', 'sensitive_outliers'],
                'complexity': 'low',
                'training_speed': 'fast',
                'prediction_speed': 'fast'
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'type': 'traditional',
                'suitable_for': ['classification'],
                'strengths': ['interpretable', 'probabilistic', 'fast', 'regularization'],
                'weaknesses': ['assumes_linearity', 'sensitive_outliers'],
                'complexity': 'low',
                'training_speed': 'fast',
                'prediction_speed': 'fast'
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'type': 'traditional',
                'suitable_for': ['classification', 'regression'],
                'strengths': ['high_performance', 'handles_mixed_types', 'feature_importance'],
                'weaknesses': ['overfitting_prone', 'slow_training', 'many_hyperparameters'],
                'complexity': 'high',
                'training_speed': 'slow',
                'prediction_speed': 'medium'
            },
            'naive_bayes': {
                'name': 'Naive Bayes',
                'type': 'traditional',
                'suitable_for': ['classification'],
                'strengths': ['fast', 'simple', 'works_small_datasets', 'probabilistic'],
                'weaknesses': ['independence_assumption', 'categorical_bias'],
                'complexity': 'low',
                'training_speed': 'fast',
                'prediction_speed': 'fast'
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'type': 'traditional',
                'suitable_for': ['classification', 'regression'],
                'strengths': ['simple', 'no_assumptions', 'works_irregular_boundaries'],
                'weaknesses': ['slow_prediction', 'curse_dimensionality', 'sensitive_irrelevant_features'],
                'complexity': 'low',
                'training_speed': 'fast',
                'prediction_speed': 'slow'
            }
        }      
  
        # Deep learning models (for task 8.2)
        self.deep_learning_models = {
            'cnn': {
                'name': 'Convolutional Neural Network',
                'type': 'deep_learning',
                'suitable_for': ['image_classification', 'computer_vision'],
                'strengths': ['spatial_features', 'translation_invariant', 'hierarchical_learning'],
                'weaknesses': ['requires_large_data', 'computationally_intensive'],
                'complexity': 'high',
                'training_speed': 'slow',
                'prediction_speed': 'medium'
            },
            'rnn_lstm': {
                'name': 'RNN/LSTM',
                'type': 'deep_learning',
                'suitable_for': ['sequence_modeling', 'time_series', 'nlp'],
                'strengths': ['sequential_data', 'memory', 'variable_length'],
                'weaknesses': ['vanishing_gradient', 'slow_training', 'requires_large_data'],
                'complexity': 'high',
                'training_speed': 'slow',
                'prediction_speed': 'medium'
            },
            'transformer': {
                'name': 'Transformer',
                'type': 'deep_learning',
                'suitable_for': ['nlp', 'sequence_modeling', 'attention_tasks'],
                'strengths': ['attention_mechanism', 'parallelizable', 'long_range_dependencies'],
                'weaknesses': ['requires_very_large_data', 'computationally_intensive', 'memory_intensive'],
                'complexity': 'very_high',
                'training_speed': 'slow',
                'prediction_speed': 'medium'
            },
            'autoencoder': {
                'name': 'AutoEncoder',
                'type': 'deep_learning',
                'suitable_for': ['dimensionality_reduction', 'anomaly_detection', 'unsupervised'],
                'strengths': ['unsupervised', 'dimensionality_reduction', 'feature_learning'],
                'weaknesses': ['requires_large_data', 'black_box', 'hyperparameter_sensitive'],
                'complexity': 'high',
                'training_speed': 'slow',
                'prediction_speed': 'fast'
            }
        }
    
    def recommend_traditional_ml(self, 
                               task_analysis: SupervisedTaskAnalysis,
                               statistical_properties: Optional[StatisticalProperties] = None,
                               data_characteristics: Optional[Dict[str, Any]] = None) -> List[ModelRecommendation]:
        """
        Recommend traditional ML models based on task analysis.
        
        Args:
            task_analysis: Analysis of the supervised learning task
            statistical_properties: Statistical properties of the dataset
            data_characteristics: Additional data characteristics
            
        Returns:
            List of ranked model recommendations
        """
        if data_characteristics is None:
            data_characteristics = {}
            
        # Get suitable models for the task type
        suitable_models = self._filter_models_by_task(task_analysis.task_type, 'traditional')
        
        # Score each model based on suitability
        model_scores = []
        for model_key, model_info in suitable_models.items():
            score = self._score_traditional_model(
                model_key, model_info, task_analysis, statistical_properties, data_characteristics
            )
            model_scores.append(score)
        
        # Rank models by suitability score
        model_scores.sort(key=lambda x: x.suitability_score, reverse=True)
        
        # Convert to ModelRecommendation objects
        recommendations = []
        for score in model_scores:
            if score.suitability_score >= self.confidence_threshold:
                recommendation = ModelRecommendation(
                    model_name=score.model_name,
                    model_type='traditional',
                    confidence=score.suitability_score,
                    rationale=self._generate_rationale(score),
                    hyperparameters=score.hyperparameter_suggestions,
                    expected_performance=score.performance_estimate
                )
                recommendations.append(recommendation)
        
        return recommendations   
 
    def _filter_models_by_task(self, task_type: str, model_category: str) -> Dict[str, Dict]:
        """
        Filter models that are suitable for the given task type.
        
        Args:
            task_type: Type of ML task (classification, regression, etc.)
            model_category: Category of models (traditional, deep_learning)
            
        Returns:
            Dictionary of suitable models
        """
        if model_category == 'traditional':
            model_dict = self.traditional_ml_models
        else:
            model_dict = self.deep_learning_models
            
        suitable_models = {}
        for model_key, model_info in model_dict.items():
            if task_type in model_info['suitable_for'] or 'all' in model_info['suitable_for']:
                suitable_models[model_key] = model_info
                
        return suitable_models
    
    def _score_traditional_model(self, 
                               model_key: str,
                               model_info: Dict[str, Any],
                               task_analysis: SupervisedTaskAnalysis,
                               statistical_properties: Optional[StatisticalProperties],
                               data_characteristics: Dict[str, Any]) -> ModelSuitabilityScore:
        """
        Score a traditional ML model based on its suitability for the task.
        
        Args:
            model_key: Key identifier for the model
            model_info: Model information dictionary
            task_analysis: Analysis of the supervised learning task
            statistical_properties: Statistical properties of the dataset
            data_characteristics: Additional data characteristics
            
        Returns:
            ModelSuitabilityScore object
        """
        rationale_factors = {}
        base_score = 0.5  # Base suitability score
        
        # Task type compatibility
        if task_analysis.task_type in model_info['suitable_for']:
            base_score += 0.2
            rationale_factors['task_compatibility'] = 0.2
        
        # Dataset size considerations
        n_samples = task_analysis.complexity_indicators.get('n_samples', 0)
        n_features = task_analysis.complexity_indicators.get('n_features', 0)
        
        size_score = self._evaluate_size_suitability(model_key, n_samples, n_features)
        base_score += size_score * 0.15
        rationale_factors['dataset_size'] = size_score * 0.15
        
        # Task subtype specific scoring
        subtype_score = self._evaluate_subtype_suitability(model_key, task_analysis.task_subtype)
        base_score += subtype_score * 0.15
        rationale_factors['task_subtype'] = subtype_score * 0.15
        
        # Dataset balance considerations (for classification)
        if task_analysis.task_type == 'classification':
            balance_score = self._evaluate_balance_suitability(model_key, task_analysis.dataset_balance)
            base_score += balance_score * 0.1
            rationale_factors['dataset_balance'] = balance_score * 0.1
        
        # Complexity and interpretability trade-offs
        complexity_score = self._evaluate_complexity_suitability(
            model_key, task_analysis.complexity_indicators, data_characteristics
        )
        base_score += complexity_score * 0.1
        rationale_factors['complexity_handling'] = complexity_score * 0.1
        
        # Statistical properties considerations
        if statistical_properties:
            stats_score = self._evaluate_statistical_suitability(model_key, statistical_properties)
            base_score += stats_score * 0.1
            rationale_factors['statistical_properties'] = stats_score * 0.1
        
        # Performance and efficiency considerations
        efficiency_score = self._evaluate_efficiency_requirements(model_key, data_characteristics)
        base_score += efficiency_score * 0.1
        rationale_factors['efficiency'] = efficiency_score * 0.1
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, base_score))
        
        # Generate hyperparameter suggestions
        hyperparameters = self._suggest_hyperparameters(model_key, task_analysis, data_characteristics)
        
        # Estimate performance
        performance_estimate = self._estimate_performance(model_key, task_analysis, final_score)
        
        return ModelSuitabilityScore(
            model_name=model_info['name'],
            suitability_score=final_score,
            rationale_factors=rationale_factors,
            performance_estimate=performance_estimate,
            hyperparameter_suggestions=hyperparameters
        )  
  
    def _evaluate_size_suitability(self, model_key: str, n_samples: int, n_features: int) -> float:
        """
        Evaluate model suitability based on dataset size.
        
        Args:
            model_key: Model identifier
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Size suitability score (0-1)
        """
        if n_samples == 0:
            return 0.5  # Default if unknown
            
        # Model-specific size preferences
        size_preferences = {
            'decision_tree': {'min_samples': 20, 'optimal_samples': 1000, 'handles_high_dim': False},
            'random_forest': {'min_samples': 50, 'optimal_samples': 5000, 'handles_high_dim': True},
            'svm': {'min_samples': 100, 'optimal_samples': 10000, 'handles_high_dim': True},
            'linear_regression': {'min_samples': 30, 'optimal_samples': 1000, 'handles_high_dim': False},
            'logistic_regression': {'min_samples': 50, 'optimal_samples': 2000, 'handles_high_dim': False},
            'gradient_boosting': {'min_samples': 100, 'optimal_samples': 10000, 'handles_high_dim': True},
            'naive_bayes': {'min_samples': 20, 'optimal_samples': 500, 'handles_high_dim': False},
            'knn': {'min_samples': 30, 'optimal_samples': 2000, 'handles_high_dim': False}
        }
        
        prefs = size_preferences.get(model_key, {'min_samples': 50, 'optimal_samples': 1000, 'handles_high_dim': True})
        
        # Sample size scoring
        if n_samples < prefs['min_samples']:
            sample_score = 0.3
        elif n_samples >= prefs['optimal_samples']:
            sample_score = 1.0
        else:
            # Linear interpolation between min and optimal
            sample_score = 0.3 + 0.7 * (n_samples - prefs['min_samples']) / (prefs['optimal_samples'] - prefs['min_samples'])
        
        # High dimensionality penalty
        if n_features > 0:
            if n_features > n_samples and not prefs['handles_high_dim']:
                sample_score *= 0.5  # Penalty for high-dimensional data
            elif n_features > 100 and not prefs['handles_high_dim']:
                sample_score *= 0.8  # Smaller penalty for moderately high dimensions
        
        return sample_score
    
    def _evaluate_subtype_suitability(self, model_key: str, task_subtype: str) -> float:
        """
        Evaluate model suitability based on task subtype.
        
        Args:
            model_key: Model identifier
            task_subtype: Task subtype (binary, multiclass, linear, non_linear)
            
        Returns:
            Subtype suitability score (0-1)
        """
        subtype_preferences = {
            'decision_tree': {'binary': 0.9, 'multiclass': 0.9, 'linear': 0.7, 'non_linear': 0.9},
            'random_forest': {'binary': 0.9, 'multiclass': 0.9, 'linear': 0.8, 'non_linear': 0.9},
            'svm': {'binary': 0.9, 'multiclass': 0.8, 'linear': 0.9, 'non_linear': 0.9},
            'linear_regression': {'binary': 0.0, 'multiclass': 0.0, 'linear': 0.9, 'non_linear': 0.3},
            'logistic_regression': {'binary': 0.9, 'multiclass': 0.8, 'linear': 0.9, 'non_linear': 0.4},
            'gradient_boosting': {'binary': 0.9, 'multiclass': 0.9, 'linear': 0.8, 'non_linear': 0.9},
            'naive_bayes': {'binary': 0.8, 'multiclass': 0.8, 'linear': 0.6, 'non_linear': 0.6},
            'knn': {'binary': 0.8, 'multiclass': 0.8, 'linear': 0.6, 'non_linear': 0.8}
        }
        
        preferences = subtype_preferences.get(model_key, {})
        return preferences.get(task_subtype, 0.5)
    
    def _evaluate_balance_suitability(self, model_key: str, dataset_balance: Dict[str, Any]) -> float:
        """
        Evaluate model suitability based on dataset balance.
        
        Args:
            model_key: Model identifier
            dataset_balance: Dataset balance information
            
        Returns:
            Balance suitability score (0-1)
        """
        if not dataset_balance:
            return 0.5
            
        balance_status = dataset_balance.get('balance_status', 'unknown')
        
        # Model preferences for imbalanced data
        imbalance_handling = {
            'decision_tree': {'balanced': 0.8, 'slightly_imbalanced': 0.7, 'moderately_imbalanced': 0.6, 'severely_imbalanced': 0.4},
            'random_forest': {'balanced': 0.9, 'slightly_imbalanced': 0.8, 'moderately_imbalanced': 0.7, 'severely_imbalanced': 0.6},
            'svm': {'balanced': 0.8, 'slightly_imbalanced': 0.7, 'moderately_imbalanced': 0.5, 'severely_imbalanced': 0.3},
            'logistic_regression': {'balanced': 0.8, 'slightly_imbalanced': 0.7, 'moderately_imbalanced': 0.6, 'severely_imbalanced': 0.4},
            'gradient_boosting': {'balanced': 0.9, 'slightly_imbalanced': 0.8, 'moderately_imbalanced': 0.7, 'severely_imbalanced': 0.6},
            'naive_bayes': {'balanced': 0.7, 'slightly_imbalanced': 0.6, 'moderately_imbalanced': 0.5, 'severely_imbalanced': 0.4},
            'knn': {'balanced': 0.7, 'slightly_imbalanced': 0.6, 'moderately_imbalanced': 0.4, 'severely_imbalanced': 0.3}
        }
        
        preferences = imbalance_handling.get(model_key, {})
        return preferences.get(balance_status, 0.5)
    
    def _evaluate_complexity_suitability(self, model_key: str, 
                                       complexity_indicators: Dict[str, Any],
                                       data_characteristics: Dict[str, Any]) -> float:
        """
        Evaluate model suitability based on data complexity.
        
        Args:
            model_key: Model identifier
            complexity_indicators: Complexity indicators from task analysis
            data_characteristics: Additional data characteristics
            
        Returns:
            Complexity suitability score (0-1)
        """
        score = 0.5  # Base score
        
        # Multicollinearity handling
        multicollinearity_risk = complexity_indicators.get('multicollinearity_risk', 'unknown')
        if multicollinearity_risk == 'high':
            if model_key in ['random_forest', 'gradient_boosting', 'decision_tree']:
                score += 0.2  # These models handle multicollinearity well
            elif model_key in ['linear_regression', 'logistic_regression']:
                score -= 0.2  # Linear models struggle with multicollinearity
        
        # High dimensionality handling
        dimensionality = complexity_indicators.get('dimensionality', 'normal')
        if dimensionality == 'high':
            if model_key in ['svm', 'random_forest', 'gradient_boosting']:
                score += 0.1  # Good for high-dimensional data
            elif model_key in ['knn', 'naive_bayes']:
                score -= 0.2  # Struggle with high dimensions
        
        # Interpretability requirements
        interpretability_needed = data_characteristics.get('interpretability_required', False)
        if interpretability_needed:
            if model_key in ['decision_tree', 'linear_regression', 'logistic_regression']:
                score += 0.2  # Highly interpretable
            elif model_key in ['random_forest', 'gradient_boosting']:
                score -= 0.1  # Less interpretable
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_statistical_suitability(self, model_key: str, 
                                        statistical_properties: StatisticalProperties) -> float:
        """
        Evaluate model suitability based on statistical properties.
        
        Args:
            model_key: Model identifier
            statistical_properties: Statistical properties of the dataset
            
        Returns:
            Statistical suitability score (0-1)
        """
        score = 0.5  # Base score
        
        # Distribution considerations
        distribution = statistical_properties.distribution.lower()
        
        if 'normal' in distribution:
            if model_key in ['linear_regression', 'logistic_regression', 'svm']:
                score += 0.1  # Benefit from normal distribution
        elif 'skewed' in distribution:
            if model_key in ['decision_tree', 'random_forest', 'gradient_boosting']:
                score += 0.1  # Handle skewed data well
        
        # Outlier considerations (if outlier information is available)
        if hasattr(statistical_properties, 'outlier_indices') and statistical_properties.outlier_indices:
            outlier_ratio = len(statistical_properties.outlier_indices) / 1000  # Assume reasonable dataset size
            if outlier_ratio > 0.05:  # More than 5% outliers
                if model_key in ['decision_tree', 'random_forest', 'gradient_boosting']:
                    score += 0.1  # Robust to outliers
                elif model_key in ['linear_regression', 'logistic_regression', 'svm']:
                    score -= 0.1  # Sensitive to outliers
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_efficiency_requirements(self, model_key: str, 
                                        data_characteristics: Dict[str, Any]) -> float:
        """
        Evaluate model suitability based on efficiency requirements.
        
        Args:
            model_key: Model identifier
            data_characteristics: Data characteristics including efficiency requirements
            
        Returns:
            Efficiency suitability score (0-1)
        """
        score = 0.5  # Base score
        
        # Training speed requirements
        fast_training_needed = data_characteristics.get('fast_training_required', False)
        if fast_training_needed:
            model_info = self.traditional_ml_models.get(model_key, {})
            training_speed = model_info.get('training_speed', 'medium')
            if training_speed == 'fast':
                score += 0.2
            elif training_speed == 'slow':
                score -= 0.2
        
        # Prediction speed requirements
        fast_prediction_needed = data_characteristics.get('fast_prediction_required', False)
        if fast_prediction_needed:
            model_info = self.traditional_ml_models.get(model_key, {})
            prediction_speed = model_info.get('prediction_speed', 'medium')
            if prediction_speed == 'fast':
                score += 0.2
            elif prediction_speed == 'slow':
                score -= 0.2
        
        # Memory constraints
        memory_constrained = data_characteristics.get('memory_constrained', False)
        if memory_constrained:
            if model_key in ['linear_regression', 'logistic_regression', 'naive_bayes', 'svm']:
                score += 0.1  # Memory efficient
            elif model_key in ['random_forest', 'gradient_boosting']:
                score -= 0.1  # More memory intensive
        
        return max(0.0, min(1.0, score)) 
   
    def _suggest_hyperparameters(self, model_key: str, 
                               task_analysis: SupervisedTaskAnalysis,
                               data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the given model.
        
        Args:
            model_key: Model identifier
            task_analysis: Task analysis results
            data_characteristics: Data characteristics
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        n_samples = task_analysis.complexity_indicators.get('n_samples', 1000)
        n_features = task_analysis.complexity_indicators.get('n_features', 10)
        
        hyperparameters = {}
        
        if model_key == 'decision_tree':
            hyperparameters = {
                'max_depth': min(10, max(3, int(np.log2(n_samples)))),
                'min_samples_split': max(2, int(n_samples * 0.01)),
                'min_samples_leaf': max(1, int(n_samples * 0.005)),
                'random_state': 42
            }
            
        elif model_key == 'random_forest':
            hyperparameters = {
                'n_estimators': min(200, max(50, int(n_samples / 10))),
                'max_depth': min(15, max(5, int(np.log2(n_samples)))),
                'min_samples_split': max(2, int(n_samples * 0.01)),
                'min_samples_leaf': max(1, int(n_samples * 0.005)),
                'random_state': 42
            }
            
        elif model_key == 'svm':
            hyperparameters = {
                'C': 1.0,
                'kernel': 'rbf' if n_features < 1000 else 'linear',
                'gamma': 'scale',
                'random_state': 42
            }
            
        elif model_key == 'linear_regression':
            hyperparameters = {
                'fit_intercept': True,
                'normalize': False  # Deprecated in newer sklearn versions
            }
            
        elif model_key == 'logistic_regression':
            hyperparameters = {
                'C': 1.0,
                'max_iter': min(1000, max(100, n_features * 10)),
                'random_state': 42,
                'solver': 'liblinear' if n_samples < 10000 else 'lbfgs'
            }
            
        elif model_key == 'gradient_boosting':
            hyperparameters = {
                'n_estimators': min(200, max(50, int(n_samples / 20))),
                'learning_rate': 0.1,
                'max_depth': min(8, max(3, int(np.log2(n_samples)))),
                'random_state': 42
            }
            
        elif model_key == 'naive_bayes':
            if task_analysis.task_type == 'classification':
                hyperparameters = {
                    'alpha': 1.0  # Laplace smoothing
                }
                
        elif model_key == 'knn':
            hyperparameters = {
                'n_neighbors': min(20, max(3, int(np.sqrt(n_samples)))),
                'weights': 'distance',
                'algorithm': 'auto'
            }
        
        return hyperparameters
    
    def _estimate_performance(self, model_key: str, 
                            task_analysis: SupervisedTaskAnalysis,
                            suitability_score: float) -> Dict[str, float]:
        """
        Estimate expected performance for the model.
        
        Args:
            model_key: Model identifier
            task_analysis: Task analysis results
            suitability_score: Model suitability score
            
        Returns:
            Dictionary of performance estimates
        """
        # Base performance estimates (these would ideally come from empirical studies)
        base_performance = {
            'decision_tree': {'accuracy': 0.75, 'precision': 0.75, 'recall': 0.75, 'f1': 0.75},
            'random_forest': {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1': 0.85},
            'svm': {'accuracy': 0.82, 'precision': 0.82, 'recall': 0.82, 'f1': 0.82},
            'linear_regression': {'r2': 0.70, 'mse': 0.30, 'mae': 0.25},
            'logistic_regression': {'accuracy': 0.78, 'precision': 0.78, 'recall': 0.78, 'f1': 0.78},
            'gradient_boosting': {'accuracy': 0.87, 'precision': 0.87, 'recall': 0.87, 'f1': 0.87},
            'naive_bayes': {'accuracy': 0.72, 'precision': 0.72, 'recall': 0.72, 'f1': 0.72},
            'knn': {'accuracy': 0.76, 'precision': 0.76, 'recall': 0.76, 'f1': 0.76}
        }
        
        base_perf = base_performance.get(model_key, {'accuracy': 0.70})
        
        # Adjust based on suitability score
        adjusted_performance = {}
        for metric, value in base_perf.items():
            # Adjust performance based on suitability (higher suitability = better expected performance)
            adjustment_factor = 0.8 + (suitability_score * 0.4)  # Range: 0.8 to 1.2
            adjusted_value = min(1.0, value * adjustment_factor)
            adjusted_performance[metric] = round(adjusted_value, 3)
        
        # Add confidence intervals
        confidence_intervals = {}
        for metric, value in adjusted_performance.items():
            confidence_intervals[f'{metric}_confidence_lower'] = max(0.0, value - 0.1)
            confidence_intervals[f'{metric}_confidence_upper'] = min(1.0, value + 0.1)
        
        adjusted_performance.update(confidence_intervals)
        
        return adjusted_performance
    
    def _generate_rationale(self, score: ModelSuitabilityScore) -> str:
        """
        Generate human-readable rationale for the model recommendation.
        
        Args:
            score: Model suitability score object
            
        Returns:
            Rationale string explaining why the model is recommended
        """
        rationale_parts = []
        
        # Overall suitability
        if score.suitability_score >= 0.8:
            rationale_parts.append(f"{score.model_name} is highly suitable for this task")
        elif score.suitability_score >= 0.6:
            rationale_parts.append(f"{score.model_name} is moderately suitable for this task")
        else:
            rationale_parts.append(f"{score.model_name} has limited suitability for this task")
        
        # Key factors
        top_factors = sorted(score.rationale_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for factor, impact in top_factors:
            if impact > 0.1:
                if factor == 'task_compatibility':
                    rationale_parts.append("excellent compatibility with the task type")
                elif factor == 'dataset_size':
                    rationale_parts.append("well-suited for the dataset size")
                elif factor == 'task_subtype':
                    rationale_parts.append("good match for the specific task characteristics")
                elif factor == 'dataset_balance':
                    rationale_parts.append("handles dataset imbalance appropriately")
                elif factor == 'complexity_handling':
                    rationale_parts.append("effective at handling data complexity")
                elif factor == 'statistical_properties':
                    rationale_parts.append("aligns well with the data's statistical properties")
                elif factor == 'efficiency':
                    rationale_parts.append("meets efficiency requirements")
            elif impact < -0.1:
                if factor == 'dataset_size':
                    rationale_parts.append("may struggle with the current dataset size")
                elif factor == 'complexity_handling':
                    rationale_parts.append("may have difficulty with data complexity")
        
        # Performance expectations
        expected_acc = score.performance_estimate.get('accuracy', score.performance_estimate.get('r2', 0))
        if expected_acc >= 0.85:
            rationale_parts.append("expected to deliver high performance")
        elif expected_acc >= 0.75:
            rationale_parts.append("expected to deliver good performance")
        else:
            rationale_parts.append("expected performance may be limited")
        
        return ". ".join(rationale_parts) + "." 
   
    def recommend_deep_learning(self, 
                              task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                              domain_analysis: Optional[Dict[str, Any]] = None,
                              data_characteristics: Optional[Dict[str, Any]] = None) -> List[ModelRecommendation]:
        """
        Recommend deep learning models based on task analysis and domain characteristics.
        
        Args:
            task_analysis: Analysis of the learning task (supervised or unsupervised)
            domain_analysis: Domain-specific analysis results (CV, NLP, time series)
            data_characteristics: Additional data characteristics
            
        Returns:
            List of ranked deep learning model recommendations
        """
        if data_characteristics is None:
            data_characteristics = {}
        if domain_analysis is None:
            domain_analysis = {}
            
        # Determine suitable deep learning models based on domain and task
        suitable_models = self._filter_deep_learning_models(task_analysis, domain_analysis)
        
        # Score each model based on suitability
        model_scores = []
        for model_key, model_info in suitable_models.items():
            score = self._score_deep_learning_model(
                model_key, model_info, task_analysis, domain_analysis, data_characteristics
            )
            model_scores.append(score)
        
        # Rank models by suitability score
        model_scores.sort(key=lambda x: x.suitability_score, reverse=True)
        
        # Convert to ModelRecommendation objects
        recommendations = []
        for score in model_scores:
            if score.suitability_score >= self.confidence_threshold:
                recommendation = ModelRecommendation(
                    model_name=score.model_name,
                    model_type='deep_learning',
                    confidence=score.suitability_score,
                    rationale=self._generate_deep_learning_rationale(score, domain_analysis),
                    hyperparameters=score.hyperparameter_suggestions,
                    expected_performance=score.performance_estimate
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _filter_deep_learning_models(self, 
                                   task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                                   domain_analysis: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Filter deep learning models based on task type and domain.
        
        Args:
            task_analysis: Task analysis results
            domain_analysis: Domain-specific analysis
            
        Returns:
            Dictionary of suitable deep learning models
        """
        suitable_models = {}
        
        # Check for computer vision tasks
        if 'cv_analysis' in domain_analysis:
            cv_analysis = domain_analysis['cv_analysis']
            if hasattr(cv_analysis, 'task_type') and cv_analysis.task_type in ['classification', 'object_detection', 'segmentation']:
                suitable_models['cnn'] = self.deep_learning_models['cnn']
        
        # Check for NLP tasks
        if 'nlp_analysis' in domain_analysis:
            nlp_analysis = domain_analysis['nlp_analysis']
            if hasattr(nlp_analysis, 'task_type'):
                if nlp_analysis.task_type in ['sentiment', 'classification', 'ner']:
                    suitable_models['transformer'] = self.deep_learning_models['transformer']
                if nlp_analysis.task_type in ['ner', 'sequence_modeling']:
                    suitable_models['rnn_lstm'] = self.deep_learning_models['rnn_lstm']
        
        # Check for time series tasks
        if 'timeseries_analysis' in domain_analysis:
            ts_analysis = domain_analysis['timeseries_analysis']
            if hasattr(ts_analysis, 'series_type'):
                suitable_models['rnn_lstm'] = self.deep_learning_models['rnn_lstm']
                # Transformers can also be good for time series
                suitable_models['transformer'] = self.deep_learning_models['transformer']
        
        # Check for unsupervised tasks
        if isinstance(task_analysis, UnsupervisedTaskAnalysis):
            if task_analysis.dimensionality_reduction_needed or task_analysis.clustering_potential > 0.7:
                suitable_models['autoencoder'] = self.deep_learning_models['autoencoder']
        
        # If no domain-specific models found, consider general deep learning models
        if not suitable_models and isinstance(task_analysis, SupervisedTaskAnalysis):
            # For general supervised tasks, consider all models with lower confidence
            if task_analysis.complexity_indicators.get('n_samples', 0) > 1000:  # Need large datasets for DL
                suitable_models.update(self.deep_learning_models)
        
        return suitable_models
    
    def _score_deep_learning_model(self, 
                                 model_key: str,
                                 model_info: Dict[str, Any],
                                 task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                                 domain_analysis: Dict[str, Any],
                                 data_characteristics: Dict[str, Any]) -> ModelSuitabilityScore:
        """
        Score a deep learning model based on its suitability for the task.
        
        Args:
            model_key: Key identifier for the model
            model_info: Model information dictionary
            task_analysis: Analysis of the learning task
            domain_analysis: Domain-specific analysis
            data_characteristics: Additional data characteristics
            
        Returns:
            ModelSuitabilityScore object
        """
        rationale_factors = {}
        base_score = 0.4  # Lower base score for deep learning (requires more data/compute)
        
        # Domain compatibility scoring
        domain_score = self._evaluate_domain_compatibility(model_key, domain_analysis)
        base_score += domain_score * 0.3
        rationale_factors['domain_compatibility'] = domain_score * 0.3
        
        # Dataset size requirements for deep learning
        if isinstance(task_analysis, SupervisedTaskAnalysis):
            n_samples = task_analysis.complexity_indicators.get('n_samples', 0)
        else:
            n_samples = data_characteristics.get('n_samples', 0)
            
        size_score = self._evaluate_deep_learning_size_requirements(model_key, n_samples)
        base_score += size_score * 0.25
        rationale_factors['dataset_size'] = size_score * 0.25
        
        # Computational requirements assessment
        compute_score = self._evaluate_computational_requirements(model_key, data_characteristics)
        base_score += compute_score * 0.2
        rationale_factors['computational_feasibility'] = compute_score * 0.2
        
        # Task complexity suitability
        complexity_score = self._evaluate_deep_learning_complexity_suitability(model_key, task_analysis, domain_analysis)
        base_score += complexity_score * 0.15
        rationale_factors['task_complexity'] = complexity_score * 0.15
        
        # Data type and structure compatibility
        structure_score = self._evaluate_data_structure_compatibility(model_key, domain_analysis, data_characteristics)
        base_score += structure_score * 0.1
        rationale_factors['data_structure'] = structure_score * 0.1
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, base_score))
        
        # Generate hyperparameter suggestions
        hyperparameters = self._suggest_deep_learning_hyperparameters(model_key, task_analysis, domain_analysis, data_characteristics)
        
        # Estimate performance
        performance_estimate = self._estimate_deep_learning_performance(model_key, task_analysis, final_score, domain_analysis)
        
        return ModelSuitabilityScore(
            model_name=model_info['name'],
            suitability_score=final_score,
            rationale_factors=rationale_factors,
            performance_estimate=performance_estimate,
            hyperparameter_suggestions=hyperparameters
        )
    
    def _evaluate_domain_compatibility(self, model_key: str, domain_analysis: Dict[str, Any]) -> float:
        """
        Evaluate model compatibility with the specific domain.
        
        Args:
            model_key: Model identifier
            domain_analysis: Domain-specific analysis results
            
        Returns:
            Domain compatibility score (0-1)
        """
        score = 0.5  # Base score
        
        # CNN for computer vision
        if model_key == 'cnn' and 'cv_analysis' in domain_analysis:
            cv_analysis = domain_analysis['cv_analysis']
            if hasattr(cv_analysis, 'task_type'):
                if cv_analysis.task_type in ['classification', 'object_detection', 'segmentation']:
                    score = 0.9
                    # Bonus for image classification
                    if cv_analysis.task_type == 'classification':
                        score = 0.95
        
        # RNN/LSTM for sequential data
        elif model_key == 'rnn_lstm':
            if 'timeseries_analysis' in domain_analysis:
                score = 0.85
            elif 'nlp_analysis' in domain_analysis:
                nlp_analysis = domain_analysis['nlp_analysis']
                if hasattr(nlp_analysis, 'task_type') and nlp_analysis.task_type in ['ner', 'sequence_modeling']:
                    score = 0.8
        
        # Transformer for NLP and some time series
        elif model_key == 'transformer':
            if 'nlp_analysis' in domain_analysis:
                nlp_analysis = domain_analysis['nlp_analysis']
                if hasattr(nlp_analysis, 'task_type'):
                    if nlp_analysis.task_type in ['sentiment', 'classification', 'ner']:
                        score = 0.9
                    elif nlp_analysis.task_type == 'conversation':
                        score = 0.95
            elif 'timeseries_analysis' in domain_analysis:
                # Transformers can be good for time series too
                score = 0.75
        
        # AutoEncoder for unsupervised tasks
        elif model_key == 'autoencoder':
            if 'unsupervised_potential' in domain_analysis:
                score = 0.8
            # Good for dimensionality reduction
            if domain_analysis.get('dimensionality_reduction_needed', False):
                score = 0.85
        
        return score
    
    def _evaluate_deep_learning_size_requirements(self, model_key: str, n_samples: int) -> float:
        """
        Evaluate if dataset size is sufficient for deep learning model.
        
        Args:
            model_key: Model identifier
            n_samples: Number of samples in dataset
            
        Returns:
            Size suitability score (0-1)
        """
        # Deep learning size requirements (generally higher than traditional ML)
        size_requirements = {
            'cnn': {'min_samples': 1000, 'optimal_samples': 10000},
            'rnn_lstm': {'min_samples': 500, 'optimal_samples': 5000},
            'transformer': {'min_samples': 5000, 'optimal_samples': 50000},
            'autoencoder': {'min_samples': 1000, 'optimal_samples': 10000}
        }
        
        requirements = size_requirements.get(model_key, {'min_samples': 1000, 'optimal_samples': 10000})
        
        if n_samples < requirements['min_samples']:
            return 0.2  # Very low score for insufficient data
        elif n_samples >= requirements['optimal_samples']:
            return 1.0  # Perfect score for optimal data size
        else:
            # Linear interpolation between min and optimal
            return 0.2 + 0.8 * (n_samples - requirements['min_samples']) / (requirements['optimal_samples'] - requirements['min_samples'])
    
    def _evaluate_computational_requirements(self, model_key: str, data_characteristics: Dict[str, Any]) -> float:
        """
        Evaluate computational feasibility for the deep learning model.
        
        Args:
            model_key: Model identifier
            data_characteristics: Data characteristics including compute constraints
            
        Returns:
            Computational feasibility score (0-1)
        """
        score = 0.5  # Base score
        
        # Check for GPU availability
        has_gpu = data_characteristics.get('gpu_available', False)
        if has_gpu:
            score += 0.3  # Significant bonus for GPU availability
        else:
            # Penalty for models that really need GPU
            if model_key in ['cnn', 'transformer']:
                score -= 0.2
        
        # Check computational constraints
        compute_constrained = data_characteristics.get('compute_constrained', False)
        if compute_constrained:
            # Penalty for computationally intensive models
            if model_key in ['transformer', 'cnn']:
                score -= 0.3
            elif model_key in ['rnn_lstm', 'autoencoder']:
                score -= 0.1
        
        # Check memory constraints
        memory_constrained = data_characteristics.get('memory_constrained', False)
        if memory_constrained:
            # All deep learning models are memory intensive
            score -= 0.2
            # Transformers are especially memory hungry
            if model_key == 'transformer':
                score -= 0.1
        
        # Training time constraints
        fast_training_needed = data_characteristics.get('fast_training_required', False)
        if fast_training_needed:
            # Deep learning models are generally slow to train
            score -= 0.2
            # Transformers are especially slow
            if model_key == 'transformer':
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_deep_learning_complexity_suitability(self, 
                                                      model_key: str,
                                                      task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                                                      domain_analysis: Dict[str, Any]) -> float:
        """
        Evaluate model suitability based on task complexity.
        
        Args:
            model_key: Model identifier
            task_analysis: Task analysis results
            domain_analysis: Domain-specific analysis
            
        Returns:
            Complexity suitability score (0-1)
        """
        score = 0.5  # Base score
        
        # Deep learning excels at complex tasks
        if isinstance(task_analysis, SupervisedTaskAnalysis):
            # High-dimensional data
            n_features = task_analysis.complexity_indicators.get('n_features', 0)
            if n_features > 100:
                score += 0.2
            elif n_features > 1000:
                score += 0.3
            
            # Complex task subtypes
            if task_analysis.task_subtype == 'non_linear':
                score += 0.2
            elif task_analysis.task_subtype == 'multiclass' and task_analysis.target_characteristics.get('unique_values', 0) > 10:
                score += 0.1
        
        # Domain-specific complexity
        if 'cv_analysis' in domain_analysis and model_key == 'cnn':
            cv_analysis = domain_analysis['cv_analysis']
            if hasattr(cv_analysis, 'num_classes') and cv_analysis.num_classes and cv_analysis.num_classes > 10:
                score += 0.1  # Bonus for many classes
        
        if 'nlp_analysis' in domain_analysis and model_key in ['transformer', 'rnn_lstm']:
            nlp_analysis = domain_analysis['nlp_analysis']
            if hasattr(nlp_analysis, 'text_characteristics'):
                text_chars = nlp_analysis.text_characteristics
                if text_chars.get('avg_length', 0) > 100:  # Long texts
                    score += 0.1
                if text_chars.get('vocabulary_size', 0) > 10000:  # Large vocabulary
                    score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_data_structure_compatibility(self, 
                                             model_key: str,
                                             domain_analysis: Dict[str, Any],
                                             data_characteristics: Dict[str, Any]) -> float:
        """
        Evaluate compatibility with data structure and format.
        
        Args:
            model_key: Model identifier
            domain_analysis: Domain-specific analysis
            data_characteristics: Data characteristics
            
        Returns:
            Data structure compatibility score (0-1)
        """
        score = 0.5  # Base score
        
        data_structure = data_characteristics.get('data_structure', 'unknown')
        
        # CNN for image data
        if model_key == 'cnn':
            if data_structure == 'image' or 'cv_analysis' in domain_analysis:
                score = 0.9
            else:
                score = 0.1  # CNN not suitable for non-image data
        
        # RNN/LSTM for sequential data
        elif model_key == 'rnn_lstm':
            if data_structure in ['time_series', 'text'] or 'timeseries_analysis' in domain_analysis or 'nlp_analysis' in domain_analysis:
                score = 0.8
            else:
                score = 0.3  # Can work with other data but not optimal
        
        # Transformer for text and sequential data
        elif model_key == 'transformer':
            if data_structure == 'text' or 'nlp_analysis' in domain_analysis:
                score = 0.9
            elif data_structure == 'time_series' or 'timeseries_analysis' in domain_analysis:
                score = 0.7
            else:
                score = 0.2  # Not ideal for other data types
        
        # AutoEncoder is more flexible
        elif model_key == 'autoencoder':
            if data_structure in ['tabular', 'image', 'text']:
                score = 0.7
            else:
                score = 0.5
        
        return score 
   
    def _suggest_deep_learning_hyperparameters(self, 
                                             model_key: str,
                                             task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                                             domain_analysis: Dict[str, Any],
                                             data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest hyperparameters for deep learning models.
        
        Args:
            model_key: Model identifier
            task_analysis: Task analysis results
            domain_analysis: Domain-specific analysis
            data_characteristics: Data characteristics
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if isinstance(task_analysis, SupervisedTaskAnalysis):
            n_samples = task_analysis.complexity_indicators.get('n_samples', 1000)
            n_features = task_analysis.complexity_indicators.get('n_features', 10)
        else:
            n_samples = data_characteristics.get('n_samples', 1000)
            n_features = data_characteristics.get('n_features', 10)
        
        hyperparameters = {}
        
        if model_key == 'cnn':
            # CNN hyperparameters for image classification
            hyperparameters = {
                'architecture': 'sequential',
                'conv_layers': [
                    {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
                    {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                    {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
                ],
                'pooling': 'max_pooling',
                'dropout_rate': 0.25,
                'dense_layers': [128, 64],
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': min(64, max(16, n_samples // 100)),
                'epochs': min(100, max(20, n_samples // 500))
            }
            
            # Adjust based on domain analysis
            if 'cv_analysis' in domain_analysis:
                cv_analysis = domain_analysis['cv_analysis']
                if hasattr(cv_analysis, 'num_classes') and cv_analysis.num_classes:
                    hyperparameters['output_units'] = cv_analysis.num_classes
                    if cv_analysis.num_classes == 2:
                        hyperparameters['output_activation'] = 'sigmoid'
                        hyperparameters['loss'] = 'binary_crossentropy'
                    else:
                        hyperparameters['output_activation'] = 'softmax'
                        hyperparameters['loss'] = 'categorical_crossentropy'
        
        elif model_key == 'rnn_lstm':
            # RNN/LSTM hyperparameters
            hyperparameters = {
                'architecture': 'lstm',
                'lstm_units': [64, 32],
                'dropout_rate': 0.2,
                'recurrent_dropout': 0.2,
                'return_sequences': True,  # For stacked LSTM
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': min(32, max(8, n_samples // 200)),
                'epochs': min(50, max(10, n_samples // 1000))
            }
            
            # Adjust for time series vs NLP
            if 'timeseries_analysis' in domain_analysis:
                ts_analysis = domain_analysis['timeseries_analysis']
                if hasattr(ts_analysis, 'series_type'):
                    if ts_analysis.series_type == 'multivariate':
                        hyperparameters['lstm_units'] = [128, 64]  # More units for multivariate
                    
                    if hasattr(ts_analysis, 'recommended_task'):
                        if ts_analysis.recommended_task == 'forecasting':
                            hyperparameters['loss'] = 'mse'
                            hyperparameters['output_activation'] = 'linear'
                        else:
                            hyperparameters['loss'] = 'categorical_crossentropy'
                            hyperparameters['output_activation'] = 'softmax'
            
            elif 'nlp_analysis' in domain_analysis:
                nlp_analysis = domain_analysis['nlp_analysis']
                if hasattr(nlp_analysis, 'text_characteristics'):
                    text_chars = nlp_analysis.text_characteristics
                    avg_length = text_chars.get('avg_length', 100)
                    if avg_length > 200:
                        hyperparameters['lstm_units'] = [128, 64]  # More units for longer texts
        
        elif model_key == 'transformer':
            # Transformer hyperparameters
            hyperparameters = {
                'architecture': 'transformer',
                'num_layers': 6,
                'num_heads': 8,
                'hidden_size': 512,
                'intermediate_size': 2048,
                'dropout_rate': 0.1,
                'attention_dropout': 0.1,
                'optimizer': 'adamw',
                'learning_rate': 5e-5,
                'batch_size': min(16, max(4, n_samples // 1000)),  # Smaller batches for transformers
                'epochs': min(20, max(3, n_samples // 5000)),
                'warmup_steps': 500
            }
            
            # Adjust based on task complexity
            if n_samples > 10000:
                hyperparameters['num_layers'] = 12
                hyperparameters['hidden_size'] = 768
            elif n_samples < 5000:
                hyperparameters['num_layers'] = 4
                hyperparameters['hidden_size'] = 256
                hyperparameters['num_heads'] = 4
            
            # NLP-specific adjustments
            if 'nlp_analysis' in domain_analysis:
                nlp_analysis = domain_analysis['nlp_analysis']
                if hasattr(nlp_analysis, 'task_type'):
                    if nlp_analysis.task_type == 'sentiment':
                        hyperparameters['output_units'] = 2
                        hyperparameters['output_activation'] = 'softmax'
                    elif nlp_analysis.task_type == 'ner':
                        hyperparameters['architecture'] = 'transformer_token_classification'
                        hyperparameters['output_activation'] = 'softmax'
        
        elif model_key == 'autoencoder':
            # AutoEncoder hyperparameters
            encoder_dims = []
            current_dim = n_features
            
            # Create encoder dimensions (gradually reducing)
            while current_dim > 32 and len(encoder_dims) < 4:
                current_dim = max(32, current_dim // 2)
                encoder_dims.append(current_dim)
            
            hyperparameters = {
                'architecture': 'autoencoder',
                'encoder_dims': encoder_dims,
                'decoder_dims': encoder_dims[::-1][1:] + [n_features],  # Mirror encoder + original dim
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': min(128, max(32, n_samples // 50)),
                'epochs': min(200, max(50, n_samples // 100)),
                'loss': 'mse'
            }
            
            # Adjust for different data types
            data_structure = data_characteristics.get('data_structure', 'tabular')
            if data_structure == 'image':
                hyperparameters['architecture'] = 'convolutional_autoencoder'
                hyperparameters['conv_encoder'] = [
                    {'filters': 32, 'kernel_size': (3, 3)},
                    {'filters': 64, 'kernel_size': (3, 3)},
                    {'filters': 128, 'kernel_size': (3, 3)}
                ]
        
        return hyperparameters
    
    def _estimate_deep_learning_performance(self, 
                                          model_key: str,
                                          task_analysis: Union[SupervisedTaskAnalysis, UnsupervisedTaskAnalysis],
                                          suitability_score: float,
                                          domain_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate expected performance for deep learning models.
        
        Args:
            model_key: Model identifier
            task_analysis: Task analysis results
            suitability_score: Model suitability score
            domain_analysis: Domain-specific analysis
            
        Returns:
            Dictionary of performance estimates
        """
        # Base performance estimates for deep learning models
        # These are generally higher than traditional ML when sufficient data is available
        base_performance = {
            'cnn': {'accuracy': 0.88, 'precision': 0.88, 'recall': 0.88, 'f1': 0.88},
            'rnn_lstm': {'accuracy': 0.82, 'precision': 0.82, 'recall': 0.82, 'f1': 0.82, 'mse': 0.15},
            'transformer': {'accuracy': 0.92, 'precision': 0.92, 'recall': 0.92, 'f1': 0.92},
            'autoencoder': {'reconstruction_loss': 0.05, 'compression_ratio': 0.8, 'anomaly_detection_auc': 0.85}
        }
        
        base_perf = base_performance.get(model_key, {'accuracy': 0.80})
        
        # Adjust based on suitability score and domain
        adjusted_performance = {}
        
        for metric, value in base_perf.items():
            # Adjust performance based on suitability
            adjustment_factor = 0.7 + (suitability_score * 0.6)  # Range: 0.7 to 1.3
            adjusted_value = min(1.0, value * adjustment_factor)
            
            # Domain-specific adjustments
            if model_key == 'cnn' and 'cv_analysis' in domain_analysis:
                # CNN performs very well on image tasks
                adjusted_value = min(1.0, adjusted_value * 1.1)
            elif model_key == 'transformer' and 'nlp_analysis' in domain_analysis:
                # Transformers excel at NLP
                adjusted_value = min(1.0, adjusted_value * 1.15)
            elif model_key == 'rnn_lstm' and 'timeseries_analysis' in domain_analysis:
                # LSTM good for time series
                adjusted_value = min(1.0, adjusted_value * 1.05)
            
            adjusted_performance[metric] = round(adjusted_value, 3)
        
        # Add confidence intervals
        confidence_intervals = {}
        for metric, value in adjusted_performance.items():
            # Wider confidence intervals for deep learning due to higher variance
            confidence_intervals[f'{metric}_confidence_lower'] = max(0.0, value - 0.15)
            confidence_intervals[f'{metric}_confidence_upper'] = min(1.0, value + 0.1)
        
        adjusted_performance.update(confidence_intervals)
        
        # Add training time estimates
        if isinstance(task_analysis, SupervisedTaskAnalysis):
            n_samples = task_analysis.complexity_indicators.get('n_samples', 1000)
        else:
            n_samples = 1000
            
        # Rough training time estimates (in minutes)
        training_time_estimates = {
            'cnn': max(30, n_samples / 100),
            'rnn_lstm': max(20, n_samples / 200),
            'transformer': max(60, n_samples / 50),
            'autoencoder': max(15, n_samples / 300)
        }
        
        adjusted_performance['estimated_training_time_minutes'] = training_time_estimates.get(model_key, 30)
        
        return adjusted_performance
    
    def _generate_deep_learning_rationale(self, 
                                        score: ModelSuitabilityScore,
                                        domain_analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable rationale for deep learning model recommendations.
        
        Args:
            score: Model suitability score object
            domain_analysis: Domain-specific analysis results
            
        Returns:
            Rationale string explaining why the model is recommended
        """
        rationale_parts = []
        
        # Overall suitability
        if score.suitability_score >= 0.8:
            rationale_parts.append(f"{score.model_name} is highly suitable for this deep learning task")
        elif score.suitability_score >= 0.6:
            rationale_parts.append(f"{score.model_name} is moderately suitable for this deep learning task")
        else:
            rationale_parts.append(f"{score.model_name} has limited suitability for this deep learning task")
        
        # Key factors
        top_factors = sorted(score.rationale_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for factor, impact in top_factors:
            if impact > 0.15:
                if factor == 'domain_compatibility':
                    if 'cv_analysis' in domain_analysis:
                        rationale_parts.append("excellent fit for computer vision tasks")
                    elif 'nlp_analysis' in domain_analysis:
                        rationale_parts.append("well-suited for natural language processing")
                    elif 'timeseries_analysis' in domain_analysis:
                        rationale_parts.append("appropriate for time series analysis")
                    else:
                        rationale_parts.append("good domain compatibility")
                elif factor == 'dataset_size':
                    rationale_parts.append("sufficient data available for deep learning")
                elif factor == 'computational_feasibility':
                    rationale_parts.append("computationally feasible with available resources")
                elif factor == 'task_complexity':
                    rationale_parts.append("well-suited for complex pattern recognition")
                elif factor == 'data_structure':
                    rationale_parts.append("compatible with the data structure and format")
            elif impact < -0.15:
                if factor == 'dataset_size':
                    rationale_parts.append("may require more data for optimal performance")
                elif factor == 'computational_feasibility':
                    rationale_parts.append("may be computationally demanding")
        
        # Performance expectations
        expected_acc = score.performance_estimate.get('accuracy', 
                      score.performance_estimate.get('f1', 
                      score.performance_estimate.get('reconstruction_loss', 0)))
        
        if expected_acc >= 0.9:
            rationale_parts.append("expected to deliver excellent performance")
        elif expected_acc >= 0.8:
            rationale_parts.append("expected to deliver strong performance")
        elif expected_acc >= 0.7:
            rationale_parts.append("expected to deliver good performance")
        else:
            rationale_parts.append("performance may be limited by data or computational constraints")
        
        # Training considerations
        training_time = score.performance_estimate.get('estimated_training_time_minutes', 0)
        if training_time > 120:  # More than 2 hours
            rationale_parts.append("requires significant training time")
        elif training_time > 60:  # More than 1 hour
            rationale_parts.append("requires moderate training time")
        
        return ". ".join(rationale_parts) + "."