"""
Complexity analyzer for performance estimation and overfitting risk assessment.

This module provides comprehensive complexity analysis including computational
resource estimation, overfitting risk detection, and model complexity assessment.
"""

import numpy as np
import pandas as pd
import psutil
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings

from ..core.data_models import (
    ResourceEstimate, OverfittingRisk, DatasetComplexity, ComplexityAnalysis
)
from ..core.exceptions import NeuroLiteException


class ComplexityAnalyzer:
    """
    Comprehensive complexity analyzer for performance estimation and risk assessment.
    
    This class provides methods for estimating computational requirements,
    detecting overfitting risks, and assessing dataset complexity to support
    automated ML pipeline optimization.
    """
    
    def __init__(self):
        """Initialize the ComplexityAnalyzer."""
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = psutil.cpu_count()
        
        # Complexity thresholds
        self.high_dimensionality_threshold = 100
        self.small_sample_threshold = 1000
        self.high_cardinality_threshold = 50
        
    def estimate_computational_requirements(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        task_type: str = "classification"
    ) -> ResourceEstimate:
        """
        Estimate computational resource requirements for the dataset.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            task_type: Type of ML task ('classification', 'regression', 'clustering')
            
        Returns:
            Resource requirement estimates
        """
        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
            memory_usage_mb = X.memory_usage(deep=True).sum() / (1024**2)
        else:
            n_samples, n_features = X.shape
            memory_usage_mb = X.nbytes / (1024**2)
        
        # CPU vs GPU suitability assessment
        cpu_suitability, gpu_suitability = self._assess_hardware_suitability(
            n_samples, n_features, task_type
        )
        
        # Memory requirement prediction
        predicted_memory_mb = self._predict_memory_requirements(
            n_samples, n_features, memory_usage_mb, task_type
        )
        
        # Processing time estimation
        processing_time = self._estimate_processing_time(
            n_samples, n_features, task_type
        )
        
        # Hardware recommendation
        if gpu_suitability > cpu_suitability and gpu_suitability > 0.7:
            recommended_hardware = "gpu"
        elif n_samples > 100000 or n_features > 1000:
            recommended_hardware = "distributed"
        else:
            recommended_hardware = "cpu"
        
        # Generate rationale
        rationale = self._generate_resource_rationale(
            n_samples, n_features, cpu_suitability, gpu_suitability, 
            predicted_memory_mb, processing_time
        )
        
        # Scaling factors
        scaling_factors = {
            "sample_scaling": min(1.0, n_samples / 10000),
            "feature_scaling": min(1.0, n_features / 100),
            "memory_scaling": min(1.0, predicted_memory_mb / 1024),
            "complexity_scaling": min(1.0, (n_samples * n_features) / 1000000)
        }
        
        # Confidence based on data characteristics
        confidence = self._calculate_resource_confidence(n_samples, n_features, task_type)
        
        return ResourceEstimate(
            cpu_suitability=cpu_suitability,
            gpu_suitability=gpu_suitability,
            memory_requirement_mb=predicted_memory_mb,
            processing_time_seconds=processing_time,
            recommended_hardware=recommended_hardware,
            confidence=confidence,
            rationale=rationale,
            scaling_factors=scaling_factors
        )
    
    def _assess_hardware_suitability(
        self, 
        n_samples: int, 
        n_features: int, 
        task_type: str
    ) -> Tuple[float, float]:
        """
        Assess CPU vs GPU suitability for the given dataset characteristics.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            task_type: Type of ML task
            
        Returns:
            Tuple of (cpu_suitability, gpu_suitability) scores
        """
        # Base suitability scores
        cpu_base = 0.8
        gpu_base = 0.3
        
        # Adjust based on dataset size
        if n_samples > 50000:
            gpu_base += 0.3
            cpu_base -= 0.1
        
        if n_features > 500:
            gpu_base += 0.2
            cpu_base -= 0.1
        
        # Adjust based on task type
        if task_type in ["deep_learning", "neural_network"]:
            gpu_base += 0.4
            cpu_base -= 0.2
        elif task_type in ["tree_based", "linear"]:
            cpu_base += 0.1
            gpu_base -= 0.1
        
        # Adjust based on computational complexity
        complexity_factor = (n_samples * n_features) / 1000000
        if complexity_factor > 1:
            gpu_base += min(0.3, complexity_factor * 0.1)
            cpu_base -= min(0.2, complexity_factor * 0.05)
        
        # Normalize to [0, 1] range
        cpu_suitability = max(0.0, min(1.0, cpu_base))
        gpu_suitability = max(0.0, min(1.0, gpu_base))
        
        return cpu_suitability, gpu_suitability
    
    def _predict_memory_requirements(
        self, 
        n_samples: int, 
        n_features: int, 
        current_memory_mb: float,
        task_type: str
    ) -> float:
        """
        Predict memory requirements for ML processing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            current_memory_mb: Current memory usage
            task_type: Type of ML task
            
        Returns:
            Predicted memory requirement in MB
        """
        # Base memory requirement (current data + overhead)
        base_memory = current_memory_mb * 2  # 2x for processing overhead
        
        # Model-specific memory requirements
        model_memory_factor = {
            "linear": 1.2,
            "tree_based": 1.5,
            "ensemble": 2.0,
            "neural_network": 3.0,
            "deep_learning": 5.0,
            "clustering": 1.3
        }.get(task_type, 1.5)
        
        # Feature engineering overhead
        feature_memory = (n_features * n_samples * 8) / (1024**2)  # 8 bytes per float64
        
        # Cross-validation memory overhead
        cv_memory = base_memory * 0.5  # Additional memory for CV folds
        
        # Total predicted memory
        total_memory = (base_memory + feature_memory + cv_memory) * model_memory_factor
        
        # Add safety margin
        total_memory *= 1.3
        
        return total_memory
    
    def _estimate_processing_time(
        self, 
        n_samples: int, 
        n_features: int, 
        task_type: str
    ) -> float:
        """
        Estimate processing time for the ML task.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            task_type: Type of ML task
            
        Returns:
            Estimated processing time in seconds
        """
        # Base time complexity factors
        complexity_factors = {
            "linear": 1.0,
            "tree_based": 2.0,
            "ensemble": 5.0,
            "neural_network": 10.0,
            "deep_learning": 30.0,
            "clustering": 3.0
        }
        
        base_factor = complexity_factors.get(task_type, 2.0)
        
        # Time estimation based on empirical observations
        # Base time for 1000 samples, 10 features
        base_time = 1.0  # seconds
        
        # Scale with sample size (roughly linear)
        sample_factor = n_samples / 1000
        
        # Scale with feature size (roughly quadratic for some algorithms)
        feature_factor = (n_features / 10) ** 1.5
        
        # CPU performance factor (assume average modern CPU)
        cpu_factor = max(1.0, 8 / self.cpu_count)
        
        estimated_time = base_time * base_factor * sample_factor * feature_factor * cpu_factor
        
        # Add cross-validation overhead (5-fold CV)
        estimated_time *= 5
        
        return estimated_time
    
    def _generate_resource_rationale(
        self,
        n_samples: int,
        n_features: int,
        cpu_suitability: float,
        gpu_suitability: float,
        memory_mb: float,
        processing_time: float
    ) -> str:
        """Generate human-readable rationale for resource recommendations."""
        rationale_parts = []
        
        # Dataset size assessment
        if n_samples > 100000:
            rationale_parts.append("Large dataset benefits from parallel processing")
        elif n_samples < 1000:
            rationale_parts.append("Small dataset suitable for standard processing")
        
        # Feature dimensionality
        if n_features > 1000:
            rationale_parts.append("High-dimensional data may benefit from GPU acceleration")
        
        # Hardware recommendation
        if gpu_suitability > cpu_suitability:
            rationale_parts.append("GPU processing recommended for computational efficiency")
        else:
            rationale_parts.append("CPU processing sufficient for this dataset size")
        
        # Memory considerations
        if memory_mb > self.system_memory_gb * 1024 * 0.8:
            rationale_parts.append("High memory requirements may need distributed processing")
        
        # Processing time
        if processing_time > 300:  # 5 minutes
            rationale_parts.append("Long processing time expected, consider optimization")
        
        return "; ".join(rationale_parts) if rationale_parts else "Standard processing requirements"
    
    def _calculate_resource_confidence(
        self, 
        n_samples: int, 
        n_features: int, 
        task_type: str
    ) -> float:
        """Calculate confidence in resource estimates."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on dataset characteristics
        if 1000 <= n_samples <= 100000:
            confidence += 0.1  # Good sample size range
        
        if 10 <= n_features <= 1000:
            confidence += 0.1  # Good feature range
        
        # Adjust based on task type familiarity
        if task_type in ["linear", "tree_based", "ensemble"]:
            confidence += 0.05  # Well-understood algorithms
        
        return min(1.0, confidence)
    
    def assess_dataset_complexity(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> DatasetComplexity:
        """
        Assess the complexity of the dataset for ML modeling.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Dataset complexity assessment
        """
        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
            X_array = X.values
        else:
            n_samples, n_features = X.shape
            X_array = X
        
        # Dimensionality complexity
        dimensionality_complexity = self._assess_dimensionality_complexity(n_samples, n_features)
        
        # Sample complexity
        sample_complexity = self._assess_sample_complexity(n_samples, n_features)
        
        # Class imbalance complexity (if classification task)
        class_imbalance_complexity = self._assess_class_imbalance_complexity(y)
        
        # Feature interaction complexity
        feature_interaction_complexity = self._assess_feature_interaction_complexity(X_array)
        
        # Noise level estimation
        noise_level = self._estimate_noise_level(X_array, y)
        
        # Overall complexity score
        complexity_factors = {
            "dimensionality": dimensionality_complexity,
            "sample_size": sample_complexity,
            "class_imbalance": class_imbalance_complexity,
            "feature_interactions": feature_interaction_complexity,
            "noise": noise_level
        }
        
        # Weighted average of complexity factors
        weights = {
            "dimensionality": 0.25,
            "sample_size": 0.20,
            "class_imbalance": 0.20,
            "feature_interactions": 0.20,
            "noise": 0.15
        }
        
        complexity_score = sum(
            complexity_factors[factor] * weights[factor] 
            for factor in complexity_factors
        )
        
        # Determine complexity level
        if complexity_score < 0.3:
            complexity_level = "low"
        elif complexity_score < 0.7:
            complexity_level = "medium"
        else:
            complexity_level = "high"
        
        # Confidence based on data quality and size
        confidence = self._calculate_complexity_confidence(n_samples, n_features)
        
        return DatasetComplexity(
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            dimensionality_complexity=dimensionality_complexity,
            sample_complexity=sample_complexity,
            class_imbalance_complexity=class_imbalance_complexity,
            feature_interaction_complexity=feature_interaction_complexity,
            noise_level=noise_level,
            confidence=confidence,
            complexity_factors=complexity_factors
        )
    
    def _assess_dimensionality_complexity(self, n_samples: int, n_features: int) -> float:
        """Assess complexity due to high dimensionality (curse of dimensionality)."""
        # Ratio of features to samples
        feature_sample_ratio = n_features / n_samples
        
        # High dimensionality indicators
        if feature_sample_ratio > 0.1:  # More than 10% features to samples
            return min(1.0, feature_sample_ratio * 2)
        elif n_features > self.high_dimensionality_threshold:
            return min(1.0, n_features / 1000)
        else:
            return max(0.0, feature_sample_ratio)
    
    def _assess_sample_complexity(self, n_samples: int, n_features: int) -> float:
        """Assess complexity due to sample size relative to problem difficulty."""
        # Ideal sample-to-feature ratio is typically 10:1 or higher
        ideal_ratio = 10
        actual_ratio = n_samples / max(1, n_features)
        
        if actual_ratio >= ideal_ratio:
            return 0.0  # Low complexity
        elif actual_ratio >= 5:
            return 0.3  # Medium complexity
        elif actual_ratio >= 2:
            return 0.6  # High complexity
        else:
            return 1.0  # Very high complexity
    
    def _assess_class_imbalance_complexity(self, y: Optional[Union[pd.Series, np.ndarray]]) -> float:
        """Assess complexity due to class imbalance."""
        if y is None:
            return 0.0  # No target variable, assume regression or unsupervised
        
        try:
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            # Check if it's a classification problem
            unique_values = np.unique(y_array)
            
            # If too many unique values, likely regression
            if len(unique_values) > 20:
                return 0.0
            
            # Calculate class distribution
            value_counts = pd.Series(y_array).value_counts()
            proportions = value_counts / len(y_array)
            
            # Calculate imbalance using entropy-based measure
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(len(unique_values))
            
            # Normalized entropy (1 = balanced, 0 = completely imbalanced)
            balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
            
            # Convert to complexity score (higher imbalance = higher complexity)
            imbalance_complexity = 1.0 - balance_score
            
            # Additional penalty for severe imbalance
            min_proportion = proportions.min()
            if min_proportion < 0.01:  # Less than 1% of minority class
                imbalance_complexity = min(1.0, imbalance_complexity + 0.3)
            
            return imbalance_complexity
            
        except Exception:
            return 0.0  # Default to no complexity if analysis fails
    
    def _assess_feature_interaction_complexity(self, X: np.ndarray) -> float:
        """Assess complexity due to feature interactions."""
        try:
            n_samples, n_features = X.shape
            
            # For large datasets, sample for efficiency
            if n_samples > 5000:
                sample_indices = np.random.choice(n_samples, 5000, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            # Remove non-numeric columns and handle missing values
            if X_sample.dtype == 'object':
                return 0.5  # Default complexity for non-numeric data
            
            # Handle missing values
            X_clean = X_sample[~np.isnan(X_sample).any(axis=1)]
            if len(X_clean) < 10:
                return 0.5  # Not enough clean data
            
            # Calculate correlation matrix
            try:
                correlation_matrix = np.corrcoef(X_clean.T)
                
                # Remove diagonal (self-correlations)
                correlation_matrix = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
                
                # Calculate interaction complexity based on correlation strength
                high_correlations = np.abs(correlation_matrix) > 0.7
                interaction_ratio = np.sum(high_correlations) / len(correlation_matrix)
                
                return min(1.0, interaction_ratio * 2)
                
            except Exception:
                return 0.5  # Default if correlation calculation fails
                
        except Exception:
            return 0.5  # Default complexity
    
    def _estimate_noise_level(
        self, 
        X: np.ndarray, 
        y: Optional[Union[pd.Series, np.ndarray]]
    ) -> float:
        """Estimate noise level in the data."""
        try:
            # For features, estimate noise using coefficient of variation
            feature_noise = 0.0
            numeric_features = 0
            
            for i in range(X.shape[1]):
                column = X[:, i]
                if np.issubdtype(column.dtype, np.number):
                    clean_column = column[~np.isnan(column)]
                    if len(clean_column) > 10:
                        mean_val = np.mean(clean_column)
                        std_val = np.std(clean_column)
                        if mean_val != 0:
                            cv = std_val / abs(mean_val)
                            feature_noise += min(1.0, cv / 2)  # Normalize CV
                            numeric_features += 1
            
            if numeric_features > 0:
                feature_noise /= numeric_features
            else:
                feature_noise = 0.5  # Default for non-numeric data
            
            # For target variable, estimate noise using simple model fit
            target_noise = 0.0
            if y is not None:
                try:
                    # Quick noise estimation using linear model residuals
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split
                    
                    # Prepare data
                    X_clean = X[~np.isnan(X).any(axis=1)]
                    if isinstance(y, pd.Series):
                        y_clean = y.values[~np.isnan(X).any(axis=1)]
                    else:
                        y_clean = y[~np.isnan(X).any(axis=1)]
                    
                    if len(X_clean) > 50 and len(np.unique(y_clean)) > 10:  # Regression case
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_clean, y_clean, test_size=0.3, random_state=42
                        )
                        
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        # Calculate normalized residuals
                        residuals = np.abs(y_test - predictions)
                        target_range = np.max(y_test) - np.min(y_test)
                        if target_range > 0:
                            normalized_residuals = residuals / target_range
                            target_noise = min(1.0, np.mean(normalized_residuals))
                        
                except Exception:
                    target_noise = 0.5  # Default if model fitting fails
            
            # Combine feature and target noise
            overall_noise = (feature_noise + target_noise) / 2
            return min(1.0, overall_noise)
            
        except Exception:
            return 0.5  # Default noise level
    
    def _calculate_complexity_confidence(self, n_samples: int, n_features: int) -> float:
        """Calculate confidence in complexity assessment."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence with more data
        if n_samples > 1000:
            confidence += 0.1
        if n_samples > 10000:
            confidence += 0.1
        
        # Adjust for feature count
        if 10 <= n_features <= 100:
            confidence += 0.1  # Good feature range for analysis
        
        return min(1.0, confidence)
    
    def detect_overfitting_risk(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        task_type: str = "classification"
    ) -> OverfittingRisk:
        """
        Detect overfitting risk and recommend mitigation strategies.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of ML task ('classification' or 'regression')
            
        Returns:
            Overfitting risk assessment
        """
        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
            X_array = X.values
        else:
            n_samples, n_features = X.shape
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Risk factors analysis
        risk_factors = []
        risk_scores = []
        
        # 1. Sample size vs feature count ratio
        sample_feature_ratio = n_samples / n_features
        if sample_feature_ratio < 10:
            risk_factors.append("Low sample-to-feature ratio")
            risk_scores.append(min(1.0, 1.0 - (sample_feature_ratio / 10)))
        
        # 2. Model complexity vs data size
        complexity_risk = self._assess_model_complexity_risk(n_samples, n_features)
        if complexity_risk > 0.3:
            risk_factors.append("High model complexity relative to data size")
            risk_scores.append(complexity_risk)
        
        # 3. Feature correlation and multicollinearity
        correlation_risk = self._assess_correlation_risk(X_array)
        if correlation_risk > 0.5:
            risk_factors.append("High feature correlation/multicollinearity")
            risk_scores.append(correlation_risk)
        
        # 4. Class imbalance (for classification)
        if task_type == "classification":
            imbalance_risk = self._assess_imbalance_overfitting_risk(y_array)
            if imbalance_risk > 0.4:
                risk_factors.append("Class imbalance may lead to overfitting")
                risk_scores.append(imbalance_risk)
        
        # 5. Noise level
        noise_risk = self._assess_noise_overfitting_risk(X_array, y_array)
        if noise_risk > 0.4:
            risk_factors.append("High noise level increases overfitting risk")
            risk_scores.append(noise_risk)
        
        # Calculate overall risk score
        if risk_scores:
            risk_score = np.mean(risk_scores)
        else:
            risk_score = 0.2  # Low default risk
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Generate regularization recommendations
        regularization_recommendations = self._generate_regularization_recommendations(
            risk_factors, risk_score, n_samples, n_features
        )
        
        # Recommend cross-validation strategy
        cv_strategy = self._recommend_cv_strategy(n_samples, task_type, risk_level)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            risk_factors, risk_score, n_samples, n_features
        )
        
        # Calculate confidence
        confidence = self._calculate_overfitting_confidence(n_samples, n_features, len(risk_factors))
        
        return OverfittingRisk(
            risk_level=risk_level,
            risk_score=risk_score,
            contributing_factors=risk_factors,
            regularization_recommendations=regularization_recommendations,
            cross_validation_strategy=cv_strategy,
            confidence=confidence,
            mitigation_strategies=mitigation_strategies
        )
    
    def _assess_model_complexity_risk(self, n_samples: int, n_features: int) -> float:
        """Assess overfitting risk due to model complexity."""
        # Rule of thumb: need at least 10 samples per feature for linear models
        # More complex models need even more data
        
        complexity_factors = {
            "linear": 10,
            "polynomial": 20,
            "tree": 15,
            "ensemble": 25,
            "neural_network": 50
        }
        
        # Use ensemble as default (conservative estimate)
        required_samples = n_features * complexity_factors["ensemble"]
        
        if n_samples >= required_samples:
            return 0.0
        else:
            return min(1.0, 1.0 - (n_samples / required_samples))
    
    def _assess_correlation_risk(self, X: np.ndarray) -> float:
        """Assess overfitting risk due to feature correlations."""
        try:
            # Handle missing values
            X_clean = X[~np.isnan(X).any(axis=1)]
            if len(X_clean) < 10:
                return 0.5  # Default risk if not enough clean data
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X_clean.T)
            
            # Count high correlations (excluding diagonal)
            high_corr_mask = (np.abs(correlation_matrix) > 0.8) & ~np.eye(correlation_matrix.shape[0], dtype=bool)
            high_corr_count = np.sum(high_corr_mask) / 2  # Divide by 2 for symmetric matrix
            
            total_pairs = (correlation_matrix.shape[0] * (correlation_matrix.shape[0] - 1)) / 2
            correlation_ratio = high_corr_count / total_pairs if total_pairs > 0 else 0
            
            return min(1.0, correlation_ratio * 3)  # Scale up the risk
            
        except Exception:
            return 0.5  # Default risk if calculation fails
    
    def _assess_imbalance_overfitting_risk(self, y: np.ndarray) -> float:
        """Assess overfitting risk due to class imbalance."""
        try:
            unique_values, counts = np.unique(y, return_counts=True)
            
            if len(unique_values) <= 1:
                return 0.0  # No imbalance with single class
            
            # Calculate imbalance ratio
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_ratio = min_count / max_count
            
            # Higher risk with more severe imbalance
            if imbalance_ratio < 0.1:  # Severe imbalance
                return 0.8
            elif imbalance_ratio < 0.3:  # Moderate imbalance
                return 0.5
            else:
                return 0.2  # Mild imbalance
                
        except Exception:
            return 0.3  # Default risk
    
    def _assess_noise_overfitting_risk(self, X: np.ndarray, y: np.ndarray) -> float:
        """Assess overfitting risk due to noise in the data."""
        try:
            # Use a simple model to estimate signal-to-noise ratio
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            
            # Clean the data
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 50:
                return 0.5  # Default risk for small datasets
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.3, random_state=42
            )
            
            # Fit simple model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate R² score
            r2 = r2_score(y_test, model.predict(X_test))
            
            # Low R² indicates high noise or complex relationships
            if r2 < 0.1:
                return 0.8  # High noise risk
            elif r2 < 0.3:
                return 0.5  # Moderate noise risk
            else:
                return 0.2  # Low noise risk
                
        except Exception:
            return 0.5  # Default risk
    
    def _generate_regularization_recommendations(
        self, 
        risk_factors: List[str], 
        risk_score: float,
        n_samples: int,
        n_features: int
    ) -> List[str]:
        """Generate regularization recommendations based on risk factors."""
        recommendations = []
        
        if risk_score > 0.5:
            recommendations.append("Apply L1 (Lasso) regularization for feature selection")
            recommendations.append("Apply L2 (Ridge) regularization to reduce model complexity")
        
        if "Low sample-to-feature ratio" in risk_factors:
            recommendations.append("Use elastic net regularization combining L1 and L2")
            recommendations.append("Consider dimensionality reduction (PCA, feature selection)")
        
        if "High feature correlation/multicollinearity" in risk_factors:
            recommendations.append("Apply Ridge regularization to handle multicollinearity")
            recommendations.append("Remove highly correlated features")
        
        if "Class imbalance may lead to overfitting" in risk_factors:
            recommendations.append("Use class-balanced regularization")
            recommendations.append("Apply SMOTE or other resampling techniques")
        
        if n_features > n_samples:
            recommendations.append("Strong regularization required due to high dimensionality")
            recommendations.append("Consider feature selection before modeling")
        
        if not recommendations:
            recommendations.append("Light regularization may be beneficial")
        
        return recommendations
    
    def _recommend_cv_strategy(self, n_samples: int, task_type: str, risk_level: str) -> str:
        """Recommend appropriate cross-validation strategy."""
        if n_samples < 100:
            return "Leave-one-out cross-validation due to small sample size"
        elif n_samples < 1000:
            if risk_level == "high":
                return "10-fold cross-validation with stratification"
            else:
                return "5-fold cross-validation"
        else:
            if risk_level == "high":
                return "10-fold cross-validation with multiple repetitions"
            elif task_type == "classification":
                return "Stratified 5-fold cross-validation"
            else:
                return "5-fold cross-validation"
    
    def _generate_mitigation_strategies(
        self,
        risk_factors: List[str],
        risk_score: float,
        n_samples: int,
        n_features: int
    ) -> Dict[str, str]:
        """Generate comprehensive mitigation strategies."""
        strategies = {}
        
        if risk_score > 0.7:
            strategies["data_collection"] = "Collect more training data to reduce overfitting risk"
        
        if "Low sample-to-feature ratio" in risk_factors:
            strategies["feature_engineering"] = "Apply feature selection or dimensionality reduction"
        
        if "High feature correlation/multicollinearity" in risk_factors:
            strategies["preprocessing"] = "Remove correlated features or apply PCA"
        
        if "Class imbalance may lead to overfitting" in risk_factors:
            strategies["sampling"] = "Apply balanced sampling or cost-sensitive learning"
        
        if risk_score > 0.5:
            strategies["model_selection"] = "Use simpler models or ensemble methods with regularization"
            strategies["validation"] = "Implement robust cross-validation and early stopping"
        
        strategies["monitoring"] = "Monitor training/validation curves for overfitting signs"
        
        return strategies
    
    def _calculate_overfitting_confidence(
        self, 
        n_samples: int, 
        n_features: int, 
        n_risk_factors: int
    ) -> float:
        """Calculate confidence in overfitting risk assessment."""
        confidence = 0.8  # Base confidence
        
        # Higher confidence with more data
        if n_samples > 1000:
            confidence += 0.1
        
        # Lower confidence with very high dimensionality
        if n_features > n_samples:
            confidence -= 0.2
        
        # Adjust based on number of risk factors identified
        if n_risk_factors > 0:
            confidence += 0.1  # More confident when risks are identified
        
        return max(0.3, min(1.0, confidence))
    
    def analyze_comprehensive(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        task_type: str = "classification"
    ) -> ComplexityAnalysis:
        """
        Perform comprehensive complexity analysis combining all methods.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            task_type: Type of ML task
            
        Returns:
            Comprehensive complexity analysis results
        """
        try:
            # Dataset complexity assessment
            dataset_complexity = self.assess_dataset_complexity(X, y)
            
            # Resource requirement estimation
            resource_estimate = self.estimate_computational_requirements(X, y, task_type)
            
            # Overfitting risk detection (if target is provided)
            if y is not None:
                overfitting_risk = self.detect_overfitting_risk(X, y, task_type)
            else:
                # Default overfitting risk for unsupervised tasks
                overfitting_risk = OverfittingRisk(
                    risk_level="low",
                    risk_score=0.2,
                    contributing_factors=["Unsupervised learning task"],
                    regularization_recommendations=["Consider regularization for clustering algorithms"],
                    cross_validation_strategy="K-fold validation for stability assessment",
                    confidence=0.7,
                    mitigation_strategies={"validation": "Use stability metrics for model selection"}
                )
            
            # Generate overall recommendations
            recommended_approach = self._generate_overall_recommendation(
                dataset_complexity, resource_estimate, overfitting_risk
            )
            
            # Performance expectations
            performance_expectations = self._generate_performance_expectations(
                dataset_complexity, overfitting_risk
            )
            
            # Optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                dataset_complexity, resource_estimate, overfitting_risk
            )
            
            # Overall confidence
            overall_confidence = np.mean([
                dataset_complexity.confidence,
                resource_estimate.confidence,
                overfitting_risk.confidence
            ])
            
            return ComplexityAnalysis(
                dataset_complexity=dataset_complexity,
                resource_estimate=resource_estimate,
                overfitting_risk=overfitting_risk,
                recommended_approach=recommended_approach,
                performance_expectations=performance_expectations,
                optimization_suggestions=optimization_suggestions,
                overall_confidence=overall_confidence
            )
            
        except Exception as e:
            # Return minimal analysis if comprehensive analysis fails
            warnings.warn(f"Comprehensive complexity analysis failed: {str(e)}")
            
            # Create minimal default objects
            default_complexity = DatasetComplexity(
                complexity_level="medium",
                complexity_score=0.5,
                dimensionality_complexity=0.5,
                sample_complexity=0.5,
                class_imbalance_complexity=0.5,
                feature_interaction_complexity=0.5,
                noise_level=0.5,
                confidence=0.5
            )
            
            default_resources = ResourceEstimate(
                cpu_suitability=0.8,
                gpu_suitability=0.3,
                memory_requirement_mb=1024.0,
                processing_time_seconds=60.0,
                recommended_hardware="cpu",
                confidence=0.5,
                rationale="Default resource estimation due to analysis failure"
            )
            
            default_overfitting = OverfittingRisk(
                risk_level="medium",
                risk_score=0.5,
                contributing_factors=["Analysis failed - using defaults"],
                regularization_recommendations=["Apply standard regularization"],
                cross_validation_strategy="5-fold cross-validation",
                confidence=0.5
            )
            
            return ComplexityAnalysis(
                dataset_complexity=default_complexity,
                resource_estimate=default_resources,
                overfitting_risk=default_overfitting,
                recommended_approach="Standard ML approach with regularization",
                performance_expectations={"accuracy": "moderate", "training_time": "reasonable"},
                optimization_suggestions=["Apply standard preprocessing and regularization"],
                overall_confidence=0.5
            )
    
    def _generate_overall_recommendation(
        self,
        dataset_complexity: DatasetComplexity,
        resource_estimate: ResourceEstimate,
        overfitting_risk: OverfittingRisk
    ) -> str:
        """Generate overall approach recommendation."""
        recommendations = []
        
        if dataset_complexity.complexity_level == "high":
            recommendations.append("Use ensemble methods or deep learning")
        elif dataset_complexity.complexity_level == "low":
            recommendations.append("Simple linear models may be sufficient")
        else:
            recommendations.append("Tree-based models or moderate complexity algorithms")
        
        if overfitting_risk.risk_level == "high":
            recommendations.append("with strong regularization")
        elif overfitting_risk.risk_level == "medium":
            recommendations.append("with moderate regularization")
        
        if resource_estimate.recommended_hardware == "gpu":
            recommendations.append("Consider GPU acceleration for training")
        elif resource_estimate.recommended_hardware == "distributed":
            recommendations.append("Consider distributed computing")
        
        return " ".join(recommendations)
    
    def _generate_performance_expectations(
        self,
        dataset_complexity: DatasetComplexity,
        overfitting_risk: OverfittingRisk
    ) -> Dict[str, str]:
        """Generate performance expectations."""
        expectations = {}
        
        # Accuracy expectations
        if dataset_complexity.complexity_level == "low" and overfitting_risk.risk_level == "low":
            expectations["accuracy"] = "high"
        elif dataset_complexity.complexity_level == "high" or overfitting_risk.risk_level == "high":
            expectations["accuracy"] = "moderate"
        else:
            expectations["accuracy"] = "good"
        
        # Training time expectations
        if dataset_complexity.complexity_score > 0.7:
            expectations["training_time"] = "long"
        elif dataset_complexity.complexity_score < 0.3:
            expectations["training_time"] = "short"
        else:
            expectations["training_time"] = "moderate"
        
        # Generalization expectations
        if overfitting_risk.risk_level == "high":
            expectations["generalization"] = "challenging"
        elif overfitting_risk.risk_level == "low":
            expectations["generalization"] = "good"
        else:
            expectations["generalization"] = "moderate"
        
        return expectations
    
    def _generate_optimization_suggestions(
        self,
        dataset_complexity: DatasetComplexity,
        resource_estimate: ResourceEstimate,
        overfitting_risk: OverfittingRisk
    ) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Data-based optimizations
        if dataset_complexity.dimensionality_complexity > 0.7:
            suggestions.append("Apply dimensionality reduction techniques")
        
        if dataset_complexity.noise_level > 0.6:
            suggestions.append("Implement noise reduction preprocessing")
        
        if dataset_complexity.class_imbalance_complexity > 0.6:
            suggestions.append("Address class imbalance with sampling techniques")
        
        # Resource-based optimizations
        if resource_estimate.memory_requirement_mb > 8192:  # > 8GB
            suggestions.append("Implement batch processing for memory efficiency")
        
        if resource_estimate.processing_time_seconds > 3600:  # > 1 hour
            suggestions.append("Consider parallel processing or model simplification")
        
        # Overfitting-based optimizations
        if overfitting_risk.risk_level == "high":
            suggestions.extend(overfitting_risk.regularization_recommendations[:2])  # Top 2 recommendations
        
        # General optimizations
        suggestions.append("Use cross-validation for robust model evaluation")
        suggestions.append("Monitor learning curves during training")
        
        return suggestions