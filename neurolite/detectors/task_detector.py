"""
Task detector for ML task identification and analysis.

This module provides functionality to detect supervised and unsupervised learning tasks,
assess dataset characteristics, and provide task-specific recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

from ..core.data_models import (
    TaskIdentification, SupervisedTaskAnalysis, UnsupervisedTaskAnalysis
)
from ..core.exceptions import NeuroLiteException


class TaskDetector:
    """Detector for identifying ML tasks and analyzing dataset characteristics."""
    
    def __init__(self):
        """Initialize the TaskDetector."""
        self.confidence_threshold = 0.7
        self.min_samples_for_analysis = 20
        self.max_clusters_to_test = 10
        self.dimensionality_threshold = 0.95  # Explained variance threshold for PCA
        
    def detect_supervised_task(self, X: Union[pd.DataFrame, np.ndarray], 
                             y: Union[pd.Series, np.ndarray]) -> SupervisedTaskAnalysis:
        """
        Detect supervised learning task type and characteristics.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            SupervisedTaskAnalysis: Analysis results for supervised task
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
            
        if len(X_array) < self.min_samples_for_analysis:
            raise NeuroLiteException(
                f"Insufficient samples for analysis. Need at least {self.min_samples_for_analysis}, got {len(X_array)}"
            )
            
        if len(X_array) != len(y_array):
            raise NeuroLiteException("Feature matrix and target variable must have same length")
        
        # Determine if classification or regression
        task_type, task_subtype, confidence = self._classify_supervised_task(y_array)
        
        # Analyze target characteristics
        target_characteristics = self._analyze_target_characteristics(y_array, task_type)
        
        # Assess dataset balance (for classification)
        dataset_balance = {}
        if task_type == 'classification':
            dataset_balance = self._assess_dataset_balance(y_array)
        
        # Analyze complexity indicators
        complexity_indicators = self._analyze_complexity_indicators(X_array, y_array, task_type)
        
        return SupervisedTaskAnalysis(
            task_type=task_type,
            task_subtype=task_subtype,
            confidence=confidence,
            target_characteristics=target_characteristics,
            dataset_balance=dataset_balance,
            complexity_indicators=complexity_indicators
        )
    
    def _classify_supervised_task(self, y: np.ndarray) -> Tuple[str, str, float]:
        """
        Classify whether the task is classification or regression.
        
        Args:
            y: Target variable array
            
        Returns:
            Tuple of (task_type, task_subtype, confidence)
        """
        # Check if target is numeric
        try:
            y_numeric = pd.to_numeric(y, errors='coerce')
            is_numeric = not np.isnan(y_numeric).all()
        except:
            is_numeric = False
        
        # Count unique values
        unique_values = np.unique(y[~pd.isna(y)])
        num_unique = len(unique_values)
        total_samples = len(y)
        
        # Classification indicators
        is_categorical = not is_numeric or num_unique <= max(10, total_samples * 0.05)
        
        if is_categorical:
            # Classification task
            if num_unique == 2:
                return 'classification', 'binary', 0.95
            elif num_unique <= 10:
                return 'classification', 'multiclass', 0.90
            else:
                # Many unique values but still categorical
                return 'classification', 'multiclass', 0.75
        else:
            # Regression task - determine if linear or non-linear
            if self._is_linear_relationship(y_numeric):
                return 'regression', 'linear', 0.85
            else:
                return 'regression', 'non_linear', 0.80
    
    def _is_linear_relationship(self, y: np.ndarray) -> bool:
        """
        Determine if the target variable suggests a linear relationship.
        
        Args:
            y: Numeric target variable
            
        Returns:
            bool: True if linear relationship is likely
        """
        # Simple heuristic: check if values are roughly evenly distributed
        # More sophisticated analysis would require X features
        try:
            # Remove any NaN values
            y_clean = y[~np.isnan(y)]
            
            # If all values are the same, default to linear
            if len(np.unique(y_clean)) <= 1:
                return True
                
            # Check for uniform distribution vs skewed
            _, p_value = stats.normaltest(y_clean)
            return p_value > 0.05  # Normal distribution suggests linear potential
        except Exception:
            return True  # Default to linear if test fails
    
    def _analyze_target_characteristics(self, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """
        Analyze characteristics of the target variable.
        
        Args:
            y: Target variable array
            task_type: Type of task (classification or regression)
            
        Returns:
            Dict containing target characteristics
        """
        characteristics = {
            'total_samples': len(y),
            'missing_values': np.sum(pd.isna(y)),
            'unique_values': len(np.unique(y[~pd.isna(y)]))
        }
        
        if task_type == 'classification':
            # Classification-specific characteristics
            unique_values, counts = np.unique(y[~pd.isna(y)], return_counts=True)
            characteristics.update({
                'class_labels': unique_values.tolist(),
                'class_counts': counts.tolist(),
                'most_frequent_class': unique_values[np.argmax(counts)],
                'least_frequent_class': unique_values[np.argmin(counts)]
            })
        else:
            # Regression-specific characteristics
            y_numeric = pd.to_numeric(y, errors='coerce')
            y_clean = y_numeric[~np.isnan(y_numeric)]
            
            if len(y_clean) > 0:
                characteristics.update({
                    'min_value': float(np.min(y_clean)),
                    'max_value': float(np.max(y_clean)),
                    'mean_value': float(np.mean(y_clean)),
                    'std_value': float(np.std(y_clean)),
                    'median_value': float(np.median(y_clean)),
                    'skewness': float(stats.skew(y_clean)),
                    'kurtosis': float(stats.kurtosis(y_clean))
                })
        
        return characteristics
    
    def _assess_dataset_balance(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Assess balance of classification dataset.
        
        Args:
            y: Target variable array for classification
            
        Returns:
            Dict containing balance assessment
        """
        unique_values, counts = np.unique(y[~pd.isna(y)], return_counts=True)
        total_samples = len(y[~pd.isna(y)])
        
        # Calculate balance metrics
        class_proportions = counts / total_samples
        max_proportion = np.max(class_proportions)
        min_proportion = np.min(class_proportions)
        
        # Balance ratio (closer to 1.0 means more balanced)
        balance_ratio = min_proportion / max_proportion
        
        # Imbalance severity
        if balance_ratio >= 0.8:
            balance_status = 'balanced'
        elif balance_ratio >= 0.5:
            balance_status = 'slightly_imbalanced'
        elif balance_ratio >= 0.2:
            balance_status = 'moderately_imbalanced'
        else:
            balance_status = 'severely_imbalanced'
        
        return {
            'balance_ratio': float(balance_ratio),
            'balance_status': balance_status,
            'class_proportions': {str(cls): float(prop) for cls, prop in zip(unique_values, class_proportions)},
            'majority_class': str(unique_values[np.argmax(counts)]),
            'minority_class': str(unique_values[np.argmin(counts)]),
            'majority_proportion': float(max_proportion),
            'minority_proportion': float(min_proportion)
        }
    
    def _analyze_complexity_indicators(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """
        Analyze complexity indicators for the supervised learning task.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of task (classification or regression)
            
        Returns:
            Dict containing complexity indicators
        """
        n_samples, n_features = X.shape
        
        complexity = {
            'n_samples': n_samples,
            'n_features': n_features,
            'samples_to_features_ratio': n_samples / n_features if n_features > 0 else 0,
            'dimensionality': 'high' if n_features > n_samples else 'normal'
        }
        
        # Feature correlation analysis
        try:
            if n_features > 1:
                # Calculate correlation matrix for numerical features
                X_numeric = self._convert_to_numeric(X)
                if X_numeric.shape[1] > 1:
                    corr_matrix = np.corrcoef(X_numeric.T)
                    # Find high correlations (excluding diagonal)
                    high_corr_mask = (np.abs(corr_matrix) > 0.8) & (corr_matrix != 1.0)
                    high_corr_pairs = np.sum(high_corr_mask) // 2  # Divide by 2 for symmetric matrix
                    
                    complexity.update({
                        'high_correlation_pairs': int(high_corr_pairs),
                        'multicollinearity_risk': 'high' if high_corr_pairs > n_features * 0.1 else 'low'
                    })
        except Exception as e:
            complexity.update({
                'correlation_analysis_error': str(e),
                'multicollinearity_risk': 'unknown'
            })
        
        # Task-specific complexity
        if task_type == 'classification':
            unique_classes = len(np.unique(y[~pd.isna(y)]))
            complexity.update({
                'n_classes': unique_classes,
                'classification_complexity': 'high' if unique_classes > 10 else 'normal'
            })
        else:
            # Regression complexity based on target variance
            y_numeric = pd.to_numeric(y, errors='coerce')
            y_clean = y_numeric[~np.isnan(y_numeric)]
            if len(y_clean) > 1:
                target_variance = np.var(y_clean)
                target_range = np.max(y_clean) - np.min(y_clean)
                complexity.update({
                    'target_variance': float(target_variance),
                    'target_range': float(target_range),
                    'regression_complexity': 'high' if target_variance > np.mean(y_clean) else 'normal'
                })
        
        return complexity
    
    def _convert_to_numeric(self, X: np.ndarray) -> np.ndarray:
        """
        Convert features to numeric format for correlation analysis.
        
        Args:
            X: Feature matrix
            
        Returns:
            Numeric feature matrix
        """
        if X.dtype.kind in 'biufc':  # Already numeric
            return X
        
        # Try to convert to numeric, replacing non-numeric with NaN
        X_numeric = []
        for col in range(X.shape[1]):
            try:
                col_numeric = pd.to_numeric(X[:, col], errors='coerce')
                X_numeric.append(col_numeric)
            except:
                # If conversion fails, create array of NaNs
                X_numeric.append(np.full(X.shape[0], np.nan))
        
        return np.column_stack(X_numeric)
    
    def detect_unsupervised_task(self, X: Union[pd.DataFrame, np.ndarray]) -> UnsupervisedTaskAnalysis:
        """
        Detect unsupervised learning task potential and characteristics.
        
        Args:
            X: Feature matrix
            
        Returns:
            UnsupervisedTaskAnalysis: Analysis results for unsupervised task
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if len(X_array) < self.min_samples_for_analysis:
            raise NeuroLiteException(
                f"Insufficient samples for analysis. Need at least {self.min_samples_for_analysis}, got {len(X_array)}"
            )
        
        # Convert to numeric for analysis
        X_numeric = self._convert_to_numeric(X_array)
        
        # Remove columns that are all NaN
        valid_cols = ~np.isnan(X_numeric).all(axis=0)
        X_clean = X_numeric[:, valid_cols]
        
        if X_clean.shape[1] == 0:
            raise NeuroLiteException("No valid numeric features found for unsupervised analysis")
        
        # Remove rows with any NaN values for clustering analysis
        valid_rows = ~np.isnan(X_clean).any(axis=1)
        X_for_clustering = X_clean[valid_rows]
        
        if len(X_for_clustering) < self.min_samples_for_analysis:
            raise NeuroLiteException("Insufficient clean samples for clustering analysis")
        
        # Assess clustering potential
        clustering_potential, optimal_clusters, clustering_characteristics = self._assess_clustering_potential(X_for_clustering)
        
        # Assess dimensionality reduction need
        dimensionality_reduction_needed, dimensionality_info = self._assess_dimensionality_reduction_need(X_for_clustering)
        
        # Calculate overall confidence
        confidence = self._calculate_unsupervised_confidence(
            clustering_potential, dimensionality_reduction_needed, X_for_clustering.shape
        )
        
        return UnsupervisedTaskAnalysis(
            clustering_potential=clustering_potential,
            optimal_clusters=optimal_clusters,
            dimensionality_reduction_needed=dimensionality_reduction_needed,
            confidence=confidence,
            clustering_characteristics=clustering_characteristics,
            dimensionality_info=dimensionality_info
        )
    
    def _assess_clustering_potential(self, X: np.ndarray) -> Tuple[float, Optional[int], Dict[str, Any]]:
        """
        Assess the potential for clustering in the dataset.
        
        Args:
            X: Clean numeric feature matrix
            
        Returns:
            Tuple of (clustering_potential, optimal_clusters, characteristics)
        """
        n_samples, n_features = X.shape
        
        # Scale the data for clustering analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'data_scaled': True
        }
        
        # Test different numbers of clusters
        max_clusters = min(self.max_clusters_to_test, n_samples // 2)
        silhouette_scores = []
        inertias = []
        
        best_score = -1
        optimal_clusters = None
        
        for k in range(2, max_clusters + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Calculate silhouette score
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                    inertias.append(kmeans.inertia_)
                    
                    if sil_score > best_score:
                        best_score = sil_score
                        optimal_clusters = k
                        
            except Exception as e:
                characteristics[f'clustering_error_k{k}'] = str(e)
                continue
        
        # Calculate clustering potential based on best silhouette score
        if silhouette_scores:
            clustering_potential = max(0.0, min(1.0, (best_score + 1) / 2))  # Normalize from [-1,1] to [0,1]
            characteristics.update({
                'silhouette_scores': silhouette_scores,
                'best_silhouette_score': best_score,
                'inertias': inertias,
                'clusters_tested': list(range(2, len(silhouette_scores) + 2))
            })
        else:
            clustering_potential = 0.0
            characteristics['clustering_analysis_failed'] = True
        
        # Additional clustering characteristics
        try:
            # Calculate pairwise distances to assess data spread
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(X_scaled[:min(1000, n_samples)])  # Sample for large datasets
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            characteristics.update({
                'average_pairwise_distance': float(avg_distance),
                'distance_std': float(std_distance),
                'distance_coefficient_variation': float(std_distance / avg_distance) if avg_distance > 0 else 0
            })
        except Exception as e:
            characteristics['distance_analysis_error'] = str(e)
        
        return clustering_potential, optimal_clusters, characteristics
    
    def _assess_dimensionality_reduction_need(self, X: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Assess whether dimensionality reduction is needed.
        
        Args:
            X: Clean numeric feature matrix
            
        Returns:
            Tuple of (dimensionality_reduction_needed, dimensionality_info)
        """
        n_samples, n_features = X.shape
        
        dimensionality_info = {
            'n_samples': n_samples,
            'n_features': n_features,
            'samples_to_features_ratio': n_samples / n_features if n_features > 0 else 0
        }
        
        # Simple heuristic: if features > samples or features > 50, likely need reduction
        high_dimensionality = n_features > n_samples or n_features > 50
        
        # Try PCA analysis if feasible
        dimensionality_reduction_needed = high_dimensionality
        
        try:
            if n_features > 1 and n_samples > n_features:
                # Scale data for PCA
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform PCA
                pca = PCA()
                pca.fit(X_scaled)
                
                # Calculate cumulative explained variance
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                
                # Find number of components needed for threshold variance
                n_components_needed = np.argmax(cumulative_variance >= self.dimensionality_threshold) + 1
                
                dimensionality_info.update({
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_explained_variance': cumulative_variance.tolist(),
                    'components_for_95_percent': int(n_components_needed),
                    'dimensionality_reduction_benefit': float(1 - n_components_needed / n_features)
                })
                
                # Update recommendation based on PCA results
                if n_components_needed < n_features * 0.8:
                    dimensionality_reduction_needed = True
                elif n_components_needed >= n_features * 0.95:
                    dimensionality_reduction_needed = False
                    
        except Exception as e:
            dimensionality_info['pca_analysis_error'] = str(e)
            # Fall back to simple heuristic
            dimensionality_reduction_needed = high_dimensionality
        
        # Additional dimensionality characteristics
        try:
            # Check for highly correlated features
            corr_matrix = np.corrcoef(X.T)
            high_corr_pairs = np.sum((np.abs(corr_matrix) > 0.9) & (corr_matrix != 1.0)) // 2
            
            dimensionality_info.update({
                'high_correlation_pairs': int(high_corr_pairs),
                'correlation_based_reduction_potential': high_corr_pairs > 0
            })
            
            if high_corr_pairs > n_features * 0.1:
                dimensionality_reduction_needed = True
                
        except Exception as e:
            dimensionality_info['correlation_analysis_error'] = str(e)
        
        return dimensionality_reduction_needed, dimensionality_info
    
    def _calculate_unsupervised_confidence(self, clustering_potential: float, 
                                         dimensionality_reduction_needed: bool, 
                                         data_shape: Tuple[int, int]) -> float:
        """
        Calculate overall confidence for unsupervised learning recommendations.
        
        Args:
            clustering_potential: Assessed clustering potential (0-1)
            dimensionality_reduction_needed: Whether dimensionality reduction is needed
            data_shape: Shape of the data (n_samples, n_features)
            
        Returns:
            Overall confidence score
        """
        n_samples, n_features = data_shape
        
        # Base confidence from clustering potential
        confidence = clustering_potential
        
        # Adjust based on data characteristics
        if n_samples >= 100:  # Good sample size
            confidence *= 1.1
        elif n_samples < 50:  # Small sample size
            confidence *= 0.8
        
        # Adjust based on dimensionality
        if dimensionality_reduction_needed and n_features > 10:
            confidence *= 1.05  # Dimensionality reduction can help
        elif n_features < 3:
            confidence *= 0.9  # Very low dimensional data
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))