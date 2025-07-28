"""
Preprocessing recommender for data preparation strategies.

This module provides functionality to recommend appropriate preprocessing steps
including normalization, standardization, encoding, and feature scaling based on
data characteristics and ML task requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings

from ..core.data_models import (
    ColumnType, NumericalAnalysis, CategoricalAnalysis, TemporalAnalysis,
    ScalingRecommendation, EncodingRecommendation, FeatureEngineeringRecommendation,
    PreprocessingPipeline, TaskIdentification, QualityMetrics
)
from ..core.exceptions import NeuroLiteException, InsufficientDataError


class PreprocessingRecommender:
    """Recommender for data preprocessing strategies and transformations."""
    
    def __init__(self):
        """Initialize the PreprocessingRecommender."""
        self.high_cardinality_threshold = 50
        self.low_cardinality_threshold = 10
        self.outlier_threshold = 0.05  # 5% outliers threshold
        self.skewness_threshold = 1.0
        self.correlation_threshold = 0.8
        self.min_samples_for_analysis = 30
        
    def recommend_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        task_info: Optional[TaskIdentification] = None,
        quality_metrics: Optional[QualityMetrics] = None,
        target_column: Optional[str] = None
    ) -> PreprocessingPipeline:
        """
        Recommend a complete preprocessing pipeline based on data characteristics.
        
        Args:
            df: Input DataFrame
            column_types: Dictionary mapping column names to their types
            task_info: Information about the ML task
            quality_metrics: Data quality assessment results
            target_column: Name of target column for supervised tasks
            
        Returns:
            PreprocessingPipeline with all recommendations
        """
        if df.empty:
            raise NeuroLiteException("Cannot recommend preprocessing for empty DataFrame")
            
        if len(df) < self.min_samples_for_analysis:
            raise InsufficientDataError(
                len(df), self.min_samples_for_analysis, "samples"
            )
        
        # Get individual recommendations
        scaling_recs = self._recommend_scaling(df, column_types, task_info)
        encoding_recs = self._recommend_encoding(df, column_types, task_info, target_column)
        feature_eng_recs = self._recommend_feature_engineering(
            df, column_types, task_info, quality_metrics, target_column
        )
        
        # Determine pipeline order
        pipeline_order = self._determine_pipeline_order(scaling_recs, encoding_recs, feature_eng_recs)
        
        # Calculate overall confidence
        all_recs = scaling_recs + encoding_recs + feature_eng_recs
        overall_confidence = np.mean([rec.confidence for rec in all_recs]) if all_recs else 0.0
        
        # Estimate processing time (simplified heuristic)
        estimated_time = self._estimate_processing_time(df, len(all_recs))
        
        return PreprocessingPipeline(
            scaling_recommendations=scaling_recs,
            encoding_recommendations=encoding_recs,
            feature_engineering_recommendations=feature_eng_recs,
            pipeline_order=pipeline_order,
            overall_confidence=overall_confidence,
            estimated_processing_time=estimated_time
        )
    
    def _recommend_scaling(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        task_info: Optional[TaskIdentification] = None
    ) -> List[ScalingRecommendation]:
        """Recommend scaling strategies for numerical features."""
        recommendations = []
        
        numerical_columns = [
            col for col, col_type in column_types.items()
            if col_type.primary_type == 'numerical' and col in df.columns
        ]
        
        if not numerical_columns:
            return recommendations
            
        for col in numerical_columns:
            series = df[col].dropna()
            if len(series) < 10:  # Skip columns with too few values
                continue
                
            scaling_rec = self._analyze_scaling_need(series, col, task_info)
            recommendations.append(scaling_rec)  # Always add recommendation, even if 'none'
        
        return recommendations
    
    def _analyze_scaling_need(
        self,
        series: pd.Series,
        column_name: str,
        task_info: Optional[TaskIdentification] = None
    ) -> ScalingRecommendation:
        """Analyze scaling requirements for a numerical column."""
        # Calculate basic statistics
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        
        # Detect outliers using IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        outliers = series[(series < outlier_bounds[0]) | (series > outlier_bounds[1])]
        outlier_ratio = len(outliers) / len(series)
        
        # Calculate skewness
        skewness = stats.skew(series)
        
        # Decision logic for scaling type
        confidence = 0.7  # Base confidence
        
        # If many outliers, recommend robust scaling
        if outlier_ratio > self.outlier_threshold:
            scaling_type = 'robust_scaling'
            rationale = f"High outlier ratio ({outlier_ratio:.2%}) detected. Robust scaling handles outliers better."
            confidence = 0.9
            
        # If data is highly skewed, recommend robust scaling
        elif abs(skewness) > self.skewness_threshold:
            scaling_type = 'robust_scaling'
            rationale = f"High skewness ({skewness:.2f}) detected. Robust scaling is more suitable."
            confidence = 0.8
            
        # If range is very large, recommend normalization
        elif range_val > 1000 or (max_val > 100 and min_val < 0.01):
            scaling_type = 'normalization'
            rationale = f"Large value range ({min_val:.2f} to {max_val:.2f}). Normalization will help."
            confidence = 0.8
            
        # For neural networks or distance-based algorithms, prefer standardization
        elif task_info and ('neural' in task_info.task_type.lower() or 
                           'svm' in task_info.task_type.lower() or
                           'clustering' in task_info.task_type.lower()):
            scaling_type = 'standardization'
            rationale = f"Task type '{task_info.task_type}' benefits from standardized features."
            confidence = 0.9
            
        # If data is already well-scaled (mean ~0, std ~1), no scaling needed
        elif abs(mean_val) < 0.1 and 0.8 < std_val < 1.2:
            scaling_type = 'none'
            rationale = "Data is already well-scaled (mean ≈ 0, std ≈ 1)."
            confidence = 0.9
            
        # Default to standardization for most cases
        else:
            scaling_type = 'standardization'
            rationale = "Standardization recommended for general ML algorithms."
            confidence = 0.7
        
        return ScalingRecommendation(
            scaling_type=scaling_type,
            rationale=rationale,
            confidence=confidence,
            affected_columns=[column_name],
            parameters={
                'mean': mean_val,
                'std': std_val,
                'range': range_val,
                'outlier_ratio': outlier_ratio,
                'skewness': skewness
            }
        )
    
    def _recommend_encoding(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        task_info: Optional[TaskIdentification] = None,
        target_column: Optional[str] = None
    ) -> List[EncodingRecommendation]:
        """Recommend encoding strategies for categorical features."""
        recommendations = []
        
        categorical_columns = [
            col for col, col_type in column_types.items()
            if col_type.primary_type == 'categorical' and col in df.columns
        ]
        
        if not categorical_columns:
            return recommendations
            
        for col in categorical_columns:
            if col == target_column:  # Skip target column
                continue
                
            series = df[col].dropna()
            if len(series) < 5:  # Skip columns with too few values
                continue
                
            encoding_rec = self._analyze_encoding_need(series, col, task_info, df, target_column)
            if encoding_rec.encoding_type != 'none':
                recommendations.append(encoding_rec)
        
        return recommendations
    
    def _analyze_encoding_need(
        self,
        series: pd.Series,
        column_name: str,
        task_info: Optional[TaskIdentification] = None,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None
    ) -> EncodingRecommendation:
        """Analyze encoding requirements for a categorical column."""
        unique_values = series.unique()
        cardinality = len(unique_values)
        
        # Check if ordinal (simple heuristic: numeric strings or clear ordering)
        is_ordinal = self._detect_ordinal_nature(unique_values)
        
        confidence = 0.7  # Base confidence
        
        # Decision logic for encoding type
        if cardinality == 2:
            # Binary categorical - use label encoding
            encoding_type = 'label_encoding'
            rationale = f"Binary categorical variable with {cardinality} categories. Label encoding is efficient."
            confidence = 0.9
            
        elif cardinality > self.high_cardinality_threshold:
            # High cardinality - consider target encoding if supervised task
            if (task_info and task_info.task_type in ['classification', 'regression'] and 
                target_column and df is not None and target_column in df.columns):
                encoding_type = 'target_encoding'
                rationale = f"High cardinality ({cardinality} categories). Target encoding reduces dimensionality."
                confidence = 0.8
            else:
                # Use binary encoding for high cardinality
                encoding_type = 'binary_encoding'
                rationale = f"High cardinality ({cardinality} categories). Binary encoding is memory efficient."
                confidence = 0.7
                
        elif is_ordinal:
            # Ordinal data - use label encoding
            encoding_type = 'label_encoding'
            rationale = f"Ordinal categorical variable detected. Label encoding preserves order."
            confidence = 0.8
            
        elif cardinality <= self.low_cardinality_threshold:
            # Low cardinality - use one-hot encoding
            encoding_type = 'one_hot'
            rationale = f"Low cardinality ({cardinality} categories). One-hot encoding is suitable."
            confidence = 0.9
            
        else:
            # Medium cardinality - default to one-hot if not too many categories
            if cardinality <= 20:
                encoding_type = 'one_hot'
                rationale = f"Medium cardinality ({cardinality} categories). One-hot encoding is manageable."
                confidence = 0.7
            else:
                encoding_type = 'binary_encoding'
                rationale = f"Medium-high cardinality ({cardinality} categories). Binary encoding balances efficiency."
                confidence = 0.6
        
        return EncodingRecommendation(
            encoding_type=encoding_type,
            rationale=rationale,
            confidence=confidence,
            affected_columns=[column_name],
            parameters={
                'cardinality': cardinality,
                'unique_values': unique_values.tolist()[:10],  # Store first 10 for reference
                'is_ordinal': is_ordinal
            }
        )
    
    def _detect_ordinal_nature(self, unique_values: np.ndarray) -> bool:
        """Detect if categorical values have ordinal nature."""
        # Convert to strings for analysis
        str_values = [str(val).lower() for val in unique_values if pd.notna(val)]
        
        # Check for common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
        ]
        
        # Check if values match any ordinal pattern
        for pattern in ordinal_patterns:
            if all(val in pattern for val in str_values):
                return True
        
        # Check if values are numeric strings
        try:
            numeric_values = [float(val) for val in str_values]
            return True  # If all can be converted to numbers, likely ordinal
        except (ValueError, TypeError):
            pass
        
        # Check for size/rating patterns
        size_patterns = ['xs', 's', 'm', 'l', 'xl', 'xxl']
        rating_patterns = ['1', '2', '3', '4', '5']
        
        if all(val in size_patterns for val in str_values):
            return True
        if all(val in rating_patterns for val in str_values):
            return True
            
        return False
    
    def _recommend_feature_engineering(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        task_info: Optional[TaskIdentification] = None,
        quality_metrics: Optional[QualityMetrics] = None,
        target_column: Optional[str] = None
    ) -> List[FeatureEngineeringRecommendation]:
        """Recommend feature engineering strategies."""
        recommendations = []
        
        # Polynomial features for regression tasks
        if task_info and 'regression' in task_info.task_type:
            poly_rec = self._recommend_polynomial_features(df, column_types)
            if poly_rec:
                recommendations.append(poly_rec)
        
        # Feature selection based on correlation
        correlation_rec = self._recommend_feature_selection(df, column_types, target_column)
        if correlation_rec:
            recommendations.append(correlation_rec)
        
        # Dimensionality reduction for high-dimensional data
        if len(df.columns) > 50:
            dim_red_rec = self._recommend_dimensionality_reduction(df, column_types, task_info)
            if dim_red_rec:
                recommendations.append(dim_red_rec)
        
        return recommendations
    
    def _recommend_polynomial_features(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType]
    ) -> Optional[FeatureEngineeringRecommendation]:
        """Recommend polynomial feature generation for regression tasks."""
        numerical_columns = [
            col for col, col_type in column_types.items()
            if col_type.primary_type == 'numerical' and col in df.columns
        ]
        
        if len(numerical_columns) < 2:
            return None
            
        # Simple heuristic: if we have few numerical features, polynomial might help
        if len(numerical_columns) <= 5:
            return FeatureEngineeringRecommendation(
                technique='polynomial_features',
                rationale=f"Few numerical features ({len(numerical_columns)}). Polynomial features may capture non-linear relationships.",
                confidence=0.6,
                affected_columns=numerical_columns,
                parameters={'degree': 2, 'interaction_only': False},
                expected_benefit="May improve model performance by capturing feature interactions"
            )
        
        return None
    
    def _recommend_feature_selection(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        target_column: Optional[str] = None
    ) -> Optional[FeatureEngineeringRecommendation]:
        """Recommend feature selection based on correlation analysis."""
        if not target_column or target_column not in df.columns:
            return None
            
        numerical_columns = [
            col for col, col_type in column_types.items()
            if col_type.primary_type == 'numerical' and col in df.columns and col != target_column
        ]
        
        if len(numerical_columns) < 3:
            return None
            
        # Calculate correlation matrix
        try:
            corr_matrix = df[numerical_columns].corr().abs()
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                return FeatureEngineeringRecommendation(
                    technique='feature_selection',
                    rationale=f"Found {len(high_corr_pairs)} highly correlated feature pairs. Feature selection may reduce multicollinearity.",
                    confidence=0.8,
                    affected_columns=numerical_columns,
                    parameters={'correlation_threshold': self.correlation_threshold, 'method': 'correlation'},
                    expected_benefit="Reduces multicollinearity and may improve model interpretability"
                )
        except Exception:
            # If correlation calculation fails, skip this recommendation
            pass
            
        return None
    
    def _recommend_dimensionality_reduction(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, ColumnType],
        task_info: Optional[TaskIdentification] = None
    ) -> Optional[FeatureEngineeringRecommendation]:
        """Recommend dimensionality reduction for high-dimensional data."""
        numerical_columns = [
            col for col, col_type in column_types.items()
            if col_type.primary_type == 'numerical' and col in df.columns
        ]
        
        if len(numerical_columns) < 50:
            return None
            
        # Recommend PCA for high-dimensional numerical data
        technique = 'pca'
        confidence = 0.7
        
        # Adjust recommendation based on task type
        if task_info:
            if 'classification' in task_info.task_type:
                technique = 'pca'
                confidence = 0.8
            elif 'clustering' in task_info.task_type:
                technique = 'pca'
                confidence = 0.9
        
        return FeatureEngineeringRecommendation(
            technique=technique,
            rationale=f"High-dimensional data ({len(numerical_columns)} numerical features). Dimensionality reduction may improve performance and reduce overfitting.",
            confidence=confidence,
            affected_columns=numerical_columns,
            parameters={'n_components': min(50, len(numerical_columns) // 2)},
            expected_benefit="Reduces computational complexity and may prevent overfitting"
        )
    
    def _determine_pipeline_order(
        self,
        scaling_recs: List[ScalingRecommendation],
        encoding_recs: List[EncodingRecommendation],
        feature_eng_recs: List[FeatureEngineeringRecommendation]
    ) -> List[str]:
        """Determine the optimal order for preprocessing steps."""
        pipeline_order = []
        
        # 1. Handle missing values first (if implemented)
        # pipeline_order.append('handle_missing')
        
        # 2. Encoding should come before scaling
        if encoding_recs:
            pipeline_order.append('encoding')
        
        # 3. Feature engineering (like polynomial features) before scaling
        feature_creation_techniques = ['polynomial_features']
        if any(rec.technique in feature_creation_techniques for rec in feature_eng_recs):
            pipeline_order.append('feature_creation')
        
        # 4. Scaling after encoding and feature creation
        if scaling_recs:
            pipeline_order.append('scaling')
        
        # 5. Feature selection after scaling
        selection_techniques = ['feature_selection', 'pca']
        if any(rec.technique in selection_techniques for rec in feature_eng_recs):
            pipeline_order.append('feature_selection')
        
        return pipeline_order
    
    def _estimate_processing_time(self, df: pd.DataFrame, num_operations: int) -> float:
        """Estimate preprocessing time based on data size and operations."""
        # Simple heuristic: base time + time per row + time per operation
        base_time = 0.1  # seconds
        time_per_row = 0.00001  # seconds per row
        time_per_operation = 0.05  # seconds per operation
        
        estimated_time = base_time + (len(df) * time_per_row) + (num_operations * time_per_operation)
        return estimated_time