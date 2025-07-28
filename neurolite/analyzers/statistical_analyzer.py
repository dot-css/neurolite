"""
Statistical analyzer for comprehensive data analysis.

This module provides statistical analysis capabilities including distribution
fitting, correlation analysis, and relationship detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import normaltest, jarque_bera, anderson, kstest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings

from ..core.data_models import StatisticalProperties
from ..core.exceptions import NeuroLiteException


@dataclass
class DistributionFit:
    """Represents a fitted distribution with parameters and goodness-of-fit metrics."""
    distribution_name: str
    parameters: Dict[str, float]
    goodness_of_fit: float
    p_value: float
    confidence_interval: Tuple[float, float]
    aic: float
    bic: float


@dataclass
class DistributionAnalysis:
    """Comprehensive distribution analysis results."""
    best_fit: DistributionFit
    alternative_fits: List[DistributionFit]
    is_normal: bool
    is_multimodal: bool
    skewness: float
    kurtosis: float
    normality_tests: Dict[str, Dict[str, float]]


@dataclass
class CorrelationMatrix:
    """Correlation analysis results."""
    pearson_correlation: np.ndarray
    spearman_correlation: np.ndarray
    kendall_correlation: np.ndarray
    mutual_information: np.ndarray
    column_names: List[str]


@dataclass
class RelationshipAnalysis:
    """Non-linear relationship detection results."""
    linear_relationships: Dict[str, float]
    non_linear_relationships: Dict[str, float]
    multicollinearity_vif: Dict[str, float]
    feature_dependencies: Dict[str, List[str]]


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for data distributions and relationships.
    
    This class provides methods for distribution fitting, correlation analysis,
    and relationship detection to support automated ML pipeline recommendations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the StatisticalAnalyzer.
        
        Args:
            confidence_level: Confidence level for statistical tests and intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Common distributions to test
        self.distributions = [
            stats.norm,      # Normal
            stats.expon,     # Exponential
            stats.gamma,     # Gamma
            stats.beta,      # Beta
            stats.uniform,   # Uniform
            stats.lognorm,   # Log-normal
            stats.chi2,      # Chi-squared
            stats.t,         # Student's t
            stats.weibull_min,  # Weibull
            stats.pareto,    # Pareto
        ]
    
    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, DistributionAnalysis]:
        """
        Analyze distributions for all numerical columns in the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping column names to distribution analysis results
        """
        results = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            series = df[column].dropna()
            if len(series) < 10:  # Skip columns with too few data points
                continue
                
            try:
                results[column] = self._analyze_single_distribution(series)
            except Exception as e:
                warnings.warn(f"Failed to analyze distribution for column {column}: {str(e)}")
                continue
                
        return results
    
    def _analyze_single_distribution(self, series: pd.Series) -> DistributionAnalysis:
        """
        Analyze distribution for a single numerical series.
        
        Args:
            series: Numerical data series
            
        Returns:
            Distribution analysis results
        """
        data = series.values
        
        # Fit distributions and find best fit
        distribution_fits = self._fit_distributions(data)
        best_fit = min(distribution_fits, key=lambda x: x.aic)
        alternative_fits = [fit for fit in distribution_fits if fit != best_fit][:3]
        
        # Test for normality
        normality_tests = self._test_normality(data)
        is_normal = any(test.get('p_value', 0) > self.alpha for test in normality_tests.values() if 'p_value' in test)
        
        # Test for multimodality
        is_multimodal = self._test_multimodality(data)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        return DistributionAnalysis(
            best_fit=best_fit,
            alternative_fits=alternative_fits,
            is_normal=is_normal,
            is_multimodal=is_multimodal,
            skewness=skewness,
            kurtosis=kurtosis,
            normality_tests=normality_tests
        )
    
    def _fit_distributions(self, data: np.ndarray) -> List[DistributionFit]:
        """
        Fit multiple distributions to the data and return results.
        
        Args:
            data: Numerical data array
            
        Returns:
            List of distribution fits sorted by goodness of fit
        """
        fits = []
        
        # Use a subset of faster distributions to avoid hanging
        fast_distributions = [
            stats.norm,      # Normal
            stats.expon,     # Exponential
            stats.uniform,   # Uniform
            stats.lognorm,   # Log-normal
        ]
        
        for distribution in fast_distributions:
            try:
                # Add timeout protection by limiting data size for complex distributions
                sample_data = data
                if len(data) > 1000:
                    # Sample data for faster fitting
                    sample_indices = np.random.choice(len(data), 1000, replace=False)
                    sample_data = data[sample_indices]
                
                # Fit distribution with error handling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = distribution.fit(sample_data)
                
                # Validate parameters
                if not all(np.isfinite(param) for param in params):
                    continue
                
                # Calculate goodness of fit using Kolmogorov-Smirnov test
                ks_stat, p_value = kstest(sample_data, lambda x: distribution.cdf(x, *params))
                
                # Calculate AIC and BIC
                try:
                    log_likelihood = np.sum(distribution.logpdf(sample_data, *params))
                    if not np.isfinite(log_likelihood):
                        continue
                except:
                    continue
                    
                k = len(params)  # Number of parameters
                n = len(sample_data)
                aic = float(2 * k - 2 * log_likelihood)
                bic = float(k * np.log(n) - 2 * log_likelihood)
                
                # Simplified confidence interval calculation
                confidence_interval = self._calculate_confidence_interval(distribution, params, sample_data)
                
                # Create parameter dictionary
                param_names = []
                if distribution.shapes:
                    param_names = [name.strip() for name in distribution.shapes.split(',')]
                param_names.extend(['loc', 'scale'])
                param_dict = dict(zip(param_names[:len(params)], params))
                
                fits.append(DistributionFit(
                    distribution_name=distribution.name,
                    parameters=param_dict,
                    goodness_of_fit=max(0, 1 - ks_stat),  # Convert to goodness measure
                    p_value=p_value,
                    confidence_interval=confidence_interval,
                    aic=aic,
                    bic=bic
                ))
                
            except Exception as e:
                # Skip distributions that fail to fit
                continue
        
        # If no distributions fit successfully, create a default normal fit
        if not fits:
            try:
                mean = np.mean(data)
                std = np.std(data)
                fits.append(DistributionFit(
                    distribution_name="norm",
                    parameters={"loc": mean, "scale": std},
                    goodness_of_fit=0.5,
                    p_value=0.5,
                    confidence_interval=(mean - std, mean + std),
                    aic=float(len(data)),
                    bic=float(len(data))
                ))
            except:
                pass
        
        return sorted(fits, key=lambda x: x.aic)
    
    def _calculate_confidence_interval(self, distribution, params, data) -> Tuple[float, float]:
        """
        Calculate confidence interval for distribution parameters.
        
        Args:
            distribution: Scipy distribution object
            params: Fitted parameters
            data: Original data
            
        Returns:
            Confidence interval tuple
        """
        try:
            # Use simple standard error estimation for speed
            if len(params) > 0:
                # For the first parameter (usually location/mean)
                std_error = np.std(data) / np.sqrt(len(data))
                margin = 1.96 * std_error  # 95% confidence interval
                return (float(params[0] - margin), float(params[0] + margin))
            else:
                # Fallback if no parameters
                mean = np.mean(data)
                std = np.std(data)
                return (float(mean - std), float(mean + std))
                
        except Exception:
            # Final fallback
            try:
                mean = np.mean(data)
                std = np.std(data)
                return (float(mean - std), float(mean + std))
            except:
                return (0.0, 1.0)
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Perform multiple normality tests on the data.
        
        Args:
            data: Numerical data array
            
        Returns:
            Dictionary of normality test results
        """
        tests = {}
        
        # Shapiro-Wilk test (for small samples)
        if len(data) <= 5000:
            try:
                stat, p_value = stats.shapiro(data)
                tests['shapiro_wilk'] = {'statistic': stat, 'p_value': p_value}
            except Exception:
                pass
        
        # D'Agostino's normality test
        try:
            stat, p_value = normaltest(data)
            tests['dagostino'] = {'statistic': stat, 'p_value': p_value}
        except Exception:
            pass
        
        # Jarque-Bera test
        try:
            stat, p_value = jarque_bera(data)
            tests['jarque_bera'] = {'statistic': stat, 'p_value': p_value}
        except Exception:
            pass
        
        # Anderson-Darling test
        try:
            result = anderson(data, dist='norm')
            tests['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist()
            }
        except Exception:
            pass
        
        return tests
    
    def _test_multimodality(self, data: np.ndarray) -> bool:
        """
        Test if the data has multiple modes (multimodal distribution).
        
        Args:
            data: Numerical data array
            
        Returns:
            True if multimodal, False otherwise
        """
        try:
            # Use kernel density estimation to detect peaks
            from scipy.signal import find_peaks
            
            # Create histogram
            hist, bin_edges = np.histogram(data, bins='auto', density=True)
            
            # Find peaks in the histogram
            peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
            
            # Consider multimodal if more than one significant peak
            return len(peaks) > 1
            
        except Exception:
            # Fallback: use simple statistical test
            # If kurtosis is very negative, might indicate multimodality
            return stats.kurtosis(data) < -1.2    

    def compute_correlations(self, df: pd.DataFrame) -> CorrelationMatrix:
        """
        Compute various correlation matrices for the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Correlation matrix results
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            raise NeuroLiteException("No numerical columns found for correlation analysis")
        
        # Remove columns with all NaN values
        numerical_df = numerical_df.dropna(axis=1, how='all')
        column_names = numerical_df.columns.tolist()
        
        # Pearson correlation
        pearson_corr = numerical_df.corr(method='pearson').values
        
        # Spearman correlation
        spearman_corr = numerical_df.corr(method='spearman').values
        
        # Kendall correlation
        kendall_corr = numerical_df.corr(method='kendall').values
        
        # Mutual information
        mutual_info = self._compute_mutual_information(numerical_df)
        
        return CorrelationMatrix(
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            kendall_correlation=kendall_corr,
            mutual_information=mutual_info,
            column_names=column_names
        )
    
    def _compute_mutual_information(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute mutual information matrix for numerical features.
        
        Args:
            df: Numerical dataframe
            
        Returns:
            Mutual information matrix
        """
        n_features = len(df.columns)
        mi_matrix = np.zeros((n_features, n_features))
        
        # Fill diagonal with 1s (perfect mutual information with self)
        np.fill_diagonal(mi_matrix, 1.0)
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Only compute upper triangle
                    try:
                        # Remove NaN values
                        data = df[[col1, col2]].dropna()
                        if len(data) < 10:
                            mi_value = 0.0
                        else:
                            # Discretize continuous variables for MI calculation
                            x = data[col1].values.reshape(-1, 1)
                            y = data[col2].values
                            
                            # Use mutual_info_regression for continuous target
                            mi_value = mutual_info_regression(x, y, random_state=42)[0]
                            
                            # Normalize to [0, 1] range
                            mi_value = min(mi_value, 1.0)
                        
                        mi_matrix[i, j] = mi_value
                        mi_matrix[j, i] = mi_value  # Symmetric matrix
                        
                    except Exception:
                        mi_matrix[i, j] = 0.0
                        mi_matrix[j, i] = 0.0
        
        return mi_matrix
    
    def detect_relationships(self, df: pd.DataFrame) -> RelationshipAnalysis:
        """
        Detect linear and non-linear relationships between features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Relationship analysis results
        """
        numerical_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        if numerical_df.empty:
            raise NeuroLiteException("No numerical columns found for relationship analysis")
        
        # Linear relationships (Pearson correlation)
        linear_relationships = self._detect_linear_relationships(numerical_df)
        
        # Non-linear relationships
        non_linear_relationships = self._detect_nonlinear_relationships(numerical_df)
        
        # Multicollinearity detection using VIF
        multicollinearity_vif = self._calculate_vif(numerical_df)
        
        # Feature dependencies
        feature_dependencies = self._detect_feature_dependencies(numerical_df)
        
        return RelationshipAnalysis(
            linear_relationships=linear_relationships,
            non_linear_relationships=non_linear_relationships,
            multicollinearity_vif=multicollinearity_vif,
            feature_dependencies=feature_dependencies
        )
    
    def _detect_linear_relationships(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect significant linear relationships between features.
        
        Args:
            df: Numerical dataframe
            
        Returns:
            Dictionary of significant linear relationships
        """
        relationships = {}
        corr_matrix = df.corr(method='pearson')
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # Only check upper triangle
                    correlation = abs(corr_matrix.loc[col1, col2])
                    if correlation > 0.3:  # Threshold for significant correlation
                        relationships[f"{col1}_vs_{col2}"] = correlation
        
        return relationships
    
    def _detect_nonlinear_relationships(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect non-linear relationships using mutual information and rank correlation.
        
        Args:
            df: Numerical dataframe
            
        Returns:
            Dictionary of non-linear relationship strengths
        """
        relationships = {}
        
        # Compute Spearman correlation (rank-based, captures monotonic non-linear)
        spearman_corr = df.corr(method='spearman')
        pearson_corr = df.corr(method='pearson')
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:
                    spearman_val = abs(spearman_corr.loc[col1, col2])
                    pearson_val = abs(pearson_corr.loc[col1, col2])
                    
                    # Non-linear relationship if Spearman >> Pearson
                    if spearman_val > 0.3 and (spearman_val - pearson_val) > 0.1:
                        relationships[f"{col1}_vs_{col2}"] = spearman_val - pearson_val
        
        return relationships
    
    def _calculate_vif(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factor for multicollinearity detection.
        
        Args:
            df: Numerical dataframe
            
        Returns:
            Dictionary of VIF values for each feature
        """
        from sklearn.linear_model import LinearRegression
        
        vif_dict = {}
        
        # Need at least 2 features for VIF calculation
        if len(df.columns) < 2:
            return vif_dict
        
        for i, feature in enumerate(df.columns):
            try:
                # Prepare data
                X = df.drop(columns=[feature]).values
                y = df[feature].values
                
                # Remove rows with NaN values
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 10:  # Not enough data
                    vif_dict[feature] = 1.0
                    continue
                
                # Fit linear regression
                reg = LinearRegression()
                reg.fit(X_clean, y_clean)
                
                # Calculate R-squared
                r_squared = reg.score(X_clean, y_clean)
                
                # Calculate VIF
                if r_squared >= 0.999:  # Avoid division by zero
                    vif_dict[feature] = float('inf')
                else:
                    vif_dict[feature] = 1 / (1 - r_squared)
                    
            except Exception:
                vif_dict[feature] = 1.0  # Default value if calculation fails
        
        return vif_dict
    
    def _detect_feature_dependencies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect which features are dependent on others.
        
        Args:
            df: Numerical dataframe
            
        Returns:
            Dictionary mapping features to their dependencies
        """
        dependencies = {}
        correlation_matrix = df.corr(method='pearson').abs()
        
        for feature in df.columns:
            # Find features highly correlated with this feature
            correlated_features = []
            for other_feature in df.columns:
                if feature != other_feature:
                    correlation = correlation_matrix.loc[feature, other_feature]
                    if correlation > 0.7:  # High correlation threshold
                        correlated_features.append(other_feature)
            
            if correlated_features:
                dependencies[feature] = correlated_features
        
        return dependencies
    
    def analyze_comprehensive(self, df: pd.DataFrame) -> StatisticalProperties:
        """
        Perform comprehensive statistical analysis combining all methods.
        
        Args:
            df: Input dataframe
            
        Returns:
            Statistical properties object
        """
        try:
            # Distribution analysis
            distribution_results = self.analyze_distributions(df)
            
            # Find the most common distribution across all columns
            distribution_counts = {}
            for col_analysis in distribution_results.values():
                dist_name = col_analysis.best_fit.distribution_name
                distribution_counts[dist_name] = distribution_counts.get(dist_name, 0) + 1
            
            most_common_distribution = max(distribution_counts.items(), key=lambda x: x[1])[0] if distribution_counts else "unknown"
            
            # Correlation analysis
            correlation_results = self.compute_correlations(df)
            
            # Relationship analysis
            relationship_results = self.detect_relationships(df)
            
            # Combine parameters from all distribution analyses
            combined_parameters = {}
            for col, analysis in distribution_results.items():
                for param_name, param_value in analysis.best_fit.parameters.items():
                    combined_parameters[f"{col}_{param_name}"] = param_value
            
            # Feature importance based on correlation strength
            feature_importance = {}
            if len(correlation_results.column_names) > 1:
                for i, col in enumerate(correlation_results.column_names):
                    # Average absolute correlation with other features
                    correlations = correlation_results.pearson_correlation[i, :]
                    avg_correlation = np.mean(np.abs(correlations[correlations != 1.0]))  # Exclude self-correlation
                    feature_importance[col] = avg_correlation
            
            # Detect outliers using IQR method
            outlier_indices = self._detect_outliers(df)
            
            return StatisticalProperties(
                distribution=most_common_distribution,
                parameters=combined_parameters,
                correlation_matrix=correlation_results.pearson_correlation,
                feature_importance=feature_importance,
                outlier_indices=outlier_indices
            )
            
        except Exception as e:
            # Return minimal statistical properties if analysis fails
            warnings.warn(f"Statistical analysis failed: {str(e)}")
            return StatisticalProperties(
                distribution="unknown",
                parameters={},
                correlation_matrix=None,
                feature_importance={},
                outlier_indices=[]
            )
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[int]:
        """
        Detect outliers using the IQR method across all numerical columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of row indices that are outliers
        """
        numerical_df = df.select_dtypes(include=[np.number])
        outlier_indices = set()
        
        for column in numerical_df.columns:
            series = numerical_df[column].dropna()
            if len(series) < 4:  # Need at least 4 points for quartiles
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            column_outliers = series[(series < lower_bound) | (series > upper_bound)].index
            outlier_indices.update(column_outliers.tolist())
        
        return sorted(list(outlier_indices))