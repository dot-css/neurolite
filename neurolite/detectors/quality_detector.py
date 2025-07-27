"""
Quality detector for comprehensive data quality assessment.

This module provides functionality for detecting data quality issues including
missing data patterns, duplicates, consistency issues, and validation problems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
import re
from dataclasses import dataclass

from ..core.data_models import QualityMetrics, MissingDataAnalysis
from ..core.exceptions import NeuroLiteException, InsufficientDataError


@dataclass
class DuplicateAnalysis:
    """Analysis results for duplicate detection."""
    duplicate_count: int
    duplicate_percentage: float
    duplicate_rows: List[int]
    exact_duplicates: int
    partial_duplicates: int
    
    def __post_init__(self):
        """Validate DuplicateAnalysis data after initialization."""
        if self.duplicate_count < 0:
            raise ValueError("Duplicate count cannot be negative")
        if not 0.0 <= self.duplicate_percentage <= 1.0:
            raise ValueError("Duplicate percentage must be between 0.0 and 1.0")


@dataclass
class ConsistencyReport:
    """Report for data consistency validation."""
    format_consistency_score: float
    range_consistency_score: float
    referential_integrity_score: float
    inconsistent_formats: Dict[str, List[str]]
    range_violations: Dict[str, List[int]]
    integrity_violations: List[str]
    
    def __post_init__(self):
        """Validate ConsistencyReport data after initialization."""
        scores = [self.format_consistency_score, self.range_consistency_score, self.referential_integrity_score]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("Consistency scores must be between 0.0 and 1.0")


class QualityDetector:
    """
    Detector for comprehensive data quality assessment.
    
    Provides methods for analyzing missing data patterns, detecting duplicates,
    validating data consistency, and recommending quality improvement strategies.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        """
        Initialize QualityDetector.
        
        Args:
            confidence_threshold: Minimum confidence threshold for classifications
        """
        self.confidence_threshold = confidence_threshold
        
    def analyze_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            QualityMetrics object containing all quality assessments
            
        Raises:
            InsufficientDataError: If dataset is too small for analysis
        """
        if df.empty:
            raise InsufficientDataError(0, 1, "dataset")
            
        if len(df) < 2:
            raise InsufficientDataError(len(df), 2, "dataset")
        
        # Calculate quality metrics
        completeness = self._calculate_completeness(df)
        consistency = self._calculate_consistency(df)
        validity = self._calculate_validity(df)
        uniqueness = self._calculate_uniqueness(df)
        missing_pattern = self._detect_missing_pattern_type(df)
        duplicate_count = self._count_duplicates(df)
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
            missing_pattern=missing_pattern,
            duplicate_count=duplicate_count
        )
    
    def detect_missing_patterns(self, df: pd.DataFrame) -> MissingDataAnalysis:
        """
        Detect and analyze missing data patterns.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            MissingDataAnalysis object with detailed missing data information
        """
        if df.empty:
            return MissingDataAnalysis(
                missing_percentage=0.0,
                missing_pattern_type='UNKNOWN',
                missing_columns=[],
                imputation_strategy='none'
            )
        
        # Calculate missing percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells if total_cells > 0 else 0.0
        
        # Identify columns with missing data
        missing_columns = df.columns[df.isnull().any()].tolist()
        
        # Classify missing data pattern
        pattern_type = self._classify_missing_pattern(df)
        
        # Recommend imputation strategy
        imputation_strategy = self._recommend_imputation_strategy(df, pattern_type)
        
        return MissingDataAnalysis(
            missing_percentage=missing_percentage,
            missing_pattern_type=pattern_type,
            missing_columns=missing_columns,
            imputation_strategy=imputation_strategy
        )
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        if df.empty:
            return 0.0
        
        total_cells = df.size
        non_null_cells = df.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        if df.empty:
            return 0.0
        
        consistency_scores = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # For text columns, check format consistency
                consistency_scores.append(self._check_format_consistency(df[column]))
            else:
                # For numeric columns, check range consistency
                consistency_scores.append(self._check_range_consistency(df[column]))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """Calculate data validity score."""
        if df.empty:
            return 0.0
        
        validity_scores = []
        
        for column in df.columns:
            valid_ratio = self._check_column_validity(df[column])
            validity_scores.append(valid_ratio)
        
        return np.mean(validity_scores) if validity_scores else 0.0
    
    def _calculate_uniqueness(self, df: pd.DataFrame) -> float:
        """Calculate data uniqueness score."""
        if df.empty:
            return 0.0
        
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        return unique_rows / total_rows if total_rows > 0 else 0.0
    
    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count duplicate rows in the dataset."""
        if df.empty:
            return 0
        
        return len(df) - len(df.drop_duplicates())
    
    def _detect_missing_pattern_type(self, df: pd.DataFrame) -> str:
        """Detect the overall missing data pattern type."""
        if df.empty or not df.isnull().any().any():
            return 'UNKNOWN'
        
        return self._classify_missing_pattern(df)
    
    def _classify_missing_pattern(self, df: pd.DataFrame) -> str:
        """
        Classify missing data pattern as MCAR, MAR, or MNAR.
        
        Args:
            df: Input DataFrame
            
        Returns:
            String indicating pattern type: 'MCAR', 'MAR', 'MNAR', or 'UNKNOWN'
        """
        if df.empty or not df.isnull().any().any():
            return 'UNKNOWN'
        
        # Get missing data indicator matrix
        missing_matrix = df.isnull()
        
        # If no missing data, return UNKNOWN
        if not missing_matrix.any().any():
            return 'UNKNOWN'
        
        # Test for MCAR using Little's MCAR test approximation
        if self._test_mcar(df):
            return 'MCAR'
        
        # Test for MAR by checking correlations between missingness patterns
        if self._test_mar(df):
            return 'MAR'
        
        # If neither MCAR nor MAR, assume MNAR
        return 'MNAR'
    
    def _test_mcar(self, df: pd.DataFrame) -> bool:
        """
        Test if missing data is Missing Completely At Random (MCAR).
        
        Uses a simplified approach based on missing data patterns.
        """
        missing_matrix = df.isnull()
        
        # If very little missing data, assume MCAR
        missing_percentage = missing_matrix.sum().sum() / df.size
        if missing_percentage < 0.15:  # Less than 15% missing, more lenient
            return True
        
        # Check if missing data is randomly distributed across rows
        missing_per_row = missing_matrix.sum(axis=1)
        
        # Test if missing counts per row follow expected random distribution
        try:
            # Use chi-square test for randomness with more lenient threshold
            observed_counts = np.bincount(missing_per_row)
            expected_mean = missing_percentage * len(df.columns)
            
            # If missing data is truly random, it should follow Poisson distribution
            expected_counts = []
            for i in range(len(observed_counts)):
                expected = len(df) * stats.poisson.pmf(i, expected_mean)
                expected_counts.append(max(expected, 1))  # Avoid zero expected counts
            
            chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
            return p_value > 0.01  # More lenient threshold for randomness
            
        except (ValueError, ZeroDivisionError):
            # If test fails, use simple heuristic
            # Check if missing data is roughly evenly distributed
            missing_per_column = missing_matrix.sum(axis=0)
            if len(missing_per_column) > 0:
                cv = missing_per_column.std() / (missing_per_column.mean() + 1e-10)
                return cv < 1.0  # Low coefficient of variation suggests randomness
            return False
    
    def _test_mar(self, df: pd.DataFrame) -> bool:
        """
        Test if missing data is Missing At Random (MAR).
        
        Checks if missingness in one column is related to observed values in other columns.
        """
        missing_matrix = df.isnull()
        
        # For each column with missing data, check correlation with other columns
        for col in df.columns:
            if missing_matrix[col].any():
                # Create missingness indicator
                missing_indicator = missing_matrix[col].astype(int)
                
                # Check correlation with other non-missing columns
                for other_col in df.columns:
                    if col != other_col and not missing_matrix[other_col].all():
                        try:
                            # For numeric columns, use correlation
                            if pd.api.types.is_numeric_dtype(df[other_col]):
                                correlation = df[other_col].corr(missing_indicator)
                                if abs(correlation) > 0.3:  # Moderate correlation
                                    return True
                            
                            # For categorical columns, use chi-square test
                            else:
                                observed_data = df[other_col].dropna()
                                if len(observed_data.unique()) > 1:
                                    # Create contingency table
                                    contingency = pd.crosstab(
                                        df[other_col].fillna('Missing'),
                                        missing_indicator
                                    )
                                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                                    if p_value < 0.05:  # Significant association
                                        return True
                                        
                        except (ValueError, TypeError):
                            continue
        
        return False
    
    def _recommend_imputation_strategy(self, df: pd.DataFrame, pattern_type: str) -> str:
        """
        Recommend appropriate imputation strategy based on missing data pattern.
        
        Args:
            df: Input DataFrame
            pattern_type: Type of missing data pattern
            
        Returns:
            String describing recommended imputation strategy
        """
        if not df.isnull().any().any():
            return 'none'
        
        missing_percentage = df.isnull().sum().sum() / df.size
        
        # High missing data percentage
        if missing_percentage > 0.5:
            return 'consider_dropping_columns'
        
        # Very low missing data (less than 10% for simple deletion)
        if missing_percentage < 0.1:
            return 'simple_deletion'
        
        # Strategy based on pattern type
        if pattern_type == 'MCAR':
            return 'mean_median_mode_imputation'
        elif pattern_type == 'MAR':
            return 'multiple_imputation'
        elif pattern_type == 'MNAR':
            return 'domain_specific_imputation'
        else:
            return 'mean_median_mode_imputation'
    
    def _check_format_consistency(self, series: pd.Series) -> float:
        """Check format consistency for text columns."""
        if series.empty or series.isnull().all():
            return 0.0
        
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return 0.0
        
        # Check for consistent patterns (e.g., email, phone, date formats)
        patterns_found = set()
        
        for value in non_null_values.astype(str):
            # Simple pattern detection
            if '@' in value and '.' in value:
                patterns_found.add('email')
            elif value.replace('-', '').replace('(', '').replace(')', '').replace(' ', '').isdigit():
                patterns_found.add('phone')
            elif len(value.split('/')) == 3 or len(value.split('-')) == 3:
                patterns_found.add('date')
            else:
                patterns_found.add('text')
        
        # Consistency is higher when fewer patterns are found
        return 1.0 / len(patterns_found) if patterns_found else 0.0
    
    def _check_range_consistency(self, series: pd.Series) -> float:
        """Check range consistency for numeric columns."""
        if series.empty or series.isnull().all():
            return 0.0
        
        non_null_values = series.dropna()
        if len(non_null_values) == 0:
            return 0.0
        
        # Check for outliers using IQR method
        Q1 = non_null_values.quantile(0.25)
        Q3 = non_null_values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return 1.0  # All values are the same
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = non_null_values[(non_null_values < lower_bound) | (non_null_values > upper_bound)]
        outlier_ratio = len(outliers) / len(non_null_values)
        
        return 1.0 - outlier_ratio
    
    def _check_column_validity(self, series: pd.Series) -> float:
        """Check validity of values in a column."""
        if series.empty:
            return 0.0
        
        total_values = len(series)
        valid_values = 0
        
        for value in series:
            if pd.isnull(value):
                continue  # Skip null values in validity check
            
            # Basic validity checks
            if pd.api.types.is_numeric_dtype(series):
                # For numeric data, check if finite
                if np.isfinite(value):
                    valid_values += 1
            else:
                # For non-numeric data, check if not empty string
                if str(value).strip():
                    valid_values += 1
        
        return valid_values / total_values if total_values > 0 else 0.0
    
    def find_duplicates(self, df: pd.DataFrame) -> DuplicateAnalysis:
        """
        Detect and analyze duplicate records in the dataset.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            DuplicateAnalysis object with detailed duplicate information
        """
        if df.empty:
            return DuplicateAnalysis(
                duplicate_count=0,
                duplicate_percentage=0.0,
                duplicate_rows=[],
                exact_duplicates=0,
                partial_duplicates=0
            )
        
        # Find exact duplicates
        exact_duplicates = df.duplicated()
        exact_duplicate_count = exact_duplicates.sum()
        exact_duplicate_rows = df.index[exact_duplicates].tolist()
        
        # Find partial duplicates (similar but not identical rows)
        partial_duplicate_count = self._find_partial_duplicates(df)
        
        total_duplicates = exact_duplicate_count + partial_duplicate_count
        duplicate_percentage = min(1.0, total_duplicates / len(df)) if len(df) > 0 else 0.0
        
        return DuplicateAnalysis(
            duplicate_count=total_duplicates,
            duplicate_percentage=duplicate_percentage,
            duplicate_rows=exact_duplicate_rows,
            exact_duplicates=exact_duplicate_count,
            partial_duplicates=partial_duplicate_count
        )
    
    def validate_consistency(self, df: pd.DataFrame) -> ConsistencyReport:
        """
        Validate data consistency across multiple dimensions.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            ConsistencyReport object with detailed consistency analysis
        """
        if df.empty:
            return ConsistencyReport(
                format_consistency_score=0.0,
                range_consistency_score=0.0,
                referential_integrity_score=0.0,
                inconsistent_formats={},
                range_violations={},
                integrity_violations=[]
            )
        
        # Check format consistency
        format_score, format_issues = self._validate_format_consistency(df)
        
        # Check range consistency
        range_score, range_issues = self._validate_range_consistency(df)
        
        # Check referential integrity
        integrity_score, integrity_issues = self._validate_referential_integrity(df)
        
        return ConsistencyReport(
            format_consistency_score=format_score,
            range_consistency_score=range_score,
            referential_integrity_score=integrity_score,
            inconsistent_formats=format_issues,
            range_violations=range_issues,
            integrity_violations=integrity_issues
        )
    
    def _find_partial_duplicates(self, df: pd.DataFrame) -> int:
        """
        Find partial duplicates using similarity measures.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Number of partial duplicates found
        """
        if df.empty or len(df) < 2:
            return 0
        
        partial_duplicates = 0
        
        # For text columns, use string similarity
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            non_null_values = df[col].dropna().astype(str)
            if len(non_null_values) < 2:
                continue
            
            # Simple similarity check using string length and first few characters
            for i, val1 in enumerate(non_null_values):
                for j, val2 in enumerate(non_null_values[i+1:], i+1):
                    if self._calculate_string_similarity(val1, val2) > 0.8:
                        partial_duplicates += 1
                        break  # Avoid counting same pair multiple times
        
        return partial_duplicates
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using simple metrics.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Simple similarity based on common characters and length
        common_chars = set(str1.lower()) & set(str2.lower())
        total_chars = set(str1.lower()) | set(str2.lower())
        
        if not total_chars:
            return 0.0
        
        char_similarity = len(common_chars) / len(total_chars)
        
        # Length similarity
        max_len = max(len(str1), len(str2))
        min_len = min(len(str1), len(str2))
        length_similarity = min_len / max_len if max_len > 0 else 0.0
        
        # Combined similarity
        return (char_similarity + length_similarity) / 2.0
    
    def _validate_format_consistency(self, df: pd.DataFrame) -> Tuple[float, Dict[str, List[str]]]:
        """
        Validate format consistency across columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (consistency_score, format_issues_dict)
        """
        if df.empty:
            return 0.0, {}
        
        format_scores = []
        format_issues = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                score = self._check_format_consistency(df[column])
                format_scores.append(score)
                
                # Identify specific format issues
                if score < 0.8:  # Threshold for format inconsistency
                    issues = self._identify_format_issues(df[column])
                    if issues:
                        format_issues[column] = issues
        
        overall_score = np.mean(format_scores) if format_scores else 1.0
        return overall_score, format_issues
    
    def _validate_range_consistency(self, df: pd.DataFrame) -> Tuple[float, Dict[str, List[int]]]:
        """
        Validate range consistency for numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (consistency_score, range_violations_dict)
        """
        if df.empty:
            return 0.0, {}
        
        range_scores = []
        range_violations = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            score = self._check_range_consistency(df[column])
            range_scores.append(score)
            
            # Identify specific range violations
            if score < 0.8:  # Threshold for range inconsistency
                violations = self._identify_range_violations(df[column])
                if violations:
                    range_violations[column] = violations
        
        overall_score = np.mean(range_scores) if range_scores else 1.0
        return overall_score, range_violations
    
    def _validate_referential_integrity(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Validate referential integrity within the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (integrity_score, integrity_violations_list)
        """
        if df.empty:
            return 0.0, []
        
        violations = []
        
        # Check for potential foreign key relationships
        # This is a simplified check - in practice, this would require domain knowledge
        
        # Look for columns that might be related (e.g., ID columns)
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        
        for col in id_columns:
            # Check if all values in ID column are unique (primary key constraint)
            if df[col].duplicated().any():
                violations.append(f"Duplicate values found in potential primary key column: {col}")
            
            # Check for null values in ID columns
            if df[col].isnull().any():
                violations.append(f"Null values found in ID column: {col}")
        
        # Check for orphaned references (simplified)
        for col in df.columns:
            if col.endswith('_id') and col != col.replace('_id', ''):
                parent_col = col.replace('_id', '')
                if parent_col in df.columns:
                    # Check if all foreign key values exist in parent column
                    foreign_values = set(df[col].dropna())
                    parent_values = set(df[parent_col].dropna())
                    orphaned = foreign_values - parent_values
                    if orphaned:
                        violations.append(f"Orphaned references in {col}: {list(orphaned)[:5]}")  # Show first 5
        
        # Calculate integrity score based on violations
        total_checks = len(id_columns) * 2 + len([col for col in df.columns if col.endswith('_id')])
        integrity_score = max(0.0, 1.0 - len(violations) / max(total_checks, 1))
        
        return integrity_score, violations
    
    def _identify_format_issues(self, series: pd.Series) -> List[str]:
        """
        Identify specific format inconsistencies in a column.
        
        Args:
            series: Input pandas Series
            
        Returns:
            List of format issue descriptions
        """
        issues = []
        non_null_values = series.dropna().astype(str)
        
        if len(non_null_values) == 0:
            return issues
        
        # Check for mixed date formats
        date_formats = set()
        for value in non_null_values:
            if re.match(r'\d{4}-\d{2}-\d{2}', value):
                date_formats.add('YYYY-MM-DD')
            elif re.match(r'\d{2}/\d{2}/\d{4}', value):
                date_formats.add('MM/DD/YYYY')
            elif re.match(r'\d{2}-\d{2}-\d{4}', value):
                date_formats.add('MM-DD-YYYY')
        
        if len(date_formats) > 1:
            issues.append(f"Mixed date formats found: {', '.join(date_formats)}")
        
        # Check for mixed email formats
        email_patterns = set()
        for value in non_null_values:
            if '@' in value:
                if value.count('@') == 1:
                    email_patterns.add('standard_email')
                else:
                    email_patterns.add('malformed_email')
        
        if 'malformed_email' in email_patterns:
            issues.append("Malformed email addresses found")
        
        # Check for mixed phone number formats
        phone_patterns = set()
        for value in non_null_values:
            clean_value = re.sub(r'[^\d]', '', value)
            if clean_value.isdigit():
                if len(clean_value) == 10:
                    phone_patterns.add('10_digit')
                elif len(clean_value) == 11:
                    phone_patterns.add('11_digit')
                else:
                    phone_patterns.add('other_length')
        
        if len(phone_patterns) > 1:
            issues.append(f"Mixed phone number formats: {', '.join(phone_patterns)}")
        
        return issues
    
    def _identify_range_violations(self, series: pd.Series) -> List[int]:
        """
        Identify specific range violations in a numeric column.
        
        Args:
            series: Input pandas Series
            
        Returns:
            List of row indices with range violations
        """
        violations = []
        non_null_values = series.dropna()
        
        if len(non_null_values) == 0:
            return violations
        
        # Use IQR method to identify outliers
        Q1 = non_null_values.quantile(0.25)
        Q3 = non_null_values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return violations  # All values are the same
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find indices of outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        violations = series.index[outlier_mask].tolist()
        
        return violations