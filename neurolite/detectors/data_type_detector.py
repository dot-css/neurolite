"""
Data type detector for column classification and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy import stats
from datetime import datetime
import re

from ..core.data_models import (
    ColumnType, NumericalAnalysis, CategoricalAnalysis, 
    TemporalAnalysis, TextAnalysis
)
from ..core.exceptions import NeuroLiteException


class DataTypeDetector:
    """Detector for classifying data types and analyzing column characteristics."""
    
    def __init__(self):
        """Initialize the DataTypeDetector."""
        self.confidence_threshold = 0.8
        self.max_categorical_cardinality = 50
        self.min_samples_for_analysis = 10
        self.min_text_samples = 5  # Lower threshold for text analysis
    
    def classify_columns(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        """Classify all columns in a DataFrame."""
        if df.empty:
            raise NeuroLiteException("Cannot classify columns in empty DataFrame")
        
        column_types = {}
        for column in df.columns:
            column_types[column] = self._classify_single_column(df[column])
        return column_types
    
    def _classify_single_column(self, series: pd.Series) -> ColumnType:
        """Classify a single column/series."""
        clean_series = series.dropna()
        
        if len(clean_series) < self.min_samples_for_analysis:
            return ColumnType(
                primary_type='text',
                subtype='insufficient_data',
                confidence=0.1,
                properties={'sample_size': len(clean_series)}
            )
        
        if self._is_numerical(clean_series):
            return self._classify_numerical(clean_series)
        
        if self._is_temporal(clean_series):
            return self._classify_temporal(clean_series)
        
        if self._is_categorical(clean_series):
            return self._classify_categorical(clean_series)
        
        return self._classify_text(clean_series)
    
    def analyze_numerical(self, series: pd.Series) -> NumericalAnalysis:
        """Perform detailed numerical analysis on a series."""
        clean_series = series.dropna()
        
        if not self._is_numerical(clean_series):
            raise NeuroLiteException("Series is not numerical")
        
        data_type = self._determine_numerical_type(clean_series)
        is_continuous = self._is_continuous(clean_series)
        range_min = float(clean_series.min())
        range_max = float(clean_series.max())
        distribution_type = self._determine_distribution(clean_series)
        outlier_count = self._count_outliers(clean_series)
        
        return NumericalAnalysis(
            data_type=data_type,
            is_continuous=is_continuous,
            range_min=range_min,
            range_max=range_max,
            distribution_type=distribution_type,
            outlier_count=outlier_count
        )
    
    def _is_numerical(self, series: pd.Series) -> bool:
        """Check if a series contains numerical data."""
        if series.dtype == 'bool':
            return False
        
        try:
            pd.to_numeric(series, errors='raise')
            
            # Check if numeric strings might actually be dates in %Y%m%d format
            if series.dtype == 'object':
                sample_value = str(series.iloc[0])
                if len(sample_value) == 8 and sample_value.isdigit():
                    # Could be %Y%m%d format, check if it's a valid date
                    if self._detect_datetime_format(sample_value) == '%Y%m%d':
                        return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def _classify_numerical(self, series: pd.Series) -> ColumnType:
        """Classify numerical data and return ColumnType."""
        analysis = self.analyze_numerical(series)
        
        subtype = f"{analysis.data_type}_{'continuous' if analysis.is_continuous else 'discrete'}"
        confidence = 0.95 if len(series) > 100 else 0.85
        
        properties = {
            'range': (analysis.range_min, analysis.range_max),
            'distribution': analysis.distribution_type,
            'outlier_count': analysis.outlier_count,
            'is_continuous': analysis.is_continuous
        }
        
        return ColumnType(
            primary_type='numerical',
            subtype=subtype,
            confidence=confidence,
            properties=properties
        )
    
    def _determine_numerical_type(self, series: pd.Series) -> str:
        """Determine if numerical data is integer or float."""
        if all(float(x).is_integer() for x in series):
            if series.min() >= -2**31 and series.max() <= 2**31 - 1:
                return 'integer'
        return 'float'
    
    def _is_continuous(self, series: pd.Series) -> bool:
        """Determine if numerical data is continuous or discrete."""
        unique_ratio = len(series.unique()) / len(series)
        
        if unique_ratio > 0.5:
            return True
        
        if unique_ratio < 0.1:
            return False
        
        if series.dtype in ['float64', 'float32']:
            non_integer_count = sum(1 for x in series if not float(x).is_integer())
            if non_integer_count > len(series) * 0.1:
                return True
        
        return False
    
    def _determine_distribution(self, series: pd.Series) -> str:
        """Determine the likely distribution type of numerical data."""
        normalized = (series - series.mean()) / series.std()
        
        _, p_normal = stats.normaltest(normalized)
        if p_normal > 0.05:
            return 'normal'
        
        _, p_uniform = stats.kstest(normalized, 'uniform')
        if p_uniform > 0.05:
            return 'uniform'
        
        skewness = stats.skew(series)
        if abs(skewness) > 1:
            return 'skewed_right' if skewness > 0 else 'skewed_left'
        
        hist, _ = np.histogram(series, bins=20)
        peaks = len([i for i in range(1, len(hist)-1) 
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
        if peaks >= 2:
            return 'multimodal'
        
        return 'unknown'
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using the IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    def analyze_categorical(self, series: pd.Series) -> CategoricalAnalysis:
        """Perform detailed categorical analysis on a series."""
        clean_series = series.dropna()
        
        if not self._is_categorical(clean_series):
            raise NeuroLiteException("Series is not categorical")
        
        value_counts = clean_series.value_counts()
        unique_values = value_counts.index.astype(str).tolist()
        frequency_distribution = value_counts.to_dict()
        
        category_type = self._determine_categorical_type(clean_series)
        encoding_recommendation = self._recommend_encoding(len(unique_values), category_type)
        
        return CategoricalAnalysis(
            category_type=category_type,
            cardinality=len(unique_values),
            unique_values=unique_values,
            frequency_distribution={str(k): v for k, v in frequency_distribution.items()},
            encoding_recommendation=encoding_recommendation
        )
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if a series contains categorical data."""
        if series.dtype.name == 'category':
            return True
        
        unique_ratio = len(series.unique()) / len(series)
        cardinality = len(series.unique())
        
        # More lenient criteria for categorical detection
        if cardinality <= self.max_categorical_cardinality and unique_ratio < 0.8:
            return True
        
        return False
    
    def _classify_categorical(self, series: pd.Series) -> ColumnType:
        """Classify categorical data and return ColumnType."""
        analysis = self.analyze_categorical(series)
        
        subtype = f"{analysis.category_type}_cardinality_{analysis.cardinality}"
        unique_ratio = analysis.cardinality / len(series)
        confidence = max(0.6, 1.0 - unique_ratio)
        
        properties = {
            'cardinality': analysis.cardinality,
            'category_type': analysis.category_type,
            'encoding_recommendation': analysis.encoding_recommendation,
            'most_frequent': max(analysis.frequency_distribution.items(), 
                               key=lambda x: x[1])[0]
        }
        
        return ColumnType(
            primary_type='categorical',
            subtype=subtype,
            confidence=confidence,
            properties=properties
        )
    
    def _determine_categorical_type(self, series: pd.Series) -> str:
        """Determine if categorical data is nominal or ordinal."""
        unique_values = series.unique()
        
        # Check for obvious ordinal patterns
        ordinal_patterns = [
            r'^(low|medium|high)$',
            r'^(small|medium|large)$', 
            r'^(poor|fair|good|excellent)$',
            r'^(never|rarely|sometimes|often|always)$',
            r'^\d+$',  # Pure numbers as strings
            r'^(first|second|third|fourth|fifth)$'
        ]
        
        str_values = [str(v).lower() for v in unique_values]
        
        for pattern in ordinal_patterns:
            if any(re.match(pattern, val) for val in str_values):
                return 'ordinal'
        
        return 'nominal'
    
    def _recommend_encoding(self, cardinality: int, category_type: str) -> str:
        """Recommend encoding strategy based on categorical analysis."""
        if cardinality == 2:
            return 'binary_encoding'
        elif cardinality <= 10 and category_type == 'nominal':
            return 'one_hot_encoding'
        elif category_type == 'ordinal':
            return 'ordinal_encoding'
        else:
            return 'target_encoding'
    
    def analyze_temporal(self, series: pd.Series) -> TemporalAnalysis:
        """Perform detailed temporal analysis on a series."""
        clean_series = series.dropna()
        
        if not self._is_temporal(clean_series):
            raise NeuroLiteException("Series is not temporal")
        
        datetime_series = pd.to_datetime(clean_series, infer_datetime_format=True)
        datetime_format = self._detect_datetime_format(clean_series.iloc[0])
        frequency = self._detect_frequency(datetime_series)
        has_seasonality = self._detect_seasonality(datetime_series)
        has_trend = self._detect_trend(datetime_series)
        is_stationary = self._test_stationarity(datetime_series)
        time_range = (datetime_series.min(), datetime_series.max())
        
        return TemporalAnalysis(
            datetime_format=datetime_format,
            frequency=frequency,
            has_seasonality=has_seasonality,
            has_trend=has_trend,
            is_stationary=is_stationary,
            time_range=time_range
        )
    
    def _is_temporal(self, series: pd.Series) -> bool:
        """Check if a series contains temporal data."""
        # Exclude numerical data
        if self._is_numerical(series):
            return False
        
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        # First try pandas to_datetime
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return True
        except (ValueError, TypeError):
            pass
        
        # If pandas fails, try manual format detection
        # Require at least 80% of samples to be valid dates for temporal classification
        valid_dates = 0
        for sample_value in sample:
            if self._detect_datetime_format(sample_value) != 'unknown':
                valid_dates += 1
        
        return valid_dates / len(sample) >= 0.8
    
    def _classify_temporal(self, series: pd.Series) -> ColumnType:
        """Classify temporal data and return ColumnType."""
        analysis = self.analyze_temporal(series)
        
        subtype_parts = ['datetime']
        if analysis.frequency:
            subtype_parts.append(f"freq_{analysis.frequency}")
        if analysis.has_seasonality:
            subtype_parts.append('seasonal')
        if analysis.has_trend:
            subtype_parts.append('trending')
        
        subtype = '_'.join(subtype_parts)
        confidence = 0.9
        
        properties = {
            'datetime_format': analysis.datetime_format,
            'frequency': analysis.frequency,
            'has_seasonality': analysis.has_seasonality,
            'has_trend': analysis.has_trend,
            'is_stationary': analysis.is_stationary,
            'time_span_days': (analysis.time_range[1] - analysis.time_range[0]).days
        }
        
        return ColumnType(
            primary_type='temporal',
            subtype=subtype,
            confidence=confidence,
            properties=properties
        )
    
    def _detect_datetime_format(self, sample_value: Any) -> str:
        """Detect the datetime format of a sample value."""
        common_formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%Y%m%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        str_value = str(sample_value)
        
        for fmt in common_formats:
            try:
                datetime.strptime(str_value, fmt)
                return fmt
            except ValueError:
                continue
        
        return 'unknown'
    
    def _detect_frequency(self, datetime_series: pd.Series) -> Optional[str]:
        """Detect the frequency of a datetime series."""
        if len(datetime_series) < 3:
            return None
        
        sorted_series = datetime_series.sort_values()
        diffs = sorted_series.diff().dropna()
        
        # Convert to Series to use mode()
        diffs_series = pd.Series(diffs)
        mode_diff = diffs_series.mode()
        if len(mode_diff) == 0:
            return None
        
        mode_diff = mode_diff.iloc[0]
        
        if mode_diff.days == 1:
            return 'D'
        elif mode_diff.days == 7:
            return 'W'
        elif 28 <= mode_diff.days <= 31:
            return 'M'
        elif 365 <= mode_diff.days <= 366:
            return 'Y'
        elif mode_diff.seconds == 3600:
            return 'H'
        elif mode_diff.seconds == 60:
            return 'T'
        
        return None
    
    def _detect_seasonality(self, datetime_series: pd.Series) -> bool:
        """Simple seasonality detection based on time span."""
        if len(datetime_series) < 24:
            return False
        
        time_span = datetime_series.max() - datetime_series.min()
        return time_span.days > 730
    
    def _detect_trend(self, datetime_series: pd.Series) -> bool:
        """Simple trend detection based on time span."""
        if len(datetime_series) < 10:
            return False
        
        indices = np.arange(len(datetime_series))
        timestamps = datetime_series.sort_values().astype(np.int64)
        correlation = np.corrcoef(indices, timestamps)[0, 1]
        
        return abs(correlation) > 0.7
    
    def _test_stationarity(self, datetime_series: pd.Series) -> bool:
        """Simple stationarity test based on variance."""
        if len(datetime_series) < 10:
            return True
        
        # Convert to Series if it's a DatetimeIndex
        if isinstance(datetime_series, pd.DatetimeIndex):
            datetime_series = pd.Series(datetime_series)
        
        mid = len(datetime_series) // 2
        first_half = datetime_series.iloc[:mid]
        second_half = datetime_series.iloc[mid:]
        
        first_numeric = first_half.astype(np.int64)
        second_numeric = second_half.astype(np.int64)
        
        var1 = np.var(first_numeric)
        var2 = np.var(second_numeric)
        
        if var1 == 0 and var2 == 0:
            return True
        
        ratio = max(var1, var2) / (min(var1, var2) + 1e-10)
        return bool(ratio < 2.0)
    
    def analyze_text(self, series: pd.Series) -> TextAnalysis:
        """Perform detailed text analysis on a series."""
        clean_series = series.dropna()
        
        if len(clean_series) < self.min_text_samples:
            raise NeuroLiteException("Insufficient data for text analysis")
        
        # Convert to string and analyze
        str_series = clean_series.astype(str)
        
        # Basic text metrics
        lengths = str_series.str.len()
        avg_length = float(lengths.mean())
        max_length = int(lengths.max())
        min_length = int(lengths.min())
        unique_ratio = len(str_series.unique()) / len(str_series)
        
        # Text type classification
        text_type = self._determine_text_type(str_series)
        
        # Language detection
        language = self._detect_language(str_series)
        
        # Encoding detection
        encoding = self._detect_encoding(str_series)
        
        # Character analysis
        contains_special_chars = self._contains_special_characters(str_series)
        contains_numbers = self._contains_numbers(str_series)
        
        # Readability score (for natural language text)
        readability_score = None
        if text_type == 'natural_language' and avg_length > 20:
            readability_score = self._calculate_readability_score(str_series)
        
        return TextAnalysis(
            text_type=text_type,
            language=language,
            encoding=encoding,
            avg_length=avg_length,
            max_length=max_length,
            min_length=min_length,
            unique_ratio=unique_ratio,
            contains_special_chars=contains_special_chars,
            contains_numbers=contains_numbers,
            readability_score=readability_score
        )
    
    def _classify_text(self, series: pd.Series) -> ColumnType:
        """Classify text data and return ColumnType."""
        try:
            analysis = self.analyze_text(series)
            
            # Create subtype based on analysis
            subtype_parts = [analysis.text_type]
            if analysis.language:
                subtype_parts.append(f"lang_{analysis.language}")
            if analysis.avg_length > 100:
                subtype_parts.append('long')
            elif analysis.avg_length < 20:
                subtype_parts.append('short')
            
            subtype = '_'.join(subtype_parts)
            
            # Determine confidence based on various factors
            confidence = self._calculate_text_confidence(analysis)
            
            properties = {
                'text_type': analysis.text_type,
                'language': analysis.language,
                'encoding': analysis.encoding,
                'avg_length': analysis.avg_length,
                'unique_ratio': analysis.unique_ratio,
                'contains_special_chars': analysis.contains_special_chars,
                'contains_numbers': analysis.contains_numbers,
                'readability_score': analysis.readability_score
            }
            
            return ColumnType(
                primary_type='text',
                subtype=subtype,
                confidence=confidence,
                properties=properties
            )
            
        except NeuroLiteException:
            # Fallback to basic classification
            str_series = series.astype(str)
            avg_length = str_series.str.len().mean()
            unique_ratio = len(str_series.unique()) / len(str_series)
            
            if unique_ratio < 0.1:
                subtype = 'categorical_text'
            elif avg_length > 100:
                subtype = 'long_text'
            else:
                subtype = 'short_text'
            
            return ColumnType(
                primary_type='text',
                subtype=subtype,
                confidence=0.5,
                properties={
                    'avg_length': avg_length,
                    'unique_ratio': unique_ratio
                }
            )
    
    def _determine_text_type(self, str_series: pd.Series) -> str:
        """Determine the type of text data."""
        unique_ratio = len(str_series.unique()) / len(str_series)
        avg_length = str_series.str.len().mean()
        
        # Check for structured text patterns first (highest priority)
        if self._is_structured_text(str_series):
            return 'structured_text'
        
        # Check for categorical text (low unique ratio)
        if unique_ratio <= 0.5:
            # For categorical text, be more lenient with natural language detection
            # Only classify as natural language if it has strong NL indicators
            if avg_length > 50 or self._has_strong_natural_language_indicators(str_series):
                return 'natural_language'
            else:
                return 'categorical_text'
        
        # Check for natural language indicators
        if self._is_natural_language(str_series):
            return 'natural_language'
        
        # Mixed or unclear type
        if unique_ratio > 0.8 and avg_length > 10:
            return 'mixed'
        
        return 'categorical_text'
    
    def _is_structured_text(self, str_series: pd.Series) -> bool:
        """Check if text follows structured patterns."""
        sample = str_series.head(min(20, len(str_series)))
        
        # Common structured patterns
        structured_patterns = [
            r'^[A-Z]{2,3}-\d+$',  # Codes like ABC-123
            r'^\d{3}-\d{2}-\d{4}$',  # SSN-like patterns
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # Email
            r'^https?://[^\s]+$',  # URLs
            r'^\d{3}-\d{4}$',  # Phone numbers like 555-1234
            r'^\+?1?-?\d{3}-?\d{3}-?\d{4}$',  # Full phone numbers
            r'^[A-Z0-9]{8,}$',  # IDs/Codes
        ]
        
        for pattern in structured_patterns:
            matches = sum(1 for text in sample if re.match(pattern, str(text)))
            if matches / len(sample) > 0.7:
                return True
        
        return False
    
    def _is_natural_language(self, str_series: pd.Series) -> bool:
        """Check if text appears to be natural language."""
        sample = str_series.head(min(10, len(str_series)))
        
        natural_language_indicators = 0
        total_indicators = 0
        
        for text in sample:
            text_str = str(text).lower()
            
            # Check for common English words
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            has_common_words = any(word in text_str for word in common_words)
            if has_common_words:
                natural_language_indicators += 1
            total_indicators += 1
            
            # Check for sentence structure (spaces, punctuation)
            has_spaces = ' ' in text_str
            has_punctuation = any(char in text_str for char in '.,!?;:')
            if has_spaces and len(text_str.split()) > 3:
                natural_language_indicators += 1
            total_indicators += 1
            
            # Check average word length (natural language typically 4-6 chars)
            if has_spaces:
                words = text_str.split()
                if words:
                    avg_word_len = sum(len(word) for word in words) / len(words)
                    if 3 <= avg_word_len <= 8:
                        natural_language_indicators += 1
                total_indicators += 1
        
        return total_indicators > 0 and (natural_language_indicators / total_indicators) > 0.5
    
    def _has_strong_natural_language_indicators(self, str_series: pd.Series) -> bool:
        """Check if text has strong natural language indicators (stricter than _is_natural_language)."""
        sample = str_series.head(min(10, len(str_series)))
        
        strong_indicators = 0
        total_checks = 0
        
        for text in sample:
            text_str = str(text).lower()
            
            # Check for multiple common words (stronger indicator)
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
            word_count = sum(1 for word in common_words if word in text_str)
            if word_count >= 2:  # At least 2 common words
                strong_indicators += 1
            total_checks += 1
            
            # Check for sentence-like structure with punctuation
            has_spaces = ' ' in text_str
            has_punctuation = any(char in text_str for char in '.,!?;:')
            word_count = len(text_str.split()) if has_spaces else 0
            if has_spaces and has_punctuation and word_count >= 4:
                strong_indicators += 1
            total_checks += 1
            
            # Check for articles and prepositions (strong NL indicators)
            articles_prepositions = ['a', 'an', 'the', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to']
            if any(f' {word} ' in f' {text_str} ' for word in articles_prepositions):
                strong_indicators += 1
            total_checks += 1
        
        return total_checks > 0 and (strong_indicators / total_checks) > 0.6
    
    def _detect_language(self, str_series: pd.Series) -> Optional[str]:
        """Detect the language of text data."""
        # Simple language detection based on character patterns
        sample_text = ' '.join(str_series.head(min(5, len(str_series))).astype(str))
        
        if len(sample_text) < 20:
            return None
        
        # Basic language detection patterns - order matters for accuracy
        if re.search(r'[а-яё]', sample_text.lower()):
            return 'ru'
        elif re.search(r'[äöüß]', sample_text.lower()):
            return 'de'
        elif re.search(r'[ñ¿¡]', sample_text.lower()):
            return 'es'
        elif re.search(r'[áéíóúü]', sample_text.lower()) and 'español' in sample_text.lower():
            return 'es'
        elif re.search(r'[àáâãçéêíóôõú]', sample_text.lower()):
            # Check for Portuguese-specific patterns first
            if re.search(r'[ãõ]', sample_text.lower()):
                return 'pt'
            else:
                return 'fr'
        elif re.search(r'[àáâäæçèéêëìíîïñòóôöøùúûüÿ]', sample_text.lower()):
            return 'fr'
        elif re.search(r'[一-龯]', sample_text):
            return 'zh'
        elif re.search(r'[ひらがなカタカナ]', sample_text):
            return 'ja'
        elif re.search(r'[가-힣]', sample_text):
            return 'ko'
        elif re.search(r'[a-zA-Z]', sample_text):
            return 'en'
        
        return None
    
    def _detect_encoding(self, str_series: pd.Series) -> str:
        """Detect the encoding of text data."""
        # Since we're working with pandas Series, text is already decoded
        # We can infer likely original encoding from character patterns
        sample_text = ' '.join(str_series.head(min(5, len(str_series))).astype(str))
        
        # Check for common encoding indicators
        if any(ord(char) > 127 for char in sample_text):
            # Contains non-ASCII characters
            if any(ord(char) > 255 for char in sample_text):
                return 'utf-8'
            else:
                return 'latin-1'
        
        return 'ascii'
    
    def _contains_special_characters(self, str_series: pd.Series) -> bool:
        """Check if text contains special characters."""
        sample = str_series.head(min(10, len(str_series)))
        special_char_pattern = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`]'
        
        for text in sample:
            if re.search(special_char_pattern, str(text)):
                return True
        
        return False
    
    def _contains_numbers(self, str_series: pd.Series) -> bool:
        """Check if text contains numbers."""
        sample = str_series.head(min(10, len(str_series)))
        
        for text in sample:
            if re.search(r'\d', str(text)):
                return True
        
        return False
    
    def _calculate_readability_score(self, str_series: pd.Series) -> float:
        """Calculate a simple readability score for natural language text."""
        # Simple readability score based on average sentence and word length
        sample_text = ' '.join(str_series.head(min(5, len(str_series))).astype(str))
        
        if len(sample_text) < 20:
            return 50.0  # Neutral score for short text
        
        # Count sentences (approximate)
        sentences = len(re.split(r'[.!?]+', sample_text))
        if sentences == 0:
            sentences = 1
        
        # Count words
        words = len(sample_text.split())
        if words == 0:
            return 50.0
        
        # Count syllables (approximate by counting vowels)
        syllables = len(re.findall(r'[aeiouAEIOU]', sample_text))
        
        # Simple Flesch-like formula (simplified)
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words if words > 0 else 1
        
        # Simplified readability score (0-100, higher = more readable)
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Clamp to 0-100 range
        return max(0.0, min(100.0, score))
    
    def _calculate_text_confidence(self, analysis: TextAnalysis) -> float:
        """Calculate confidence score for text classification."""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for clear patterns
        if analysis.text_type == 'categorical_text' and analysis.unique_ratio < 0.05:
            confidence += 0.2
        elif analysis.text_type == 'natural_language' and analysis.readability_score and analysis.readability_score > 30:
            confidence += 0.15
        elif analysis.text_type == 'structured_text':
            confidence += 0.15
        
        # Adjust based on sample characteristics
        if analysis.avg_length > 50:
            confidence += 0.05
        
        if analysis.language:
            confidence += 0.05
        
        return min(0.95, confidence)