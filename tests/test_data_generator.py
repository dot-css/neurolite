"""
Test data generator for creating various data types for integration testing.

This module provides utilities to generate realistic test datasets
for different data types and scenarios.
"""

import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings


class TestDataGenerator:
    """Generate various types of test data for integration testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible data generation."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_tabular_dataset(self, 
                               n_rows: int = 1000,
                               n_numerical: int = 5,
                               n_categorical: int = 3,
                               n_text: int = 2,
                               missing_rate: float = 0.05,
                               include_target: bool = True) -> pd.DataFrame:
        """
        Generate a realistic tabular dataset with mixed data types.
        
        Args:
            n_rows: Number of rows to generate
            n_numerical: Number of numerical columns
            n_categorical: Number of categorical columns
            n_text: Number of text columns
            missing_rate: Proportion of missing values to introduce
            include_target: Whether to include a target variable
            
        Returns:
            Generated DataFrame
        """
        data = {}
        
        # Generate numerical columns
        for i in range(n_numerical):
            if i == 0:
                # ID column (integer)
                data[f'id'] = range(n_rows)
            elif i == 1:
                # Age-like column
                data[f'age'] = np.random.randint(18, 80, n_rows)
            elif i == 2:
                # Income-like column (normal distribution)
                data[f'income'] = np.random.normal(50000, 15000, n_rows)
            elif i == 3:
                # Score-like column (bounded)
                data[f'score'] = np.random.uniform(0, 100, n_rows)
            else:
                # Generic numerical column
                data[f'num_{i}'] = np.random.normal(i * 10, 5, n_rows)
        
        # Generate categorical columns
        for i in range(n_categorical):
            if i == 0:
                # Low cardinality categorical
                data[f'category'] = np.random.choice(['A', 'B', 'C'], n_rows, p=[0.5, 0.3, 0.2])
            elif i == 1:
                # Medium cardinality categorical
                categories = [f'type_{j}' for j in range(10)]
                data[f'type'] = np.random.choice(categories, n_rows)
            else:
                # High cardinality categorical
                cardinality = min(50, n_rows // 20)  # Reasonable cardinality
                categories = [f'cat_{i}_{j}' for j in range(cardinality)]
                data[f'cat_{i}'] = np.random.choice(categories, n_rows)
        
        # Generate text columns
        for i in range(n_text):
            if i == 0:
                # Short text (names, codes)
                data[f'name'] = [f'Item_{j:04d}' for j in range(n_rows)]
            else:
                # Longer text descriptions
                descriptions = [
                    f'This is a description for item {j} with some additional details and information.'
                    for j in range(n_rows)
                ]
                data[f'description_{i}'] = descriptions
        
        # Add temporal column
        data['created_date'] = pd.date_range('2020-01-01', periods=n_rows, freq='D')[:n_rows]
        
        # Add boolean column
        data['is_active'] = np.random.choice([True, False], n_rows, p=[0.7, 0.3])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add target variable if requested
        if include_target:
            # Create target based on some features for realistic relationships
            if 'income' in df.columns and 'age' in df.columns:
                # Create target with some relationship to features
                target_prob = (df['income'] / 100000 + df['age'] / 100) / 2
                target_prob = np.clip(target_prob, 0.1, 0.9)
                df['target'] = np.random.binomial(1, target_prob)
            else:
                # Random target
                df['target'] = np.random.choice([0, 1], n_rows, p=[0.6, 0.4])
        
        # Introduce missing values
        if missing_rate > 0:
            self._introduce_missing_values(df, missing_rate)
        
        return df
    
    def generate_time_series_dataset(self,
                                   n_points: int = 2000,
                                   freq: str = 'H',
                                   n_series: int = 3,
                                   trend: bool = True,
                                   seasonality: bool = True,
                                   noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate time series dataset with configurable characteristics.
        
        Args:
            n_points: Number of time points
            freq: Frequency string (e.g., 'H', 'D', 'M')
            n_series: Number of time series
            trend: Whether to include trend
            seasonality: Whether to include seasonality
            noise_level: Level of random noise
            
        Returns:
            Generated time series DataFrame
        """
        # Generate time index
        dates = pd.date_range('2020-01-01', periods=n_points, freq=freq)
        
        data = {'timestamp': dates}
        
        for i in range(n_series):
            # Base value
            base_value = 100 + i * 50
            values = np.full(n_points, base_value)
            
            # Add trend
            if trend:
                trend_component = np.linspace(0, base_value * 0.5, n_points)
                values += trend_component
            
            # Add seasonality
            if seasonality:
                if freq == 'H':
                    # Daily seasonality for hourly data
                    seasonal_component = 20 * np.sin(2 * np.pi * np.arange(n_points) / 24)
                elif freq == 'D':
                    # Weekly seasonality for daily data
                    seasonal_component = 15 * np.sin(2 * np.pi * np.arange(n_points) / 7)
                else:
                    # Generic seasonality
                    seasonal_component = 10 * np.sin(2 * np.pi * np.arange(n_points) / 12)
                
                values += seasonal_component
            
            # Add noise
            noise = np.random.normal(0, base_value * noise_level, n_points)
            values += noise
            
            data[f'series_{i}'] = values
        
        # Add categorical features
        data['day_of_week'] = dates.dayofweek
        data['month'] = dates.month
        data['is_weekend'] = dates.weekday >= 5
        
        return pd.DataFrame(data)
    
    def generate_text_dataset(self,
                            n_documents: int = 1000,
                            n_topics: int = 4,
                            avg_doc_length: int = 100,
                            vocabulary_size: int = 500) -> pd.DataFrame:
        """
        Generate text dataset for NLP tasks.
        
        Args:
            n_documents: Number of documents to generate
            n_topics: Number of topics/classes
            avg_doc_length: Average document length in words
            vocabulary_size: Size of vocabulary
            
        Returns:
            Generated text DataFrame
        """
        # Define topic-specific vocabularies
        topic_vocabularies = {
            'technology': ['computer', 'software', 'algorithm', 'data', 'programming', 'AI', 
                          'machine', 'learning', 'neural', 'network', 'code', 'system'],
            'sports': ['game', 'player', 'team', 'score', 'match', 'championship', 
                      'victory', 'defeat', 'training', 'competition', 'athlete', 'coach'],
            'politics': ['government', 'policy', 'election', 'candidate', 'vote', 
                        'democracy', 'law', 'citizen', 'parliament', 'minister', 'debate', 'reform'],
            'entertainment': ['movie', 'actor', 'music', 'concert', 'show', 'performance', 
                             'artist', 'entertainment', 'film', 'theater', 'celebrity', 'award']
        }
        
        # Generate common vocabulary
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                       'with', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 
                       'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under']
        
        topics = list(topic_vocabularies.keys())[:n_topics]
        documents = []
        labels = []
        
        for i in range(n_documents):
            # Choose topic
            topic = np.random.choice(topics)
            labels.append(topic)
            
            # Generate document
            doc_length = max(10, int(np.random.normal(avg_doc_length, avg_doc_length * 0.3)))
            
            # Mix topic-specific and common words
            topic_words = topic_vocabularies[topic]
            
            words = []
            for _ in range(doc_length):
                if np.random.random() < 0.3:  # 30% topic-specific words
                    words.append(np.random.choice(topic_words))
                else:  # 70% common words
                    words.append(np.random.choice(common_words))
            
            documents.append(' '.join(words))
        
        # Create DataFrame
        df = pd.DataFrame({
            'document_id': range(n_documents),
            'text': documents,
            'topic': labels,
            'word_count': [len(doc.split()) for doc in documents],
            'char_count': [len(doc) for doc in documents],
            'avg_word_length': [np.mean([len(word) for word in doc.split()]) for doc in documents]
        })
        
        return df
    
    def generate_image_metadata_dataset(self,
                                      n_images: int = 1000,
                                      n_classes: int = 5) -> pd.DataFrame:
        """
        Generate image metadata dataset (simulating computer vision data).
        
        Args:
            n_images: Number of image records
            n_classes: Number of image classes
            
        Returns:
            Generated image metadata DataFrame
        """
        # Define image classes
        classes = [f'class_{i}' for i in range(n_classes)]
        
        data = {
            'image_id': [f'img_{i:06d}' for i in range(n_images)],
            'filename': [f'image_{i:06d}.jpg' for i in range(n_images)],
            'width': np.random.randint(224, 2048, n_images),
            'height': np.random.randint(224, 2048, n_images),
            'channels': np.random.choice([1, 3, 4], n_images, p=[0.1, 0.8, 0.1]),
            'file_size_kb': np.random.exponential(500, n_images),
            'format': np.random.choice(['JPEG', 'PNG', 'TIFF'], n_images, p=[0.7, 0.2, 0.1]),
            'color_depth': np.random.choice([8, 16, 24, 32], n_images, p=[0.1, 0.2, 0.6, 0.1]),
            'has_alpha': np.random.choice([True, False], n_images, p=[0.2, 0.8]),
            'compression_ratio': np.random.uniform(0.1, 0.9, n_images),
            'brightness': np.random.normal(128, 30, n_images),
            'contrast': np.random.normal(1.0, 0.3, n_images),
            'label': np.random.choice(classes, n_images)
        }
        
        # Ensure positive values
        data['file_size_kb'] = np.abs(data['file_size_kb'])
        data['brightness'] = np.clip(data['brightness'], 0, 255)
        data['contrast'] = np.clip(data['contrast'], 0.1, 3.0)
        
        return pd.DataFrame(data)
    
    def generate_audio_metadata_dataset(self,
                                      n_files: int = 500,
                                      n_genres: int = 6) -> pd.DataFrame:
        """
        Generate audio metadata dataset.
        
        Args:
            n_files: Number of audio files
            n_genres: Number of music genres
            
        Returns:
            Generated audio metadata DataFrame
        """
        genres = ['rock', 'jazz', 'classical', 'electronic', 'pop', 'country'][:n_genres]
        
        data = {
            'file_id': [f'audio_{i:05d}' for i in range(n_files)],
            'filename': [f'audio_{i:05d}.wav' for i in range(n_files)],
            'duration_seconds': np.random.exponential(180, n_files),  # Average 3 minutes
            'sample_rate': np.random.choice([22050, 44100, 48000], n_files, p=[0.2, 0.6, 0.2]),
            'bit_depth': np.random.choice([16, 24, 32], n_files, p=[0.6, 0.3, 0.1]),
            'channels': np.random.choice([1, 2], n_files, p=[0.3, 0.7]),
            'file_size_mb': np.random.exponential(5, n_files),
            'genre': np.random.choice(genres, n_files),
            'tempo_bpm': np.random.normal(120, 30, n_files),
            'loudness_db': np.random.normal(-15, 5, n_files),
            'dynamic_range': np.random.uniform(5, 25, n_files),
            'has_vocals': np.random.choice([True, False], n_files, p=[0.7, 0.3]),
            'energy': np.random.uniform(0, 1, n_files),
            'valence': np.random.uniform(0, 1, n_files)  # Musical positivity
        }
        
        # Ensure positive values where appropriate
        data['duration_seconds'] = np.abs(data['duration_seconds'])
        data['file_size_mb'] = np.abs(data['file_size_mb'])
        data['tempo_bpm'] = np.abs(data['tempo_bpm'])
        
        return pd.DataFrame(data)
    
    def generate_mixed_quality_dataset(self,
                                     n_rows: int = 1000,
                                     missing_patterns: List[str] = None) -> pd.DataFrame:
        """
        Generate dataset with various data quality issues for testing quality detection.
        
        Args:
            n_rows: Number of rows
            missing_patterns: Types of missing patterns to introduce
            
        Returns:
            DataFrame with quality issues
        """
        if missing_patterns is None:
            missing_patterns = ['MCAR', 'MAR', 'MNAR']
        
        # Generate base dataset
        df = self.generate_tabular_dataset(n_rows, include_target=True)
        
        # Introduce different types of missing patterns
        if 'MCAR' in missing_patterns:
            # Missing Completely At Random - random 5% missing in income
            mcar_indices = np.random.choice(df.index, int(0.05 * n_rows), replace=False)
            df.loc[mcar_indices, 'income'] = np.nan
        
        if 'MAR' in missing_patterns:
            # Missing At Random - missing category more likely for older people
            mar_condition = df['age'] > 60
            mar_indices = df[mar_condition].sample(frac=0.3).index
            df.loc[mar_indices, 'category'] = np.nan
        
        if 'MNAR' in missing_patterns:
            # Missing Not At Random - high income values more likely to be missing
            mnar_condition = df['income'] > df['income'].quantile(0.9)
            mnar_indices = df[mnar_condition].sample(frac=0.4).index
            df.loc[mnar_indices, 'income'] = np.nan
        
        # Add duplicate records
        duplicate_indices = np.random.choice(df.index, int(0.02 * n_rows), replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Add inconsistent formatting
        if 'name' in df.columns:
            # Mix case inconsistencies
            inconsistent_indices = np.random.choice(df.index, int(0.1 * len(df)), replace=False)
            df.loc[inconsistent_indices, 'name'] = df.loc[inconsistent_indices, 'name'].str.lower()
        
        # Add outliers
        if 'score' in df.columns:
            outlier_indices = np.random.choice(df.index, int(0.02 * len(df)), replace=False)
            df.loc[outlier_indices, 'score'] = np.random.uniform(200, 1000, len(outlier_indices))
        
        return df
    
    def _introduce_missing_values(self, df: pd.DataFrame, missing_rate: float):
        """Introduce missing values randomly across the DataFrame."""
        for col in df.columns:
            if col == 'id':  # Don't add missing values to ID columns
                continue
            
            n_missing = int(len(df) * missing_rate)
            if n_missing > 0:
                missing_indices = np.random.choice(df.index, n_missing, replace=False)
                df.loc[missing_indices, col] = np.nan
    
    def save_datasets_to_files(self, output_dir: str = None) -> Dict[str, str]:
        """
        Generate and save various test datasets to files.
        
        Args:
            output_dir: Directory to save files (uses temp dir if None)
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_paths = {}
        
        # Generate and save different datasets
        datasets = {
            'tabular_clean': self.generate_tabular_dataset(1000, missing_rate=0.0),
            'tabular_missing': self.generate_tabular_dataset(1000, missing_rate=0.1),
            'time_series': self.generate_time_series_dataset(2000),
            'text_data': self.generate_text_dataset(500),
            'image_metadata': self.generate_image_metadata_dataset(800),
            'audio_metadata': self.generate_audio_metadata_dataset(400),
            'quality_issues': self.generate_mixed_quality_dataset(1000)
        }
        
        for name, df in datasets.items():
            # Save as CSV
            csv_path = output_path / f'{name}.csv'
            df.to_csv(csv_path, index=False)
            file_paths[f'{name}_csv'] = str(csv_path)
            
            # Save as JSON (for some datasets)
            if name in ['text_data', 'image_metadata', 'audio_metadata']:
                json_path = output_path / f'{name}.json'
                df.to_json(json_path, orient='records', indent=2)
                file_paths[f'{name}_json'] = str(json_path)
        
        return file_paths


if __name__ == '__main__':
    # Example usage
    generator = TestDataGenerator()
    
    # Generate sample datasets
    tabular_df = generator.generate_tabular_dataset(100)
    print("Tabular dataset shape:", tabular_df.shape)
    print("Columns:", list(tabular_df.columns))
    
    time_series_df = generator.generate_time_series_dataset(200)
    print("\nTime series dataset shape:", time_series_df.shape)
    print("Columns:", list(time_series_df.columns))
    
    text_df = generator.generate_text_dataset(50)
    print("\nText dataset shape:", text_df.shape)
    print("Topics:", text_df['topic'].unique())