"""
Test data generators for consistent and reproducible testing.
Provides utilities to generate synthetic datasets for testing.
"""

import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image
from pathlib import Path
import json
import shutil


class TestDataGenerator:
    """Generate synthetic test datasets for various ML tasks."""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_classification_data(self, n_samples=1000, n_features=10, n_classes=3, 
                                   noise=0.1, save_path=None):
        """Generate synthetic classification dataset."""
        np.random.seed(self.seed)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Create some structure in the data
        weights = np.random.randn(n_features, n_classes)
        logits = X @ weights + noise * np.random.randn(n_samples, n_classes)
        y = np.argmax(logits, axis=1)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = dict(zip(feature_names, X.T))
        data['target'] = y
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_regression_data(self, n_samples=1000, n_features=10, noise=0.1, 
                               save_path=None):
        """Generate synthetic regression dataset."""
        np.random.seed(self.seed)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Create target with linear relationship + noise
        true_weights = np.random.randn(n_features)
        y = X @ true_weights + noise * np.random.randn(n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = dict(zip(feature_names, X.T))
        data['target'] = y
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_text_classification_data(self, n_samples=500, save_path=None):
        """Generate synthetic text classification dataset."""
        np.random.seed(self.seed)
        
        # Positive sentiment words
        positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'awesome',
            'brilliant', 'outstanding', 'superb', 'magnificent', 'perfect',
            'love', 'best', 'incredible', 'remarkable', 'exceptional'
        ]
        
        # Negative sentiment words
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate',
            'disappointing', 'poor', 'useless', 'pathetic', 'disgusting',
            'annoying', 'frustrating', 'boring', 'waste', 'regret'
        ]
        
        # Neutral words
        neutral_words = [
            'okay', 'average', 'normal', 'standard', 'typical', 'regular',
            'common', 'usual', 'ordinary', 'moderate', 'fair', 'decent'
        ]
        
        texts = []
        labels = []
        
        for _ in range(n_samples):
            # Choose sentiment
            sentiment = np.random.choice([0, 1, 2])  # 0: negative, 1: neutral, 2: positive
            
            if sentiment == 0:
                words = np.random.choice(negative_words, size=np.random.randint(3, 8))
            elif sentiment == 1:
                words = np.random.choice(neutral_words, size=np.random.randint(3, 8))
            else:
                words = np.random.choice(positive_words, size=np.random.randint(3, 8))
            
            # Add some random common words
            common_words = ['the', 'is', 'was', 'and', 'but', 'very', 'really', 'quite']
            words = np.concatenate([words, np.random.choice(common_words, size=2)])
            
            text = ' '.join(words)
            texts.append(text)
            labels.append(sentiment)
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_image_classification_data(self, n_classes=3, n_images_per_class=20, 
                                         image_size=(64, 64), save_dir=None):
        """Generate synthetic image classification dataset."""
        np.random.seed(self.seed)
        
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        
        class_names = [f'class_{i}' for i in range(n_classes)]
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(n_images_per_class):
                # Generate synthetic image with class-specific patterns
                if class_idx == 0:
                    # Red-ish images
                    img_array = np.random.randint(100, 255, (*image_size, 3), dtype=np.uint8)
                    img_array[:, :, 1] = np.random.randint(0, 100, image_size)  # Low green
                    img_array[:, :, 2] = np.random.randint(0, 100, image_size)  # Low blue
                elif class_idx == 1:
                    # Green-ish images
                    img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                    img_array[:, :, 1] = np.random.randint(100, 255, image_size)  # High green
                else:
                    # Blue-ish images
                    img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                    img_array[:, :, 2] = np.random.randint(100, 255, image_size)  # High blue
                
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f'image_{img_idx:03d}.png')
                img.save(img_path)
        
        return save_dir
    
    def generate_time_series_data(self, n_samples=1000, n_features=5, save_path=None):
        """Generate synthetic time series dataset."""
        np.random.seed(self.seed)
        
        # Generate time index
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        
        # Generate features with trends and seasonality
        data = {'date': dates}
        
        for i in range(n_features):
            # Add trend
            trend = np.linspace(0, 10, n_samples) * np.random.randn()
            
            # Add seasonality
            seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
            
            # Add noise
            noise = np.random.randn(n_samples)
            
            data[f'feature_{i}'] = trend + seasonal + noise
        
        # Create target (next day's feature_0)
        data['target'] = np.roll(data['feature_0'], -1)
        data['target'][-1] = data['target'][-2]  # Fill last value
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_multilabel_classification_data(self, n_samples=1000, n_features=20, 
                                              n_labels=5, save_path=None):
        """Generate synthetic multilabel classification dataset."""
        np.random.seed(self.seed)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate correlated binary labels
        label_weights = np.random.randn(n_features, n_labels)
        label_probs = 1 / (1 + np.exp(-(X @ label_weights)))  # Sigmoid
        Y = (label_probs > 0.5).astype(int)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        label_names = [f'label_{i}' for i in range(n_labels)]
        
        data = dict(zip(feature_names, X.T))
        data.update(dict(zip(label_names, Y.T)))
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_imbalanced_classification_data(self, n_samples=1000, n_features=10, 
                                              imbalance_ratio=0.1, save_path=None):
        """Generate imbalanced classification dataset."""
        np.random.seed(self.seed)
        
        # Calculate samples per class
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority
        
        # Generate majority class
        X_majority = np.random.randn(n_majority, n_features)
        y_majority = np.zeros(n_majority)
        
        # Generate minority class (shifted distribution)
        X_minority = np.random.randn(n_minority, n_features) + 2
        y_minority = np.ones(n_minority)
        
        # Combine
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = dict(zip(feature_names, X.T))
        data['target'] = y
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_missing_data_dataset(self, n_samples=1000, n_features=10, 
                                    missing_rate=0.2, save_path=None):
        """Generate dataset with missing values."""
        np.random.seed(self.seed)
        
        # Generate base data
        df = self.generate_classification_data(n_samples, n_features, save_path=None)
        
        # Introduce missing values
        for col in df.columns[:-1]:  # Don't add missing to target
            missing_mask = np.random.random(len(df)) < missing_rate
            df.loc[missing_mask, col] = np.nan
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_mixed_types_dataset(self, n_samples=1000, save_path=None):
        """Generate dataset with mixed data types."""
        np.random.seed(self.seed)
        
        data = {
            # Numerical features
            'numeric_int': np.random.randint(0, 100, n_samples),
            'numeric_float': np.random.randn(n_samples),
            
            # Categorical features
            'category_low_card': np.random.choice(['A', 'B', 'C'], n_samples),
            'category_high_card': np.random.choice([f'cat_{i}' for i in range(50)], n_samples),
            
            # Boolean feature
            'boolean': np.random.choice([True, False], n_samples),
            
            # Date feature
            'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            
            # Text feature
            'text': [f'text_sample_{i}' for i in range(n_samples)],
            
            # Target
            'target': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_audio_classification_data(self, n_classes=3, n_samples_per_class=10, 
                                         sample_rate=16000, duration=2.0, save_dir=None):
        """Generate synthetic audio classification dataset."""
        np.random.seed(self.seed)
        
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        
        class_names = [f'audio_class_{i}' for i in range(n_classes)]
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for sample_idx in range(n_samples_per_class):
                # Generate synthetic audio with class-specific characteristics
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                if class_idx == 0:
                    # Low frequency sine wave
                    audio = np.sin(2 * np.pi * 440 * t)  # A4 note
                elif class_idx == 1:
                    # Higher frequency sine wave
                    audio = np.sin(2 * np.pi * 880 * t)  # A5 note
                else:
                    # White noise
                    audio = np.random.normal(0, 0.1, len(t))
                
                # Add some noise
                audio += np.random.normal(0, 0.05, len(audio))
                
                # Normalize
                audio = audio / np.max(np.abs(audio))
                
                # Save as WAV file (simplified - just save as numpy array)
                audio_path = os.path.join(class_dir, f'audio_{sample_idx:03d}.npy')
                np.save(audio_path, audio.astype(np.float32))
        
        return save_dir
    
    def generate_object_detection_data(self, n_images=50, image_size=(256, 256), 
                                     max_objects=3, save_dir=None):
        """Generate synthetic object detection dataset."""
        np.random.seed(self.seed)
        
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        
        images_dir = os.path.join(save_dir, 'images')
        annotations_dir = os.path.join(save_dir, 'annotations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        object_classes = ['circle', 'square', 'triangle']
        
        for img_idx in range(n_images):
            # Create blank image
            img_array = np.random.randint(200, 255, (*image_size, 3), dtype=np.uint8)
            
            # Generate random number of objects
            n_objects = np.random.randint(1, max_objects + 1)
            annotations = []
            
            for obj_idx in range(n_objects):
                # Random object class
                obj_class = np.random.choice(object_classes)
                class_id = object_classes.index(obj_class)
                
                # Random position and size
                obj_size = np.random.randint(20, 60)
                x_center = np.random.randint(obj_size, image_size[1] - obj_size)
                y_center = np.random.randint(obj_size, image_size[0] - obj_size)
                
                # Draw object
                if obj_class == 'circle':
                    y, x = np.ogrid[:image_size[0], :image_size[1]]
                    mask = (x - x_center)**2 + (y - y_center)**2 <= (obj_size//2)**2
                    img_array[mask] = [255, 0, 0]  # Red
                elif obj_class == 'square':
                    x1, x2 = x_center - obj_size//2, x_center + obj_size//2
                    y1, y2 = y_center - obj_size//2, y_center + obj_size//2
                    img_array[y1:y2, x1:x2] = [0, 255, 0]  # Green
                else:  # triangle
                    # Simplified triangle
                    for i in range(obj_size//2):
                        start = x_center - i
                        end = x_center + i
                        y_pos = y_center - obj_size//2 + i
                        if 0 <= y_pos < image_size[0]:
                            x_start = max(0, start)
                            x_end = min(image_size[1], end)
                            img_array[y_pos, x_start:x_end] = [0, 0, 255]  # Blue
                
                # Create bounding box annotation
                bbox = {
                    'class_id': class_id,
                    'class_name': obj_class,
                    'x_center': x_center / image_size[1],  # Normalized
                    'y_center': y_center / image_size[0],  # Normalized
                    'width': obj_size / image_size[1],     # Normalized
                    'height': obj_size / image_size[0]     # Normalized
                }
                annotations.append(bbox)
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = os.path.join(images_dir, f'image_{img_idx:03d}.png')
            img.save(img_path)
            
            # Save annotations
            ann_path = os.path.join(annotations_dir, f'image_{img_idx:03d}.json')
            with open(ann_path, 'w') as f:
                json.dump(annotations, f, indent=2)
        
        return save_dir
    
    def generate_sequence_to_sequence_data(self, n_samples=1000, save_path=None):
        """Generate synthetic sequence-to-sequence dataset (e.g., translation)."""
        np.random.seed(self.seed)
        
        # Simple vocabulary
        vocab_source = ['hello', 'world', 'good', 'morning', 'evening', 'how', 'are', 'you', 'fine', 'thanks']
        vocab_target = ['hola', 'mundo', 'bueno', 'mañana', 'tarde', 'como', 'estas', 'tu', 'bien', 'gracias']
        
        # Create mapping
        translation_map = dict(zip(vocab_source, vocab_target))
        
        source_sequences = []
        target_sequences = []
        
        for _ in range(n_samples):
            # Generate source sequence
            seq_length = np.random.randint(2, 6)
            source_words = np.random.choice(vocab_source, size=seq_length, replace=True)
            source_seq = ' '.join(source_words)
            
            # Generate target sequence (translate)
            target_words = [translation_map[word] for word in source_words]
            target_seq = ' '.join(target_words)
            
            source_sequences.append(source_seq)
            target_sequences.append(target_seq)
        
        df = pd.DataFrame({
            'source': source_sequences,
            'target': target_sequences
        })
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_anomaly_detection_data(self, n_normal=900, n_anomaly=100, 
                                      n_features=10, save_path=None):
        """Generate synthetic anomaly detection dataset."""
        np.random.seed(self.seed)
        
        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_normal
        )
        
        # Generate anomalous data (shifted and scaled)
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,  # Shifted mean
            cov=np.eye(n_features) * 4,    # Larger variance
            size=n_anomaly
        )
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = dict(zip(feature_names, X.T))
        data['is_anomaly'] = y
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_clustering_data(self, n_clusters=3, n_samples_per_cluster=100, 
                               n_features=2, cluster_std=1.0, save_path=None):
        """Generate synthetic clustering dataset."""
        np.random.seed(self.seed)
        
        # Generate cluster centers
        cluster_centers = np.random.uniform(-10, 10, (n_clusters, n_features))
        
        X = []
        y = []
        
        for cluster_id, center in enumerate(cluster_centers):
            # Generate samples around cluster center
            cluster_samples = np.random.multivariate_normal(
                mean=center,
                cov=np.eye(n_features) * cluster_std**2,
                size=n_samples_per_cluster
            )
            
            X.append(cluster_samples)
            y.extend([cluster_id] * n_samples_per_cluster)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = dict(zip(feature_names, X.T))
        data['cluster'] = y
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_recommendation_data(self, n_users=100, n_items=50, n_ratings=1000, 
                                   save_path=None):
        """Generate synthetic recommendation dataset."""
        np.random.seed(self.seed)
        
        ratings = []
        
        for _ in range(n_ratings):
            user_id = np.random.randint(0, n_users)
            item_id = np.random.randint(0, n_items)
            
            # Generate rating with some user/item bias
            user_bias = np.random.normal(0, 0.5)
            item_bias = np.random.normal(0, 0.5)
            rating = 3 + user_bias + item_bias + np.random.normal(0, 0.5)
            rating = np.clip(rating, 1, 5)  # Rating scale 1-5
            
            ratings.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
        
        df = pd.DataFrame(ratings)
        
        if save_path:
            df.to_csv(save_path, index=False)
            return save_path
        
        return df
    
    def generate_graph_data(self, n_nodes=100, n_edges=200, save_dir=None):
        """Generate synthetic graph dataset."""
        np.random.seed(self.seed)
        
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        
        # Generate nodes with features
        node_features = np.random.randn(n_nodes, 5)  # 5 features per node
        node_labels = np.random.randint(0, 3, n_nodes)  # 3 classes
        
        # Generate edges
        edges = []
        for _ in range(n_edges):
            source = np.random.randint(0, n_nodes)
            target = np.random.randint(0, n_nodes)
            if source != target:  # No self-loops
                edges.append([source, target])
        
        # Remove duplicates
        edges = list(set(tuple(edge) for edge in edges))
        
        # Save nodes
        nodes_df = pd.DataFrame(node_features, columns=[f'feature_{i}' for i in range(5)])
        nodes_df['label'] = node_labels
        nodes_df['node_id'] = range(n_nodes)
        nodes_path = os.path.join(save_dir, 'nodes.csv')
        nodes_df.to_csv(nodes_path, index=False)
        
        # Save edges
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        edges_path = os.path.join(save_dir, 'edges.csv')
        edges_df.to_csv(edges_path, index=False)
        
        return save_dir
    
    def create_test_suite_datasets(self, output_dir):
        """Create a complete suite of test datasets."""
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {}
        
        # Basic datasets
        datasets['classification'] = self.generate_classification_data(
            save_path=os.path.join(output_dir, 'classification.csv')
        )
        
        datasets['regression'] = self.generate_regression_data(
            save_path=os.path.join(output_dir, 'regression.csv')
        )
        
        datasets['text_classification'] = self.generate_text_classification_data(
            save_path=os.path.join(output_dir, 'text_classification.csv')
        )
        
        datasets['image_classification'] = self.generate_image_classification_data(
            save_dir=os.path.join(output_dir, 'image_classification')
        )
        
        # Advanced datasets
        datasets['time_series'] = self.generate_time_series_data(
            save_path=os.path.join(output_dir, 'time_series.csv')
        )
        
        datasets['multilabel'] = self.generate_multilabel_classification_data(
            save_path=os.path.join(output_dir, 'multilabel.csv')
        )
        
        datasets['imbalanced'] = self.generate_imbalanced_classification_data(
            save_path=os.path.join(output_dir, 'imbalanced.csv')
        )
        
        datasets['missing_data'] = self.generate_missing_data_dataset(
            save_path=os.path.join(output_dir, 'missing_data.csv')
        )
        
        datasets['mixed_types'] = self.generate_mixed_types_dataset(
            save_path=os.path.join(output_dir, 'mixed_types.csv')
        )
        
        # New comprehensive datasets
        datasets['audio_classification'] = self.generate_audio_classification_data(
            save_dir=os.path.join(output_dir, 'audio_classification')
        )
        
        datasets['object_detection'] = self.generate_object_detection_data(
            save_dir=os.path.join(output_dir, 'object_detection')
        )
        
        datasets['sequence_to_sequence'] = self.generate_sequence_to_sequence_data(
            save_path=os.path.join(output_dir, 'sequence_to_sequence.csv')
        )
        
        datasets['anomaly_detection'] = self.generate_anomaly_detection_data(
            save_path=os.path.join(output_dir, 'anomaly_detection.csv')
        )
        
        datasets['clustering'] = self.generate_clustering_data(
            save_path=os.path.join(output_dir, 'clustering.csv')
        )
        
        datasets['recommendation'] = self.generate_recommendation_data(
            save_path=os.path.join(output_dir, 'recommendation.csv')
        )
        
        datasets['graph_data'] = self.generate_graph_data(
            save_dir=os.path.join(output_dir, 'graph_data')
        )
        
        # Performance test datasets
        datasets['large_classification'] = self.generate_classification_data(
            n_samples=10000, n_features=50,
            save_path=os.path.join(output_dir, 'large_classification.csv')
        )
        
        datasets['wide_dataset'] = self.generate_classification_data(
            n_samples=1000, n_features=200,
            save_path=os.path.join(output_dir, 'wide_dataset.csv')
        )
        
        # Edge case datasets
        datasets['single_class'] = self.generate_classification_data(
            n_samples=100, n_classes=1,
            save_path=os.path.join(output_dir, 'single_class.csv')
        )
        
        datasets['high_dimensional'] = self.generate_classification_data(
            n_samples=500, n_features=100,
            save_path=os.path.join(output_dir, 'high_dimensional.csv')
        )
        
        # Save dataset metadata
        metadata = {
            'datasets': datasets,
            'generator_seed': self.seed,
            'created_at': pd.Timestamp.now().isoformat(),
            'dataset_descriptions': {
                'classification': 'Basic multi-class classification dataset',
                'regression': 'Basic regression dataset',
                'text_classification': 'Text sentiment classification dataset',
                'image_classification': 'Synthetic image classification dataset',
                'time_series': 'Time series forecasting dataset',
                'multilabel': 'Multi-label classification dataset',
                'imbalanced': 'Imbalanced binary classification dataset',
                'missing_data': 'Dataset with missing values',
                'mixed_types': 'Dataset with mixed data types',
                'audio_classification': 'Synthetic audio classification dataset',
                'object_detection': 'Synthetic object detection dataset',
                'sequence_to_sequence': 'Sequence-to-sequence translation dataset',
                'anomaly_detection': 'Anomaly detection dataset',
                'clustering': 'Clustering dataset with known clusters',
                'recommendation': 'User-item rating dataset',
                'graph_data': 'Graph dataset with nodes and edges',
                'large_classification': 'Large classification dataset for performance testing',
                'wide_dataset': 'High-dimensional dataset for performance testing',
                'single_class': 'Edge case: single class dataset',
                'high_dimensional': 'High-dimensional dataset'
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return datasets


def pytest_configure(config):
    """Configure pytest with test data generation."""
    # Generate test datasets if they don't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    
    if not os.path.exists(test_data_dir):
        print("Generating test datasets...")
        generator = TestDataGenerator()
        generator.create_test_suite_datasets(test_data_dir)
        print(f"Test datasets created in {test_data_dir}")


if __name__ == "__main__":
    # Generate test datasets when run directly
    output_dir = "test_data"
    generator = TestDataGenerator()
    datasets = generator.create_test_suite_datasets(output_dir)
    
    print("Generated test datasets:")
    for name, path in datasets.items():
        print(f"  {name}: {path}")