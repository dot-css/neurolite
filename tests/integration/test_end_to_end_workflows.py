"""
End-to-end integration tests for NeuroLite workflows.
Tests complete user workflows with real datasets.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import requests
import zipfile
from urllib.parse import urlparse

from neurolite import train
from neurolite.core.exceptions import NeuroLiteError


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_tabular_data(self, temp_dir):
        """Create sample tabular dataset."""
        # Generate synthetic classification data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.randint(0, 5, n_samples),
            'feature4': np.random.uniform(0, 100, n_samples),
            'target': np.random.randint(0, 3, n_samples)
        }
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'tabular_data.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def sample_image_data(self, temp_dir):
        """Create sample image dataset."""
        from PIL import Image
        
        # Create directory structure for image classification
        classes = ['class_a', 'class_b', 'class_c']
        image_dir = os.path.join(temp_dir, 'images')
        
        for class_name in classes:
            class_dir = os.path.join(image_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 10 synthetic images per class
            for i in range(10):
                # Create random RGB image
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f'image_{i}.png')
                img.save(img_path)
        
        return image_dir
    
    @pytest.fixture
    def sample_text_data(self, temp_dir):
        """Create sample text dataset."""
        texts = [
            "This is a positive review. Great product!",
            "Terrible experience. Would not recommend.",
            "Average product. Nothing special.",
            "Excellent quality and fast delivery.",
            "Poor customer service and low quality.",
            "Good value for money. Satisfied.",
            "Worst purchase ever. Complete waste.",
            "Outstanding product. Highly recommended.",
            "Mediocre quality. Could be better.",
            "Perfect! Exactly what I needed."
        ] * 50  # Repeat to have enough samples
        
        sentiments = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 50
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        file_path = os.path.join(temp_dir, 'text_data.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    def test_tabular_classification_workflow(self, sample_tabular_data):
        """Test complete tabular classification workflow."""
        # Train model
        model = train(
            data=sample_tabular_data,
            task="classification",
            target="target",
            validation_split=0.2,
            test_split=0.1
        )
        
        # Verify model was trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'evaluate')
        
        # Test prediction
        test_data = {
            'feature1': 0.5,
            'feature2': -0.3,
            'feature3': 2,
            'feature4': 45.0
        }
        
        prediction = model.predict(test_data)
        assert prediction is not None
        assert isinstance(prediction, (int, np.integer, float, np.floating))
        
        # Test batch prediction
        test_df = pd.DataFrame([test_data] * 5)
        batch_predictions = model.predict(test_df)
        assert len(batch_predictions) == 5
    
    def test_tabular_regression_workflow(self, temp_dir):
        """Test complete tabular regression workflow."""
        # Create regression dataset
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples),
            'x3': np.random.uniform(0, 10, n_samples),
        }
        
        # Create target with some relationship to features
        data['y'] = (2 * data['x1'] + 3 * data['x2'] + 
                    0.5 * data['x3'] + np.random.normal(0, 0.1, n_samples))
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'regression_data.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="regression",
            target="y",
            validation_split=0.2
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_data = {'x1': 1.0, 'x2': -0.5, 'x3': 5.0}
        prediction = model.predict(test_data)
        assert isinstance(prediction, (float, np.floating))
    
    def test_image_classification_workflow(self, sample_image_data):
        """Test complete image classification workflow."""
        # Train model with minimal epochs for speed
        model = train(
            data=sample_image_data,
            task="image_classification",
            validation_split=0.3,
            config={'epochs': 2, 'batch_size': 4}  # Minimal for testing
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction on single image
        test_image_path = os.path.join(sample_image_data, 'class_a', 'image_0.png')
        prediction = model.predict(test_image_path)
        assert prediction is not None
    
    def test_text_classification_workflow(self, sample_text_data):
        """Test complete text classification workflow."""
        # Train model
        model = train(
            data=sample_text_data,
            task="text_classification",
            target="sentiment",
            validation_split=0.2,
            config={'epochs': 1}  # Minimal for testing
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_text = "This is a great product!"
        prediction = model.predict(test_text)
        assert prediction is not None
        assert isinstance(prediction, (int, np.integer))
    
    def test_auto_task_detection(self, sample_tabular_data):
        """Test automatic task type detection."""
        model = train(
            data=sample_tabular_data,
            target="target"  # Let it auto-detect task type
        )
        
        assert model is not None
        # Should detect classification based on target values
    
    def test_hyperparameter_optimization_workflow(self, sample_tabular_data):
        """Test workflow with hyperparameter optimization."""
        model = train(
            data=sample_tabular_data,
            task="classification",
            target="target",
            optimize=True,
            config={'optimization_trials': 3}  # Minimal for testing
        )
        
        assert model is not None
        assert hasattr(model, 'best_params_')
    
    def test_model_export_workflow(self, sample_tabular_data):
        """Test model training and export workflow."""
        model = train(
            data=sample_tabular_data,
            task="classification",
            target="target"
        )
        
        # Test model export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, 'model.pkl')
            model.export(export_path)
            assert os.path.exists(export_path)
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        with pytest.raises(NeuroLiteError):
            train(data="nonexistent_file.csv", task="classification")
    
    def test_error_handling_invalid_task(self, sample_tabular_data):
        """Test error handling with invalid task type."""
        with pytest.raises(NeuroLiteError):
            train(
                data=sample_tabular_data,
                task="invalid_task_type",
                target="target"
            )
    
    def test_minimal_code_interface(self, sample_tabular_data):
        """Test the minimal code interface requirement."""
        # This should work in just 2 lines
        model = train(data=sample_tabular_data, target="target")
        prediction = model.predict({'feature1': 0.5, 'feature2': -0.3, 
                                  'feature3': 2, 'feature4': 45.0})
        
        assert model is not None
        assert prediction is not None


class TestRealDatasetWorkflows:
    """Test workflows with real-world datasets."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def download_dataset(self, url, extract_path):
        """Download and extract dataset if not already present."""
        if os.path.exists(extract_path):
            return extract_path
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create extract directory
            os.makedirs(extract_path, exist_ok=True)
            
            # Save and extract zip file
            zip_path = os.path.join(extract_path, 'dataset.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            os.remove(zip_path)
            return extract_path
            
        except Exception as e:
            pytest.skip(f"Could not download dataset: {e}")
    
    @pytest.mark.slow
    def test_iris_classification_workflow(self, temp_dir):
        """Test with the classic Iris dataset."""
        from sklearn.datasets import load_iris
        
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        file_path = os.path.join(temp_dir, 'iris.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="classification",
            target="target",
            validation_split=0.3
        )
        
        # Verify model performance
        assert model is not None
        
        # Test prediction
        test_sample = {
            'sepal length (cm)': 5.1,
            'sepal width (cm)': 3.5,
            'petal length (cm)': 1.4,
            'petal width (cm)': 0.2
        }
        
        prediction = model.predict(test_sample)
        assert prediction in [0, 1, 2]
        
        # Evaluate model
        evaluation = model.evaluate()
        assert evaluation['accuracy'] > 0.8  # Should achieve good accuracy on Iris
    
    @pytest.mark.slow
    def test_boston_housing_regression_workflow(self, temp_dir):
        """Test with Boston Housing dataset."""
        # Create synthetic housing data (Boston dataset is deprecated)
        np.random.seed(42)
        n_samples = 506
        
        # Generate realistic housing features
        data = {
            'CRIM': np.random.exponential(3, n_samples),  # Crime rate
            'ZN': np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples),  # Zoning
            'INDUS': np.random.uniform(0, 30, n_samples),  # Industrial
            'CHAS': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # River
            'NOX': np.random.uniform(0.3, 0.9, n_samples),  # Nitric oxide
            'RM': np.random.normal(6.3, 0.7, n_samples),  # Rooms
            'AGE': np.random.uniform(0, 100, n_samples),  # Age
            'DIS': np.random.exponential(3, n_samples),  # Distance
            'RAD': np.random.choice(range(1, 25), n_samples),  # Accessibility
            'TAX': np.random.uniform(150, 700, n_samples),  # Tax rate
            'PTRATIO': np.random.uniform(12, 22, n_samples),  # Pupil-teacher ratio
            'B': np.random.uniform(300, 400, n_samples),  # Black proportion
            'LSTAT': np.random.uniform(1, 40, n_samples),  # Lower status
        }
        
        # Create realistic target (median home value)
        target = (
            20 + 
            5 * data['RM'] - 
            0.5 * data['CRIM'] - 
            0.3 * data['LSTAT'] + 
            np.random.normal(0, 3, n_samples)
        )
        target = np.clip(target, 5, 50)  # Realistic price range
        data['MEDV'] = target
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'housing.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="regression",
            target="MEDV",
            validation_split=0.2
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_sample = {
            'CRIM': 0.1, 'ZN': 12.5, 'INDUS': 7.87, 'CHAS': 0,
            'NOX': 0.524, 'RM': 6.012, 'AGE': 66.6, 'DIS': 5.5605,
            'RAD': 5, 'TAX': 311, 'PTRATIO': 15.2, 'B': 395.6, 'LSTAT': 12.43
        }
        
        prediction = model.predict(test_sample)
        assert isinstance(prediction, (float, np.floating))
        assert 5 <= prediction <= 50  # Reasonable price range
        
        # Evaluate model
        evaluation = model.evaluate()
        assert 'mse' in evaluation or 'rmse' in evaluation
    
    @pytest.mark.slow
    def test_wine_quality_workflow(self, temp_dir):
        """Test with wine quality dataset."""
        # Generate synthetic wine quality data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
            'volatile_acidity': np.random.normal(0.5, 0.18, n_samples),
            'citric_acid': np.random.normal(0.27, 0.19, n_samples),
            'residual_sugar': np.random.exponential(2.5, n_samples),
            'chlorides': np.random.normal(0.087, 0.047, n_samples),
            'free_sulfur_dioxide': np.random.normal(15.9, 10.5, n_samples),
            'total_sulfur_dioxide': np.random.normal(46, 32.9, n_samples),
            'density': np.random.normal(0.996, 0.002, n_samples),
            'pH': np.random.normal(3.31, 0.15, n_samples),
            'sulphates': np.random.normal(0.66, 0.17, n_samples),
            'alcohol': np.random.normal(10.4, 1.07, n_samples),
        }
        
        # Create quality score based on features
        quality_score = (
            0.3 * data['alcohol'] +
            0.2 * (10 - data['volatile_acidity']) +
            0.1 * data['citric_acid'] +
            0.1 * (4 - data['pH']) +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convert to discrete quality ratings (3-9)
        data['quality'] = np.clip(np.round(quality_score), 3, 9).astype(int)
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'wine_quality.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="classification",
            target="quality",
            validation_split=0.2
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_sample = {
            'fixed_acidity': 7.4, 'volatile_acidity': 0.7, 'citric_acid': 0.0,
            'residual_sugar': 1.9, 'chlorides': 0.076, 'free_sulfur_dioxide': 11.0,
            'total_sulfur_dioxide': 34.0, 'density': 0.9978, 'pH': 3.51,
            'sulphates': 0.56, 'alcohol': 9.4
        }
        
        prediction = model.predict(test_sample)
        assert 3 <= prediction <= 9
    
    @pytest.mark.slow
    def test_sentiment_analysis_workflow(self, temp_dir):
        """Test with movie review sentiment dataset."""
        # Generate synthetic movie reviews
        np.random.seed(42)
        
        positive_phrases = [
            "excellent movie", "great acting", "wonderful story", "amazing cinematography",
            "brilliant performance", "outstanding film", "loved it", "highly recommend",
            "masterpiece", "incredible", "fantastic", "superb direction"
        ]
        
        negative_phrases = [
            "terrible movie", "poor acting", "boring story", "awful cinematography",
            "bad performance", "disappointing film", "hated it", "waste of time",
            "disaster", "horrible", "worst movie", "terrible direction"
        ]
        
        neutral_phrases = [
            "okay movie", "average acting", "decent story", "standard film",
            "not bad", "watchable", "mediocre", "nothing special"
        ]
        
        reviews = []
        sentiments = []
        
        for _ in range(300):  # 100 per sentiment
            sentiment = np.random.choice([0, 1, 2])  # negative, neutral, positive
            
            if sentiment == 0:
                phrases = np.random.choice(negative_phrases, size=np.random.randint(2, 5))
            elif sentiment == 1:
                phrases = np.random.choice(neutral_phrases, size=np.random.randint(2, 4))
            else:
                phrases = np.random.choice(positive_phrases, size=np.random.randint(2, 5))
            
            review = " ".join(phrases) + "."
            reviews.append(review)
            sentiments.append(sentiment)
        
        df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })
        
        file_path = os.path.join(temp_dir, 'movie_reviews.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="text_classification",
            target="sentiment",
            validation_split=0.2,
            config={'epochs': 2}  # Minimal for testing
        )
        
        # Verify model
        assert model is not None
        
        # Test predictions
        test_reviews = [
            "This movie is absolutely fantastic and amazing!",
            "Terrible film, complete waste of time.",
            "It was okay, nothing special but watchable."
        ]
        
        for review in test_reviews:
            prediction = model.predict(review)
            assert prediction in [0, 1, 2]
    
    @pytest.mark.slow
    def test_time_series_workflow(self, temp_dir):
        """Test with time series data."""
        # Generate synthetic stock price data
        np.random.seed(42)
        n_days = 1000
        
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Generate realistic stock price movement
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        price = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
        
        # Add some technical indicators
        volume = np.random.lognormal(15, 0.5, n_days)
        
        # Simple moving averages
        sma_5 = pd.Series(price).rolling(5).mean()
        sma_20 = pd.Series(price).rolling(20).mean()
        
        df = pd.DataFrame({
            'date': dates,
            'price': price,
            'volume': volume,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'returns': returns
        })
        
        # Create target (next day's price movement)
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        df = df.dropna()
        
        file_path = os.path.join(temp_dir, 'stock_data.csv')
        df.to_csv(file_path, index=False)
        
        # Train model
        model = train(
            data=file_path,
            task="classification",
            target="target",
            validation_split=0.2,
            config={'epochs': 3}
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_sample = {
            'price': 105.5,
            'volume': 1000000,
            'sma_5': 104.2,
            'sma_20': 103.8,
            'returns': 0.01
        }
        
        prediction = model.predict(test_sample)
        assert prediction in [0, 1]
    
    @pytest.mark.slow
    def test_multi_class_image_workflow(self, temp_dir):
        """Test with multi-class image classification."""
        # Create synthetic image dataset with more realistic patterns
        np.random.seed(42)
        
        classes = ['circles', 'squares', 'triangles', 'stars']
        image_dir = os.path.join(temp_dir, 'shapes')
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(image_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_idx in range(15):  # More images per class
                # Create 128x128 RGB image
                img_array = np.random.randint(200, 255, (128, 128, 3), dtype=np.uint8)
                
                # Add class-specific patterns
                center_x, center_y = 64, 64
                
                if class_name == 'circles':
                    # Draw circle pattern
                    y, x = np.ogrid[:128, :128]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
                    img_array[mask] = [255, 0, 0]  # Red circle
                
                elif class_name == 'squares':
                    # Draw square pattern
                    img_array[34:94, 34:94] = [0, 255, 0]  # Green square
                
                elif class_name == 'triangles':
                    # Draw triangle pattern (simplified)
                    for i in range(30):
                        start = center_x - i
                        end = center_x + i
                        y_pos = center_y - 30 + i
                        if 0 <= y_pos < 128:
                            img_array[y_pos, max(0, start):min(128, end)] = [0, 0, 255]  # Blue triangle
                
                else:  # stars
                    # Draw star pattern (simplified as cross)
                    img_array[center_y-20:center_y+20, center_x-3:center_x+3] = [255, 255, 0]  # Yellow cross
                    img_array[center_y-3:center_y+3, center_x-20:center_x+20] = [255, 255, 0]
                
                from PIL import Image
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f'image_{img_idx:03d}.png')
                img.save(img_path)
        
        # Train model
        model = train(
            data=image_dir,
            task="image_classification",
            validation_split=0.3,
            config={'epochs': 2, 'batch_size': 8}  # Minimal for testing
        )
        
        # Verify model
        assert model is not None
        
        # Test prediction
        test_image_path = os.path.join(image_dir, 'circles', 'image_000.png')
        prediction = model.predict(test_image_path)
        assert prediction is not None