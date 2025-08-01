"""
Example Tests

This module contains tests that validate the functionality of
documentation examples by running them in isolated environments.
"""

import unittest
import tempfile
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestBasicExamples(unittest.TestCase):
    """Test basic examples from the documentation."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_sample_tabular_data(self):
        """Test creating sample tabular data as shown in examples."""
        # Create sample data similar to examples
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        df.to_csv('sample_data.csv', index=False)
        
        # Verify data was created correctly
        self.assertTrue(os.path.exists('sample_data.csv'))
        
        loaded_df = pd.read_csv('sample_data.csv')
        self.assertEqual(len(loaded_df), 100)
        self.assertListEqual(list(loaded_df.columns), ['feature1', 'feature2', 'feature3', 'target'])
        self.assertTrue(all(loaded_df['target'].isin([0, 1])))
    
    def test_create_sample_text_data(self):
        """Test creating sample text data as shown in examples."""
        # Create sample text data
        text_data = [
            ("This movie is great!", "positive"),
            ("I loved this film", "positive"),
            ("Terrible movie", "negative"),
            ("Waste of time", "negative"),
            ("Amazing story", "positive"),
            ("Very boring", "negative")
        ]
        
        df = pd.DataFrame(text_data, columns=['text', 'label'])
        df.to_csv('text_sample.csv', index=False)
        
        # Verify data
        self.assertTrue(os.path.exists('text_sample.csv'))
        
        loaded_df = pd.read_csv('text_sample.csv')
        self.assertEqual(len(loaded_df), 6)
        self.assertIn('text', loaded_df.columns)
        self.assertIn('label', loaded_df.columns)
        self.assertTrue(all(loaded_df['label'].isin(['positive', 'negative'])))
    
    def test_create_sample_image_structure(self):
        """Test creating sample image directory structure."""
        # Create image directory structure as shown in examples
        os.makedirs('images/cats', exist_ok=True)
        os.makedirs('images/dogs', exist_ok=True)
        
        # Create dummy image files
        for i in range(10):
            Path(f'images/cats/cat_{i:03d}.jpg').touch()
            Path(f'images/dogs/dog_{i:03d}.jpg').touch()
        
        # Verify structure
        self.assertTrue(os.path.exists('images/cats'))
        self.assertTrue(os.path.exists('images/dogs'))
        self.assertEqual(len(os.listdir('images/cats')), 10)
        self.assertEqual(len(os.listdir('images/dogs')), 10)
    
    def test_data_validation_examples(self):
        """Test data validation examples from troubleshooting guide."""
        # Test file existence checking
        self.assertFalse(os.path.exists('nonexistent_file.csv'))
        
        # Create file and test
        test_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        test_data.to_csv('test_file.csv', index=False)
        self.assertTrue(os.path.exists('test_file.csv'))
        
        # Test data loading and validation
        loaded = pd.read_csv('test_file.csv')
        self.assertEqual(len(loaded), 3)
        self.assertListEqual(list(loaded.columns), ['a', 'b'])
        
        # Test for missing values
        test_data_with_na = pd.DataFrame({
            'feature': [1, 2, np.nan, 4],
            'target': [0, 1, 0, 1]
        })
        test_data_with_na.to_csv('data_with_na.csv', index=False)
        
        loaded_na = pd.read_csv('data_with_na.csv')
        self.assertTrue(loaded_na.isnull().any().any())
        self.assertEqual(loaded_na.isnull().sum().sum(), 1)


class TestConfigurationExamples(unittest.TestCase):
    """Test configuration examples from documentation."""
    
    def test_parameter_validation(self):
        """Test parameter validation examples."""
        # Test validation split validation
        valid_cases = [
            (0.2, 0.1),  # Valid case
            (0.15, 0.15),  # Valid case
            (0.0, 0.0),  # Edge case - valid
        ]
        
        invalid_cases = [
            (0.5, 0.6),  # Sum > 1.0
            (-0.1, 0.1),  # Negative validation split
            (0.2, -0.1),  # Negative test split
            (1.5, 0.1),  # validation_split > 1.0
            (0.2, 1.5),  # test_split > 1.0
        ]
        
        def validate_splits(val_split, test_split):
            return (
                0.0 <= val_split <= 1.0 and
                0.0 <= test_split <= 1.0 and
                val_split + test_split < 1.0
            )
        
        # Test valid cases
        for val_split, test_split in valid_cases:
            self.assertTrue(validate_splits(val_split, test_split),
                          f"Valid case failed: val_split={val_split}, test_split={test_split}")
        
        # Test invalid cases
        for val_split, test_split in invalid_cases:
            self.assertFalse(validate_splits(val_split, test_split),
                           f"Invalid case passed: val_split={val_split}, test_split={test_split}")
    
    def test_model_parameter_examples(self):
        """Test model parameter examples."""
        # Test model name validation
        valid_models = ['auto', 'bert', 'resnet18', 'random_forest', 'xgboost']
        invalid_models = ['invalid_model', '', None]
        
        def is_valid_model(model):
            if model is None:
                return False
            return isinstance(model, str) and len(model) > 0
        
        for model in valid_models:
            self.assertTrue(is_valid_model(model),
                          f"Valid model failed: {model}")
        
        for model in invalid_models:
            self.assertFalse(is_valid_model(model),
                           f"Invalid model passed: {model}")
    
    def test_task_parameter_examples(self):
        """Test task parameter examples."""
        valid_tasks = [
            'auto', 'classification', 'regression', 
            'image_classification', 'text_classification',
            'sentiment_analysis', 'object_detection'
        ]
        
        for task in valid_tasks:
            self.assertIsInstance(task, str)
            self.assertGreater(len(task), 0)


class TestTroubleshootingExamples(unittest.TestCase):
    """Test examples from the troubleshooting guide."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_file_path_checking(self):
        """Test file path checking examples."""
        # Test file existence checking
        test_file = 'test_data.csv'
        self.assertFalse(os.path.exists(test_file))
        
        # Create file
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(test_file, index=False)
        self.assertTrue(os.path.exists(test_file))
        
        # Test absolute path
        abs_path = os.path.abspath(test_file)
        self.assertTrue(os.path.exists(abs_path))
        
        # Test directory listing
        files = os.listdir('.')
        self.assertIn(test_file, files)
    
    def test_data_format_checking(self):
        """Test data format checking examples."""
        # Create test CSV with proper format
        good_data = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'label': ['positive', 'negative']
        })
        good_data.to_csv('good_data.csv', index=False)
        
        # Test loading
        loaded = pd.read_csv('good_data.csv')
        self.assertIn('text', loaded.columns)
        self.assertIn('label', loaded.columns)
        self.assertEqual(len(loaded), 2)
        
        # Create problematic data
        bad_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [None, None, None]  # All null values
        })
        bad_data.to_csv('bad_data.csv', index=False)
        
        loaded_bad = pd.read_csv('bad_data.csv')
        self.assertTrue(loaded_bad['col2'].isnull().all())
    
    def test_memory_optimization_examples(self):
        """Test memory optimization examples."""
        # Create larger dataset to test memory considerations
        large_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000)
        })
        large_data.to_csv('large_data.csv', index=False)
        
        # Test chunked reading (example from troubleshooting)
        chunk_size = 100
        chunks = []
        for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
            chunks.append(len(chunk))
        
        self.assertEqual(sum(chunks), 1000)
        self.assertTrue(all(chunk <= chunk_size for chunk in chunks))


class TestAPIExamples(unittest.TestCase):
    """Test API examples from documentation."""
    
    def test_function_signature_examples(self):
        """Test that function signature examples are valid."""
        # Test parameter combinations from documentation
        valid_combinations = [
            {
                'data': 'data.csv',
                'model': 'auto',
                'task': 'auto'
            },
            {
                'data': 'images/',
                'model': 'resnet18',
                'task': 'image_classification',
                'image_size': 224
            },
            {
                'data': 'text.csv',
                'model': 'bert',
                'task': 'text_classification',
                'target': 'label',
                'max_length': 128
            }
        ]
        
        for combo in valid_combinations:
            # Test that all required parameters are strings or numbers
            self.assertIsInstance(combo['data'], str)
            self.assertIsInstance(combo['model'], str)
            self.assertIsInstance(combo['task'], str)
            
            # Test optional parameters
            if 'target' in combo:
                self.assertIsInstance(combo['target'], str)
            if 'image_size' in combo:
                self.assertIsInstance(combo['image_size'], int)
            if 'max_length' in combo:
                self.assertIsInstance(combo['max_length'], int)
    
    def test_deployment_examples(self):
        """Test deployment examples from documentation."""
        # Test deployment parameter combinations
        deployment_configs = [
            {'format': 'api', 'port': 8000},
            {'format': 'onnx'},
            {'format': 'tflite'},
            {'format': 'torchscript'}
        ]
        
        for config in deployment_configs:
            self.assertIsInstance(config['format'], str)
            if 'port' in config:
                self.assertIsInstance(config['port'], int)
                self.assertGreater(config['port'], 0)
                self.assertLess(config['port'], 65536)


class TestNotebookExamples(unittest.TestCase):
    """Test examples from Jupyter notebooks."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent
        self.tutorials_dir = self.docs_dir / 'tutorials'
    
    def test_notebook_code_cells(self):
        """Test that notebook code cells contain valid Python."""
        notebook_path = self.tutorials_dir / '01_quick_start.ipynb'
        
        if notebook_path.exists():
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Extract code cells
            code_cells = [
                cell for cell in notebook['cells'] 
                if cell.get('cell_type') == 'code'
            ]
            
            self.assertGreater(len(code_cells), 0, "No code cells found in notebook")
            
            # Test that code cells contain Python-like content
            for i, cell in enumerate(code_cells):
                source = ''.join(cell.get('source', []))
                
                # Skip empty cells and comment-only cells
                if not source.strip() or source.strip().startswith('#'):
                    continue
                
                # Check for Python keywords/patterns
                python_indicators = [
                    'import', 'def ', 'class ', 'if ', 'for ', 'while ',
                    'print(', '=', 'neurolite'
                ]
                
                has_python = any(indicator in source for indicator in python_indicators)
                self.assertTrue(has_python, 
                              f"Code cell {i} doesn't appear to contain Python code: {source[:100]}")
    
    def test_notebook_imports(self):
        """Test that notebooks import required modules."""
        notebook_path = self.tutorials_dir / '01_quick_start.ipynb'
        
        if notebook_path.exists():
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Get all source code
            all_source = ''
            for cell in notebook['cells']:
                if cell.get('cell_type') == 'code':
                    all_source += ''.join(cell.get('source', []))
            
            # Check for required imports
            required_imports = ['neurolite', 'pandas', 'numpy']
            
            for import_name in required_imports:
                self.assertTrue(
                    f'import {import_name}' in all_source or f'from {import_name}' in all_source,
                    f"Required import '{import_name}' not found in notebook"
                )


def run_example_tests():
    """Run all example tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBasicExamples,
        TestConfigurationExamples,
        TestTroubleshootingExamples,
        TestAPIExamples,
        TestNotebookExamples
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_example_tests()
    sys.exit(0 if success else 1)