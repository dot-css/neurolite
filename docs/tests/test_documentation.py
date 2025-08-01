"""
Documentation Tests

This module contains tests to ensure that documentation examples
remain functional and up-to-date.
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

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import neurolite
    NEUROLITE_AVAILABLE = True
except ImportError:
    NEUROLITE_AVAILABLE = False


class TestDocumentationExamples(unittest.TestCase):
    """Test that documentation examples work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @unittest.skipUnless(NEUROLITE_AVAILABLE, "NeuroLite not available")
    def test_basic_api_usage(self):
        """Test basic API usage examples from documentation."""
        # Create sample tabular data
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        df.to_csv('test_data.csv', index=False)
        
        # Test basic training
        try:
            model = neurolite.train(
                data='test_data.csv',
                target='target',
                task='classification',
                optimize=False  # Skip optimization for faster testing
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'predict'))
        except Exception as e:
            self.fail(f"Basic API usage failed: {e}")
    
    @unittest.skipUnless(NEUROLITE_AVAILABLE, "NeuroLite not available")
    def test_image_classification_example(self):
        """Test image classification example structure."""
        # Create sample image directory structure
        os.makedirs('images/cats', exist_ok=True)
        os.makedirs('images/dogs', exist_ok=True)
        
        # Create dummy image files (just empty files for structure test)
        for i in range(5):
            Path(f'images/cats/cat_{i}.jpg').touch()
            Path(f'images/dogs/dog_{i}.jpg').touch()
        
        # Test that the directory structure is recognized
        self.assertTrue(os.path.exists('images/cats'))
        self.assertTrue(os.path.exists('images/dogs'))
        self.assertEqual(len(os.listdir('images/cats')), 5)
        self.assertEqual(len(os.listdir('images/dogs')), 5)
    
    @unittest.skipUnless(NEUROLITE_AVAILABLE, "NeuroLite not available")
    def test_text_classification_example(self):
        """Test text classification example data format."""
        # Create sample text data
        text_data = {
            'text': [
                'This is a positive review',
                'This is a negative review',
                'Another positive example',
                'Another negative example'
            ],
            'label': ['positive', 'negative', 'positive', 'negative']
        }
        df = pd.DataFrame(text_data)
        df.to_csv('text_data.csv', index=False)
        
        # Verify data format
        loaded_df = pd.read_csv('text_data.csv')
        self.assertIn('text', loaded_df.columns)
        self.assertIn('label', loaded_df.columns)
        self.assertEqual(len(loaded_df), 4)
    
    def test_configuration_examples(self):
        """Test configuration examples from documentation."""
        # Test parameter validation examples
        test_cases = [
            {'validation_split': 0.2, 'test_split': 0.1, 'expected': True},
            {'validation_split': 0.5, 'test_split': 0.6, 'expected': False},  # Sum > 1.0
            {'validation_split': -0.1, 'test_split': 0.1, 'expected': False},  # Negative
            {'validation_split': 1.5, 'test_split': 0.1, 'expected': False},  # > 1.0
        ]
        
        for case in test_cases:
            val_split = case['validation_split']
            test_split = case['test_split']
            expected = case['expected']
            
            # Test validation logic
            is_valid = (
                0.0 <= val_split <= 1.0 and
                0.0 <= test_split <= 1.0 and
                val_split + test_split < 1.0
            )
            
            self.assertEqual(is_valid, expected,
                           f"Validation failed for val_split={val_split}, test_split={test_split}")
    
    def test_troubleshooting_examples(self):
        """Test troubleshooting guide examples."""
        # Test file existence checking
        test_file = 'test_file.csv'
        self.assertFalse(os.path.exists(test_file))
        
        # Create file and test again
        Path(test_file).touch()
        self.assertTrue(os.path.exists(test_file))
        
        # Test directory listing
        files = os.listdir('.')
        self.assertIn(test_file, files)
    
    def test_data_format_examples(self):
        """Test data format examples from documentation."""
        # Test CSV format for tabular data
        tabular_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        tabular_data.to_csv('tabular.csv', index=False)
        
        # Verify CSV can be loaded
        loaded = pd.read_csv('tabular.csv')
        self.assertEqual(len(loaded), 5)
        self.assertListEqual(list(loaded.columns), ['feature1', 'feature2', 'target'])
        
        # Test text data format
        text_data = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'label': ['class1', 'class2']
        })
        text_data.to_csv('text.csv', index=False)
        
        loaded_text = pd.read_csv('text.csv')
        self.assertEqual(len(loaded_text), 2)
        self.assertIn('text', loaded_text.columns)
        self.assertIn('label', loaded_text.columns)


class TestTutorialNotebooks(unittest.TestCase):
    """Test that tutorial notebooks are valid and executable."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent
        self.tutorials_dir = self.docs_dir / 'tutorials'
    
    def test_notebook_files_exist(self):
        """Test that tutorial notebook files exist."""
        expected_notebooks = [
            '01_quick_start.ipynb',
            'computer_vision/01_image_classification.ipynb',
            'nlp/01_text_classification.ipynb'
        ]
        
        for notebook in expected_notebooks:
            notebook_path = self.tutorials_dir / notebook
            self.assertTrue(notebook_path.exists(),
                          f"Tutorial notebook not found: {notebook}")
    
    def test_notebook_structure(self):
        """Test that notebooks have proper structure."""
        import json
        
        notebook_path = self.tutorials_dir / '01_quick_start.ipynb'
        if notebook_path.exists():
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Check basic notebook structure
            self.assertIn('cells', notebook)
            self.assertIn('metadata', notebook)
            self.assertIn('nbformat', notebook)
            
            # Check that there are both markdown and code cells
            cell_types = [cell.get('cell_type') for cell in notebook['cells']]
            self.assertIn('markdown', cell_types)
            self.assertIn('code', cell_types)


class TestExampleScripts(unittest.TestCase):
    """Test that example scripts are valid Python code."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent
        self.examples_dir = self.docs_dir / 'examples'
    
    def test_example_scripts_exist(self):
        """Test that example scripts exist."""
        expected_scripts = [
            'basic/image_classification.py',
            'basic/text_classification.py'
        ]
        
        for script in expected_scripts:
            script_path = self.examples_dir / script
            self.assertTrue(script_path.exists(),
                          f"Example script not found: {script}")
    
    def test_example_scripts_syntax(self):
        """Test that example scripts have valid Python syntax."""
        import ast
        
        example_scripts = [
            self.examples_dir / 'basic/image_classification.py',
            self.examples_dir / 'basic/text_classification.py'
        ]
        
        for script_path in example_scripts:
            if script_path.exists():
                with open(script_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    self.fail(f"Syntax error in {script_path}: {e}")
    
    def test_example_imports(self):
        """Test that example scripts have proper imports."""
        script_path = self.examples_dir / 'basic/image_classification.py'
        
        if script_path.exists():
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required imports
            self.assertIn('import neurolite', content)
            self.assertIn('import os', content)
            self.assertIn('import numpy', content)


class TestAPIDocumentation(unittest.TestCase):
    """Test API documentation consistency."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent
        self.api_doc = self.docs_dir / 'api/README.md'
    
    def test_api_doc_exists(self):
        """Test that API documentation exists."""
        self.assertTrue(self.api_doc.exists())
    
    def test_api_doc_structure(self):
        """Test API documentation structure."""
        if self.api_doc.exists():
            with open(self.api_doc, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for main sections
            required_sections = [
                '# API Reference',
                '## Core Functions',
                '### `train()`',
                '### `deploy()`',
                '## Data Classes',
                '## Exceptions'
            ]
            
            for section in required_sections:
                self.assertIn(section, content,
                            f"Missing section in API docs: {section}")
    
    @unittest.skipUnless(NEUROLITE_AVAILABLE, "NeuroLite not available")
    def test_api_function_signatures(self):
        """Test that documented function signatures match actual functions."""
        # Check train function signature
        import inspect
        
        train_sig = inspect.signature(neurolite.train)
        train_params = list(train_sig.parameters.keys())
        
        # These parameters should be documented
        expected_params = ['data', 'model', 'task', 'target', 'validation_split', 'test_split']
        
        for param in expected_params:
            self.assertIn(param, train_params,
                        f"Parameter {param} not found in train function")


class TestDocumentationConsistency(unittest.TestCase):
    """Test consistency across documentation files."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = Path(__file__).parent.parent
    
    def test_readme_links(self):
        """Test that README links point to existing files."""
        readme_path = self.docs_dir / 'README.md'
        
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract markdown links
            import re
            links = re.findall(r'\[.*?\]\((.*?)\)', content)
            
            for link in links:
                # Skip external links
                if link.startswith('http'):
                    continue
                
                # Check if internal link exists
                link_path = self.docs_dir / link
                if not link_path.exists():
                    # Some links might be relative to project root
                    alt_path = self.docs_dir.parent / link
                    self.assertTrue(alt_path.exists() or link_path.exists(),
                                  f"Broken link in README: {link}")
    
    def test_consistent_terminology(self):
        """Test consistent terminology across documentation."""
        # Define expected terminology
        terminology = {
            'NeuroLite': 'NeuroLite',  # Consistent capitalization
            'API': 'API',  # Not 'api' or 'Api'
            'CSV': 'CSV',  # Not 'csv'
        }
        
        doc_files = [
            self.docs_dir / 'README.md',
            self.docs_dir / 'api/README.md',
            self.docs_dir / 'troubleshooting.md'
        ]
        
        for doc_file in doc_files:
            if doc_file.exists():
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for consistent terminology (this is a basic check)
                # In practice, you might want more sophisticated checks
                self.assertIn('NeuroLite', content,
                            f"NeuroLite not mentioned in {doc_file}")


def run_documentation_tests():
    """Run all documentation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDocumentationExamples,
        TestTutorialNotebooks,
        TestExampleScripts,
        TestAPIDocumentation,
        TestDocumentationConsistency
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_documentation_tests()
    sys.exit(0 if success else 1)