"""
Final integration and validation tests for NeuroLite.

This module contains comprehensive tests that validate the complete user workflow,
ensure all requirements are met, and verify the library is ready for release.
"""

import pytest
import tempfile
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock, patch
import importlib
import inspect

from neurolite import train, deploy, __version__
from neurolite.core import NeuroLiteError, get_logger
from neurolite.data import DataType
from neurolite.models import TaskType


class TestFinalIntegration:
    """Final integration tests validating complete user workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_datasets(self, temp_dir):
        """Create comprehensive sample datasets for testing."""
        datasets = {}
        
        # Tabular classification dataset
        np.random.seed(42)
        tabular_data = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randint(0, 5, 200),
            'feature4': np.random.choice(['A', 'B', 'C'], 200),
            'target': np.random.randint(0, 3, 200)
        })
        tabular_path = os.path.join(temp_dir, 'tabular_classification.csv')
        tabular_data.to_csv(tabular_path, index=False)
        datasets['tabular_classification'] = tabular_path
        
        # Tabular regression dataset
        regression_data = pd.DataFrame({
            'x1': np.random.randn(200),
            'x2': np.random.randn(200),
            'x3': np.random.uniform(0, 10, 200),
        })
        regression_data['y'] = (
            2 * regression_data['x1'] + 
            3 * regression_data['x2'] + 
            0.5 * regression_data['x3'] + 
            np.random.normal(0, 0.1, 200)
        )
        regression_path = os.path.join(temp_dir, 'tabular_regression.csv')
        regression_data.to_csv(regression_path, index=False)
        datasets['tabular_regression'] = regression_path
        
        return datasets  
  
    def test_minimal_code_interface_requirement_1_1(self, sample_datasets):
        """
        Test Requirement 1.1: Minimal Code Interface
        Verify that users can train models with less than 10 lines of code.
        """
        # Test 1: Single line training (tabular)
        model = train(sample_datasets['tabular_classification'], target='target')
        assert model is not None
        
        # Test 2: Two line training and prediction
        model = train(sample_datasets['tabular_regression'], target='y')
        prediction = model.predict({'x1': 0.5, 'x2': -0.3, 'x3': 5.0})
        assert prediction is not None
        
        # Verify models work
        assert model is not None
        assert prediction is not None
    
    def test_automatic_data_processing_requirement_2(self, sample_datasets):
        """
        Test Requirement 2: Automatic Data Processing
        Verify automatic data detection, preprocessing, and splitting.
        """
        # Test automatic data type detection
        for dataset_name, dataset_path in sample_datasets.items():
            if 'tabular' in dataset_name:
                target = 'target' if 'classification' in dataset_name else 'y'
            else:
                target = None
            
            model = train(
                dataset_path, 
                target=target,
                validation_split=0.2,
                test_split=0.1
            )
            
            assert model is not None
            # Verify that data was automatically processed
            assert hasattr(model, 'metadata')
            assert hasattr(model, 'training_history')
    
    def test_model_zoo_integration_requirement_3(self, sample_datasets):
        """
        Test Requirement 3: Model Zoo Integration
        Verify access to pre-configured models and automatic selection.
        """
        # Test automatic model selection
        model_auto = train(sample_datasets['tabular_classification'], target='target')
        assert model_auto is not None
        
        # Test explicit model selection
        model_explicit = train(
            sample_datasets['tabular_classification'], 
            target='target',
            model='random_forest_classifier'
        )
        assert model_explicit is not None
        
        # Verify different models can be used
        assert hasattr(model_auto, 'model')
        assert hasattr(model_explicit, 'model')
    
    def test_automated_training_evaluation_requirement_4(self, sample_datasets):
        """
        Test Requirement 4: Automated Training and Evaluation
        Verify automatic training orchestration and evaluation.
        """
        model = train(sample_datasets['tabular_classification'], target='target')
        
        # Verify training was automated
        assert hasattr(model, 'training_history')
        assert len(model.training_history) > 0
        
        # Verify evaluation was performed
        assert hasattr(model, 'evaluation_results')
        evaluation = model.evaluate()
        assert evaluation is not None
        assert isinstance(evaluation, dict)
        assert len(evaluation) > 0
    
    def test_hyperparameter_optimization_requirement_5(self, sample_datasets):
        """
        Test Requirement 5: Hyperparameter Optimization
        Verify automatic hyperparameter tuning capability.
        """
        model = train(
            sample_datasets['tabular_classification'], 
            target='target',
            optimize=True,
            config={'optimization_trials': 2}  # Minimal for testing
        )
        
        assert model is not None
        # Verify optimization was performed
        assert hasattr(model, 'metadata')
    
    def test_deployment_requirement_6(self, sample_datasets):
        """
        Test Requirement 6: One-Click Deployment
        Verify model export and deployment capabilities.
        """
        model = train(sample_datasets['tabular_classification'], target='target')
        
        # Test model export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, 'model.pkl')
            model.export(export_path)
            assert os.path.exists(export_path)
        
        # Test API deployment (mock to avoid actual server startup)
        with patch('neurolite.api.create_api_server') as mock_create_api:
            mock_server = Mock()
            mock_create_api.return_value = mock_server
            
            api_server = deploy(model, format='api', port=8080)
            assert api_server == mock_server
            mock_create_api.assert_called_once()
    
    def test_production_quality_requirement_10(self):
        """
        Test Requirement 10: Production-Ready Quality
        Verify documentation, error handling, and quality standards.
        """
        # Test that main functions have proper docstrings
        assert train.__doc__ is not None
        assert len(train.__doc__.strip()) > 100  # Substantial documentation
        
        assert deploy.__doc__ is not None
        assert len(deploy.__doc__.strip()) > 50
        
        # Test version information is available
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        
        # Test error handling provides informative messages
        with pytest.raises(NeuroLiteError) as exc_info:
            train("nonexistent_file.csv")
        
        error_msg = str(exc_info.value)
        assert "does not exist" in error_msg.lower()
        assert len(error_msg) > 50  # Informative error message
    
    def test_complete_user_workflow_integration(self, sample_datasets):
        """
        Test complete end-to-end user workflow integration.
        This test simulates a real user's complete workflow.
        """
        # Step 1: Load and train model
        start_time = time.time()
        model = train(
            sample_datasets['tabular_classification'],
            target='target',
            validation_split=0.2,
            test_split=0.1
        )
        training_time = time.time() - start_time
        
        # Verify training completed successfully
        assert model is not None
        assert training_time < 60  # Should complete within reasonable time
        
        # Step 2: Make predictions
        test_data = {
            'feature1': 0.5,
            'feature2': -0.3,
            'feature3': 2,
            'feature4': 'A'
        }
        
        prediction = model.predict(test_data)
        assert prediction is not None
        assert isinstance(prediction, (int, np.integer, float, np.floating))
        
        # Step 3: Evaluate model
        evaluation = model.evaluate()
        assert evaluation is not None
        assert isinstance(evaluation, dict)
        assert len(evaluation) > 0
        
        # Step 4: Export model
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, 'final_model.pkl')
            model.export(export_path)
            assert os.path.exists(export_path)
            assert os.path.getsize(export_path) > 0
        
        # Step 5: Batch predictions
        test_df = pd.DataFrame([test_data] * 5)
        batch_predictions = model.predict(test_df)
        assert len(batch_predictions) == 5
        
        # Verify workflow completed successfully
        logger = get_logger(__name__)
        logger.info(f"Complete user workflow test passed in {training_time:.2f}s")


class TestComponentIntegration:
    """Test integration between all library components."""
    
    def test_core_components_integration(self):
        """Test that all core components are properly integrated."""
        # Test that all main modules can be imported
        from neurolite import core, data, models, training, evaluation, deployment
        from neurolite import visualization, workflows, cli
        
        # Verify core components exist
        assert hasattr(core, 'get_logger')
        assert hasattr(core, 'NeuroLiteError')
        assert hasattr(data, 'detect_data_type')
        assert hasattr(models, 'get_model_registry')
        assert hasattr(training, 'TrainingEngine')
        assert hasattr(evaluation, 'evaluate_model')
        assert hasattr(deployment, 'create_api_server')
        
        # Test that components can interact
        logger = core.get_logger('test')
        assert logger is not None
        
        registry = models.get_model_registry()
        assert registry is not None
        assert hasattr(registry, 'list_models')
    
    def test_api_consistency(self):
        """Test that all API interfaces are consistent."""
        # Test main API functions
        assert callable(train)
        assert callable(deploy)
        
        # Check function signatures
        train_sig = inspect.signature(train)
        assert 'data' in train_sig.parameters
        assert 'model' in train_sig.parameters
        assert 'task' in train_sig.parameters
        
        deploy_sig = inspect.signature(deploy)
        assert 'model' in deploy_sig.parameters
        assert 'format' in deploy_sig.parameters
    
    def test_error_handling_integration(self):
        """Test that error handling is consistent across components."""
        # Test that all custom exceptions inherit from NeuroLiteError
        from neurolite.core import (
            NeuroLiteError, DataError, ModelError, 
            TrainingError, ConfigurationError
        )
        
        assert issubclass(DataError, NeuroLiteError)
        assert issubclass(ModelError, NeuroLiteError)
        assert issubclass(TrainingError, NeuroLiteError)
        assert issubclass(ConfigurationError, NeuroLiteError)
        
        # Test that errors provide helpful messages
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"


class TestEdgeCaseHandling:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            # Create empty CSV
            pd.DataFrame().to_csv(f.name, index=False)
            
            with pytest.raises(NeuroLiteError):
                train(f.name)
        
        os.unlink(f.name)
    
    def test_invalid_data_format_handling(self):
        """Test handling of invalid data formats."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not valid data")
            f.flush()
            
            with pytest.raises(NeuroLiteError):
                train(f.name)
        
        os.unlink(f.name)
    
    def test_missing_target_column_handling(self):
        """Test handling of missing target columns."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            df.to_csv(f.name, index=False)
            
            with pytest.raises(NeuroLiteError):
                train(f.name, target='nonexistent_column')
        
        os.unlink(f.name)
    
    def test_insufficient_data_handling(self):
        """Test handling of datasets with insufficient data."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            # Create dataset with only 1 sample
            df = pd.DataFrame({'feature1': [1], 'target': [0]})
            df.to_csv(f.name, index=False)
            
            with pytest.raises(NeuroLiteError):
                train(f.name, target='target')
        
        os.unlink(f.name)


class TestRequirementsValidation:
    """Validate that all specified requirements are met."""
    
    def test_requirement_1_1_minimal_code_interface(self):
        """Validate Requirement 1.1: Users can train models with <10 lines of code."""
        # Count required parameters for basic usage
        sig = inspect.signature(train)
        required_params = [
            p for p in sig.parameters.values() 
            if p.default == inspect.Parameter.empty
        ]
        
        # Should only require 'data' parameter for basic usage
        assert len(required_params) <= 1
        assert 'data' in [p.name for p in required_params]
    
    def test_requirement_1_2_intelligent_defaults(self):
        """Validate Requirement 1.2: System applies intelligent defaults."""
        sig = inspect.signature(train)
        
        # Check that most parameters have defaults
        params_with_defaults = [
            p for p in sig.parameters.values()
            if p.default != inspect.Parameter.empty
        ]
        
        # Most parameters should have defaults
        assert len(params_with_defaults) >= 5
        
        # Check specific important defaults
        param_dict = {p.name: p.default for p in sig.parameters.values()}
        assert param_dict.get('model') == 'auto'
        assert param_dict.get('task') == 'auto'
        assert param_dict.get('validation_split') == 0.2
        assert param_dict.get('test_split') == 0.1
    
    def test_requirement_10_1_comprehensive_documentation(self):
        """Validate Requirement 10.1: Comprehensive documentation."""
        # Check that main functions have substantial documentation
        assert train.__doc__ is not None
        assert len(train.__doc__.strip()) > 200
        
        assert deploy.__doc__ is not None
        assert len(deploy.__doc__.strip()) > 100
        
        # Check that docstrings include examples
        assert 'Example' in train.__doc__ or 'example' in train.__doc__
    
    def test_requirement_10_2_clear_error_messages(self):
        """Validate Requirement 10.2: Clear, actionable error messages."""
        # Test various error conditions and verify messages are helpful
        
        # Non-existent file
        with pytest.raises(NeuroLiteError) as exc_info:
            train("nonexistent_file.csv")
        
        error_msg = str(exc_info.value)
        assert len(error_msg) > 50  # Substantial error message
        assert "does not exist" in error_msg.lower()
        assert "check" in error_msg.lower()  # Suggests action


class TestReleaseReadiness:
    """Test that the library is ready for release."""
    
    def test_package_structure(self):
        """Test that package structure is correct."""
        # Check that main package exists
        import neurolite
        assert neurolite is not None
        
        # Check that main modules exist
        from neurolite import core, data, models, training
        from neurolite import evaluation, deployment, visualization
        from neurolite import workflows, cli
        
        # Check that main functions are exported
        from neurolite import train, deploy
        assert callable(train)
        assert callable(deploy)
    
    def test_version_information(self):
        """Test that version information is properly configured."""
        from neurolite import __version__
        from neurolite._version import get_version, get_version_info
        
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        
        version = get_version()
        assert version == __version__
        
        version_info = get_version_info()
        assert isinstance(version_info, tuple)
        assert len(version_info) >= 3
    
    def test_example_workflows(self):
        """Test that example workflows work correctly."""
        # Create a simple example that should work
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            # Create Iris-like dataset
            np.random.seed(42)
            df = pd.DataFrame({
                'sepal_length': np.random.normal(5.8, 0.8, 150),
                'sepal_width': np.random.normal(3.0, 0.4, 150),
                'petal_length': np.random.normal(3.8, 1.8, 150),
                'petal_width': np.random.normal(1.2, 0.8, 150),
                'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
            })
            df.to_csv(f.name, index=False)
            
            # Test the example workflow from documentation
            model = train(f.name, target='species')
            prediction = model.predict({
                'sepal_length': 5.1,
                'sepal_width': 3.5,
                'petal_length': 1.4,
                'petal_width': 0.2
            })
            
            assert model is not None
            assert prediction is not None
        
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])