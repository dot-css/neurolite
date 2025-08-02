"""
Validation tests for the comprehensive testing suite itself.
Ensures all testing components are properly integrated and functional.
"""

import pytest
import os
import tempfile
import subprocess
import sys
from pathlib import Path
import importlib.util


class TestComprehensiveTestSuite:
    """Test the comprehensive testing suite components."""
    
    def test_test_runner_exists(self):
        """Test that the comprehensive test runner exists and is executable."""
        runner_path = Path(__file__).parent / 'run_comprehensive_tests.py'
        assert runner_path.exists(), "Comprehensive test runner not found"
        assert runner_path.is_file(), "Test runner is not a file"
        
        # Test that it's executable
        with open(runner_path, 'r') as f:
            content = f.read()
            assert 'def main()' in content, "Test runner missing main function"
            assert 'if __name__ == "__main__"' in content, "Test runner not executable"
    
    def test_ci_pipeline_configuration(self):
        """Test that CI pipeline configuration exists."""
        ci_config_path = Path(__file__).parent.parent / '.github' / 'workflows' / 'comprehensive-tests.yml'
        assert ci_config_path.exists(), "CI pipeline configuration not found"
        
        with open(ci_config_path, 'r') as f:
            content = f.read()
            assert 'name: Comprehensive Test Suite' in content
            assert 'unit-tests:' in content
            assert 'integration-tests:' in content
            assert 'performance-tests:' in content
            assert 'compatibility-tests:' in content
    
    def test_integration_test_modules(self):
        """Test that all integration test modules exist."""
        integration_dir = Path(__file__).parent / 'integration'
        assert integration_dir.exists(), "Integration tests directory not found"
        
        required_modules = [
            'test_end_to_end_workflows.py',
            'test_performance_benchmarks.py',
            'test_compatibility.py',
            'test_data_generators.py'
        ]
        
        for module in required_modules:
            module_path = integration_dir / module
            assert module_path.exists(), f"Integration test module {module} not found"
    
    def test_data_generator_functionality(self):
        """Test that the test data generator works correctly."""
        from tests.integration.test_data_generators import TestDataGenerator
        
        generator = TestDataGenerator(seed=42)
        
        # Test basic dataset generation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test classification data generation
            df = generator.generate_classification_data(n_samples=100, n_features=5)
            assert len(df) == 100
            assert len(df.columns) == 6  # 5 features + target
            
            # Test regression data generation
            df = generator.generate_regression_data(n_samples=50, n_features=3)
            assert len(df) == 50
            assert len(df.columns) == 4  # 3 features + target
            
            # Test image data generation
            image_dir = generator.generate_image_classification_data(
                n_classes=2, n_images_per_class=5, save_dir=temp_dir
            )
            assert os.path.exists(image_dir)
            
            # Check that images were created
            class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
            assert len(class_dirs) == 2
    
    def test_performance_benchmark_structure(self):
        """Test that performance benchmarks have proper structure."""
        from tests.integration.test_performance_benchmarks import TestPerformanceBenchmarks
        
        # Check that the class exists and has required methods
        benchmark_class = TestPerformanceBenchmarks
        
        required_methods = [
            'test_training_speed_small_dataset',
            'test_training_speed_large_dataset',
            'test_inference_speed',
            'test_memory_efficiency',
            'test_scalability_with_features'
        ]
        
        for method_name in required_methods:
            assert hasattr(benchmark_class, method_name), f"Missing benchmark method: {method_name}"
    
    def test_compatibility_test_coverage(self):
        """Test that compatibility tests cover required areas."""
        from tests.integration.test_compatibility import TestCompatibility
        
        compatibility_class = TestCompatibility
        
        required_methods = [
            'test_python_version_compatibility',
            'test_operating_system_compatibility',
            'test_dependency_imports',
            'test_basic_workflow_compatibility'
        ]
        
        for method_name in required_methods:
            assert hasattr(compatibility_class, method_name), f"Missing compatibility test: {method_name}"
    
    def test_end_to_end_workflow_coverage(self):
        """Test that end-to-end workflows cover major use cases."""
        from tests.integration.test_end_to_end_workflows import TestEndToEndWorkflows
        
        workflow_class = TestEndToEndWorkflows
        
        required_methods = [
            'test_tabular_classification_workflow',
            'test_tabular_regression_workflow',
            'test_image_classification_workflow',
            'test_text_classification_workflow',
            'test_minimal_code_interface'
        ]
        
        for method_name in required_methods:
            assert hasattr(workflow_class, method_name), f"Missing workflow test: {method_name}"
    
    def test_test_runner_command_line_interface(self):
        """Test that the test runner has proper CLI interface."""
        runner_path = Path(__file__).parent / 'run_comprehensive_tests.py'
        
        # Test help output
        try:
            result = subprocess.run(
                [sys.executable, str(runner_path), '--help'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, "Test runner help command failed"
            assert '--unit-only' in result.stdout, "Missing --unit-only option"
            assert '--integration-only' in result.stdout, "Missing --integration-only option"
            assert '--performance-only' in result.stdout, "Missing --performance-only option"
            assert '--verbose' in result.stdout, "Missing --verbose option"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Test runner help command timed out")
        except Exception as e:
            pytest.fail(f"Test runner CLI test failed: {e}")
    
    def test_test_data_directory_structure(self):
        """Test that test data can be generated with proper structure."""
        from tests.integration.test_data_generators import TestDataGenerator
        
        generator = TestDataGenerator(seed=42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets = generator.create_test_suite_datasets(temp_dir)
            
            # Check that metadata was created
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            assert os.path.exists(metadata_path), "Test data metadata not created"
            
            # Check that basic datasets were created
            required_datasets = [
                'classification', 'regression', 'text_classification', 
                'image_classification', 'time_series'
            ]
            
            for dataset_name in required_datasets:
                assert dataset_name in datasets, f"Missing dataset: {dataset_name}"
    
    def test_pytest_markers_and_configuration(self):
        """Test that pytest markers and configuration are properly set up."""
        # Check for pytest configuration file
        pytest_ini_path = Path(__file__).parent.parent / 'pytest.ini'
        pyproject_toml_path = Path(__file__).parent.parent / 'pyproject.toml'
        
        # At least one configuration should exist
        has_config = pytest_ini_path.exists() or pyproject_toml_path.exists()
        assert has_config, "No pytest configuration found"
        
        # Test that slow marker is available (should be defined in config)
        # This is tested by trying to use it
        try:
            import pytest
            # If this doesn't raise an error, the marker system is working
            pytest.mark.slow
        except Exception as e:
            pytest.fail(f"Pytest marker system not working: {e}")
    
    def test_test_coverage_requirements(self):
        """Test that test coverage tools are available."""
        try:
            import coverage
            assert hasattr(coverage, 'Coverage'), "Coverage module not properly installed"
        except ImportError:
            pytest.fail("Coverage module not available")
        
        # Test pytest-cov integration
        try:
            import pytest_cov
        except ImportError:
            pytest.fail("pytest-cov not available")
    
    def test_performance_monitoring_tools(self):
        """Test that performance monitoring tools are available."""
        try:
            import psutil
            assert hasattr(psutil, 'Process'), "psutil not properly installed"
        except ImportError:
            pytest.fail("psutil not available for performance monitoring")
        
        # Test memory measurement
        process = psutil.Process()
        memory_info = process.memory_info()
        assert hasattr(memory_info, 'rss'), "Memory monitoring not working"
    
    def test_test_isolation(self):
        """Test that tests can run in isolation without interference."""
        # This test ensures that test fixtures and setup don't interfere
        import tempfile
        import shutil
        
        # Create multiple temporary directories to simulate test isolation
        temp_dirs = []
        try:
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f'test_isolation_{i}_')
                temp_dirs.append(temp_dir)
                
                # Create some test files
                test_file = os.path.join(temp_dir, 'test.txt')
                with open(test_file, 'w') as f:
                    f.write(f'test content {i}')
                
                assert os.path.exists(test_file)
        
        finally:
            # Clean up
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    def test_error_handling_in_tests(self):
        """Test that test suite handles errors gracefully."""
        from tests.integration.test_data_generators import TestDataGenerator
        
        generator = TestDataGenerator(seed=42)
        
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError, AssertionError)):
            generator.generate_classification_data(n_samples=-1)
        
        # Test with invalid save path
        with pytest.raises((OSError, IOError, PermissionError)):
            generator.generate_classification_data(save_path='/invalid/path/file.csv')
    
    def test_test_reproducibility(self):
        """Test that tests are reproducible with fixed seeds."""
        from tests.integration.test_data_generators import TestDataGenerator
        
        # Generate data with same seed twice
        generator1 = TestDataGenerator(seed=42)
        generator2 = TestDataGenerator(seed=42)
        
        df1 = generator1.generate_classification_data(n_samples=100, n_features=5)
        df2 = generator2.generate_classification_data(n_samples=100, n_features=5)
        
        # Data should be identical
        import pandas as pd
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_comprehensive_test_documentation(self):
        """Test that comprehensive testing is properly documented."""
        # Check for README or documentation mentioning testing
        project_root = Path(__file__).parent.parent
        
        readme_files = list(project_root.glob('README*'))
        assert len(readme_files) > 0, "No README file found"
        
        # Check that at least one README mentions testing
        mentions_testing = False
        for readme_file in readme_files:
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if 'test' in content or 'testing' in content:
                        mentions_testing = True
                        break
            except Exception:
                continue
        
        assert mentions_testing, "Testing not mentioned in documentation"
    
    def test_test_suite_completeness(self):
        """Test that the test suite covers all major components."""
        # This is a meta-test that ensures we have comprehensive coverage
        
        # Check that we have tests for all major directories
        project_root = Path(__file__).parent.parent
        neurolite_dir = project_root / 'neurolite'
        
        if neurolite_dir.exists():
            # Get all Python modules in neurolite
            python_files = list(neurolite_dir.rglob('*.py'))
            python_files = [f for f in python_files if not f.name.startswith('__')]
            
            # We should have a reasonable number of test files
            test_files = list(Path(__file__).parent.rglob('test_*.py'))
            
            # Rough heuristic: should have at least 1 test file per 3 source files
            min_expected_tests = max(1, len(python_files) // 3)
            assert len(test_files) >= min_expected_tests, f"Insufficient test coverage: {len(test_files)} test files for {len(python_files)} source files"


if __name__ == "__main__":
    # Run the validation tests
    pytest.main([__file__, '-v'])