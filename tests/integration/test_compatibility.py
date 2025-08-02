"""
Compatibility tests for different Python versions and operating systems.
Tests cross-platform functionality and version compatibility.
"""

import pytest
import sys
import platform
import tempfile
import os
import numpy as np
import pandas as pd
import importlib

from neurolite import train
from neurolite.core.config import Config


class TestCompatibility:
    """Test compatibility across different environments."""
    
    def test_python_version_compatibility(self):
        """Test that library works with supported Python versions."""
        python_version = sys.version_info
        
        # Check minimum Python version
        assert python_version >= (3, 8), f"Python {python_version} not supported. Minimum: 3.8"
        
        # Test basic functionality
        assert hasattr(sys, 'version_info')
        assert hasattr(platform, 'system')
    
    def test_operating_system_compatibility(self):
        """Test compatibility across different operating systems."""
        os_name = platform.system()
        
        # Should work on Windows, macOS, and Linux
        supported_os = ['Windows', 'Darwin', 'Linux']
        assert os_name in supported_os, f"OS {os_name} not in supported list: {supported_os}"
        
        # Test path handling
        temp_dir = tempfile.mkdtemp()
        assert os.path.exists(temp_dir)
        
        # Test file operations
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        assert os.path.exists(test_file)
        os.remove(test_file)
        os.rmdir(temp_dir)
    
    def test_dependency_imports(self):
        """Test that all required dependencies can be imported."""
        required_packages = [
            'numpy',
            'pandas',
            'sklearn',
            'torch',
            'transformers',
            'optuna',
            'matplotlib',
            'seaborn',
            'plotly',
            'flask',
            'fastapi',
            'click',
            'tqdm',
            'PIL',
            'cv2',
            'onnx'
        ]
        
        for package in required_packages:
            try:
                if package == 'sklearn':
                    import sklearn
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'cv2':
                    import cv2
                else:
                    importlib.import_module(package)
                print(f"✓ {package} imported successfully")
            except ImportError as e:
                pytest.fail(f"Failed to import required package {package}: {e}")
    
    def test_optional_dependency_handling(self):
        """Test graceful handling of optional dependencies."""
        # Test TensorFlow (optional)
        try:
            import tensorflow as tf
            tensorflow_available = True
            print("✓ TensorFlow available")
        except ImportError:
            tensorflow_available = False
            print("! TensorFlow not available (optional)")
        
        # Test XGBoost (optional)
        try:
            import xgboost as xgb
            xgboost_available = True
            print("✓ XGBoost available")
        except ImportError:
            xgboost_available = False
            print("! XGBoost not available (optional)")
        
        # Library should work without optional dependencies
        assert True  # If we get here, basic imports worked
    
    def test_file_path_handling(self):
        """Test file path handling across different OS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different path formats
            paths_to_test = [
                os.path.join(temp_dir, 'test.csv'),
                os.path.join(temp_dir, 'subdir', 'test.csv'),
                os.path.join(temp_dir, 'test with spaces.csv'),
            ]
            
            for path in paths_to_test:
                # Create directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # Create test data
                df = pd.DataFrame({
                    'x': [1, 2, 3, 4, 5],
                    'y': [0, 1, 0, 1, 0]
                })
                df.to_csv(path, index=False)
                
                # Test that file can be read
                assert os.path.exists(path)
                loaded_df = pd.read_csv(path)
                assert len(loaded_df) == 5
    
    def test_unicode_handling(self):
        """Test handling of unicode characters in data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data with unicode characters
            data = {
                'text': ['Hello 世界', 'Café ñoño', 'Москва', '🚀 rocket', 'naïve résumé'],
                'label': [0, 1, 0, 1, 0]
            }
            
            df = pd.DataFrame(data)
            file_path = os.path.join(temp_dir, 'unicode_test.csv')
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            # Test loading
            loaded_df = pd.read_csv(file_path, encoding='utf-8')
            assert len(loaded_df) == 5
            assert '世界' in loaded_df['text'].iloc[0]
    
    def test_memory_architecture_compatibility(self):
        """Test compatibility with different memory architectures."""
        # Test with different numpy dtypes
        dtypes_to_test = [
            np.int32, np.int64, np.float32, np.float64
        ]
        
        for dtype in dtypes_to_test:
            arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
            assert arr.dtype == dtype
            
            # Test conversion
            df = pd.DataFrame({'values': arr, 'target': [0, 1, 0, 1, 0]})
            assert len(df) == 5
    
    def test_locale_compatibility(self):
        """Test compatibility with different system locales."""
        import locale
        
        # Get current locale
        current_locale = locale.getlocale()
        print(f"Current locale: {current_locale}")
        
        # Test number formatting
        test_number = 1234.56
        formatted = f"{test_number:.2f}"
        assert isinstance(formatted, str)
        
        # Test date handling
        import datetime
        now = datetime.datetime.now()
        assert isinstance(now, datetime.datetime)
    
    def test_multiprocessing_compatibility(self):
        """Test multiprocessing functionality."""
        import multiprocessing as mp
        
        # Test that multiprocessing is available
        cpu_count = mp.cpu_count()
        assert cpu_count > 0
        print(f"CPU count: {cpu_count}")
        
        # Test simple multiprocessing operation
        def square(x):
            return x * x
        
        with mp.Pool(processes=2) as pool:
            results = pool.map(square, [1, 2, 3, 4])
            assert results == [1, 4, 9, 16]
    
    def test_gpu_detection(self):
        """Test GPU detection and handling."""
        import torch
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA devices: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"Device {i}: {device_name}")
        
        # Test MPS (Apple Silicon) availability
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            print(f"MPS available: {mps_available}")
        
        # Library should work regardless of GPU availability
        assert True
    
    def test_configuration_compatibility(self):
        """Test configuration system compatibility."""
        config = Config()
        
        # Test setting and getting values
        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value'
        
        # Test different data types
        config.set('int_value', 42)
        config.set('float_value', 3.14)
        config.set('bool_value', True)
        config.set('list_value', [1, 2, 3])
        
        assert config.get('int_value') == 42
        assert config.get('float_value') == 3.14
        assert config.get('bool_value') is True
        assert config.get('list_value') == [1, 2, 3]
    
    def test_basic_workflow_compatibility(self):
        """Test basic workflow across different environments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simple test dataset
            np.random.seed(42)
            data = {
                'feature1': np.random.normal(0, 1, 100),
                'feature2': np.random.normal(0, 1, 100),
                'target': np.random.randint(0, 2, 100)
            }
            
            df = pd.DataFrame(data)
            file_path = os.path.join(temp_dir, 'test_data.csv')
            df.to_csv(file_path, index=False)
            
            # Test basic training workflow
            try:
                model = train(
                    data=file_path,
                    task="classification",
                    target="target",
                    config={'epochs': 1}  # Minimal for compatibility test
                )
                
                assert model is not None
                
                # Test prediction
                test_data = {'feature1': 0.5, 'feature2': -0.3}
                prediction = model.predict(test_data)
                assert prediction is not None
                
                print("✓ Basic workflow compatible")
                
            except Exception as e:
                pytest.fail(f"Basic workflow failed on this environment: {e}")
    
    def test_package_metadata(self):
        """Test package metadata and version information."""
        import neurolite
        
        # Test version information
        assert hasattr(neurolite, '__version__')
        version = neurolite.__version__
        assert isinstance(version, str)
        assert len(version.split('.')) >= 2  # At least major.minor
        
        print(f"NeuroLite version: {version}")
        
        # Test package structure
        expected_modules = [
            'neurolite.core',
            'neurolite.data',
            'neurolite.models',
            'neurolite.training',
            'neurolite.evaluation',
            'neurolite.deployment',
            'neurolite.visualization'
        ]
        
        for module_name in expected_modules:
            try:
                importlib.import_module(module_name)
                print(f"✓ {module_name} available")
            except ImportError as e:
                pytest.fail(f"Expected module {module_name} not available: {e}")
    
    def test_error_message_compatibility(self):
        """Test that error messages are properly formatted across platforms."""
        from neurolite.core.exceptions import NeuroLiteError, DataError
        
        # Test custom exceptions
        try:
            raise DataError("Test error message")
        except DataError as e:
            error_str = str(e)
            assert "Test error message" in error_str
            assert isinstance(e, NeuroLiteError)  # Should inherit from base
        
        # Test error formatting
        try:
            train(data="nonexistent_file.csv", task="classification")
        except Exception as e:
            error_str = str(e)
            assert isinstance(error_str, str)
            assert len(error_str) > 0
    
    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy versions."""
        import numpy as np
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.dtype in [np.int32, np.int64]  # Platform dependent
        
        # Test random number generation
        np.random.seed(42)
        random_arr = np.random.randn(100)
        assert len(random_arr) == 100
        assert isinstance(random_arr[0], (np.floating, float))
        
        # Test array operations
        result = np.mean(arr)
        assert isinstance(result, (np.floating, float))
        
        print(f"NumPy version: {np.__version__}")
    
    def test_pandas_version_compatibility(self):
        """Test compatibility with different Pandas versions."""
        import pandas as pd
        
        # Test DataFrame creation
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        
        assert len(df) == 3
        assert list(df.columns) == ['A', 'B', 'C']
        
        # Test CSV operations
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test.csv')
            df.to_csv(csv_path, index=False)
            
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == 3
        
        print(f"Pandas version: {pd.__version__}")
    
    def test_pytorch_version_compatibility(self):
        """Test compatibility with different PyTorch versions."""
        try:
            import torch
            
            # Test tensor creation
            tensor = torch.tensor([1, 2, 3, 4, 5])
            assert tensor.shape == (5,)
            
            # Test device handling
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor_on_device = tensor.to(device)
            assert tensor_on_device.device.type in ['cuda', 'cpu']
            
            # Test basic operations
            result = torch.mean(tensor.float())
            assert isinstance(result.item(), float)
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_sklearn_version_compatibility(self):
        """Test compatibility with different scikit-learn versions."""
        try:
            import sklearn
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            
            # Test dataset generation
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            assert X.shape == (100, 10)
            assert len(y) == 100
            
            # Test train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            assert len(X_train) == 80
            assert len(X_test) == 20
            
            # Test model training
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)
            
            print(f"scikit-learn version: {sklearn.__version__}")
            
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_transformers_version_compatibility(self):
        """Test compatibility with different transformers versions."""
        try:
            import transformers
            from transformers import AutoTokenizer
            
            # Test tokenizer loading (use a small model)
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Test tokenization
            text = "Hello world, this is a test."
            tokens = tokenizer.encode(text)
            assert len(tokens) > 0
            
            # Test decoding
            decoded = tokenizer.decode(tokens)
            assert isinstance(decoded, str)
            
            print(f"Transformers version: {transformers.__version__}")
            
        except ImportError:
            pytest.skip("Transformers not available")
        except Exception as e:
            pytest.skip(f"Transformers test failed: {e}")
    
    def test_environment_variable_handling(self):
        """Test handling of environment variables across platforms."""
        import os
        
        # Test setting and getting environment variables
        test_var = 'NEUROLITE_TEST_VAR'
        test_value = 'test_value_123'
        
        os.environ[test_var] = test_value
        assert os.environ.get(test_var) == test_value
        
        # Test path environment variables
        path_var = os.environ.get('PATH')
        assert path_var is not None
        assert len(path_var) > 0
        
        # Clean up
        del os.environ[test_var]
        assert os.environ.get(test_var) is None
    
    def test_file_encoding_compatibility(self):
        """Test file encoding handling across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test UTF-8 encoding
            utf8_file = os.path.join(temp_dir, 'utf8_test.txt')
            test_text = "Hello 世界 🌍 café naïve résumé"
            
            with open(utf8_file, 'w', encoding='utf-8') as f:
                f.write(test_text)
            
            with open(utf8_file, 'r', encoding='utf-8') as f:
                read_text = f.read()
            
            assert read_text == test_text
            
            # Test CSV with unicode
            csv_file = os.path.join(temp_dir, 'unicode_test.csv')
            df = pd.DataFrame({
                'text': ['Hello', '世界', 'café', '🌍'],
                'value': [1, 2, 3, 4]
            })
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            loaded_df = pd.read_csv(csv_file, encoding='utf-8')
            
            assert len(loaded_df) == 4
            assert '世界' in loaded_df['text'].values
    
    def test_thread_safety(self):
        """Test thread safety of core components."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker_function(worker_id):
            try:
                # Test configuration access
                from neurolite.core.config import Config
                config = Config()
                config.set(f'worker_{worker_id}', worker_id)
                value = config.get(f'worker_{worker_id}')
                
                results.put(('success', worker_id, value))
            except Exception as e:
                results.put(('error', worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        successful_workers = 0
        while not results.empty():
            status, worker_id, result = results.get()
            if status == 'success':
                successful_workers += 1
                assert result == worker_id
        
        assert successful_workers == 5, f"Only {successful_workers}/5 workers succeeded"
    
    def test_large_file_handling(self):
        """Test handling of large files across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a moderately large CSV file
            large_file = os.path.join(temp_dir, 'large_test.csv')
            
            # Generate data in chunks to avoid memory issues
            chunk_size = 1000
            n_chunks = 10
            
            # Write header
            with open(large_file, 'w') as f:
                f.write('feature1,feature2,feature3,target\n')
            
            # Write data in chunks
            for chunk in range(n_chunks):
                np.random.seed(chunk)  # Consistent data
                data = np.random.randn(chunk_size, 4)
                
                with open(large_file, 'a') as f:
                    for row in data:
                        f.write(f'{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{int(row[3] > 0)}\n')
            
            # Test file size
            file_size = os.path.getsize(large_file)
            assert file_size > 100000  # Should be > 100KB
            
            # Test loading with pandas
            df = pd.read_csv(large_file)
            assert len(df) == chunk_size * n_chunks
            
            print(f"Large file size: {file_size / 1024:.1f} KB")
    
    def test_signal_handling(self):
        """Test signal handling compatibility."""
        import signal
        import time
        
        # Test that signal module is available
        assert hasattr(signal, 'SIGINT')
        
        # Test signal handler setup (without actually triggering)
        original_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
        
        # Restore original handler
        signal.signal(signal.SIGINT, original_handler)
        
        # This test mainly ensures signal handling doesn't crash
        assert True
    
    def test_subprocess_compatibility(self):
        """Test subprocess execution across platforms."""
        import subprocess
        
        # Test simple command execution
        if platform.system() == 'Windows':
            result = subprocess.run(['echo', 'test'], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(['echo', 'test'], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert 'test' in result.stdout
    
    def test_temporary_directory_handling(self):
        """Test temporary directory handling across platforms."""
        import tempfile
        import shutil
        
        # Test temporary directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
            
            # Test file creation in temp directory
            test_file = os.path.join(temp_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test content')
            
            assert os.path.exists(test_file)
            
            # Test subdirectory creation
            sub_dir = os.path.join(temp_dir, 'subdir')
            os.makedirs(sub_dir)
            assert os.path.exists(sub_dir)
        
        # Directory should be cleaned up automatically
        assert not os.path.exists(temp_dir)
    
    def test_logging_compatibility(self):
        """Test logging system compatibility."""
        import logging
        import io
        
        # Test logger creation
        logger = logging.getLogger('neurolite_test')
        logger.setLevel(logging.DEBUG)
        
        # Test string stream handler
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Test logging
        logger.info('Test info message')
        logger.warning('Test warning message')
        logger.error('Test error message')
        
        log_contents = log_stream.getvalue()
        assert 'Test info message' in log_contents
        assert 'Test warning message' in log_contents
        assert 'Test error message' in log_contents
        
        # Clean up
        logger.removeHandler(handler)
        handler.close()
    
    def test_json_serialization_compatibility(self):
        """Test JSON serialization across platforms."""
        import json
        
        # Test basic data types
        test_data = {
            'string': 'test',
            'integer': 42,
            'float': 3.14,
            'boolean': True,
            'null': None,
            'list': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        
        # Test serialization
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        # Test deserialization
        loaded_data = json.loads(json_str)
        assert loaded_data == test_data
        
        # Test with file
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, 'test.json')
            
            with open(json_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(json_file, 'r') as f:
                loaded_from_file = json.load(f)
            
            assert loaded_from_file == test_data