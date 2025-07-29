"""
Tests for core utilities.
"""

import os
import sys
import time
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from neurolite.core.utils import (
    ensure_dir,
    get_file_hash,
    get_size_mb,
    format_duration,
    format_bytes,
    safe_import,
    check_dependencies,
    get_available_device,
    get_memory_usage,
    timer,
    retry,
    cache_result,
    validate_input,
    flatten_dict,
    unflatten_dict,
    get_class_from_string,
    is_notebook,
    setup_matplotlib_backend
)
from neurolite.core.exceptions import MissingDependencyError


class TestFileUtilities:
    """Test file-related utilities."""
    
    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new_dir" / "nested_dir"
            
            result = ensure_dir(test_path)
            
            assert test_path.exists()
            assert test_path.is_dir()
            assert result == test_path
    
    def test_ensure_dir_existing_directory(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            
            result = ensure_dir(test_path)
            
            assert result == test_path
            assert test_path.exists()
    
    def test_get_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test content")
            temp_file_path = temp_file.name
        
        try:
            # Calculate hash manually for comparison
            expected_hash = hashlib.md5(b"test content").hexdigest()
            
            result = get_file_hash(temp_file_path)
            
            assert result == expected_hash
        finally:
            Path(temp_file_path).unlink()
    
    def test_get_file_hash_different_algorithms(self):
        """Test file hash with different algorithms."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test")
            temp_file_path = temp_file.name
        
        try:
            md5_hash = get_file_hash(temp_file_path, "md5")
            sha1_hash = get_file_hash(temp_file_path, "sha1")
            sha256_hash = get_file_hash(temp_file_path, "sha256")
            
            assert len(md5_hash) == 32
            assert len(sha1_hash) == 40
            assert len(sha256_hash) == 64
            assert md5_hash != sha1_hash != sha256_hash
        finally:
            Path(temp_file_path).unlink()
    
    def test_get_size_mb_file(self):
        """Test getting file size in MB."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            # Write 1MB of data
            temp_file.write("x" * (1024 * 1024))
            temp_file_path = temp_file.name
        
        try:
            size = get_size_mb(temp_file_path)
            assert abs(size - 1.0) < 0.1  # Should be approximately 1MB
        finally:
            Path(temp_file_path).unlink()
    
    def test_get_size_mb_directory(self):
        """Test getting directory size in MB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files
            for i in range(3):
                file_path = Path(temp_dir) / f"file_{i}.txt"
                with open(file_path, 'w') as f:
                    f.write("x" * (512 * 1024))  # 0.5MB each
            
            size = get_size_mb(temp_dir)
            assert abs(size - 1.5) < 0.1  # Should be approximately 1.5MB
    
    def test_get_size_mb_nonexistent(self):
        """Test getting size of nonexistent path."""
        size = get_size_mb("/nonexistent/path")
        assert size == 0.0


class TestFormatUtilities:
    """Test formatting utilities."""
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(59.9) == "59.9s"
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        assert format_duration(60) == "1.0m"
        assert format_duration(150) == "2.5m"
        assert format_duration(3599) == "60.0m"
    
    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        assert format_duration(3600) == "1.0h"
        assert format_duration(7200) == "2.0h"
    
    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(512) == "512.0B"
        assert format_bytes(1024) == "1.0KB"
        assert format_bytes(1024 * 1024) == "1.0MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0TB"


class TestImportUtilities:
    """Test import-related utilities."""
    
    def test_safe_import_success(self):
        """Test successful import."""
        module = safe_import("os")
        assert module is os
    
    def test_safe_import_failure(self):
        """Test failed import."""
        with pytest.raises(MissingDependencyError) as exc_info:
            safe_import("nonexistent_module", "test_feature")
        
        error = exc_info.value
        assert error.context["dependency"] == "nonexistent_module"
        assert error.context["feature"] == "test_feature"
        assert "pip install nonexistent_module" in error.context["install_command"]
    
    def test_safe_import_with_mapping(self):
        """Test import with package name mapping."""
        with pytest.raises(MissingDependencyError) as exc_info:
            safe_import("cv2", "computer_vision")
        
        error = exc_info.value
        assert "pip install opencv-python" in error.context["install_command"]
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        dependencies = {
            "os": "operating_system",
            "nonexistent_module": "fake_feature"
        }
        
        results = check_dependencies(dependencies)
        
        assert results["os"] is True
        assert results["nonexistent_module"] is False
    
    @patch('neurolite.core.utils.importlib.import_module')
    def test_get_available_device_cuda(self, mock_import):
        """Test device detection with CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_import.return_value = mock_torch
        
        device = get_available_device()
        assert device == "cuda"
    
    @patch('neurolite.core.utils.importlib.import_module')
    def test_get_available_device_mps(self, mock_import):
        """Test device detection with MPS available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_import.return_value = mock_torch
        
        device = get_available_device()
        assert device == "mps"
    
    @patch('neurolite.core.utils.importlib.import_module')
    def test_get_available_device_cpu(self, mock_import):
        """Test device detection fallback to CPU."""
        mock_import.side_effect = ImportError()
        
        device = get_available_device()
        assert device == "cpu"


class TestMemoryUtilities:
    """Test memory-related utilities."""
    
    @patch('neurolite.core.utils.psutil')
    def test_get_memory_usage_success(self, mock_psutil):
        """Test successful memory usage retrieval."""
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.0
        mock_psutil.Process.return_value = mock_process
        
        result = get_memory_usage()
        
        assert result["rss"] == 100.0
        assert result["vms"] == 200.0
        assert result["percent"] == 5.0
    
    @patch('neurolite.core.utils.psutil', None)
    def test_get_memory_usage_no_psutil(self):
        """Test memory usage when psutil is not available."""
        result = get_memory_usage()
        
        assert result["rss"] == 0.0
        assert result["vms"] == 0.0
        assert result["percent"] == 0.0


class TestTimerUtility:
    """Test timer context manager."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        with timer("test_operation") as timing_info:
            time.sleep(0.1)
        
        assert "duration" in timing_info
        assert timing_info["duration"] >= 0.1
        assert "memory_delta" in timing_info
        assert "start_memory" in timing_info
        assert "end_memory" in timing_info


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_retry_success_first_attempt(self):
        """Test retry with success on first attempt."""
        call_count = 0
        
        @retry(max_attempts=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test retry with success after failures."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_all_attempts_fail(self):
        """Test retry when all attempts fail."""
        call_count = 0
        
        @retry(max_attempts=2, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError, match="Persistent failure"):
            test_function()
        
        assert call_count == 2


class TestCacheDecorator:
    """Test cache decorator."""
    
    def test_cache_result_basic(self):
        """Test basic cache functionality."""
        call_count = 0
        
        @cache_result()
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different argument should execute function
        result3 = test_function(10)
        assert result3 == 20
        assert call_count == 2


class TestValidationUtilities:
    """Test validation utilities."""
    
    def test_validate_input_success(self):
        """Test successful input validation."""
        # Should not raise any exception
        validate_input(5, int, "test_param")
        validate_input("hello", str, "test_param")
        validate_input([1, 2, 3], list, "test_param")
    
    def test_validate_input_type_error(self):
        """Test validation with wrong type."""
        with pytest.raises(ValueError, match="test_param must be of type int"):
            validate_input("5", int, "test_param")
    
    def test_validate_input_min_value(self):
        """Test validation with minimum value."""
        validate_input(10, int, "test_param", min_value=5)
        
        with pytest.raises(ValueError, match="test_param must be >= 5"):
            validate_input(3, int, "test_param", min_value=5)
    
    def test_validate_input_max_value(self):
        """Test validation with maximum value."""
        validate_input(10, int, "test_param", max_value=15)
        
        with pytest.raises(ValueError, match="test_param must be <= 15"):
            validate_input(20, int, "test_param", max_value=15)
    
    def test_validate_input_allowed_values(self):
        """Test validation with allowed values."""
        validate_input("red", str, "color", allowed_values=["red", "green", "blue"])
        
        with pytest.raises(ValueError, match="color must be one of"):
            validate_input("yellow", str, "color", allowed_values=["red", "green", "blue"])


class TestDictUtilities:
    """Test dictionary utilities."""
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = flatten_dict(nested_dict)
        
        expected = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }
        
        assert flattened == expected
    
    def test_unflatten_dict(self):
        """Test dictionary unflattening."""
        flattened_dict = {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }
        
        unflattened = unflatten_dict(flattened_dict)
        
        expected = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        assert unflattened == expected
    
    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten and unflatten are inverse operations."""
        original = {
            "level1": {
                "level2": {
                    "level3": "value"
                },
                "other": 42
            },
            "top_level": "test"
        }
        
        flattened = flatten_dict(original)
        restored = unflatten_dict(flattened)
        
        assert restored == original


class TestClassUtilities:
    """Test class-related utilities."""
    
    def test_get_class_from_string(self):
        """Test getting class from string."""
        cls = get_class_from_string("pathlib.Path")
        assert cls is Path
    
    def test_get_class_from_string_invalid(self):
        """Test getting class from invalid string."""
        with pytest.raises((ImportError, AttributeError)):
            get_class_from_string("nonexistent.module.Class")


class TestEnvironmentUtilities:
    """Test environment detection utilities."""
    
    @patch('neurolite.core.utils.get_ipython')
    def test_is_notebook_true(self, mock_get_ipython):
        """Test notebook detection when in notebook."""
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = 'ZMQInteractiveShell'
        mock_get_ipython.return_value = mock_ipython
        
        assert is_notebook() is True
    
    @patch('neurolite.core.utils.get_ipython')
    def test_is_notebook_false(self, mock_get_ipython):
        """Test notebook detection when not in notebook."""
        mock_get_ipython.return_value = None
        
        assert is_notebook() is False
    
    def test_is_notebook_no_ipython(self):
        """Test notebook detection when IPython is not available."""
        with patch.dict('sys.modules', {'IPython': None}):
            assert is_notebook() is False
    
    @patch('neurolite.core.utils.matplotlib')
    @patch('neurolite.core.utils.is_notebook')
    def test_setup_matplotlib_backend_notebook(self, mock_is_notebook, mock_matplotlib):
        """Test matplotlib backend setup in notebook."""
        mock_is_notebook.return_value = True
        
        setup_matplotlib_backend()
        
        mock_matplotlib.use.assert_called_with('inline')
    
    @patch('neurolite.core.utils.matplotlib')
    @patch('neurolite.core.utils.is_notebook')
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_matplotlib_backend_headless_linux(self, mock_is_notebook, mock_matplotlib):
        """Test matplotlib backend setup on headless Linux."""
        mock_is_notebook.return_value = False
        
        with patch('sys.platform', 'linux'):
            setup_matplotlib_backend()
            mock_matplotlib.use.assert_called_with('Agg')


if __name__ == '__main__':
    pytest.main([__file__])