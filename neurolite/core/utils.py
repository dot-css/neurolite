"""
Common utilities for NeuroLite.

Provides helper functions and utilities used across the library.
"""

import os
import sys
import time
import hashlib
import pickle
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
import importlib.util

from .logger import get_logger
from .exceptions import DependencyError, MissingDependencyError


logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_size_mb(path: Union[str, Path]) -> float:
    """
    Get size of file or directory in MB.
    
    Args:
        path: File or directory path
        
    Returns:
        Size in megabytes
    """
    path = Path(path)
    
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    else:
        return 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes in human-readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def safe_import(module_name: str, feature_name: str = None) -> Any:
    """
    Safely import a module with informative error handling.
    
    Args:
        module_name: Name of module to import
        feature_name: Name of feature that requires this module
        
    Returns:
        Imported module
        
    Raises:
        MissingDependencyError: If module cannot be imported
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        feature = feature_name or module_name
        install_cmd = f"pip install {module_name}"
        
        # Common package mappings
        install_mappings = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
        }
        
        if module_name in install_mappings:
            install_cmd = f"pip install {install_mappings[module_name]}"
        
        raise MissingDependencyError(
            dependency=module_name,
            feature=feature,
            install_command=install_cmd
        ) from e


def check_dependencies(dependencies: Dict[str, str]) -> Dict[str, bool]:
    """
    Check if dependencies are available.
    
    Args:
        dependencies: Dict mapping module names to feature names
        
    Returns:
        Dict mapping module names to availability status
    """
    results = {}
    for module_name, feature_name in dependencies.items():
        try:
            importlib.import_module(module_name)
            results[module_name] = True
        except ImportError:
            results[module_name] = False
            logger.debug(f"Optional dependency '{module_name}' not available for {feature_name}")
    
    return results


def get_available_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    
    # Check for MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    
    return 'cpu'


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict with memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}


@contextmanager
def timer(operation_name: str = "Operation"):
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        
    Yields:
        Dict that will contain timing information
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    timing_info = {}
    
    try:
        yield timing_info
    finally:
        end_time = time.time()
        end_memory = get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory['rss'] - start_memory['rss']
        
        timing_info.update({
            'duration': duration,
            'memory_delta': memory_delta,
            'start_memory': start_memory['rss'],
            'end_memory': end_memory['rss']
        })
        
        logger.debug(
            f"{operation_name} completed in {format_duration(duration)} "
            f"(Memory: {memory_delta:+.1f} MB)"
        )


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def cache_result(cache_dir: Optional[Union[str, Path]] = None, ttl: Optional[float] = None):
    """
    Decorator for caching function results to disk.
    
    Args:
        cache_dir: Directory to store cache files
        ttl: Time to live in seconds (None for no expiration)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = hashlib.md5(
                f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}".encode()
            ).hexdigest()
            
            # Determine cache directory
            if cache_dir is None:
                from .config import get_config
                cache_path = Path(get_config().data.cache_dir) / "function_cache"
            else:
                cache_path = Path(cache_dir)
            
            ensure_dir(cache_path)
            cache_file = cache_path / f"{cache_key}.pkl"
            
            # Check if cached result exists and is valid
            if cache_file.exists():
                try:
                    if ttl is None or (time.time() - cache_file.stat().st_mtime) < ttl:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                except Exception as e:
                    logger.warning(f"Failed to load cache for {func.__name__}: {e}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to cache result for {func.__name__}: {e}")
            
            return result
        
        return wrapper
    return decorator


def validate_input(
    value: Any,
    expected_type: type,
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allowed_values: Optional[List[Any]] = None
) -> None:
    """
    Validate input parameter.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allowed_values: List of allowed values
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"{name} must be one of {allowed_values}, got {value}")


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with nested keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def get_class_from_string(class_string: str) -> type:
    """
    Import and return class from string.
    
    Args:
        class_string: String in format 'module.submodule.ClassName'
        
    Returns:
        Class object
        
    Raises:
        ImportError: If module or class cannot be imported
    """
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def is_notebook() -> bool:
    """
    Check if running in Jupyter notebook.
    
    Returns:
        True if running in notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


def setup_matplotlib_backend():
    """Set up appropriate matplotlib backend based on environment."""
    try:
        import matplotlib
        
        if is_notebook():
            matplotlib.use('inline')
        elif 'DISPLAY' not in os.environ and sys.platform.startswith('linux'):
            matplotlib.use('Agg')  # Non-interactive backend for headless systems
        
        logger.debug(f"Matplotlib backend: {matplotlib.get_backend()}")
    except ImportError:
        pass