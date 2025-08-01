"""
Performance optimization utilities for NeuroLite.

Provides lazy loading, caching, parallel processing, and GPU acceleration
detection for improved performance and resource utilization.
"""

import os
import sys
import time
import pickle
import hashlib
import threading
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
from contextlib import contextmanager
import weakref

from .logger import get_logger
from .config import get_config
from .utils import ensure_dir, format_bytes, format_duration, get_memory_usage


logger = get_logger(__name__)


class LazyLoader:
    """
    Lazy loading utility for expensive objects like models and datasets.
    
    Objects are only loaded when first accessed, reducing memory usage
    and startup time.
    """
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        """
        Initialize lazy loader.
        
        Args:
            loader_func: Function to call for loading the object
            *args: Arguments to pass to loader function
            **kwargs: Keyword arguments to pass to loader function
        """
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._loaded_object = None
        self._lock = threading.Lock()
        self._load_time = None
    
    def __call__(self):
        """Load and return the object."""
        if self._loaded_object is None:
            with self._lock:
                if self._loaded_object is None:  # Double-check locking
                    start_time = time.time()
                    logger.debug(f"Lazy loading {self._loader_func.__name__}")
                    
                    self._loaded_object = self._loader_func(*self._args, **self._kwargs)
                    self._load_time = time.time() - start_time
                    
                    logger.debug(
                        f"Lazy loaded {self._loader_func.__name__} in "
                        f"{format_duration(self._load_time)}"
                    )
        
        return self._loaded_object
    
    @property
    def is_loaded(self) -> bool:
        """Check if object has been loaded."""
        return self._loaded_object is not None
    
    @property
    def load_time(self) -> Optional[float]:
        """Get load time in seconds."""
        return self._load_time
    
    def unload(self):
        """Unload the object to free memory."""
        with self._lock:
            if self._loaded_object is not None:
                logger.debug(f"Unloading {self._loader_func.__name__}")
                self._loaded_object = None
                self._load_time = None


class CacheManager:
    """
    Advanced caching system for preprocessed data and model artifacts.
    
    Supports memory and disk caching with TTL, size limits, and automatic
    cleanup of expired entries.
    """
    
    def __init__(self, 
                 cache_dir: Optional[Union[str, Path]] = None,
                 max_memory_size: int = 512 * 1024 * 1024,  # 512MB
                 max_disk_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
                 default_ttl: Optional[float] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum memory cache size in bytes
            max_disk_size: Maximum disk cache size in bytes
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir or get_config().data.cache_dir) / "performance_cache"
        ensure_dir(self.cache_dir)
        
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.default_ttl = default_ttl
        
        # Memory cache with weak references to allow garbage collection
        self._memory_cache: Dict[str, Any] = {}
        self._memory_metadata: Dict[str, Dict[str, Any]] = {}
        self._memory_lock = threading.RLock()
        
        # Disk cache metadata
        self._disk_metadata_file = self.cache_dir / "metadata.pkl"
        self._disk_metadata: Dict[str, Dict[str, Any]] = self._load_disk_metadata()
        self._disk_lock = threading.RLock()
        
        # Cleanup expired entries on initialization
        self._cleanup_expired()
    
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace."""
        return f"{namespace}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _load_disk_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load disk cache metadata."""
        if self._disk_metadata_file.exists():
            try:
                with open(self._disk_metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk cache metadata: {e}")
        return {}
    
    def _save_disk_metadata(self):
        """Save disk cache metadata."""
        try:
            with open(self._disk_metadata_file, 'wb') as f:
                pickle.dump(self._disk_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save disk cache metadata: {e}")
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if 'ttl' not in metadata or metadata['ttl'] is None:
            return False
        return time.time() > metadata['created_at'] + metadata['ttl']
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        # Clean memory cache
        with self._memory_lock:
            expired_keys = [
                key for key, metadata in self._memory_metadata.items()
                if self._is_expired(metadata)
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                del self._memory_metadata[key]
        
        # Clean disk cache
        with self._disk_lock:
            expired_keys = [
                key for key, metadata in self._disk_metadata.items()
                if self._is_expired(metadata)
            ]
            for key in expired_keys:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                del self._disk_metadata[key]
            
            if expired_keys:
                self._save_disk_metadata()
    
    def _evict_memory_cache(self):
        """Evict least recently used items from memory cache."""
        with self._memory_lock:
            if not self._memory_cache:
                return
            
            # Sort by last access time
            sorted_items = sorted(
                self._memory_metadata.items(),
                key=lambda x: x[1].get('last_access', 0)
            )
            
            # Remove oldest 25% of items
            num_to_remove = max(1, len(sorted_items) // 4)
            for key, _ in sorted_items[:num_to_remove]:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    del self._memory_metadata[key]
    
    def _evict_disk_cache(self):
        """Evict least recently used items from disk cache."""
        with self._disk_lock:
            if not self._disk_metadata:
                return
            
            # Calculate total disk usage
            total_size = sum(
                metadata.get('size', 0) 
                for metadata in self._disk_metadata.values()
            )
            
            if total_size <= self.max_disk_size:
                return
            
            # Sort by last access time
            sorted_items = sorted(
                self._disk_metadata.items(),
                key=lambda x: x[1].get('last_access', 0)
            )
            
            # Remove items until under size limit
            for key, metadata in sorted_items:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                del self._disk_metadata[key]
                total_size -= metadata.get('size', 0)
                
                if total_size <= self.max_disk_size * 0.8:  # Leave some headroom
                    break
            
            self._save_disk_metadata()
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_key(key, namespace)
        
        # Try memory cache first
        with self._memory_lock:
            if cache_key in self._memory_cache:
                metadata = self._memory_metadata[cache_key]
                if not self._is_expired(metadata):
                    metadata['last_access'] = time.time()
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    logger.debug(f"Memory cache hit for {key}")
                    return self._memory_cache[cache_key]
                else:
                    # Remove expired entry
                    del self._memory_cache[cache_key]
                    del self._memory_metadata[cache_key]
        
        # Try disk cache
        with self._disk_lock:
            if cache_key in self._disk_metadata:
                metadata = self._disk_metadata[cache_key]
                if not self._is_expired(metadata):
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'rb') as f:
                                value = pickle.load(f)
                            
                            metadata['last_access'] = time.time()
                            metadata['access_count'] = metadata.get('access_count', 0) + 1
                            self._save_disk_metadata()
                            
                            logger.debug(f"Disk cache hit for {key}")
                            return value
                        except Exception as e:
                            logger.warning(f"Failed to load from disk cache: {e}")
                            # Remove corrupted entry
                            cache_file.unlink(missing_ok=True)
                            del self._disk_metadata[cache_key]
                            self._save_disk_metadata()
                else:
                    # Remove expired entry
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    cache_file.unlink(missing_ok=True)
                    del self._disk_metadata[cache_key]
                    self._save_disk_metadata()
        
        return None
    
    def set(self, 
            key: str, 
            value: Any, 
            namespace: str = "default",
            ttl: Optional[float] = None,
            prefer_memory: bool = True) -> bool:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time-to-live in seconds
            prefer_memory: Whether to prefer memory cache
            
        Returns:
            True if successfully cached
        """
        cache_key = self._generate_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        # Estimate object size
        try:
            serialized = pickle.dumps(value)
            size = len(serialized)
        except Exception as e:
            logger.warning(f"Failed to serialize value for caching: {e}")
            return False
        
        metadata = {
            'created_at': time.time(),
            'last_access': time.time(),
            'access_count': 0,
            'size': size,
            'ttl': ttl,
            'namespace': namespace
        }
        
        # Try memory cache first if preferred and size allows
        if prefer_memory and size <= self.max_memory_size // 10:  # Max 10% of memory cache
            with self._memory_lock:
                # Check if we need to evict
                current_memory_usage = sum(
                    self._memory_metadata[k].get('size', 0)
                    for k in self._memory_cache
                )
                
                if current_memory_usage + size > self.max_memory_size:
                    self._evict_memory_cache()
                
                self._memory_cache[cache_key] = value
                self._memory_metadata[cache_key] = metadata
                logger.debug(f"Cached {key} in memory ({format_bytes(size)})")
                return True
        
        # Use disk cache
        with self._disk_lock:
            try:
                # Check if we need to evict
                current_disk_usage = sum(
                    self._disk_metadata[k].get('size', 0)
                    for k in self._disk_metadata
                )
                
                if current_disk_usage + size > self.max_disk_size:
                    self._evict_disk_cache()
                
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    f.write(serialized)
                
                self._disk_metadata[cache_key] = metadata
                self._save_disk_metadata()
                
                logger.debug(f"Cached {key} on disk ({format_bytes(size)})")
                return True
            except Exception as e:
                logger.warning(f"Failed to cache to disk: {e}")
                return False
    
    def clear(self, namespace: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            namespace: Namespace to clear (None for all)
        """
        # Clear memory cache
        with self._memory_lock:
            if namespace is None:
                self._memory_cache.clear()
                self._memory_metadata.clear()
            else:
                keys_to_remove = [
                    key for key, metadata in self._memory_metadata.items()
                    if metadata.get('namespace') == namespace
                ]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                    del self._memory_metadata[key]
        
        # Clear disk cache
        with self._disk_lock:
            if namespace is None:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    if cache_file.name != "metadata.pkl":
                        cache_file.unlink()
                self._disk_metadata.clear()
            else:
                keys_to_remove = [
                    key for key, metadata in self._disk_metadata.items()
                    if metadata.get('namespace') == namespace
                ]
                for key in keys_to_remove:
                    cache_file = self.cache_dir / f"{key}.pkl"
                    cache_file.unlink(missing_ok=True)
                    del self._disk_metadata[key]
            
            self._save_disk_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._memory_lock, self._disk_lock:
            memory_size = sum(
                metadata.get('size', 0)
                for metadata in self._memory_metadata.values()
            )
            disk_size = sum(
                metadata.get('size', 0)
                for metadata in self._disk_metadata.values()
            )
            
            return {
                'memory_entries': len(self._memory_cache),
                'memory_size': memory_size,
                'memory_size_formatted': format_bytes(memory_size),
                'memory_utilization': memory_size / self.max_memory_size,
                'disk_entries': len(self._disk_metadata),
                'disk_size': disk_size,
                'disk_size_formatted': format_bytes(disk_size),
                'disk_utilization': disk_size / self.max_disk_size,
                'total_entries': len(self._memory_cache) + len(self._disk_metadata),
                'total_size': memory_size + disk_size,
                'total_size_formatted': format_bytes(memory_size + disk_size)
            }


class ParallelProcessor:
    """
    Parallel processing utilities for data loading and preprocessing.
    
    Provides both thread-based and process-based parallelization with
    automatic worker count optimization and progress tracking.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (None for auto)
            use_processes: Whether to use processes instead of threads
            chunk_size: Chunk size for batch processing
        """
        if max_workers is None:
            if use_processes:
                max_workers = min(multiprocessing.cpu_count(), get_config().data.num_workers)
            else:
                max_workers = min(multiprocessing.cpu_count() * 2, get_config().data.num_workers * 2)
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size or max(1, 1000 // max_workers)
        
        logger.debug(
            f"Initialized ParallelProcessor with {max_workers} "
            f"{'processes' if use_processes else 'threads'}"
        )
    
    def map(self, 
            func: Callable,
            iterable: List[Any],
            progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """
        Apply function to iterable in parallel.
        
        Args:
            func: Function to apply
            iterable: Iterable to process
            progress_callback: Optional progress callback (completed, total)
            
        Returns:
            List of results
        """
        if len(iterable) == 0:
            return []
        
        if len(iterable) == 1 or self.max_workers == 1:
            # Single-threaded execution for small datasets
            results = []
            for i, item in enumerate(iterable):
                results.append(func(item))
                if progress_callback:
                    progress_callback(i + 1, len(iterable))
            return results
        
        # Parallel execution
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(iterable)
            }
            
            # Collect results in order
            results = [None] * len(iterable)
            completed = 0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    results[index] = None
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(iterable))
        
        return results
    
    def map_batches(self,
                   func: Callable,
                   iterable: List[Any],
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """
        Apply function to batches of items in parallel.
        
        Args:
            func: Function to apply to each batch
            iterable: Iterable to process
            progress_callback: Optional progress callback
            
        Returns:
            Flattened list of results
        """
        if len(iterable) <= self.chunk_size:
            return func(iterable)
        
        # Create batches
        batches = [
            iterable[i:i + self.chunk_size]
            for i in range(0, len(iterable), self.chunk_size)
        ]
        
        # Process batches in parallel
        batch_results = self.map(func, batches, progress_callback)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results


class GPUAccelerator:
    """
    GPU acceleration detection and automatic utilization.
    
    Detects available GPU devices and provides utilities for
    automatic device selection and memory management.
    """
    
    def __init__(self):
        """Initialize GPU accelerator."""
        self._device_info = None
        self._torch_available = None
        self._tensorflow_available = None
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """Detect GPU capabilities and available frameworks."""
        self._device_info = {
            'cuda_available': False,
            'mps_available': False,
            'cuda_devices': [],
            'mps_devices': [],
            'recommended_device': 'cpu'
        }
        
        # Check PyTorch CUDA
        try:
            import torch
            self._torch_available = True
            
            if torch.cuda.is_available():
                self._device_info['cuda_available'] = True
                self._device_info['cuda_devices'] = [
                    {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                        'memory_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    }
                    for i in range(torch.cuda.device_count())
                ]
                self._device_info['recommended_device'] = 'cuda'
            
            # Check PyTorch MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device_info['mps_available'] = True
                self._device_info['mps_devices'] = [{'id': 0, 'name': 'Apple Silicon GPU'}]
                if not self._device_info['cuda_available']:
                    self._device_info['recommended_device'] = 'mps'
        
        except ImportError:
            self._torch_available = False
        
        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            self._tensorflow_available = True
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and not self._device_info['cuda_available']:
                self._device_info['cuda_available'] = True
                self._device_info['recommended_device'] = 'cuda'
        
        except ImportError:
            self._tensorflow_available = False
        
        logger.debug(f"GPU detection complete: {self._device_info}")
    
    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return (self._device_info['cuda_available'] or 
                self._device_info['mps_available'])
    
    @property
    def recommended_device(self) -> str:
        """Get recommended device for computation."""
        config_device = get_config().model.device
        
        if config_device != 'auto':
            return config_device
        
        return self._device_info['recommended_device']
    
    @property
    def device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        return self._device_info.copy()
    
    def get_optimal_batch_size(self, 
                              model_size_mb: float,
                              input_size_mb: float,
                              safety_factor: float = 0.8) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_size_mb: Model size in MB
            input_size_mb: Single input size in MB
            safety_factor: Safety factor for memory usage
            
        Returns:
            Recommended batch size
        """
        if not self.is_available or not self._device_info['cuda_devices']:
            return 32  # Default batch size for CPU
        
        # Use the GPU with most free memory
        best_gpu = max(
            self._device_info['cuda_devices'],
            key=lambda x: x['memory_free']
        )
        
        available_memory_mb = best_gpu['memory_free'] / (1024 * 1024)
        usable_memory_mb = available_memory_mb * safety_factor
        
        # Reserve memory for model and overhead
        memory_for_batches = usable_memory_mb - model_size_mb - 100  # 100MB overhead
        
        if memory_for_batches <= 0:
            return 1
        
        batch_size = int(memory_for_batches / input_size_mb)
        return max(1, min(batch_size, 256))  # Cap at 256
    
    @contextmanager
    def device_context(self, device: Optional[str] = None):
        """
        Context manager for device selection.
        
        Args:
            device: Device to use (None for auto)
        """
        device = device or self.recommended_device
        
        if self._torch_available:
            import torch
            old_device = None
            try:
                if device != 'cpu':
                    old_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.set_device(0)
                    elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        pass  # MPS doesn't need explicit device setting
                
                yield device
            finally:
                if old_device is not None and torch.cuda.is_available():
                    torch.cuda.set_device(old_device)
        else:
            yield device
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage."""
        if not self.is_available:
            return
        
        if self._torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache")
            except Exception as e:
                logger.warning(f"Failed to clear PyTorch CUDA cache: {e}")
        
        if self._tensorflow_available:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                logger.debug("Cleared TensorFlow session")
            except Exception as e:
                logger.warning(f"Failed to clear TensorFlow session: {e}")


# Global instances
_cache_manager = None
_gpu_accelerator = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_gpu_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator


def lazy_load(loader_func: Callable, *args, **kwargs) -> LazyLoader:
    """
    Create a lazy loader for expensive objects.
    
    Args:
        loader_func: Function to call for loading
        *args: Arguments for loader function
        **kwargs: Keyword arguments for loader function
        
    Returns:
        LazyLoader instance
    """
    return LazyLoader(loader_func, *args, **kwargs)


def cached(namespace: str = "default", 
           ttl: Optional[float] = None,
           prefer_memory: bool = True):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl: Time-to-live in seconds
        prefer_memory: Whether to prefer memory cache
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            
            cache_manager = get_cache_manager()
            
            # Try to get from cache
            result = cache_manager.get(key, namespace)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, namespace, ttl, prefer_memory)
            
            return result
        
        return wrapper
    return decorator


def parallel_map(func: Callable,
                iterable: List[Any],
                max_workers: Optional[int] = None,
                use_processes: bool = False,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
    """
    Apply function to iterable in parallel.
    
    Args:
        func: Function to apply
        iterable: Iterable to process
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        progress_callback: Optional progress callback
        
    Returns:
        List of results
    """
    processor = ParallelProcessor(max_workers, use_processes)
    return processor.map(func, iterable, progress_callback)


@contextmanager
def gpu_context(device: Optional[str] = None):
    """
    Context manager for GPU device selection.
    
    Args:
        device: Device to use (None for auto)
    """
    accelerator = get_gpu_accelerator()
    with accelerator.device_context(device) as dev:
        yield dev