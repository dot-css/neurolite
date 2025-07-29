"""
Performance optimization utilities for NeuroLite library.

This module provides lazy loading, streaming, parallel processing, and memory-efficient
sampling strategies to optimize performance for large datasets.
"""

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Union, Iterator, Callable, Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import gc
from functools import wraps
import warnings

from .exceptions import ResourceLimitError, ProcessingError


class MemoryMonitor:
    """Monitor memory usage during processing."""
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        self.max_memory_mb = max_memory_mb
        self.initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
    def check_memory(self):
        """Check current memory usage against limits."""
        if self.max_memory_mb is not None:
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_used = current_memory - self.initial_memory
            
            if memory_used > self.max_memory_mb:
                raise ResourceLimitError(
                    "memory", 
                    f"Exceeded {self.max_memory_mb}MB limit",
                    memory_used
                )
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        current_memory = psutil.virtual_memory().used / (1024 * 1024)
        return current_memory - self.initial_memory


class LazyDataLoader:
    """Lazy loading for large datasets with streaming capabilities."""
    
    def __init__(self, data_source: Union[str, pd.DataFrame], chunk_size: int = 10000):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self._total_rows = None
        self._columns = None
        self._dtypes = None
        
    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over data chunks."""
        if isinstance(self.data_source, str):
            # File-based streaming
            file_path = Path(self.data_source)
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.csv':
                yield from pd.read_csv(self.data_source, chunksize=self.chunk_size)
            elif file_ext in ['.xlsx', '.xls']:
                # Excel doesn't support native chunking, load and split
                df = pd.read_excel(self.data_source)
                yield from self._chunk_dataframe(df)
            elif file_ext == '.parquet':
                # Parquet supports row group reading
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(self.data_source)
                for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                    yield batch.to_pandas()
            else:
                raise ValueError(f"Streaming not supported for {file_ext} files")
                
        elif isinstance(self.data_source, pd.DataFrame):
            # DataFrame chunking
            yield from self._chunk_dataframe(self.data_source)
        else:
            raise ValueError("Unsupported data source type for lazy loading")
    
    def _chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split DataFrame into chunks."""
        for i in range(0, len(df), self.chunk_size):
            yield df.iloc[i:i + self.chunk_size].copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset without loading all data."""
        if isinstance(self.data_source, str):
            # Sample first chunk to get metadata
            first_chunk = next(iter(self))
            self._columns = first_chunk.columns.tolist()
            self._dtypes = first_chunk.dtypes.to_dict()
            
            # Estimate total rows
            if self.data_source.endswith('.csv'):
                # Quick row count for CSV
                with open(self.data_source, 'r') as f:
                    self._total_rows = sum(1 for _ in f) - 1  # Subtract header
            else:
                # For other formats, we need to load to count
                self._total_rows = len(pd.read_csv(self.data_source) if self.data_source.endswith('.csv') else pd.read_excel(self.data_source))
                
        elif isinstance(self.data_source, pd.DataFrame):
            self._columns = self.data_source.columns.tolist()
            self._dtypes = self.data_source.dtypes.to_dict()
            self._total_rows = len(self.data_source)
        
        return {
            'total_rows': self._total_rows,
            'columns': self._columns,
            'dtypes': self._dtypes,
            'estimated_chunks': (self._total_rows // self.chunk_size) + 1
        }


class SamplingStrategy:
    """Memory-efficient sampling strategies for large datasets."""
    
    @staticmethod
    def stratified_sample(df: pd.DataFrame, target_col: str, sample_size: int, 
                         random_state: int = 42) -> pd.DataFrame:
        """
        Perform stratified sampling to maintain class distribution.
        
        Args:
            df: Input DataFrame
            target_col: Target column for stratification
            sample_size: Desired sample size
            random_state: Random seed for reproducibility
            
        Returns:
            Stratified sample DataFrame
        """
        try:
            # Calculate samples per class
            class_counts = df[target_col].value_counts()
            total_samples = len(df)
            
            sampled_dfs = []
            for class_val, class_count in class_counts.items():
                # Proportional sampling
                class_sample_size = int((class_count / total_samples) * sample_size)
                class_sample_size = max(1, class_sample_size)  # At least 1 sample per class
                
                class_df = df[df[target_col] == class_val]
                if len(class_df) >= class_sample_size:
                    sampled_class = class_df.sample(n=class_sample_size, random_state=random_state)
                else:
                    sampled_class = class_df  # Use all samples if class is small
                
                sampled_dfs.append(sampled_class)
            
            return pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
            
        except Exception as e:
            # Fallback to simple random sampling
            warnings.warn(f"Stratified sampling failed: {e}. Using random sampling.")
            return SamplingStrategy.random_sample(df, sample_size, random_state)
    
    @staticmethod
    def random_sample(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        """
        Perform simple random sampling.
        
        Args:
            df: Input DataFrame
            sample_size: Desired sample size
            random_state: Random seed for reproducibility
            
        Returns:
            Random sample DataFrame
        """
        if len(df) <= sample_size:
            return df.copy()
        
        return df.sample(n=sample_size, random_state=random_state)
    
    @staticmethod
    def systematic_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        Perform systematic sampling (every nth row).
        
        Args:
            df: Input DataFrame
            sample_size: Desired sample size
            
        Returns:
            Systematic sample DataFrame
        """
        if len(df) <= sample_size:
            return df.copy()
        
        step = len(df) // sample_size
        indices = range(0, len(df), step)[:sample_size]
        return df.iloc[list(indices)].copy()
    
    @staticmethod
    def reservoir_sample(data_iterator: Iterator[pd.DataFrame], sample_size: int, 
                        random_state: int = 42) -> pd.DataFrame:
        """
        Perform reservoir sampling for streaming data.
        
        Args:
            data_iterator: Iterator over data chunks
            sample_size: Desired sample size
            random_state: Random seed for reproducibility
            
        Returns:
            Reservoir sample DataFrame
        """
        np.random.seed(random_state)
        reservoir = []
        total_seen = 0
        
        for chunk in data_iterator:
            for _, row in chunk.iterrows():
                total_seen += 1
                
                if len(reservoir) < sample_size:
                    reservoir.append(row)
                else:
                    # Replace with probability sample_size/total_seen
                    replace_idx = np.random.randint(0, total_seen)
                    if replace_idx < sample_size:
                        reservoir[replace_idx] = row
        
        if not reservoir:
            raise ValueError("No data found in iterator")
        
        return pd.DataFrame(reservoir)
    
    @staticmethod
    def adaptive_sample_size(total_rows: int, total_cols: int, 
                           target_memory_mb: float = 500) -> int:
        """
        Calculate adaptive sample size based on data characteristics and memory constraints.
        
        Args:
            total_rows: Total number of rows in dataset
            total_cols: Total number of columns in dataset
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Recommended sample size
        """
        # Estimate memory per row (rough approximation)
        estimated_bytes_per_row = total_cols * 8  # Assume 8 bytes per value on average
        target_bytes = target_memory_mb * 1024 * 1024
        
        max_rows_for_memory = int(target_bytes / estimated_bytes_per_row)
        
        # Statistical considerations
        # For statistical significance, we need at least sqrt(total_rows) samples
        min_statistical_sample = max(1000, int(np.sqrt(total_rows)))
        
        # Balance between memory and statistical requirements
        recommended_sample = min(
            total_rows,  # Can't sample more than available
            max(min_statistical_sample, max_rows_for_memory)
        )
        
        return recommended_sample


class ParallelProcessor:
    """Parallel processing utilities for CPU-intensive operations."""
    
    def __init__(self, max_workers: Optional[int] = None, use_threads: bool = False):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to 8 to avoid overhead
        self.use_threads = use_threads
        
    def process_chunks_parallel(self, data_chunks: List[pd.DataFrame], 
                              processing_func: Callable, 
                              *args, **kwargs) -> List[Any]:
        """
        Process data chunks in parallel.
        
        Args:
            data_chunks: List of DataFrame chunks to process
            processing_func: Function to apply to each chunk
            *args, **kwargs: Additional arguments for processing function
            
        Returns:
            List of processing results
        """
        if len(data_chunks) == 1:
            # No need for parallel processing with single chunk
            return [processing_func(data_chunks[0], *args, **kwargs)]
        
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(processing_func, chunk, *args, **kwargs): i 
                    for i, chunk in enumerate(data_chunks)
                }
                
                # Collect results in order
                results = [None] * len(data_chunks)
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        results[chunk_idx] = future.result()
                    except Exception as e:
                        raise ProcessingError(f"parallel_processing_chunk_{chunk_idx}", e)
                
                return results
                
        except Exception as e:
            # Fallback to sequential processing
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            return [processing_func(chunk, *args, **kwargs) for chunk in data_chunks]
    
    def map_reduce_parallel(self, data_chunks: List[pd.DataFrame],
                           map_func: Callable, reduce_func: Callable,
                           *args, **kwargs) -> Any:
        """
        Apply map-reduce pattern with parallel processing.
        
        Args:
            data_chunks: List of DataFrame chunks to process
            map_func: Function to apply to each chunk (map phase)
            reduce_func: Function to combine results (reduce phase)
            *args, **kwargs: Additional arguments for map function
            
        Returns:
            Reduced result
        """
        # Map phase (parallel)
        map_results = self.process_chunks_parallel(data_chunks, map_func, *args, **kwargs)
        
        # Reduce phase (sequential)
        return reduce_func(map_results)


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Add performance metrics to result if it's a dict
            if isinstance(result, dict):
                result['_performance_metrics'] = {
                    'execution_time_seconds': execution_time,
                    'memory_used_mb': memory_used,
                    'function_name': func.__name__
                }
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log performance even for failed operations
            print(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper


class OptimizedDataProfiler:
    """
    Performance-optimized version of DataProfiler with lazy loading,
    streaming, and parallel processing capabilities.
    """
    
    def __init__(self, chunk_size: int = 10000, max_workers: Optional[int] = None,
                 target_memory_mb: float = 500, enable_parallel: bool = True):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.target_memory_mb = target_memory_mb
        self.enable_parallel = enable_parallel
        
        self.parallel_processor = ParallelProcessor(max_workers, use_threads=False)
        self.memory_monitor = MemoryMonitor(target_memory_mb * 2)  # Allow 2x target for processing
    
    def analyze_large_dataset(self, data_source: Union[str, pd.DataFrame],
                            sample_for_analysis: bool = True) -> Dict[str, Any]:
        """
        Analyze large datasets using streaming and parallel processing.
        
        Args:
            data_source: File path or DataFrame to analyze
            sample_for_analysis: Whether to use sampling for detailed analysis
            
        Returns:
            Analysis results with performance metrics
        """
        start_time = time.time()
        
        # Initialize lazy loader
        lazy_loader = LazyDataLoader(data_source, self.chunk_size)
        metadata = lazy_loader.get_metadata()
        
        # Determine if we need sampling
        total_rows = metadata['total_rows']
        if sample_for_analysis and total_rows > 50000:
            # Use sampling for detailed analysis
            sample_size = SamplingStrategy.adaptive_sample_size(
                total_rows, len(metadata['columns']), self.target_memory_mb
            )
            
            # Get representative sample
            sample_df = self._get_representative_sample(lazy_loader, sample_size)
            
            # Perform detailed analysis on sample
            detailed_results = self._analyze_sample(sample_df)
            
            # Perform streaming analysis for full dataset statistics
            streaming_results = self._analyze_streaming(lazy_loader)
            
            # Combine results
            results = {
                'metadata': metadata,
                'sample_analysis': detailed_results,
                'full_dataset_stats': streaming_results,
                'sampling_info': {
                    'sample_size': len(sample_df),
                    'total_size': total_rows,
                    'sampling_ratio': len(sample_df) / total_rows
                }
            }
        else:
            # Analyze full dataset with streaming
            results = {
                'metadata': metadata,
                'full_analysis': self._analyze_streaming(lazy_loader)
            }
        
        # Add performance metrics
        results['performance'] = {
            'total_execution_time': time.time() - start_time,
            'memory_used_mb': self.memory_monitor.get_memory_usage(),
            'processing_mode': 'sampled' if sample_for_analysis and total_rows > 50000 else 'full'
        }
        
        return results
    
    def _get_representative_sample(self, lazy_loader: LazyDataLoader, 
                                 sample_size: int) -> pd.DataFrame:
        """Get a representative sample from the dataset."""
        # Use reservoir sampling for streaming data
        return SamplingStrategy.reservoir_sample(iter(lazy_loader), sample_size)
    
    def _analyze_sample(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed analysis on a sample."""
        # This would integrate with existing DataProfiler components
        # For now, return basic analysis
        return {
            'shape': sample_df.shape,
            'dtypes': sample_df.dtypes.to_dict(),
            'missing_values': sample_df.isnull().sum().to_dict(),
            'basic_stats': sample_df.describe().to_dict() if len(sample_df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
    
    def _analyze_streaming(self, lazy_loader: LazyDataLoader) -> Dict[str, Any]:
        """Perform streaming analysis of the full dataset."""
        # Initialize accumulators
        total_rows = 0
        column_stats = {}
        
        # Process chunks
        for chunk in lazy_loader:
            self.memory_monitor.check_memory()
            
            total_rows += len(chunk)
            
            # Update column statistics
            for col in chunk.columns:
                if col not in column_stats:
                    column_stats[col] = {
                        'count': 0,
                        'null_count': 0,
                        'dtype': str(chunk[col].dtype)
                    }
                
                column_stats[col]['count'] += len(chunk)
                column_stats[col]['null_count'] += chunk[col].isnull().sum()
            
            # Force garbage collection to manage memory
            gc.collect()
        
        return {
            'total_rows_processed': total_rows,
            'column_statistics': column_stats
        }