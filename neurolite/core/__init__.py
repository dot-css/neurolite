"""
Core utilities and infrastructure for NeuroLite.

This module provides the foundational components including configuration
management, logging, exception handling, and common utilities.
"""

from .config import (
    NeuroLiteConfig,
    LoggingConfig,
    TrainingConfig,
    ModelConfig,
    DataConfig,
    PerformanceConfig,
    DeploymentConfig,
    Environment,
    ConfigManager,
    get_config,
    update_config
)

from .exceptions import (
    NeuroLiteError,
    DataError,
    DataNotFoundError,
    DataFormatError,
    DataValidationError,
    ModelError,
    ModelNotFoundError,
    ModelCompatibilityError,
    ModelLoadError,
    TrainingError,
    TrainingConfigurationError,
    TrainingFailedError,
    EvaluationError,
    MetricError,
    DeploymentError,
    ExportError,
    APIServerError,
    ConfigurationError,
    InvalidConfigurationError,
    DependencyError,
    MissingDependencyError,
    ResourceError,
    InsufficientMemoryError,
    GPUError
)

from .logger import (
    NeuroLiteLogger,
    LoggerManager,
    get_logger,
    setup_logging,
    log_system_info,
    log_performance_metrics
)

from .utils import (
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

from .plugins import (
    PluginRegistry,
    PluginLoader,
    ModelPlugin,
    PreprocessorPlugin,
    PluginMetadata,
    PluginError,
    PluginValidationError,
    PluginLoadError,
    get_plugin_registry,
    register_model_plugin,
    register_preprocessor_plugin,
    load_plugin_from_file,
    load_plugins_from_directory,
    load_plugins_from_config
)

from .plugin_integration import (
    PluginIntegration,
    get_plugin_integration,
    auto_discover_and_load_plugins,
    initialize_plugin_system
)

from .plugin_templates import (
    PluginTemplateGenerator,
    create_plugin_template
)

from .performance import (
    LazyLoader,
    CacheManager,
    ParallelProcessor,
    GPUAccelerator,
    get_cache_manager,
    get_gpu_accelerator,
    lazy_load,
    cached,
    parallel_map,
    gpu_context
)

from .benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    PerformanceMonitor,
    BenchmarkRunner,
    get_benchmark_runner,
    benchmark_data_loading,
    benchmark_preprocessing,
    benchmark_model_inference
)

__all__ = [
    # Configuration
    'NeuroLiteConfig',
    'LoggingConfig',
    'TrainingConfig',
    'ModelConfig',
    'DataConfig',
    'PerformanceConfig',
    'DeploymentConfig',
    'Environment',
    'ConfigManager',
    'get_config',
    'update_config',
    
    # Exceptions
    'NeuroLiteError',
    'DataError',
    'DataNotFoundError',
    'DataFormatError',
    'DataValidationError',
    'ModelError',
    'ModelNotFoundError',
    'ModelCompatibilityError',
    'ModelLoadError',
    'TrainingError',
    'TrainingConfigurationError',
    'TrainingFailedError',
    'EvaluationError',
    'MetricError',
    'DeploymentError',
    'ExportError',
    'APIServerError',
    'ConfigurationError',
    'InvalidConfigurationError',
    'DependencyError',
    'MissingDependencyError',
    'ResourceError',
    'InsufficientMemoryError',
    'GPUError',
    
    # Logging
    'NeuroLiteLogger',
    'LoggerManager',
    'get_logger',
    'setup_logging',
    'log_system_info',
    'log_performance_metrics',
    
    # Utilities
    'ensure_dir',
    'get_file_hash',
    'get_size_mb',
    'format_duration',
    'format_bytes',
    'safe_import',
    'check_dependencies',
    'get_available_device',
    'get_memory_usage',
    'timer',
    'retry',
    'cache_result',
    'validate_input',
    'flatten_dict',
    'unflatten_dict',
    'get_class_from_string',
    'is_notebook',
    'setup_matplotlib_backend',
    
    # Plugin System
    'PluginRegistry',
    'PluginLoader',
    'ModelPlugin',
    'PreprocessorPlugin',
    'PluginMetadata',
    'PluginError',
    'PluginValidationError',
    'PluginLoadError',
    'get_plugin_registry',
    'register_model_plugin',
    'register_preprocessor_plugin',
    'load_plugin_from_file',
    'load_plugins_from_directory',
    'load_plugins_from_config',
    'PluginIntegration',
    'get_plugin_integration',
    'auto_discover_and_load_plugins',
    'initialize_plugin_system',
    'PluginTemplateGenerator',
    'create_plugin_template',
    
    # Performance Optimization
    'LazyLoader',
    'CacheManager',
    'ParallelProcessor',
    'GPUAccelerator',
    'get_cache_manager',
    'get_gpu_accelerator',
    'lazy_load',
    'cached',
    'parallel_map',
    'gpu_context',
    
    # Benchmarking
    'BenchmarkResult',
    'BenchmarkSuite',
    'PerformanceMonitor',
    'BenchmarkRunner',
    'get_benchmark_runner',
    'benchmark_data_loading',
    'benchmark_preprocessing',
    'benchmark_model_inference'
]