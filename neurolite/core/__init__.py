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

__all__ = [
    # Configuration
    'NeuroLiteConfig',
    'LoggingConfig',
    'TrainingConfig',
    'ModelConfig',
    'DataConfig',
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
    'setup_matplotlib_backend'
]