"""
Logging infrastructure for NeuroLite.

Provides configurable logging with different levels, formatters,
and output destinations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import get_config, LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class NeuroLiteLogger:
    """Custom logger for NeuroLite with enhanced functionality."""
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """
        Initialize NeuroLite logger.
        
        Args:
            name: Logger name
            config: Logging configuration (uses global config if None)
        """
        self.name = name
        self.config = config or get_config().logging
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with configured handlers and formatters."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level.upper()))
            
            # Use colored formatter for console
            console_formatter = ColoredFormatter(self.config.format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_path:
            file_path = Path(self.config.file_path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level.upper()))
            
            # Use standard formatter for file
            file_formatter = logging.Formatter(self.config.format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)
    
    def log_training_start(self, model_name: str, dataset_info: Dict[str, Any]) -> None:
        """Log training start with context."""
        self.info(
            f"Starting training - Model: {model_name}, "
            f"Data shape: {dataset_info.get('shape', 'unknown')}, "
            f"Task: {dataset_info.get('task_type', 'unknown')}"
        )
    
    def log_training_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training progress."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch} - {metrics_str}")
    
    def log_training_complete(self, duration: float, final_metrics: Dict[str, float]) -> None:
        """Log training completion."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
        self.info(f"Training completed in {duration:.2f}s - Final metrics: {metrics_str}")
    
    def log_model_export(self, model_name: str, export_format: str, file_path: str) -> None:
        """Log model export."""
        self.info(f"Exported model '{model_name}' to {export_format} format: {file_path}")
    
    def log_api_start(self, host: str, port: int) -> None:
        """Log API server start."""
        self.info(f"Starting API server at http://{host}:{port}")
    
    def log_prediction_request(self, input_shape: tuple, processing_time: float) -> None:
        """Log prediction request."""
        self.debug(f"Prediction request - Input shape: {input_shape}, Processing time: {processing_time:.3f}s")


class LoggerManager:
    """Manager for NeuroLite loggers."""
    
    _loggers: Dict[str, NeuroLiteLogger] = {}
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[LoggingConfig] = None) -> NeuroLiteLogger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name
            config: Optional logging configuration
            
        Returns:
            NeuroLiteLogger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = NeuroLiteLogger(name, config)
        return cls._loggers[name]
    
    @classmethod
    def configure_all_loggers(cls, config: LoggingConfig) -> None:
        """
        Reconfigure all existing loggers with new configuration.
        
        Args:
            config: New logging configuration
        """
        for logger in cls._loggers.values():
            logger.config = config
            logger._logger = logger._setup_logger()
    
    @classmethod
    def shutdown_all_loggers(cls) -> None:
        """Shutdown all loggers and handlers."""
        for logger in cls._loggers.values():
            for handler in logger._logger.handlers:
                handler.close()
            logger._logger.handlers.clear()
        cls._loggers.clear()


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> NeuroLiteLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        config: Optional logging configuration
        
    Returns:
        NeuroLiteLogger instance
    """
    return LoggerManager.get_logger(name, config)


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        config: Logging configuration (uses global config if None)
    """
    if config is None:
        config = get_config().logging
    
    # Configure root logger to prevent other libraries from interfering
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Only show warnings and errors from other libraries
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure all NeuroLite loggers
    LoggerManager.configure_all_loggers(config)


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name()}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            logger.info("  GPU: Not available")
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")


def log_performance_metrics(
    operation: str,
    duration: float,
    memory_used: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        memory_used: Memory used in MB
        additional_metrics: Additional metrics to log
    """
    logger = get_logger(__name__)
    
    metrics = [f"Duration: {duration:.3f}s"]
    if memory_used is not None:
        metrics.append(f"Memory: {memory_used:.1f} MB")
    
    if additional_metrics:
        for key, value in additional_metrics.items():
            metrics.append(f"{key}: {value}")
    
    logger.info(f"Performance [{operation}] - {', '.join(metrics)}")


# Initialize logging on module import
setup_logging()