"""
Tests for core logging functionality.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from neurolite.core.logger import (
    ColoredFormatter,
    NeuroLiteLogger,
    LoggerManager,
    get_logger,
    setup_logging,
    log_system_info,
    log_performance_metrics
)
from neurolite.core.config import LoggingConfig


class TestColoredFormatter:
    """Test ColoredFormatter class."""
    
    def test_format_with_colors(self):
        """Test formatting with colors."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should contain ANSI color codes
        assert "\033[32m" in formatted  # Green for INFO
        assert "\033[0m" in formatted   # Reset
        assert "Test message" in formatted
    
    def test_format_different_levels(self):
        """Test formatting different log levels."""
        formatter = ColoredFormatter("%(levelname)s")
        
        levels = [
            (logging.DEBUG, "\033[36m"),    # Cyan
            (logging.INFO, "\033[32m"),     # Green
            (logging.WARNING, "\033[33m"),  # Yellow
            (logging.ERROR, "\033[31m"),    # Red
            (logging.CRITICAL, "\033[35m")  # Magenta
        ]
        
        for level, color_code in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=0,
                msg="Test",
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            assert color_code in formatted


class TestNeuroLiteLogger:
    """Test NeuroLiteLogger class."""
    
    def test_logger_creation(self):
        """Test logger creation with default config."""
        logger = NeuroLiteLogger("test_logger")
        
        assert logger.name == "test_logger"
        assert isinstance(logger.config, LoggingConfig)
        assert logger._logger.name == "test_logger"
    
    def test_logger_with_custom_config(self):
        """Test logger creation with custom config."""
        config = LoggingConfig(level="DEBUG", console_output=False)
        logger = NeuroLiteLogger("test_logger", config)
        
        assert logger.config.level == "DEBUG"
        assert logger.config.console_output is False
    
    def test_logging_methods(self):
        """Test different logging methods."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = NeuroLiteLogger("test_logger")
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            logger.exception("Exception message")
            
            mock_logger.debug.assert_called_with("Debug message")
            mock_logger.info.assert_called_with("Info message")
            mock_logger.warning.assert_called_with("Warning message")
            mock_logger.error.assert_called_with("Error message")
            mock_logger.critical.assert_called_with("Critical message")
            mock_logger.exception.assert_called_with("Exception message")
    
    def test_file_logging(self):
        """Test file logging configuration."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            config = LoggingConfig(
                level="INFO",
                file_path=temp_file.name,
                console_output=False
            )
            
            logger = NeuroLiteLogger("test_logger", config)
            logger.info("Test file logging")
            
            # Check that file handler was added
            file_handlers = [
                h for h in logger._logger.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
            
            # Clean up
            Path(temp_file.name).unlink()
    
    def test_training_logging_methods(self):
        """Test training-specific logging methods."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = NeuroLiteLogger("test_logger")
            
            # Test training start logging
            dataset_info = {"shape": (1000, 224, 224, 3), "task_type": "classification"}
            logger.log_training_start("resnet50", dataset_info)
            
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "resnet50" in call_args
            assert "classification" in call_args
            
            # Test training progress logging
            metrics = {"loss": 0.5, "accuracy": 0.85}
            logger.log_training_progress(10, metrics)
            
            call_args = mock_logger.info.call_args[0][0]
            assert "Epoch 10" in call_args
            assert "loss: 0.5000" in call_args
            assert "accuracy: 0.8500" in call_args
            
            # Test training completion logging
            final_metrics = {"loss": 0.2, "accuracy": 0.92}
            logger.log_training_complete(120.5, final_metrics)
            
            call_args = mock_logger.info.call_args[0][0]
            assert "120.50s" in call_args
            assert "loss: 0.2000" in call_args
            assert "accuracy: 0.9200" in call_args


class TestLoggerManager:
    """Test LoggerManager class."""
    
    def test_get_logger(self):
        """Test getting logger from manager."""
        logger1 = LoggerManager.get_logger("test_logger")
        logger2 = LoggerManager.get_logger("test_logger")
        
        assert isinstance(logger1, NeuroLiteLogger)
        assert logger1 is logger2  # Should return same instance
    
    def test_get_different_loggers(self):
        """Test getting different loggers."""
        logger1 = LoggerManager.get_logger("logger1")
        logger2 = LoggerManager.get_logger("logger2")
        
        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
    
    def test_configure_all_loggers(self):
        """Test configuring all loggers."""
        # Create some loggers
        logger1 = LoggerManager.get_logger("logger1")
        logger2 = LoggerManager.get_logger("logger2")
        
        original_level1 = logger1.config.level
        original_level2 = logger2.config.level
        
        # Configure all loggers
        new_config = LoggingConfig(level="ERROR")
        LoggerManager.configure_all_loggers(new_config)
        
        assert logger1.config.level == "ERROR"
        assert logger2.config.level == "ERROR"
    
    def test_shutdown_all_loggers(self):
        """Test shutting down all loggers."""
        # Create some loggers
        LoggerManager.get_logger("logger1")
        LoggerManager.get_logger("logger2")
        
        # Shutdown all loggers
        LoggerManager.shutdown_all_loggers()
        
        # Manager should have no loggers
        assert len(LoggerManager._loggers) == 0


class TestGlobalFunctions:
    """Test global logging functions."""
    
    def test_get_logger_function(self):
        """Test get_logger global function."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, NeuroLiteLogger)
        assert logger.name == "test_logger"
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        config = LoggingConfig(level="WARNING")
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            setup_logging(config)
            
            # Should configure root logger
            mock_root_logger.setLevel.assert_called_with(logging.WARNING)
    
    @patch('neurolite.core.logger.platform')
    @patch('neurolite.core.logger.psutil')
    def test_log_system_info(self, mock_psutil, mock_platform):
        """Test log_system_info function."""
        # Mock system information
        mock_platform.platform.return_value = "Linux-5.4.0"
        mock_platform.python_version.return_value = "3.9.0"
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        with patch('neurolite.core.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_system_info()
            
            # Should log system information
            assert mock_logger.info.call_count >= 4
            
            # Check some of the logged information
            call_args_list = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Linux-5.4.0" in arg for arg in call_args_list)
            assert any("3.9.0" in arg for arg in call_args_list)
            assert any("8" in arg for arg in call_args_list)
            assert any("16.0 GB" in arg for arg in call_args_list)
    
    def test_log_performance_metrics(self):
        """Test log_performance_metrics function."""
        with patch('neurolite.core.logger.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_performance_metrics(
                "test_operation",
                1.234,
                memory_used=512.5,
                additional_metrics={"accuracy": 0.95, "f1_score": 0.92}
            )
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            
            assert "test_operation" in call_args
            assert "1.234s" in call_args
            assert "512.5 MB" in call_args
            assert "accuracy: 0.95" in call_args
            assert "f1_score: 0.92" in call_args


if __name__ == '__main__':
    pytest.main([__file__])