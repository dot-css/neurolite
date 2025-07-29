"""
Tests for core configuration management.
"""

import os
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from neurolite.core.config import (
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


class TestNeuroLiteConfig:
    """Test NeuroLiteConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NeuroLiteConfig()
        
        assert config.environment == Environment.DEVELOPMENT
        # Note: debug becomes True after __post_init__ for DEVELOPMENT environment
        assert config.debug is True  # Changed from False to True
        assert config.verbose is False
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.deployment, DeploymentConfig)
    
    def test_post_init_development(self):
        """Test post-initialization for development environment."""
        config = NeuroLiteConfig(environment=Environment.DEVELOPMENT)
        
        assert config.debug is True
        assert config.logging.level == "DEBUG"
    
    def test_post_init_testing(self):
        """Test post-initialization for testing environment."""
        config = NeuroLiteConfig(environment=Environment.TESTING)
        
        assert config.logging.level == "WARNING"
        assert config.training.epochs == 2
    
    def test_post_init_production(self):
        """Test post-initialization for production environment."""
        config = NeuroLiteConfig(environment=Environment.PRODUCTION)
        
        assert config.debug is False
        assert config.logging.level == "INFO"
    
    def test_cache_directories_created(self):
        """Test that cache directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_cache = os.path.join(temp_dir, "models")
            data_cache = os.path.join(temp_dir, "data")
            
            config = NeuroLiteConfig()
            config.model.cache_dir = model_cache
            config.data.cache_dir = data_cache
            config.__post_init__()
            
            assert Path(model_cache).exists()
            assert Path(data_cache).exists()


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        
        assert manager1 is manager2
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.config
        
        assert isinstance(config, NeuroLiteConfig)
    
    @patch.dict(os.environ, {
        'NEUROLITE_ENV': 'production',
        'NEUROLITE_DEBUG': 'true',
        'NEUROLITE_LOG_LEVEL': 'ERROR',
        'NEUROLITE_BATCH_SIZE': '64',
        'NEUROLITE_EPOCHS': '200',
        'NEUROLITE_LEARNING_RATE': '0.01'
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        manager = ConfigManager()
        config = manager._load_config()
        
        assert config.environment == Environment.PRODUCTION
        assert config.debug is True
        assert config.logging.level == "ERROR"
        assert config.training.batch_size == 64
        assert config.training.epochs == 200
        assert config.training.learning_rate == 0.01
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'debug': True,
            'logging': {
                'level': 'WARNING',
                'console_output': False
            },
            'training': {
                'batch_size': 128,
                'epochs': 50
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            config = NeuroLiteConfig()
            manager._load_from_file(config, config_file)
            
            assert config.debug is True
            assert config.logging.level == 'WARNING'
            assert config.logging.console_output is False
            assert config.training.batch_size == 128
            assert config.training.epochs == 50
        finally:
            config_file.unlink()
    
    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            'verbose': True,
            'model': {
                'device': 'cuda',
                'precision': 'float16'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            config = NeuroLiteConfig()
            manager._load_from_file(config, config_file)
            
            assert config.verbose is True
            assert config.model.device == 'cuda'
            assert config.model.precision == 'float16'
        finally:
            config_file.unlink()
    
    def test_update_config(self):
        """Test updating configuration."""
        manager = ConfigManager()
        original_debug = manager.config.debug
        
        manager.update_config(debug=not original_debug)
        
        assert manager.config.debug == (not original_debug)
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            manager.save_config(config_file)
            
            assert config_file.exists()
            
            with open(config_file, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert 'environment' in saved_data
            assert 'logging' in saved_data
            assert 'training' in saved_data
        finally:
            config_file.unlink()
    
    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = Path(f.name)
        
        try:
            manager = ConfigManager()
            manager.save_config(config_file)
            
            assert config_file.exists()
            
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            
            assert 'environment' in saved_data
            assert 'logging' in saved_data
            assert 'training' in saved_data
        finally:
            config_file.unlink()


class TestConfigComponents:
    """Test individual configuration components."""
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.console_output is True
        assert config.file_path is None
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.early_stopping is True
        assert config.patience == 10
        assert config.validation_split == 0.2
        assert config.test_split == 0.1
        assert config.random_seed == 42
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.cache_dir == "~/.neurolite/models"
        assert config.auto_download is True
        assert config.device == "auto"
        assert config.precision == "float32"
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        
        assert config.cache_dir == "~/.neurolite/data"
        assert config.max_cache_size == 1024 * 1024 * 1024
        assert config.num_workers == 4
        assert config.prefetch_factor == 2
        assert config.pin_memory is True
    
    def test_deployment_config_defaults(self):
        """Test DeploymentConfig default values."""
        config = DeploymentConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.timeout == 30
        assert config.max_request_size == 16 * 1024 * 1024


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        
        assert isinstance(config, NeuroLiteConfig)
    
    def test_update_config(self):
        """Test update_config function."""
        original_config = get_config()
        original_debug = original_config.debug
        
        update_config(debug=not original_debug)
        
        updated_config = get_config()
        assert updated_config.debug == (not original_debug)
        
        # Reset to original value
        update_config(debug=original_debug)


if __name__ == '__main__':
    pytest.main([__file__])