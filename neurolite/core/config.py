"""
Configuration management system for NeuroLite.

Provides environment-specific settings and configuration management
with intelligent defaults and validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class TrainingConfig:
    """Default training configuration."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration settings."""
    cache_dir: str = "~/.neurolite/models"
    auto_download: bool = True
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: str = "float32"  # float32, float16, mixed


@dataclass
class DataConfig:
    """Data processing configuration."""
    cache_dir: str = "~/.neurolite/data"
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    # Caching settings
    enable_caching: bool = True
    memory_cache_size: int = 512 * 1024 * 1024  # 512MB
    disk_cache_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    cache_ttl: Optional[int] = None  # No expiration by default
    
    # Parallel processing settings
    max_workers: Optional[int] = None  # Auto-detect
    use_processes: bool = False  # Use threads by default
    chunk_size: Optional[int] = None  # Auto-calculate
    
    # GPU settings
    auto_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    allow_growth: bool = True
    
    # Lazy loading settings
    lazy_load_models: bool = True
    lazy_load_datasets: bool = True
    
    # Benchmarking settings
    benchmark_warmup_runs: int = 1
    benchmark_measurement_runs: int = 3
    benchmark_results_dir: Optional[str] = None


@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    max_request_size: int = 16 * 1024 * 1024  # 16MB


@dataclass
class NeuroLiteConfig:
    """Main configuration class for NeuroLite."""
    environment: Environment = Environment.DEVELOPMENT
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Additional settings
    debug: bool = False
    verbose: bool = False
    project_root: str = field(default_factory=lambda: str(Path.cwd()))
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Expand user paths
        self.model.cache_dir = os.path.expanduser(self.model.cache_dir)
        self.data.cache_dir = os.path.expanduser(self.data.cache_dir)
        
        # Create cache directories if they don't exist
        Path(self.model.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Environment-specific adjustments
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.logging.level = "DEBUG"
        elif self.environment == Environment.TESTING:
            self.logging.level = "WARNING"
            self.training.epochs = 2  # Faster testing
        elif self.environment == Environment.PRODUCTION:
            self.debug = False
            self.logging.level = "INFO"


class ConfigManager:
    """Configuration manager for loading and managing settings."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[NeuroLiteConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self._config = self._load_config()
    
    @property
    def config(self) -> NeuroLiteConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> NeuroLiteConfig:
        """Load configuration from various sources."""
        config = NeuroLiteConfig()
        
        # Load from environment variables
        self._load_from_env(config)
        
        # Load from config files
        config_file = self._find_config_file()
        if config_file:
            self._load_from_file(config, config_file)
        
        return config
    
    def _load_from_env(self, config: NeuroLiteConfig) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            'NEUROLITE_ENV': ('environment', lambda x: Environment(x)),
            'NEUROLITE_DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'NEUROLITE_VERBOSE': ('verbose', lambda x: x.lower() == 'true'),
            'NEUROLITE_LOG_LEVEL': ('logging.level', str),
            'NEUROLITE_BATCH_SIZE': ('training.batch_size', int),
            'NEUROLITE_EPOCHS': ('training.epochs', int),
            'NEUROLITE_LEARNING_RATE': ('training.learning_rate', float),
            'NEUROLITE_DEVICE': ('model.device', str),
            'NEUROLITE_CACHE_DIR': ('model.cache_dir', str),
            'NEUROLITE_NUM_WORKERS': ('data.num_workers', int),
            'NEUROLITE_ENABLE_CACHING': ('performance.enable_caching', lambda x: x.lower() == 'true'),
            'NEUROLITE_MEMORY_CACHE_SIZE': ('performance.memory_cache_size', int),
            'NEUROLITE_DISK_CACHE_SIZE': ('performance.disk_cache_size', int),
            'NEUROLITE_MAX_WORKERS': ('performance.max_workers', int),
            'NEUROLITE_USE_PROCESSES': ('performance.use_processes', lambda x: x.lower() == 'true'),
            'NEUROLITE_AUTO_GPU': ('performance.auto_gpu', lambda x: x.lower() == 'true'),
            'NEUROLITE_LAZY_LOAD_MODELS': ('performance.lazy_load_models', lambda x: x.lower() == 'true'),
        }
        
        for env_var, (attr_path, converter) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_attr(config, attr_path, converted_value)
                except (ValueError, TypeError) as e:
                    # Log warning but continue with default value
                    pass
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations."""
        config_names = [
            'neurolite.yaml',
            'neurolite.yml',
            'neurolite.json',
            '.neurolite.yaml',
            '.neurolite.yml',
            '.neurolite.json'
        ]
        
        search_paths = [
            Path.cwd(),
            Path.cwd() / '.neurolite',
            Path.home() / '.neurolite',
            Path.home()
        ]
        
        for path in search_paths:
            for name in config_names:
                config_file = path / name
                if config_file.exists():
                    return config_file
        
        return None
    
    def _load_from_file(self, config: NeuroLiteConfig, config_file: Path) -> None:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    data = json.load(f)
                else:
                    return
            
            if data:
                self._update_config_from_dict(config, data)
        except Exception as e:
            # Log warning but continue with current config
            pass
    
    def _update_config_from_dict(self, config: NeuroLiteConfig, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    # Handle nested configuration
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
    
    def _set_nested_attr(self, obj: Any, attr_path: str, value: Any) -> None:
        """Set nested attribute using dot notation."""
        attrs = attr_path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def save_config(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        file_path = Path(file_path)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(self._config)
        
        with open(file_path, 'w') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif file_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: NeuroLiteConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif hasattr(field_value, '__dict__') and not isinstance(field_value, type):
                # Handle nested config objects (dataclass instances)
                nested_dict = {}
                for nested_key, nested_value in field_value.__dict__.items():
                    if isinstance(nested_value, Enum):
                        nested_dict[nested_key] = nested_value.value
                    else:
                        nested_dict[nested_key] = nested_value
                result[field_name] = nested_dict
            elif isinstance(field_value, type):
                # Skip type objects (like enum classes)
                continue
            else:
                result[field_name] = field_value
        return result


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> NeuroLiteConfig:
    """Get the current configuration."""
    return config_manager.config


def update_config(**kwargs) -> None:
    """Update the current configuration."""
    config_manager.update_config(**kwargs)