"""
Unit tests for NeuroLite CLI functionality.

Tests CLI commands for training, evaluation, deployment, and configuration management.
"""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from neurolite.cli.main import cli, _load_config_file, _save_model, _load_model
from neurolite.core.exceptions import NeuroLiteError
from neurolite.training import TrainedModel


class TestConfigFileHandling:
    """Test configuration file loading and validation."""
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        config_data = {
            'train': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'model': {
                'type': 'resnet18'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = _load_config_file(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        config_data = {
            'train': {
                'epochs': 50,
                'batch_size': 16
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = _load_config_file(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(Exception):
            _load_config_file('nonexistent_config.yaml')
    
    def test_load_invalid_format_config(self):
        """Test loading configuration file with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid config content")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):
                _load_config_file(config_path)
        finally:
            Path(config_path).unlink()


class TestModelSaveLoad:
    """Test model saving and loading functionality."""
    
    def test_save_and_load_model(self):
        """Test saving and loading a trained model."""
        # Create a simple object that can be pickled
        class MockTrainedModel:
            def __init__(self):
                self.model = "test_model"
                self.config = {"epochs": 10}
        
        mock_model = MockTrainedModel()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Test saving
            _save_model(mock_model, model_path)
            assert Path(model_path).exists()
            
            # Test loading
            loaded_model = _load_model(model_path)
            assert loaded_model.model == "test_model"
            assert loaded_model.config == {"epochs": 10}
        finally:
            Path(model_path).unlink()
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        with pytest.raises(Exception):
            _load_model('nonexistent_model.pkl')


class TestCLICommands:
    """Test CLI command functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'NeuroLite' in result.output
        assert 'Train and deploy machine learning models' in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
    
    @patch('neurolite.cli.main.train')
    def test_train_command_basic(self, mock_train):
        """Test basic train command."""
        runner = CliRunner()
        
        # Mock the train function
        mock_trained_model = Mock(spec=TrainedModel)
        mock_trained_model.evaluation_results = Mock()
        mock_trained_model.evaluation_results.primary_metric = 0.85
        mock_train.return_value = mock_trained_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            result = runner.invoke(cli, [
                'train',
                str(data_path),
                '--model', 'resnet18',
                '--task', 'classification'
            ])
            
            assert result.exit_code == 0
            assert 'Training completed successfully' in result.output
            mock_train.assert_called_once()
    
    @patch('neurolite.cli.main.train')
    def test_train_command_with_config(self, mock_train):
        """Test train command with configuration file."""
        runner = CliRunner()
        
        # Mock the train function
        mock_trained_model = Mock(spec=TrainedModel)
        mock_train.return_value = mock_trained_model
        
        config_data = {
            'train': {
                'epochs': 50,
                'batch_size': 16,
                'learning_rate': 0.01
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            config_path = Path(temp_dir) / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            result = runner.invoke(cli, [
                '--config', str(config_path),
                'train',
                str(data_path)
            ])
            
            assert result.exit_code == 0
            mock_train.assert_called_once()
            
            # Check that config was loaded
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['epochs'] == 50
            assert call_kwargs['batch_size'] == 16
    
    @patch('neurolite.cli.main.train')
    def test_train_command_with_output(self, mock_train):
        """Test train command with model output."""
        runner = CliRunner()
        
        # Mock the train function
        mock_trained_model = Mock(spec=TrainedModel)
        mock_train.return_value = mock_trained_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            output_path = Path(temp_dir) / "model.pkl"
            
            result = runner.invoke(cli, [
                'train',
                str(data_path),
                '--output', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert 'Model saved to' in result.output
            mock_train.assert_called_once()
    
    @patch('neurolite.cli.main.train')
    def test_train_command_failure(self, mock_train):
        """Test train command with training failure."""
        runner = CliRunner()
        
        # Mock the train function to raise an error
        mock_train.side_effect = NeuroLiteError("Training failed")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            result = runner.invoke(cli, [
                'train',
                str(data_path)
            ])
            
            assert result.exit_code == 1
            assert 'Error: Training failed' in result.output
    
    @patch('neurolite.cli.main._load_model')
    def test_evaluate_command(self, mock_load_model):
        """Test evaluate command."""
        runner = CliRunner()
        
        # Mock the model loading
        mock_model = Mock(spec=TrainedModel)
        mock_load_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            model_path.touch()
            
            test_data_path = Path(temp_dir) / "test_data"
            test_data_path.mkdir()
            
            result = runner.invoke(cli, [
                'evaluate',
                str(model_path),
                str(test_data_path),
                '--metrics', 'accuracy', 'f1'
            ])
            
            assert result.exit_code == 0
            assert 'Evaluation completed' in result.output
            assert 'accuracy:' in result.output
            mock_load_model.assert_called_once()
    
    @patch('neurolite.cli.main._load_model')
    def test_evaluate_command_with_output(self, mock_load_model):
        """Test evaluate command with output file."""
        runner = CliRunner()
        
        # Mock the model loading
        mock_model = Mock(spec=TrainedModel)
        mock_load_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            model_path.touch()
            
            test_data_path = Path(temp_dir) / "test_data"
            test_data_path.mkdir()
            
            output_path = Path(temp_dir) / "results.json"
            
            result = runner.invoke(cli, [
                'evaluate',
                str(model_path),
                str(test_data_path),
                '--output', str(output_path),
                '--format', 'json'
            ])
            
            assert result.exit_code == 0
            assert 'Results saved to' in result.output
            assert output_path.exists()
    
    @patch('neurolite.cli.main._load_model')
    def test_deploy_command_api(self, mock_load_model):
        """Test deploy command for API deployment."""
        runner = CliRunner()
        
        # Mock the model loading
        mock_model = Mock(spec=TrainedModel)
        mock_load_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            model_path.touch()
            
            result = runner.invoke(cli, [
                'deploy',
                str(model_path),
                '--format', 'api',
                '--port', '8080'
            ])
            
            assert result.exit_code == 0
            assert 'API server would be available' in result.output
            assert '8080' in result.output
            mock_load_model.assert_called_once()
    
    @patch('neurolite.cli.main._load_model')
    def test_deploy_command_export(self, mock_load_model):
        """Test deploy command for model export."""
        runner = CliRunner()
        
        # Mock the model loading
        mock_model = Mock(spec=TrainedModel)
        mock_load_model.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            model_path.touch()
            
            output_path = Path(temp_dir) / "model.onnx"
            
            result = runner.invoke(cli, [
                'deploy',
                str(model_path),
                '--format', 'onnx',
                '--output', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert 'Model exported to' in result.output
            mock_load_model.assert_called_once()
    
    @patch('neurolite.cli.main.get_config')
    @patch('neurolite.cli.main.log_system_info')
    def test_info_command(self, mock_log_system_info, mock_get_config):
        """Test info command."""
        runner = CliRunner()
        
        # Mock configuration
        mock_config = Mock()
        mock_config.environment.value = "development"
        mock_config.debug = False
        mock_config.model.cache_dir = "/tmp/models"
        mock_config.data.cache_dir = "/tmp/data"
        mock_get_config.return_value = mock_config
        
        result = runner.invoke(cli, ['info'])
        
        assert result.exit_code == 0
        assert 'NeuroLite System Information' in result.output
        assert 'Environment: development' in result.output
        mock_log_system_info.assert_called_once()
    
    @patch('neurolite.cli.main.get_model_registry')
    def test_list_models_command(self, mock_get_registry):
        """Test list-models command."""
        runner = CliRunner()
        
        # Mock model registry
        mock_registry = Mock()
        mock_registry.list_models.return_value = ['resnet18', 'bert', 'xgboost']
        mock_get_registry.return_value = mock_registry
        
        result = runner.invoke(cli, ['list-models'])
        
        assert result.exit_code == 0
        assert 'Available Models:' in result.output
        assert 'resnet18' in result.output
        assert 'bert' in result.output
        assert 'xgboost' in result.output
        assert 'Total: 3 models' in result.output
    
    def test_init_config_command(self):
        """Test init-config command."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.yaml"
            
            result = runner.invoke(cli, [
                'init-config',
                '--output', str(output_path),
                '--task', 'classification'
            ])
            
            assert result.exit_code == 0
            assert 'Example configuration created' in result.output
            assert output_path.exists()
            
            # Verify config content
            with open(output_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'train' in config
            assert 'data' in config
            assert 'model' in config
            assert config['model']['task'] == 'classification'
    
    def test_validate_config_command_valid(self):
        """Test validate-config command with valid configuration."""
        runner = CliRunner()
        
        config_data = {
            'train': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = runner.invoke(cli, ['validate-config', config_path])
            
            assert result.exit_code == 0
            assert 'Configuration is valid' in result.output
        finally:
            Path(config_path).unlink()
    
    def test_validate_config_command_invalid(self):
        """Test validate-config command with invalid configuration."""
        runner = CliRunner()
        
        config_data = {
            'train': {
                'epochs': 'invalid',  # Should be integer
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = runner.invoke(cli, ['validate-config', config_path])
            
            assert result.exit_code == 1
            assert 'Configuration validation failed' in result.output
            assert 'train.epochs must be an integer' in result.output
        finally:
            Path(config_path).unlink()
    
    @patch('neurolite.cli.main.config_manager')
    def test_export_config_command(self, mock_config_manager):
        """Test export-config command."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "exported_config.yaml"
            
            result = runner.invoke(cli, [
                'export-config',
                '--format', 'yaml',
                str(output_path)
            ])
            
            assert result.exit_code == 0
            assert 'Configuration exported to' in result.output
            mock_config_manager.save_config.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_verbose_mode(self):
        """Test CLI with verbose mode enabled."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ['--verbose', 'info'])
        
        assert result.exit_code == 0
        # Verbose mode should be enabled
    
    def test_debug_mode(self):
        """Test CLI with debug mode enabled."""
        runner = CliRunner()
        
        with patch('neurolite.cli.main.log_system_info') as mock_log:
            result = runner.invoke(cli, ['--debug', 'info'])
            
            assert result.exit_code == 0
            # Debug mode should trigger system info logging
            assert mock_log.call_count >= 1
    
    def test_config_file_integration(self):
        """Test CLI with configuration file integration."""
        runner = CliRunner()
        
        config_data = {
            'train': {
                'epochs': 25,
                'batch_size': 64
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                '--config', config_path,
                'info'
            ])
            
            assert result.exit_code == 0
            assert 'Loaded configuration from' in result.output
        finally:
            Path(config_path).unlink()
    
    def test_nonexistent_data_path(self):
        """Test CLI with non-existent data path."""
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'train',
            '/nonexistent/path'
        ])
        
        assert result.exit_code == 2  # Click validation error
        assert 'does not exist' in result.output
    
    def test_nonexistent_model_path(self):
        """Test CLI with non-existent model path."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_path = Path(temp_dir) / "test_data"
            test_data_path.mkdir()
            
            result = runner.invoke(cli, [
                'evaluate',
                '/nonexistent/model.pkl',
                str(test_data_path)
            ])
            
            assert result.exit_code == 2  # Click validation error
            assert 'does not exist' in result.output


if __name__ == '__main__':
    pytest.main([__file__])