#!/usr/bin/env python3
"""
Simple CLI test script to verify basic functionality.
"""

import tempfile
import yaml
import json
from pathlib import Path
from click.testing import CliRunner

from neurolite.cli.main import cli


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'NeuroLite' in result.output
    print("✓ CLI help test passed")


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    print("✓ CLI version test passed")


def test_train_help():
    """Test train command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--help'])
    assert result.exit_code == 0
    assert 'Train a machine learning model' in result.output
    print("✓ Train help test passed")


def test_evaluate_help():
    """Test evaluate command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ['evaluate', '--help'])
    assert result.exit_code == 0
    assert 'Evaluate a trained model' in result.output
    print("✓ Evaluate help test passed")


def test_deploy_help():
    """Test deploy command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ['deploy', '--help'])
    assert result.exit_code == 0
    assert 'Deploy a trained model' in result.output
    print("✓ Deploy help test passed")


def test_list_models_help():
    """Test list-models command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ['list-models', '--help'])
    assert result.exit_code == 0
    assert 'List available models' in result.output
    print("✓ List models help test passed")


def test_info_command():
    """Test info command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['info'])
    assert result.exit_code == 0
    assert 'NeuroLite System Information' in result.output
    print("✓ Info command test passed")


def test_init_config_command():
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
        
    print("✓ Init config command test passed")


def test_validate_config_command():
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
    
    print("✓ Validate config command test passed")


def test_export_config_command():
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
        assert output_path.exists()
        
        # Verify exported config content
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'environment' in config
        assert 'debug' in config
        
    print("✓ Export config command test passed")


def test_config_file_loading():
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
    
    print("✓ Config file loading test passed")


def main():
    """Run all CLI tests."""
    print("Running CLI functionality tests...")
    print("=" * 50)
    
    try:
        test_cli_help()
        test_cli_version()
        test_train_help()
        test_evaluate_help()
        test_deploy_help()
        test_list_models_help()
        test_info_command()
        test_init_config_command()
        test_validate_config_command()
        test_export_config_command()
        test_config_file_loading()
        
        print("=" * 50)
        print("✓ All CLI tests passed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == '__main__':
    main()