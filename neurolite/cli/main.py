"""
Main CLI entry point for NeuroLite.

Provides command-line interface for NeuroLite operations.
"""

import click
from pathlib import Path

from ..core import get_logger, get_config, log_system_info
from ..api import train, deploy

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.version_option()
def cli(verbose: bool, debug: bool):
    """
    NeuroLite - AI/ML/DL/NLP Productivity Library
    
    Train and deploy machine learning models with minimal code.
    """
    config = get_config()
    
    if verbose:
        config.verbose = True
        config.logging.level = "DEBUG"
    
    if debug:
        config.debug = True
        log_system_info()


@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.option('--model', '-m', default='auto', help='Model type to use')
@click.option('--task', '-t', default='auto', help='Task type')
@click.option('--target', help='Target column for tabular data')
@click.option('--validation-split', default=0.2, type=float, help='Validation split ratio')
@click.option('--test-split', default=0.1, type=float, help='Test split ratio')
@click.option('--optimize/--no-optimize', default=True, help='Enable hyperparameter optimization')
@click.option('--deploy/--no-deploy', default=False, help='Deploy after training')
@click.option('--output', '-o', type=click.Path(), help='Output directory for model artifacts')
def train_cmd(
    data: str,
    model: str,
    task: str,
    target: str,
    validation_split: float,
    test_split: float,
    optimize: bool,
    deploy: bool,
    output: str
):
    """Train a machine learning model."""
    try:
        logger.info(f"Starting training with data: {data}")
        
        # Call the main train function
        trained_model = train(
            data=data,
            model=model,
            task=task,
            target=target,
            validation_split=validation_split,
            test_split=test_split,
            optimize=optimize,
            deploy=deploy,
            output_dir=output
        )
        
        click.echo(f"Training completed successfully!")
        if output:
            click.echo(f"Model saved to: {output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--format', '-f', default='api', help='Deployment format')
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, type=int, help='Port number')
@click.option('--output', '-o', type=click.Path(), help='Output path for exported model')
def deploy_cmd(
    model_path: str,
    format: str,
    host: str,
    port: int,
    output: str
):
    """Deploy a trained model."""
    try:
        logger.info(f"Deploying model from: {model_path}")
        
        # Load model (placeholder - will be implemented in later tasks)
        # model = load_model(model_path)
        
        # Deploy model
        # deployed = deploy(
        #     model=model,
        #     format=format,
        #     host=host,
        #     port=port,
        #     output_path=output
        # )
        
        click.echo("Deployment functionality will be implemented in subsequent tasks.")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
def info():
    """Show system and library information."""
    click.echo("NeuroLite System Information")
    click.echo("=" * 30)
    
    # Log system info which will be displayed
    log_system_info()
    
    # Show configuration
    config = get_config()
    click.echo(f"Environment: {config.environment.value}")
    click.echo(f"Debug mode: {config.debug}")
    click.echo(f"Model cache: {config.model.cache_dir}")
    click.echo(f"Data cache: {config.data.cache_dir}")


@cli.command()
@click.option('--format', '-f', default='yaml', type=click.Choice(['yaml', 'json']), help='Config format')
@click.argument('output_path', type=click.Path())
def export_config(format: str, output_path: str):
    """Export current configuration to file."""
    try:
        from ..core.config import config_manager
        
        output_file = Path(output_path)
        if not output_file.suffix:
            output_file = output_file.with_suffix(f'.{format}')
        
        config_manager.save_config(output_file)
        click.echo(f"Configuration exported to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()