# NeuroLite Documentation

Welcome to NeuroLite - the AI/ML/DL/NLP productivity library that enables you to build, train, and deploy machine learning models with minimal code (under 10 lines).

## Quick Start

```python
import neurolite

# Train a model with just one line
model = neurolite.train('data/my_dataset.csv', task='classification')

# Deploy it with another line
neurolite.deploy(model, format='api', port=8080)
```

## Documentation Structure

- [API Reference](api/README.md) - Complete API documentation
- [Tutorials](tutorials/README.md) - Step-by-step tutorials for different domains
- [Examples](examples/README.md) - Gallery of examples with sample datasets
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Contributing](contributing.md) - How to contribute to NeuroLite

## Features

- **Minimal Code Interface**: Train and deploy models with less than 10 lines of code
- **Automatic Data Processing**: Automatically detect and preprocess your data
- **Model Zoo Integration**: Access to pre-configured state-of-the-art models
- **Automated Training**: Fully automated training with intelligent monitoring
- **Hyperparameter Optimization**: Automatic hyperparameter tuning
- **One-Click Deployment**: Deploy trained models to production with a single command
- **Multi-Domain Support**: Unified interfaces for computer vision, NLP, and traditional ML
- **Extensibility**: Plugin system for custom models and preprocessing

## Supported Domains

- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Text classification, sentiment analysis, generation
- **Traditional ML**: Regression, classification, clustering with tabular data

## Installation

```bash
pip install neurolite
```

## License

MIT License - see [LICENSE](../LICENSE) for details.