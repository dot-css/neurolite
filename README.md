# NeuroLite 🧠⚡

[![PyPI version](https://badge.fury.io/py/neurolite.svg)](https://badge.fury.io/py/neurolite)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/dot-css/neurolite/branch/main/graph/badge.svg)](https://codecov.io/gh/dot-css/neurolite)

**NeuroLite** is a revolutionary AI/ML/DL/NLP productivity library that enables you to build, train, and deploy machine learning models with **minimal code**. Transform complex ML workflows into simple, intuitive operations.

## 🚀 Why NeuroLite?

- **🎯 Minimal Code**: Train state-of-the-art models in less than 10 lines of code
- **🤖 Auto-Everything**: Automatic data processing, model selection, and hyperparameter tuning
- **🌍 Multi-Domain**: Unified interface for Computer Vision, NLP, and Traditional ML
- **⚡ Production Ready**: One-click deployment to production environments
- **🔧 Extensible**: Plugin system for custom models and workflows
- **📊 Rich Visualization**: Built-in dashboards and reporting tools

## 📦 Installation

### Quick Install
```bash
pip install neurolite
```

### Development Install
```bash
git clone https://github.com/dot-css/neurolite.git
cd neurolite
pip install -e ".[dev]"
```

### Optional Dependencies
```bash
# For TensorFlow support
pip install neurolite[tensorflow]

# For XGBoost support  
pip install neurolite[xgboost]

# Install everything
pip install neurolite[all]
```

## 🎯 Quick Start

### Image Classification in 3 Lines
```python
from neurolite import train

# Train a computer vision model
model = train(data="path/to/images", task="image_classification")
predictions = model.predict("path/to/new/image.jpg")
```

### Text Classification
```python
from neurolite import train

# Train an NLP model
model = train(data="reviews.csv", task="sentiment_analysis", target="sentiment")
result = model.predict("This product is amazing!")
```

### Tabular Data Prediction
```python
from neurolite import train

# Train on structured data
model = train(data="sales.csv", task="regression", target="revenue")
forecast = model.predict({"feature1": 100, "feature2": "category_a"})
```

### One-Click Deployment
```python
from neurolite import deploy

# Deploy your model instantly
endpoint = deploy(model, platform="cloud", auto_scale=True)
print(f"Model deployed at: {endpoint.url}")
```

## 🌟 Key Features

### 🤖 Automatic Intelligence
- **Auto Data Processing**: Handles missing values, encoding, scaling automatically
- **Auto Model Selection**: Chooses the best model architecture for your data
- **Auto Hyperparameter Tuning**: Optimizes model parameters using advanced algorithms
- **Auto Feature Engineering**: Creates and selects relevant features

### 🎨 Multi-Domain Support

#### Computer Vision
```python
# Image classification, object detection, segmentation
model = train(data="images/", task="object_detection")
results = model.predict("test_image.jpg")
```

#### Natural Language Processing
```python
# Text classification, sentiment analysis, translation
model = train(data="texts.csv", task="text_generation")
generated = model.predict("Once upon a time")
```

#### Traditional ML
```python
# Regression, classification, clustering
model = train(data="tabular.csv", task="classification")
predictions = model.predict(new_data)
```

### 🚀 Production Deployment
```python
from neurolite import deploy

# Deploy to various platforms
deploy(model, platform="aws")        # AWS Lambda/SageMaker
deploy(model, platform="gcp")        # Google Cloud
deploy(model, platform="azure")      # Azure ML
deploy(model, platform="docker")     # Docker container
deploy(model, platform="kubernetes") # Kubernetes cluster
```

## 📊 Advanced Features

### Hyperparameter Optimization
```python
from neurolite import train

model = train(
    data="data.csv",
    task="classification",
    optimization="bayesian",  # bayesian, grid, random
    trials=100,
    timeout=3600  # 1 hour
)
```

### Model Ensembles
```python
from neurolite import train

# Automatic ensemble creation
model = train(
    data="data.csv",
    task="regression",
    ensemble=True,
    ensemble_size=5
)
```

### Custom Workflows
```python
from neurolite.workflows import create_workflow

# Define custom ML pipeline
workflow = create_workflow([
    "data_cleaning",
    "feature_engineering", 
    "model_training",
    "evaluation",
    "deployment"
])

result = workflow.run(data="data.csv")
```

### Real-time Monitoring
```python
from neurolite import monitor

# Monitor deployed models
monitor.track(model, metrics=["accuracy", "latency", "drift"])
dashboard = monitor.dashboard(model)
```

## 🔧 Configuration

### Global Settings
```python
import neurolite

# Configure global settings
neurolite.config.set_device("gpu")  # cpu, gpu, auto
neurolite.config.set_cache_dir("./cache")
neurolite.config.set_log_level("INFO")
```

### Model-Specific Configuration
```python
model = train(
    data="data.csv",
    task="classification",
    config={
        "model_type": "neural_network",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "early_stopping": True
    }
)
```

## 📈 Performance Benchmarks

| Task | Dataset | NeuroLite | Traditional Approach | Time Saved |
|------|---------|-----------|---------------------|-------------|
| Image Classification | CIFAR-10 | 3 lines | 200+ lines | 98.5% |
| Sentiment Analysis | IMDB | 2 lines | 150+ lines | 98.7% |
| Sales Forecasting | Custom | 4 lines | 300+ lines | 98.7% |

## 🛠️ Supported Models

### Computer Vision
- **Classification**: ResNet, EfficientNet, Vision Transformer
- **Object Detection**: YOLO, Faster R-CNN, SSD
- **Segmentation**: U-Net, DeepLab, FCN

### Natural Language Processing
- **Text Classification**: BERT, RoBERTa, DistilBERT
- **Text Generation**: GPT-2, T5, BART
- **Translation**: MarianMT, T5
- **Question Answering**: BERT, RoBERTa

### Traditional ML
- **Classification**: Random Forest, XGBoost, SVM, Logistic Regression
- **Regression**: Linear Regression, Random Forest, Gradient Boosting
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Ensemble**: Voting, Stacking, Bagging

## 🔌 Plugin System

Extend NeuroLite with custom models and workflows:

```python
from neurolite.plugins import register_model

@register_model("my_custom_model")
class CustomModel:
    def train(self, data):
        # Custom training logic
        pass
    
    def predict(self, data):
        # Custom prediction logic
        pass

# Use your custom model
model = train(data="data.csv", model="my_custom_model")
```

## 📚 Documentation

- **[Getting Started Guide](https://neurolite.readthedocs.io/getting-started)**
- **[API Reference](https://neurolite.readthedocs.io/api)**
- **[Tutorials](https://neurolite.readthedocs.io/tutorials)**
- **[Examples](https://github.com/dot-css/neurolite/tree/main/examples)**
- **[Plugin Development](https://neurolite.readthedocs.io/plugins)**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/dot-css/neurolite.git
cd neurolite
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
black neurolite/ tests/
flake8 neurolite/ tests/
mypy neurolite/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by the NeuroLite Team
- Powered by PyTorch, Transformers, Scikit-learn, and other amazing open-source libraries
- Special thanks to our contributors and the ML community

## 📞 Support

- **Documentation**: [https://neurolite.readthedocs.io](https://neurolite.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/dot-css/neurolite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dot-css/neurolite/discussions)
- **Email**: saqibshaikhdz@gmail.com


---

**Made with ❤️ for the AI/ML community**