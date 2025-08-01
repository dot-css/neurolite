# API Reference

This section provides comprehensive documentation for all NeuroLite APIs.

## Core Functions

### `train()`

The main training function that handles the complete ML workflow.

```python
def train(
    data: Union[str, Path],
    model: str = "auto",
    task: str = "auto",
    target: Optional[str] = None,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    optimize: bool = True,
    deploy: bool = False,
    **kwargs
) -> TrainedModel
```

**Parameters:**

- `data` (str | Path): Path to data file or directory
  - For images: Directory containing subdirectories for each class
  - For text: CSV file with text and labels
  - For tabular: CSV file with features and target column

- `model` (str, default="auto"): Model type to use
  - "auto": Automatic model selection based on data type
  - Computer Vision: "resnet18", "resnet50", "efficientnet", "vit"
  - NLP: "bert", "roberta", "distilbert", "gpt2"
  - Traditional ML: "random_forest", "xgboost", "svm", "logistic_regression"

- `task` (str, default="auto"): Task type
  - "auto": Automatic task detection
  - "classification": Multi-class classification
  - "regression": Regression tasks
  - "image_classification": Image classification
  - "object_detection": Object detection
  - "text_classification": Text classification
  - "sentiment_analysis": Sentiment analysis

- `target` (str, optional): Target column name for tabular data

- `validation_split` (float, default=0.2): Fraction of data for validation (0.0-1.0)

- `test_split` (float, default=0.1): Fraction of data for testing (0.0-1.0)

- `optimize` (bool, default=True): Whether to perform hyperparameter optimization

- `deploy` (bool, default=False): Whether to create deployment artifacts

- `**kwargs`: Additional configuration options (see domain-specific parameters below)

**Returns:**

`TrainedModel`: A trained model ready for inference and deployment

**Domain-Specific Parameters:**

**Computer Vision:**
- `image_size` (int, default=224): Input image size
- `augmentation` (bool, default=True): Whether to apply data augmentation
- `confidence_threshold` (float, default=0.5): Confidence threshold for object detection
- `nms_threshold` (float, default=0.4): NMS threshold for object detection

**NLP:**
- `max_length` (int, default=512): Maximum sequence length
- `tokenizer` (str, optional): Custom tokenizer to use
- `remove_stopwords` (bool, default=False): Whether to remove stopwords
- `temperature` (float, default=1.0): Temperature for text generation
- `top_p` (float, default=0.9): Top-p sampling for text generation

**Tabular:**
- `feature_engineering` (bool, default=True): Whether to apply automatic feature engineering
- `scaling` (str, default="standard"): Feature scaling method ("standard", "minmax", "robust")
- `categorical_encoding` (str, default="auto"): Categorical encoding method
- `missing_value_strategy` (str, default="auto"): Strategy for handling missing values

**Examples:**

```python
# Computer Vision - Image Classification
model = neurolite.train(
    data='data/images/',
    model='resnet18',
    task='image_classification',
    image_size=224,
    augmentation=True
)

# NLP - Text Classification
model = neurolite.train(
    data='data/reviews.csv',
    model='bert',
    task='text_classification',
    target='sentiment',
    max_length=256
)

# Tabular - Regression
model = neurolite.train(
    data='data/housing.csv',
    model='random_forest',
    task='regression',
    target='price',
    feature_engineering=True
)
```

**Raises:**

- `ConfigurationError`: Invalid parameters or configuration
- `DataError`: Issues with data loading or preprocessing
- `ModelError`: Model-related errors
- `TrainingError`: Training process failures

### `deploy()`

Deploy a trained model for inference.

```python
def deploy(
    model: TrainedModel,
    format: str = "api",
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs
) -> Union[ExportedModel, Any]
```

**Parameters:**

- `model` (TrainedModel): Trained model to deploy
- `format` (str, default="api"): Deployment format
  - "api": Create REST API server
  - "onnx": Export to ONNX format
  - "tflite": Export to TensorFlow Lite
  - "torchscript": Export to TorchScript
- `host` (str, default="0.0.0.0"): Host address for API deployment
- `port` (int, default=8000): Port for API deployment
- `**kwargs`: Additional deployment options

**Returns:**

`Union[ExportedModel, Any]`: Deployed model instance (API server or exported model)

**Examples:**

```python
# Deploy as REST API
api_server = neurolite.deploy(model, format='api', port=8080)

# Export to ONNX
exported_model = neurolite.deploy(model, format='onnx', optimize=True)
```

## Data Classes

### `TrainedModel`

Represents a trained model with all associated metadata.

**Attributes:**

- `model`: The actual trained model object
- `config`: Training configuration used
- `training_history`: Training metrics history
- `evaluation_results`: Model evaluation results
- `metadata`: Additional model metadata
- `framework`: ML framework used ("pytorch", "tensorflow", "sklearn")

**Methods:**

- `predict(data)`: Make predictions on new data
- `evaluate(data)`: Evaluate model on test data
- `save(path)`: Save model to disk
- `load(path)`: Load model from disk

## Exceptions

### `NeuroLiteError`

Base exception for all NeuroLite errors.

### `ConfigurationError`

Raised when there are issues with configuration parameters.

### `DataError`

Raised when there are issues with data loading or preprocessing.

### `ModelError`

Raised when there are model-related errors.

### `TrainingError`

Raised when training process fails.

## Advanced Usage

### Custom Models

Register custom models with the model registry:

```python
from neurolite.models import register_model, BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
    
    def fit(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass

# Register the model
register_model("my_custom_model", MyCustomModel)

# Use it
model = neurolite.train(data='data.csv', model='my_custom_model')
```

### Plugin System

Create custom preprocessing plugins:

```python
from neurolite.core.plugins import register_preprocessor, BasePreprocessor

class MyPreprocessor(BasePreprocessor):
    def transform(self, data):
        # Custom preprocessing logic
        return processed_data

# Register the preprocessor
register_preprocessor("my_preprocessor", MyPreprocessor)
```

### Configuration Management

Use configuration files for complex workflows:

```python
import neurolite

# Load configuration from file
config = neurolite.load_config('config.yaml')

# Train with configuration
model = neurolite.train(**config)
```

## Performance Optimization

### GPU Acceleration

NeuroLite automatically detects and uses GPU when available:

```python
# GPU will be used automatically if available
model = neurolite.train(data='data/', model='resnet50')
```

### Parallel Processing

Enable parallel processing for data loading:

```python
model = neurolite.train(
    data='data/',
    num_workers=4,  # Number of parallel workers
    batch_size=64   # Larger batch size for better GPU utilization
)
```

### Memory Optimization

For large datasets, use lazy loading:

```python
model = neurolite.train(
    data='large_dataset/',
    lazy_loading=True,  # Load data on-demand
    cache_size=1000     # Cache size for preprocessed data
)
```