Frequently Asked Questions
==========================

This page answers common questions about NeuroLite.

General Questions
-----------------

What is NeuroLite?
~~~~~~~~~~~~~~~~~~

NeuroLite is an AI/ML/DL/NLP productivity library that enables you to build, train, and deploy machine learning models with minimal code. It automates the complex parts of machine learning while providing flexibility for advanced users.

How is NeuroLite different from other ML libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite focuses on:

- **Minimal Code**: Train models in less than 10 lines of code
- **Automation**: Automatic data processing, model selection, and hyperparameter tuning
- **Multi-Domain**: Unified interface for computer vision, NLP, and traditional ML
- **Production Ready**: One-click deployment to multiple platforms

What machine learning tasks does NeuroLite support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite supports:

- **Computer Vision**: Image classification, object detection, segmentation
- **NLP**: Text classification, sentiment analysis, text generation
- **Traditional ML**: Regression, classification, clustering, forecasting

Installation and Setup
----------------------

How do I install NeuroLite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install neurolite

For the latest development version:

.. code-block:: bash

   pip install git+https://github.com/dot-css/neurolite.git

What are the system requirements?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB storage

**Recommended:**
- Python 3.9+
- 16GB RAM
- GPU with CUDA support
- 10GB storage

Do I need a GPU?
~~~~~~~~~~~~~~~~

No, NeuroLite works on CPU. However, GPU acceleration significantly speeds up training for deep learning models. NeuroLite automatically detects and uses available GPUs.

Data and Training
-----------------

What data formats does NeuroLite support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Images**: JPEG, PNG, BMP, TIFF
- **Text**: CSV, TSV, JSON, TXT
- **Tabular**: CSV, Excel, Parquet
- **Directories**: Organized folder structures

How should I organize my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For images:**

.. code-block:: text

   images/
   ├── class1/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── class2/
       ├── img3.jpg
       └── img4.jpg

**For text/tabular:**

CSV files with headers, where one column contains the target variable.

How much data do I need?
~~~~~~~~~~~~~~~~~~~~~~~~

It depends on the task:

- **Simple classification**: 100-1000 samples per class
- **Complex tasks**: 1000+ samples per class
- **Transfer learning**: Can work with fewer samples

NeuroLite uses transfer learning and data augmentation to work with smaller datasets.

Can I use my own data preprocessing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! You can customize preprocessing:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={
           "preprocessing": {
               "handle_missing": "median",
               "scale_features": "standard",
               "encode_categorical": "onehot"
           }
       }
   )

Models and Performance
----------------------

How does NeuroLite choose the best model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite considers:

- Data size and type
- Task complexity
- Available computational resources
- Performance benchmarks

You can also specify the model type manually:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={"model_type": "random_forest"}
   )

What models are available?
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Computer Vision:**
- ResNet, EfficientNet, Vision Transformer
- YOLO, Faster R-CNN (object detection)
- U-Net, DeepLab (segmentation)

**NLP:**
- BERT, RoBERTa, DistilBERT
- GPT-2, T5 (text generation)

**Traditional ML:**
- Random Forest, XGBoost, SVM
- Linear/Logistic Regression
- K-Means, DBSCAN (clustering)

How can I improve model performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **More/Better Data**: Quality data is most important
2. **Hyperparameter Tuning**: Use ``optimization="bayesian"``
3. **Transfer Learning**: Use pre-trained models
4. **Data Augmentation**: Especially for images
5. **Ensemble Methods**: Combine multiple models

Can I use custom models?
~~~~~~~~~~~~~~~~~~~~~~~~

Yes! NeuroLite supports custom models through the plugin system:

.. code-block:: python

   from neurolite.plugins import register_model

   @register_model("my_custom_model")
   class CustomModel:
       def train(self, data):
           # Your training logic
           pass
       
       def predict(self, data):
           # Your prediction logic
           pass

Deployment
----------

How do I deploy my model?
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Deploy as REST API
   endpoint = neurolite.deploy(model, platform="api")

   # Deploy to cloud
   endpoint = neurolite.deploy(model, platform="aws")

What deployment platforms are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Local**: REST API server
- **Cloud**: AWS, Google Cloud, Azure
- **Containers**: Docker, Kubernetes
- **Edge**: ONNX, TensorFlow Lite

Can I deploy multiple models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! You can deploy multiple models to the same endpoint or separate endpoints:

.. code-block:: python

   # Multiple models on same endpoint
   endpoint = neurolite.deploy([model1, model2], platform="api")

   # Separate endpoints
   endpoint1 = neurolite.deploy(model1, platform="api", port=8080)
   endpoint2 = neurolite.deploy(model2, platform="api", port=8081)

How do I monitor deployed models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neurolite import monitor

   # Monitor model performance
   monitor.track(model, metrics=["accuracy", "latency"])
   
   # View dashboard
   dashboard = monitor.dashboard(model)

Troubleshooting
---------------

Training is very slow. What can I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use GPU**: Enable GPU acceleration
2. **Reduce batch size**: If running out of memory
3. **Smaller model**: Use a lighter model architecture
4. **Reduce data size**: Use a subset for initial experiments
5. **Optimize data loading**: Ensure data is on fast storage

I'm getting "Out of Memory" errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reduce batch size
   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={"batch_size": 16}
   )

   # Use CPU instead of GPU
   neurolite.config.set_device("cpu")

My model accuracy is low
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Check data quality**: Ensure correct labels and clean data
2. **More data**: Collect more training samples
3. **Data augmentation**: Especially for images
4. **Hyperparameter tuning**: Use automatic optimization
5. **Transfer learning**: Use pre-trained models
6. **Feature engineering**: For tabular data

The model is overfitting
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={
           "validation_split": 0.2,
           "early_stopping": True,
           "dropout": 0.3,
           "regularization": 0.01
       }
   )

I can't load my saved model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure you're using the same NeuroLite version that was used to save the model. If versions differ, try:

.. code-block:: python

   # Export to a standard format
   model.export("model.onnx", format="onnx")
   
   # Load with ONNX runtime
   import onnxruntime as ort
   session = ort.InferenceSession("model.onnx")

Advanced Usage
--------------

Can I use NeuroLite in production?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! NeuroLite is designed for production use with:

- Robust error handling
- Model versioning
- Performance monitoring
- Scalable deployment options

How do I handle large datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use data streaming
   model = neurolite.train(
       data="large_dataset.csv",
       task="classification",
       config={
           "batch_size": 1000,
           "streaming": True,
           "cache_data": False
       }
   )

Can I use NeuroLite with other ML libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! NeuroLite is compatible with:

- **Scikit-learn**: For preprocessing and evaluation
- **PyTorch/TensorFlow**: For custom models
- **Pandas**: For data manipulation
- **MLflow**: For experiment tracking

How do I contribute to NeuroLite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See our :doc:`contributing` guide for:

- Code contributions
- Documentation improvements
- Bug reports
- Feature requests

Getting Help
------------

Where can I get help?
~~~~~~~~~~~~~~~~~~~~~

1. **Documentation**: This documentation site
2. **GitHub Issues**: Report bugs and request features
3. **Discussions**: Community discussions on GitHub
4. **Examples**: Check the examples repository

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~

Please report bugs on our `GitHub Issues <https://github.com/dot-css/neurolite/issues>`_ page with:

- NeuroLite version
- Python version
- Operating system
- Complete error message
- Minimal code to reproduce the issue

Can I request new features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Please use our `GitHub Issues <https://github.com/dot-css/neurolite/issues>`_ page to request new features. Include:

- Use case description
- Expected behavior
- Any relevant examples

Performance and Scaling
-----------------------

How fast is NeuroLite?
~~~~~~~~~~~~~~~~~~~~~~

Performance depends on:

- Hardware (CPU/GPU)
- Data size and complexity
- Model architecture
- Task type

See our :doc:`getting_started/quickstart` for benchmark results.

Can NeuroLite handle big data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, through:

- Data streaming
- Distributed training (coming soon)
- Cloud deployment
- Efficient data loading

How do I optimize for speed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use GPU**: Enable GPU acceleration
2. **Batch processing**: Process multiple samples together
3. **Model optimization**: Use lighter models for inference
4. **Caching**: Enable data and model caching
5. **Parallel processing**: Use multiple workers

Licensing and Commercial Use
----------------------------

What license does NeuroLite use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite is released under the MIT License, which allows:

- Commercial use
- Modification
- Distribution
- Private use

Can I use NeuroLite commercially?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! The MIT License allows commercial use without restrictions.

Do I need to credit NeuroLite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While not required, we appreciate attribution. You can mention:

"Powered by NeuroLite - https://github.com/dot-css/neurolite"