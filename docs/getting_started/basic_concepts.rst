Basic Concepts
==============

This guide introduces the core concepts and philosophy behind NeuroLite.

The NeuroLite Philosophy
-------------------------

NeuroLite is built on three core principles:

1. **Simplicity First**: Complex ML workflows should be simple to use
2. **Automation**: Let the library handle the technical details
3. **Flexibility**: Provide escape hatches for advanced users

Core Components
---------------

The Train Function
~~~~~~~~~~~~~~~~~~

The ``train()`` function is the heart of NeuroLite:

.. code-block:: python

   import neurolite

   model = neurolite.train(
       data="your_data.csv",      # Your dataset
       task="classification",      # What you want to do
       target="label",            # Target column (optional)
       config={}                  # Advanced configuration (optional)
   )

**Parameters:**

- ``data``: Path to your dataset or data directory
- ``task``: The type of machine learning task
- ``target``: Column name for supervised learning (auto-detected if not provided)
- ``config``: Dictionary of advanced configuration options

Supported Tasks
~~~~~~~~~~~~~~~

NeuroLite supports a wide range of machine learning tasks:

**Computer Vision:**
- ``image_classification`` - Classify images into categories
- ``object_detection`` - Detect and locate objects in images
- ``image_segmentation`` - Pixel-level image segmentation

**Natural Language Processing:**
- ``text_classification`` - Classify text documents
- ``sentiment_analysis`` - Analyze sentiment in text
- ``text_generation`` - Generate text content
- ``question_answering`` - Answer questions based on context

**Traditional ML:**
- ``classification`` - Multi-class classification
- ``binary_classification`` - Binary classification
- ``regression`` - Predict continuous values
- ``clustering`` - Unsupervised grouping
- ``forecasting`` - Time series prediction

The TrainedModel Object
~~~~~~~~~~~~~~~~~~~~~~~

When you call ``train()``, you get back a ``TrainedModel`` object:

.. code-block:: python

   model = neurolite.train("data.csv", task="classification")

   # Make predictions
   prediction = model.predict(new_data)

   # Evaluate performance
   metrics = model.evaluate()

   # Save the model
   model.save("my_model.pkl")

   # Get model information
   info = model.info()

**Key Methods:**

- ``predict(data)``: Make predictions on new data
- ``evaluate(test_data=None)``: Evaluate model performance
- ``save(path)``: Save the model to disk
- ``info()``: Get model information and statistics

Data Handling
-------------

Automatic Data Detection
~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite automatically detects your data format and structure:

.. code-block:: python

   # These all work automatically
   model = neurolite.train("images/")              # Image directory
   model = neurolite.train("data.csv")             # CSV file
   model = neurolite.train("data.json")            # JSON file
   model = neurolite.train("data.xlsx")            # Excel file
   model = neurolite.train("data.parquet")         # Parquet file

Data Preprocessing
~~~~~~~~~~~~~~~~~~

NeuroLite handles preprocessing automatically:

- **Missing Values**: Intelligent imputation strategies
- **Categorical Encoding**: One-hot encoding, label encoding
- **Feature Scaling**: Standardization, normalization
- **Text Processing**: Tokenization, vectorization
- **Image Processing**: Resizing, normalization, augmentation

You can also customize preprocessing:

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

Model Selection
---------------

Automatic Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

NeuroLite automatically chooses the best model for your data:

- **Data Size**: Considers dataset size for model complexity
- **Data Type**: Matches model architecture to data type
- **Task Type**: Selects appropriate model family
- **Performance**: Benchmarks multiple models

Manual Model Selection
~~~~~~~~~~~~~~~~~~~~~~

You can also specify the model type:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={
           "model_type": "random_forest"  # or "neural_network", "xgboost", etc.
       }
   )

**Available Model Types:**

- ``neural_network`` - Deep neural networks
- ``random_forest`` - Random forest ensemble
- ``xgboost`` - Gradient boosting
- ``svm`` - Support vector machines
- ``logistic_regression`` - Linear models
- ``transformer`` - Transformer models (NLP)
- ``cnn`` - Convolutional networks (vision)

Hyperparameter Optimization
----------------------------

Automatic Optimization
~~~~~~~~~~~~~~~~~~~~~~

NeuroLite automatically optimizes hyperparameters:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       optimization="bayesian"  # Automatic hyperparameter tuning
   )

**Optimization Methods:**

- ``bayesian`` - Bayesian optimization (recommended)
- ``grid`` - Grid search
- ``random`` - Random search
- ``none`` - Use default parameters

Custom Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~

You can also set specific hyperparameters:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={
           "epochs": 100,
           "batch_size": 32,
           "learning_rate": 0.001,
           "dropout": 0.2
       }
   )

Evaluation and Metrics
----------------------

Automatic Evaluation
~~~~~~~~~~~~~~~~~~~~

NeuroLite automatically evaluates your model:

.. code-block:: python

   model = neurolite.train("data.csv", task="classification")
   
   # Get evaluation metrics
   metrics = model.evaluate()
   print(f"Accuracy: {metrics['accuracy']:.3f}")
   print(f"F1 Score: {metrics['f1_score']:.3f}")

**Available Metrics:**

- **Classification**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Regression**: MAE, MSE, RMSE, R²
- **Clustering**: Silhouette score, Calinski-Harabasz index

Visualization
~~~~~~~~~~~~~

Built-in visualization tools:

.. code-block:: python

   # Plot training metrics
   model.plot_metrics()

   # Plot confusion matrix
   model.plot_confusion_matrix()

   # Plot feature importance
   model.plot_feature_importance()

   # Plot predictions vs actual
   model.plot_predictions()

Deployment
----------

Simple Deployment
~~~~~~~~~~~~~~~~~

Deploy your model with one line:

.. code-block:: python

   endpoint = neurolite.deploy(model, platform="api")

**Deployment Platforms:**

- ``api`` - REST API server
- ``docker`` - Docker container
- ``aws`` - AWS Lambda/SageMaker
- ``gcp`` - Google Cloud Platform
- ``azure`` - Microsoft Azure

Advanced Deployment
~~~~~~~~~~~~~~~~~~~

Customize your deployment:

.. code-block:: python

   endpoint = neurolite.deploy(
       model,
       platform="api",
       config={
           "port": 8080,
           "host": "0.0.0.0",
           "workers": 4,
           "timeout": 30
       }
   )

Configuration System
--------------------

Global Configuration
~~~~~~~~~~~~~~~~~~~~

Set global preferences:

.. code-block:: python

   import neurolite

   # Set device preference
   neurolite.config.set_device("gpu")  # or "cpu", "auto"

   # Set cache directory
   neurolite.config.set_cache_dir("./cache")

   # Set logging level
   neurolite.config.set_log_level("INFO")

Model-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure individual models:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       config={
           "model_type": "neural_network",
           "epochs": 100,
           "batch_size": 32,
           "validation_split": 0.2,
           "early_stopping": True,
           "save_best_only": True
       }
   )

Error Handling
--------------

NeuroLite provides clear error messages and suggestions:

.. code-block:: python

   try:
       model = neurolite.train("nonexistent.csv", task="classification")
   except neurolite.DataError as e:
       print(f"Data error: {e}")
       print(f"Suggestion: {e.suggestion}")

**Common Exception Types:**

- ``DataError`` - Issues with data loading or format
- ``ModelError`` - Problems with model training
- ``ConfigError`` - Configuration issues
- ``DeploymentError`` - Deployment problems

Best Practices
--------------

1. **Start Simple**: Use default settings first, then customize
2. **Validate Early**: Check your data before training
3. **Monitor Training**: Use built-in logging and metrics
4. **Save Models**: Always save successful models
5. **Test Thoroughly**: Evaluate on held-out test data

Next Steps
----------

Now that you understand the basics:

- Explore :doc:`../user_guide/computer_vision` for vision tasks
- Learn about :doc:`../user_guide/nlp` for text processing
- Check out :doc:`../tutorials/image_classification` for hands-on practice
- Browse :doc:`../examples/basic_examples` for more examples