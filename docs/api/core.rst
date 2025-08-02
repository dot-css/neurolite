Core API Reference
==================

This section documents the core functions and classes in NeuroLite.

Core Functions
--------------

.. automodule:: neurolite.api
   :members:
   :undoc-members:
   :show-inheritance:

train()
~~~~~~~

.. autofunction:: neurolite.train

The main training function that creates and trains machine learning models.

**Parameters:**

* **data** (*str or pandas.DataFrame*) -- Path to dataset or DataFrame object
* **task** (*str*) -- Type of machine learning task to perform
* **target** (*str, optional*) -- Target column name for supervised learning
* **config** (*dict, optional*) -- Configuration dictionary for advanced options

**Returns:**

* **TrainedModel** -- Trained model object ready for predictions

**Example:**

.. code-block:: python

   import neurolite
   
   # Basic usage
   model = neurolite.train('data.csv', task='classification')
   
   # With configuration
   model = neurolite.train(
       data='data.csv',
       task='classification',
       target='label',
       config={
           'model_type': 'neural_network',
           'epochs': 100,
           'batch_size': 32
       }
   )

deploy()
~~~~~~~~

.. autofunction:: neurolite.deploy

Deploy a trained model to various platforms.

**Parameters:**

* **model** (*TrainedModel*) -- Trained model to deploy
* **platform** (*str*) -- Deployment platform ('api', 'docker', 'aws', etc.)
* **config** (*dict, optional*) -- Platform-specific configuration

**Returns:**

* **Endpoint** -- Deployment endpoint object

**Example:**

.. code-block:: python

   # Deploy as REST API
   endpoint = neurolite.deploy(model, platform='api', port=8080)
   
   # Deploy to AWS
   endpoint = neurolite.deploy(
       model, 
       platform='aws',
       config={'region': 'us-east-1'}
   )

load()
~~~~~~

.. autofunction:: neurolite.load

Load a previously saved model.

**Parameters:**

* **path** (*str*) -- Path to saved model file

**Returns:**

* **TrainedModel** -- Loaded model object

**Example:**

.. code-block:: python

   # Load saved model
   model = neurolite.load('my_model.pkl')
   
   # Make predictions
   predictions = model.predict(new_data)

Core Classes
------------

TrainedModel
~~~~~~~~~~~~

.. autoclass:: neurolite.core.TrainedModel
   :members:
   :undoc-members:
   :show-inheritance:

The main model class returned by the ``train()`` function.

**Methods:**

predict(data)
^^^^^^^^^^^^^

Make predictions on new data.

**Parameters:**

* **data** -- Input data for prediction

**Returns:**

* Predictions in appropriate format

**Example:**

.. code-block:: python

   # Single prediction
   prediction = model.predict('new_image.jpg')
   
   # Batch predictions
   predictions = model.predict(['img1.jpg', 'img2.jpg'])
   
   # DataFrame predictions
   predictions = model.predict(test_df)

evaluate(test_data=None)
^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate model performance.

**Parameters:**

* **test_data** (*optional*) -- Test dataset for evaluation

**Returns:**

* **dict** -- Dictionary of evaluation metrics

**Example:**

.. code-block:: python

   # Evaluate on validation set
   metrics = model.evaluate()
   
   # Evaluate on custom test set
   metrics = model.evaluate(test_data)
   
   print(f"Accuracy: {metrics['accuracy']:.3f}")

save(path)
^^^^^^^^^^

Save the model to disk.

**Parameters:**

* **path** (*str*) -- File path to save the model

**Example:**

.. code-block:: python

   # Save model
   model.save('my_model.pkl')
   
   # Save with timestamp
   import datetime
   timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   model.save(f'model_{timestamp}.pkl')

info()
^^^^^^

Get information about the model.

**Returns:**

* **dict** -- Model information and statistics

**Example:**

.. code-block:: python

   info = model.info()
   print(f"Model type: {info['model_type']}")
   print(f"Training time: {info['training_time']}")
   print(f"Parameters: {info['num_parameters']}")

plot_metrics()
^^^^^^^^^^^^^^

Plot training metrics.

**Example:**

.. code-block:: python

   # Plot training history
   model.plot_metrics()

plot_confusion_matrix()
^^^^^^^^^^^^^^^^^^^^^^^

Plot confusion matrix for classification tasks.

**Example:**

.. code-block:: python

   # Plot confusion matrix
   model.plot_confusion_matrix()

plot_feature_importance()
^^^^^^^^^^^^^^^^^^^^^^^^^

Plot feature importance for applicable models.

**Example:**

.. code-block:: python

   # Plot feature importance
   model.plot_feature_importance()

Endpoint
~~~~~~~~

.. autoclass:: neurolite.deployment.Endpoint
   :members:
   :undoc-members:
   :show-inheritance:

Represents a deployed model endpoint.

**Attributes:**

* **url** (*str*) -- Endpoint URL
* **status** (*str*) -- Deployment status
* **platform** (*str*) -- Deployment platform

**Methods:**

predict(data)
^^^^^^^^^^^^^

Make predictions via the endpoint.

**Parameters:**

* **data** -- Input data for prediction

**Returns:**

* Prediction results

**Example:**

.. code-block:: python

   # Deploy model
   endpoint = neurolite.deploy(model, platform='api')
   
   # Make predictions via endpoint
   prediction = endpoint.predict(new_data)

stop()
^^^^^^

Stop the deployed endpoint.

**Example:**

.. code-block:: python

   # Stop the endpoint
   endpoint.stop()

Configuration
-------------

Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. automodule:: neurolite.config
   :members:
   :undoc-members:
   :show-inheritance:

**Functions:**

set_device(device)
^^^^^^^^^^^^^^^^^^

Set the default device for training.

**Parameters:**

* **device** (*str*) -- Device to use ('cpu', 'gpu', 'auto')

**Example:**

.. code-block:: python

   import neurolite
   
   # Use GPU if available
   neurolite.config.set_device('gpu')
   
   # Force CPU usage
   neurolite.config.set_device('cpu')

set_cache_dir(path)
^^^^^^^^^^^^^^^^^^^

Set the cache directory for models and data.

**Parameters:**

* **path** (*str*) -- Path to cache directory

**Example:**

.. code-block:: python

   # Set custom cache directory
   neurolite.config.set_cache_dir('./my_cache')

set_log_level(level)
^^^^^^^^^^^^^^^^^^^^

Set the logging level.

**Parameters:**

* **level** (*str*) -- Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

**Example:**

.. code-block:: python

   # Set verbose logging
   neurolite.config.set_log_level('DEBUG')

Exceptions
----------

.. automodule:: neurolite.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

NeuroLiteError
~~~~~~~~~~~~~~

Base exception class for all NeuroLite errors.

DataError
~~~~~~~~~

Raised when there are issues with data loading or processing.

**Example:**

.. code-block:: python

   try:
       model = neurolite.train('invalid_data.csv', task='classification')
   except neurolite.DataError as e:
       print(f"Data error: {e}")
       print(f"Suggestion: {e.suggestion}")

ModelError
~~~~~~~~~~

Raised when there are issues with model training or inference.

ConfigError
~~~~~~~~~~~

Raised when there are configuration issues.

DeploymentError
~~~~~~~~~~~~~~~

Raised when there are deployment issues.

Utilities
---------

.. automodule:: neurolite.utils
   :members:
   :undoc-members:
   :show-inheritance:

**Functions:**

get_version()
~~~~~~~~~~~~~

Get the current NeuroLite version.

**Returns:**

* **str** -- Version string

**Example:**

.. code-block:: python

   import neurolite
   print(f"NeuroLite version: {neurolite.get_version()}")

list_models()
~~~~~~~~~~~~~

List available model types.

**Returns:**

* **list** -- List of available model types

**Example:**

.. code-block:: python

   models = neurolite.list_models()
   print("Available models:", models)

list_tasks()
~~~~~~~~~~~~

List supported tasks.

**Returns:**

* **list** -- List of supported tasks

**Example:**

.. code-block:: python

   tasks = neurolite.list_tasks()
   print("Supported tasks:", tasks)