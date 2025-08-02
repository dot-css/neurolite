Quick Start Guide
=================

Welcome to NeuroLite! This guide will get you up and running with your first machine learning model in just 5 minutes.

Your First Model
-----------------

Let's start with a simple example - training an image classifier:

.. code-block:: python

   import neurolite

   # Train an image classification model
   model = neurolite.train(
       data="path/to/your/images",
       task="image_classification"
   )

   # Make predictions
   prediction = model.predict("path/to/test_image.jpg")
   print(f"Prediction: {prediction}")

That's it! NeuroLite automatically:

- Detected your data format
- Chose the best model architecture
- Preprocessed your images
- Trained the model with optimal hyperparameters
- Made it ready for predictions

Text Classification Example
---------------------------

Working with text data is just as simple:

.. code-block:: python

   import neurolite

   # Train a sentiment analysis model
   model = neurolite.train(
       data="reviews.csv",
       task="sentiment_analysis",
       target="sentiment"
   )

   # Analyze sentiment
   result = model.predict("This product is amazing!")
   print(f"Sentiment: {result}")

Tabular Data Example
--------------------

For structured/tabular data:

.. code-block:: python

   import neurolite

   # Train a regression model
   model = neurolite.train(
       data="sales_data.csv",
       task="regression",
       target="revenue"
   )

   # Make predictions
   forecast = model.predict({
       "marketing_spend": 10000,
       "season": "summer",
       "region": "north"
   })
   print(f"Predicted revenue: ${forecast:,.2f}")

Deploy Your Model
-----------------

Once you're happy with your model, deploy it instantly:

.. code-block:: python

   # Deploy as a REST API
   endpoint = neurolite.deploy(
       model,
       platform="api",
       port=8080
   )

   print(f"Model deployed at: http://localhost:8080")

Your model is now accessible via HTTP requests:

.. code-block:: bash

   curl -X POST http://localhost:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"data": "your input data"}'

Advanced Configuration
----------------------

For more control over the training process:

.. code-block:: python

   import neurolite

   model = neurolite.train(
       data="data.csv",
       task="classification",
       target="label",
       config={
           "model_type": "neural_network",
           "epochs": 50,
           "batch_size": 32,
           "learning_rate": 0.001,
           "validation_split": 0.2
       }
   )

Hyperparameter Optimization
----------------------------

Let NeuroLite find the best hyperparameters automatically:

.. code-block:: python

   model = neurolite.train(
       data="data.csv",
       task="classification",
       optimization="bayesian",  # or "grid", "random"
       trials=100,
       timeout=3600  # 1 hour
   )

Working with Different Data Formats
------------------------------------

NeuroLite supports various data formats:

Images
~~~~~~

.. code-block:: python

   # From directory structure
   model = neurolite.train("images/", task="image_classification")

   # From CSV with image paths
   model = neurolite.train("image_data.csv", task="image_classification")

Text
~~~~

.. code-block:: python

   # CSV file
   model = neurolite.train("text_data.csv", task="text_classification")

   # JSON file
   model = neurolite.train("data.json", task="sentiment_analysis")

   # Direct text files
   model = neurolite.train("documents/", task="document_classification")

Tabular
~~~~~~~

.. code-block:: python

   # CSV file
   model = neurolite.train("data.csv", task="regression")

   # Excel file
   model = neurolite.train("data.xlsx", task="classification")

   # Parquet file
   model = neurolite.train("data.parquet", task="clustering")

Model Evaluation
----------------

Evaluate your model's performance:

.. code-block:: python

   # Get detailed metrics
   metrics = model.evaluate()
   print(f"Accuracy: {metrics['accuracy']:.3f}")
   print(f"F1 Score: {metrics['f1_score']:.3f}")

   # Visualize results
   model.plot_metrics()
   model.plot_confusion_matrix()

Saving and Loading Models
-------------------------

Save your trained models for later use:

.. code-block:: python

   # Save model
   model.save("my_model.pkl")

   # Load model
   loaded_model = neurolite.load("my_model.pkl")

   # Make predictions with loaded model
   prediction = loaded_model.predict(new_data)

Common Tasks
------------

Here are some common machine learning tasks you can accomplish with NeuroLite:

Classification Tasks
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Binary classification
   model = neurolite.train("data.csv", task="binary_classification")

   # Multi-class classification
   model = neurolite.train("data.csv", task="classification")

   # Multi-label classification
   model = neurolite.train("data.csv", task="multilabel_classification")

Regression Tasks
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple regression
   model = neurolite.train("data.csv", task="regression")

   # Time series forecasting
   model = neurolite.train("timeseries.csv", task="forecasting")

Clustering
~~~~~~~~~~

.. code-block:: python

   # Unsupervised clustering
   model = neurolite.train("data.csv", task="clustering")

   # Get cluster assignments
   clusters = model.predict(new_data)

Next Steps
----------

Now that you've got the basics down, explore:

- :doc:`basic_concepts` - Understand NeuroLite's core concepts
- :doc:`../user_guide/computer_vision` - Deep dive into computer vision
- :doc:`../user_guide/nlp` - Natural language processing guide
- :doc:`../tutorials/image_classification` - Detailed tutorials
- :doc:`../examples/basic_examples` - More examples to learn from

Tips for Success
-----------------

1. **Start Simple**: Begin with the default settings and gradually add complexity
2. **Quality Data**: Good data is more important than complex models
3. **Iterate Quickly**: Use NeuroLite's speed to try many approaches
4. **Monitor Performance**: Use built-in evaluation tools to track progress
5. **Deploy Early**: Get your model in production quickly and improve iteratively

Need Help?
----------

- Check the :doc:`../troubleshooting` guide for common issues
- Browse :doc:`../examples/basic_examples` for more examples
- Visit our `GitHub repository <https://github.com/dot-css/neurolite>`_ for support