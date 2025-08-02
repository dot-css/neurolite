Image Classification Tutorial
=============================

This tutorial will guide you through building an image classification model with NeuroLite.

Overview
--------

In this tutorial, you'll learn how to:

- Prepare image data for training
- Train an image classification model
- Evaluate model performance
- Make predictions on new images
- Deploy your model

Prerequisites
-------------

- NeuroLite installed (``pip install neurolite``)
- Basic Python knowledge
- Image dataset (we'll use a sample dataset)

Dataset Preparation
-------------------

First, let's organize our image data. NeuroLite expects images to be organized in folders by class:

.. code-block:: text

   images/
   ├── cats/
   │   ├── cat1.jpg
   │   ├── cat2.jpg
   │   └── ...
   ├── dogs/
   │   ├── dog1.jpg
   │   ├── dog2.jpg
   │   └── ...
   └── birds/
       ├── bird1.jpg
       ├── bird2.jpg
       └── ...

Download Sample Dataset
~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we'll use a sample dataset:

.. code-block:: python

   import neurolite
   
   # Download sample image dataset
   dataset_path = neurolite.datasets.download('animals_sample')
   print(f"Dataset downloaded to: {dataset_path}")

Basic Image Classification
--------------------------

Let's start with the simplest approach:

.. code-block:: python

   import neurolite
   
   # Train an image classification model
   model = neurolite.train(
       data="images/",
       task="image_classification"
   )
   
   print("Training completed!")

That's it! NeuroLite automatically:

- Detected the image classes from folder names
- Chose an appropriate CNN architecture
- Preprocessed the images (resizing, normalization)
- Split data into train/validation sets
- Trained the model with optimal hyperparameters

Making Predictions
------------------

Now let's use our trained model to classify new images:

.. code-block:: python

   # Predict a single image
   prediction = model.predict("test_images/unknown_animal.jpg")
   print(f"Prediction: {prediction}")
   
   # Predict multiple images
   predictions = model.predict([
       "test_images/img1.jpg",
       "test_images/img2.jpg",
       "test_images/img3.jpg"
   ])
   
   for i, pred in enumerate(predictions):
       print(f"Image {i+1}: {pred}")

Getting Prediction Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get confidence scores for predictions:

.. code-block:: python

   # Get probabilities for all classes
   probabilities = model.predict_proba("test_images/unknown_animal.jpg")
   
   for class_name, prob in probabilities.items():
       print(f"{class_name}: {prob:.3f}")

Model Evaluation
----------------

Let's evaluate our model's performance:

.. code-block:: python

   # Get evaluation metrics
   metrics = model.evaluate()
   
   print(f"Accuracy: {metrics['accuracy']:.3f}")
   print(f"Precision: {metrics['precision']:.3f}")
   print(f"Recall: {metrics['recall']:.3f}")
   print(f"F1 Score: {metrics['f1_score']:.3f}")

Visualizing Results
~~~~~~~~~~~~~~~~~~~

NeuroLite provides built-in visualization tools:

.. code-block:: python

   # Plot training metrics
   model.plot_metrics()
   
   # Plot confusion matrix
   model.plot_confusion_matrix()
   
   # Plot sample predictions
   model.plot_predictions(num_samples=9)

Advanced Configuration
----------------------

For more control over the training process:

.. code-block:: python

   model = neurolite.train(
       data="images/",
       task="image_classification",
       config={
           "model_type": "resnet50",        # Specific architecture
           "epochs": 50,                    # Training epochs
           "batch_size": 32,                # Batch size
           "learning_rate": 0.001,          # Learning rate
           "image_size": (224, 224),        # Input image size
           "validation_split": 0.2,         # Validation split
           "data_augmentation": True,       # Enable augmentation
           "early_stopping": True,          # Early stopping
           "save_best_only": True          # Save best model only
       }
   )

Data Augmentation
~~~~~~~~~~~~~~~~~

Enable data augmentation to improve model generalization:

.. code-block:: python

   model = neurolite.train(
       data="images/",
       task="image_classification",
       config={
           "data_augmentation": {
               "rotation_range": 20,
               "width_shift_range": 0.2,
               "height_shift_range": 0.2,
               "horizontal_flip": True,
               "zoom_range": 0.2,
               "brightness_range": [0.8, 1.2]
           }
       }
   )

Transfer Learning
~~~~~~~~~~~~~~~~~

Use pre-trained models for better performance:

.. code-block:: python

   model = neurolite.train(
       data="images/",
       task="image_classification",
       config={
           "model_type": "efficientnet_b0",
           "pretrained": True,              # Use pre-trained weights
           "freeze_base": True,             # Freeze base layers
           "fine_tune_epochs": 10           # Fine-tuning epochs
       }
   )

Hyperparameter Optimization
---------------------------

Let NeuroLite find the best hyperparameters automatically:

.. code-block:: python

   model = neurolite.train(
       data="images/",
       task="image_classification",
       optimization="bayesian",
       trials=50,                          # Number of trials
       timeout=7200                        # 2 hours timeout
   )
   
   # Get the best hyperparameters
   best_params = model.get_best_params()
   print("Best hyperparameters:", best_params)

Working with CSV Data
---------------------

If your image paths are in a CSV file:

.. code-block:: python

   # CSV format:
   # image_path,label
   # images/cat1.jpg,cat
   # images/dog1.jpg,dog
   
   model = neurolite.train(
       data="image_data.csv",
       task="image_classification",
       target="label"
   )

Custom Image Preprocessing
--------------------------

Customize image preprocessing:

.. code-block:: python

   model = neurolite.train(
       data="images/",
       task="image_classification",
       config={
           "preprocessing": {
               "resize": (256, 256),
               "crop": (224, 224),
               "normalize": "imagenet",      # ImageNet normalization
               "convert_to_rgb": True
           }
       }
   )

Model Deployment
----------------

Deploy your trained model:

.. code-block:: python

   # Deploy as REST API
   endpoint = neurolite.deploy(
       model,
       platform="api",
       port=8080
   )
   
   print(f"Model deployed at: http://localhost:8080")

Test the deployed model:

.. code-block:: bash

   # Test with curl
   curl -X POST http://localhost:8080/predict \
        -F "file=@test_image.jpg"

Or using Python:

.. code-block:: python

   import requests
   
   # Send image for prediction
   with open("test_image.jpg", "rb") as f:
       response = requests.post(
           "http://localhost:8080/predict",
           files={"file": f}
       )
   
   prediction = response.json()
   print(f"Prediction: {prediction}")

Saving and Loading Models
-------------------------

Save your trained model:

.. code-block:: python

   # Save the model
   model.save("image_classifier.pkl")
   
   # Load the model later
   loaded_model = neurolite.load("image_classifier.pkl")
   
   # Use loaded model
   prediction = loaded_model.predict("new_image.jpg")

Export for Production
~~~~~~~~~~~~~~~~~~~~~

Export to different formats:

.. code-block:: python

   # Export to ONNX
   model.export("model.onnx", format="onnx")
   
   # Export to TensorFlow Lite
   model.export("model.tflite", format="tflite")
   
   # Export to TorchScript
   model.export("model.pt", format="torchscript")

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Out of Memory Error:**

.. code-block:: python

   # Reduce batch size
   model = neurolite.train(
       data="images/",
       task="image_classification",
       config={"batch_size": 16}  # Smaller batch size
   )

**Low Accuracy:**

- Ensure images are properly labeled
- Try data augmentation
- Use transfer learning
- Increase training epochs
- Check for class imbalance

**Slow Training:**

- Use GPU if available
- Reduce image size
- Increase batch size
- Use mixed precision training

Performance Tips
----------------

1. **Use GPU**: Enable GPU acceleration for faster training
2. **Batch Size**: Larger batch sizes can improve training speed
3. **Image Size**: Smaller images train faster but may reduce accuracy
4. **Transfer Learning**: Use pre-trained models for better results
5. **Data Augmentation**: Helps prevent overfitting

Next Steps
----------

- Try :doc:`object_detection` for detecting objects in images
- Learn about :doc:`text_classification` for NLP tasks
- Explore :doc:`../examples/advanced_examples` for complex scenarios
- Check out :doc:`../user_guide/deployment` for production deployment