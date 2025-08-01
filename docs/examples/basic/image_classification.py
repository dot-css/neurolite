"""
Example: Basic Image Classification

This example demonstrates how to train an image classification model
using NeuroLite with minimal code.

Requirements:
- neurolite
- PIL (Pillow)
- matplotlib

Usage:
    python image_classification.py
"""

import neurolite
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_sample_dataset():
    """Create a simple sample dataset for demonstration."""
    print("Creating sample dataset...")
    
    # Create directories
    os.makedirs('sample_data/cats', exist_ok=True)
    os.makedirs('sample_data/dogs', exist_ok=True)
    
    np.random.seed(42)
    
    # Generate synthetic "cat" images (blue-tinted)
    for i in range(50):
        img = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
        img[:, :, 2] += 100  # Add blue tint
        img = np.clip(img, 0, 255)
        Image.fromarray(img).save(f'sample_data/cats/cat_{i:03d}.jpg')
    
    # Generate synthetic "dog" images (red-tinted)
    for i in range(50):
        img = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
        img[:, :, 0] += 100  # Add red tint
        img = np.clip(img, 0, 255)
        Image.fromarray(img).save(f'sample_data/dogs/dog_{i:03d}.jpg')
    
    print(f"Created {len(os.listdir('sample_data/cats'))} cat images")
    print(f"Created {len(os.listdir('sample_data/dogs'))} dog images")


def train_model():
    """Train an image classification model."""
    print("\nTraining image classification model...")
    
    # Train with minimal configuration
    model = neurolite.train(
        data='sample_data/',
        task='image_classification',
        model='resnet18',
        image_size=64,
        validation_split=0.2,
        optimize=False  # Skip optimization for faster demo
    )
    
    print("Training completed!")
    return model


def evaluate_model(model):
    """Evaluate the trained model."""
    print("\nModel Evaluation:")
    print("================")
    
    metrics = model.evaluation_results.metrics
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name.capitalize()}: {value:.4f}")
        else:
            print(f"{metric_name.capitalize()}: {value}")


def make_predictions(model):
    """Make predictions on new images."""
    print("\nMaking predictions on test images...")
    
    # Create test images
    np.random.seed(123)
    
    # Test cat image (blue-tinted)
    test_cat = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
    test_cat[:, :, 2] += 100
    test_cat = np.clip(test_cat, 0, 255)
    
    # Test dog image (red-tinted)
    test_dog = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
    test_dog[:, :, 0] += 100
    test_dog = np.clip(test_dog, 0, 255)
    
    # Make predictions
    test_images = [test_cat, test_dog]
    predictions = model.predict(test_images)
    
    # Display results
    class_names = ['cats', 'dogs']
    
    plt.figure(figsize=(10, 4))
    
    for i, (img, pred) in enumerate(zip(test_images, predictions)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        predicted_class = class_names[pred] if isinstance(pred, (int, np.integer)) else pred
        plt.title(f'Predicted: {predicted_class}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    
    print("Predictions:")
    for i, pred in enumerate(predictions):
        predicted_class = class_names[pred] if isinstance(pred, (int, np.integer)) else pred
        print(f"Test image {i+1}: {predicted_class}")


def deploy_model(model):
    """Deploy the model for inference."""
    print("\nDeploying model...")
    
    try:
        # Export to ONNX format
        exported_model = neurolite.deploy(model, format='onnx')
        print("✓ Model exported to ONNX format")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    print("\nTo deploy as API:")
    print("api_server = neurolite.deploy(model, format='api', port=8080)")
    print("# Access at http://localhost:8080/predict")


def main():
    """Main function to run the complete example."""
    print("NeuroLite Image Classification Example")
    print("=====================================")
    
    # Step 1: Create sample dataset
    if not os.path.exists('sample_data'):
        create_sample_dataset()
    else:
        print("Sample dataset already exists, skipping creation...")
    
    # Step 2: Train model
    model = train_model()
    
    # Step 3: Evaluate model
    evaluate_model(model)
    
    # Step 4: Make predictions
    make_predictions(model)
    
    # Step 5: Deploy model
    deploy_model(model)
    
    print("\nExample completed successfully!")
    print("Check 'predictions.png' to see the prediction results.")


if __name__ == "__main__":
    main()