"""
Example: Basic Text Classification

This example demonstrates how to train a text classification model
for sentiment analysis using NeuroLite.

Requirements:
- neurolite
- pandas
- matplotlib
- seaborn

Usage:
    python text_classification.py
    python text_classification.py --data custom_data.csv --model bert
"""

import neurolite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def create_sample_dataset():
    """Create a sample movie review dataset."""
    print("Creating sample text dataset...")
    
    # Sample movie reviews with sentiment labels
    reviews_data = [
        # Positive reviews
        ("This movie is absolutely fantastic! Great acting and storyline.", "positive"),
        ("I loved every minute of this film. Highly recommended!", "positive"),
        ("Outstanding performance by the lead actor. Must watch!", "positive"),
        ("Brilliant cinematography and excellent direction.", "positive"),
        ("One of the best movies I've seen this year.", "positive"),
        ("Amazing special effects and compelling characters.", "positive"),
        ("Wonderful story that kept me engaged throughout.", "positive"),
        ("Superb acting and beautiful soundtrack.", "positive"),
        ("This film exceeded all my expectations.", "positive"),
        ("Perfect blend of action and emotion.", "positive"),
        ("Incredible movie with outstanding performances.", "positive"),
        ("Loved the plot twists and character development.", "positive"),
        ("Excellent movie with great visual effects.", "positive"),
        ("This is a masterpiece of modern cinema.", "positive"),
        ("Fantastic storytelling and amazing cast.", "positive"),
        ("Beautifully crafted film with deep meaning.", "positive"),
        ("Exceptional direction and powerful performances.", "positive"),
        ("This movie touched my heart and soul.", "positive"),
        ("Brilliant script and flawless execution.", "positive"),
        ("A cinematic gem that deserves all the praise.", "positive"),
        
        # Negative reviews
        ("This movie was terrible. Waste of time and money.", "negative"),
        ("Poor acting and confusing plot. Very disappointing.", "negative"),
        ("I couldn't even finish watching this boring film.", "negative"),
        ("Worst movie I've ever seen. Completely pointless.", "negative"),
        ("Bad direction and terrible screenplay.", "negative"),
        ("This film lacks any coherent storyline.", "negative"),
        ("Awful acting and poor production quality.", "negative"),
        ("Complete waste of time. Very boring and predictable.", "negative"),
        ("I regret watching this movie. Total disappointment.", "negative"),
        ("Poor character development and weak plot.", "negative"),
        ("This movie is painfully slow and uninteresting.", "negative"),
        ("Bad script and unconvincing performances.", "negative"),
        ("I fell asleep halfway through this boring film.", "negative"),
        ("Terrible movie with no redeeming qualities.", "negative"),
        ("Poorly executed and completely forgettable.", "negative"),
        ("Confusing plot and mediocre acting throughout.", "negative"),
        ("This film failed to deliver on its promises.", "negative"),
        ("Disappointing sequel that ruined the franchise.", "negative"),
        ("Overrated movie with shallow characters.", "negative"),
        ("Boring dialogue and predictable ending.", "negative"),
    ]
    
    # Create DataFrame
    df = pd.DataFrame(reviews_data, columns=['text', 'label'])
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('movie_reviews.csv', index=False)
    
    print(f"Created dataset with {len(df)} reviews")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    return df


def train_model(data_file='movie_reviews.csv', model_name='distilbert'):
    """Train a text classification model."""
    print(f"\nTraining text classification model with {model_name}...")
    
    # Train the model
    model = neurolite.train(
        data=data_file,
        model=model_name,
        task='text_classification',
        target='label',
        max_length=128,
        validation_split=0.2,
        test_split=0.1,
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
    
    # Plot confusion matrix if available
    if hasattr(model.evaluation_results, 'confusion_matrix') and model.evaluation_results.confusion_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            model.evaluation_results.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['negative', 'positive'],
            yticklabels=['negative', 'positive']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()


def make_predictions(model):
    """Make predictions on new text samples."""
    print("\nMaking predictions on new text samples...")
    
    # Test samples
    test_texts = [
        "This movie is absolutely incredible! I loved every second of it.",
        "Boring and predictable. I want my money back.",
        "The acting was decent but the plot was confusing.",
        "Amazing cinematography and outstanding performances by all actors.",
        "I fell asleep during the movie. Very disappointing.",
        "This film is a masterpiece of storytelling and visual effects.",
        "The movie was okay, nothing special but not terrible either.",
        "Worst film I've ever seen. Complete waste of time.",
        "Great movie with excellent character development.",
        "Poor script and unconvincing dialogue throughout."
    ]
    
    # Make predictions
    predictions = model.predict(test_texts)
    
    print("\nPredictions:")
    print("============")
    
    for i, (text, prediction) in enumerate(zip(test_texts, predictions)):
        # Truncate long text for display
        display_text = text if len(text) <= 60 else text[:57] + "..."
        print(f"{i+1:2d}. \"{display_text}\"")
        print(f"    Predicted: {prediction}")
        print()


def analyze_predictions(model):
    """Analyze model predictions with confidence scores."""
    print("Analyzing prediction confidence...")
    
    # Test texts with varying sentiment strength
    test_cases = [
        ("This movie is absolutely amazing and perfect!", "Strong Positive"),
        ("I really enjoyed this film.", "Moderate Positive"),
        ("The movie was okay.", "Neutral"),
        ("I didn't like this movie much.", "Moderate Negative"),
        ("This is the worst movie ever made!", "Strong Negative"),
    ]
    
    predictions = model.predict([text for text, _ in test_cases])
    
    print("\nPrediction Analysis:")
    print("===================")
    
    for (text, expected), prediction in zip(test_cases, predictions):
        print(f"Text: \"{text}\"")
        print(f"Expected: {expected}")
        print(f"Predicted: {prediction}")
        print("-" * 50)


def compare_models():
    """Compare different text classification models."""
    print("\nComparing different models...")
    
    models_to_test = ['distilbert', 'bert']  # Add more models as available
    results = {}
    
    for model_name in models_to_test:
        print(f"Training {model_name}...")
        try:
            model = neurolite.train(
                data='movie_reviews.csv',
                model=model_name,
                task='text_classification',
                target='label',
                max_length=64,  # Shorter for faster training
                validation_split=0.2,
                optimize=False
            )
            
            accuracy = model.evaluation_results.metrics.get('accuracy', 0.0)
            f1_score = model.evaluation_results.metrics.get('f1', 0.0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1': f1_score
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
            
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
            results[model_name] = {'accuracy': 0.0, 'f1': 0.0}
    
    # Plot comparison
    if len(results) > 1:
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add value labels
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
            plt.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()


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
    print("\nAPI Usage:")
    print("POST /predict")
    print("Content-Type: application/json")
    print("Body: {\"text\": \"Your text to classify\"}")
    print("Response: {\"prediction\": \"positive\", \"confidence\": 0.95}")


def main():
    """Main function to run the complete example."""
    parser = argparse.ArgumentParser(description='Text Classification Example')
    parser.add_argument('--data', default='movie_reviews.csv', help='Path to data file')
    parser.add_argument('--model', default='distilbert', help='Model to use')
    parser.add_argument('--compare', action='store_true', help='Compare different models')
    
    args = parser.parse_args()
    
    print("NeuroLite Text Classification Example")
    print("====================================")
    
    # Step 1: Create sample dataset if it doesn't exist
    if args.data == 'movie_reviews.csv' and not pd.io.common.file_exists(args.data):
        create_sample_dataset()
    
    # Step 2: Train model
    model = train_model(args.data, args.model)
    
    # Step 3: Evaluate model
    evaluate_model(model)
    
    # Step 4: Make predictions
    make_predictions(model)
    
    # Step 5: Analyze predictions
    analyze_predictions(model)
    
    # Step 6: Compare models (optional)
    if args.compare:
        compare_models()
    
    # Step 7: Deploy model
    deploy_model(model)
    
    print("\nExample completed successfully!")
    print("Check generated plots for visual results.")


if __name__ == "__main__":
    main()