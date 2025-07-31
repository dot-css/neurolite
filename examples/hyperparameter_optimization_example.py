"""
Example demonstrating hyperparameter optimization with NeuroLite.

This example shows how to use the hyperparameter optimization functionality
to automatically find the best parameters for a machine learning model.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from neurolite.training.optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptimizationBounds,
    ResourceConstraints,
    SearchStrategy,
    PruningStrategy
)
from neurolite.models.ml.sklearn_models import create_random_forest_classifier, create_linear_regression
from neurolite.models.base import TaskType
from neurolite.data.detector import DataType


def create_sample_classification_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_sample_regression_data():
    """Create sample regression dataset."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def classification_model_factory(**kwargs):
    """Factory function for creating classification models."""
    # Extract model-specific parameters
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
    
    return create_random_forest_classifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )


def regression_model_factory(**kwargs):
    """Factory function for creating regression models."""
    return create_linear_regression()


def example_classification_optimization():
    """Example of hyperparameter optimization for classification."""
    print("=" * 60)
    print("Classification Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create sample data
    X_train, y_train, X_val, y_val, X_test, y_test = create_sample_classification_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Configure optimization bounds with model-specific parameters
    bounds = OptimizationBounds(
        learning_rate=(1e-4, 1e-1),  # Not used for RandomForest but kept for consistency
        batch_size=[32, 64, 128],    # Not used for RandomForest but kept for consistency
        epochs=(10, 50)              # Not used for RandomForest but kept for consistency
    )
    
    # Configure resource constraints for fast example
    constraints = ResourceConstraints(
        max_trials=10,           # Small number for quick demo
        max_time_minutes=2,      # 2 minutes max
        early_stopping_rounds=3
    )
    
    # Create optimization configuration
    config = OptimizationConfig(
        search_strategy=SearchStrategy.RANDOM,  # Random search for speed
        pruning_strategy=PruningStrategy.NONE,  # No pruning for simplicity
        optimization_bounds=bounds,
        resource_constraints=constraints,
        objective_metric="val_accuracy",
        objective_direction="maximize",
        use_cross_validation=False,  # Use validation split for speed
        verbose=True,
        custom_parameter_space={
            'n_estimators': {
                'type': 'int',
                'low': 10,
                'high': 200
            },
            'max_depth': {
                'type': 'int',
                'low': 3,
                'high': 20
            },
            'min_samples_split': {
                'type': 'int',
                'low': 2,
                'high': 20
            },
            'min_samples_leaf': {
                'type': 'int',
                'low': 1,
                'high': 10
            }
        }
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    print("\nStarting hyperparameter optimization...")
    result = optimizer.optimize(
        model_factory=classification_model_factory,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type=TaskType.CLASSIFICATION,
        data_type=DataType.TABULAR
    )
    
    # Print results
    print(f"\nOptimization completed!")
    print(f"Best validation accuracy: {result.best_value:.4f}")
    print(f"Best parameters: {result.best_params}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    
    # Train final model with best parameters
    if result.best_model:
        print("\nEvaluating best model on test set...")
        test_predictions = result.best_model.predict(X_test)
        test_accuracy = np.mean(test_predictions.predictions == y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get optimization insights
    insights = optimizer.get_optimization_insights()
    if 'param_importance' in insights:
        print("\nParameter importance:")
        for param, importance in insights['param_importance'].items():
            print(f"  {param}: {importance:.4f}")
    
    return result


def example_regression_optimization():
    """Example of hyperparameter optimization for regression."""
    print("\n" + "=" * 60)
    print("Regression Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Create sample data
    X_train, y_train, X_val, y_val, X_test, y_test = create_sample_regression_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Configure optimization for regression
    config = OptimizationConfig(
        search_strategy=SearchStrategy.RANDOM,
        pruning_strategy=PruningStrategy.NONE,
        resource_constraints=ResourceConstraints(
            max_trials=5,  # Very small for quick demo
            max_time_minutes=1
        ),
        objective_metric="val_loss",
        objective_direction="minimize",
        use_cross_validation=True,  # Use CV for regression
        cv_folds=3,
        verbose=True
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    print("\nStarting hyperparameter optimization...")
    result = optimizer.optimize(
        model_factory=regression_model_factory,
        X_train=X_train,
        y_train=y_train,
        task_type=TaskType.REGRESSION,
        data_type=DataType.TABULAR
    )
    
    # Print results
    print(f"\nOptimization completed!")
    print(f"Best validation loss: {result.best_value:.4f}")
    print(f"Best parameters: {result.best_params}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    
    return result


def main():
    """Run hyperparameter optimization examples."""
    print("NeuroLite Hyperparameter Optimization Examples")
    print("This example demonstrates automated hyperparameter search")
    print("using Optuna integration with different model types.\n")
    
    try:
        # Run classification example
        classification_result = example_classification_optimization()
        
        # Run regression example
        regression_result = example_regression_optimization()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("Both optimization examples completed successfully!")
        print(f"Classification best accuracy: {classification_result.best_value:.4f}")
        print(f"Regression best loss: {regression_result.best_value:.4f}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure Optuna is installed: pip install optuna")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()