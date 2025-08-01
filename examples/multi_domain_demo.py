#!/usr/bin/env python3
"""
Multi-Domain Task Support Integration Demo

This script demonstrates the multi-domain workflow coordination for computer vision,
NLP, and tabular data tasks with appropriate preprocessing, model selection, and
consistent API patterns.
"""

import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import NeuroLite components
from neurolite.workflows import create_workflow, get_workflow_factory
from neurolite.data import DataType
from neurolite.models import TaskType


def create_sample_tabular_data():
    """Create sample tabular data for demonstration."""
    print("Creating sample tabular data...")
    
    # Create synthetic tabular data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randint(0, 5, n_samples),
        'feature4': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    print(f"Created tabular data with {len(df)} samples at: {temp_file.name}")
    return temp_file.name


def create_sample_text_data():
    """Create sample text data for demonstration."""
    print("Creating sample text data...")
    
    # Create synthetic text data
    texts = [
        "This is a positive example of text classification.",
        "This is a negative example for sentiment analysis.",
        "Another positive text sample for testing.",
        "A negative sentiment text for classification.",
        "Neutral text that could go either way.",
    ] * 20  # Repeat to get 100 samples
    
    labels = [1, 0, 1, 0, 1] * 20  # Corresponding labels
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    print(f"Created text data with {len(df)} samples at: {temp_file.name}")
    return temp_file.name


def create_sample_image_data():
    """Create sample image directory structure for demonstration."""
    print("Creating sample image data structure...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create class directories
    for class_name in ['class_0', 'class_1']:
        class_dir = Path(temp_dir) / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create dummy image files
        for i in range(10):
            image_file = class_dir / f"image_{i}.jpg"
            # Write dummy image data (just bytes for demonstration)
            image_file.write_bytes(b"fake_image_data_" + str(i).encode())
    
    print(f"Created image data structure at: {temp_dir}")
    return temp_dir


def demonstrate_workflow_factory():
    """Demonstrate the workflow factory functionality."""
    print("\n" + "="*60)
    print("WORKFLOW FACTORY DEMONSTRATION")
    print("="*60)
    
    factory = get_workflow_factory()
    
    # Show supported data types
    supported_types = factory.get_supported_data_types()
    print(f"\nSupported data types: {[dt.value for dt in supported_types]}")
    
    # Show workflow information for each data type
    for data_type in supported_types:
        print(f"\n--- {data_type.value.upper()} WORKFLOW INFO ---")
        info = factory.get_workflow_info(data_type)
        print(f"Class: {info['class_name']}")
        print(f"Supported tasks: {info['supported_tasks']}")
        print(f"Default models: {info['default_models']}")


def demonstrate_tabular_workflow():
    """Demonstrate tabular data workflow."""
    print("\n" + "="*60)
    print("TABULAR DATA WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data_file = create_sample_tabular_data()
    
    try:
        # Create workflow configuration
        print("\nCreating tabular workflow...")
        workflow = create_workflow(
            data_path=data_file,
            model="random_forest_classifier",
            task="classification",
            target="target",
            validation_split=0.2,
            test_split=0.1,
            domain_config={
                'feature_engineering': True,
                'scaling': 'standard',
                'categorical_encoding': 'onehot'
            }
        )
        
        print(f"Created workflow: {workflow.__class__.__name__}")
        print(f"Supported data types: {[dt.value for dt in workflow.supported_data_types]}")
        print(f"Supported tasks: {[tt.value for tt in workflow.supported_tasks]}")
        print(f"Default models: {workflow.default_models}")
        
        # Validate configuration
        print("\nValidating workflow configuration...")
        workflow.validate_config()
        print("Configuration validation passed!")
        
    except Exception as e:
        print(f"Error in tabular workflow: {e}")
    finally:
        # Clean up
        if os.path.exists(data_file):
            os.unlink(data_file)


def demonstrate_nlp_workflow():
    """Demonstrate NLP workflow."""
    print("\n" + "="*60)
    print("NLP WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data_file = create_sample_text_data()
    
    try:
        # Create workflow configuration
        print("\nCreating NLP workflow...")
        workflow = create_workflow(
            data_path=data_file,
            model="bert",
            task="text_classification",
            target="label",
            validation_split=0.2,
            test_split=0.1,
            domain_config={
                'max_length': 512,
                'tokenizer': 'bert-base-uncased'
            }
        )
        
        print(f"Created workflow: {workflow.__class__.__name__}")
        print(f"Supported data types: {[dt.value for dt in workflow.supported_data_types]}")
        print(f"Supported tasks: {[tt.value for tt in workflow.supported_tasks]}")
        print(f"Default models: {workflow.default_models}")
        
        # Validate configuration
        print("\nValidating workflow configuration...")
        workflow.validate_config()
        print("Configuration validation passed!")
        
    except Exception as e:
        print(f"Error in NLP workflow: {e}")
    finally:
        # Clean up
        if os.path.exists(data_file):
            os.unlink(data_file)


def demonstrate_vision_workflow():
    """Demonstrate computer vision workflow."""
    print("\n" + "="*60)
    print("COMPUTER VISION WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    data_dir = create_sample_image_data()
    
    try:
        # Create workflow configuration
        print("\nCreating vision workflow...")
        workflow = create_workflow(
            data_path=data_dir,
            model="resnet18",
            task="image_classification",
            validation_split=0.2,
            test_split=0.1,
            domain_config={
                'image_size': (224, 224),
                'augmentation': True
            }
        )
        
        print(f"Created workflow: {workflow.__class__.__name__}")
        print(f"Supported data types: {[dt.value for dt in workflow.supported_data_types]}")
        print(f"Supported tasks: {[tt.value for tt in workflow.supported_tasks]}")
        print(f"Default models: {workflow.default_models}")
        
        # Validate configuration
        print("\nValidating workflow configuration...")
        workflow.validate_config()
        print("Configuration validation passed!")
        
    except Exception as e:
        print(f"Error in vision workflow: {e}")
    finally:
        # Clean up
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)


def demonstrate_api_consistency():
    """Demonstrate consistent API patterns across domains."""
    print("\n" + "="*60)
    print("API CONSISTENCY DEMONSTRATION")
    print("="*60)
    
    print("\nDemonstrating consistent API patterns across all domains:")
    
    # Show that all workflows have the same base interface
    from neurolite.workflows import VisionWorkflow, NLPWorkflow, TabularWorkflow
    from neurolite.workflows.base import WorkflowConfig
    
    workflows = [
        ("Vision", VisionWorkflow),
        ("NLP", NLPWorkflow),
        ("Tabular", TabularWorkflow)
    ]
    
    print("\nBase interface consistency:")
    for name, workflow_class in workflows:
        # Create dummy config
        config = WorkflowConfig(data_path="dummy")
        workflow = workflow_class(config)
        
        print(f"\n{name} Workflow:")
        print(f"  - Has execute() method: {hasattr(workflow, 'execute')}")
        print(f"  - Has validate_config() method: {hasattr(workflow, 'validate_config')}")
        print(f"  - Has supported_data_types property: {hasattr(workflow, 'supported_data_types')}")
        print(f"  - Has supported_tasks property: {hasattr(workflow, 'supported_tasks')}")
        print(f"  - Has default_models property: {hasattr(workflow, 'default_models')}")
    
    print("\nAll workflows implement the same base interface for consistent usage!")


def main():
    """Main demonstration function."""
    print("NeuroLite Multi-Domain Task Support Integration Demo")
    print("="*60)
    
    try:
        # Demonstrate workflow factory
        demonstrate_workflow_factory()
        
        # Demonstrate individual workflows
        demonstrate_tabular_workflow()
        demonstrate_nlp_workflow()
        demonstrate_vision_workflow()
        
        # Demonstrate API consistency
        demonstrate_api_consistency()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Multi-domain workflow factory")
        print("✓ Domain-specific preprocessing and model selection")
        print("✓ Consistent API patterns across all domains")
        print("✓ Configuration validation and error handling")
        print("✓ Tabular, NLP, and Computer Vision workflows")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()