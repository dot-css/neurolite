"""
Simple test to verify NeuroLite basic functionality.
"""

import pandas as pd
import numpy as np
from neurolite import DataProfiler

def test_basic_functionality():
    """Test basic NeuroLite functionality."""
    print("Testing NeuroLite basic functionality...")
    
    # Create sample data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 55000, 60000, 52000, 58000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing']
    })
    
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data:\n{df}\n")
    
    # Initialize profiler
    profiler = DataProfiler()
    
    # Perform analysis
    print("Performing analysis...")
    results = profiler.analyze(df)
    
    # Display results
    print("=== Analysis Results ===")
    print(f"Data structure type: {results['data_structure'].structure_type}")
    print(f"Dimensions: {results['data_structure'].dimensions}")
    print(f"Completeness: {results['quality_metrics'].completeness:.2%}")
    print(f"Uniqueness: {results['quality_metrics'].uniqueness:.2%}")
    print(f"Missing pattern: {results['quality_metrics'].missing_pattern}")
    
    print("\nColumn types:")
    for col, col_type in results['column_analysis'].items():
        print(f"  {col}: {col_type.primary_type} ({col_type.subtype}) [{col_type.confidence:.2%}]")
    
    print(f"\nMissing data percentage: {results['missing_analysis'].missing_percentage:.2%}")
    print(f"Imputation strategy: {results['missing_analysis'].imputation_strategy}")
    
    print("\n✅ Basic functionality test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()