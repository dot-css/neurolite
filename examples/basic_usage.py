"""
Basic usage examples for NeuroLite.

This script demonstrates the core functionality of NeuroLite including
data quality assessment, type detection, and basic analysis.
"""

import pandas as pd
import numpy as np
from neurolite.detectors import QualityDetector, DataTypeDetector, FileDetector


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create a sample dataset with various data quality issues
    data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'age': [25, 30, 35, 28, np.nan, 45, 32, 38, 29, 41],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@invalid', 
                 'david@example.com', 'eve@example.com', 'frank@example.com',
                 'grace@example.com', 'henry@example.com', 'ivy@example.com', 'jack@example.com'],
        'salary': [50000, 55000, 60000, 52000, 58000, 75000, 48000, 62000, 51000, 68000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Marketing',
                      'Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales'],
        'join_date': ['2020-01-15', '2019-03-22', '2021-07-10', '2020-11-05', '2019-09-18',
                     '2018-12-03', '2021-02-28', '2020-06-14', '2019-11-30', '2021-04-12']
    }
    
    return pd.DataFrame(data)


def demonstrate_quality_detection():
    """Demonstrate quality detection capabilities."""
    print("=" * 60)
    print("QUALITY DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Sample dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}\n")
    
    # Initialize quality detector
    quality_detector = QualityDetector()
    
    # Perform comprehensive quality analysis
    print("1. Comprehensive Quality Analysis")
    print("-" * 40)
    quality_metrics = quality_detector.analyze_quality(df)
    
    print(f"Completeness: {quality_metrics.completeness:.2%}")
    print(f"Consistency: {quality_metrics.consistency:.2%}")
    print(f"Validity: {quality_metrics.validity:.2%}")
    print(f"Uniqueness: {quality_metrics.uniqueness:.2%}")
    print(f"Missing Pattern: {quality_metrics.missing_pattern}")
    print(f"Duplicate Count: {quality_metrics.duplicate_count}")
    print()
    
    # Analyze missing data patterns
    print("2. Missing Data Analysis")
    print("-" * 40)
    missing_analysis = quality_detector.detect_missing_patterns(df)
    
    print(f"Missing Percentage: {missing_analysis.missing_percentage:.2%}")
    print(f"Missing Pattern Type: {missing_analysis.missing_pattern_type}")
    print(f"Missing Columns: {missing_analysis.missing_columns}")
    print(f"Imputation Strategy: {missing_analysis.imputation_strategy}")
    print()
    
    # Find duplicates
    print("3. Duplicate Detection")
    print("-" * 40)
    duplicate_analysis = quality_detector.find_duplicates(df)
    
    print(f"Total Duplicates: {duplicate_analysis.duplicate_count}")
    print(f"Duplicate Percentage: {duplicate_analysis.duplicate_percentage:.2%}")
    print(f"Exact Duplicates: {duplicate_analysis.exact_duplicates}")
    print(f"Partial Duplicates: {duplicate_analysis.partial_duplicates}")
    print()
    
    # Validate consistency
    print("4. Consistency Validation")
    print("-" * 40)
    consistency_report = quality_detector.validate_consistency(df)
    
    print(f"Format Consistency Score: {consistency_report.format_consistency_score:.2%}")
    print(f"Range Consistency Score: {consistency_report.range_consistency_score:.2%}")
    print(f"Referential Integrity Score: {consistency_report.referential_integrity_score:.2%}")
    
    if consistency_report.inconsistent_formats:
        print("Format Issues:")
        for col, issues in consistency_report.inconsistent_formats.items():
            print(f"  {col}: {issues}")
    
    if consistency_report.integrity_violations:
        print("Integrity Violations:")
        for violation in consistency_report.integrity_violations:
            print(f"  - {violation}")
    print()


def demonstrate_type_detection():
    """Demonstrate data type detection capabilities."""
    print("=" * 60)
    print("DATA TYPE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize type detector
    type_detector = DataTypeDetector()
    
    # Classify columns
    print("Column Type Classification")
    print("-" * 40)
    column_types = type_detector.classify_columns(df)
    
    for column, col_type in column_types.items():
        print(f"{column:15} -> {col_type.primary_type:12} ({col_type.subtype}) "
              f"[confidence: {col_type.confidence:.2%}]")
    print()
    
    # Analyze specific columns
    print("Detailed Column Analysis")
    print("-" * 40)
    
    # Numerical analysis
    if 'age' in df.columns:
        numerical_analysis = type_detector.analyze_numerical(df['age'])
        print(f"Age column analysis:")
        print(f"  Data type: {numerical_analysis.data_type}")
        print(f"  Is continuous: {numerical_analysis.is_continuous}")
        print(f"  Range: {numerical_analysis.range_min} - {numerical_analysis.range_max}")
        print(f"  Distribution: {numerical_analysis.distribution_type}")
        print(f"  Outliers: {numerical_analysis.outlier_count}")
        print()
    
    # Categorical analysis
    if 'department' in df.columns:
        categorical_analysis = type_detector.analyze_categorical(df['department'])
        print(f"Department column analysis:")
        print(f"  Category type: {categorical_analysis.category_type}")
        print(f"  Cardinality: {categorical_analysis.cardinality}")
        print(f"  Unique values: {categorical_analysis.unique_values}")
        print(f"  Encoding recommendation: {categorical_analysis.encoding_recommendation}")
        print()
    
    # Temporal analysis
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date'])
        temporal_analysis = type_detector.analyze_temporal(df['join_date'])
        print(f"Join date column analysis:")
        print(f"  DateTime format: {temporal_analysis.datetime_format}")
        print(f"  Has seasonality: {temporal_analysis.has_seasonality}")
        print(f"  Has trend: {temporal_analysis.has_trend}")
        print(f"  Is stationary: {temporal_analysis.is_stationary}")
        print(f"  Time range: {temporal_analysis.time_range[0]} to {temporal_analysis.time_range[1]}")
        print()


def demonstrate_file_detection():
    """Demonstrate file detection capabilities."""
    print("=" * 60)
    print("FILE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize file detector
    file_detector = FileDetector()
    
    # Detect data structure
    print("Data Structure Detection")
    print("-" * 40)
    data_structure = file_detector.detect_structure(df)
    
    print(f"Structure type: {data_structure.structure_type}")
    print(f"Dimensions: {data_structure.dimensions}")
    print(f"Sample size: {data_structure.sample_size}")
    print(f"Memory usage: {data_structure.memory_usage} bytes")
    print()


def main():
    """Run all demonstrations."""
    print("NeuroLite Basic Usage Examples")
    print("=" * 60)
    print("This script demonstrates the core functionality of NeuroLite.")
    print("=" * 60)
    print()
    
    try:
        # Run demonstrations
        demonstrate_quality_detection()
        demonstrate_type_detection()
        demonstrate_file_detection()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All examples completed successfully!")
        print("For more advanced usage, see the documentation at:")
        print("https://neurolite.readthedocs.io/")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Please check your NeuroLite installation.")


if __name__ == "__main__":
    main()