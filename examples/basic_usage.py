"""
NeuroLite Basic Usage Examples

This script demonstrates the basic usage of NeuroLite for automated data analysis
and ML recommendations with minimal code.
"""

import pandas as pd
import numpy as np
import neurolite as nl
from pathlib import Path
import tempfile
import os


def create_sample_data():
    """Create sample datasets for demonstration."""
    print("Creating sample datasets...")
    
    # Create a sample CSV file
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'experience_years': np.random.randint(0, 40, 1000),
        'satisfaction_score': np.random.uniform(1, 10, 1000),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 1000),
        'performance_rating': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], 1000, 
                                             p=[0.1, 0.2, 0.4, 0.3]),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000],
        'is_remote': np.random.choice([True, False], 1000, p=[0.3, 0.7])
    }
    
    # Add some missing values
    data['income'][np.random.choice(1000, 50, replace=False)] = np.nan
    data['satisfaction_score'][np.random.choice(1000, 30, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    
    # Save to temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name, df


def example_1_basic_analysis():
    """Example 1: Basic analysis with minimal code."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Analysis (3 lines of code)")
    print("="*60)
    
    csv_file, df = create_sample_data()
    
    try:
        # The core NeuroLite promise: analyze any dataset in 3 lines
        import neurolite as nl
        report = nl.analyze(csv_file)
        print(f"Detected task: {report.task_identification.task_type}")
        
        print(f"\nFile format: {report.file_info.format_type}")
        print(f"Data structure: {report.data_structure.structure_type}")
        print(f"Dataset shape: {report.data_structure.dimensions}")
        print(f"Analysis completed in: {report.execution_time:.2f} seconds")
        
        print(f"\nData Quality Scores:")
        print(f"- Completeness: {report.quality_metrics.completeness:.1%}")
        print(f"- Consistency: {report.quality_metrics.consistency:.1%}")
        print(f"- Validity: {report.quality_metrics.validity:.1%}")
        print(f"- Uniqueness: {report.quality_metrics.uniqueness:.1%}")
        
        print(f"\nTop 3 Model Recommendations:")
        for i, rec in enumerate(report.model_recommendations[:3], 1):
            print(f"{i}. {rec.model_name} ({rec.confidence:.1%} confidence)")
            print(f"   Rationale: {rec.rationale}")
        
    finally:
        os.unlink(csv_file)


def example_2_quick_analysis():
    """Example 2: Quick analysis for fast exploration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Quick Analysis for Fast Exploration")
    print("="*60)
    
    csv_file, df = create_sample_data()
    
    try:
        # Quick analysis for initial data exploration
        quick_report = nl.quick_analyze(csv_file)
        
        print(f"Quick analysis completed in: {quick_report.execution_time:.2f} seconds")
        print(f"Dataset shape: {quick_report.basic_stats['shape']}")
        print(f"Memory usage: {quick_report.basic_stats['memory_usage_mb']:.1f} MB")
        
        print(f"\nMissing values by column:")
        for col, missing_count in quick_report.basic_stats['missing_values'].items():
            if missing_count > 0:
                print(f"- {col}: {missing_count} missing values")
        
        print(f"\nQuick recommendations:")
        for rec in quick_report.quick_recommendations:
            print(f"- {rec}")
            
    finally:
        os.unlink(csv_file)


def example_3_dataframe_analysis():
    """Example 3: Analyzing pandas DataFrames directly."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Analyzing Pandas DataFrames")
    print("="*60)
    
    # Create DataFrame directly
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_1': np.random.randn(500),
        'feature_2': np.random.randint(0, 100, 500),
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'target': np.random.choice([0, 1], 500)
    })
    
    # Analyze DataFrame directly
    report = nl.analyze(df)
    
    print(f"DataFrame analysis completed in: {report.execution_time:.2f} seconds")
    print(f"Detected task: {report.task_identification.task_type}")
    print(f"Task subtype: {report.task_identification.task_subtype}")
    
    print(f"\nColumn types detected:")
    for col_name, col_type in report.column_analysis.items():
        print(f"- {col_name}: {col_type.primary_type} ({col_type.confidence:.1%})")


def example_4_specialized_functions():
    """Example 4: Using specialized analysis functions."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Specialized Analysis Functions")
    print("="*60)
    
    csv_file, df = create_sample_data()
    
    try:
        # Data type detection
        print("Data Type Detection:")
        types = nl.detect_data_types(csv_file)
        for col, dtype in types.items():
            print(f"- {col}: {dtype}")
        
        print(f"\nData Quality Assessment:")
        quality = nl.assess_data_quality(csv_file)
        print(f"- Overall quality score: {quality['overall_score']:.1%}")
        print(f"- Missing pattern: {quality['missing_pattern']}")
        print(f"- Duplicate records: {quality['duplicate_count']}")
        
        print(f"\nModel Recommendations:")
        recommendations = nl.get_recommendations(csv_file)
        print(f"- Detected task: {recommendations['task_detected']}")
        print(f"- Confidence: {recommendations['confidence']:.1%}")
        print(f"- Recommended models: {recommendations['models']}")
        print(f"- Preprocessing steps: {recommendations['preprocessing']}")
        
    finally:
        os.unlink(csv_file)


def example_5_formatting_and_export():
    """Example 5: Formatting results and exporting reports."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Formatting and Exporting Results")
    print("="*60)
    
    csv_file, df = create_sample_data()
    
    try:
        # Perform analysis
        report = nl.analyze(csv_file)
        
        # Format summary in different formats
        print("Text Summary:")
        text_summary = nl.format_summary(report, 'text')
        print(text_summary[:500] + "..." if len(text_summary) > 500 else text_summary)
        
        # Create DataFrame summary
        print(f"\nDataFrame Summary:")
        summary_df = nl.create_dataframe_summary(report)
        print(summary_df.head(10))
        
        # Export to different formats
        print(f"\nExporting reports...")
        
        # Export to JSON
        nl.export_report(report, 'analysis_report.json', 'json')
        print("- Exported to analysis_report.json")
        
        # Export to HTML
        nl.export_report(report, 'analysis_report.html', 'html')
        print("- Exported to analysis_report.html")
        
        # Export to Markdown
        nl.export_report(report, 'analysis_report.md', 'markdown')
        print("- Exported to analysis_report.md")
        
        # Export DataFrame summary to CSV
        nl.export_report(report, 'analysis_summary.csv', 'csv')
        print("- Exported summary to analysis_summary.csv")
        
        print(f"\nReport files created successfully!")
        
    finally:
        os.unlink(csv_file)


def example_6_error_handling():
    """Example 6: Proper error handling."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Handling")
    print("="*60)
    
    # Test with non-existent file
    try:
        report = nl.analyze('non_existent_file.csv')
    except nl.UnsupportedFormatError as e:
        print(f"Format error: {e}")
    except nl.InsufficientDataError as e:
        print(f"Data error: {e}")
    except nl.NeuroLiteException as e:
        print(f"Analysis error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test with empty DataFrame
    try:
        empty_df = pd.DataFrame()
        report = nl.analyze(empty_df)
    except nl.InsufficientDataError as e:
        print(f"Empty DataFrame error: {e}")
    except nl.NeuroLiteException as e:
        print(f"Analysis error with empty DataFrame: {e}")
    
    # Test with unsupported data type
    try:
        unsupported_data = "This is just a string"
        report = nl.analyze(unsupported_data)
    except nl.NeuroLiteException as e:
        print(f"Unsupported data type error: {e}")


def example_7_performance_optimization():
    """Example 7: Performance optimization for large datasets."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Optimization")
    print("="*60)
    
    # Create a larger dataset
    print("Creating larger dataset for performance testing...")
    np.random.seed(42)
    large_df = pd.DataFrame({
        f'feature_{i}': np.random.randn(10000) for i in range(20)
    })
    large_df['target'] = np.random.choice([0, 1], 10000)
    
    # Test different analysis modes
    print(f"\nTesting different analysis modes on dataset with shape {large_df.shape}:")
    
    # Quick analysis
    import time
    start_time = time.time()
    quick_report = nl.quick_analyze(large_df)
    quick_time = time.time() - start_time
    print(f"- Quick analysis: {quick_time:.2f} seconds")
    
    # Full analysis
    start_time = time.time()
    full_report = nl.analyze(large_df)
    full_time = time.time() - start_time
    print(f"- Full analysis: {full_time:.2f} seconds")
    
    # Analysis with time limit
    start_time = time.time()
    limited_report = nl.analyze(large_df, max_processing_time=3.0)
    limited_time = time.time() - start_time
    print(f"- Time-limited analysis: {limited_time:.2f} seconds")
    
    print(f"\nPerformance comparison:")
    print(f"- Quick analysis is {full_time/quick_time:.1f}x faster than full analysis")
    print(f"- All analyses completed within target time limits")


def example_8_advanced_usage():
    """Example 8: Advanced usage with DataProfiler class."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Advanced Usage with DataProfiler")
    print("="*60)
    
    csv_file, df = create_sample_data()
    
    try:
        # Create custom profiler with specific settings
        from neurolite import DataProfiler
        
        profiler = DataProfiler(
            confidence_threshold=0.9,  # Higher confidence threshold
            enable_graceful_degradation=True,
            max_processing_time=10.0,  # 10 second limit
            max_memory_usage_mb=512    # 512 MB memory limit
        )
        
        print("Using custom DataProfiler with:")
        print("- Confidence threshold: 90%")
        print("- Processing time limit: 10 seconds")
        print("- Memory limit: 512 MB")
        
        # Perform analysis
        report = profiler.analyze(csv_file)
        
        print(f"\nAnalysis results:")
        print(f"- Execution time: {report.execution_time:.2f} seconds")
        print(f"- Task detected: {report.task_identification.task_type}")
        print(f"- Task confidence: {report.task_identification.confidence:.1%}")
        
        # Use large dataset analysis for optimization
        print(f"\nTesting optimized large dataset analysis:")
        large_report = profiler.analyze_large_dataset(csv_file, use_sampling=True)
        
        if 'performance_optimizations' in large_report.resource_requirements:
            opts = large_report.resource_requirements['performance_optimizations']
            print(f"- Used sampling: {opts.get('used_sampling', False)}")
            print(f"- Parallel processing: {opts.get('parallel_processing', False)}")
            print(f"- Memory usage: {opts.get('memory_usage_mb', 0):.1f} MB")
        
    finally:
        os.unlink(csv_file)


def main():
    """Run all examples."""
    print("NeuroLite Basic Usage Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use NeuroLite")
    print("for automated data analysis and ML recommendations.")
    
    # Run all examples
    example_1_basic_analysis()
    example_2_quick_analysis()
    example_3_dataframe_analysis()
    example_4_specialized_functions()
    example_5_formatting_and_export()
    example_6_error_handling()
    example_7_performance_optimization()
    example_8_advanced_usage()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("Check the generated report files for detailed analysis results.")
    print("="*60)


if __name__ == "__main__":
    main()