# NeuroLite API Documentation

NeuroLite is an automated AI/ML library that detects data types and automatically applies the best models with minimal code. This documentation covers the complete public API.

## Quick Start

```python
import neurolite as nl

# Analyze any dataset with one line
report = nl.analyze('your_data.csv')
print(f"Detected task: {report.task_identification.task_type}")
print(f"Recommended models: {[m.model_name for m in report.model_recommendations]}")
```

## Main API Functions

### `analyze(data_source, quick=False, confidence_threshold=0.8, max_processing_time=5.0, enable_parallel=True)`

Analyze any dataset with comprehensive AI/ML insights in one function call.

**Parameters:**
- `data_source` (str | DataFrame | ndarray): File path, pandas DataFrame, or numpy array to analyze
- `quick` (bool): If True, performs fast basic analysis. If False, comprehensive analysis
- `confidence_threshold` (float): Minimum confidence for classifications (0.0-1.0)
- `max_processing_time` (float): Maximum processing time in seconds (None for no limit)
- `enable_parallel` (bool): Whether to use parallel processing for better performance

**Returns:**
- `ProfileReport`: Comprehensive analysis results (if quick=False)
- `QuickReport`: Basic analysis results (if quick=True)

**Examples:**

```python
# Basic usage
report = nl.analyze('data.csv')

# Quick analysis for large datasets
quick_report = nl.analyze('large_data.csv', quick=True)

# Custom settings
report = nl.analyze('data.csv', 
                   confidence_threshold=0.9,
                   max_processing_time=10.0)

# Analyze DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
report = nl.analyze(df)

# Analyze numpy array
import numpy as np
arr = np.random.randn(1000, 10)
report = nl.analyze(arr)
```

### `quick_analyze(data_source)`

Perform quick analysis with basic metrics for fast initial assessment.

**Parameters:**
- `data_source` (str | DataFrame | ndarray): Data source to analyze

**Returns:**
- `QuickReport`: Basic analysis results

**Example:**

```python
# Quick analysis of any data source
report = nl.quick_analyze('data.csv')
print(f"Shape: {report.basic_stats['shape']}")
print(f"Missing values: {report.basic_stats['missing_values']}")
```

### `get_recommendations(data_source, task_type=None)`

Get ML model and preprocessing recommendations for a dataset.

**Parameters:**
- `data_source` (str | DataFrame | ndarray): Data source to analyze
- `task_type` (str, optional): Task type hint ('classification', 'regression', 'clustering')

**Returns:**
- `dict`: Dictionary with 'models' and 'preprocessing' recommendation lists

**Example:**

```python
# Get recommendations for any dataset
recs = nl.get_recommendations('data.csv')
print("Recommended models:", recs['models'])
print("Preprocessing steps:", recs['preprocessing'])

# Get recommendations with task hint
recs = nl.get_recommendations('data.csv', task_type='classification')
```

### `detect_data_types(data_source)`

Detect data types for all columns in a dataset.

**Parameters:**
- `data_source` (str | DataFrame | ndarray): Data source to analyze

**Returns:**
- `dict`: Dictionary mapping column names to detected data types

**Example:**

```python
# Detect data types
types = nl.detect_data_types('data.csv')
print(types)
# {'age': 'numerical', 'name': 'text', 'category': 'categorical'}
```

### `assess_data_quality(data_source)`

Assess data quality metrics for a dataset.

**Parameters:**
- `data_source` (str | DataFrame | ndarray): Data source to analyze

**Returns:**
- `dict`: Dictionary with quality metrics and recommendations

**Example:**

```python
# Assess data quality
quality = nl.assess_data_quality('data.csv')
print(f"Completeness: {quality['completeness']:.2%}")
print(f"Missing pattern: {quality['missing_pattern']}")
print(f"Overall score: {quality['overall_score']:.2%}")
```

## Convenience Aliases

### `profile(data_source, ...)`

Alias for `analyze()` function.

### `scan(data_source)`

Alias for `quick_analyze()` function.

## Visualization and Formatting

### `format_summary(report, format_type='text')`

Format analysis results into a human-readable summary.

**Parameters:**
- `report` (ProfileReport | QuickReport): Analysis report to format
- `format_type` (str): Output format ('text', 'markdown', 'html')

**Returns:**
- `str`: Formatted summary string

**Example:**

```python
report = nl.analyze('data.csv')
summary = nl.format_summary(report, 'text')
print(summary)

# Generate markdown report
md_summary = nl.format_summary(report, 'markdown')
with open('report.md', 'w') as f:
    f.write(md_summary)
```

### `create_dataframe_summary(report)`

Create a pandas DataFrame summary of the analysis results.

**Parameters:**
- `report` (ProfileReport | QuickReport): Analysis report to summarize

**Returns:**
- `DataFrame`: DataFrame with key metrics and findings

**Example:**

```python
report = nl.analyze('data.csv')
summary_df = nl.create_dataframe_summary(report)
print(summary_df)
```

### `export_report(report, filepath, format_type='json')`

Export analysis report to file.

**Parameters:**
- `report` (ProfileReport | QuickReport): Analysis report to export
- `filepath` (str): Output file path
- `format_type` (str): Export format ('json', 'csv', 'html', 'markdown')

**Example:**

```python
report = nl.analyze('data.csv')
nl.export_report(report, 'analysis_report.html', 'html')
nl.export_report(report, 'analysis_report.json', 'json')
```

## Result Objects

### ProfileReport

Comprehensive analysis results containing:

- `file_info`: File format information
- `data_structure`: Data structure details
- `column_analysis`: Column type classifications
- `quality_metrics`: Data quality assessment
- `statistical_properties`: Statistical analysis results
- `domain_analysis`: Domain-specific analysis
- `task_identification`: ML task identification
- `model_recommendations`: Recommended models
- `preprocessing_recommendations`: Preprocessing steps
- `resource_requirements`: Resource estimates
- `execution_time`: Analysis execution time

### QuickReport

Basic analysis results containing:

- `file_info`: File format information
- `data_structure`: Data structure details
- `basic_stats`: Basic statistics
- `quick_recommendations`: Quick recommendations
- `execution_time`: Analysis execution time

### ModelRecommendation

Model recommendation details:

- `model_name`: Name of the recommended model
- `model_type`: Type of model (traditional_ml, deep_learning)
- `confidence`: Confidence in the recommendation
- `rationale`: Explanation for the recommendation
- `hyperparameters`: Suggested hyperparameters
- `expected_performance`: Expected performance metrics

## Advanced API

### DataProfiler

For advanced users who need more control over the analysis process.

```python
from neurolite import DataProfiler

profiler = DataProfiler(
    confidence_threshold=0.8,
    enable_graceful_degradation=True,
    max_processing_time=10.0,
    max_memory_usage_mb=1024
)

# Standard analysis
report = profiler.analyze('data.csv')

# Quick analysis
quick_report = profiler.quick_analyze('data.csv')

# Large dataset analysis with optimizations
large_report = profiler.analyze_large_dataset('big_data.csv', use_sampling=True)
```

## Error Handling

NeuroLite provides specific exceptions for different error conditions:

```python
from neurolite import (
    NeuroLiteException,
    UnsupportedFormatError,
    InsufficientDataError,
    ResourceLimitError
)

try:
    report = nl.analyze('data.csv')
except UnsupportedFormatError as e:
    print(f"File format not supported: {e}")
except InsufficientDataError as e:
    print(f"Dataset too small: {e}")
except ResourceLimitError as e:
    print(f"Resource limit exceeded: {e}")
except NeuroLiteException as e:
    print(f"Analysis failed: {e}")
```

## Performance Optimization

NeuroLite automatically optimizes performance for large datasets:

- **Lazy Loading**: Streams large files in chunks
- **Intelligent Sampling**: Uses statistical sampling for analysis
- **Parallel Processing**: Utilizes multiple CPU cores
- **Memory Management**: Monitors and manages memory usage

### Large Dataset Handling

```python
# For datasets > 100MB or > 50,000 rows, NeuroLite automatically:
# 1. Uses streaming processing
# 2. Applies intelligent sampling
# 3. Enables parallel processing
# 4. Monitors memory usage

report = nl.analyze('large_dataset.csv')  # Automatically optimized

# Force optimization for smaller datasets
from neurolite import DataProfiler
profiler = DataProfiler()
report = profiler.analyze_large_dataset('data.csv', use_sampling=True)
```

## Supported Data Formats

NeuroLite supports 20+ data formats:

**Tabular Data:**
- CSV, TSV
- Excel (XLSX, XLS)
- JSON, JSONL
- Parquet
- HDF5

**Text Data:**
- TXT, MD
- PDF (text extraction)

**Image Data:**
- PNG, JPG, JPEG
- TIFF, BMP
- GIF

**Audio Data:**
- WAV, MP3
- FLAC, OGG

**Video Data:**
- MP4, AVI
- MOV, WMV

**Other:**
- XML
- Pickle files
- NumPy arrays
- Pandas DataFrames

## Performance Targets

NeuroLite is designed to meet specific performance targets:

- **5-second analysis** for datasets up to 1GB
- **95%+ accuracy** in data type detection
- **Memory efficient** processing with configurable limits
- **Graceful degradation** when components fail

## Best Practices

### 1. Start with Quick Analysis

```python
# For initial exploration, use quick analysis
quick_report = nl.quick_analyze('data.csv')
print(f"Dataset overview: {quick_report.basic_stats}")

# Then perform full analysis if needed
if quick_report.data_structure.sample_size > 10000:
    full_report = nl.analyze('data.csv')
```

### 2. Handle Large Datasets

```python
# For large datasets, set appropriate limits
report = nl.analyze('large_data.csv', 
                   max_processing_time=30.0,  # 30 seconds max
                   confidence_threshold=0.7)  # Lower threshold for speed
```

### 3. Export Results

```python
# Save results for later use
report = nl.analyze('data.csv')
nl.export_report(report, 'analysis.json', 'json')
nl.export_report(report, 'summary.html', 'html')
```

### 4. Error Handling

```python
# Always handle potential errors
try:
    report = nl.analyze('data.csv')
    recommendations = nl.get_recommendations('data.csv')
except nl.NeuroLiteException as e:
    print(f"Analysis failed: {e}")
    # Fallback to basic pandas analysis
```

## Integration Examples

### With Scikit-learn

```python
import neurolite as nl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Analyze dataset
report = nl.analyze('data.csv')

# Get recommendations
recs = nl.get_recommendations('data.csv')
print(f"Recommended models: {recs['models']}")

# Use recommendations to select model
if 'RandomForest' in recs['models']:
    # Load data and train model
    import pandas as pd
    df = pd.read_csv('data.csv')
    # ... training code
```

### With Pandas

```python
import neurolite as nl
import pandas as pd

# Analyze DataFrame
df = pd.read_csv('data.csv')
report = nl.analyze(df)

# Use type detection for preprocessing
types = nl.detect_data_types(df)
for col, dtype in types.items():
    if dtype == 'categorical':
        df[col] = pd.Categorical(df[col])
    elif dtype == 'temporal':
        df[col] = pd.to_datetime(df[col])
```

### With Jupyter Notebooks

```python
import neurolite as nl

# Quick analysis for exploration
report = nl.quick_analyze('data.csv')
display(nl.create_dataframe_summary(report))

# Full analysis with formatted output
full_report = nl.analyze('data.csv')
print(nl.format_summary(full_report, 'markdown'))
```

This API documentation provides comprehensive coverage of all NeuroLite functionality with practical examples and best practices for different use cases.