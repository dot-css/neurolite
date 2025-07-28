#!/usr/bin/env python3

import numpy as np
import pandas as pd
from neurolite.analyzers.statistical_analyzer import StatisticalAnalyzer

# Create simple test data
np.random.seed(42)
test_data = pd.DataFrame({
    'col1': np.random.normal(0, 1, 100),
    'col2': np.random.normal(0, 1, 100)
})

print("Creating analyzer...")
analyzer = StatisticalAnalyzer()

print("Testing distribution analysis...")
try:
    dist_results = analyzer.analyze_distributions(test_data)
    print(f"Distribution analysis completed: {len(dist_results)} columns analyzed")
except Exception as e:
    print(f"Distribution analysis failed: {e}")

print("Testing correlation analysis...")
try:
    corr_results = analyzer.compute_correlations(test_data)
    print(f"Correlation analysis completed: {corr_results.pearson_correlation.shape}")
except Exception as e:
    print(f"Correlation analysis failed: {e}")

print("Testing relationship detection...")
try:
    rel_results = analyzer.detect_relationships(test_data)
    print(f"Relationship detection completed")
except Exception as e:
    print(f"Relationship detection failed: {e}")

print("All tests completed!")