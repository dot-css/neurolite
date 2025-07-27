"""
Additional tests for data structure identification with sample files.

Tests comprehensive structure detection across different data types and formats.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json

from neurolite.detectors.file_detector import FileDetector
from neurolite.core.data_models import DataStructure
from neurolite.core.exceptions import UnsupportedFormatError, NeuroLiteException


class TestStructureIdentificationSamples:
    """Test structure identification with various sample files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FileDetector()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_csv(self, filename: str, rows: int = 100, has_datetime: bool = False) -> Path:
        """Create a sample CSV file."""
        file_path = Path(self.temp_dir) / filename
        
        if has_datetime:
            # Create time series CSV
            dates = pd.date_range('2023-01-01', periods=rows, freq='D')
            df = pd.DataFrame({
                'timestamp': dates,
                'value': np.random.randn(rows),
                'category': np.random.choice(['A', 'B', 'C'], rows)
            })
        else:
            # Create regular tabular CSV
            df = pd.DataFrame({
                'id': range(1, rows + 1),
                'name': [f'Person_{i}' for i in range(1, rows + 1)],
                'age': np.random.randint(18, 80, rows),
                'salary': np.random.randint(30000, 120000, rows),
                'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], rows)
            })
        
        df.to_csv(file_path, index=False)
        return file_path
    
    def create_sample_image(self, filename: str, width: int = 640, height: int = 480, mode: str = 'RGB') -> Path:
        """Create a sample image file."""
        file_path = Path(self.temp_dir) / filename
        
        if mode == 'RGB':
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        elif mode == 'L':
            image_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        else:
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        image = Image.fromarray(image_array, mode=mode)
        image.save(file_path)
        return file_path
    
    def create_sample_json(self, filename: str, is_lines: bool = False, records: int = 10) -> Path:
        """Create a sample JSON file."""
        file_path = Path(self.temp_dir) / filename
        
        if is_lines:
            # JSON Lines format
            with open(file_path, 'w') as f:
                for i in range(records):
                    record = {
                        'id': i + 1,
                        'name': f'Person_{i + 1}',
                        'score': np.random.randint(0, 100)
                    }
                    f.write(json.dumps(record) + '\n')
        else:
            # Single JSON document
            data = {
                'metadata': {'version': '1.0', 'created': '2023-01-01'},
                'records': [
                    {'id': i + 1, 'name': f'Person_{i + 1}', 'score': np.random.randint(0, 100)}
                    for i in range(records)
                ]
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return file_path
    
    def create_sample_text(self, filename: str, lines: int = 50) -> Path:
        """Create a sample text file."""
        file_path = Path(self.temp_dir) / filename
        
        with open(file_path, 'w') as f:
            for i in range(lines):
                f.write(f"This is line {i + 1} of the sample text file.\n")
        
        return file_path
    
    def test_tabular_csv_structure(self):
        """Test structure detection for tabular CSV files."""
        file_path = self.create_sample_csv("tabular.csv", rows=500)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'tabular'
        assert result.dimensions == (500, 5)  # 500 rows, 5 columns
        assert result.sample_size == 500
        assert result.memory_usage > 0
    
    def test_timeseries_csv_structure(self):
        """Test structure detection for time series CSV files."""
        file_path = self.create_sample_csv("timeseries.csv", rows=365, has_datetime=True)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'time_series'
        assert result.dimensions == (365, 3)  # 365 rows, 3 columns
        assert result.sample_size == 365
        assert result.memory_usage > 0
    
    def test_large_csv_estimation(self):
        """Test memory and dimension estimation for large CSV files."""
        file_path = self.create_sample_csv("large.csv", rows=10000)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'tabular'
        assert result.dimensions == (10000, 5)
        assert result.sample_size == 10000
        assert result.memory_usage > 50000  # Should be substantial for 10k rows
    
    def test_image_rgb_structure(self):
        """Test structure detection for RGB images."""
        file_path = self.create_sample_image("rgb_image.png", width=800, height=600, mode='RGB')
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'image'
        assert result.dimensions == (600, 800, 3)  # height, width, channels
        assert result.sample_size == 1
        assert result.memory_usage > 0
    
    def test_image_grayscale_structure(self):
        """Test structure detection for grayscale images."""
        file_path = self.create_sample_image("gray_image.png", width=400, height=300, mode='L')
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'image'
        assert result.dimensions == (300, 400)  # height, width (no channel dimension for grayscale)
        assert result.sample_size == 1
        assert result.memory_usage > 0
    
    def test_json_single_document_structure(self):
        """Test structure detection for single JSON documents."""
        file_path = self.create_sample_json("single.json", is_lines=False, records=20)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'text'
        assert result.dimensions == (1,)  # Single document
        assert result.sample_size == 1
        assert result.memory_usage > 0
    
    def test_jsonl_structure(self):
        """Test structure detection for JSON Lines files."""
        file_path = self.create_sample_json("lines.jsonl", is_lines=True, records=100)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'tabular'  # JSONL treated as tabular
        assert result.dimensions == (100, 1)  # 100 records
        assert result.sample_size == 100
        assert result.memory_usage > 0
    
    def test_text_file_structure(self):
        """Test structure detection for plain text files."""
        file_path = self.create_sample_text("sample.txt", lines=200)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'text'
        assert result.dimensions == (200,)  # 200 lines
        assert result.sample_size == 200
        assert result.memory_usage > 0
    
    def test_markdown_file_structure(self):
        """Test structure detection for Markdown files."""
        file_path = Path(self.temp_dir) / "sample.md"
        
        markdown_content = """# Sample Markdown
        
## Introduction
This is a sample markdown file for testing.

## Features
- Feature 1
- Feature 2
- Feature 3

## Conclusion
This concludes the sample.
"""
        
        with open(file_path, 'w') as f:
            f.write(markdown_content)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'text'
        assert result.sample_size > 0  # Should count lines
        assert result.memory_usage > 0
    
    def test_excel_structure_simulation(self):
        """Test Excel structure detection (simulated with CSV)."""
        # Create a CSV that simulates Excel data
        file_path = Path(self.temp_dir) / "test.xlsx"
        
        # Create Excel-like content but save as CSV for testing
        df = pd.DataFrame({
            'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'] * 25,
            'Revenue': np.random.randint(100000, 500000, 100),
            'Expenses': np.random.randint(50000, 200000, 100),
            'Profit': np.random.randint(10000, 100000, 100)
        })
        
        # Write as CSV but with .xlsx extension to test extension-based detection
        df.to_csv(file_path, index=False)
        
        # This should detect as Excel format but fail structure analysis
        # since we can't actually read Excel without proper libraries
        with pytest.raises(NeuroLiteException):
            self.detector.detect_structure(file_path)
    
    def test_memory_usage_accuracy(self):
        """Test accuracy of memory usage estimation."""
        # Create a DataFrame and compare estimated vs actual memory usage
        df = pd.DataFrame({
            'col1': np.random.randn(1000),
            'col2': np.random.randint(0, 100, 1000),
            'col3': ['text_' + str(i) for i in range(1000)]
        })
        
        actual_memory = df.memory_usage(deep=True).sum()
        
        result = self.detector.detect_structure(df)
        estimated_memory = result.memory_usage
        
        # Estimated memory should be reasonably close to actual
        # Allow for some variance due to estimation methods
        assert abs(estimated_memory - actual_memory) / actual_memory < 0.5  # Within 50%
    
    def test_dimension_analysis_accuracy(self):
        """Test accuracy of dimension analysis."""
        # Test various array shapes
        test_cases = [
            (np.random.randn(100), (100,)),
            (np.random.randn(50, 10), (50, 10)),
            (np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), (32, 32, 3)),
            (np.random.randn(10, 20, 30, 4), (10, 20, 30, 4))
        ]
        
        for array, expected_dims in test_cases:
            result = self.detector.detect_structure(array)
            assert result.dimensions == expected_dims
            assert result.sample_size == array.shape[0] if array.ndim > 0 else 1
    
    def test_time_series_detection_heuristics(self):
        """Test various heuristics for time series detection."""
        # Test different datetime column names
        datetime_columns = ['date', 'timestamp', 'time', 'datetime', 'created_at']
        
        for col_name in datetime_columns:
            df = pd.DataFrame({
                col_name: pd.date_range('2023-01-01', periods=50),
                'value': np.random.randn(50)
            })
            
            result = self.detector.detect_structure(df)
            assert result.structure_type == 'time_series', f"Failed for column: {col_name}"
    
    def test_structure_consistency(self):
        """Test that structure detection is consistent across multiple calls."""
        file_path = self.create_sample_csv("consistency.csv", rows=100)
        
        # Run detection multiple times
        results = [self.detector.detect_structure(file_path) for _ in range(5)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.structure_type == first_result.structure_type
            assert result.dimensions == first_result.dimensions
            assert result.sample_size == first_result.sample_size
            # Memory usage might vary slightly due to system conditions
            assert abs(result.memory_usage - first_result.memory_usage) < 1000


if __name__ == "__main__":
    pytest.main([__file__])