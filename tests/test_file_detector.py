"""
Unit tests for FileDetector class.

Tests file format detection, structure identification, and related functionality.
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
from neurolite.core.data_models import FileFormat, DataStructure
from neurolite.core.exceptions import UnsupportedFormatError, NeuroLiteException


class TestFileDetector:
    """Test cases for FileDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FileDetector()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_file(self, filename: str, content: bytes = b'', text_content: str = None) -> Path:
        """Helper to create temporary test files."""
        file_path = Path(self.temp_dir) / filename
        
        if text_content is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        else:
            with open(file_path, 'wb') as f:
                f.write(content)
        
        return file_path
    
    def test_detect_format_csv(self):
        """Test CSV file format detection."""
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        file_path = self.create_temp_file("test.csv", text_content=csv_content)
        
        result = self.detector.detect_format(file_path)
        
        assert isinstance(result, FileFormat)
        assert result.format_type == "CSV"
        assert result.mime_type == "text/csv"
        assert 0.0 <= result.confidence <= 1.0
        assert result.encoding is not None
        assert "file_size" in result.metadata
    
    def test_detect_format_json(self):
        """Test JSON file format detection."""
        json_content = '{"name": "John", "age": 25, "city": "NYC"}'
        file_path = self.create_temp_file("test.json", text_content=json_content)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "JSON"
        assert result.mime_type == "application/json"
        assert result.confidence > 0.0
    
    def test_detect_format_png_magic_number(self):
        """Test PNG detection using magic number."""
        # PNG magic number
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 20
        file_path = self.create_temp_file("test.png", content=png_header)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "PNG"
        assert result.mime_type == "image/png"
        assert result.confidence >= 0.9  # High confidence for magic number detection
    
    def test_detect_format_jpeg_magic_number(self):
        """Test JPEG detection using magic number."""
        # JPEG magic number
        jpeg_header = b'\xff\xd8\xff' + b'\x00' * 20
        file_path = self.create_temp_file("test.jpg", content=jpeg_header)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "JPEG"
        assert result.mime_type == "image/jpeg"
        assert result.confidence >= 0.9
    
    def test_detect_format_pdf_magic_number(self):
        """Test PDF detection using magic number."""
        pdf_header = b'%PDF-1.4' + b'\x00' * 20
        file_path = self.create_temp_file("test.pdf", content=pdf_header)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "PDF"
        assert result.mime_type == "application/pdf"
        assert result.confidence >= 0.9
    
    def test_detect_format_extension_fallback(self):
        """Test format detection falls back to extension when magic number fails."""
        # File with .xlsx extension but no magic number
        file_path = self.create_temp_file("test.xlsx", content=b'some content')
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "EXCEL"
        assert result.confidence > 0.0
    
    def test_detect_format_magic_extension_agreement(self):
        """Test high confidence when magic number and extension agree."""
        # PNG magic number with .png extension
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 20
        file_path = self.create_temp_file("test.png", content=png_header)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "PNG"
        assert result.confidence >= 0.95  # Very high confidence when both agree
    
    def test_detect_format_magic_extension_disagreement(self):
        """Test handling when magic number and extension disagree."""
        # PNG magic number with .jpg extension
        png_header = b'\x89PNG\r\n\x1a\n' + b'\x00' * 20
        file_path = self.create_temp_file("test.jpg", content=png_header)
        
        result = self.detector.detect_format(file_path)
        
        # Should trust magic number over extension
        assert result.format_type == "PNG"
        assert result.confidence >= 0.8
    
    def test_detect_format_riff_wav(self):
        """Test RIFF WAV file detection."""
        # RIFF WAV header
        wav_header = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 20
        file_path = self.create_temp_file("test.wav", content=wav_header)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "WAV"
        assert result.mime_type == "audio/wav"
        assert result.confidence >= 0.9
    
    def test_detect_format_unsupported(self):
        """Test handling of unsupported file formats."""
        # Unknown file with no recognizable magic number or extension
        file_path = self.create_temp_file("test.unknown", content=b'random content')
        
        with pytest.raises(UnsupportedFormatError):
            self.detector.detect_format(file_path)
    
    def test_detect_format_nonexistent_file(self):
        """Test handling of non-existent files."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.csv"
        
        with pytest.raises(NeuroLiteException, match="File not found"):
            self.detector.detect_format(nonexistent_path)
    
    def test_detect_format_directory(self):
        """Test handling when path is a directory."""
        with pytest.raises(NeuroLiteException, match="Path is not a file"):
            self.detector.detect_format(Path(self.temp_dir))
    
    def test_encoding_detection_utf8(self):
        """Test UTF-8 encoding detection."""
        utf8_content = "Hello, 世界! 🌍"
        file_path = self.create_temp_file("test.txt", text_content=utf8_content)
        
        result = self.detector.detect_format(file_path)
        
        assert result.encoding in ['utf-8', 'UTF-8']
    
    def test_csv_metadata_extraction(self):
        """Test CSV-specific metadata extraction."""
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        file_path = self.create_temp_file("test.csv", text_content=csv_content)
        
        result = self.detector.detect_format(file_path)
        
        assert "estimated_delimiter" in result.metadata
        assert "estimated_columns" in result.metadata
        assert "has_header" in result.metadata
        assert result.metadata["estimated_delimiter"] == ","
        assert result.metadata["estimated_columns"] == 3
        assert result.metadata["has_header"] is True
    
    def test_json_metadata_extraction(self):
        """Test JSON-specific metadata extraction."""
        json_content = '{"name": "John", "age": 25}'
        file_path = self.create_temp_file("test.json", text_content=json_content)
        
        result = self.detector.detect_format(file_path)
        
        assert "is_json_lines" in result.metadata
        assert result.metadata["is_json_lines"] is False
    
    def test_jsonl_detection(self):
        """Test JSON Lines format detection."""
        jsonl_content = '{"name": "John", "age": 25}\n{"name": "Jane", "age": 30}'
        file_path = self.create_temp_file("test.jsonl", text_content=jsonl_content)
        
        result = self.detector.detect_format(file_path)
        
        assert result.format_type == "JSONL"
        assert "is_json_lines" in result.metadata
        assert result.metadata["is_json_lines"] is True


class TestFileDetectorStructure:
    """Test cases for data structure detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FileDetector()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_temp_file(self, filename: str, content: bytes = b'', text_content: str = None) -> Path:
        """Helper to create temporary test files."""
        file_path = Path(self.temp_dir) / filename
        
        if text_content is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
        else:
            with open(file_path, 'wb') as f:
                f.write(content)
        
        return file_path
    
    def test_detect_structure_dataframe_tabular(self):
        """Test structure detection for tabular DataFrame."""
        df = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })
        
        result = self.detector.detect_structure(df)
        
        assert isinstance(result, DataStructure)
        assert result.structure_type == 'tabular'
        assert result.dimensions == (3, 3)
        assert result.sample_size == 3
        assert result.memory_usage > 0
    
    def test_detect_structure_dataframe_timeseries(self):
        """Test structure detection for time series DataFrame."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100),
            'category': ['A'] * 50 + ['B'] * 50
        })
        
        result = self.detector.detect_structure(df)
        
        assert result.structure_type == 'time_series'
        assert result.dimensions == (100, 3)
        assert result.sample_size == 100
    
    def test_detect_structure_array_1d(self):
        """Test structure detection for 1D numpy array."""
        arr = np.random.randn(1000)
        
        result = self.detector.detect_structure(arr)
        
        assert result.structure_type == 'tabular'
        assert result.dimensions == (1000,)
        assert result.sample_size == 1000
        assert result.memory_usage == arr.nbytes
    
    def test_detect_structure_array_2d_tabular(self):
        """Test structure detection for 2D numpy array (tabular)."""
        arr = np.random.randn(100, 10)  # Small 2D array, likely tabular
        
        result = self.detector.detect_structure(arr)
        
        assert result.structure_type == 'tabular'
        assert result.dimensions == (100, 10)
        assert result.sample_size == 100
    
    def test_detect_structure_array_2d_image(self):
        """Test structure detection for 2D numpy array (image)."""
        arr = np.random.randint(0, 255, (1920, 1080), dtype=np.uint8)  # Large 2D array, likely image
        
        result = self.detector.detect_structure(arr)
        
        assert result.structure_type == 'image'
        assert result.dimensions == (1920, 1080)
        assert result.sample_size == 1920
    
    def test_detect_structure_array_3d_image(self):
        """Test structure detection for 3D numpy array (RGB image)."""
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # RGB image
        
        result = self.detector.detect_structure(arr)
        
        assert result.structure_type == 'image'
        assert result.dimensions == (480, 640, 3)
        assert result.sample_size == 480
    
    def test_detect_structure_csv_file(self):
        """Test structure detection from CSV file."""
        csv_content = "name,age,city\n" + "\n".join([f"Person{i},{20+i},City{i}" for i in range(1000)])
        file_path = self.create_temp_file("test.csv", text_content=csv_content)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'tabular'
        assert result.dimensions[0] == 1000  # 1000 data rows
        assert result.dimensions[1] == 3     # 3 columns
        assert result.sample_size == 1000
        assert result.memory_usage > 0
    
    def test_detect_structure_csv_timeseries(self):
        """Test structure detection for time series CSV."""
        csv_content = "timestamp,value\n"
        csv_content += "\n".join([f"2023-01-{i+1:02d},{i*10}" for i in range(31)])
        file_path = self.create_temp_file("timeseries.csv", text_content=csv_content)
        
        result = self.detector.detect_structure(file_path)
        
        # Should detect as time series due to timestamp column
        assert result.structure_type == 'time_series'
        assert result.dimensions == (31, 2)
    
    def test_detect_structure_json_single(self):
        """Test structure detection for single JSON document."""
        json_content = '{"name": "John", "age": 25, "hobbies": ["reading", "coding"]}'
        file_path = self.create_temp_file("test.json", text_content=json_content)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'text'
        assert result.dimensions == (1,)
        assert result.sample_size == 1
    
    def test_detect_structure_jsonl(self):
        """Test structure detection for JSON Lines."""
        jsonl_content = '{"name": "John", "age": 25}\n{"name": "Jane", "age": 30}\n{"name": "Bob", "age": 35}'
        file_path = self.create_temp_file("test.jsonl", text_content=jsonl_content)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'tabular'  # JSONL is treated as tabular
        assert result.dimensions == (3, 1)  # 3 records
        assert result.sample_size == 3
    
    def test_detect_structure_text_file(self):
        """Test structure detection for plain text."""
        text_content = "Line 1\nLine 2\nLine 3\nLine 4"
        file_path = self.create_temp_file("test.txt", text_content=text_content)
        
        result = self.detector.detect_structure(file_path)
        
        assert result.structure_type == 'text'
        assert result.dimensions == (4,)  # 4 lines
        assert result.sample_size == 4
    
    def test_detect_structure_unsupported_type(self):
        """Test handling of unsupported data source types."""
        with pytest.raises(NeuroLiteException, match="Unsupported data source type"):
            self.detector.detect_structure(123)  # Integer is not supported
    
    def test_detect_structure_unsupported_format(self):
        """Test handling of unsupported file formats for structure detection."""
        # Create a file with unsupported format
        unknown_content = b'unknown binary content'
        file_path = self.create_temp_file("test.unknown", content=unknown_content)
        
        with pytest.raises(UnsupportedFormatError):
            self.detector.detect_structure(file_path)


if __name__ == "__main__":
    pytest.main([__file__])