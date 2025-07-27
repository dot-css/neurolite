"""
File format and structure detection module.

This module provides functionality to detect file formats using magic numbers
and file extensions, as well as identify data structure types.
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import json
import xml.etree.ElementTree as ET

from ..core.data_models import FileFormat, DataStructure
from ..core.exceptions import UnsupportedFormatError, NeuroLiteException


class FileDetector:
    """Detects file formats and data structures."""
    
    # Magic number signatures for common file types
    MAGIC_NUMBERS = {
        # Images
        b'\x89PNG\r\n\x1a\n': ('PNG', 'image/png'),
        b'\xff\xd8\xff': ('JPEG', 'image/jpeg'),
        b'GIF87a': ('GIF', 'image/gif'),
        b'GIF89a': ('GIF', 'image/gif'),
        b'RIFF': ('WEBP', 'image/webp'),  # Will need additional validation
        b'II*\x00': ('TIFF', 'image/tiff'),
        b'MM\x00*': ('TIFF', 'image/tiff'),
        
        # Audio
        b'ID3': ('MP3', 'audio/mpeg'),
        b'\xff\xfb': ('MP3', 'audio/mpeg'),
        b'\xff\xf3': ('MP3', 'audio/mpeg'),
        b'\xff\xf2': ('MP3', 'audio/mpeg'),
        b'RIFF': ('WAV', 'audio/wav'),  # Will need additional validation
        b'fLaC': ('FLAC', 'audio/flac'),
        
        # Video
        b'\x00\x00\x00\x18ftypmp4': ('MP4', 'video/mp4'),
        b'\x00\x00\x00\x20ftypM4V': ('MP4', 'video/mp4'),
        b'RIFF': ('AVI', 'video/x-msvideo'),  # Will need additional validation
        
        # Documents
        b'%PDF': ('PDF', 'application/pdf'),
        b'PK\x03\x04': ('ZIP', 'application/zip'),  # Also Excel/Word files
        
        # Data formats
        b'PAR1': ('PARQUET', 'application/octet-stream'),
        b'\x89HDF\r\n\x1a\n': ('HDF5', 'application/x-hdf'),
    }
    
    # File extension mappings
    EXTENSION_MAPPINGS = {
        '.csv': ('CSV', 'text/csv'),
        '.tsv': ('TSV', 'text/tab-separated-values'),
        '.json': ('JSON', 'application/json'),
        '.jsonl': ('JSONL', 'application/jsonlines'),
        '.xml': ('XML', 'application/xml'),
        '.xlsx': ('EXCEL', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        '.xls': ('EXCEL', 'application/vnd.ms-excel'),
        '.parquet': ('PARQUET', 'application/octet-stream'),
        '.h5': ('HDF5', 'application/x-hdf'),
        '.hdf5': ('HDF5', 'application/x-hdf'),
        '.txt': ('TEXT', 'text/plain'),
        '.md': ('MARKDOWN', 'text/markdown'),
        '.png': ('PNG', 'image/png'),
        '.jpg': ('JPEG', 'image/jpeg'),
        '.jpeg': ('JPEG', 'image/jpeg'),
        '.gif': ('GIF', 'image/gif'),
        '.tiff': ('TIFF', 'image/tiff'),
        '.tif': ('TIFF', 'image/tiff'),
        '.webp': ('WEBP', 'image/webp'),
        '.wav': ('WAV', 'audio/wav'),
        '.mp3': ('MP3', 'audio/mpeg'),
        '.flac': ('FLAC', 'audio/flac'),
        '.mp4': ('MP4', 'video/mp4'),
        '.avi': ('AVI', 'video/x-msvideo'),
        '.mov': ('MOV', 'video/quicktime'),
        '.pdf': ('PDF', 'application/pdf'),
    }
    
    def __init__(self):
        """Initialize the FileDetector."""
        # Initialize mimetypes
        mimetypes.init()
    
    def detect_format(self, file_path: Union[str, Path]) -> FileFormat:
        """
        Detect file format using magic numbers and file extensions.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileFormat object with detected format information
            
        Raises:
            UnsupportedFormatError: If file format is not supported
            NeuroLiteException: If file cannot be read or analyzed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise NeuroLiteException(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise NeuroLiteException(f"Path is not a file: {file_path}")
        
        # Try magic number detection first (more reliable)
        magic_result = self._detect_by_magic_number(file_path)
        
        # Try extension detection
        extension_result = self._detect_by_extension(file_path)
        
        # Combine results and determine confidence
        format_type, mime_type, confidence = self._combine_detection_results(
            magic_result, extension_result
        )
        
        # Detect encoding for text-based files
        encoding = self._detect_encoding(file_path, format_type)
        
        # Gather additional metadata
        metadata = self._gather_metadata(file_path, format_type)
        
        return FileFormat(
            format_type=format_type,
            confidence=confidence,
            mime_type=mime_type,
            encoding=encoding,
            metadata=metadata
        )
    
    def _detect_by_magic_number(self, file_path: Path) -> Optional[Tuple[str, str, float]]:
        """Detect format using magic numbers."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 32 bytes for magic number detection
                header = f.read(32)
                
            for magic_bytes, (format_type, mime_type) in self.MAGIC_NUMBERS.items():
                if header.startswith(magic_bytes):
                    # Special handling for RIFF files (WAV, AVI, WEBP)
                    if magic_bytes == b'RIFF':
                        return self._handle_riff_format(header, format_type, mime_type)
                    return (format_type, mime_type, 0.9)
                    
        except (IOError, OSError):
            pass
            
        return None
    
    def _handle_riff_format(self, header: bytes, default_format: str, default_mime: str) -> Tuple[str, str, float]:
        """Handle RIFF format detection (WAV, AVI, WEBP)."""
        if len(header) >= 12:
            riff_type = header[8:12]
            if riff_type == b'WAVE':
                return ('WAV', 'audio/wav', 0.9)
            elif riff_type == b'AVI ':
                return ('AVI', 'video/x-msvideo', 0.9)
            elif riff_type == b'WEBP':
                return ('WEBP', 'image/webp', 0.9)
        
        return (default_format, default_mime, 0.7)
    
    def _detect_by_extension(self, file_path: Path) -> Optional[Tuple[str, str, float]]:
        """Detect format using file extension."""
        extension = file_path.suffix.lower()
        
        if extension in self.EXTENSION_MAPPINGS:
            format_type, mime_type = self.EXTENSION_MAPPINGS[extension]
            return (format_type, mime_type, 0.7)
        
        # Try system mimetypes as fallback
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            # Map common mime types to our format types
            format_type = self._mime_to_format(mime_type)
            if format_type:
                return (format_type, mime_type, 0.5)
        
        return None
    
    def _mime_to_format(self, mime_type: str) -> Optional[str]:
        """Convert mime type to our format type."""
        mime_mappings = {
            'text/csv': 'CSV',
            'application/json': 'JSON',
            'application/xml': 'XML',
            'text/xml': 'XML',
            'text/plain': 'TEXT',
            'text/markdown': 'MARKDOWN',
            'image/png': 'PNG',
            'image/jpeg': 'JPEG',
            'image/gif': 'GIF',
            'image/tiff': 'TIFF',
            'image/webp': 'WEBP',
            'audio/wav': 'WAV',
            'audio/mpeg': 'MP3',
            'audio/flac': 'FLAC',
            'video/mp4': 'MP4',
            'video/x-msvideo': 'AVI',
            'video/quicktime': 'MOV',
            'application/pdf': 'PDF',
        }
        return mime_mappings.get(mime_type)
    
    def _combine_detection_results(
        self, 
        magic_result: Optional[Tuple[str, str, float]], 
        extension_result: Optional[Tuple[str, str, float]]
    ) -> Tuple[str, str, float]:
        """Combine magic number and extension detection results."""
        
        if magic_result and extension_result:
            magic_format, magic_mime, magic_conf = magic_result
            ext_format, ext_mime, ext_conf = extension_result
            
            # If both agree, high confidence
            if magic_format == ext_format:
                return (magic_format, magic_mime, 0.95)
            
            # If they disagree, trust magic number more
            return (magic_format, magic_mime, 0.8)
        
        elif magic_result:
            return magic_result
        
        elif extension_result:
            return extension_result
        
        else:
            raise UnsupportedFormatError("Unable to detect file format")
    
    def _detect_encoding(self, file_path: Path, format_type: str) -> Optional[str]:
        """Detect text encoding for text-based files."""
        text_formats = {'CSV', 'TSV', 'JSON', 'JSONL', 'XML', 'TEXT', 'MARKDOWN'}
        
        if format_type not in text_formats:
            return None
        
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for encoding detection
                
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8') if result else 'utf-8'
            
        except ImportError:
            # Fallback if chardet is not available
            encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read some content
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return 'utf-8'  # Default fallback
        
        except (IOError, OSError):
            return 'utf-8'
    
    def _gather_metadata(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """Gather additional metadata about the file."""
        metadata = {}
        
        try:
            stat = file_path.stat()
            metadata.update({
                'file_size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'file_name': file_path.name,
                'file_extension': file_path.suffix,
            })
            
            # Format-specific metadata
            if format_type in {'PNG', 'JPEG', 'GIF', 'TIFF', 'WEBP'}:
                metadata.update(self._get_image_metadata(file_path))
            elif format_type in {'CSV', 'TSV'}:
                metadata.update(self._get_csv_metadata(file_path))
            elif format_type in {'JSON', 'JSONL'}:
                metadata.update(self._get_json_metadata(file_path))
                
        except (IOError, OSError, Exception):
            # Don't fail if metadata gathering fails
            pass
        
        return metadata
    
    def _get_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata for image files."""
        try:
            with Image.open(file_path) as img:
                return {
                    'image_width': img.width,
                    'image_height': img.height,
                    'image_mode': img.mode,
                    'image_format': img.format,
                }
        except Exception:
            return {}
    
    def _get_csv_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata for CSV files."""
        try:
            # Quick peek at CSV structure
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            # Estimate delimiter
            delimiters = [',', '\t', ';', '|']
            delimiter_counts = {d: first_line.count(d) for d in delimiters}
            likely_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            return {
                'estimated_delimiter': likely_delimiter,
                'estimated_columns': delimiter_counts[likely_delimiter] + 1,
                'has_header': self._likely_has_header(first_line, likely_delimiter),
            }
        except Exception:
            return {}
    
    def _likely_has_header(self, first_line: str, delimiter: str) -> bool:
        """Heuristic to determine if CSV has header."""
        if not first_line:
            return False
            
        fields = first_line.split(delimiter)
        
        # Check if fields look like column names (contain letters, not just numbers)
        text_fields = sum(1 for field in fields if any(c.isalpha() for c in field.strip()))
        
        return text_fields > len(fields) / 2
    
    def _get_json_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get metadata for JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to parse a small portion to understand structure
                content = f.read(1000)
                
            # Check if it's JSON Lines format
            lines = content.strip().split('\n')
            is_jsonl = len(lines) > 1 and all(
                line.strip().startswith('{') and line.strip().endswith('}') 
                for line in lines[:3] if line.strip()
            )
            
            return {
                'is_json_lines': is_jsonl,
                'estimated_records': len(lines) if is_jsonl else 1,
            }
        except Exception:
            return {} 
   
    def detect_structure(self, data_source: Union[str, Path, pd.DataFrame, np.ndarray]) -> DataStructure:
        """
        Detect data structure type and characteristics.
        
        Args:
            data_source: File path, DataFrame, or array to analyze
            
        Returns:
            DataStructure object with structure information
            
        Raises:
            NeuroLiteException: If structure cannot be determined
        """
        if isinstance(data_source, (str, Path)):
            return self._detect_structure_from_file(Path(data_source))
        elif isinstance(data_source, pd.DataFrame):
            return self._detect_structure_from_dataframe(data_source)
        elif isinstance(data_source, np.ndarray):
            return self._detect_structure_from_array(data_source)
        else:
            raise NeuroLiteException(f"Unsupported data source type: {type(data_source)}")
    
    def _detect_structure_from_file(self, file_path: Path) -> DataStructure:
        """Detect structure from file path."""
        # First detect format to understand how to process
        file_format = self.detect_format(file_path)
        format_type = file_format.format_type
        
        if format_type in {'CSV', 'TSV', 'EXCEL', 'PARQUET', 'HDF5'}:
            return self._analyze_tabular_structure(file_path, format_type)
        elif format_type in {'PNG', 'JPEG', 'GIF', 'TIFF', 'WEBP'}:
            return self._analyze_image_structure(file_path)
        elif format_type in {'WAV', 'MP3', 'FLAC'}:
            return self._analyze_audio_structure(file_path)
        elif format_type in {'MP4', 'AVI', 'MOV'}:
            return self._analyze_video_structure(file_path)
        elif format_type in {'JSON', 'JSONL', 'XML'}:
            return self._analyze_structured_text(file_path, format_type)
        elif format_type in {'TEXT', 'MARKDOWN', 'PDF'}:
            return self._analyze_text_structure(file_path)
        else:
            raise UnsupportedFormatError(f"Structure detection not supported for format: {format_type}")
    
    def _detect_structure_from_dataframe(self, df: pd.DataFrame) -> DataStructure:
        """Detect structure from pandas DataFrame."""
        # Check if it's time series data
        is_time_series = self._is_time_series_dataframe(df)
        
        structure_type = 'time_series' if is_time_series else 'tabular'
        dimensions = (df.shape[0], df.shape[1])
        sample_size = len(df)
        memory_usage = df.memory_usage(deep=True).sum()
        
        return DataStructure(
            structure_type=structure_type,
            dimensions=dimensions,
            sample_size=sample_size,
            memory_usage=int(memory_usage)
        )
    
    def _detect_structure_from_array(self, arr: np.ndarray) -> DataStructure:
        """Detect structure from numpy array."""
        # Determine structure type based on dimensions and content
        if arr.ndim == 1:
            structure_type = 'tabular'  # 1D array, likely a single column
        elif arr.ndim == 2:
            # Could be tabular data or image
            if arr.shape[0] > 1000 or arr.shape[1] > 1000:
                structure_type = 'image'  # Large 2D array likely an image
            else:
                structure_type = 'tabular'
        elif arr.ndim == 3:
            # Could be RGB image, time series, or 3D data
            if arr.shape[2] in [1, 3, 4]:  # Grayscale, RGB, or RGBA
                structure_type = 'image'
            else:
                structure_type = 'tabular'  # 3D tabular data
        elif arr.ndim == 4:
            structure_type = 'image'  # Batch of images
        else:
            structure_type = 'tabular'  # Default for higher dimensions
        
        dimensions = arr.shape
        sample_size = arr.shape[0] if arr.ndim > 0 else 1
        memory_usage = arr.nbytes
        
        return DataStructure(
            structure_type=structure_type,
            dimensions=dimensions,
            sample_size=sample_size,
            memory_usage=memory_usage
        )
    
    def _analyze_tabular_structure(self, file_path: Path, format_type: str) -> DataStructure:
        """Analyze tabular data structure."""
        try:
            if format_type == 'CSV':
                # Quick peek at CSV structure
                df_sample = pd.read_csv(file_path, nrows=100)
            elif format_type == 'TSV':
                df_sample = pd.read_csv(file_path, sep='\t', nrows=100)
            elif format_type == 'EXCEL':
                df_sample = pd.read_excel(file_path, nrows=100)
            elif format_type == 'PARQUET':
                df_sample = pd.read_parquet(file_path)
                df_sample = df_sample.head(100)
            else:
                raise UnsupportedFormatError(f"Tabular analysis not implemented for {format_type}")
            
            # Estimate full dataset size
            if format_type in {'CSV', 'TSV'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
                estimated_rows = max(1, total_lines - 1)  # Subtract header
            else:
                estimated_rows = len(df_sample)  # For other formats, use actual size
            
            # Check if it's time series
            is_time_series = self._is_time_series_dataframe(df_sample)
            
            structure_type = 'time_series' if is_time_series else 'tabular'
            dimensions = (estimated_rows, len(df_sample.columns))
            
            # Estimate memory usage
            memory_per_row = df_sample.memory_usage(deep=True).sum() / len(df_sample)
            estimated_memory = int(memory_per_row * estimated_rows)
            
            return DataStructure(
                structure_type=structure_type,
                dimensions=dimensions,
                sample_size=estimated_rows,
                memory_usage=estimated_memory
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze tabular structure: {str(e)}")
    
    def _analyze_image_structure(self, file_path: Path) -> DataStructure:
        """Analyze image data structure."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                
                # Determine dimensions based on image mode
                if mode in ['L', '1']:  # Grayscale or binary
                    dimensions = (height, width)
                elif mode in ['RGB', 'YCbCr']:
                    dimensions = (height, width, 3)
                elif mode == 'RGBA':
                    dimensions = (height, width, 4)
                else:
                    dimensions = (height, width)  # Default
                
                # Estimate memory usage (rough calculation)
                bytes_per_pixel = {'1': 0.125, 'L': 1, 'RGB': 3, 'RGBA': 4}.get(mode, 3)
                memory_usage = int(width * height * bytes_per_pixel)
                
                return DataStructure(
                    structure_type='image',
                    dimensions=dimensions,
                    sample_size=1,  # Single image
                    memory_usage=memory_usage
                )
                
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze image structure: {str(e)}")
    
    def _analyze_audio_structure(self, file_path: Path) -> DataStructure:
        """Analyze audio data structure."""
        try:
            # Try to get basic audio info without loading full file
            file_size = file_path.stat().st_size
            
            # Rough estimates based on common audio formats
            # This is a simplified approach - in practice, you'd use librosa or similar
            estimated_duration = file_size / 44100 / 2  # Rough estimate for 16-bit, 44.1kHz
            estimated_samples = int(estimated_duration * 44100)
            
            return DataStructure(
                structure_type='audio',
                dimensions=(estimated_samples,),
                sample_size=estimated_samples,
                memory_usage=file_size
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze audio structure: {str(e)}")
    
    def _analyze_video_structure(self, file_path: Path) -> DataStructure:
        """Analyze video data structure."""
        try:
            file_size = file_path.stat().st_size
            
            # This is a very rough estimate - in practice, you'd use opencv or ffmpeg
            # Assuming typical video parameters
            estimated_frames = 1000  # Placeholder
            estimated_width = 1920
            estimated_height = 1080
            
            return DataStructure(
                structure_type='video',
                dimensions=(estimated_frames, estimated_height, estimated_width, 3),
                sample_size=estimated_frames,
                memory_usage=file_size
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze video structure: {str(e)}")
    
    def _analyze_structured_text(self, file_path: Path, format_type: str) -> DataStructure:
        """Analyze structured text data (JSON, XML)."""
        try:
            file_size = file_path.stat().st_size
            
            if format_type == 'JSON':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Sample first 1KB
                
                # Check if it's JSON Lines
                lines = content.strip().split('\n')
                is_jsonl = len(lines) > 1 and all(
                    line.strip().startswith('{') for line in lines[:3] if line.strip()
                )
                
                if is_jsonl:
                    # Count total lines for JSONL
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_records = sum(1 for _ in f)
                    structure_type = 'tabular'  # JSONL is essentially tabular
                    dimensions = (total_records, 1)  # Approximate
                else:
                    structure_type = 'text'
                    dimensions = (1,)  # Single JSON document
                    total_records = 1
                    
            elif format_type == 'JSONL':
                # Count total lines for JSONL
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_records = sum(1 for _ in f)
                structure_type = 'tabular'  # JSONL is essentially tabular
                dimensions = (total_records, 1)  # Approximate
                
            elif format_type == 'XML':
                structure_type = 'text'
                dimensions = (1,)
                total_records = 1
            else:
                structure_type = 'text'
                dimensions = (1,)
                total_records = 1
            
            return DataStructure(
                structure_type=structure_type,
                dimensions=dimensions,
                sample_size=total_records,
                memory_usage=file_size
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze structured text: {str(e)}")
    
    def _analyze_text_structure(self, file_path: Path) -> DataStructure:
        """Analyze plain text structure."""
        try:
            file_size = file_path.stat().st_size
            
            # Count lines and estimate structure
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            
            return DataStructure(
                structure_type='text',
                dimensions=(lines,),
                sample_size=lines,
                memory_usage=file_size
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze text structure: {str(e)}")
    
    def _is_time_series_dataframe(self, df: pd.DataFrame) -> bool:
        """Heuristic to determine if DataFrame represents time series data."""
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return True
        
        # Check for columns that might be dates
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                # Try to parse a few values as dates
                sample_values = df[col].dropna().head(10)
                try:
                    pd.to_datetime(sample_values)
                    return True
                except:
                    continue
        
        # Check if index looks like a time series
        if isinstance(df.index, pd.DatetimeIndex):
            return True
        
        return False