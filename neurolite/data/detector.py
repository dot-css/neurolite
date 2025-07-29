"""
Automatic data type detection for NeuroLite.

Provides utilities to automatically detect data types from file paths,
extensions, and content analysis.
"""

import os
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

from ..core import get_logger, DataError, safe_import
from ..core.utils import get_file_hash


logger = get_logger(__name__)


class DataType(Enum):
    """Enumeration of supported data types."""
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class DataTypeDetector:
    """
    Automatic data type detection system.
    
    Detects data types based on file extensions, MIME types, and content analysis.
    Supports images, text, CSV/tabular data, audio, and video files.
    """
    
    # File extension mappings
    EXTENSION_MAPPINGS = {
        # Image formats
        '.jpg': DataType.IMAGE,
        '.jpeg': DataType.IMAGE,
        '.png': DataType.IMAGE,
        '.gif': DataType.IMAGE,
        '.bmp': DataType.IMAGE,
        '.tiff': DataType.IMAGE,
        '.tif': DataType.IMAGE,
        '.webp': DataType.IMAGE,
        '.svg': DataType.IMAGE,
        '.ico': DataType.IMAGE,
        
        # Text formats
        '.txt': DataType.TEXT,
        '.md': DataType.TEXT,
        '.rst': DataType.TEXT,
        '.log': DataType.TEXT,
        '.json': DataType.TEXT,
        '.xml': DataType.TEXT,
        '.html': DataType.TEXT,
        '.htm': DataType.TEXT,
        
        # Tabular formats
        '.csv': DataType.TABULAR,
        '.tsv': DataType.TABULAR,
        '.xlsx': DataType.TABULAR,
        '.xls': DataType.TABULAR,
        '.parquet': DataType.TABULAR,
        '.feather': DataType.TABULAR,
        '.h5': DataType.TABULAR,
        '.hdf5': DataType.TABULAR,
        
        # Audio formats
        '.wav': DataType.AUDIO,
        '.mp3': DataType.AUDIO,
        '.flac': DataType.AUDIO,
        '.aac': DataType.AUDIO,
        '.ogg': DataType.AUDIO,
        '.m4a': DataType.AUDIO,
        '.wma': DataType.AUDIO,
        
        # Video formats
        '.mp4': DataType.VIDEO,
        '.avi': DataType.VIDEO,
        '.mov': DataType.VIDEO,
        '.mkv': DataType.VIDEO,
        '.wmv': DataType.VIDEO,
        '.flv': DataType.VIDEO,
        '.webm': DataType.VIDEO,
        '.m4v': DataType.VIDEO,
    }
    
    # MIME type mappings
    MIME_MAPPINGS = {
        'image/': DataType.IMAGE,
        'text/': DataType.TEXT,
        'application/json': DataType.TEXT,
        'application/xml': DataType.TEXT,
        'text/csv': DataType.TABULAR,
        'application/vnd.ms-excel': DataType.TABULAR,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DataType.TABULAR,
        'audio/': DataType.AUDIO,
        'video/': DataType.VIDEO,
    }
    
    def __init__(self):
        """Initialize the data type detector."""
        self._init_mimetypes()
    
    def _init_mimetypes(self):
        """Initialize MIME types database."""
        mimetypes.init()
        # Add custom MIME types
        mimetypes.add_type('text/csv', '.csv')
        mimetypes.add_type('application/parquet', '.parquet')
        mimetypes.add_type('application/feather', '.feather')
    
    def detect_file_type(self, file_path: Union[str, Path]) -> DataType:
        """
        Detect data type of a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected data type
            
        Raises:
            DataError: If file doesn't exist or cannot be analyzed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise DataError(f"Path is not a file: {file_path}")
        
        logger.debug(f"Detecting data type for file: {file_path}")
        
        # Try extension-based detection first
        extension = file_path.suffix.lower()
        if extension in self.EXTENSION_MAPPINGS:
            detected_type = self.EXTENSION_MAPPINGS[extension]
            logger.debug(f"Detected type from extension '{extension}': {detected_type.value}")
            return detected_type
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            detected_type = self._detect_from_mime_type(mime_type)
            if detected_type != DataType.UNKNOWN:
                logger.debug(f"Detected type from MIME type '{mime_type}': {detected_type.value}")
                return detected_type
        
        # Try content-based detection
        try:
            detected_type = self._detect_from_content(file_path)
            if detected_type != DataType.UNKNOWN:
                logger.debug(f"Detected type from content analysis: {detected_type.value}")
                return detected_type
        except Exception as e:
            logger.warning(f"Content-based detection failed for {file_path}: {e}")
        
        logger.warning(f"Could not determine data type for {file_path}")
        return DataType.UNKNOWN
    
    def detect_directory_type(self, dir_path: Union[str, Path]) -> DataType:
        """
        Detect data type of files in a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Most common data type in the directory
            
        Raises:
            DataError: If directory doesn't exist or is empty
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            raise DataError(f"Directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise DataError(f"Path is not a directory: {dir_path}")
        
        logger.debug(f"Detecting data type for directory: {dir_path}")
        
        # Get all files in directory (recursively)
        files = []
        for pattern in ['*', '*/*', '*/*/*']:  # Up to 3 levels deep
            files.extend(dir_path.glob(pattern))
        
        files = [f for f in files if f.is_file()]
        
        if not files:
            raise DataError(f"No files found in directory: {dir_path}")
        
        # Count data types
        type_counts = {}
        for file_path in files:
            try:
                file_type = self.detect_file_type(file_path)
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            except Exception as e:
                logger.debug(f"Failed to detect type for {file_path}: {e}")
        
        if not type_counts:
            return DataType.UNKNOWN
        
        # Return most common type (excluding UNKNOWN)
        valid_types = {k: v for k, v in type_counts.items() if k != DataType.UNKNOWN}
        if valid_types:
            most_common_type = max(valid_types, key=valid_types.get)
            logger.debug(f"Most common type in directory: {most_common_type.value} ({valid_types[most_common_type]} files)")
            return most_common_type
        
        return DataType.UNKNOWN
    
    def _detect_from_mime_type(self, mime_type: str) -> DataType:
        """
        Detect data type from MIME type.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            Detected data type
        """
        # Check exact matches first
        if mime_type in self.MIME_MAPPINGS:
            return self.MIME_MAPPINGS[mime_type]
        
        # Check prefix matches
        for mime_prefix, data_type in self.MIME_MAPPINGS.items():
            if mime_prefix.endswith('/') and mime_type.startswith(mime_prefix):
                return data_type
        
        return DataType.UNKNOWN
    
    def _detect_from_content(self, file_path: Path) -> DataType:
        """
        Detect data type from file content analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected data type
        """
        try:
            # Read first few bytes to check for binary vs text
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for binary file indicators
            if b'\x00' in header:
                # Binary file - try to detect specific formats
                return self._detect_binary_format(header)
            
            # Text file - try to detect structure
            try:
                text_content = header.decode('utf-8', errors='ignore')
                return self._detect_text_format(text_content, file_path)
            except UnicodeDecodeError:
                return DataType.UNKNOWN
                
        except Exception as e:
            logger.debug(f"Content analysis failed for {file_path}: {e}")
            return DataType.UNKNOWN
    
    def _detect_binary_format(self, header: bytes) -> DataType:
        """
        Detect binary file format from header bytes.
        
        Args:
            header: First bytes of the file
            
        Returns:
            Detected data type
        """
        # Image format signatures
        image_signatures = [
            (b'\xff\xd8\xff', DataType.IMAGE),  # JPEG
            (b'\x89PNG\r\n\x1a\n', DataType.IMAGE),  # PNG
            (b'GIF8', DataType.IMAGE),  # GIF
            (b'BM', DataType.IMAGE),  # BMP
            (b'RIFF', DataType.IMAGE),  # WEBP (also used by WAV)
            (b'\x00\x00\x01\x00', DataType.IMAGE),  # ICO
        ]
        
        # Audio format signatures
        audio_signatures = [
            (b'ID3', DataType.AUDIO),  # MP3
            (b'\xff\xfb', DataType.AUDIO),  # MP3
            (b'RIFF', DataType.AUDIO),  # WAV (also used by WEBP)
            (b'fLaC', DataType.AUDIO),  # FLAC
            (b'OggS', DataType.AUDIO),  # OGG
        ]
        
        # Video format signatures
        video_signatures = [
            (b'\x00\x00\x00\x18ftypmp4', DataType.VIDEO),  # MP4
            (b'\x00\x00\x00\x20ftypmp4', DataType.VIDEO),  # MP4
            (b'RIFF', DataType.VIDEO),  # AVI (also used by WAV/WEBP)
        ]
        
        # Check signatures in order of specificity
        for signatures, data_type in [(image_signatures, DataType.IMAGE),
                                     (audio_signatures, DataType.AUDIO),
                                     (video_signatures, DataType.VIDEO)]:
            for signature, detected_type in signatures:
                if header.startswith(signature):
                    return detected_type
        
        return DataType.UNKNOWN
    
    def _detect_text_format(self, content: str, file_path: Path) -> DataType:
        """
        Detect text file format from content.
        
        Args:
            content: Text content sample
            file_path: Path to the file
            
        Returns:
            Detected data type
        """
        # Check for CSV-like structure
        lines = content.strip().split('\n')
        if len(lines) >= 2:
            # Check if it looks like CSV
            first_line = lines[0]
            if ',' in first_line or '\t' in first_line or ';' in first_line:
                # Count delimiters to see if it's consistent
                delimiters = [',', '\t', ';']
                for delimiter in delimiters:
                    if delimiter in first_line:
                        first_count = first_line.count(delimiter)
                        if first_count > 0:
                            # Check if other lines have similar delimiter counts
                            consistent = True
                            for line in lines[1:3]:  # Check next 2 lines
                                if abs(line.count(delimiter) - first_count) > 1:
                                    consistent = False
                                    break
                            if consistent:
                                return DataType.TABULAR
        
        # Check for JSON structure
        content_stripped = content.strip()
        if (content_stripped.startswith('{') and '}' in content_stripped) or \
           (content_stripped.startswith('[') and ']' in content_stripped):
            try:
                import json
                json.loads(content_stripped[:500])  # Try to parse first 500 chars
                return DataType.TEXT  # JSON is treated as text for NLP tasks
            except:
                pass
        
        # Check for XML structure
        if content_stripped.startswith('<?xml') or content_stripped.startswith('<'):
            return DataType.TEXT
        
        # Default to text
        return DataType.TEXT
    
    def get_supported_extensions(self, data_type: DataType) -> List[str]:
        """
        Get list of supported file extensions for a data type.
        
        Args:
            data_type: Data type to get extensions for
            
        Returns:
            List of supported extensions
        """
        return [ext for ext, dtype in self.EXTENSION_MAPPINGS.items() if dtype == data_type]
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file type is supported
        """
        try:
            detected_type = self.detect_file_type(file_path)
            return detected_type != DataType.UNKNOWN
        except Exception:
            return False


# Global detector instance
_detector = DataTypeDetector()


def detect_data_type(path: Union[str, Path]) -> DataType:
    """
    Detect data type of a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        Detected data type
        
    Raises:
        DataError: If path doesn't exist or cannot be analyzed
    """
    path = Path(path)
    
    if path.is_file():
        return _detector.detect_file_type(path)
    elif path.is_dir():
        return _detector.detect_directory_type(path)
    else:
        raise DataError(f"Path does not exist: {path}")