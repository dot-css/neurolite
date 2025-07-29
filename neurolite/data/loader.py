"""
Unified data loading interfaces for NeuroLite.

Provides unified data loading for different file formats and directory structures,
handling images, text, CSV, audio, and video files with automatic preprocessing.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import numpy as np

from ..core import get_logger, DataError, DataNotFoundError, DataFormatError, safe_import
from .detector import DataType, detect_data_type


logger = get_logger(__name__)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    data_type: DataType
    num_samples: int
    shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    target_column: Optional[str] = None
    file_paths: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Dataset:
    """
    Unified dataset container for different data types.
    
    Provides a consistent interface for accessing data regardless of the
    underlying format (images, text, tabular, etc.).
    """
    
    def __init__(
        self,
        data: Any,
        targets: Optional[Any] = None,
        info: Optional[DatasetInfo] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: The actual data (numpy array, list, etc.)
            targets: Target labels/values (optional)
            info: Dataset information
            transform: Optional transform function to apply to data
        """
        self.data = data
        self.targets = targets
        self.info = info or DatasetInfo(DataType.UNKNOWN, 0)
        self.transform = transform
    
    def __len__(self) -> int:
        """Get number of samples in dataset."""
        if hasattr(self.data, '__len__'):
            return len(self.data)
        return self.info.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, target)
        """
        if isinstance(self.data, (list, tuple)):
            sample = self.data[idx]
        elif hasattr(self.data, '__getitem__'):
            sample = self.data[idx]
        else:
            raise DataError(f"Cannot index data of type {type(self.data)}")
        
        if self.transform:
            sample = self.transform(sample)
        
        target = None
        if self.targets is not None:
            if isinstance(self.targets, (list, tuple)):
                target = self.targets[idx]
            elif hasattr(self.targets, '__getitem__'):
                target = self.targets[idx]
        
        return sample, target
    
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """Iterate over dataset samples."""
        for i in range(len(self)):
            yield self[i]
    
    def get_sample(self, idx: int) -> Any:
        """Get data sample without target."""
        sample, _ = self[idx]
        return sample
    
    def get_target(self, idx: int) -> Any:
        """Get target for sample."""
        _, target = self[idx]
        return target
    
    def get_batch(self, indices: List[int]) -> Tuple[List[Any], List[Any]]:
        """
        Get batch of samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (data_batch, target_batch)
        """
        data_batch = []
        target_batch = []
        
        for idx in indices:
            sample, target = self[idx]
            data_batch.append(sample)
            target_batch.append(target)
        
        return data_batch, target_batch


class DataLoader:
    """
    Unified data loader for different data types and formats.
    
    Automatically detects data type and loads data using appropriate methods.
    Supports images, text, CSV/tabular data, audio, and video files.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self._loaders = {
            DataType.IMAGE: self._load_image_data,
            DataType.TEXT: self._load_text_data,
            DataType.TABULAR: self._load_tabular_data,
            DataType.AUDIO: self._load_audio_data,
            DataType.VIDEO: self._load_video_data,
        }
    
    def load(
        self,
        path: Union[str, Path],
        data_type: Optional[DataType] = None,
        target_column: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load data from file or directory.
        
        Args:
            path: Path to data file or directory
            data_type: Override automatic data type detection
            target_column: Target column name for tabular data
            **kwargs: Additional loader-specific arguments
            
        Returns:
            Loaded dataset
            
        Raises:
            DataNotFoundError: If path doesn't exist
            DataFormatError: If data format is not supported
            DataError: If loading fails
        """
        path = Path(path)
        
        if not path.exists():
            raise DataNotFoundError(str(path))
        
        # Detect data type if not provided
        if data_type is None:
            data_type = detect_data_type(path)
            logger.debug(f"Auto-detected data type: {data_type.value}")
        
        if data_type == DataType.UNKNOWN:
            supported_types = [dt.value for dt in self._loaders.keys()]
            raise DataFormatError("unknown", supported_types)
        
        if data_type not in self._loaders:
            supported_types = [dt.value for dt in self._loaders.keys()]
            raise DataFormatError(data_type.value, supported_types)
        
        logger.info(f"Loading {data_type.value} data from: {path}")
        
        try:
            return self._loaders[data_type](path, target_column=target_column, **kwargs)
        except Exception as e:
            raise DataError(f"Failed to load {data_type.value} data: {str(e)}") from e
    
    def _load_image_data(
        self,
        path: Path,
        target_column: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dataset:
        """
        Load image data from file or directory.
        
        Args:
            path: Path to image file or directory
            target_column: Not used for images
            image_size: Target image size (width, height)
            **kwargs: Additional arguments
            
        Returns:
            Image dataset
        """
        # Import required libraries
        PIL = safe_import('PIL', 'image processing')
        Image = PIL.Image
        
        if path.is_file():
            # Single image file
            try:
                image = Image.open(path)
                if image_size:
                    image = image.resize(image_size)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy array
                data = [np.array(image)]
                file_paths = [str(path)]
                
                info = DatasetInfo(
                    data_type=DataType.IMAGE,
                    num_samples=1,
                    shape=data[0].shape,
                    file_paths=file_paths
                )
                
                return Dataset(data, info=info)
                
            except Exception as e:
                raise DataError(f"Failed to load image {path}: {e}")
        
        else:
            # Directory of images
            return self._load_image_directory(path, image_size, **kwargs)
    
    def _load_image_directory(
        self,
        path: Path,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dataset:
        """Load images from directory structure."""
        PIL = safe_import('PIL', 'image processing')
        Image = PIL.Image
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find all image files
        image_files = []
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            raise DataError(f"No image files found in directory: {path}")
        
        # Sort for consistent ordering
        image_files.sort()
        
        # Check if directory structure indicates classification task
        class_names = None
        targets = None
        
        # Look for subdirectory structure (class folders)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            class_names = sorted([d.name for d in subdirs])
            targets = []
            
            # Map file paths to class labels
            for img_path in image_files:
                # Find which subdirectory this image belongs to
                for i, class_name in enumerate(class_names):
                    if (path / class_name) in img_path.parents:
                        targets.append(i)
                        break
                else:
                    # Image not in any class subdirectory
                    targets.append(-1)  # Unknown class
        
        # Load images
        data = []
        valid_files = []
        valid_targets = []
        
        for i, img_path in enumerate(image_files):
            try:
                image = Image.open(img_path)
                if image_size:
                    image = image.resize(image_size)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                data.append(np.array(image))
                valid_files.append(str(img_path))
                
                if targets is not None:
                    valid_targets.append(targets[i])
                    
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
        
        if not data:
            raise DataError(f"No valid images could be loaded from: {path}")
        
        # Create dataset info
        info = DatasetInfo(
            data_type=DataType.IMAGE,
            num_samples=len(data),
            shape=data[0].shape if data else None,
            num_classes=len(class_names) if class_names else None,
            class_names=class_names,
            file_paths=valid_files
        )
        
        return Dataset(data, targets=valid_targets if valid_targets else None, info=info)
    
    def _load_text_data(
        self,
        path: Path,
        target_column: Optional[str] = None,
        encoding: str = 'utf-8',
        **kwargs
    ) -> Dataset:
        """
        Load text data from file or directory.
        
        Args:
            path: Path to text file or directory
            target_column: Not used for plain text
            encoding: Text encoding
            **kwargs: Additional arguments
            
        Returns:
            Text dataset
        """
        if path.is_file():
            # Single text file
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Split into lines or sentences for processing
                if path.suffix.lower() in ['.json', '.jsonl']:
                    data = self._load_json_text(path, encoding)
                else:
                    # Split by lines, filter empty lines
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    data = lines if lines else [content]
                
                info = DatasetInfo(
                    data_type=DataType.TEXT,
                    num_samples=len(data),
                    file_paths=[str(path)]
                )
                
                return Dataset(data, info=info)
                
            except Exception as e:
                raise DataError(f"Failed to load text file {path}: {e}")
        
        else:
            # Directory of text files
            return self._load_text_directory(path, encoding, **kwargs)
    
    def _load_json_text(self, path: Path, encoding: str) -> List[str]:
        """Load text data from JSON file."""
        import json
        
        with open(path, 'r', encoding=encoding) as f:
            if path.suffix.lower() == '.jsonl':
                # JSON Lines format
                data = []
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # Extract text field (common field names)
                            text_fields = ['text', 'content', 'message', 'body', 'description']
                            for field in text_fields:
                                if field in obj:
                                    data.append(str(obj[field]))
                                    break
                            else:
                                # Use string representation if no text field found
                                data.append(str(obj))
                        else:
                            data.append(str(obj))
                return data
            else:
                # Regular JSON
                obj = json.load(f)
                if isinstance(obj, list):
                    # Handle list of objects - try to extract text fields
                    data = []
                    for item in obj:
                        if isinstance(item, dict):
                            # Extract text field (common field names)
                            text_fields = ['text', 'content', 'message', 'body', 'description']
                            for field in text_fields:
                                if field in item:
                                    data.append(str(item[field]))
                                    break
                            else:
                                # Use string representation if no text field found
                                data.append(str(item))
                        else:
                            data.append(str(item))
                    return data
                elif isinstance(obj, dict):
                    # Try to extract text content
                    text_fields = ['text', 'content', 'data', 'items']
                    for field in text_fields:
                        if field in obj:
                            content = obj[field]
                            if isinstance(content, list):
                                return [str(item) for item in content]
                            else:
                                return [str(content)]
                    return [str(obj)]
                else:
                    return [str(obj)]
    
    def _load_text_directory(
        self,
        path: Path,
        encoding: str = 'utf-8',
        **kwargs
    ) -> Dataset:
        """Load text files from directory."""
        # Supported text extensions
        text_extensions = {'.txt', '.md', '.rst', '.log', '.json', '.jsonl', '.xml', '.html', '.htm'}
        
        # Find all text files
        text_files = []
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                text_files.append(file_path)
        
        if not text_files:
            raise DataError(f"No text files found in directory: {path}")
        
        # Sort for consistent ordering
        text_files.sort()
        
        # Check for classification structure (subdirectories)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        class_names = None
        targets = None
        
        if subdirs:
            class_names = sorted([d.name for d in subdirs])
            targets = []
        
        # Load text files
        data = []
        valid_files = []
        valid_targets = []
        
        for text_path in text_files:
            try:
                with open(text_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read().strip()
                
                if content:
                    data.append(content)
                    valid_files.append(str(text_path))
                    
                    # Determine class if using directory structure
                    if class_names:
                        for i, class_name in enumerate(class_names):
                            if (path / class_name) in text_path.parents:
                                valid_targets.append(i)
                                break
                        else:
                            valid_targets.append(-1)  # Unknown class
                            
            except Exception as e:
                logger.warning(f"Failed to load text file {text_path}: {e}")
                continue
        
        if not data:
            raise DataError(f"No valid text files could be loaded from: {path}")
        
        info = DatasetInfo(
            data_type=DataType.TEXT,
            num_samples=len(data),
            num_classes=len(class_names) if class_names else None,
            class_names=class_names,
            file_paths=valid_files
        )
        
        return Dataset(data, targets=valid_targets if valid_targets else None, info=info)
    
    def _load_tabular_data(
        self,
        path: Path,
        target_column: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load tabular data from CSV, Excel, or other formats.
        
        Args:
            path: Path to tabular data file
            target_column: Name of target column
            **kwargs: Additional arguments for pandas
            
        Returns:
            Tabular dataset
        """
        pd = safe_import('pandas', 'tabular data processing')
        
        try:
            # Load data based on file extension
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path, **kwargs)
            elif path.suffix.lower() == '.tsv':
                df = pd.read_csv(path, sep='\t', **kwargs)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(path, **kwargs)
            elif path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path, **kwargs)
            elif path.suffix.lower() == '.feather':
                df = pd.read_feather(path, **kwargs)
            elif path.suffix.lower() in ['.h5', '.hdf5']:
                # For HDF5, we need a key - try common ones
                keys = kwargs.get('key', ['data', 'df', 'table'])
                if isinstance(keys, str):
                    keys = [keys]
                
                df = None
                for key in keys:
                    try:
                        df = pd.read_hdf(path, key=key)
                        break
                    except KeyError:
                        continue
                
                if df is None:
                    raise DataError(f"Could not find valid key in HDF5 file. Available keys: {pd.HDFStore(path).keys()}")
            else:
                raise DataFormatError(path.suffix, ['.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.feather', '.h5'])
            
            # Separate features and targets
            targets = None
            feature_names = list(df.columns)
            
            if target_column:
                if target_column not in df.columns:
                    raise DataError(f"Target column '{target_column}' not found in data. Available columns: {list(df.columns)}")
                
                targets = df[target_column].values
                data = df.drop(columns=[target_column])
                feature_names = list(data.columns)
            else:
                data = df
            
            # Convert to numpy array
            data_array = data.values
            
            # Determine if this is a classification task
            num_classes = None
            class_names = None
            
            if targets is not None:
                unique_targets = pd.Series(targets).unique()
                if len(unique_targets) < len(targets) * 0.5:  # Heuristic for classification
                    num_classes = len(unique_targets)
                    class_names = [str(t) for t in sorted(unique_targets)]
            
            info = DatasetInfo(
                data_type=DataType.TABULAR,
                num_samples=len(data_array),
                shape=data_array.shape,
                num_classes=num_classes,
                class_names=class_names,
                feature_names=feature_names,
                target_column=target_column,
                file_paths=[str(path)]
            )
            
            return Dataset(data_array, targets=targets, info=info)
            
        except Exception as e:
            raise DataError(f"Failed to load tabular data from {path}: {e}")
    
    def _load_audio_data(
        self,
        path: Path,
        target_column: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> Dataset:
        """
        Load audio data from file or directory.
        
        Args:
            path: Path to audio file or directory
            target_column: Not used for audio
            sample_rate: Target sample rate for resampling
            **kwargs: Additional arguments
            
        Returns:
            Audio dataset
        """
        # Try to import audio processing library
        try:
            librosa = safe_import('librosa', 'audio processing')
        except:
            # Fallback to basic audio loading
            logger.warning("librosa not available, using basic audio loading")
            return self._load_audio_basic(path, sample_rate, **kwargs)
        
        if path.is_file():
            # Single audio file
            try:
                audio_data, sr = librosa.load(str(path), sr=sample_rate)
                
                info = DatasetInfo(
                    data_type=DataType.AUDIO,
                    num_samples=1,
                    shape=audio_data.shape,
                    file_paths=[str(path)],
                    metadata={'sample_rate': sr}
                )
                
                return Dataset([audio_data], info=info)
                
            except Exception as e:
                raise DataError(f"Failed to load audio file {path}: {e}")
        
        else:
            # Directory of audio files
            return self._load_audio_directory(path, sample_rate, **kwargs)
    
    def _load_audio_basic(self, path: Path, sample_rate: Optional[int] = None, **kwargs) -> Dataset:
        """Basic audio loading without librosa."""
        # This is a placeholder - in practice, you'd implement basic WAV loading
        # or require librosa for audio processing
        raise DataError("Audio loading requires librosa. Install with: pip install librosa")
    
    def _load_audio_directory(
        self,
        path: Path,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> Dataset:
        """Load audio files from directory."""
        librosa = safe_import('librosa', 'audio processing')
        
        # Supported audio extensions
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(path.rglob(f'*{ext}'))
            audio_files.extend(path.rglob(f'*{ext.upper()}'))
        
        if not audio_files:
            raise DataError(f"No audio files found in directory: {path}")
        
        # Sort for consistent ordering
        audio_files.sort()
        
        # Check for classification structure
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        class_names = None
        targets = None
        
        if subdirs:
            class_names = sorted([d.name for d in subdirs])
            targets = []
        
        # Load audio files
        data = []
        valid_files = []
        valid_targets = []
        sample_rates = []
        
        for audio_path in audio_files:
            try:
                audio_data, sr = librosa.load(str(audio_path), sr=sample_rate)
                
                data.append(audio_data)
                valid_files.append(str(audio_path))
                sample_rates.append(sr)
                
                # Determine class if using directory structure
                if class_names:
                    for i, class_name in enumerate(class_names):
                        if (path / class_name) in audio_path.parents:
                            valid_targets.append(i)
                            break
                    else:
                        valid_targets.append(-1)  # Unknown class
                        
            except Exception as e:
                logger.warning(f"Failed to load audio file {audio_path}: {e}")
                continue
        
        if not data:
            raise DataError(f"No valid audio files could be loaded from: {path}")
        
        info = DatasetInfo(
            data_type=DataType.AUDIO,
            num_samples=len(data),
            shape=data[0].shape if data else None,
            num_classes=len(class_names) if class_names else None,
            class_names=class_names,
            file_paths=valid_files,
            metadata={'sample_rates': sample_rates}
        )
        
        return Dataset(data, targets=valid_targets if valid_targets else None, info=info)
    
    def _load_video_data(
        self,
        path: Path,
        target_column: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Load video data from file or directory.
        
        Args:
            path: Path to video file or directory
            target_column: Not used for video
            **kwargs: Additional arguments
            
        Returns:
            Video dataset
        """
        # Video loading is complex and requires specialized libraries
        # This is a placeholder implementation
        logger.warning("Video data loading is not fully implemented yet")
        
        if path.is_file():
            info = DatasetInfo(
                data_type=DataType.VIDEO,
                num_samples=1,
                file_paths=[str(path)]
            )
            return Dataset([str(path)], info=info)  # Return file path as placeholder
        
        else:
            # Find video files
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(path.rglob(f'*{ext}'))
                video_files.extend(path.rglob(f'*{ext.upper()}'))
            
            if not video_files:
                raise DataError(f"No video files found in directory: {path}")
            
            video_files.sort()
            file_paths = [str(f) for f in video_files]
            
            info = DatasetInfo(
                data_type=DataType.VIDEO,
                num_samples=len(file_paths),
                file_paths=file_paths
            )
            
            return Dataset(file_paths, info=info)  # Return file paths as placeholder


# Global loader instance
_loader = DataLoader()


def load_data(
    path: Union[str, Path],
    data_type: Optional[DataType] = None,
    target_column: Optional[str] = None,
    **kwargs
) -> Dataset:
    """
    Load data from file or directory with automatic type detection.
    
    Args:
        path: Path to data file or directory
        data_type: Override automatic data type detection
        target_column: Target column name for tabular data
        **kwargs: Additional loader-specific arguments
        
    Returns:
        Loaded dataset
        
    Raises:
        DataNotFoundError: If path doesn't exist
        DataFormatError: If data format is not supported
        DataError: If loading fails
    """
    return _loader.load(path, data_type=data_type, target_column=target_column, **kwargs)