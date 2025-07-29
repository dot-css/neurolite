"""
Unit tests for data loading functionality.
"""

import pytest
import tempfile
import os
import json
import csv
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import numpy as np

from neurolite.data.loader import DataLoader, Dataset, DatasetInfo, load_data
from neurolite.data.detector import DataType
from neurolite.core.exceptions import DataError, DataNotFoundError, DataFormatError


class TestDataset:
    """Test cases for Dataset class."""
    
    def test_init(self):
        """Test dataset initialization."""
        data = [1, 2, 3, 4, 5]
        targets = [0, 1, 0, 1, 0]
        info = DatasetInfo(DataType.TABULAR, 5)
        
        dataset = Dataset(data, targets, info)
        
        assert dataset.data == data
        assert dataset.targets == targets
        assert dataset.info == info
        assert dataset.transform is None
    
    def test_len(self):
        """Test dataset length."""
        data = [1, 2, 3, 4, 5]
        dataset = Dataset(data)
        assert len(dataset) == 5
    
    def test_getitem(self):
        """Test dataset item access."""
        data = [10, 20, 30]
        targets = [0, 1, 0]
        dataset = Dataset(data, targets)
        
        sample, target = dataset[0]
        assert sample == 10
        assert target == 0
        
        sample, target = dataset[1]
        assert sample == 20
        assert target == 1
    
    def test_getitem_with_transform(self):
        """Test dataset item access with transform."""
        data = [1, 2, 3]
        transform = lambda x: x * 2
        dataset = Dataset(data, transform=transform)
        
        sample, target = dataset[0]
        assert sample == 2  # 1 * 2
        assert target is None
    
    def test_getitem_no_targets(self):
        """Test dataset item access without targets."""
        data = [1, 2, 3]
        dataset = Dataset(data)
        
        sample, target = dataset[0]
        assert sample == 1
        assert target is None
    
    def test_iter(self):
        """Test dataset iteration."""
        data = [1, 2, 3]
        targets = [0, 1, 0]
        dataset = Dataset(data, targets)
        
        samples_and_targets = list(dataset)
        assert len(samples_and_targets) == 3
        assert samples_and_targets[0] == (1, 0)
        assert samples_and_targets[1] == (2, 1)
        assert samples_and_targets[2] == (3, 0)
    
    def test_get_sample(self):
        """Test getting sample without target."""
        data = [10, 20, 30]
        targets = [0, 1, 0]
        dataset = Dataset(data, targets)
        
        assert dataset.get_sample(0) == 10
        assert dataset.get_sample(1) == 20
    
    def test_get_target(self):
        """Test getting target for sample."""
        data = [10, 20, 30]
        targets = [0, 1, 0]
        dataset = Dataset(data, targets)
        
        assert dataset.get_target(0) == 0
        assert dataset.get_target(1) == 1
    
    def test_get_batch(self):
        """Test getting batch of samples."""
        data = [10, 20, 30, 40]
        targets = [0, 1, 0, 1]
        dataset = Dataset(data, targets)
        
        data_batch, target_batch = dataset.get_batch([0, 2])
        assert data_batch == [10, 30]
        assert target_batch == [0, 0]


class TestDatasetInfo:
    """Test cases for DatasetInfo class."""
    
    def test_init(self):
        """Test DatasetInfo initialization."""
        info = DatasetInfo(
            data_type=DataType.IMAGE,
            num_samples=100,
            shape=(224, 224, 3),
            num_classes=10,
            class_names=['cat', 'dog'],
            feature_names=['feature1', 'feature2'],
            target_column='label',
            file_paths=['image1.jpg', 'image2.jpg'],
            metadata={'key': 'value'}
        )
        
        assert info.data_type == DataType.IMAGE
        assert info.num_samples == 100
        assert info.shape == (224, 224, 3)
        assert info.num_classes == 10
        assert info.class_names == ['cat', 'dog']
        assert info.feature_names == ['feature1', 'feature2']
        assert info.target_column == 'label'
        assert info.file_paths == ['image1.jpg', 'image2.jpg']
        assert info.metadata == {'key': 'value'}


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_init(self):
        """Test loader initialization."""
        assert self.loader is not None
        assert hasattr(self.loader, '_loaders')
        assert DataType.IMAGE in self.loader._loaders
        assert DataType.TEXT in self.loader._loaders
        assert DataType.TABULAR in self.loader._loaders
        assert DataType.AUDIO in self.loader._loaders
        assert DataType.VIDEO in self.loader._loaders
    
    def test_load_nonexistent_path(self):
        """Test loading from non-existent path."""
        with pytest.raises(DataNotFoundError):
            self.loader.load("nonexistent_path.txt")
    
    @patch('neurolite.data.loader.detect_data_type')
    def test_load_unknown_data_type(self, mock_detect):
        """Test loading with unknown data type."""
        mock_detect.return_value = DataType.UNKNOWN
        
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(DataFormatError):
                self.loader.load(temp_file.name)
    
    @patch('neurolite.data.loader.detect_data_type')
    def test_load_unsupported_data_type(self, mock_detect):
        """Test loading with unsupported data type."""
        # Create a custom unsupported data type
        class UnsupportedType:
            value = "unsupported"
        
        mock_detect.return_value = UnsupportedType()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(DataFormatError):
                self.loader.load(temp_file.name)
    
    @patch('neurolite.data.loader.safe_import')
    def test_load_image_data_single_file(self, mock_safe_import):
        """Test loading single image file."""
        # Mock PIL
        mock_pil = MagicMock()
        mock_image = MagicMock()
        mock_img_obj = MagicMock()
        mock_img_obj.mode = 'RGB'
        mock_img_obj.resize.return_value = mock_img_obj
        mock_img_obj.convert.return_value = mock_img_obj
        mock_image.open.return_value = mock_img_obj
        mock_pil.Image = mock_image
        mock_safe_import.return_value = mock_pil
        
        # Mock numpy array conversion
        with patch('numpy.array') as mock_array:
            mock_array.return_value = np.zeros((224, 224, 3))
            
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
                dataset = self.loader.load(temp_file.name, data_type=DataType.IMAGE)
                
                assert isinstance(dataset, Dataset)
                assert dataset.info.data_type == DataType.IMAGE
                assert dataset.info.num_samples == 1
                assert len(dataset) == 1
    
    def test_load_text_data_single_file(self):
        """Test loading single text file."""
        test_content = "This is a test text file.\nWith multiple lines.\nFor testing purposes."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.txt"
            with open(temp_file_path, 'w') as f:
                f.write(test_content)
            
            dataset = self.loader.load(temp_file_path, data_type=DataType.TEXT)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT
            assert len(dataset) > 0
            
            # Should split into lines
            sample, _ = dataset[0]
            assert isinstance(sample, str)
    
    def test_load_json_text_file(self):
        """Test loading JSON text file."""
        test_data = [
            {"text": "First text sample"},
            {"text": "Second text sample"},
            {"text": "Third text sample"}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.json"
            with open(temp_file_path, 'w') as f:
                json.dump(test_data, f)
            
            dataset = self.loader.load(temp_file_path, data_type=DataType.TEXT)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT
            assert len(dataset) == 3
            
            sample, _ = dataset[0]
            assert sample == "First text sample"
    
    def test_load_jsonl_text_file(self):
        """Test loading JSON Lines text file."""
        test_data = [
            {"text": "First line"},
            {"text": "Second line"},
            {"text": "Third line"}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.jsonl"
            with open(temp_file_path, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
            
            dataset = self.loader.load(temp_file_path, data_type=DataType.TEXT)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT
            assert len(dataset) == 3
            
            sample, _ = dataset[0]
            assert sample == "First line"
    
    @patch('neurolite.data.loader.safe_import')
    def test_load_tabular_data_csv(self, mock_safe_import):
        """Test loading CSV tabular data."""
        # Mock pandas
        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ['feature1', 'feature2', 'target']
        mock_df.drop.return_value.values = np.array([[1, 2], [3, 4], [5, 6]])
        mock_df.__getitem__.return_value.values = np.array([0, 1, 0])
        mock_pd.read_csv.return_value = mock_df
        mock_safe_import.return_value = mock_pd
        
        test_data = "feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.csv"
            with open(temp_file_path, 'w') as f:
                f.write(test_data)
            
            dataset = self.loader.load(temp_file_path, data_type=DataType.TABULAR, target_column='target')
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TABULAR
            assert dataset.info.target_column == 'target'
    
    @patch('neurolite.data.loader.safe_import')
    def test_load_audio_data_single_file(self, mock_safe_import):
        """Test loading single audio file."""
        # Mock librosa
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (np.zeros(44100), 44100)  # 1 second of audio
        mock_safe_import.return_value = mock_librosa
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            dataset = self.loader.load(temp_file.name, data_type=DataType.AUDIO)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.AUDIO
            assert dataset.info.num_samples == 1
            assert len(dataset) == 1
    
    def test_load_video_data_single_file(self):
        """Test loading single video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            dataset = self.loader.load(temp_file.name, data_type=DataType.VIDEO)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.VIDEO
            assert dataset.info.num_samples == 1
            assert len(dataset) == 1
    
    def test_load_image_directory_with_classes(self):
        """Test loading image directory with class structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create class directories
            (temp_path / 'cats').mkdir()
            (temp_path / 'dogs').mkdir()
            
            # Create dummy image files
            (temp_path / 'cats' / 'cat1.jpg').touch()
            (temp_path / 'cats' / 'cat2.jpg').touch()
            (temp_path / 'dogs' / 'dog1.jpg').touch()
            
            with patch('neurolite.data.loader.safe_import') as mock_safe_import:
                # Mock PIL
                mock_pil = MagicMock()
                mock_image = MagicMock()
                mock_img_obj = MagicMock()
                mock_img_obj.mode = 'RGB'
                mock_img_obj.resize.return_value = mock_img_obj
                mock_img_obj.convert.return_value = mock_img_obj
                mock_image.open.return_value = mock_img_obj
                mock_pil.Image = mock_image
                mock_safe_import.return_value = mock_pil
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((224, 224, 3))
                    
                    dataset = self.loader.load(temp_dir, data_type=DataType.IMAGE)
                    
                    assert isinstance(dataset, Dataset)
                    assert dataset.info.data_type == DataType.IMAGE
                    assert dataset.info.num_classes == 2
                    assert set(dataset.info.class_names) == {'cats', 'dogs'}
                    assert len(dataset) == 3
    
    def test_load_text_directory_with_classes(self):
        """Test loading text directory with class structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create class directories
            (temp_path / 'positive').mkdir()
            (temp_path / 'negative').mkdir()
            
            # Create text files
            with open(temp_path / 'positive' / 'pos1.txt', 'w') as f:
                f.write("This is positive text.")
            with open(temp_path / 'positive' / 'pos2.txt', 'w') as f:
                f.write("Another positive example.")
            with open(temp_path / 'negative' / 'neg1.txt', 'w') as f:
                f.write("This is negative text.")
            
            dataset = self.loader.load(temp_dir, data_type=DataType.TEXT)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT
            assert dataset.info.num_classes == 2
            assert set(dataset.info.class_names) == {'negative', 'positive'}
            assert len(dataset) == 3
    
    @patch('neurolite.data.loader.safe_import')
    def test_load_tabular_missing_target_column(self, mock_safe_import):
        """Test loading tabular data with missing target column."""
        # Mock pandas
        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ['feature1', 'feature2']
        mock_pd.read_csv.return_value = mock_df
        mock_safe_import.return_value = mock_pd
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            with pytest.raises(DataError, match="Target column 'missing_column' not found"):
                self.loader.load(temp_file.name, data_type=DataType.TABULAR, target_column='missing_column')
    
    def test_load_with_data_type_override(self):
        """Test loading with data type override."""
        test_content = "This is text content."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.unknown"
            with open(temp_file_path, 'w') as f:
                f.write(test_content)
            
            # Override detection to treat as text
            dataset = self.loader.load(temp_file_path, data_type=DataType.TEXT)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT


class TestLoadDataFunction:
    """Test cases for the load_data function."""
    
    def test_load_data_function(self):
        """Test the load_data convenience function."""
        test_content = "Test content for loading."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.txt"
            with open(temp_file_path, 'w') as f:
                f.write(test_content)
            
            dataset = load_data(temp_file_path)
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT
    
    def test_load_data_with_parameters(self):
        """Test load_data function with parameters."""
        test_content = "Test content."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "test.txt"
            with open(temp_file_path, 'w') as f:
                f.write(test_content)
            
            dataset = load_data(
                temp_file_path,
                data_type=DataType.TEXT,
                encoding='utf-8'
            )
            
            assert isinstance(dataset, Dataset)
            assert dataset.info.data_type == DataType.TEXT