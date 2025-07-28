"""
Unit tests for DomainDetector computer vision functionality.
"""

import pytest
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from PIL import Image
import tempfile
import os

from neurolite.detectors.domain_detector import DomainDetector
from neurolite.core.data_models import CVTaskAnalysis
from neurolite.core.exceptions import NeuroLiteException, UnsupportedFormatError


class TestDomainDetectorCV:
    """Test cases for computer vision detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DomainDetector()
    
    def test_init(self):
        """Test DomainDetector initialization."""
        detector = DomainDetector()
        assert detector is not None
    
    @patch('neurolite.detectors.domain_detector.Path.exists')
    @patch('neurolite.detectors.domain_detector.Path.is_dir')
    def test_detect_cv_task_directory(self, mock_is_dir, mock_exists):
        """Test CV task detection from directory."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        
        with patch.object(self.detector, '_analyze_image_directory') as mock_analyze:
            expected_result = CVTaskAnalysis(
                task_type='classification',
                task_subtype='multi_class',
                confidence=0.85,
                num_classes=3
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_cv_task('/fake/path')
            
            assert result == expected_result
            mock_analyze.assert_called_once()
    
    @patch('neurolite.detectors.domain_detector.Path.exists')
    @patch('neurolite.detectors.domain_detector.Path.is_file')
    def test_detect_cv_task_file(self, mock_is_file, mock_exists):
        """Test CV task detection from annotation file."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        with patch.object(self.detector, '_analyze_annotation_file') as mock_analyze:
            expected_result = CVTaskAnalysis(
                task_type='object_detection',
                task_subtype='bbox_detection',
                confidence=0.9
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_cv_task('/fake/file.json')
            
            assert result == expected_result
            mock_analyze.assert_called_once()
    
    def test_detect_cv_task_image_list(self):
        """Test CV task detection from image list."""
        image_list = ['/fake/img1.jpg', '/fake/img2.jpg']
        
        with patch.object(self.detector, '_analyze_image_list') as mock_analyze:
            expected_result = CVTaskAnalysis(
                task_type='classification',
                task_subtype='unknown',
                confidence=0.5
            )
            mock_analyze.return_value = expected_result
            
            result = self.detector.detect_cv_task(image_list)
            
            assert result == expected_result
            mock_analyze.assert_called_once_with(image_list)
    
    @patch('neurolite.detectors.domain_detector.Path.exists')
    def test_detect_cv_task_nonexistent_path(self, mock_exists):
        """Test CV task detection with nonexistent path."""
        mock_exists.return_value = False
        
        with pytest.raises(NeuroLiteException, match="Path does not exist"):
            self.detector.detect_cv_task('/nonexistent/path')
    
    def test_detect_cv_task_unsupported_type(self):
        """Test CV task detection with unsupported data type."""
        with pytest.raises(NeuroLiteException, match="Unsupported data source type"):
            self.detector.detect_cv_task(123)
    
    def test_analyze_json_annotations_coco_object_detection(self):
        """Test COCO JSON format detection for object detection."""
        coco_data = {
            "images": [{"id": 1, "file_name": "image1.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 50]
                }
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"}
            ]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(coco_data))):
            result = self.detector._analyze_json_annotations(Path('fake.json'))
            
            assert result.task_type == 'object_detection'
            assert result.task_subtype == 'bbox_detection'
            assert result.confidence == 0.9
            assert result.num_classes == 2
            assert result.annotation_format == 'coco_json'
    
    def test_analyze_json_annotations_coco_segmentation(self):
        """Test COCO JSON format detection for segmentation."""
        coco_data = {
            "images": [{"id": 1, "file_name": "image1.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]
                }
            ],
            "categories": [{"id": 1, "name": "person"}]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(coco_data))):
            result = self.detector._analyze_json_annotations(Path('fake.json'))
            
            assert result.task_type == 'segmentation'
            assert result.task_subtype == 'instance_segmentation'
            assert result.confidence == 0.9
            assert result.num_classes == 1
    
    def test_analyze_json_annotations_coco_classification(self):
        """Test COCO JSON format detection for classification."""
        coco_data = {
            "images": [{"id": 1, "file_name": "image1.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1
                }
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(coco_data))):
            result = self.detector._analyze_json_annotations(Path('fake.json'))
            
            assert result.task_type == 'classification'
            assert result.task_subtype == 'binary'
            assert result.confidence == 0.85
            assert result.num_classes == 2
    
    def test_analyze_json_annotations_custom_format(self):
        """Test custom JSON format detection."""
        custom_data = [
            {"image": "img1.jpg", "bbox": [10, 10, 50, 50], "class": "person"},
            {"image": "img2.jpg", "bbox": [20, 20, 60, 60], "class": "car"}
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(custom_data))):
            result = self.detector._analyze_json_annotations(Path('fake.json'))
            
            assert result.task_type == 'object_detection'
            assert result.task_subtype == 'bbox_detection'
            assert result.confidence == 0.7
            assert result.annotation_format == 'custom_json'
    
    def test_analyze_json_annotations_invalid_json(self):
        """Test handling of invalid JSON."""
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with pytest.raises(NeuroLiteException, match="Invalid JSON annotation file"):
                self.detector._analyze_json_annotations(Path('fake.json'))
    
    def test_analyze_xml_annotations_pascal_voc(self):
        """Test Pascal VOC XML format detection."""
        xml_content = '''<?xml version="1.0"?>
        <annotation>
            <filename>image.jpg</filename>
            <object>
                <name>person</name>
                <bndbox>
                    <xmin>10</xmin>
                    <ymin>10</ymin>
                    <xmax>50</xmax>
                    <ymax>50</ymax>
                </bndbox>
            </object>
            <object>
                <name>car</name>
                <bndbox>
                    <xmin>60</xmin>
                    <ymin>60</ymin>
                    <xmax>100</xmax>
                    <ymax>100</ymax>
                </bndbox>
            </object>
        </annotation>'''
        
        with patch('builtins.open', mock_open(read_data=xml_content)):
            with patch('xml.etree.ElementTree.parse') as mock_parse:
                root = ET.fromstring(xml_content)
                mock_tree = Mock()
                mock_tree.getroot.return_value = root
                mock_parse.return_value = mock_tree
                
                result = self.detector._analyze_xml_annotations(Path('fake.xml'))
                
                assert result.task_type == 'object_detection'
                assert result.task_subtype == 'bbox_detection'
                assert result.confidence == 0.9
                assert result.num_classes == 2
                assert result.annotation_format == 'pascal_voc'
    
    def test_analyze_text_annotations_yolo_format(self):
        """Test YOLO text format detection."""
        yolo_content = '''0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.2
0 0.8 0.2 0.15 0.25'''
        
        with patch('builtins.open', mock_open(read_data=yolo_content)):
            result = self.detector._analyze_text_annotations(Path('fake.txt'))
            
            assert result.task_type == 'object_detection'
            assert result.task_subtype == 'bbox_detection'
            assert result.confidence == 0.85
            assert result.num_classes == 2  # Classes 0 and 1
            assert result.annotation_format == 'yolo'
    
    def test_analyze_text_annotations_classification_labels(self):
        """Test text classification labels format."""
        labels_content = '''cat
dog
cat
bird
dog'''
        
        with patch('builtins.open', mock_open(read_data=labels_content)):
            result = self.detector._analyze_text_annotations(Path('fake.txt'))
            
            assert result.task_type == 'classification'
            assert result.task_subtype == 'multi_class'
            assert result.confidence == 0.7
            assert result.num_classes == 3  # cat, dog, bird
            assert result.annotation_format == 'text_labels'
    
    @patch('pandas.read_csv')
    def test_analyze_csv_annotations_object_detection(self, mock_read_csv):
        """Test CSV format detection for object detection."""
        import pandas as pd
        
        mock_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg'],
            'xmin': [10, 20],
            'ymin': [10, 20],
            'xmax': [50, 60],
            'ymax': [50, 60],
            'class': ['person', 'car']
        })
        mock_read_csv.return_value = mock_df
        
        result = self.detector._analyze_csv_annotations(Path('fake.csv'))
        
        assert result.task_type == 'object_detection'
        assert result.task_subtype == 'bbox_detection'
        assert result.confidence == 0.8
        assert result.num_classes == 2
        assert result.annotation_format == 'csv'
    
    @patch('pandas.read_csv')
    def test_analyze_csv_annotations_classification(self, mock_read_csv):
        """Test CSV format detection for classification."""
        import pandas as pd
        
        mock_df = pd.DataFrame({
            'image': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'label': ['cat', 'dog', 'cat']
        })
        mock_read_csv.return_value = mock_df
        
        result = self.detector._analyze_csv_annotations(Path('fake.csv'))
        
        assert result.task_type == 'classification'
        assert result.task_subtype == 'binary'
        assert result.confidence == 0.8
        assert result.num_classes == 2
        assert result.annotation_format == 'csv'
    
    def test_analyze_annotation_file_unsupported_format(self):
        """Test handling of unsupported annotation format."""
        with pytest.raises(UnsupportedFormatError, match="Unsupported annotation format"):
            self.detector._analyze_annotation_file(Path('fake.xyz'))
    
    @patch('neurolite.detectors.domain_detector.Path.exists')
    @patch('neurolite.detectors.domain_detector.Path.is_file')
    def test_analyze_image_list_valid_paths(self, mock_is_file, mock_exists):
        """Test analysis of valid image list."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        image_paths = ['/fake/img1.jpg', '/fake/img2.jpg']
        
        with patch.object(self.detector, '_analyze_image_characteristics') as mock_analyze_chars:
            mock_analyze_chars.return_value = {'total_images': 2}
            
            result = self.detector._analyze_image_list(image_paths)
            
            assert result.task_type == 'classification'
            assert result.task_subtype == 'unknown'
            assert result.confidence == 0.5
            assert result.annotation_format == 'image_list'
    
    def test_analyze_image_list_empty_list(self):
        """Test analysis of empty image list."""
        with pytest.raises(NeuroLiteException, match="Empty image list provided"):
            self.detector._analyze_image_list([])
    
    @patch('neurolite.detectors.domain_detector.Path.exists')
    def test_analyze_image_list_no_valid_files(self, mock_exists):
        """Test analysis when no valid image files exist."""
        mock_exists.return_value = False
        
        image_paths = ['/fake/img1.jpg', '/fake/img2.jpg']
        
        with pytest.raises(NeuroLiteException, match="No valid image files found in list"):
            self.detector._analyze_image_list(image_paths)
    
    @patch('PIL.Image.open')
    def test_analyze_image_characteristics(self, mock_image_open):
        """Test image characteristics analysis."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.format = 'JPEG'
        mock_img.size = (640, 480)
        mock_img.mode = 'RGB'
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        image_paths = [Path('/fake/img1.jpg'), Path('/fake/img2.jpg')]
        
        result = self.detector._analyze_image_characteristics(image_paths)
        
        assert result['total_images'] == 2
        assert result['formats']['JPEG'] == 2
        assert result['avg_width'] == 640
        assert result['avg_height'] == 480
        assert result['color_modes']['RGB'] == 2
        assert (640, 480) in result['resolutions']
    
    def test_find_annotation_files(self):
        """Test finding annotation files in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'annotations.json').touch()
            (temp_path / 'labels.xml').touch()
            (temp_path / 'data.txt').touch()
            (temp_path / 'readme.txt').touch()  # Should be excluded
            (temp_path / 'image.jpg').touch()  # Not an annotation file
            
            result = self.detector._find_annotation_files(temp_path)
            
            # Should find 3 annotation files (excluding readme.txt)
            assert len(result) == 3
            filenames = [f.name for f in result]
            assert 'annotations.json' in filenames
            assert 'labels.xml' in filenames
            assert 'data.txt' in filenames
            assert 'readme.txt' not in filenames
            assert 'image.jpg' not in filenames


if __name__ == '__main__':
    pytest.main([__file__])