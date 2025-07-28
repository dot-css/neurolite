"""
Domain-specific data detection module.

This module provides functionality to detect domain-specific data patterns
for computer vision, NLP, and time series data.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import re

from ..core.data_models import CVTaskAnalysis, NLPTaskAnalysis, TimeSeriesAnalysis
from ..core.exceptions import NeuroLiteException, UnsupportedFormatError


class DomainDetector:
    """Detects domain-specific data patterns and task types."""
    
    def __init__(self):
        """Initialize the DomainDetector."""
        pass
    
    # Computer Vision Detection Methods (Task 6.1)
    
    def detect_cv_task(self, data_source: Union[str, Path, List[str]]) -> CVTaskAnalysis:
        """
        Detect computer vision task type and format.
        
        Args:
            data_source: Path to image directory, annotation file, or list of image paths
            
        Returns:
            CVTaskAnalysis object with detected CV task information
            
        Raises:
            NeuroLiteException: If CV task cannot be determined
        """
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            
            if data_path.is_dir():
                return self._analyze_image_directory(data_path)
            elif data_path.is_file():
                return self._analyze_annotation_file(data_path)
            else:
                raise NeuroLiteException(f"Path does not exist: {data_path}")
                
        elif isinstance(data_source, list):
            return self._analyze_image_list(data_source)
        else:
            raise NeuroLiteException(f"Unsupported data source type: {type(data_source)}")
    
    def _analyze_image_directory(self, dir_path: Path) -> CVTaskAnalysis:
        """Analyze image directory structure to detect CV task."""
        try:
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(dir_path.glob(f"**/*{ext}"))
                image_files.extend(dir_path.glob(f"**/*{ext.upper()}"))
            
            if not image_files:
                raise NeuroLiteException("No image files found in directory")
            
            # Analyze directory structure for classification patterns
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            
            # Check for classification structure (subdirectories as classes)
            if subdirs and all(any(f.suffix.lower() in image_extensions 
                                 for f in subdir.iterdir() if f.is_file()) 
                             for subdir in subdirs):
                
                num_classes = len(subdirs)
                task_subtype = "multi_class" if num_classes > 2 else "binary"
                
                # Analyze image characteristics
                image_chars = self._analyze_image_characteristics(image_files[:10])
                
                return CVTaskAnalysis(
                    task_type='classification',
                    task_subtype=task_subtype,
                    confidence=0.85,
                    num_classes=num_classes,
                    annotation_format='directory_structure',
                    image_characteristics=image_chars
                )
            
            # Check for annotation files that might indicate other tasks
            annotation_files = self._find_annotation_files(dir_path)
            
            if annotation_files:
                # Analyze the first annotation file found
                return self._analyze_annotation_file(annotation_files[0])
            
            # Default to unknown classification task
            image_chars = self._analyze_image_characteristics(image_files[:10])
            
            return CVTaskAnalysis(
                task_type='classification',
                task_subtype='unknown',
                confidence=0.5,
                annotation_format='none_detected',
                image_characteristics=image_chars
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze image directory: {str(e)}")
    
    def _analyze_annotation_file(self, file_path: Path) -> CVTaskAnalysis:
        """Analyze annotation file to detect CV task type."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.json':
                return self._analyze_json_annotations(file_path)
            elif file_ext == '.xml':
                return self._analyze_xml_annotations(file_path)
            elif file_ext == '.txt':
                return self._analyze_text_annotations(file_path)
            elif file_ext == '.csv':
                return self._analyze_csv_annotations(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported annotation format: {file_ext}")
                
        except UnsupportedFormatError:
            # Re-raise UnsupportedFormatError without wrapping
            raise
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze annotation file: {str(e)}")
    
    def _analyze_json_annotations(self, file_path: Path) -> CVTaskAnalysis:
        """Analyze JSON annotation file (COCO format detection)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check for COCO format structure
            if isinstance(data, dict) and 'annotations' in data and 'images' in data:
                annotations = data['annotations']
                
                if not annotations:
                    return CVTaskAnalysis(
                        task_type='unknown',
                        task_subtype='empty_annotations',
                        confidence=0.3,
                        annotation_format='coco_json'
                    )
                
                # Check annotation structure to determine task type
                first_annotation = annotations[0]
                
                if 'bbox' in first_annotation and 'segmentation' not in first_annotation:
                    # Object detection (bounding boxes only)
                    num_classes = len(data.get('categories', []))
                    
                    return CVTaskAnalysis(
                        task_type='object_detection',
                        task_subtype='bbox_detection',
                        confidence=0.9,
                        num_classes=num_classes,
                        annotation_format='coco_json',
                        image_characteristics={'total_images': len(data['images'])}
                    )
                
                elif 'segmentation' in first_annotation:
                    # Segmentation task
                    if isinstance(first_annotation['segmentation'], list):
                        # Polygon segmentation (instance)
                        task_subtype = 'instance_segmentation'
                    else:
                        # RLE segmentation (could be semantic or instance)
                        task_subtype = 'segmentation'
                    
                    num_classes = len(data.get('categories', []))
                    
                    return CVTaskAnalysis(
                        task_type='segmentation',
                        task_subtype=task_subtype,
                        confidence=0.9,
                        num_classes=num_classes,
                        annotation_format='coco_json',
                        image_characteristics={'total_images': len(data['images'])}
                    )
                
                elif 'category_id' in first_annotation:
                    # Classification task
                    num_classes = len(data.get('categories', []))
                    task_subtype = "multi_class" if num_classes > 2 else "binary"
                    
                    return CVTaskAnalysis(
                        task_type='classification',
                        task_subtype=task_subtype,
                        confidence=0.85,
                        num_classes=num_classes,
                        annotation_format='coco_json',
                        image_characteristics={'total_images': len(data['images'])}
                    )
            
            # Check for other JSON formats (custom formats)
            elif isinstance(data, list) and data:
                # List of annotations
                first_item = data[0]
                
                if 'bbox' in first_item or 'bounding_box' in first_item:
                    return CVTaskAnalysis(
                        task_type='object_detection',
                        task_subtype='bbox_detection',
                        confidence=0.7,
                        annotation_format='custom_json'
                    )
                elif 'label' in first_item or 'class' in first_item:
                    return CVTaskAnalysis(
                        task_type='classification',
                        task_subtype='unknown',
                        confidence=0.7,
                        annotation_format='custom_json'
                    )
            
            # Unknown JSON format
            return CVTaskAnalysis(
                task_type='unknown',
                task_subtype='unknown_json',
                confidence=0.3,
                annotation_format='custom_json'
            )
            
        except json.JSONDecodeError:
            raise NeuroLiteException("Invalid JSON annotation file")
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze JSON annotations: {str(e)}")
    
    def _analyze_xml_annotations(self, file_path: Path) -> CVTaskAnalysis:
        """Analyze XML annotation file (Pascal VOC format detection)."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Check for Pascal VOC format
            if root.tag == 'annotation':
                objects = root.findall('object')
                
                if objects:
                    # Check if objects have bounding boxes
                    first_obj = objects[0]
                    bndbox = first_obj.find('bndbox')
                    
                    if bndbox is not None:
                        # Object detection with bounding boxes
                        class_names = set()
                        for obj in objects:
                            name_elem = obj.find('name')
                            if name_elem is not None:
                                class_names.add(name_elem.text)
                        
                        return CVTaskAnalysis(
                            task_type='object_detection',
                            task_subtype='bbox_detection',
                            confidence=0.9,
                            num_classes=len(class_names) if class_names else None,
                            annotation_format='pascal_voc'
                        )
                
                # Classification annotation
                return CVTaskAnalysis(
                    task_type='classification',
                    task_subtype='unknown',
                    confidence=0.7,
                    annotation_format='pascal_voc'
                )
            
            # Unknown XML format
            return CVTaskAnalysis(
                task_type='unknown',
                task_subtype='unknown_xml',
                confidence=0.3,
                annotation_format='custom_xml'
            )
            
        except ET.ParseError:
            raise NeuroLiteException("Invalid XML annotation file")
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze XML annotations: {str(e)}")
    
    def _analyze_text_annotations(self, file_path: Path) -> CVTaskAnalysis:
        """Analyze text annotation file (YOLO format detection)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return CVTaskAnalysis(
                    task_type='unknown',
                    task_subtype='empty_file',
                    confidence=0.3,
                    annotation_format='text'
                )
            
            # Check for YOLO format (class_id x_center y_center width height)
            yolo_pattern = re.compile(r'^\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+')
            
            yolo_lines = 0
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and yolo_pattern.match(line):
                    yolo_lines += 1
            
            if yolo_lines > len(lines) * 0.5:  # More than 50% match YOLO format
                # Extract class IDs to estimate number of classes
                class_ids = set()
                for line in lines:
                    line = line.strip()
                    if line and yolo_pattern.match(line):
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
                
                return CVTaskAnalysis(
                    task_type='object_detection',
                    task_subtype='bbox_detection',
                    confidence=0.85,
                    num_classes=len(class_ids) if class_ids else None,
                    annotation_format='yolo'
                )
            
            # Check for simple classification labels (one label per line)
            if all(len(line.strip().split()) == 1 for line in lines if line.strip()):
                unique_labels = set(line.strip() for line in lines if line.strip())
                
                return CVTaskAnalysis(
                    task_type='classification',
                    task_subtype='multi_class' if len(unique_labels) > 2 else 'binary',
                    confidence=0.7,
                    num_classes=len(unique_labels),
                    annotation_format='text_labels'
                )
            
            # Unknown text format
            return CVTaskAnalysis(
                task_type='unknown',
                task_subtype='unknown_text',
                confidence=0.3,
                annotation_format='custom_text'
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze text annotations: {str(e)}")
    
    def _analyze_csv_annotations(self, file_path: Path) -> CVTaskAnalysis:
        """Analyze CSV annotation file."""
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                return CVTaskAnalysis(
                    task_type='unknown',
                    task_subtype='empty_file',
                    confidence=0.3,
                    annotation_format='csv'
                )
            
            columns = [col.lower() for col in df.columns]
            
            # Check for bounding box columns
            bbox_indicators = ['bbox', 'x1', 'y1', 'x2', 'y2', 'xmin', 'ymin', 'xmax', 'ymax']
            has_bbox = any(indicator in ' '.join(columns) for indicator in bbox_indicators)
            
            if has_bbox:
                # Count unique classes if class column exists
                class_cols = [col for col in columns if 'class' in col or 'label' in col]
                num_classes = None
                
                if class_cols:
                    num_classes = df[class_cols[0]].nunique()
                
                return CVTaskAnalysis(
                    task_type='object_detection',
                    task_subtype='bbox_detection',
                    confidence=0.8,
                    num_classes=num_classes,
                    annotation_format='csv'
                )
            
            # Check for classification labels
            label_cols = [col for col in columns if 'label' in col or 'class' in col or 'category' in col]
            
            if label_cols:
                num_classes = df[label_cols[0]].nunique()
                task_subtype = 'multi_class' if num_classes > 2 else 'binary'
                
                return CVTaskAnalysis(
                    task_type='classification',
                    task_subtype=task_subtype,
                    confidence=0.8,
                    num_classes=num_classes,
                    annotation_format='csv'
                )
            
            # Unknown CSV format
            return CVTaskAnalysis(
                task_type='unknown',
                task_subtype='unknown_csv',
                confidence=0.3,
                annotation_format='csv'
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze CSV annotations: {str(e)}")
    
    def _analyze_image_list(self, image_paths: List[str]) -> CVTaskAnalysis:
        """Analyze list of image paths."""
        try:
            if not image_paths:
                raise NeuroLiteException("Empty image list provided")
            
            # Convert to Path objects and validate
            valid_paths = []
            for path_str in image_paths:
                path = Path(path_str)
                if path.exists() and path.is_file():
                    valid_paths.append(path)
            
            if not valid_paths:
                raise NeuroLiteException("No valid image files found in list")
            
            # Analyze image characteristics
            image_chars = self._analyze_image_characteristics(valid_paths[:10])
            
            # Without additional context, assume classification task
            return CVTaskAnalysis(
                task_type='classification',
                task_subtype='unknown',
                confidence=0.5,
                annotation_format='image_list',
                image_characteristics=image_chars
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze image list: {str(e)}")
    
    def _find_annotation_files(self, dir_path: Path) -> List[Path]:
        """Find annotation files in directory."""
        annotation_extensions = {'.json', '.xml', '.txt', '.csv'}
        annotation_files = set()  # Use set to avoid duplicates
        
        for ext in annotation_extensions:
            # Only search in current directory to avoid duplicates
            annotation_files.update(dir_path.glob(f"*{ext}"))
        
        # Filter out common non-annotation files
        exclude_patterns = ['readme', 'license', 'changelog', 'requirements']
        
        filtered_files = []
        for file_path in annotation_files:
            filename_lower = file_path.name.lower()
            if not any(pattern in filename_lower for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _analyze_image_characteristics(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Analyze characteristics of image files."""
        characteristics = {
            'total_images': len(image_paths),
            'formats': {},
            'resolutions': [],
            'avg_width': 0,
            'avg_height': 0,
            'color_modes': {}
        }
        
        try:
            widths, heights = [], []
            
            for img_path in image_paths:
                try:
                    with Image.open(img_path) as img:
                        # Format
                        fmt = img.format or 'UNKNOWN'
                        characteristics['formats'][fmt] = characteristics['formats'].get(fmt, 0) + 1
                        
                        # Resolution
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                        characteristics['resolutions'].append((width, height))
                        
                        # Color mode
                        mode = img.mode
                        characteristics['color_modes'][mode] = characteristics['color_modes'].get(mode, 0) + 1
                        
                except Exception:
                    # Skip problematic images
                    continue
            
            if widths and heights:
                characteristics['avg_width'] = sum(widths) / len(widths)
                characteristics['avg_height'] = sum(heights) / len(heights)
                characteristics['min_width'] = min(widths)
                characteristics['max_width'] = max(widths)
                characteristics['min_height'] = min(heights)
                characteristics['max_height'] = max(heights)
            
        except Exception:
            # Return basic characteristics if analysis fails
            pass
        
        return characteristics
    
    # NLP Detection Methods (Task 6.2)
    
    def detect_nlp_task(self, data_source: Union[str, Path, pd.DataFrame, List[str]]) -> NLPTaskAnalysis:
        """
        Detect NLP task type and characteristics.
        
        Args:
            data_source: Path to text file, DataFrame with text columns, or list of text samples
            
        Returns:
            NLPTaskAnalysis object with detected NLP task information
            
        Raises:
            NeuroLiteException: If NLP task cannot be determined
        """
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            
            if data_path.is_file():
                return self._analyze_text_file(data_path)
            else:
                raise NeuroLiteException(f"File does not exist: {data_path}")
                
        elif isinstance(data_source, pd.DataFrame):
            return self._analyze_text_dataframe(data_source)
            
        elif isinstance(data_source, list):
            return self._analyze_text_list(data_source)
            
        else:
            raise NeuroLiteException(f"Unsupported data source type: {type(data_source)}")
    
    def _analyze_text_file(self, file_path: Path) -> NLPTaskAnalysis:
        """Analyze text file to detect NLP task type."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.json':
                return self._analyze_json_text_data(file_path)
            elif file_ext == '.jsonl':
                return self._analyze_jsonl_text_data(file_path)
            elif file_ext == '.csv':
                return self._analyze_csv_text_data(file_path)
            elif file_ext in {'.txt', '.md'}:
                return self._analyze_plain_text_file(file_path)
            else:
                # Try to analyze as plain text
                return self._analyze_plain_text_file(file_path)
                
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze text file: {str(e)}")
    
    def _analyze_json_text_data(self, file_path: Path) -> NLPTaskAnalysis:
        """Analyze JSON file for NLP task patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # Single document or structured data
                return self._detect_nlp_task_from_dict(data)
                
            elif isinstance(data, list) and data:
                # List of documents/samples
                return self._detect_nlp_task_from_list(data)
            
            else:
                return NLPTaskAnalysis(
                    task_type='unknown',
                    task_subtype='empty_or_invalid',
                    confidence=0.3
                )
                
        except json.JSONDecodeError:
            raise NeuroLiteException("Invalid JSON text data file")
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze JSON text data: {str(e)}")
    
    def _analyze_jsonl_text_data(self, file_path: Path) -> NLPTaskAnalysis:
        """Analyze JSONL file for NLP task patterns."""
        try:
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 100:  # Limit sample size for analysis
                        break
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            
            if not samples:
                return NLPTaskAnalysis(
                    task_type='unknown',
                    task_subtype='no_valid_samples',
                    confidence=0.3
                )
            
            return self._detect_nlp_task_from_list(samples)
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze JSONL text data: {str(e)}")
    
    def _analyze_csv_text_data(self, file_path: Path) -> NLPTaskAnalysis:
        """Analyze CSV file for NLP task patterns."""
        try:
            df = pd.read_csv(file_path)
            return self._analyze_text_dataframe(df)
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze CSV text data: {str(e)}")
    
    def _analyze_plain_text_file(self, file_path: Path) -> NLPTaskAnalysis:
        """Analyze plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze the text content
            text_chars = self._analyze_text_content([content])
            
            # For plain text files, it's usually document classification or general text
            return NLPTaskAnalysis(
                task_type='classification',
                task_subtype='document_classification',
                confidence=0.6,
                text_characteristics=text_chars
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze plain text file: {str(e)}")
    
    def _analyze_text_dataframe(self, df: pd.DataFrame) -> NLPTaskAnalysis:
        """Analyze DataFrame for NLP task patterns."""
        try:
            if df.empty:
                return NLPTaskAnalysis(
                    task_type='unknown',
                    task_subtype='empty_dataframe',
                    confidence=0.3
                )
            
            # Find text columns
            text_columns = self._find_text_columns(df)
            
            if not text_columns:
                return NLPTaskAnalysis(
                    task_type='unknown',
                    task_subtype='no_text_columns',
                    confidence=0.3
                )
            
            # Analyze column names and structure for task detection
            column_names = [col.lower() for col in df.columns]
            
            # Check for sentiment analysis patterns
            if any(keyword in ' '.join(column_names) for keyword in ['sentiment', 'emotion', 'polarity', 'rating']):
                return self._analyze_sentiment_task(df, text_columns)
            
            # Check for NER/sequence labeling patterns
            if any(keyword in ' '.join(column_names) for keyword in ['entit', 'tag', 'ner', 'pos', 'token']):
                return self._analyze_sequence_task(df, text_columns)
            
            # Check for Q&A patterns
            if any(keyword in ' '.join(column_names) for keyword in ['question', 'answer', 'query', 'response']):
                return self._analyze_qa_task(df, text_columns)
            
            # Check for conversation patterns
            if any(keyword in ' '.join(column_names) for keyword in ['conversation', 'dialogue', 'chat', 'message']):
                return self._analyze_conversation_task(df, text_columns)
            
            # Default to classification
            return self._analyze_classification_task(df, text_columns)
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze text DataFrame: {str(e)}")
    
    def _analyze_text_list(self, text_list: List[str]) -> NLPTaskAnalysis:
        """Analyze list of text samples."""
        try:
            if not text_list:
                return NLPTaskAnalysis(
                    task_type='unknown',
                    task_subtype='empty_list',
                    confidence=0.3
                )
            
            # Analyze text characteristics
            text_chars = self._analyze_text_content(text_list)
            
            # Without additional context, assume classification
            return NLPTaskAnalysis(
                task_type='classification',
                task_subtype='text_classification',
                confidence=0.5,
                text_characteristics=text_chars
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze text list: {str(e)}")
    
    def _detect_nlp_task_from_dict(self, data: Dict[str, Any]) -> NLPTaskAnalysis:
        """Detect NLP task from dictionary structure."""
        keys = [k.lower() for k in data.keys()]
        
        # Check for Q&A patterns
        if any(keyword in keys for keyword in ['question', 'answer', 'query', 'response']):
            return NLPTaskAnalysis(
                task_type='qa',
                task_subtype='question_answering',
                confidence=0.8,
                text_characteristics=self._extract_text_chars_from_dict(data)
            )
        
        # Check for conversation patterns
        if any(keyword in keys for keyword in ['conversation', 'dialogue', 'messages', 'turns']):
            return NLPTaskAnalysis(
                task_type='conversation',
                task_subtype='dialogue',
                confidence=0.8,
                text_characteristics=self._extract_text_chars_from_dict(data)
            )
        
        # Check for sentiment/classification patterns
        if any(keyword in keys for keyword in ['text', 'content', 'document']) and \
           any(keyword in keys for keyword in ['label', 'class', 'sentiment', 'category']):
            return NLPTaskAnalysis(
                task_type='classification',
                task_subtype='text_classification',
                confidence=0.7,
                text_characteristics=self._extract_text_chars_from_dict(data)
            )
        
        # Default
        return NLPTaskAnalysis(
            task_type='unknown',
            task_subtype='unknown_structure',
            confidence=0.4,
            text_characteristics=self._extract_text_chars_from_dict(data)
        )
    
    def _detect_nlp_task_from_list(self, data: List[Dict[str, Any]]) -> NLPTaskAnalysis:
        """Detect NLP task from list of dictionaries."""
        if not data:
            return NLPTaskAnalysis(
                task_type='unknown',
                task_subtype='empty_list',
                confidence=0.3
            )
        
        # Analyze first few samples to understand structure
        sample_keys = set()
        for item in data[:10]:
            if isinstance(item, dict):
                sample_keys.update(k.lower() for k in item.keys())
        
        # Check for Q&A patterns
        if any(keyword in sample_keys for keyword in ['question', 'answer', 'query', 'response']):
            return NLPTaskAnalysis(
                task_type='qa',
                task_subtype='question_answering',
                confidence=0.8,
                text_characteristics={'total_samples': len(data)}
            )
        
        # Check for NER/sequence patterns
        if any(keyword in sample_keys for keyword in ['tokens', 'entities', 'tags', 'labels']) and \
           any(keyword in sample_keys for keyword in ['text', 'sentence', 'words']):
            return NLPTaskAnalysis(
                task_type='ner',
                task_subtype='named_entity_recognition',
                confidence=0.8,
                text_characteristics={'total_samples': len(data)},
                sequence_info={'has_token_labels': True}
            )
        
        # Check for sentiment/classification patterns
        if any(keyword in sample_keys for keyword in ['text', 'content', 'document']) and \
           any(keyword in sample_keys for keyword in ['label', 'class', 'sentiment', 'category']):
            
            # Determine if it's sentiment specifically
            if any(keyword in sample_keys for keyword in ['sentiment', 'emotion', 'polarity']):
                return NLPTaskAnalysis(
                    task_type='sentiment',
                    task_subtype='sentiment_analysis',
                    confidence=0.8,
                    text_characteristics={'total_samples': len(data)}
                )
            else:
                return NLPTaskAnalysis(
                    task_type='classification',
                    task_subtype='text_classification',
                    confidence=0.7,
                    text_characteristics={'total_samples': len(data)}
                )
        
        # Check for conversation patterns
        if any(keyword in sample_keys for keyword in ['conversation', 'dialogue', 'messages', 'turns']):
            return NLPTaskAnalysis(
                task_type='conversation',
                task_subtype='dialogue',
                confidence=0.8,
                text_characteristics={'total_samples': len(data)}
            )
        
        # Default to classification
        return NLPTaskAnalysis(
            task_type='classification',
            task_subtype='unknown_classification',
            confidence=0.5,
            text_characteristics={'total_samples': len(data)}
        )
    
    def _find_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that likely contain text data."""
        text_columns = []
        
        for col in df.columns:
            # Check if column contains string data
            if df[col].dtype == 'object':
                # Sample a few values to check if they're text
                sample_values = df[col].dropna().head(10)
                
                if len(sample_values) > 0:
                    # Check if values are strings and have reasonable length for text
                    text_like = 0
                    for val in sample_values:
                        if isinstance(val, str) and len(val) > 5:  # Arbitrary threshold
                            text_like += 1
                    
                    if text_like > len(sample_values) * 0.5:  # More than 50% are text-like
                        text_columns.append(col)
        
        return text_columns
    
    def _analyze_sentiment_task(self, df: pd.DataFrame, text_columns: List[str]) -> NLPTaskAnalysis:
        """Analyze DataFrame for sentiment analysis task."""
        sentiment_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['sentiment', 'emotion', 'polarity', 'rating'])]
        
        text_chars = self._analyze_text_content(df[text_columns[0]].dropna().head(100).tolist())
        
        return NLPTaskAnalysis(
            task_type='sentiment',
            task_subtype='sentiment_analysis',
            confidence=0.85,
            text_characteristics=text_chars
        )
    
    def _analyze_sequence_task(self, df: pd.DataFrame, text_columns: List[str]) -> NLPTaskAnalysis:
        """Analyze DataFrame for sequence labeling task."""
        text_chars = self._analyze_text_content(df[text_columns[0]].dropna().head(100).tolist())
        
        # Check if it looks like token-level labeling
        has_tokens = any('token' in col.lower() for col in df.columns)
        
        if has_tokens:
            task_subtype = 'token_classification'
        else:
            task_subtype = 'sequence_labeling'
        
        return NLPTaskAnalysis(
            task_type='ner',
            task_subtype=task_subtype,
            confidence=0.8,
            text_characteristics=text_chars,
            sequence_info={'has_token_labels': has_tokens}
        )
    
    def _analyze_qa_task(self, df: pd.DataFrame, text_columns: List[str]) -> NLPTaskAnalysis:
        """Analyze DataFrame for Q&A task."""
        text_chars = self._analyze_text_content(df[text_columns[0]].dropna().head(100).tolist())
        
        return NLPTaskAnalysis(
            task_type='qa',
            task_subtype='question_answering',
            confidence=0.8,
            text_characteristics=text_chars
        )
    
    def _analyze_conversation_task(self, df: pd.DataFrame, text_columns: List[str]) -> NLPTaskAnalysis:
        """Analyze DataFrame for conversation task."""
        text_chars = self._analyze_text_content(df[text_columns[0]].dropna().head(100).tolist())
        
        return NLPTaskAnalysis(
            task_type='conversation',
            task_subtype='dialogue',
            confidence=0.8,
            text_characteristics=text_chars
        )
    
    def _analyze_classification_task(self, df: pd.DataFrame, text_columns: List[str]) -> NLPTaskAnalysis:
        """Analyze DataFrame for classification task."""
        text_chars = self._analyze_text_content(df[text_columns[0]].dropna().head(100).tolist())
        
        # Look for label columns
        label_cols = [col for col in df.columns 
                     if any(keyword in col.lower() for keyword in ['label', 'class', 'category'])]
        
        if label_cols:
            num_classes = df[label_cols[0]].nunique()
            if num_classes <= 3:
                task_subtype = 'binary_or_multiclass'
            else:
                task_subtype = 'multiclass_classification'
        else:
            task_subtype = 'document_classification'
        
        return NLPTaskAnalysis(
            task_type='classification',
            task_subtype=task_subtype,
            confidence=0.7,
            text_characteristics=text_chars
        )
    
    def _analyze_text_content(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of text content."""
        if not texts:
            return {}
        
        characteristics = {
            'total_texts': len(texts),
            'avg_length': 0,
            'max_length': 0,
            'min_length': float('inf'),
            'has_punctuation': False,
            'has_numbers': False,
            'has_special_chars': False,
            'estimated_language': 'unknown'
        }
        
        try:
            lengths = []
            has_punct = False
            has_nums = False
            has_special = False
            
            for text in texts[:50]:  # Analyze first 50 texts
                if isinstance(text, str):
                    length = len(text)
                    lengths.append(length)
                    
                    # Check for various characteristics
                    if not has_punct and any(c in text for c in '.,!?;:'):
                        has_punct = True
                    if not has_nums and any(c.isdigit() for c in text):
                        has_nums = True
                    if not has_special and any(c in text for c in '@#$%^&*()[]{}'):
                        has_special = True
            
            if lengths:
                characteristics.update({
                    'avg_length': sum(lengths) / len(lengths),
                    'max_length': max(lengths),
                    'min_length': min(lengths),
                    'has_punctuation': has_punct,
                    'has_numbers': has_nums,
                    'has_special_chars': has_special
                })
            
            # Simple language detection heuristic
            sample_text = ' '.join(texts[:10])
            if len(sample_text) > 100:
                # Very basic language detection based on common words
                english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
                english_count = sum(1 for word in english_indicators if word in sample_text.lower())
                
                if english_count >= 3:
                    characteristics['estimated_language'] = 'english'
                else:
                    characteristics['estimated_language'] = 'other'
            
        except Exception:
            # Return basic characteristics if analysis fails
            pass
        
        return characteristics
    
    def _extract_text_chars_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text characteristics from dictionary data."""
        text_values = []
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10:  # Likely text content
                text_values.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 10:
                        text_values.append(item)
        
        if text_values:
            return self._analyze_text_content(text_values)
        else:
            return {'total_texts': 0}
    
    # Time Series Detection Methods (Task 6.3)
    
    def detect_timeseries_characteristics(self, data_source: Union[str, Path, pd.DataFrame, np.ndarray]) -> TimeSeriesAnalysis:
        """
        Detect time series characteristics and patterns.
        
        Args:
            data_source: Path to time series file, DataFrame, or array
            
        Returns:
            TimeSeriesAnalysis object with detected time series characteristics
            
        Raises:
            NeuroLiteException: If time series analysis cannot be performed
        """
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            
            if data_path.is_file():
                return self._analyze_timeseries_file(data_path)
            else:
                raise NeuroLiteException(f"File does not exist: {data_path}")
                
        elif isinstance(data_source, pd.DataFrame):
            return self._analyze_timeseries_dataframe(data_source)
            
        elif isinstance(data_source, np.ndarray):
            return self._analyze_timeseries_array(data_source)
            
        else:
            raise NeuroLiteException(f"Unsupported data source type: {type(data_source)}")
    
    def _analyze_timeseries_file(self, file_path: Path) -> TimeSeriesAnalysis:
        """Analyze time series file."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                return self._analyze_timeseries_dataframe(df)
            elif file_ext in {'.json', '.jsonl'}:
                # Try to load as DataFrame
                if file_ext == '.json':
                    df = pd.read_json(file_path)
                else:
                    df = pd.read_json(file_path, lines=True)
                return self._analyze_timeseries_dataframe(df)
            elif file_ext in {'.xlsx', '.xls'}:
                df = pd.read_excel(file_path)
                return self._analyze_timeseries_dataframe(df)
            else:
                raise UnsupportedFormatError(f"Unsupported time series file format: {file_ext}")
                
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze time series file: {str(e)}")
    
    def _analyze_timeseries_dataframe(self, df: pd.DataFrame) -> TimeSeriesAnalysis:
        """Analyze DataFrame for time series characteristics."""
        try:
            if df.empty:
                raise NeuroLiteException("Empty DataFrame provided")
            
            # Find datetime columns
            datetime_cols = self._find_datetime_columns(df)
            
            if not datetime_cols:
                raise NeuroLiteException("No datetime columns found in DataFrame")
            
            # Use the first datetime column as the time index
            time_col = datetime_cols[0]
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            
            # Sort by time column
            df_sorted = df.sort_values(time_col)
            
            # Determine if univariate or multivariate
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in numeric_cols:
                numeric_cols.remove(time_col)
            
            series_type = 'multivariate' if len(numeric_cols) > 1 else 'univariate'
            
            # Analyze frequency
            frequency = self._detect_frequency(df_sorted[time_col])
            
            # Analyze each numeric column for patterns
            trend_results = []
            seasonality_results = []
            stationarity_results = []
            
            for col in numeric_cols[:5]:  # Analyze up to 5 columns to avoid performance issues
                series = df_sorted[col].dropna()
                
                if len(series) < 10:  # Need minimum data points
                    continue
                
                # Trend analysis
                trend_strength = self._detect_trend(series)
                trend_results.append(trend_strength)
                
                # Seasonality analysis
                seasonal_strength = self._detect_seasonality(series, frequency)
                seasonality_results.append(seasonal_strength)
                
                # Stationarity test
                is_stationary = self._test_stationarity(series)
                stationarity_results.append(is_stationary)
            
            # Aggregate results
            has_trend = any(trend > 0.3 for trend in trend_results) if trend_results else False
            has_seasonality = any(seasonal > 0.3 for seasonal in seasonality_results) if seasonality_results else False
            is_stationary = all(stationarity_results) if stationarity_results else True
            
            avg_trend = np.mean(trend_results) if trend_results else 0.0
            avg_seasonal = np.mean(seasonality_results) if seasonality_results else 0.0
            
            # Determine seasonality period
            seasonality_period = self._estimate_seasonality_period(frequency) if has_seasonality else None
            
            # Recommend task type
            recommended_task = self._recommend_timeseries_task(df_sorted, numeric_cols)
            
            return TimeSeriesAnalysis(
                series_type=series_type,
                has_trend=has_trend,
                has_seasonality=has_seasonality,
                is_stationary=is_stationary,
                frequency=frequency,
                seasonality_period=seasonality_period,
                trend_strength=float(avg_trend),
                seasonal_strength=float(avg_seasonal),
                recommended_task=recommended_task
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze time series DataFrame: {str(e)}")
    
    def _analyze_timeseries_array(self, arr: np.ndarray) -> TimeSeriesAnalysis:
        """Analyze numpy array for time series characteristics."""
        try:
            if arr.size == 0:
                raise NeuroLiteException("Empty array provided")
            
            # Assume 1D array is univariate, 2D array is multivariate
            if arr.ndim == 1:
                series_type = 'univariate'
                data = arr
            elif arr.ndim == 2:
                series_type = 'multivariate' if arr.shape[1] > 1 else 'univariate'
                data = arr[:, 0]  # Analyze first column for patterns
            else:
                raise NeuroLiteException("Array must be 1D or 2D for time series analysis")
            
            # Since we don't have datetime information, assume regular intervals
            frequency = 'unknown'
            
            # Analyze patterns on the data
            trend_strength = self._detect_trend(pd.Series(data))
            seasonal_strength = self._detect_seasonality(pd.Series(data), frequency)
            is_stationary = self._test_stationarity(pd.Series(data))
            
            has_trend = trend_strength > 0.3
            has_seasonality = seasonal_strength > 0.3
            
            # Without datetime context, default to forecasting
            recommended_task = 'forecasting'
            
            return TimeSeriesAnalysis(
                series_type=series_type,
                has_trend=has_trend,
                has_seasonality=has_seasonality,
                is_stationary=is_stationary,
                frequency=frequency,
                seasonality_period=None,
                trend_strength=float(trend_strength),
                seasonal_strength=float(seasonal_strength),
                recommended_task=recommended_task
            )
            
        except Exception as e:
            raise NeuroLiteException(f"Failed to analyze time series array: {str(e)}")
    
    def _find_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that contain datetime information."""
        datetime_cols = []
        
        # Check for explicit datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
        
        # Check for columns that might be dates based on name
        if not datetime_cols:
            potential_date_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'datetime']):
                    potential_date_cols.append(col)
            
            # Try to parse potential date columns
            for col in potential_date_cols:
                try:
                    sample_values = df[col].dropna().head(10)
                    pd.to_datetime(sample_values)
                    datetime_cols.append(col)
                except:
                    continue
        
        # If still no datetime columns, check if index is datetime
        if not datetime_cols and isinstance(df.index, pd.DatetimeIndex):
            # Create a temporary column from index
            df_temp = df.copy()
            df_temp['_datetime_index'] = df_temp.index
            datetime_cols.append('_datetime_index')
        
        return datetime_cols
    
    def _detect_frequency(self, datetime_series: pd.Series) -> Optional[str]:
        """Detect the frequency of time series data."""
        try:
            if len(datetime_series) < 3:
                return 'unknown'
            
            # Calculate time differences
            time_diffs = datetime_series.diff().dropna()
            
            if time_diffs.empty:
                return 'unknown'
            
            # Convert to Series if it's an Index to use mode()
            if hasattr(time_diffs, 'to_series'):
                time_diffs = time_diffs.to_series()
            elif not isinstance(time_diffs, pd.Series):
                time_diffs = pd.Series(time_diffs)
            
            # Find the most common time difference
            mode_diff = time_diffs.mode()
            
            if mode_diff.empty:
                return 'unknown'
            
            most_common_diff = mode_diff.iloc[0]
            
            # Convert to frequency string based on the most common difference
            total_seconds = most_common_diff.total_seconds()
            
            if total_seconds <= 1:
                return 'S'  # Second
            elif total_seconds <= 60:
                return 'T'  # Minute
            elif total_seconds <= 3600:
                return 'H'  # Hour
            elif total_seconds <= 86400:
                return 'D'  # Day
            elif total_seconds <= 604800:
                return 'W'  # Week
            elif total_seconds <= 2678400:  # ~31 days
                return 'M'  # Month
            elif total_seconds <= 31622400:  # ~366 days
                return 'Y'  # Year
            else:
                return 'irregular'
                
        except Exception:
            return 'unknown'
    
    def _detect_trend(self, series: pd.Series) -> float:
        """Detect trend strength in time series."""
        try:
            if len(series) < 10:
                return 0.0
            
            # Simple linear trend detection using correlation with time index
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 3:
                return 0.0
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            
            # Return absolute correlation as trend strength
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _detect_seasonality(self, series: pd.Series, frequency: Optional[str]) -> float:
        """Detect seasonality strength in time series."""
        try:
            if len(series) < 20:  # Need sufficient data for seasonality
                return 0.0
            
            # Estimate seasonal period based on frequency
            if frequency == 'H':
                periods = [24, 168]  # Daily, weekly patterns
            elif frequency == 'D':
                periods = [7, 30, 365]  # Weekly, monthly, yearly patterns
            elif frequency == 'W':
                periods = [4, 52]  # Monthly, yearly patterns
            elif frequency == 'M':
                periods = [12]  # Yearly pattern
            else:
                # Try common periods
                periods = [7, 12, 24, 30]
            
            max_seasonality = 0.0
            
            for period in periods:
                if len(series) >= 2 * period:
                    seasonality = self._calculate_seasonal_strength(series, period)
                    max_seasonality = max(max_seasonality, seasonality)
            
            return max_seasonality
            
        except Exception:
            return 0.0
    
    def _calculate_seasonal_strength(self, series: pd.Series, period: int) -> float:
        """Calculate seasonal strength for a given period."""
        try:
            # Simple autocorrelation-based seasonality detection
            if len(series) < 2 * period:
                return 0.0
            
            # Calculate autocorrelation at the seasonal lag
            autocorr = series.autocorr(lag=period)
            
            return abs(autocorr) if not np.isnan(autocorr) else 0.0
            
        except Exception:
            return 0.0
    
    def _test_stationarity(self, series: pd.Series) -> bool:
        """Test if time series is stationary using simple heuristics."""
        try:
            if len(series) < 10:
                return True  # Assume stationary for short series
            
            # Simple stationarity test based on rolling statistics
            window_size = min(len(series) // 4, 10)
            
            if window_size < 2:
                return True
            
            # Calculate rolling mean and std
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            # Check if rolling statistics are relatively stable
            mean_stability = rolling_mean.std() / series.std() if series.std() > 0 else 0
            std_stability = rolling_std.std() / series.std() if series.std() > 0 else 0
            
            # Consider stationary if rolling statistics are stable (more lenient thresholds)
            return bool(mean_stability < 0.3 and std_stability < 0.3)
            
        except Exception:
            return True  # Default to stationary if test fails
    
    def _estimate_seasonality_period(self, frequency: Optional[str]) -> Optional[int]:
        """Estimate seasonality period based on frequency."""
        if frequency == 'H':
            return 24  # Daily seasonality
        elif frequency == 'D':
            return 7   # Weekly seasonality
        elif frequency == 'W':
            return 52  # Yearly seasonality
        elif frequency == 'M':
            return 12  # Yearly seasonality
        else:
            return None
    
    def _recommend_timeseries_task(self, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        """Recommend appropriate task type for time series data."""
        try:
            # Check if there are categorical target columns (classification)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove text-like columns (keep only potential labels)
            label_cols = []
            for col in categorical_cols:
                unique_values = df[col].nunique()
                if 2 <= unique_values <= 20:  # Reasonable number of classes
                    label_cols.append(col)
            
            if label_cols:
                return 'classification'
            
            # Check for anomaly detection patterns (outliers, rare events)
            for col in numeric_cols[:3]:  # Check first few numeric columns
                series = df[col].dropna()
                if len(series) > 10:
                    # Simple outlier detection
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
                    
                    if len(outliers) > len(series) * 0.05:  # More than 5% outliers
                        return 'anomaly_detection'
            
            # Default to forecasting
            return 'forecasting'
            
        except Exception:
            return 'forecasting'