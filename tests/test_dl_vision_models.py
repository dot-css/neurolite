"""
Unit tests for deep learning computer vision models.

Tests CNN architectures, object detection, and segmentation models
with synthetic data to ensure proper functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from PIL import Image

from neurolite.models.dl.vision import (
    PyTorchVisionModel, ResNetModel, EfficientNetModel, 
    VisionTransformerModel, ObjectDetectionModel, SegmentationModel,
    register_vision_models
)
from neurolite.models.base import TaskType, PredictionResult
from neurolite.models.registry import get_model_registry
from neurolite.data.detector import DataType
from neurolite.core.exceptions import DependencyError


class TestPyTorchVisionModel(unittest.TestCase):
    """Test base PyTorch vision model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pytorch_import_error(self):
        """Test handling of missing PyTorch dependencies."""
        with patch.dict('sys.modules', {'torch': None, 'torchvision': None}):
            with self.assertRaises(DependencyError) as context:
                ResNetModel()
            
            self.assertIn("Missing PyTorch dependencies", str(context.exception))
            self.assertIn("pip install torch torchvision", str(context.exception))
    
    def test_capabilities(self):
        """Test model capabilities."""
        try:
            model = ResNetModel(num_classes=10)
            capabilities = model.capabilities
            
            self.assertIn(TaskType.IMAGE_CLASSIFICATION, capabilities.supported_tasks)
            self.assertIn(DataType.IMAGE, capabilities.supported_data_types)
            self.assertEqual(capabilities.framework, "pytorch")
            self.assertTrue(capabilities.supports_probability_prediction)
            self.assertFalse(capabilities.supports_feature_importance)
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_device_selection(self):
        """Test automatic device selection."""
        try:
            # Test automatic device selection
            model = ResNetModel(device=None)
            self.assertIn(model.device, ['cpu', 'cuda'])
            
            # Test explicit device selection
            model_cpu = ResNetModel(device='cpu')
            self.assertEqual(model_cpu.device, 'cpu')
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def create_synthetic_image_data(self, num_samples=10, num_classes=3):
        """Create synthetic image data for testing."""
        # Create synthetic RGB images
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create random RGB image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
            labels.append(i % num_classes)
        
        return images, np.array(labels)
    
    def test_data_preparation(self):
        """Test data preparation and transforms."""
        try:
            model = ResNetModel(num_classes=3)
            
            # Create synthetic data
            images, labels = self.create_synthetic_image_data(5, 3)
            
            # Test data preparation
            X_tensor, y_tensor = model._prepare_data(images, labels, training=True)
            
            # Check tensor shapes
            self.assertEqual(X_tensor.shape[0], 5)  # batch size
            self.assertEqual(X_tensor.shape[1], 3)  # channels
            self.assertEqual(X_tensor.shape[2], 224)  # height
            self.assertEqual(X_tensor.shape[3], 224)  # width
            self.assertEqual(y_tensor.shape[0], 5)  # batch size
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        try:
            model = ResNetModel(num_classes=3)
            
            # Create synthetic training data
            X_train, y_train = self.create_synthetic_image_data(20, 3)
            
            # Train model (minimal epochs for testing)
            model.fit(X_train, y_train, epochs=2, batch_size=4)
            
            self.assertTrue(model.is_trained)
            self.assertIsNotNone(model.model)
            self.assertIn('loss', model.training_history)
            self.assertIn('accuracy', model.training_history)
            
            # Test prediction
            X_test, _ = self.create_synthetic_image_data(5, 3)
            result = model.predict(X_test)
            
            self.assertIsInstance(result, PredictionResult)
            self.assertEqual(len(result.predictions), 5)
            self.assertEqual(result.probabilities.shape, (5, 3))
            
            # Check predictions are valid class indices
            self.assertTrue(all(0 <= pred < 3 for pred in result.predictions))
            
            # Check probabilities sum to 1
            prob_sums = np.sum(result.probabilities, axis=1)
            np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        try:
            # Create and train model
            model = ResNetModel(num_classes=3)
            X_train, y_train = self.create_synthetic_image_data(10, 3)
            model.fit(X_train, y_train, epochs=1, batch_size=4)
            
            # Save model
            model_path = os.path.join(self.temp_dir, "test_model.pth")
            model.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            loaded_model = ResNetModel(num_classes=3)
            loaded_model.load(model_path)
            
            self.assertTrue(loaded_model.is_trained)
            self.assertEqual(loaded_model.num_classes, 3)
            self.assertEqual(loaded_model.model_name, model.model_name)
            
            # Test prediction with loaded model
            X_test, _ = self.create_synthetic_image_data(3, 3)
            result = loaded_model.predict(X_test)
            self.assertEqual(len(result.predictions), 3)
            
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestResNetModel(unittest.TestCase):
    """Test ResNet model implementations."""
    
    def test_resnet_variants(self):
        """Test different ResNet variants."""
        variants = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        
        for variant in variants:
            try:
                model = ResNetModel(variant=variant, num_classes=10)
                self.assertEqual(model.variant, variant)
                self.assertEqual(model.model_name, f"ResNet-{variant}")
                self.assertEqual(model.num_classes, 10)
                
            except DependencyError:
                self.skipTest("PyTorch not available")
    
    def test_invalid_variant(self):
        """Test handling of invalid ResNet variant."""
        try:
            model = ResNetModel(variant='invalid_variant', num_classes=10)
            with self.assertRaises(ValueError):
                model._create_model()
                
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_custom_num_classes(self):
        """Test ResNet with custom number of classes."""
        try:
            model = ResNetModel(variant='resnet18', num_classes=5)
            pytorch_model = model._create_model()
            
            # Check that final layer has correct output size
            self.assertEqual(pytorch_model.fc.out_features, 5)
            
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestEfficientNetModel(unittest.TestCase):
    """Test EfficientNet model implementations."""
    
    def test_efficientnet_variants(self):
        """Test different EfficientNet variants."""
        variants = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        
        for variant in variants:
            try:
                model = EfficientNetModel(variant=variant, num_classes=10)
                self.assertEqual(model.variant, variant)
                self.assertEqual(model.model_name, f"EfficientNet-{variant}")
                self.assertEqual(model.num_classes, 10)
                
            except DependencyError:
                self.skipTest("PyTorch not available")
    
    def test_fallback_to_simple_cnn(self):
        """Test fallback to simple CNN when EfficientNet is not available."""
        try:
            model = EfficientNetModel(variant='b0', num_classes=10)
            
            # Mock torchvision to not have EfficientNet
            with patch.object(model.torchvision.models, 'efficientnet_b0', side_effect=AttributeError):
                with patch('warnings.warn') as mock_warn:
                    pytorch_model = model._create_model()
                    
                    # Check that warning was issued
                    mock_warn.assert_called()
                    self.assertIn("not available", mock_warn.call_args[0][0])
                    
                    # Check that model was created
                    self.assertIsNotNone(pytorch_model)
                    
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestVisionTransformerModel(unittest.TestCase):
    """Test Vision Transformer model implementations."""
    
    def test_vit_variants(self):
        """Test different ViT variants."""
        variants = ['base', 'large']
        
        for variant in variants:
            try:
                model = VisionTransformerModel(variant=variant, num_classes=10)
                self.assertEqual(model.variant, variant)
                self.assertEqual(model.model_name, f"ViT-{variant}")
                self.assertEqual(model.num_classes, 10)
                
            except DependencyError:
                self.skipTest("PyTorch not available")
    
    def test_patch_size_parameter(self):
        """Test patch size parameter."""
        try:
            model = VisionTransformerModel(variant='base', patch_size=32, num_classes=10)
            self.assertEqual(model.patch_size, 32)
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_fallback_to_simple_vit(self):
        """Test fallback to simple ViT when torchvision ViT is not available."""
        try:
            model = VisionTransformerModel(variant='tiny', num_classes=10)
            
            with patch('warnings.warn') as mock_warn:
                pytorch_model = model._create_model()
                
                # Check that model was created (should use simple ViT for 'tiny')
                self.assertIsNotNone(pytorch_model)
                
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestObjectDetectionModel(unittest.TestCase):
    """Test object detection model implementations."""
    
    def test_object_detection_capabilities(self):
        """Test object detection model capabilities."""
        try:
            model = ObjectDetectionModel(architecture='fasterrcnn', num_classes=10)
            capabilities = model.capabilities
            
            self.assertIn(TaskType.OBJECT_DETECTION, capabilities.supported_tasks)
            self.assertIn(DataType.IMAGE, capabilities.supported_data_types)
            self.assertEqual(capabilities.framework, "pytorch")
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_detection_architectures(self):
        """Test different detection architectures."""
        architectures = ['fasterrcnn', 'ssd', 'yolo']
        
        for arch in architectures:
            try:
                model = ObjectDetectionModel(architecture=arch, num_classes=10)
                self.assertEqual(model.architecture, arch)
                self.assertEqual(model.model_name, f"ObjectDetection-{arch}")
                
            except DependencyError:
                self.skipTest("PyTorch not available")
    
    def test_fallback_detector(self):
        """Test fallback to simple detector."""
        try:
            model = ObjectDetectionModel(architecture='unsupported', num_classes=10)
            
            with patch('warnings.warn') as mock_warn:
                pytorch_model = model._create_model()
                
                # Check that warning was issued (may be multiple warnings)
                mock_warn.assert_called()
                
                # Check if our specific warning was called
                warning_messages = [call[0][0] for call in mock_warn.call_args_list]
                found_our_warning = any("not fully implemented" in msg for msg in warning_messages)
                self.assertTrue(found_our_warning, f"Expected warning not found in: {warning_messages}")
                
                # Check that model was created
                self.assertIsNotNone(pytorch_model)
                
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestSegmentationModel(unittest.TestCase):
    """Test image segmentation model implementations."""
    
    def test_segmentation_capabilities(self):
        """Test segmentation model capabilities."""
        try:
            model = SegmentationModel(architecture='deeplabv3', num_classes=10)
            capabilities = model.capabilities
            
            self.assertIn(TaskType.SEMANTIC_SEGMENTATION, capabilities.supported_tasks)
            self.assertIn(TaskType.INSTANCE_SEGMENTATION, capabilities.supported_tasks)
            self.assertIn(DataType.IMAGE, capabilities.supported_data_types)
            self.assertEqual(capabilities.framework, "pytorch")
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_segmentation_architectures(self):
        """Test different segmentation architectures."""
        architectures = ['deeplabv3', 'fcn', 'unet']
        
        for arch in architectures:
            try:
                model = SegmentationModel(architecture=arch, num_classes=10)
                self.assertEqual(model.architecture, arch)
                self.assertEqual(model.model_name, f"Segmentation-{arch}")
                
            except DependencyError:
                self.skipTest("PyTorch not available")
    
    def test_fallback_segmenter(self):
        """Test fallback to simple segmenter."""
        try:
            model = SegmentationModel(architecture='unsupported', num_classes=10)
            
            with patch('warnings.warn') as mock_warn:
                pytorch_model = model._create_model()
                
                # Check that warning was issued
                mock_warn.assert_called()
                self.assertIn("not fully implemented", mock_warn.call_args[0][0])
                
                # Check that model was created
                self.assertIsNotNone(pytorch_model)
                
        except DependencyError:
            self.skipTest("PyTorch not available")


class TestVisionModelRegistration(unittest.TestCase):
    """Test registration of vision models in the global registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear registry before each test
        registry = get_model_registry()
        registry.clear()
    
    def test_register_vision_models(self):
        """Test registration of all vision models."""
        register_vision_models()
        
        registry = get_model_registry()
        
        # Test ResNet models
        resnet_models = [f"resnet_{i}" for i in ['18', '34', '50', '101', '152']]
        for model_name in resnet_models:
            self.assertIn(model_name, registry.list_models())
            
            # Test model info
            info = registry.get_model_info(model_name)
            self.assertEqual(info.capabilities.framework, "pytorch")
            self.assertIn("cnn", info.tags)
            self.assertIn("image_classification", info.tags)
        
        # Test EfficientNet models
        efficientnet_models = [f"efficientnet_{v}" for v in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']]
        for model_name in efficientnet_models:
            self.assertIn(model_name, registry.list_models())
            
            info = registry.get_model_info(model_name)
            self.assertEqual(info.capabilities.framework, "pytorch")
            self.assertIn("efficient", info.tags)
        
        # Test Vision Transformer models
        vit_models = ["vit_base", "vit_large"]
        for model_name in vit_models:
            self.assertIn(model_name, registry.list_models())
            
            info = registry.get_model_info(model_name)
            self.assertEqual(info.capabilities.framework, "pytorch")
            self.assertIn("transformer", info.tags)
        
        # Test object detection models
        detection_models = ["object_detection_fasterrcnn", "object_detection_ssd", "object_detection_yolo"]
        for model_name in detection_models:
            self.assertIn(model_name, registry.list_models())
            
            info = registry.get_model_info(model_name)
            self.assertEqual(info.capabilities.framework, "pytorch")
            self.assertIn("object_detection", info.tags)
        
        # Test segmentation models
        segmentation_models = ["segmentation_deeplabv3", "segmentation_fcn", "segmentation_unet"]
        for model_name in segmentation_models:
            self.assertIn(model_name, registry.list_models())
            
            info = registry.get_model_info(model_name)
            self.assertEqual(info.capabilities.framework, "pytorch")
            self.assertIn("segmentation", info.tags)
    
    def test_model_creation_from_registry(self):
        """Test creating models from the registry."""
        register_vision_models()
        
        registry = get_model_registry()
        
        try:
            # Test creating ResNet model
            resnet_model = registry.get_model("resnet_50", num_classes=10)
            self.assertIsInstance(resnet_model, ResNetModel)
            self.assertEqual(resnet_model.num_classes, 10)
            
            # Test creating EfficientNet model
            efficientnet_model = registry.get_model("efficientnet_b0", num_classes=5)
            self.assertIsInstance(efficientnet_model, EfficientNetModel)
            self.assertEqual(efficientnet_model.num_classes, 5)
            
            # Test creating ViT model
            vit_model = registry.get_model("vit_base", num_classes=20)
            self.assertIsInstance(vit_model, VisionTransformerModel)
            self.assertEqual(vit_model.num_classes, 20)
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_model_filtering_by_task(self):
        """Test filtering models by task type."""
        register_vision_models()
        
        registry = get_model_registry()
        
        # Test filtering by image classification
        classification_models = registry.list_models(task_type=TaskType.IMAGE_CLASSIFICATION)
        self.assertGreater(len(classification_models), 0)
        
        # All returned models should support image classification
        for model_name in classification_models:
            info = registry.get_model_info(model_name)
            self.assertIn(TaskType.IMAGE_CLASSIFICATION, info.capabilities.supported_tasks)
        
        # Test filtering by object detection
        detection_models = registry.list_models(task_type=TaskType.OBJECT_DETECTION)
        self.assertGreater(len(detection_models), 0)
        
        for model_name in detection_models:
            info = registry.get_model_info(model_name)
            self.assertIn(TaskType.OBJECT_DETECTION, info.capabilities.supported_tasks)
    
    def test_model_filtering_by_data_type(self):
        """Test filtering models by data type."""
        register_vision_models()
        
        registry = get_model_registry()
        
        # Test filtering by image data type
        image_models = registry.list_models(data_type=DataType.IMAGE)
        self.assertGreater(len(image_models), 0)
        
        # All returned models should support image data
        for model_name in image_models:
            info = registry.get_model_info(model_name)
            self.assertIn(DataType.IMAGE, info.capabilities.supported_data_types)
    
    def test_auto_model_selection(self):
        """Test automatic model selection for vision tasks."""
        register_vision_models()
        
        registry = get_model_registry()
        
        # Test auto-selection for image classification
        selected_model = registry.auto_select_model(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            data_type=DataType.IMAGE,
            num_samples=1000
        )
        
        self.assertIsNotNone(selected_model)
        
        # Verify the selected model supports the task and data type
        info = registry.get_model_info(selected_model)
        self.assertIn(TaskType.IMAGE_CLASSIFICATION, info.capabilities.supported_tasks)
        self.assertIn(DataType.IMAGE, info.capabilities.supported_data_types)
        
        # Test auto-selection for object detection
        selected_detection_model = registry.auto_select_model(
            task_type=TaskType.OBJECT_DETECTION,
            data_type=DataType.IMAGE,
            num_samples=500
        )
        
        self.assertIsNotNone(selected_detection_model)
        
        info = registry.get_model_info(selected_detection_model)
        self.assertIn(TaskType.OBJECT_DETECTION, info.capabilities.supported_tasks)


class TestVisionModelIntegration(unittest.TestCase):
    """Integration tests for vision models with synthetic data."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear registry before each test
        registry = get_model_registry()
        registry.clear()
    
    def create_synthetic_dataset(self, num_samples=50, num_classes=5, image_size=(224, 224)):
        """Create a synthetic image dataset for testing."""
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create synthetic image with some pattern
            img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
            
            # Add some class-specific patterns
            class_id = i % num_classes
            if class_id == 0:
                img_array[:50, :50, 0] = 255  # Red square for class 0
            elif class_id == 1:
                img_array[:50, :50, 1] = 255  # Green square for class 1
            elif class_id == 2:
                img_array[:50, :50, 2] = 255  # Blue square for class 2
            
            img = Image.fromarray(img_array)
            images.append(img)
            labels.append(class_id)
        
        return images, np.array(labels)
    
    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline with synthetic data."""
        try:
            # Register models
            register_vision_models()
            registry = get_model_registry()
            
            # Create synthetic dataset
            X_train, y_train = self.create_synthetic_dataset(30, 3)
            X_test, y_test = self.create_synthetic_dataset(10, 3)
            
            # Test with ResNet
            model = registry.get_model("resnet_18", num_classes=3)
            
            # Train model
            model.fit(X_train, y_train, epochs=2, batch_size=8)
            
            # Make predictions
            result = model.predict(X_test)
            
            # Verify results
            self.assertEqual(len(result.predictions), 10)
            self.assertEqual(result.probabilities.shape, (10, 3))
            
            # Check that predictions are valid
            self.assertTrue(all(0 <= pred < 3 for pred in result.predictions))
            
            # Check that probabilities are valid
            prob_sums = np.sum(result.probabilities, axis=1)
            np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
            
        except DependencyError:
            self.skipTest("PyTorch not available")
    
    def test_multiple_model_comparison(self):
        """Test training and comparing multiple vision models."""
        try:
            # Register models
            register_vision_models()
            registry = get_model_registry()
            
            # Create synthetic dataset
            X_train, y_train = self.create_synthetic_dataset(40, 4)
            X_test, y_test = self.create_synthetic_dataset(12, 4)
            
            # Test multiple models
            model_names = ["resnet_18", "efficientnet_b0"]
            results = {}
            
            for model_name in model_names:
                model = registry.get_model(model_name, num_classes=4)
                
                # Train model
                model.fit(X_train, y_train, epochs=2, batch_size=8)
                
                # Make predictions
                result = model.predict(X_test)
                results[model_name] = result
                
                # Verify results
                self.assertEqual(len(result.predictions), 12)
                self.assertEqual(result.probabilities.shape, (12, 4))
            
            # Compare results (both models should produce valid outputs)
            for model_name, result in results.items():
                self.assertTrue(all(0 <= pred < 4 for pred in result.predictions))
                prob_sums = np.sum(result.probabilities, axis=1)
                np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
                
        except DependencyError:
            self.skipTest("PyTorch not available")


if __name__ == '__main__':
    unittest.main()