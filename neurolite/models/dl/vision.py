"""
Computer vision deep learning models for NeuroLite.

Implements CNN architectures, object detection, and image segmentation models
using PyTorch and other deep learning frameworks.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ...core import get_logger, DependencyError, ModelError
from ...data.detector import DataType
from ..base import BaseModel, TaskType, ModelCapabilities, PredictionResult, ModelMetadata
from ..registry import register_model


logger = get_logger(__name__)


class PyTorchVisionModel(BaseModel):
    """
    Base class for PyTorch-based computer vision models.
    
    Provides common functionality for CNN architectures, object detection,
    and segmentation models.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PyTorch vision model.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            input_size: Expected input size (C, H, W)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.input_size = input_size
        self.device = device
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = {}
        
        # Import PyTorch dependencies
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torchvision
            import torchvision.transforms as transforms
            
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.torchvision = torchvision
            self.transforms = transforms
            
        except ImportError as e:
            raise DependencyError(
                f"Missing PyTorch dependencies required for vision models: {e}",
                suggestions=[
                    "Install PyTorch: pip install torch torchvision",
                    "Install NeuroLite with vision extras: pip install neurolite[vision]",
                    "Visit https://pytorch.org for installation instructions"
                ]
            )
        
        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.debug(f"Initialized {model_name} model on device: {self.device}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        return ModelCapabilities(
            supported_tasks=[
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION
            ],
            supported_data_types=[DataType.IMAGE],
            framework="pytorch",
            requires_gpu=False,  # Can run on CPU but GPU is preferred
            min_samples=10,
            supports_probability_prediction=True,
            supports_feature_importance=False
        )
    
    def _create_model(self) -> Any:
        """Create the PyTorch model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def _get_transforms(self, training: bool = True) -> Any:
        """Get data transforms for preprocessing."""
        if training:
            return self.transforms.Compose([
                self.transforms.Resize((self.input_size[1], self.input_size[2])),
                self.transforms.RandomHorizontalFlip(p=0.5),
                self.transforms.RandomRotation(degrees=10),
                self.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return self.transforms.Compose([
                self.transforms.Resize((self.input_size[1], self.input_size[2])),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _prepare_data(self, X: Any, y: Any = None, training: bool = True) -> Tuple[Any, Any]:
        """Prepare data for training or inference."""
        from PIL import Image
        
        # Convert data to appropriate format
        if isinstance(X, (list, tuple)):
            # Handle list of image paths or PIL images
            images = []
            for item in X:
                if isinstance(item, str):
                    # Load image from path
                    img = Image.open(item).convert('RGB')
                elif isinstance(item, Image.Image):
                    img = item.convert('RGB')
                elif isinstance(item, np.ndarray):
                    # Convert numpy array to PIL Image
                    if item.shape[-1] == 3:  # RGB
                        img = Image.fromarray(item.astype(np.uint8))
                    else:
                        raise ValueError(f"Unsupported image array shape: {item.shape}")
                else:
                    raise ValueError(f"Unsupported image type: {type(item)}")
                images.append(img)
            X = images
        
        # Apply transforms
        transform = self._get_transforms(training=training)
        
        if isinstance(X, list):
            X_tensor = self.torch.stack([transform(img) for img in X])
        else:
            raise ValueError(f"Unsupported data format: {type(X)}")
        
        # Prepare targets
        y_tensor = None
        if y is not None:
            if isinstance(y, (list, tuple, np.ndarray)):
                y_tensor = self.torch.tensor(y, dtype=self.torch.long)
            else:
                y_tensor = y
        
        return X_tensor.to(self.device), y_tensor.to(self.device) if y_tensor is not None else None
    
    def fit(
        self,
        X: Union[np.ndarray, List, Any],
        y: Union[np.ndarray, List, Any],
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs
    ) -> 'PyTorchVisionModel':
        """Train the vision model."""
        logger.debug(f"Training {self.model_name} for {epochs} epochs")
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
            self.model.to(self.device)
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y, training=True)
        
        # Create data loader
        dataset = self.torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = self.torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup training components
        self.criterion = self.nn.CrossEntropyLoss()
        self.optimizer = self.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        self.training_history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = self.torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Record metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)
            
            if epoch % max(1, epochs // 10) == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        self.is_trained = True
        logger.debug(f"Training completed for {self.model_name}")
        return self
    
    def predict(self, X: Union[np.ndarray, List, Any], **kwargs) -> PredictionResult:
        """Make predictions using the vision model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare data
        X_tensor, _ = self._prepare_data(X, training=False)
        
        # Make predictions
        self.model.eval()
        with self.torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = self.torch.softmax(outputs, dim=1)
            predictions = self.torch.argmax(outputs, dim=1)
        
        return PredictionResult(
            predictions=predictions.cpu().numpy(),
            probabilities=probabilities.cpu().numpy()
        )
    
    def save(self, path: str) -> None:
        """Save the PyTorch model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'training_history': self.training_history,
            'config': self._config,
            'is_trained': self.is_trained
        }
        
        self.torch.save(save_dict, path)
        logger.debug(f"Saved {self.model_name} to {path}")
    
    def load(self, path: str) -> 'PyTorchVisionModel':
        """Load the PyTorch model."""
        save_dict = self.torch.load(path, map_location=self.device)
        
        # Restore configuration
        self.model_name = save_dict['model_name']
        self.num_classes = save_dict['num_classes']
        self.input_size = save_dict['input_size']
        self.training_history = save_dict.get('training_history', {})
        self._config = save_dict.get('config', {})
        self.is_trained = save_dict.get('is_trained', True)
        
        # Create and load model
        self.model = self._create_model()
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.model.to(self.device)
        
        logger.debug(f"Loaded {self.model_name} from {path}")
        return self


class ResNetModel(PyTorchVisionModel):
    """ResNet architecture for image classification."""
    
    def __init__(self, variant: str = "resnet50", **kwargs):
        """
        Initialize ResNet model.
        
        Args:
            variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            **kwargs: Additional arguments passed to parent
        """
        self.variant = variant
        super().__init__(model_name=f"ResNet-{variant}", **kwargs)
    
    def _create_model(self) -> Any:
        """Create ResNet model."""
        # Get the appropriate ResNet variant
        resnet_models = {
            'resnet18': self.torchvision.models.resnet18,
            'resnet34': self.torchvision.models.resnet34,
            'resnet50': self.torchvision.models.resnet50,
            'resnet101': self.torchvision.models.resnet101,
            'resnet152': self.torchvision.models.resnet152
        }
        
        if self.variant not in resnet_models:
            raise ValueError(f"Unsupported ResNet variant: {self.variant}")
        
        # Create model
        model = resnet_models[self.variant](pretrained=self.pretrained)
        
        # Modify final layer for custom number of classes
        if self.num_classes != 1000:
            model.fc = self.nn.Linear(model.fc.in_features, self.num_classes)
        
        return model


class EfficientNetModel(PyTorchVisionModel):
    """EfficientNet architecture for image classification."""
    
    def __init__(self, variant: str = "b0", **kwargs):
        """
        Initialize EfficientNet model.
        
        Args:
            variant: EfficientNet variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
            **kwargs: Additional arguments passed to parent
        """
        self.variant = variant
        super().__init__(model_name=f"EfficientNet-{variant}", **kwargs)
    
    def _create_model(self) -> Any:
        """Create EfficientNet model."""
        try:
            # Try to use torchvision's EfficientNet (available in newer versions)
            efficientnet_models = {
                'b0': self.torchvision.models.efficientnet_b0,
                'b1': self.torchvision.models.efficientnet_b1,
                'b2': self.torchvision.models.efficientnet_b2,
                'b3': self.torchvision.models.efficientnet_b3,
                'b4': self.torchvision.models.efficientnet_b4,
                'b5': self.torchvision.models.efficientnet_b5,
                'b6': self.torchvision.models.efficientnet_b6,
                'b7': self.torchvision.models.efficientnet_b7
            }
            
            if self.variant not in efficientnet_models:
                raise ValueError(f"Unsupported EfficientNet variant: {self.variant}")
            
            # Create model
            model = efficientnet_models[self.variant](pretrained=self.pretrained)
            
            # Modify final layer for custom number of classes
            if self.num_classes != 1000:
                model.classifier[1] = self.nn.Linear(model.classifier[1].in_features, self.num_classes)
            
            return model
            
        except AttributeError:
            # Fallback: create a simple CNN if EfficientNet is not available
            warnings.warn(
                f"EfficientNet-{self.variant} not available in this torchvision version. "
                "Using a simple CNN instead."
            )
            return self._create_simple_cnn()
    
    def _create_simple_cnn(self) -> Any:
        """Create a simple CNN as fallback."""
        nn = self.nn
        
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleCNN(self.num_classes)


class VisionTransformerModel(PyTorchVisionModel):
    """Vision Transformer (ViT) architecture for image classification."""
    
    def __init__(self, variant: str = "base", patch_size: int = 16, **kwargs):
        """
        Initialize Vision Transformer model.
        
        Args:
            variant: ViT variant ('tiny', 'small', 'base', 'large')
            patch_size: Size of image patches
            **kwargs: Additional arguments passed to parent
        """
        self.variant = variant
        self.patch_size = patch_size
        super().__init__(model_name=f"ViT-{variant}", **kwargs)
    
    def _create_model(self) -> Any:
        """Create Vision Transformer model."""
        try:
            # Try to use torchvision's ViT (available in newer versions)
            vit_models = {
                'base': self.torchvision.models.vit_b_16,
                'large': self.torchvision.models.vit_l_16,
            }
            
            if self.variant in vit_models:
                model = vit_models[self.variant](pretrained=self.pretrained)
                
                # Modify final layer for custom number of classes
                if self.num_classes != 1000:
                    model.heads.head = self.nn.Linear(model.heads.head.in_features, self.num_classes)
                
                return model
            else:
                # Create simple transformer-like model
                return self._create_simple_vit()
                
        except AttributeError:
            # Fallback: create a simple transformer-like model
            warnings.warn(
                f"Vision Transformer not available in this torchvision version. "
                "Using a simple transformer-like model instead."
            )
            return self._create_simple_vit()
    
    def _create_simple_vit(self) -> Any:
        """Create a simple Vision Transformer-like model."""
        nn = self.nn
        torch = self.torch
        
        class SimpleViT(nn.Module):
            def __init__(self, num_classes, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
                super().__init__()
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                
                # Positional embedding (simplified)
                self.pos_embed = nn.Parameter(torch.randn(1, 197, embed_dim))  # 14*14 + 1 cls token
                
                # Class token
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Classification head
                self.head = nn.Linear(embed_dim, num_classes)
                
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                
                # Add class token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # Add positional embedding
                x = x + self.pos_embed[:, :x.size(1), :]
                
                # Transformer
                x = self.transformer(x)
                
                # Classification
                cls_token_final = x[:, 0]
                return self.head(cls_token_final)
        
        return SimpleViT(self.num_classes, self.patch_size)


class ObjectDetectionModel(PyTorchVisionModel):
    """Object detection model using popular architectures."""
    
    def __init__(self, architecture: str = "fasterrcnn", **kwargs):
        """
        Initialize object detection model.
        
        Args:
            architecture: Detection architecture ('fasterrcnn', 'ssd', 'yolo')
            **kwargs: Additional arguments passed to parent
        """
        self.architecture = architecture
        super().__init__(model_name=f"ObjectDetection-{architecture}", **kwargs)
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities for object detection."""
        return ModelCapabilities(
            supported_tasks=[TaskType.OBJECT_DETECTION],
            supported_data_types=[DataType.IMAGE],
            framework="pytorch",
            requires_gpu=False,
            min_samples=10,
            supports_probability_prediction=True,
            supports_feature_importance=False
        )
    
    def _create_model(self) -> Any:
        """Create object detection model."""
        try:
            if self.architecture == "fasterrcnn":
                # Use Faster R-CNN from torchvision
                model = self.torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes
                )
                return model
            else:
                # Fallback to a simple detection model
                warnings.warn(
                    f"Object detection architecture '{self.architecture}' not fully implemented. "
                    "Using simplified detection model."
                )
                return self._create_simple_detector()
                
        except AttributeError:
            warnings.warn("Object detection models not available. Using simplified detector.")
            return self._create_simple_detector()
    
    def _create_simple_detector(self) -> Any:
        """Create a simplified object detection model."""
        # This is a placeholder - in practice, you'd implement a proper detector
        nn = self.nn
        torchvision = self.torchvision
        
        class SimpleDetector(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = torchvision.models.resnet50(pretrained=True)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
                
            def forward(self, x):
                return self.backbone(x)
        
        return SimpleDetector(self.num_classes)


class SegmentationModel(PyTorchVisionModel):
    """Image segmentation model."""
    
    def __init__(self, architecture: str = "deeplabv3", **kwargs):
        """
        Initialize segmentation model.
        
        Args:
            architecture: Segmentation architecture ('deeplabv3', 'fcn', 'unet')
            **kwargs: Additional arguments passed to parent
        """
        self.architecture = architecture
        super().__init__(model_name=f"Segmentation-{architecture}", **kwargs)
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities for segmentation."""
        return ModelCapabilities(
            supported_tasks=[TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION],
            supported_data_types=[DataType.IMAGE],
            framework="pytorch",
            requires_gpu=False,
            min_samples=10,
            supports_probability_prediction=True,
            supports_feature_importance=False
        )
    
    def _create_model(self) -> Any:
        """Create segmentation model."""
        try:
            if self.architecture == "deeplabv3":
                # Use DeepLabV3 from torchvision
                model = self.torchvision.models.segmentation.deeplabv3_resnet50(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes
                )
                return model
            elif self.architecture == "fcn":
                # Use FCN from torchvision
                model = self.torchvision.models.segmentation.fcn_resnet50(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes
                )
                return model
            else:
                warnings.warn(
                    f"Segmentation architecture '{self.architecture}' not fully implemented. "
                    "Using simplified segmentation model."
                )
                return self._create_simple_segmenter()
                
        except AttributeError:
            warnings.warn("Segmentation models not available. Using simplified segmenter.")
            return self._create_simple_segmenter()
    
    def _create_simple_segmenter(self) -> Any:
        """Create a simplified segmentation model."""
        nn = self.nn
        
        class SimpleSegmenter(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Simple U-Net-like architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, num_classes, 1)
                )
                
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        return SimpleSegmenter(self.num_classes)


def register_vision_models():
    """Register all computer vision models in the global registry."""
    logger.debug("Registering computer vision models")
    
    # ResNet models
    for variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        register_model(
            name=f"resnet_{variant.replace('resnet', '')}",
            model_class=ResNetModel,
            factory_function=lambda v=variant, **kwargs: ResNetModel(variant=v, **kwargs),
            priority=8,
            description=f"ResNet-{variant.replace('resnet', '')} architecture for image classification",
            tags=["cnn", "image_classification", "pretrained"]
        )
    
    # EfficientNet models
    for variant in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
        register_model(
            name=f"efficientnet_{variant}",
            model_class=EfficientNetModel,
            factory_function=lambda v=variant, **kwargs: EfficientNetModel(variant=v, **kwargs),
            priority=9,
            description=f"EfficientNet-{variant} architecture for image classification",
            tags=["cnn", "image_classification", "efficient", "pretrained"]
        )
    
    # Vision Transformer models
    for variant in ['base', 'large']:
        register_model(
            name=f"vit_{variant}",
            model_class=VisionTransformerModel,
            factory_function=lambda v=variant, **kwargs: VisionTransformerModel(variant=v, **kwargs),
            priority=7,
            description=f"Vision Transformer {variant} for image classification",
            tags=["transformer", "image_classification", "attention", "pretrained"]
        )
    
    # Object detection models
    for arch in ['fasterrcnn', 'ssd', 'yolo']:
        register_model(
            name=f"object_detection_{arch}",
            model_class=ObjectDetectionModel,
            factory_function=lambda a=arch, **kwargs: ObjectDetectionModel(architecture=a, **kwargs),
            priority=6,
            description=f"{arch.upper()} object detection model",
            tags=["object_detection", "cnn", "pretrained"]
        )
    
    # Segmentation models
    for arch in ['deeplabv3', 'fcn', 'unet']:
        register_model(
            name=f"segmentation_{arch}",
            model_class=SegmentationModel,
            factory_function=lambda a=arch, **kwargs: SegmentationModel(architecture=a, **kwargs),
            priority=6,
            description=f"{arch.upper()} segmentation model",
            tags=["segmentation", "cnn", "pretrained"]
        )
    
    logger.debug("Successfully registered computer vision models")