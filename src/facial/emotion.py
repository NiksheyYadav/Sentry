# Emotion Classification Module
# MobileNetV3-based emotion classifier with embedding extraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from ..config import FacialConfig


@dataclass
class EmotionResult:
    """Emotion classification result."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    embedding: np.ndarray  # 1280-dim embedding


class EmotionClassifier(nn.Module):
    """
    MobileNetV3-based emotion classifier.
    
    Classifies 7 basic emotions and extracts rich
    1280-dimensional embeddings for downstream fusion.
    """
    
    def __init__(self, config: Optional[FacialConfig] = None, pretrained: bool = True):
        """
        Initialize emotion classifier.
        
        Args:
            config: Facial analysis configuration.
            pretrained: Whether to use pretrained ImageNet weights.
        """
        super().__init__()
        self.config = config or FacialConfig()
        self.num_classes = len(self.config.emotion_classes)
        
        # Load pretrained MobileNetV3
        # if pretrained:
        #     weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        #     self.backbone = mobilenet_v3_large(weights=weights)
        # else:
        #     self.backbone = mobilenet_v3_large(weights=None)
        
        # Load pretrained DenseNet121
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = densenet121(weights=weights)
        else:
            self.backbone = densenet121(weights=None)
            
        # Modify first convolution for 1-channel input (Grayscale)
        # DenseNet121 features.conv0 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv = self.backbone.features.conv0
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize 1-channel weights by averaging RGB weights
        if pretrained:
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        self.backbone.features.conv0 = new_conv
        
        # Get embedding dimension from backbone
        # self.embedding_dim = self.backbone.classifier[0].in_features  # 960 (MobileNetV3)
        self.embedding_dim = self.backbone.classifier.in_features  # 1024 (DenseNet121)
        
        # Replace classifier with emotion head
        self.backbone.classifier = nn.Identity()
        
        # Projection to match expected embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Preprocessing (Grayscale)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])
        
        self._device = "cpu"
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            Tuple of (logits, embeddings).
        """
        # Get backbone features
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return logits, embeddings
    
    def predict(self, face_image: np.ndarray) -> EmotionResult:
        """
        Predict emotion for a face image.
        
        Args:
            face_image: RGB face image (HxWx3).
            
        Returns:
            EmotionResult with prediction and embedding.
        """
        self.eval()
        
        # Preprocess
        input_tensor = self.preprocess(face_image)
        input_tensor = input_tensor.unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            logits, embedding = self.forward(input_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Get prediction
        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        
        # Build probability dictionary
        prob_dict = {
            emotion: float(probs_np[i]) 
            for i, emotion in enumerate(self.config.emotion_classes)
        }
        
        return EmotionResult(
            emotion=self.config.emotion_classes[pred_idx],
            confidence=float(probs_np[pred_idx]),
            probabilities=prob_dict,
            embedding=embedding.cpu().numpy()[0]
        )
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[EmotionResult]:
        """
        Predict emotions for multiple face images.
        
        Args:
            face_images: List of RGB face images.
            
        Returns:
            List of EmotionResult objects.
        """
        if not face_images:
            return []
        
        self.eval()
        
        # Preprocess all images
        inputs = torch.stack([
            self.preprocess(img) for img in face_images
        ]).to(self._device)
        
        with torch.no_grad():
            logits, embeddings = self.forward(inputs)
            probs = F.softmax(logits, dim=1)
        
        # Build results
        probs_np = probs.cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()
        
        results = []
        for i in range(len(face_images)):
            pred_idx = np.argmax(probs_np[i])
            prob_dict = {
                emotion: float(probs_np[i, j])
                for j, emotion in enumerate(self.config.emotion_classes)
            }
            
            results.append(EmotionResult(
                emotion=self.config.emotion_classes[pred_idx],
                confidence=float(probs_np[i, pred_idx]),
                probabilities=prob_dict,
                embedding=embeddings_np[i]
            ))
        
        return results
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract only the embedding for a face.
        
        Args:
            face_image: RGB face image.
            
        Returns:
            1280-dimensional embedding.
        """
        result = self.predict(face_image)
        return result.embedding


def create_emotion_classifier(config: Optional[FacialConfig] = None, 
                              device: str = "cuda") -> EmotionClassifier:
    """
    Factory function to create emotion classifier.
    
    Args:
        config: Facial analysis configuration.
        device: Computation device.
        
    Returns:
        Initialized EmotionClassifier on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    classifier = EmotionClassifier(config=config)
    classifier.to(device)
    return classifier
