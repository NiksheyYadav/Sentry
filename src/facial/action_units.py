# Facial Action Unit Detection Module
# Detects fundamental muscle movements that comprise expressions

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import cv2

from ..config import FacialConfig


@dataclass
class ActionUnitResult:
    """Action Unit detection result."""
    au_intensities: Dict[str, float]  # AU name -> intensity (0-5)
    au_presence: Dict[str, bool]  # AU name -> present
    clinical_patterns: Dict[str, float]  # Pattern name -> confidence


class ActionUnitDetector(nn.Module):
    """
    Facial Action Unit detector for clinical pattern recognition.
    
    Detects specific AUs associated with depression and anxiety:
    - AU4 (brow lowering): Associated with distress
    - AU12 (lip corner pulling): Reduced in depression
    - AU15 (lip corner depression): Elevated in sadness
    """
    
    # AU indices relevant for mental health
    MENTAL_HEALTH_AUS = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU12', 'AU14', 'AU15', 'AU17', 'AU45']
    
    def __init__(self, config: Optional[FacialConfig] = None):
        """
        Initialize AU detector.
        
        Args:
            config: Facial analysis configuration.
        """
        super().__init__()
        self.config = config or FacialConfig()
        self.num_aus = len(self.config.action_units)
        
        # Feature extractor (shared with emotion model or standalone)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # AU intensity regression heads (one per AU)
        self.au_heads = nn.ModuleDict({
            au: nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Output 0-1, scaled to 0-5
            )
            for au in self.config.action_units
        })
        
        # Clinical pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(self.num_aus, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # Depression, anxiety, neutral, positive
            nn.Softmax(dim=1)
        )
        
        self._device = "cpu"
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            
        Returns:
            AU intensities tensor of shape (B, num_aus).
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Get all AU predictions
        au_preds = []
        for au in self.config.action_units:
            pred = self.au_heads[au](features) * 5.0  # Scale to 0-5
            au_preds.append(pred)
        
        return torch.cat(au_preds, dim=1)
    
    def predict(self, face_image: np.ndarray) -> ActionUnitResult:
        """
        Predict AUs for a face image.
        
        Args:
            face_image: RGB face image (HxWx3).
            
        Returns:
            ActionUnitResult with intensities and patterns.
        """
        self.eval()
        
        # Preprocess
        img = cv2.resize(face_image, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self._device)
        
        # Use FP16 if on CUDA
        use_amp = (self._device == "cuda" or (isinstance(self._device, torch.device) and self._device.type == "cuda"))
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                au_intensities = self.forward(input_tensor)
                patterns = self.pattern_detector(au_intensities / 5.0)  # Normalize for pattern detection
        
        au_np = au_intensities.cpu().float().numpy()[0]
        patterns_np = patterns.cpu().float().numpy()[0]
        
        # Build result
        intensity_dict = {
            au: float(au_np[i])
            for i, au in enumerate(self.config.action_units)
        }
        
        presence_dict = {
            au: float(au_np[i]) > 1.0  # Present if intensity > 1
            for i, au in enumerate(self.config.action_units)
        }
        
        pattern_names = ['depression_indicator', 'anxiety_indicator', 'neutral', 'positive']
        pattern_dict = {
            name: float(patterns_np[i])
            for i, name in enumerate(pattern_names)
        }
        
        return ActionUnitResult(
            au_intensities=intensity_dict,
            au_presence=presence_dict,
            clinical_patterns=pattern_dict
        )
    
    def get_depression_indicators(self, result: ActionUnitResult) -> Dict[str, float]:
        """
        Extract depression-related AU patterns.
        
        Depression manifests through:
        - Reduced AU12 (lip corner pulling / smiling)
        - Increased AU4 (brow lowering)
        - Sustained AU15 (lip corner depression)
        - Reduced overall facial expressivity
        """
        au = result.au_intensities
        
        # Calculate depression indicators
        indicators = {
            'reduced_smiling': max(0, 3.0 - au.get('AU12', 0)) / 3.0,
            'brow_lowering': au.get('AU4', 0) / 5.0,
            'lip_depression': au.get('AU15', 0) / 5.0,
            'flat_affect': 1.0 - (sum(au.values()) / (len(au) * 5.0)),
            'overall': result.clinical_patterns.get('depression_indicator', 0)
        }
        
        return indicators
    
    def get_anxiety_indicators(self, result: ActionUnitResult) -> Dict[str, float]:
        """
        Extract anxiety-related AU patterns.
        
        Anxiety manifests through:
        - Increased AU1+AU2 (inner/outer brow raise)
        - Increased AU4 (brow lowering)
        - Eye widening (AU5)
        - Lip tension (AU23/24)
        """
        au = result.au_intensities
        
        indicators = {
            'brow_raise': (au.get('AU1', 0) + au.get('AU2', 0)) / 10.0,
            'brow_lowering': au.get('AU4', 0) / 5.0,
            'eye_widening': au.get('AU5', 0) / 5.0,
            'overall': result.clinical_patterns.get('anxiety_indicator', 0)
        }
        
        return indicators


def create_au_detector(config: Optional[FacialConfig] = None,
                       device: str = "cuda") -> ActionUnitDetector:
    """
    Factory function to create AU detector.
    
    Args:
        config: Facial analysis configuration.
        device: Computation device.
        
    Returns:
        Initialized ActionUnitDetector on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    detector = ActionUnitDetector(config=config)
    detector.to(device)
    return detector
