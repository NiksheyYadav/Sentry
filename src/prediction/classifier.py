# Mental Health Classifier Module
# Three-headed classification for stress, depression, and anxiety

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from ..config import PredictionConfig, FusionConfig


@dataclass
class MentalHealthPrediction:
    """Mental health assessment prediction."""
    # Stress assessment
    stress_level: str
    stress_confidence: float
    stress_probabilities: Dict[str, float]
    
    # Depression assessment
    depression_level: str
    depression_confidence: float
    depression_probabilities: Dict[str, float]
    
    # Anxiety assessment
    anxiety_level: str
    anxiety_confidence: float
    anxiety_probabilities: Dict[str, float]
    
    # Overall
    primary_concern: str
    overall_severity: float
    requires_attention: bool


class ClassificationHead(nn.Module):
    """Single classification head with uncertainty estimation."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MentalHealthClassifier(nn.Module):
    """
    Three-headed classifier for mental health assessment.
    
    Produces calibrated predictions for:
    - Stress (3 levels)
    - Depression (4 levels)
    - Anxiety (4 levels)
    """
    
    def __init__(self, 
                 prediction_config: Optional[PredictionConfig] = None,
                 fusion_config: Optional[FusionConfig] = None):
        """
        Initialize classifier.
        
        Args:
            prediction_config: Prediction configuration.
            fusion_config: Fusion configuration for input dimension.
        """
        super().__init__()
        
        self.pred_config = prediction_config or PredictionConfig()
        self.fusion_config = fusion_config or FusionConfig()
        
        input_dim = self.fusion_config.fused_dim  # 1024
        
        # Shared feature extractor
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Classification heads
        self.stress_head = ClassificationHead(
            512, len(self.pred_config.stress_levels)
        )
        self.depression_head = ClassificationHead(
            512, len(self.pred_config.depression_levels)
        )
        self.anxiety_head = ClassificationHead(
            512, len(self.pred_config.anxiety_levels)
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(
            torch.tensor(self.pred_config.temperature)
        )
        
        self._device = "cpu"
        self._dropout_enabled = False
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def enable_mc_dropout(self) -> None:
        """Enable dropout for Monte Carlo uncertainty estimation."""
        self._dropout_enabled = True
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_mc_dropout(self) -> None:
        """Disable dropout for normal inference."""
        self._dropout_enabled = False
        self.eval()
    
    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            fused_features: Fused embedding (B, 1024).
            
        Returns:
            Dictionary with logits for each head.
        """
        # Shared features
        shared = self.shared_backbone(fused_features)
        
        # Get logits from each head
        stress_logits = self.stress_head(shared)
        depression_logits = self.depression_head(shared)
        anxiety_logits = self.anxiety_head(shared)
        
        # Apply temperature scaling
        stress_logits = stress_logits / self.temperature
        depression_logits = depression_logits / self.temperature
        anxiety_logits = anxiety_logits / self.temperature
        
        return {
            'stress': stress_logits,
            'depression': depression_logits,
            'anxiety': anxiety_logits
        }
    
    def predict(self, fused_features: np.ndarray, 
                use_mc_dropout: bool = False,
                num_samples: int = 10) -> MentalHealthPrediction:
        """
        Make prediction with optional uncertainty estimation.
        
        Args:
            fused_features: Fused embedding array (1024,).
            use_mc_dropout: Use Monte Carlo dropout for uncertainty.
            num_samples: Number of MC samples.
            
        Returns:
            MentalHealthPrediction dataclass.
        """
        # Convert to tensor
        x = torch.from_numpy(fused_features).float().unsqueeze(0).to(self._device)
        
        if use_mc_dropout:
            self.enable_mc_dropout()
            
            # Collect samples
            all_outputs = []
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.forward(x)
                all_outputs.append(outputs)
            
            # Average predictions
            stress_probs = torch.stack([
                F.softmax(o['stress'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            depression_probs = torch.stack([
                F.softmax(o['depression'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            anxiety_probs = torch.stack([
                F.softmax(o['anxiety'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            
            self.disable_mc_dropout()
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.forward(x)
                stress_probs = F.softmax(outputs['stress'], dim=1)
                depression_probs = F.softmax(outputs['depression'], dim=1)
                anxiety_probs = F.softmax(outputs['anxiety'], dim=1)
        
        # Convert to numpy
        stress_probs = stress_probs.cpu().numpy()[0]
        depression_probs = depression_probs.cpu().numpy()[0]
        anxiety_probs = anxiety_probs.cpu().numpy()[0]
        
        # Build prediction
        stress_idx = np.argmax(stress_probs)
        depression_idx = np.argmax(depression_probs)
        anxiety_idx = np.argmax(anxiety_probs)
        
        # Determine primary concern
        severities = {
            'stress': stress_idx / (len(self.pred_config.stress_levels) - 1),
            'depression': depression_idx / (len(self.pred_config.depression_levels) - 1),
            'anxiety': anxiety_idx / (len(self.pred_config.anxiety_levels) - 1)
        }
        primary_concern = max(severities, key=severities.get)
        overall_severity = max(severities.values())
        
        # Determine if attention required
        requires_attention = (
            stress_idx >= 2 or 
            depression_idx >= 2 or 
            anxiety_idx >= 2
        )
        
        return MentalHealthPrediction(
            stress_level=self.pred_config.stress_levels[stress_idx],
            stress_confidence=float(stress_probs[stress_idx]),
            stress_probabilities={
                level: float(stress_probs[i])
                for i, level in enumerate(self.pred_config.stress_levels)
            },
            depression_level=self.pred_config.depression_levels[depression_idx],
            depression_confidence=float(depression_probs[depression_idx]),
            depression_probabilities={
                level: float(depression_probs[i])
                for i, level in enumerate(self.pred_config.depression_levels)
            },
            anxiety_level=self.pred_config.anxiety_levels[anxiety_idx],
            anxiety_confidence=float(anxiety_probs[anxiety_idx]),
            anxiety_probabilities={
                level: float(anxiety_probs[i])
                for i, level in enumerate(self.pred_config.anxiety_levels)
            },
            primary_concern=primary_concern,
            overall_severity=float(overall_severity),
            requires_attention=requires_attention
        )


def create_classifier(prediction_config: Optional[PredictionConfig] = None,
                      fusion_config: Optional[FusionConfig] = None,
                      device: str = "cuda") -> MentalHealthClassifier:
    """
    Factory function to create classifier.
    
    Args:
        prediction_config: Prediction configuration.
        fusion_config: Fusion configuration.
        device: Computation device.
        
    Returns:
        Initialized MentalHealthClassifier on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    classifier = MentalHealthClassifier(prediction_config, fusion_config)
    classifier.to(device)
    return classifier
