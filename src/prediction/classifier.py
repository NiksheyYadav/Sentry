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
    """Mental health assessment prediction with posture-based indicators."""
    # Stress assessment
    stress_level: str
    stress_confidence: float
    stress_probabilities: Dict[str, float]
    
    # Neutral assessment
    neutral_level: str
    neutral_confidence: float
    neutral_probabilities: Dict[str, float]
    
    # Anxiety assessment
    anxiety_level: str
    anxiety_confidence: float
    anxiety_probabilities: Dict[str, float]
    
    # Posture-based predictions (moved from posture model to post-fusion)
    posture_archetype: str  # upright, slouched, open, closed
    posture_archetype_confidence: float
    stress_indicator: str  # calm, fidgeting, restless, stillness
    stress_indicator_confidence: float
    trajectory: str  # stable, deteriorating, improving
    trajectory_confidence: float
    
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
    Multi-headed classifier for mental health assessment.
    
    Produces calibrated predictions for:
    - Stress (3 levels)
    - Depression (4 levels)
    - Anxiety (4 levels)
    - Posture archetype (4 types) - moved from posture model
    - Stress indicator (4 types) - moved from posture model  
    - Trajectory (3 types) - moved from posture model
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
        
        # Mental health classification heads
        self.stress_head = ClassificationHead(
            512, len(self.pred_config.stress_levels)
        )
        self.neutral_head = ClassificationHead(
            512, len(self.pred_config.neutral_levels)
        )
        self.anxiety_head = ClassificationHead(
            512, len(self.pred_config.anxiety_levels)
        )
        
        # Posture-based classification heads (moved from posture model)
        self.posture_archetype_head = ClassificationHead(
            512, len(self.pred_config.posture_archetypes)
        )
        self.stress_indicator_head = ClassificationHead(
            512, len(self.pred_config.stress_indicators)
        )
        self.trajectory_head = ClassificationHead(
            512, len(self.pred_config.trajectories)
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
        
        # Mental health predictions
        stress_logits = self.stress_head(shared)
        neutral_logits = self.neutral_head(shared)
        anxiety_logits = self.anxiety_head(shared)
        
        # Posture-based predictions
        posture_logits = self.posture_archetype_head(shared)
        stress_ind_logits = self.stress_indicator_head(shared)
        trajectory_logits = self.trajectory_head(shared)
        
        # Apply temperature scaling
        stress_logits = stress_logits / self.temperature
        neutral_logits = neutral_logits / self.temperature
        anxiety_logits = anxiety_logits / self.temperature
        posture_logits = posture_logits / self.temperature
        stress_ind_logits = stress_ind_logits / self.temperature
        trajectory_logits = trajectory_logits / self.temperature
        
        return {
            'stress': stress_logits,
            'neutral': neutral_logits,
            'anxiety': anxiety_logits,
            'posture_archetype': posture_logits,
            'stress_indicator': stress_ind_logits,
            'trajectory': trajectory_logits
        }
    
    def predict(self, fused_features: np.ndarray, 
                use_mc_dropout: bool = False,
                num_samples: int = 10,
                emotion_hint: Optional[str] = None) -> MentalHealthPrediction:
        """
        Make prediction with optional uncertainty estimation.
        
        Args:
            fused_features: Fused embedding array (1024,).
            use_mc_dropout: Use Monte Carlo dropout for uncertainty.
            num_samples: Number of MC samples.
            emotion_hint: Optional detected emotion to guide assessment.
            
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
            
            # Average predictions for all heads
            stress_probs = torch.stack([
                F.softmax(o['stress'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            neutral_probs = torch.stack([
                F.softmax(o['neutral'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            anxiety_probs = torch.stack([
                F.softmax(o['anxiety'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            posture_probs = torch.stack([
                F.softmax(o['posture_archetype'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            stress_ind_probs = torch.stack([
                F.softmax(o['stress_indicator'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            trajectory_probs = torch.stack([
                F.softmax(o['trajectory'], dim=1) for o in all_outputs
            ]).mean(dim=0)
            
            self.disable_mc_dropout()
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.forward(x)
                stress_probs = F.softmax(outputs['stress'], dim=1)
                neutral_probs = F.softmax(outputs['neutral'], dim=1)
                anxiety_probs = F.softmax(outputs['anxiety'], dim=1)
                posture_probs = F.softmax(outputs['posture_archetype'], dim=1)
                stress_ind_probs = F.softmax(outputs['stress_indicator'], dim=1)
                trajectory_probs = F.softmax(outputs['trajectory'], dim=1)
        
        # Convert to numpy
        stress_probs = stress_probs.cpu().numpy()[0]
        neutral_probs = neutral_probs.cpu().numpy()[0]
        anxiety_probs = anxiety_probs.cpu().numpy()[0]
        
        # Apply Heuristic Safety Override
        if emotion_hint == 'happy':
            # Force high neutral and low stress/anxiety
            # Low Stress: [0.9, 0.1, 0.0]
            # High Neutral: [0.0, 0.1, 0.9]
            # Low Anxiety: [0.9, 0.1, 0.0, 0.0]
            stress_probs = np.array([0.9, 0.1, 0.0])
            neutral_probs = np.array([0.0, 0.1, 0.9])
            anxiety_probs = np.array([0.9, 0.1, 0.0, 0.0])
        elif emotion_hint == 'neutral':
            # Cap anxiety at 'mild' and favor 'normal' neutral
            stress_probs = np.array([0.8, 0.2, 0.0])
            # Boost index 1 (normal)
            neutral_probs = np.array([0.1, 0.8, 0.1])
            # Anxiety: [minimal, mild, moderate, severe]
            # Boost minimal/mild, suppress moderate/severe
            anxiety_probs = np.array([0.7, 0.3, 0.0, 0.0])
        elif emotion_hint == 'anger':
            # Force high stress and low neutral
            # Stress: [low, moderate, high]
            stress_probs = np.array([0.0, 0.2, 0.8])
            # Neutral: [low, normal, high]
            neutral_probs = np.array([0.9, 0.1, 0.0])
            # Anxiety: [minimal, mild, moderate, severe]
            anxiety_probs = np.array([0.3, 0.4, 0.3, 0.0])
            
        posture_probs = posture_probs.cpu().numpy()[0]
        stress_ind_probs = stress_ind_probs.cpu().numpy()[0]
        trajectory_probs = trajectory_probs.cpu().numpy()[0]
        
        # Build prediction indices
        stress_idx = np.argmax(stress_probs)
        neutral_idx = np.argmax(neutral_probs)
        anxiety_idx = np.argmax(anxiety_probs)
        posture_idx = np.argmax(posture_probs)
        stress_ind_idx = np.argmax(stress_ind_probs)
        trajectory_idx = np.argmax(trajectory_probs)
        
        # Determine primary concern (mental health only)
        severities = {
            'stress': stress_idx / (len(self.pred_config.stress_levels) - 1),
            'neutral': neutral_idx / (len(self.pred_config.neutral_levels) - 1),
            'anxiety': anxiety_idx / (len(self.pred_config.anxiety_levels) - 1)
        }
        primary_concern = max(severities, key=severities.get)
        overall_severity = max(severities.values())
        
        # Determine if attention required
        requires_attention = (
            stress_idx >= 2 or 
            neutral_idx >= 2 or 
            anxiety_idx >= 2
        )
        
        return MentalHealthPrediction(
            stress_level=self.pred_config.stress_levels[stress_idx],
            stress_confidence=float(stress_probs[stress_idx]),
            stress_probabilities={
                level: float(stress_probs[i])
                for i, level in enumerate(self.pred_config.stress_levels)
            },
            neutral_level=self.pred_config.neutral_levels[neutral_idx],
            neutral_confidence=float(neutral_probs[neutral_idx]),
            neutral_probabilities={
                level: float(neutral_probs[i])
                for i, level in enumerate(self.pred_config.neutral_levels)
            },
            anxiety_level=self.pred_config.anxiety_levels[anxiety_idx],
            anxiety_confidence=float(anxiety_probs[anxiety_idx]),
            anxiety_probabilities={
                level: float(anxiety_probs[i])
                for i, level in enumerate(self.pred_config.anxiety_levels)
            },
            posture_archetype=self.pred_config.posture_archetypes[posture_idx],
            posture_archetype_confidence=float(posture_probs[posture_idx]),
            stress_indicator=self.pred_config.stress_indicators[stress_ind_idx],
            stress_indicator_confidence=float(stress_ind_probs[stress_ind_idx]),
            trajectory=self.pred_config.trajectories[trajectory_idx],
            trajectory_confidence=float(trajectory_probs[trajectory_idx]),
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
