# Fusion Network Module
# Combines multimodal features with temporal consistency

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config import FusionConfig
from .cross_attention import CrossAttentionFusion


@dataclass
class FusedRepresentation:
    """Fused multimodal representation."""
    embedding: np.ndarray  # 1024-dim fused embedding
    facial_contribution: float  # 0-1 facial weight
    posture_contribution: float  # 0-1 posture weight
    temporal_consistency: float  # Smoothness of prediction


class TemporalConsistencyLayer(nn.Module):
    """
    Recurrent layer for temporal smoothing of predictions.
    
    Prevents erratic fluctuation by considering previous state.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self._hidden = None
    
    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> None:
        """Reset hidden state."""
        self._hidden = torch.zeros(1, batch_size, self.gru.hidden_size).to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Apply temporal consistency.
        
        Args:
            x: Input tensor (B, dim).
            
        Returns:
            Tuple of (smoothed_output, consistency_score).
        """
        # Expand for GRU
        x_seq = x.unsqueeze(1)  # (B, 1, dim)
        
        if self._hidden is None:
            self.reset_state(x.size(0), x.device)
        
        # GRU forward
        gru_out, self._hidden = self.gru(x_seq, self._hidden)
        gru_out = gru_out.squeeze(1)
        
        # Project back to input dimension
        memory = self.projection(gru_out)
        
        # Gated combination
        gate_input = torch.cat([x, memory], dim=-1)
        gate_weight = self.gate(gate_input)
        
        # Blend current with memory
        output = gate_weight * x + (1 - gate_weight) * memory
        
        # Consistency score (how much we rely on memory)
        consistency = 1 - float(gate_weight.mean())
        
        return output, consistency


class FusionNetwork(nn.Module):
    """
    Complete multimodal fusion network.
    
    Combines cross-attention fusion with temporal consistency
    to produce stable, integrated representations.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize fusion network.
        
        Args:
            config: Fusion configuration.
        """
        super().__init__()
        self.config = config or FusionConfig()
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(config)
        
        # Concatenation and projection
        total_dim = self.config.facial_embed_dim + self.config.posture_embed_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(768, self.config.fused_dim),
            nn.LayerNorm(self.config.fused_dim)
        )
        
        # Temporal consistency layer
        self.temporal_layer = TemporalConsistencyLayer(
            input_dim=self.config.fused_dim,
            hidden_dim=256
        )
        
        # Modality importance estimation
        self.importance_net = nn.Sequential(
            nn.Linear(self.config.fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        self._device = "cpu"
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def reset_temporal_state(self, batch_size: int = 1) -> None:
        """Reset temporal consistency state."""
        self.temporal_layer.reset_state(batch_size, self._device)
    
    def forward(self, facial_features: torch.Tensor,
                posture_features: torch.Tensor,
                apply_temporal: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.
        
        Args:
            facial_features: Facial embedding (B, 1280).
            posture_features: Posture embedding (B, 512).
            apply_temporal: Whether to apply temporal smoothing.
            
        Returns:
            Tuple of (fused_embedding, metadata_dict).
        """
        # Cross-attention fusion
        facial_attended, posture_attended = self.cross_attention(
            facial_features, posture_features
        )
        
        # Concatenate
        combined = torch.cat([facial_attended, posture_attended], dim=-1)
        
        # Project to fused dimension
        fused = self.fusion_mlp(combined)
        
        # Apply temporal consistency
        consistency = 0.0
        if apply_temporal:
            fused, consistency = self.temporal_layer(fused)
        
        # Estimate modality importance
        importance = self.importance_net(fused)
        
        metadata = {
            'facial_importance': float(importance[:, 0].mean()),
            'posture_importance': float(importance[:, 1].mean()),
            'temporal_consistency': consistency
        }
        
        return fused, metadata
    
    def fuse(self, facial_features: np.ndarray,
             posture_features: np.ndarray) -> FusedRepresentation:
        """
        Fuse features and return structured result.
        
        Args:
            facial_features: Facial embedding array (1280,).
            posture_features: Posture embedding array (512,).
            
        Returns:
            FusedRepresentation dataclass.
        """
        self.eval()
        
        # Convert to tensors
        facial_t = torch.from_numpy(facial_features).float().unsqueeze(0).to(self._device)
        posture_t = torch.from_numpy(posture_features).float().unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            fused, metadata = self.forward(facial_t, posture_t)
        
        return FusedRepresentation(
            embedding=fused.cpu().numpy()[0],
            facial_contribution=metadata['facial_importance'],
            posture_contribution=metadata['posture_importance'],
            temporal_consistency=metadata['temporal_consistency']
        )
    
    def get_attention_analysis(self, facial_features: np.ndarray,
                                posture_features: np.ndarray) -> dict:
        """
        Analyze cross-modal attention for interpretability.
        
        Args:
            facial_features: Facial embedding.
            posture_features: Posture embedding.
            
        Returns:
            Dictionary with attention analysis.
        """
        facial_t = torch.from_numpy(facial_features).float().unsqueeze(0).to(self._device)
        posture_t = torch.from_numpy(posture_features).float().unsqueeze(0).to(self._device)
        
        attn_weights = self.cross_attention.get_attention_weights(facial_t, posture_t)
        
        _, metadata = self.forward(facial_t, posture_t, apply_temporal=False)
        
        return {
            **attn_weights,
            **metadata
        }


def create_fusion_network(config: Optional[FusionConfig] = None,
                          device: str = "cuda") -> FusionNetwork:
    """
    Factory function to create fusion network.
    
    Args:
        config: Fusion configuration.
        device: Computation device.
        
    Returns:
        Initialized FusionNetwork on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    network = FusionNetwork(config=config)
    network.to(device)
    return network
