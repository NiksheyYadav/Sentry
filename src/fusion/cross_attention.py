# Cross-Attention Fusion Module
# Bidirectional attention between facial and posture modalities

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from ..config import FusionConfig


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention layer."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query tensor (B, dim).
            key: Key tensor (B, dim).
            value: Value tensor (B, dim).
            
        Returns:
            Attended output (B, dim).
        """
        batch_size = query.size(0)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        attn = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bh,bhd->bhd', attn, v)
        out = out.reshape(batch_size, self.dim)
        out = self.out_proj(out)
        
        return out


class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion for facial and posture features.
    
    Implements bidirectional attention where:
    - Facial features query posture features
    - Posture features query facial features
    
    This allows each modality to inform interpretation of the other.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize cross-attention fusion.
        
        Args:
            config: Fusion configuration.
        """
        super().__init__()
        self.config = config or FusionConfig()
        
        # Encoders to project to common dimension
        self.facial_encoder = nn.Sequential(
            nn.Linear(1280, 768),  # From emotion embedding
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(768, self.config.facial_embed_dim),
            nn.LayerNorm(self.config.facial_embed_dim)
        )
        
        self.posture_encoder = nn.Sequential(
            nn.Linear(512, 384),  # From temporal model embedding
            nn.LayerNorm(384),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(384, self.config.posture_embed_dim),
            nn.LayerNorm(self.config.posture_embed_dim)
        )
        
        # Cross-attention layers
        self.face_to_posture_attn = MultiHeadCrossAttention(
            dim=self.config.facial_embed_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout
        )
        
        self.posture_to_face_attn = MultiHeadCrossAttention(
            dim=self.config.posture_embed_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout
        )
        
        # Layer norms for residual connections
        self.facial_norm = nn.LayerNorm(self.config.facial_embed_dim)
        self.posture_norm = nn.LayerNorm(self.config.posture_embed_dim)
        
        # Gating mechanism for weighted combination
        self.face_gate = nn.Sequential(
            nn.Linear(self.config.facial_embed_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self.posture_gate = nn.Sequential(
            nn.Linear(self.config.posture_embed_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self._device = "cpu"
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def forward(self, facial_features: torch.Tensor, 
                posture_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with bidirectional cross-attention.
        
        Args:
            facial_features: Facial embedding (B, 1280).
            posture_features: Posture embedding (B, 512).
            
        Returns:
            Tuple of (attended_facial, attended_posture).
        """
        # Encode to common dimensions
        facial_enc = self.facial_encoder(facial_features)
        posture_enc = self.posture_encoder(posture_features)
        
        # Cross-attention: facial queries posture
        facial_attended = self.face_to_posture_attn(
            facial_enc, posture_enc, posture_enc
        )
        facial_attended = self.facial_norm(facial_enc + facial_attended)
        
        # Cross-attention: posture queries facial
        posture_attended = self.posture_to_face_attn(
            posture_enc, facial_enc, facial_enc
        )
        posture_attended = self.posture_norm(posture_enc + posture_attended)
        
        # Gated combination with original
        face_gate_input = torch.cat([facial_enc, facial_attended], dim=-1)
        face_weight = self.face_gate(face_gate_input)
        facial_out = face_weight * facial_attended + (1 - face_weight) * facial_enc
        
        posture_gate_input = torch.cat([posture_enc, posture_attended], dim=-1)
        posture_weight = self.posture_gate(posture_gate_input)
        posture_out = posture_weight * posture_attended + (1 - posture_weight) * posture_enc
        
        return facial_out, posture_out
    
    def get_attention_weights(self, facial_features: torch.Tensor,
                               posture_features: torch.Tensor) -> dict:
        """
        Get attention weights for interpretability.
        
        Args:
            facial_features: Facial embedding.
            posture_features: Posture embedding.
            
        Returns:
            Dictionary with attention weights.
        """
        self.eval()
        
        with torch.no_grad():
            facial_enc = self.facial_encoder(facial_features)
            posture_enc = self.posture_encoder(posture_features)
            
            # Get gate weights
            _, attended_facial = self.face_to_posture_attn(
                facial_enc, posture_enc, posture_enc
            ), None
            _, attended_posture = self.posture_to_face_attn(
                posture_enc, facial_enc, facial_enc
            ), None
            
            facial_out, posture_out = self.forward(facial_features, posture_features)
            
            face_gate_input = torch.cat([facial_enc, facial_out], dim=-1)
            face_weight = self.face_gate(face_gate_input)
            
            posture_gate_input = torch.cat([posture_enc, posture_out], dim=-1)
            posture_weight = self.posture_gate(posture_gate_input)
        
        return {
            'facial_cross_attn_weight': float(face_weight.mean()),
            'posture_cross_attn_weight': float(posture_weight.mean())
        }
