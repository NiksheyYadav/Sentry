# Posture Temporal Model
# TCN-LSTM hybrid for temporal pattern recognition

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ..config import PostureConfig


@dataclass
class TemporalPostureResult:
    """Temporal posture analysis result."""
    pattern_embedding: np.ndarray  # 512-dim embedding
    posture_trajectory: str  # 'stable', 'deteriorating', 'improving'
    stillness_duration: float  # Consecutive seconds of stillness
    state_transitions: int  # Number of archetype changes
    prediction_confidence: float


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Block with dilated convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder with dilated convolutions."""
    
    def __init__(self, input_dim: int, channels: List[int], kernel_size: int = 3):
        super().__init__()
        
        layers = []
        num_levels = len(channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dilation))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C) - batch, time, channels.
            
        Returns:
            Output tensor of shape (B, T, output_dim).
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Back to (B, T, C)
        return out.transpose(1, 2)


class PostureTemporalModel(nn.Module):
    """
    Hybrid TCN-LSTM model for temporal posture analysis.
    
    TCN captures multi-scale temporal patterns (fidgeting frequency, etc.)
    LSTM models longer-term state transitions.
    """
    
    def __init__(self, config: Optional[PostureConfig] = None):
        """
        Initialize temporal model.
        
        Args:
            config: Posture analysis configuration.
        """
        super().__init__()
        self.config = config or PostureConfig()
        
        # Input dimension (from feature extractor)
        self.input_dim = 15  # Geometric + movement features
        
        # TCN encoder
        self.tcn = TCNEncoder(
            input_dim=self.input_dim,
            channels=self.config.tcn_channels,
            kernel_size=self.config.tcn_kernel_size
        )
        
        # LSTM for long-term dependencies
        self.lstm = nn.LSTM(
            input_size=self.config.tcn_channels[-1],
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            batch_first=True,
            dropout=0.2 if self.config.lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Projection to output embedding
        self.projection = nn.Sequential(
            nn.Linear(self.config.lstm_hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512)
        )
        
        # Trajectory classifier
        self.trajectory_classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # stable, deteriorating, improving
        )
        
        self._device = "cpu"
        self._hidden = None
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)
    
    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state."""
        h0 = torch.zeros(
            self.config.lstm_num_layers, batch_size, 
            self.config.lstm_hidden_size
        ).to(self._device)
        c0 = torch.zeros(
            self.config.lstm_num_layers, batch_size,
            self.config.lstm_hidden_size
        ).to(self._device)
        self._hidden = (h0, c0)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, input_dim).
            hidden: Optional LSTM hidden state.
            
        Returns:
            Tuple of (embedding, trajectory_logits, hidden_state).
        """
        # TCN encoding
        tcn_out = self.tcn(x)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(tcn_out)
        else:
            lstm_out, hidden = self.lstm(tcn_out, hidden)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        # Project to embedding
        embedding = self.projection(last_output)
        
        # Classify trajectory
        trajectory = self.trajectory_classifier(embedding)
        
        return embedding, trajectory, hidden
    
    def process_sequence(self, feature_sequence: np.ndarray) -> TemporalPostureResult:
        """
        Process a sequence of posture features.
        
        Args:
            feature_sequence: Array of shape (T, input_dim).
            
        Returns:
            TemporalPostureResult with analysis.
        """
        self.eval()
        
        # Prepare input
        x = torch.from_numpy(feature_sequence).float()
        x = x.unsqueeze(0).to(self._device)  # Add batch dim
        
        with torch.no_grad():
            embedding, trajectory_logits, _ = self.forward(x)
            trajectory_probs = torch.softmax(trajectory_logits, dim=1)
        
        embedding_np = embedding.cpu().numpy()[0]
        probs_np = trajectory_probs.cpu().numpy()[0]
        
        trajectories = ['stable', 'deteriorating', 'improving']
        trajectory_idx = np.argmax(probs_np)
        
        # Analyze sequence for additional metrics
        stillness_duration = self._compute_stillness(feature_sequence)
        state_transitions = self._count_transitions(feature_sequence)
        
        return TemporalPostureResult(
            pattern_embedding=embedding_np,
            posture_trajectory=trajectories[trajectory_idx],
            stillness_duration=stillness_duration,
            state_transitions=state_transitions,
            prediction_confidence=float(probs_np[trajectory_idx])
        )
    
    def _compute_stillness(self, features: np.ndarray) -> float:
        """Compute consecutive stillness duration."""
        # Assume stillness_level is at index 9
        stillness_col = 9 if features.shape[1] > 9 else -1
        if stillness_col < 0:
            return 0.0
        
        stillness = features[:, stillness_col]
        
        # Count consecutive frames with high stillness
        consecutive = 0
        max_consecutive = 0
        
        for s in stillness:
            if s > 0.7:  # High stillness threshold
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive / 10.0  # Convert to seconds (10 FPS)
    
    def _count_transitions(self, features: np.ndarray) -> int:
        """Count archetype state transitions."""
        # Archetype one-hot starts at index 11
        if features.shape[1] < 15:
            return 0
        
        archetypes = features[:, 11:15]
        current_arch = np.argmax(archetypes, axis=1)
        
        transitions = np.sum(np.abs(np.diff(current_arch)) > 0)
        return int(transitions)
    
    def get_embedding(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Extract only the embedding from a sequence."""
        result = self.process_sequence(feature_sequence)
        return result.pattern_embedding


def create_temporal_model(config: Optional[PostureConfig] = None,
                          device: str = "cuda") -> PostureTemporalModel:
    """
    Factory function to create temporal posture model.
    
    Args:
        config: Posture analysis configuration.
        device: Computation device.
        
    Returns:
        Initialized PostureTemporalModel on specified device.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model = PostureTemporalModel(config=config)
    model.to(device)
    return model
