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
    """Temporal posture analysis result - embedding only.
    
    Note: Trajectory, posture archetype, and stress indicator predictions
    are now handled by MentalHealthClassifier post-fusion.
    """
    pattern_embedding: np.ndarray  # 512-dim embedding
    stillness_duration: float  # Consecutive seconds of stillness (heuristic)
    state_transitions: int  # Number of archetype changes (heuristic)


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
        
        # Input dimension - 100 for MultiPosture dataset, 15 for live feature extractor
        self.input_dim = getattr(self.config, 'input_dim', 100)
        
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
        
        # Projection to output embedding with LayerNorm for stability
        self.projection = nn.Sequential(
            nn.Linear(self.config.lstm_hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # NOTE: Classifiers (trajectory, posture, stress) moved to MentalHealthClassifier
        # post-fusion for unified predictions using both facial and posture features
        
        # Initialize weights properly
        self._init_weights()
        
        self._device = "cpu"
        self._hidden = None
    
    def _init_weights(self):
        """Initialize weights with small values to prevent exploding gradients."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
               ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass - extracts embedding only.
        
        Args:
            x: Input tensor of shape (B, T, input_dim).
            hidden: Optional LSTM hidden state.
            
        Returns:
            Tuple of (embedding, hidden_state).
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
        
        return embedding, hidden
    
    def process_sequence(self, feature_sequence: np.ndarray) -> TemporalPostureResult:
        """
        Process a sequence of posture features.
        
        Args:
            feature_sequence: Array of shape (T, input_dim).
            
        Returns:
            TemporalPostureResult with analysis.
        """
        self.eval()
        
        # Handle dimension mismatch: pad features if needed
        seq_features = feature_sequence.shape[1] if len(feature_sequence.shape) > 1 else feature_sequence.shape[0]
        if seq_features < self.input_dim:
            # Pad with zeros to match expected input dimension
            padding = np.zeros((feature_sequence.shape[0], self.input_dim - seq_features))
            feature_sequence = np.concatenate([feature_sequence, padding], axis=1)
        elif seq_features > self.input_dim:
            # Truncate to expected dimension
            feature_sequence = feature_sequence[:, :self.input_dim]
        
        # Prepare input
        x = torch.from_numpy(feature_sequence).float()
        x = x.unsqueeze(0).to(self._device)  # Add batch dim
        
        with torch.no_grad():
            embedding, _ = self.forward(x)
        
        embedding_np = embedding.cpu().numpy()[0]
        
        # Analyze sequence for additional heuristic metrics
        stillness_duration = self._compute_stillness(feature_sequence)
        state_transitions = self._count_transitions(feature_sequence)
        
        return TemporalPostureResult(
            pattern_embedding=embedding_np,
            stillness_duration=stillness_duration,
            state_transitions=state_transitions
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
                          device: str = "cuda",
                          checkpoint_path: Optional[str] = None) -> PostureTemporalModel:
    """
    Factory function to create temporal posture model.
    
    Args:
        config: Posture analysis configuration.
        device: Computation device.
        checkpoint_path: Optional path to trained model checkpoint.
        
    Returns:
        Initialized PostureTemporalModel on specified device.
    """
    import os
    
    device = device if torch.cuda.is_available() else "cpu"
    
    # Check for trained model
    if checkpoint_path is None:
        # Try default location
        default_path = "models/posture_trained/best_model.pth"
        if os.path.exists(default_path):
            checkpoint_path = default_path
            print(f"Found trained posture model: {checkpoint_path}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load checkpoint and extract input dimension
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Extract input_dim from first conv layer
            checkpoint_input_dim = 15  # default
            for key in state_dict:
                if 'tcn.network.0.conv1.weight' in key:
                    checkpoint_input_dim = state_dict[key].shape[1]
                    break
            
            # Validate against config
            requested_input_dim = config.input_dim if config else 15
            
            if checkpoint_input_dim != requested_input_dim:
                print(f"Warning: Model checkpoint has {checkpoint_input_dim} features, but {requested_input_dim} were requested.")
                print(f"Skipping incompatible model loading. Using untrained model instead.")
                model = PostureTemporalModel(config=config)
                model.to(device)
            else:
                print(f"Loading trained posture model (input_dim={checkpoint_input_dim})")
                model = PostureTemporalModel(config=config)
                model.load_state_dict(state_dict)
                model.to(device)
                print(f"Loaded trained posture model from {checkpoint_path}")
                
        except Exception as e:
            print(f"Error loading posture checkpoint: {e}")
            model = PostureTemporalModel(config=config)
            model.to(device)
    else:
        # Use untrained model
        model = PostureTemporalModel(config=config)
        model.to(device)
        if checkpoint_path:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return model
