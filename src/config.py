# Mental Health Assessment Framework - Configuration

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import yaml


@dataclass
class VideoConfig:
    """Video capture and processing configuration."""
    camera_id: int = 0
    capture_fps: int = 30
    process_fps: int = 30  # Process all frames at 30 FPS
    frame_width: int = 640
    frame_height: int = 480
    buffer_size: int = 300  # 10 seconds at 30 FPS
    

@dataclass
class FacialConfig:
    """Facial analysis configuration."""
    mtcnn_thresholds: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.7])
    face_padding: float = 0.2  # 20% padding around face
    min_face_size: int = 40
    emotion_classes: List[str] = field(default_factory=lambda: [
        'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
    ])
    embedding_dim: int = 1280  # MobileNetV3 penultimate layer
    action_units: List[str] = field(default_factory=lambda: [
        'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 
        'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 
        'AU23', 'AU25', 'AU26', 'AU45'
    ])


@dataclass
class PostureConfig:
    """Posture analysis configuration."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    # Key landmark indices for feature extraction
    key_landmarks: Dict[str, int] = field(default_factory=lambda: {
        'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_hip': 23, 'right_hip': 24, 'left_ear': 7, 'right_ear': 8
    })
    # Temporal model settings
    tcn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    tcn_kernel_size: int = 3
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2


@dataclass
class FusionConfig:
    """Multimodal fusion configuration."""
    facial_embed_dim: int = 512
    posture_embed_dim: int = 512
    fused_dim: int = 1024
    attention_heads: int = 8
    dropout: float = 0.1


@dataclass
class PredictionConfig:
    """Prediction and alert configuration."""
    stress_levels: List[str] = field(default_factory=lambda: ['low', 'moderate', 'high'])
    depression_levels: List[str] = field(default_factory=lambda: [
        'minimal', 'mild', 'moderate', 'severe'
    ])
    anxiety_levels: List[str] = field(default_factory=lambda: [
        'minimal', 'mild', 'moderate', 'severe'
    ])
    # Confidence calibration
    temperature: float = 1.5
    mc_dropout_samples: int = 10
    # Alert thresholds
    high_severity_threshold: float = 0.7
    high_confidence_threshold: float = 0.8
    alert_cooldown_seconds: int = 300  # 5 minutes between alerts


@dataclass
class Config:
    """Main configuration container."""
    video: VideoConfig = field(default_factory=VideoConfig)
    facial: FacialConfig = field(default_factory=FacialConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    
    # Paths
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    use_fp16: bool = True  # Mixed precision inference
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()
        if 'video' in data:
            config.video = VideoConfig(**data['video'])
        if 'facial' in data:
            config.facial = FacialConfig(**data['facial'])
        if 'posture' in data:
            config.posture = PostureConfig(**data['posture'])
        if 'fusion' in data:
            config.fusion = FusionConfig(**data['fusion'])
        if 'prediction' in data:
            config.prediction = PredictionConfig(**data['prediction'])
        if 'models_dir' in data:
            config.models_dir = Path(data['models_dir'])
        if 'logs_dir' in data:
            config.logs_dir = Path(data['logs_dir'])
        if 'device' in data:
            config.device = data['device']
        if 'use_fp16' in data:
            config.use_fp16 = data['use_fp16']
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'video': self.video.__dict__,
            'facial': self.facial.__dict__,
            'posture': self.posture.__dict__,
            'fusion': self.fusion.__dict__,
            'prediction': self.prediction.__dict__,
            'models_dir': str(self.models_dir),
            'logs_dir': str(self.logs_dir),
            'device': self.device,
            'use_fp16': self.use_fp16,
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Default configuration instance
default_config = Config()
