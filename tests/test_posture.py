# Posture Analysis Module Tests

import pytest
import numpy as np
from unittest.mock import Mock

import sys
sys.path.insert(0, 'c:/sentry')

from src.config import PostureConfig
from src.posture.pose_estimator import PoseResult, PoseLandmark
from src.posture.features import PostureFeatureExtractor, GeometricFeatures, MovementFeatures


def create_mock_pose_result(shoulder_angle: float = 0.0, 
                             head_tilt: float = 0.0) -> PoseResult:
    """Create a mock pose result for testing."""
    # Create landmarks with specified positions
    landmarks = {
        'nose': PoseLandmark(x=0.5, y=0.2, z=0.0, visibility=0.9),
        'left_shoulder': PoseLandmark(x=0.4, y=0.4, z=0.0, visibility=0.9),
        'right_shoulder': PoseLandmark(x=0.6, y=0.4, z=0.0, visibility=0.9),
        'left_hip': PoseLandmark(x=0.4, y=0.6, z=0.0, visibility=0.9),
        'right_hip': PoseLandmark(x=0.6, y=0.6, z=0.0, visibility=0.9),
        'left_ear': PoseLandmark(x=0.45, y=0.15, z=0.0, visibility=0.8),
        'right_ear': PoseLandmark(x=0.55, y=0.15, z=0.0, visibility=0.8),
    }
    
    # Create raw landmarks array
    raw_landmarks = np.zeros((33, 4))
    for i, name in enumerate(['nose', 'left_shoulder', 'right_shoulder', 
                              'left_hip', 'right_hip', 'left_ear', 'right_ear']):
        if name in landmarks:
            lm = landmarks[name]
            raw_landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
    
    return PoseResult(
        landmarks=landmarks,
        raw_landmarks=raw_landmarks,
        world_landmarks=None,
        confidence=0.9
    )


class TestPostureFeatureExtractor:
    """Tests for posture feature extraction."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = PostureFeatureExtractor()
        
        assert extractor._prev_landmarks is None
        assert len(extractor._movement_buffer) == 0
    
    def test_extract_geometric(self):
        """Test geometric feature extraction."""
        extractor = PostureFeatureExtractor()
        pose = create_mock_pose_result()
        
        features = extractor.extract_geometric(pose)
        
        assert isinstance(features, GeometricFeatures)
        assert isinstance(features.shoulder_angle, float)
        assert isinstance(features.head_tilt, float)
        assert isinstance(features.spine_curvature, float)
    
    def test_extract_movement_first_frame(self):
        """Test movement extraction on first frame."""
        extractor = PostureFeatureExtractor()
        pose = create_mock_pose_result()
        
        features = extractor.extract_movement(pose)
        
        assert isinstance(features, MovementFeatures)
        assert features.total_movement == 0.0  # First frame, no movement
        assert features.stillness_level == 1.0
    
    def test_extract_movement_with_motion(self):
        """Test movement extraction with actual motion."""
        extractor = PostureFeatureExtractor()
        
        # First frame
        pose1 = create_mock_pose_result()
        extractor.extract_movement(pose1)
        
        # Second frame with different positions
        pose2_landmarks = {
            'nose': PoseLandmark(x=0.52, y=0.22, z=0.0, visibility=0.9),
            'left_shoulder': PoseLandmark(x=0.42, y=0.42, z=0.0, visibility=0.9),
            'right_shoulder': PoseLandmark(x=0.62, y=0.42, z=0.0, visibility=0.9),
            'left_hip': PoseLandmark(x=0.4, y=0.6, z=0.0, visibility=0.9),
            'right_hip': PoseLandmark(x=0.6, y=0.6, z=0.0, visibility=0.9),
            'left_ear': PoseLandmark(x=0.47, y=0.17, z=0.0, visibility=0.8),
            'right_ear': PoseLandmark(x=0.57, y=0.17, z=0.0, visibility=0.8),
        }
        
        raw = np.zeros((33, 4))
        for i, name in enumerate(['nose', 'left_shoulder', 'right_shoulder']):
            if name in pose2_landmarks:
                lm = pose2_landmarks[name]
                raw[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        pose2 = PoseResult(
            landmarks=pose2_landmarks,
            raw_landmarks=raw,
            world_landmarks=None,
            confidence=0.9
        )
        
        features = extractor.extract_movement(pose2)
        
        assert features.total_movement > 0.0
    
    def test_classify_archetype(self):
        """Test posture archetype classification."""
        extractor = PostureFeatureExtractor()
        
        # Upright posture
        geometric = GeometricFeatures(
            shoulder_angle=0.0,
            head_tilt=0.0,
            spine_curvature=5.0,
            chest_openness=0.25,
            head_forward=0.0,
            shoulder_asymmetry=0.0
        )
        
        movement = MovementFeatures(
            total_movement=0.1,
            upper_body_movement=0.05,
            head_movement=0.02,
            stillness_level=0.5,
            fidgeting_score=0.1
        )
        
        state = extractor.classify_archetype(geometric, movement)
        
        assert state.archetype in extractor.ARCHETYPES
        assert 0 <= state.confidence <= 1
        assert 'spine_straight' in state.features
    
    def test_get_feature_vector(self):
        """Test flat feature vector generation."""
        extractor = PostureFeatureExtractor()
        pose = create_mock_pose_result()
        
        vector = extractor.get_feature_vector(pose)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 15  # 6 geometric + 5 movement + 4 archetype one-hot
    
    def test_reset(self):
        """Test resetting extractor state."""
        extractor = PostureFeatureExtractor()
        pose = create_mock_pose_result()
        
        extractor.extract_movement(pose)
        assert extractor._prev_landmarks is not None
        
        extractor.reset()
        
        assert extractor._prev_landmarks is None
        assert len(extractor._movement_buffer) == 0


class TestPostureTemporalModelArchitecture:
    """Tests for temporal model architecture."""
    
    def test_config_defaults(self):
        """Test posture config defaults."""
        config = PostureConfig()
        
        assert len(config.tcn_channels) == 3
        assert config.lstm_hidden_size == 128
        assert config.lstm_num_layers == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
