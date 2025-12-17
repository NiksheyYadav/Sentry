# Posture Feature Extraction Module
# Geometric and movement features from pose landmarks

import numpy as np
from typing import Optional, Dict, List, Deque
from collections import deque
from dataclasses import dataclass
import cv2

from ..config import PostureConfig
from .pose_estimator import PoseResult, PoseLandmark


@dataclass
class GeometricFeatures:
    """Geometric posture features."""
    shoulder_angle: float  # Shoulder alignment (degrees from horizontal)
    head_tilt: float  # Head tilt angle (degrees)
    spine_curvature: float  # Shoulder-hip alignment indicator
    chest_openness: float  # Distance between shoulders (normalized)
    head_forward: float  # Head position relative to shoulders
    shoulder_asymmetry: float  # Left-right shoulder height difference


@dataclass
class MovementFeatures:
    """Movement-based features."""
    total_movement: float  # Sum of all landmark movements
    upper_body_movement: float  # Shoulder/arm movements
    head_movement: float  # Head/neck movements
    stillness_level: float  # Inverse of movement (0-1)
    fidgeting_score: float  # High-frequency, low-amplitude movements


@dataclass
class PosturalState:
    """Overall postural state classification."""
    archetype: str  # 'upright', 'relaxed', 'defensive', 'collapsed'
    confidence: float
    features: Dict[str, float]


class PostureFeatureExtractor:
    """
    Extracts biomechanically meaningful features from pose landmarks.
    
    Features are invariant to camera distance and angle for
    robust posture analysis.
    """
    
    # Posture archetypes
    ARCHETYPES = ['upright_engaged', 'relaxed_reclined', 'defensive_closed', 'collapsed_withdrawn']
    
    def __init__(self, config: Optional[PostureConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Posture analysis configuration.
        """
        self.config = config or PostureConfig()
        
        # Movement tracking buffers
        self._prev_landmarks: Optional[np.ndarray] = None
        self._movement_buffer: Deque[float] = deque(maxlen=30)  # 3 seconds at 10 FPS
        self._position_buffer: Deque[np.ndarray] = deque(maxlen=30)
    
    def extract_geometric(self, result: PoseResult) -> GeometricFeatures:
        """
        Extract geometric features from pose.
        
        Args:
            result: Pose estimation result.
            
        Returns:
            GeometricFeatures dataclass.
        """
        lm = result.landmarks
        
        # Shoulder angle (deviation from horizontal)
        left_shoulder = lm.get('left_shoulder')
        right_shoulder = lm.get('right_shoulder')
        
        if left_shoulder and right_shoulder:
            dy = left_shoulder.y - right_shoulder.y
            dx = left_shoulder.x - right_shoulder.x
            shoulder_angle = np.degrees(np.arctan2(dy, dx))
        else:
            shoulder_angle = 0.0
        
        # Head tilt
        left_ear = lm.get('left_ear')
        right_ear = lm.get('right_ear')
        nose = lm.get('nose')
        
        if left_ear and right_ear:
            dy = left_ear.y - right_ear.y
            dx = left_ear.x - right_ear.x
            head_tilt = np.degrees(np.arctan2(dy, dx))
        else:
            head_tilt = 0.0
        
        # Spine curvature (shoulder midpoint to hip midpoint alignment)
        left_hip = lm.get('left_hip')
        right_hip = lm.get('right_hip')
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            ])
            hip_mid = np.array([
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            ])
            
            # Spine curvature as deviation from vertical
            spine_vec = shoulder_mid - hip_mid
            vertical = np.array([0, -1])
            cos_angle = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
            spine_curvature = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        else:
            spine_curvature = 0.0
        
        # Chest openness (shoulder distance normalized)
        if left_shoulder and right_shoulder:
            chest_openness = np.sqrt(
                (left_shoulder.x - right_shoulder.x) ** 2 +
                (left_shoulder.y - right_shoulder.y) ** 2
            )
        else:
            chest_openness = 0.0
        
        # Head forward position
        if nose and left_shoulder and right_shoulder:
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            head_forward = nose.x - shoulder_mid_x
        else:
            head_forward = 0.0
        
        # Shoulder asymmetry
        if left_shoulder and right_shoulder:
            shoulder_asymmetry = abs(left_shoulder.y - right_shoulder.y)
        else:
            shoulder_asymmetry = 0.0
        
        return GeometricFeatures(
            shoulder_angle=float(shoulder_angle),
            head_tilt=float(head_tilt),
            spine_curvature=float(spine_curvature),
            chest_openness=float(chest_openness),
            head_forward=float(head_forward),
            shoulder_asymmetry=float(shoulder_asymmetry)
        )
    
    def extract_movement(self, result: PoseResult) -> MovementFeatures:
        """
        Extract movement features from pose sequence.
        
        Args:
            result: Current pose estimation result.
            
        Returns:
            MovementFeatures dataclass.
        """
        current_landmarks = result.raw_landmarks[:, :2]  # x, y only
        
        if self._prev_landmarks is None:
            self._prev_landmarks = current_landmarks.copy()
            self._position_buffer.append(current_landmarks.copy())
            return MovementFeatures(
                total_movement=0.0,
                upper_body_movement=0.0,
                head_movement=0.0,
                stillness_level=1.0,
                fidgeting_score=0.0
            )
        
        # Calculate movement
        movement = np.linalg.norm(current_landmarks - self._prev_landmarks, axis=1)
        
        # Weight by visibility
        visibility = result.raw_landmarks[:, 3]
        movement = movement * visibility
        
        total_movement = float(np.sum(movement))
        
        # Upper body movement (shoulders, elbows, wrists)
        upper_indices = [11, 12, 13, 14, 15, 16]
        upper_body_movement = float(np.sum(movement[upper_indices]))
        
        # Head movement (nose, ears, eyes)
        head_indices = [0, 7, 8, 2, 5]
        head_movement = float(np.sum(movement[head_indices]))
        
        # Track movement over time
        self._movement_buffer.append(total_movement)
        self._position_buffer.append(current_landmarks.copy())
        
        # Stillness level (inverse of average movement)
        avg_movement = np.mean(self._movement_buffer)
        stillness_level = 1.0 / (1.0 + avg_movement * 50)
        
        # Fidgeting score (high frequency, low amplitude movements)
        fidgeting_score = self._compute_fidgeting()
        
        # Update previous landmarks
        self._prev_landmarks = current_landmarks.copy()
        
        return MovementFeatures(
            total_movement=total_movement,
            upper_body_movement=upper_body_movement,
            head_movement=head_movement,
            stillness_level=float(stillness_level),
            fidgeting_score=fidgeting_score
        )
    
    def _compute_fidgeting(self) -> float:
        """
        Compute fidgeting score from position buffer.
        
        High fidgeting = frequent small movements (oscillation).
        """
        if len(self._position_buffer) < 10:
            return 0.0
        
        positions = np.array(list(self._position_buffer))
        
        # Calculate velocity
        velocity = np.diff(positions, axis=0)
        
        # Fidgeting = high velocity variance with low total displacement
        velocity_magnitude = np.linalg.norm(velocity, axis=(1, 2))
        
        # Number of direction changes
        velocity_signs = np.sign(velocity[:, :, 0])  # X direction
        direction_changes = np.sum(np.abs(np.diff(velocity_signs, axis=0)))
        
        # Normalize
        fidgeting = float(direction_changes) / (len(velocity_signs) * 33 + 1)
        
        return float(np.clip(fidgeting, 0, 1))
    
    def classify_archetype(self, geometric: GeometricFeatures, 
                           movement: MovementFeatures) -> PosturalState:
        """
        Classify postural archetype.
        
        Args:
            geometric: Geometric features.
            movement: Movement features.
            
        Returns:
            PosturalState with classification.
        """
        features = {
            'spine_straight': 1.0 - min(abs(geometric.spine_curvature) / 45, 1),
            'chest_open': min(geometric.chest_openness / 0.3, 1),
            'head_up': 1.0 - min(abs(geometric.head_tilt) / 30, 1),
            'stillness': movement.stillness_level,
            'fidgeting': movement.fidgeting_score
        }
        
        # Score each archetype
        scores = {
            'upright_engaged': (
                features['spine_straight'] * 0.4 +
                features['chest_open'] * 0.3 +
                features['head_up'] * 0.2 +
                (1 - features['stillness']) * 0.1
            ),
            'relaxed_reclined': (
                (1 - features['spine_straight']) * 0.3 +
                features['chest_open'] * 0.2 +
                features['stillness'] * 0.3 +
                (1 - features['fidgeting']) * 0.2
            ),
            'defensive_closed': (
                (1 - features['chest_open']) * 0.4 +
                features['fidgeting'] * 0.3 +
                (1 - features['head_up']) * 0.3
            ),
            'collapsed_withdrawn': (
                (1 - features['spine_straight']) * 0.4 +
                (1 - features['chest_open']) * 0.3 +
                features['stillness'] * 0.2 +
                (1 - features['head_up']) * 0.1
            )
        }
        
        archetype = max(scores, key=scores.get)
        confidence = scores[archetype]
        
        return PosturalState(
            archetype=archetype,
            confidence=float(confidence),
            features=features
        )
    
    def get_feature_vector(self, result: PoseResult) -> np.ndarray:
        """
        Get flat feature vector for model input.
        
        Args:
            result: Pose estimation result.
            
        Returns:
            Feature vector array.
        """
        geometric = self.extract_geometric(result)
        movement = self.extract_movement(result)
        state = self.classify_archetype(geometric, movement)
        
        # Combine all features
        features = [
            geometric.shoulder_angle / 90,  # Normalize
            geometric.head_tilt / 90,
            geometric.spine_curvature / 90,
            geometric.chest_openness,
            geometric.head_forward,
            geometric.shoulder_asymmetry,
            movement.total_movement,
            movement.upper_body_movement,
            movement.head_movement,
            movement.stillness_level,
            movement.fidgeting_score,
            *[float(state.archetype == arch) for arch in self.ARCHETYPES]
        ]
        
        return np.array(features, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset tracking state."""
        self._prev_landmarks = None
        self._movement_buffer.clear()
        self._position_buffer.clear()
