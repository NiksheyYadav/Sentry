# Pose Estimation Module
# MediaPipe Tasks API-based skeletal tracking (for MediaPipe >= 0.10)

import os
import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..config import PostureConfig

# New MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Get the model path - look for it in the models directory
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker.task"


@dataclass
class PoseLandmark:
    """Single pose landmark."""
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    z: float  # Depth (relative)
    visibility: float  # Confidence 0-1


@dataclass
class PoseResult:
    """Pose estimation result."""
    landmarks: Dict[str, PoseLandmark]  # Named landmarks
    raw_landmarks: np.ndarray  # (33, 4) array [x, y, z, visibility]
    world_landmarks: Optional[np.ndarray]  # 3D world coordinates
    confidence: float


class PoseEstimator:
    """
    MediaPipe Tasks API-based pose estimator.
    
    Identifies 33 body landmarks for posture analysis
    with real-time performance.
    
    Updated for MediaPipe >= 0.10 using the new Tasks API.
    """
    
    # MediaPipe landmark indices (same as before)
    LANDMARK_NAMES = {
        0: 'nose',
        1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear',
        9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index'
    }
    
    # Pose connections for drawing skeleton
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    def __init__(self, config: Optional[PostureConfig] = None, model_path: Optional[str] = None):
        """
        Initialize pose estimator.
        
        Args:
            config: Posture analysis configuration.
            model_path: Path to the pose_landmarker.task model file.
                       If not provided, looks in the models directory.
        """
        self.config = config or PostureConfig()
        
        # Determine model path
        if model_path:
            self._model_path = Path(model_path)
        else:
            self._model_path = MODEL_PATH
        
        # Verify model exists
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found at: {self._model_path}\n"
                f"Please download it from:\n"
                f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task\n"
                f"And place it in the models directory."
            )
        
        # Create options for PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=str(self._model_path))
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False
        )
        
        # Create the PoseLandmarker
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
    
    def estimate(self, frame: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from a frame.
        
        Args:
            frame: Input frame (BGR from OpenCV).
            
        Returns:
            PoseResult or None if no pose detected.
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Increment timestamp
        self._frame_timestamp_ms += 33  # ~30fps
        
        # Detect pose
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        # Check if any pose was detected
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        
        # Get the first detected pose
        pose_landmarks = result.pose_landmarks[0]
        
        # Extract landmarks
        landmarks_dict = {}
        raw_landmarks = np.zeros((33, 4))
        
        for idx, landmark in enumerate(pose_landmarks):
            name = self.LANDMARK_NAMES.get(idx, f'landmark_{idx}')
            landmarks_dict[name] = PoseLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            )
            raw_landmarks[idx] = [
                landmark.x, 
                landmark.y, 
                landmark.z, 
                landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            ]
        
        # World landmarks (3D)
        world_landmarks = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_lms = result.pose_world_landmarks[0]
            world_landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in world_lms
            ])
        
        # Overall confidence (average visibility of key landmarks)
        key_indices = [11, 12, 23, 24, 0]  # Shoulders, hips, nose
        confidence = np.mean([raw_landmarks[i, 3] for i in key_indices])
        
        return PoseResult(
            landmarks=landmarks_dict,
            raw_landmarks=raw_landmarks,
            world_landmarks=world_landmarks,
            confidence=float(confidence)
        )
    
    def get_landmark(self, result: PoseResult, name: str) -> Optional[PoseLandmark]:
        """Get a specific landmark by name."""
        return result.landmarks.get(name)
    
    def get_landmark_position(self, result: PoseResult, name: str, 
                              frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get landmark position in pixel coordinates.
        
        Args:
            result: Pose estimation result.
            name: Landmark name.
            frame_shape: (height, width) of frame.
            
        Returns:
            (x, y) pixel coordinates or None.
        """
        landmark = self.get_landmark(result, name)
        if landmark is None:
            return None
        
        h, w = frame_shape
        return (int(landmark.x * w), int(landmark.y * h))
    
    def draw_pose(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Frame to draw on (BGR).
            result: Pose estimation result.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw connections
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_name = self.LANDMARK_NAMES.get(start_idx)
            end_name = self.LANDMARK_NAMES.get(end_idx)
            
            if start_name and end_name:
                start_lm = result.landmarks.get(start_name)
                end_lm = result.landmarks.get(end_name)
                
                if start_lm and end_lm and start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    start_point = (int(start_lm.x * w), int(start_lm.y * h))
                    end_point = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(annotated, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for name, lm in result.landmarks.items():
            if lm.visibility > 0.5:
                point = (int(lm.x * w), int(lm.y * h))
                cv2.circle(annotated, point, 4, (255, 0, 0), -1)
        
        return annotated
    
    def close(self) -> None:
        """Release resources."""
        if hasattr(self, '_landmarker'):
            self._landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
