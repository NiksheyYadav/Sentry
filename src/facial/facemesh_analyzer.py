# FaceMesh-based Expression Analysis Module
# Uses MediaPipe FaceMesh 468 landmarks for accurate expression detection

import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ExpressionScores:
    """Facial expression scores from landmark analysis."""
    # Primary emotions
    neutral: float = 0.0
    happy: float = 0.0
    sad: float = 0.0
    surprise: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    
    # Feature indicators
    eye_openness: float = 0.0  # 0=closed, 1=wide open
    eyebrow_raise: float = 0.0  # 0=normal, 1=raised
    mouth_openness: float = 0.0  # 0=closed, 1=wide open
    smile_score: float = 0.0  # 0=frown, 0.5=neutral, 1=smile
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'neutral': self.neutral,
            'happy': self.happy,
            'sad': self.sad,
            'surprise': self.surprise,
            'fear': self.fear,
            'anger': self.anger,
            'eye_openness': self.eye_openness,
            'eyebrow_raise': self.eyebrow_raise,
            'mouth_openness': self.mouth_openness,
            'smile_score': self.smile_score
        }


class FaceMeshExpressionAnalyzer:
    """
    Analyze facial expressions using MediaPipe FaceMesh 468 landmarks.
    
    Provides accurate detection of:
    - Neutral: Relaxed face, no strong expression
    - Happy: Smile (raised mouth corners)
    - Sad: Frown (lowered mouth corners, lowered eyebrows)
    - Surprise: Wide eyes + open mouth + raised eyebrows
    - Fear: Wide eyes + tense face + raised eyebrows
    - Anger: Lowered eyebrows, tense mouth
    """
    
    # Landmark indices for key facial features
    # Left eye (upper and lower eyelid points for openness calculation)
    LEFT_EYE_TOP = [386, 374, 373, 390]
    LEFT_EYE_BOTTOM = [362, 382, 381, 380]
    LEFT_EYE_VERTICAL = [(386, 374), (387, 373)]  # Upper-lower pairs
    
    # Right eye
    RIGHT_EYE_TOP = [159, 145, 144, 163]
    RIGHT_EYE_BOTTOM = [33, 7, 163, 144]
    RIGHT_EYE_VERTICAL = [(159, 145), (158, 153)]  # Upper-lower pairs
    
    # Key eye points for Eye Aspect Ratio (EAR)
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # P1-P6 for EAR
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # P1-P6 for EAR
    
    # Eyebrows
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    LEFT_EYEBROW_TOP = [336, 334, 300]  # Top points
    RIGHT_EYEBROW_TOP = [70, 105, 107]  # Top points
    
    # Lips/Mouth
    UPPER_LIP_TOP = [0, 267, 269, 270, 13, 82, 81, 37, 39, 40]
    LOWER_LIP_BOTTOM = [17, 314, 405, 321, 375, 84, 181, 91, 146, 61]
    MOUTH_LEFT_CORNER = 61
    MOUTH_RIGHT_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    
    # Reference points for normalization
    LEFT_EYE_OUTER = 263
    RIGHT_EYE_OUTER = 33
    NOSE_TIP = 4
    CHIN = 152
    FOREHEAD_CENTER = 10
    
    def __init__(self, static_image_mode: bool = True, max_faces: int = 1):
        """
        Initialize FaceMesh analyzer.
        
        Args:
            static_image_mode: Process images independently (vs video stream)
            max_faces: Maximum number of faces to detect
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Baseline values (will be calibrated per-face)
        self._baseline_eye_height = None
        self._baseline_mouth_height = None
        self._baseline_eyebrow_height = None
    
    def analyze(self, face_image: np.ndarray) -> Optional[ExpressionScores]:
        """
        Analyze facial expression from a face image.
        
        Args:
            face_image: RGB face image (cropped face)
            
        Returns:
            ExpressionScores or None if no face detected
        """
        # Ensure RGB
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
        
        # Process with FaceMesh
        results = self.face_mesh.process(face_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get landmarks as numpy array
        landmarks = results.multi_face_landmarks[0]
        h, w = face_image.shape[:2]
        
        points = np.array([
            [lm.x * w, lm.y * h, lm.z * w]  # Scale z by width for depth
            for lm in landmarks.landmark
        ])
        
        # Calculate facial metrics
        scores = ExpressionScores()
        
        # === Eye Openness (EAR - Eye Aspect Ratio) ===
        left_ear = self._calculate_ear(points, self.LEFT_EYE_EAR)
        right_ear = self._calculate_ear(points, self.RIGHT_EYE_EAR)
        avg_ear = (left_ear + right_ear) / 2
        
        # Normalize: EAR ~0.2 = closed, ~0.35 = normal, ~0.5+ = wide
        scores.eye_openness = np.clip((avg_ear - 0.15) / 0.35, 0, 1)
        
        # === Eyebrow Raise ===
        eyebrow_raise = self._calculate_eyebrow_raise(points)
        scores.eyebrow_raise = eyebrow_raise
        
        # === Mouth Openness ===
        mouth_openness = self._calculate_mouth_openness(points)
        scores.mouth_openness = mouth_openness
        
        # === Smile Score (mouth corner analysis) ===
        smile_score = self._calculate_smile(points)
        scores.smile_score = smile_score
        
        # === Calculate Expression Probabilities ===
        
        # SURPRISE: Wide eyes + Open mouth + Raised eyebrows
        if scores.eye_openness > 0.6 and scores.mouth_openness > 0.4 and scores.eyebrow_raise > 0.4:
            scores.surprise = min(0.95, 0.4 + scores.eye_openness * 0.3 + scores.mouth_openness * 0.2 + scores.eyebrow_raise * 0.2)
        elif scores.eye_openness > 0.5 and scores.mouth_openness > 0.3:
            scores.surprise = min(0.7, 0.2 + scores.eye_openness * 0.3 + scores.mouth_openness * 0.2)
        
        # FEAR: Wide eyes + Raised eyebrows + Tense mouth (not fully open)
        if scores.eye_openness > 0.55 and scores.eyebrow_raise > 0.4 and 0.2 < scores.mouth_openness < 0.6:
            scores.fear = min(0.85, 0.3 + scores.eye_openness * 0.3 + scores.eyebrow_raise * 0.3)
        
        # HAPPY: Smile
        if scores.smile_score > 0.55:
            scores.happy = min(0.95, 0.3 + (scores.smile_score - 0.5) * 1.5)
        
        # SAD: Low eyebrows + Frown (low smile score) + Normal/closed eyes
        if scores.smile_score < 0.4 and scores.eyebrow_raise < 0.3 and scores.eye_openness < 0.5:
            scores.sad = min(0.85, 0.3 + (0.5 - scores.smile_score) * 0.6 + (0.3 - scores.eyebrow_raise) * 0.3)
        
        # ANGER: Low eyebrows + Tense mouth + Normal eyes
        if scores.eyebrow_raise < 0.2 and scores.smile_score < 0.45 and 0.3 < scores.eye_openness < 0.6:
            scores.anger = min(0.8, 0.3 + (0.2 - scores.eyebrow_raise) * 1.5)
        
        # NEUTRAL: No strong expression
        max_expression = max(scores.surprise, scores.fear, scores.happy, scores.sad, scores.anger)
        if max_expression < 0.4:
            scores.neutral = 0.7 - max_expression
        else:
            scores.neutral = max(0.05, 0.4 - max_expression * 0.5)
        
        return scores
    
    def _calculate_ear(self, points: np.ndarray, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        Higher = more open
        """
        p1, p2, p3, p4, p5, p6 = [points[i][:2] for i in eye_indices]
        
        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)
        
        if h < 1:
            return 0.3  # Default
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_eyebrow_raise(self, points: np.ndarray) -> float:
        """
        Calculate eyebrow raise relative to eyes.
        
        Uses distance from eyebrow to eye relative to face height.
        """
        # Get eye center points
        left_eye_center = np.mean([points[i][:2] for i in [263, 362]], axis=0)
        right_eye_center = np.mean([points[i][:2] for i in [33, 133]], axis=0)
        
        # Get eyebrow top points
        left_brow_top = np.mean([points[i][:2] for i in self.LEFT_EYEBROW_TOP], axis=0)
        right_brow_top = np.mean([points[i][:2] for i in self.RIGHT_EYEBROW_TOP], axis=0)
        
        # Face height for normalization
        chin = points[self.CHIN][:2]
        forehead = points[self.FOREHEAD_CENTER][:2]
        face_height = np.linalg.norm(forehead - chin)
        
        if face_height < 10:
            return 0.3
        
        # Distance from eyebrow to eye (negative y = higher)
        left_dist = left_eye_center[1] - left_brow_top[1]
        right_dist = right_eye_center[1] - right_brow_top[1]
        avg_dist = (left_dist + right_dist) / 2
        
        # Normalize by face height (typical range: 0.08 - 0.15 of face height)
        normalized = avg_dist / face_height
        
        # Map to 0-1 scale (0.08 = low, 0.12 = normal, 0.16+ = raised)
        raise_score = np.clip((normalized - 0.08) / 0.08, 0, 1)
        
        return raise_score
    
    def _calculate_mouth_openness(self, points: np.ndarray) -> float:
        """
        Calculate mouth openness (vertical distance between lips).
        """
        # Upper and lower lip centers
        upper_lip = points[self.UPPER_LIP_CENTER][:2]
        lower_lip = points[self.LOWER_LIP_CENTER][:2]
        
        # Mouth corners for width reference
        left_corner = points[self.MOUTH_LEFT_CORNER][:2]
        right_corner = points[self.MOUTH_RIGHT_CORNER][:2]
        
        mouth_width = np.linalg.norm(right_corner - left_corner)
        mouth_height = np.linalg.norm(lower_lip - upper_lip)
        
        if mouth_width < 5:
            return 0.0
        
        # Openness ratio (height / width)
        # Closed: ~0.05, Normal: ~0.15, Open: ~0.4+
        ratio = mouth_height / mouth_width
        
        # Normalize to 0-1
        openness = np.clip((ratio - 0.05) / 0.5, 0, 1)
        
        return openness
    
    def _calculate_smile(self, points: np.ndarray) -> float:
        """
        Calculate smile score based on mouth corner positions.
        
        Smile: Corners up relative to center
        Frown: Corners down relative to center
        Neutral: Corners level with center
        """
        left_corner = points[self.MOUTH_LEFT_CORNER][:2]
        right_corner = points[self.MOUTH_RIGHT_CORNER][:2]
        
        # Use nose tip as reference point
        nose = points[self.NOSE_TIP][:2]
        
        # Average corner height relative to nose
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        
        # Lip center for comparison
        lip_center = points[self.UPPER_LIP_CENTER][:2]
        
        # If corners are ABOVE (lower y) the lip center = smile
        # If corners are BELOW (higher y) the lip center = frown
        corner_height_diff = lip_center[1] - corner_avg_y
        
        # Normalize by mouth width
        mouth_width = np.linalg.norm(right_corner - left_corner)
        if mouth_width < 5:
            return 0.5
        
        normalized_diff = corner_height_diff / mouth_width
        
        # Map to 0-1 (0 = frown, 0.5 = neutral, 1 = smile)
        smile_score = 0.5 + normalized_diff * 2
        smile_score = np.clip(smile_score, 0, 1)
        
        return smile_score
    
    def get_landmark_visualization(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Draw landmarks on face image for debugging."""
        results = self.face_mesh.process(face_image)
        
        if not results.multi_face_landmarks:
            return None
        
        annotated = face_image.copy()
        h, w = face_image.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw key points
            for i, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                
                # Color code by region
                if i in self.LEFT_EYE_EAR + self.RIGHT_EYE_EAR:
                    color = (0, 255, 0)  # Green for eyes
                elif i in self.LEFT_EYEBROW + self.RIGHT_EYEBROW:
                    color = (255, 0, 0)  # Blue for eyebrows
                elif i in self.UPPER_LIP_TOP + self.LOWER_LIP_BOTTOM:
                    color = (0, 0, 255)  # Red for lips
                else:
                    continue  # Skip other points
                
                cv2.circle(annotated, (x, y), 1, color, -1)
        
        return annotated
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


def create_facemesh_analyzer(**kwargs) -> FaceMeshExpressionAnalyzer:
    """Create a FaceMesh expression analyzer with optional custom settings."""
    return FaceMeshExpressionAnalyzer(**kwargs)
