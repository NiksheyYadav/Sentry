# Emotion Post-Processing Module
# Corrects model predictions using MediaPipe FaceMesh landmarks and temporal smoothing

import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque

# Import FaceMesh analyzer
from .facemesh_analyzer import FaceMeshExpressionAnalyzer, ExpressionScores


@dataclass
class PostProcessedEmotion:
    """Post-processed emotion result."""
    raw_emotion: str
    raw_confidence: float
    raw_probabilities: Dict[str, float]
    
    final_emotion: str
    final_confidence: float
    final_probabilities: Dict[str, float]
    
    correction_applied: bool
    correction_reason: str = ""


class FacialExpressionAnalyzer:
    """
    Analyze facial features to detect expressions beyond model predictions.
    
    Detects:
    - Smile (mouth curvature up, teeth visible)
    - Surprise (open mouth, raised eyebrows)
    - Neutral (relaxed features)
    - Frown (mouth curvature down)
    """
    
    def __init__(self):
        # Load Haar cascades
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def analyze(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze face image to get expression indicators.
        
        Args:
            face_image: RGB face image (cropped)
            
        Returns:
            Dict with expression scores: smile, surprise, neutral, frown
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        
        h, w = gray.shape[:2]
        
        # Analyze different regions
        scores = {
            'smile': 0.0,
            'surprise': 0.0,
            'neutral': 0.3,  # Default baseline
            'frown': 0.0,
            'open_mouth': 0.0,
            'wide_eyes': 0.0
        }
        
        # === Mouth Analysis ===
        mouth_region = gray[int(h*0.55):, int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            # Smile detection
            smiles = self.smile_cascade.detectMultiScale(
                mouth_region, scaleFactor=1.7, minNeighbors=15, minSize=(20, 20)
            )
            if len(smiles) > 0:
                scores['smile'] = min(0.9, 0.4 + 0.15 * len(smiles))
            
            # Open mouth detection (for surprise)
            mouth_openness = self._detect_open_mouth(mouth_region)
            scores['open_mouth'] = mouth_openness
            if mouth_openness > 0.5:
                scores['surprise'] += 0.4
            
            # Mouth curvature analysis
            curvature = self._analyze_mouth_curvature(mouth_region)
            if curvature > 0.2:
                scores['smile'] = max(scores['smile'], curvature * 0.7)
            elif curvature < -0.2:
                scores['frown'] = abs(curvature) * 0.7
        
        # === Eye Analysis ===
        eye_region = gray[:int(h*0.55), :]
        if eye_region.size > 0:
            eyes = self.eye_cascade.detectMultiScale(
                eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            
            # Wide eyes detection (for surprise)
            eye_openness = self._detect_wide_eyes(eye_region, eyes)
            scores['wide_eyes'] = eye_openness
            if eye_openness > 0.5:
                scores['surprise'] += 0.4
        
        # === Combine for final surprise score ===
        if scores['open_mouth'] > 0.4 and scores['wide_eyes'] > 0.4:
            scores['surprise'] = min(0.9, scores['surprise'] + 0.2)
        
        # === Neutral detection ===
        # If no strong expression detected, likely neutral
        max_expression = max(scores['smile'], scores['surprise'], scores['frown'])
        if max_expression < 0.3:
            scores['neutral'] = 0.6
        else:
            scores['neutral'] = max(0.1, 0.5 - max_expression)
        
        return scores
    
    def _detect_open_mouth(self, mouth_region: np.ndarray) -> float:
        """Detect if mouth is open (for surprise, fear, or speech)."""
        if mouth_region.size < 100:
            return 0.0
        
        # Apply threshold to find dark regions (open mouth appears dark)
        _, binary = cv2.threshold(mouth_region, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of dark regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest dark region (likely open mouth)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        total_area = mouth_region.shape[0] * mouth_region.shape[1]
        
        # Open mouth typically covers 15-40% of mouth region
        ratio = area / total_area
        
        if 0.1 < ratio < 0.5:
            # Check aspect ratio (open mouth is taller than wide)
            x, y, w, h = cv2.boundingRect(largest)
            aspect = h / (w + 1)
            
            if aspect > 0.5:  # More vertical = more open
                return min(0.9, ratio * 2 + aspect * 0.3)
        
        return max(0.0, ratio * 1.5 - 0.1)
    
    def _detect_wide_eyes(self, eye_region: np.ndarray, detected_eyes: List) -> float:
        """Detect if eyes are wide open (surprise, fear)."""
        if len(detected_eyes) < 1:
            return 0.0
        
        # Calculate average eye height relative to region
        total_height = 0
        for (x, y, w, h) in detected_eyes[:2]:  # Max 2 eyes
            total_height += h
        
        avg_height = total_height / min(2, len(detected_eyes))
        region_height = eye_region.shape[0]
        
        # Eyes taking up more vertical space = wider open
        ratio = avg_height / region_height
        
        # Normal eyes ~15-20% of upper face, wide eyes ~25-35%
        if ratio > 0.2:
            return min(0.9, (ratio - 0.15) * 3)
        
        return 0.0
    
    def _analyze_mouth_curvature(self, mouth_region: np.ndarray) -> float:
        """Analyze mouth curvature: positive = smile, negative = frown."""
        if mouth_region.size < 100:
            return 0.0
        
        # Apply edge detection
        edges = cv2.Canny(mouth_region, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 10:
            return 0.0
        
        points = largest.reshape(-1, 2)
        h, w = mouth_region.shape[:2]
        
        # Sample left, center, right y-positions
        left_mask = points[:, 0] < w * 0.3
        right_mask = points[:, 0] > w * 0.7
        center_mask = (points[:, 0] > w * 0.4) & (points[:, 0] < w * 0.6)
        
        if not (left_mask.any() and right_mask.any() and center_mask.any()):
            return 0.0
        
        left_y = np.mean(points[left_mask, 1])
        right_y = np.mean(points[right_mask, 1])
        center_y = np.mean(points[center_mask, 1])
        
        # Corners higher than center = frown, lower = smile
        corner_avg = (left_y + right_y) / 2
        curvature = (center_y - corner_avg) / h
        
        return np.clip(curvature * 3, -1, 1)


class EmotionPostProcessor:
    """
    Post-processor for emotion predictions.
    
    Applies:
    1. Facial expression analysis (smile, surprise, frown detection)
    2. Expression-based correction (override model when visual evidence is strong)
    3. Temporal smoothing (weighted moving average)
    4. Hysteresis to prevent flickering
    """
    
    def __init__(
        self,
        temporal_window: int = 15,
        expression_override_threshold: float = 0.6,
        hysteresis_threshold: int = 5,
        min_switch_confidence: float = 0.45
    ):
        """
        Initialize post-processor.
        
        Args:
            temporal_window: Number of frames for temporal smoothing
            expression_override_threshold: Threshold to override model prediction
            hysteresis_threshold: Frames required to switch emotion
            min_switch_confidence: Minimum confidence to consider switching
        """
        self.temporal_window = temporal_window
        self.expression_override_threshold = expression_override_threshold
        self.hysteresis_threshold = hysteresis_threshold
        self.min_switch_confidence = min_switch_confidence
        
        # FaceMesh Expression analyzer (468 landmarks)
        self.expression_analyzer = FaceMeshExpressionAnalyzer(static_image_mode=True)
        
        # Temporal buffers
        self._prob_buffer: deque = deque(maxlen=temporal_window)
        self._emotion_buffer: deque = deque(maxlen=temporal_window)
        self._expression_buffer: deque = deque(maxlen=temporal_window)
        
        # Current stable state
        self._stable_emotion: str = 'neutral'
        self._stable_confidence: float = 0.5
        self._frames_at_current: int = 0
        
        # Emotion weights for temporal averaging (recent frames matter more)
        self._time_weights = np.exp(np.linspace(-1, 0, temporal_window))
        self._time_weights /= self._time_weights.sum()
    
    def process(
        self,
        raw_emotion: str,
        raw_confidence: float,
        raw_probabilities: Dict[str, float],
        face_image: Optional[np.ndarray] = None
    ) -> PostProcessedEmotion:
        """
        Post-process an emotion prediction.
        
        Args:
            raw_emotion: Raw model prediction
            raw_confidence: Raw confidence score
            raw_probabilities: Raw probability distribution
            face_image: Cropped face image (RGB) for expression detection
            
        Returns:
            PostProcessedEmotion with corrected prediction
        """
        correction_applied = False
        correction_reason = ""
        
        # Step 1: Analyze facial expressions using FaceMesh (468 landmarks)
        expression_scores = None
        if face_image is not None and face_image.size > 0:
            expression_scores = self.expression_analyzer.analyze(face_image)
            if expression_scores is not None:
                self._expression_buffer.append(expression_scores.to_dict())
        
        # Step 2: Add to temporal buffer
        self._prob_buffer.append(raw_probabilities.copy())
        self._emotion_buffer.append(raw_emotion)
        
        # Step 3: Calculate temporally smoothed probabilities
        if len(self._prob_buffer) >= 3:
            smoothed_probs = self._temporal_smooth()
        else:
            smoothed_probs = raw_probabilities.copy()
        
        # Step 4: FaceMesh-based expression correction
        if len(self._expression_buffer) >= 2:
            recent_expressions = list(self._expression_buffer)[-5:]
            
            # Average expression scores over recent frames (from FaceMesh)
            avg_surprise = np.mean([e.get('surprise', 0) for e in recent_expressions])
            avg_fear = np.mean([e.get('fear', 0) for e in recent_expressions])
            avg_happy = np.mean([e.get('happy', 0) for e in recent_expressions])
            avg_sad = np.mean([e.get('sad', 0) for e in recent_expressions])
            avg_anger = np.mean([e.get('anger', 0) for e in recent_expressions])
            avg_neutral = np.mean([e.get('neutral', 0) for e in recent_expressions])
            
            # Feature indicators
            avg_eye_open = np.mean([e.get('eye_openness', 0.5) for e in recent_expressions])
            avg_eyebrow = np.mean([e.get('eyebrow_raise', 0.3) for e in recent_expressions])
            avg_mouth_open = np.mean([e.get('mouth_openness', 0) for e in recent_expressions])
            avg_smile = np.mean([e.get('smile_score', 0.5) for e in recent_expressions])
            
            # === SURPRISE Correction ===
            # FaceMesh detected surprise (wide eyes + open mouth + raised eyebrows)
            # Only apply if very confident (threshold raised to reduce flickering)
            if avg_surprise > 0.6 and avg_eye_open > 0.6 and avg_mouth_open > 0.35:
                if raw_emotion in ['sad', 'neutral', 'anger']:
                    smoothed_probs['surprise'] = max(smoothed_probs.get('surprise', 0), avg_surprise * 0.9)
                    smoothed_probs[raw_emotion] *= 0.5  # Less aggressive reduction
                    correction_applied = True
                    correction_reason = f"FaceMesh: Surprise (eye={avg_eye_open:.2f}, mouth={avg_mouth_open:.2f}, brow={avg_eyebrow:.2f})"
            
            # === FEAR Correction ===
            # Wide eyes + raised eyebrows but mouth not fully open
            elif avg_fear > 0.4:
                if raw_emotion in ['sad', 'neutral', 'surprise']:
                    smoothed_probs['fear'] = max(smoothed_probs.get('fear', 0), avg_fear)
                    smoothed_probs[raw_emotion] *= 0.4
                    correction_applied = True
                    correction_reason = f"FaceMesh: Fear (eye={avg_eye_open:.2f}, brow={avg_eyebrow:.2f})"
            
            # === HAPPY Correction ===
            # Smile detected
            elif avg_happy > 0.4 or avg_smile > 0.6:
                if raw_emotion in ['sad', 'fear', 'anger', 'neutral']:
                    smoothed_probs['happy'] = max(smoothed_probs.get('happy', 0), max(avg_happy, avg_smile * 0.9))
                    smoothed_probs[raw_emotion] *= 0.3
                    correction_applied = True
                    correction_reason = f"FaceMesh: Happy (smile={avg_smile:.2f})"
            
            # === NEUTRAL Correction ===
            # High neutral score from FaceMesh, but model says sad
            elif avg_neutral > 0.5 and raw_emotion == 'sad':
                if avg_sad < 0.3:  # FaceMesh doesn't see sadness
                    smoothed_probs['neutral'] = max(smoothed_probs.get('neutral', 0), avg_neutral)
                    smoothed_probs['sad'] *= 0.5
                    correction_applied = True
                    correction_reason = f"FaceMesh: Neutral (score={avg_neutral:.2f})"
            
            # === SAD Correction ===
            # FaceMesh detected sadness
            elif avg_sad > 0.45:
                if raw_emotion in ['neutral', 'happy']:
                    smoothed_probs['sad'] = max(smoothed_probs.get('sad', 0), avg_sad)
                    smoothed_probs[raw_emotion] *= 0.5
                    correction_applied = True
                    correction_reason = f"FaceMesh: Sad (score={avg_sad:.2f})"
            
            # === ANGER Correction ===
            elif avg_anger > 0.4:
                if raw_emotion in ['neutral', 'sad']:
                    smoothed_probs['anger'] = max(smoothed_probs.get('anger', 0), avg_anger)
                    smoothed_probs[raw_emotion] *= 0.5
                    correction_applied = True
                    correction_reason = f"FaceMesh: Anger (score={avg_anger:.2f})"
            
            # Renormalize if correction applied
            if correction_applied:
                total = sum(smoothed_probs.values())
                if total > 0:
                    smoothed_probs = {k: v/total for k, v in smoothed_probs.items()}
        
        # Step 5: Get final emotion from smoothed probabilities
        final_emotion = max(smoothed_probs, key=smoothed_probs.get)
        final_confidence = smoothed_probs[final_emotion]
        
        # Step 6: Apply hysteresis (require more consistent evidence to switch)
        effective_hysteresis = self.hysteresis_threshold
        if correction_applied and final_confidence > 0.7:
            effective_hysteresis = 3  # Faster switching only when very confident
        
        if final_emotion != self._stable_emotion:
            self._frames_at_current = 0
            
            # Count how many recent frames agree with new emotion
            recent_emotions = list(self._emotion_buffer)[-effective_hysteresis:]
            
            # For corrections, count based on smoothed probability leader, not raw emotion
            if correction_applied or final_confidence > 0.5:
                # Switch more readily when we have strong evidence
                self._stable_emotion = final_emotion
                self._stable_confidence = final_confidence
            elif len(recent_emotions) >= effective_hysteresis - 1:
                agreement = sum(1 for e in recent_emotions if e == final_emotion)
                if agreement >= effective_hysteresis - 1:
                    self._stable_emotion = final_emotion
                    self._stable_confidence = final_confidence
                else:
                    final_emotion = self._stable_emotion
                    final_confidence = smoothed_probs.get(self._stable_emotion, self._stable_confidence)
        else:
            self._frames_at_current += 1
            self._stable_confidence = 0.8 * self._stable_confidence + 0.2 * final_confidence
        
        return PostProcessedEmotion(
            raw_emotion=raw_emotion,
            raw_confidence=raw_confidence,
            raw_probabilities=raw_probabilities,
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            final_probabilities=smoothed_probs,
            correction_applied=correction_applied,
            correction_reason=correction_reason
        )
    
    def _temporal_smooth(self) -> Dict[str, float]:
        """Apply weighted temporal smoothing to probabilities."""
        probs_list = list(self._prob_buffer)
        n = len(probs_list)
        
        if n == 0:
            return {}
        
        # Get all emotion keys
        all_emotions = set()
        for p in probs_list:
            all_emotions.update(p.keys())
        
        # Use only the relevant time weights
        weights = self._time_weights[-n:]
        weights = weights / weights.sum()  # Renormalize
        
        smoothed = {}
        for emotion in all_emotions:
            values = np.array([p.get(emotion, 0) for p in probs_list])
            smoothed[emotion] = float(np.dot(values, weights))
        
        return smoothed
    
    def reset(self):
        """Reset all buffers and state."""
        self._prob_buffer.clear()
        self._emotion_buffer.clear()
        self._expression_buffer.clear()
        self._stable_emotion = 'neutral'
        self._stable_confidence = 0.5
        self._frames_at_current = 0
    
    def get_state(self) -> Dict:
        """Get current post-processor state for debugging."""
        recent_expr = list(self._expression_buffer)[-5:] if self._expression_buffer else []
        avg_surprise = np.mean([e.get('surprise', 0) for e in recent_expr]) if recent_expr else 0
        avg_smile = np.mean([e.get('smile', 0) for e in recent_expr]) if recent_expr else 0
        
        return {
            'stable_emotion': self._stable_emotion,
            'stable_confidence': self._stable_confidence,
            'buffer_size': len(self._prob_buffer),
            'frames_at_current': self._frames_at_current,
            'avg_surprise': avg_surprise,
            'avg_smile': avg_smile
        }
    
    def get_face_landmarks(self, face_image: np.ndarray = None):
        """
        Get raw FaceMesh landmarks for visualization.
        
        Returns cached landmarks from the last analyze() call.
        This avoids running FaceMesh twice per frame.
        
        Args:
            face_image: Optional, legacy parameter (ignored if cache exists)
            
        Returns:
            MediaPipe face landmarks object or None if no face detected
        """
        return self.expression_analyzer.get_landmarks(face_image)


# Convenience function to create post-processor
def create_emotion_postprocessor(**kwargs) -> EmotionPostProcessor:
    """Create an emotion post-processor with optional custom settings."""
    return EmotionPostProcessor(**kwargs)
