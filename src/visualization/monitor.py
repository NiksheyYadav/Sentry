# Real-time Monitoring Dashboard
# Visualization for debugging and demonstration

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time

from ..config import Config
from ..facial.detector import FaceDetection
from ..facial.emotion import EmotionResult
from ..posture.pose_estimator import PoseResult
from ..prediction.classifier import MentalHealthPrediction
from ..prediction.calibration import Alert


@dataclass
class MonitorState:
    """Current monitor visualization state."""
    frame: np.ndarray
    fps: float
    face_detected: bool
    pose_detected: bool
    prediction: Optional[MentalHealthPrediction]
    alert: Optional[Alert]


class RealtimeMonitor:
    """
    Real-time visualization and monitoring dashboard.
    
    Provides visual feedback on:
    - Face and pose detection
    - Emotion and posture features
    - Predictions and confidence levels
    - Active monitoring status
    """
    
    # Color scheme
    COLORS = {
        'primary': (66, 133, 244),     # Blue
        'secondary': (52, 168, 83),    # Green
        'warning': (251, 188, 5),      # Yellow
        'danger': (234, 67, 53),       # Red
        'neutral': (154, 160, 166),    # Gray
        'background': (32, 33, 36),    # Dark gray
        'text': (255, 255, 255)        # White
    }
    
    def __init__(self, config: Optional[Config] = None,
                 window_name: str = "Mental Health Monitor"):
        """
        Initialize monitor.
        
        Args:
            config: Main configuration.
            window_name: OpenCV window name.
        """
        self.config = config or Config()
        self.window_name = window_name
        
        self._fps_buffer: List[float] = []
        self._last_frame_time = time.time()
        self._is_active = False
        
        # Panel dimensions
        self.panel_width = 300
        self.graph_height = 100
    
    def start(self) -> None:
        """Initialize display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_active = True
    
    def stop(self) -> None:
        """Close display window."""
        cv2.destroyWindow(self.window_name)
        self._is_active = False
    
    def update(self, 
               frame: np.ndarray,
               face_detection: Optional[FaceDetection] = None,
               pose_result: Optional[PoseResult] = None,
               prediction: Optional[MentalHealthPrediction] = None,
               alert: Optional[Alert] = None,
               emotion_result: Optional[EmotionResult] = None,
               additional_info: Optional[Dict] = None) -> np.ndarray:
        """
        Update display with current frame and analysis results.
        
        Args:
            frame: Current video frame (BGR).
            face_detection: Face detection result.
            pose_result: Pose estimation result.
            prediction: Mental health prediction.
            alert: Current alert if any.
            emotion_result: Facial emotion result.
            additional_info: Additional metrics to display.
            
        Returns:
            Annotated frame for display.
        """
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self._last_frame_time + 1e-6)
        self._last_frame_time = current_time
        self._fps_buffer.append(fps)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        avg_fps = np.mean(self._fps_buffer)
        
        # Create display canvas
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w + self.panel_width, 3), dtype=np.uint8)
        canvas[:, :, :] = self.COLORS['background']
        
        # Draw main frame
        annotated_frame = frame.copy()
        
        # Draw face detection
        if face_detection is not None:
            annotated_frame = self._draw_face(annotated_frame, face_detection, emotion_result)
        
        # Draw pose
        if pose_result is not None:
            annotated_frame = self._draw_pose(annotated_frame, pose_result)
        
        # Place frame in canvas
        canvas[:h, :w] = annotated_frame
        
        # Draw side panel
        panel_x = w + 10
        self._draw_panel(canvas, panel_x, prediction, alert, emotion_result, avg_fps, additional_info)
        
        # Draw privacy indicator
        self._draw_privacy_indicator(canvas, panel_x, h - 40)
        
        # Display
        if self._is_active:
            cv2.imshow(self.window_name, canvas)
        
        return canvas
    
    def _draw_face(self, frame: np.ndarray, detection: FaceDetection,
                   emotion_result: Optional[EmotionResult] = None) -> np.ndarray:
        """Draw face detection overlay."""
        x1, y1, x2, y2 = detection.bbox
        
        # Draw bounding box
        color = self.COLORS['primary']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw landmarks
        for lm in detection.landmarks:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(frame, (x, y), 3, self.COLORS['secondary'], -1)
        
        # Draw confidence and emotion
        label_y = y1 - 10
        
        # Face confidence
        label = f"Face: {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Emotion label
        if emotion_result:
            label_y -= 20
            emo_label = f"{emotion_result.emotion.upper()} ({emotion_result.confidence:.2f})"
            cv2.putText(frame, emo_label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['warning'], 2)
        
        return frame
    
    def _draw_pose(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Draw pose skeleton overlay."""
        h, w = frame.shape[:2]
        
        # Draw connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
        ]
        
        for start_name, end_name in connections:
            start_lm = result.landmarks.get(start_name)
            end_lm = result.landmarks.get(end_name)
            
            if start_lm and end_lm:
                if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(frame, start_pt, end_pt, self.COLORS['secondary'], 2)
        
        # Draw key landmarks
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 
                         'left_hip', 'right_hip']
        for name in key_landmarks:
            lm = result.landmarks.get(name)
            if lm and lm.visibility > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 5, self.COLORS['primary'], -1)
        
        return frame
    
    def _draw_panel(self, canvas: np.ndarray, x: int,
                    prediction: Optional[MentalHealthPrediction],
                    alert: Optional[Alert],
                    emotion_result: Optional[EmotionResult],
                    fps: float,
                    additional_info: Optional[Dict]) -> None:
        """Draw side information panel."""
        y = 30
        
        # Title
        cv2.putText(canvas, "Mental Health Monitor", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['text'], 2)
        y += 40
        
        # FPS
        cv2.putText(canvas, f"FPS: {fps:.1f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['neutral'], 1)
        y += 30
        
        # Divider
        cv2.line(canvas, (x, y), (x + self.panel_width - 20, y), 
                self.COLORS['neutral'], 1)
        y += 20
        
        if prediction is not None:
            # Prediction section
            cv2.putText(canvas, "Assessment", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)
            y += 30
            
            # Stress
            stress_color = self._get_severity_color(prediction.stress_level, 
                                                     ['low', 'moderate', 'high'])
            cv2.putText(canvas, f"Stress: {prediction.stress_level}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, stress_color, 1)
            self._draw_confidence_bar(canvas, x + 150, y - 10, 
                                      prediction.stress_confidence)
            y += 25
            
            # Depression
            dep_color = self._get_severity_color(prediction.depression_level,
                                                  ['minimal', 'mild', 'moderate', 'severe'])
            cv2.putText(canvas, f"Depression: {prediction.depression_level}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, dep_color, 1)
            self._draw_confidence_bar(canvas, x + 150, y - 10,
                                      prediction.depression_confidence)
            y += 25
            
            # Anxiety
            anx_color = self._get_severity_color(prediction.anxiety_level,
                                                  ['minimal', 'mild', 'moderate', 'severe'])
            cv2.putText(canvas, f"Anxiety: {prediction.anxiety_level}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, anx_color, 1)
            self._draw_confidence_bar(canvas, x + 150, y - 10,
                                      prediction.anxiety_confidence)
            y += 30
            
            # Overall
            cv2.putText(canvas, f"Primary: {prediction.primary_concern}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
            y += 40
            
        # Emotion Details Section
        if emotion_result is not None:
            cv2.putText(canvas, "Facial Emotion", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)
            y += 25
            
            # Show top 3 emotions
            sorted_probs = sorted(
                emotion_result.probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for emo, prob in sorted_probs:
                # Color based on emotion type (roughly)
                if emo in ['happy', 'neutral', 'surprise']:
                    color = self.COLORS['secondary']
                elif emo in ['fear', 'sadness', 'disgust']:
                    color = self.COLORS['warning']
                else: # angry
                    color = self.COLORS['danger']
                    
                cv2.putText(canvas, f"{emo.capitalize()}: {prob:.2f}", (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                self._draw_confidence_bar(canvas, x + 150, y - 8, prob)
                y += 20
            y += 20
        
        # Alert section
        if alert is not None:
            alert_color = self.COLORS['danger'] if alert.alert_type == 'immediate' else \
                         self.COLORS['warning'] if alert.alert_type == 'follow_up' else \
                         self.COLORS['neutral']
            
            cv2.putText(canvas, f"ALERT: {alert.alert_type.upper()}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
            y += 25
            
            # Wrap description
            words = alert.description.split()
            line = ""
            for word in words:
                if len(line + word) < 30:
                    line += word + " "
                else:
                    cv2.putText(canvas, line, (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
                    y += 18
                    line = word + " "
            if line:
                cv2.putText(canvas, line, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text'], 1)
                y += 25
        
        # Additional info
        if additional_info:
            y += 10
            cv2.line(canvas, (x, y), (x + self.panel_width - 20, y),
                    self.COLORS['neutral'], 1)
            y += 20
            
            cv2.putText(canvas, "Details", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
            y += 20
            
            for key, value in additional_info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.3f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(canvas, text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['neutral'], 1)
                y += 18
    
    def _draw_confidence_bar(self, canvas: np.ndarray, x: int, y: int,
                              confidence: float, width: int = 80, height: int = 10) -> None:
        """Draw a confidence bar."""
        # Background
        cv2.rectangle(canvas, (x, y), (x + width, y + height),
                     self.COLORS['neutral'], -1)
        
        # Filled portion
        fill_width = int(width * confidence)
        color = self.COLORS['secondary'] if confidence > 0.7 else \
                self.COLORS['warning'] if confidence > 0.4 else \
                self.COLORS['danger']
        cv2.rectangle(canvas, (x, y), (x + fill_width, y + height), color, -1)
    
    def _draw_privacy_indicator(self, canvas: np.ndarray, x: int, y: int) -> None:
        """Draw active monitoring privacy indicator."""
        # Red recording dot
        cv2.circle(canvas, (x + 10, y + 10), 6, self.COLORS['danger'], -1)
        cv2.putText(canvas, "MONITORING ACTIVE", (x + 25, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['danger'], 1)
    
    def _get_severity_color(self, level: str, levels: List[str]) -> Tuple[int, int, int]:
        """Get color based on severity level."""
        try:
            idx = levels.index(level)
            ratio = idx / (len(levels) - 1)
            
            if ratio <= 0.33:
                return self.COLORS['secondary']
            elif ratio <= 0.66:
                return self.COLORS['warning']
            else:
                return self.COLORS['danger']
        except ValueError:
            return self.COLORS['neutral']
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press."""
        return cv2.waitKey(delay)
    
    def is_active(self) -> bool:
        """Check if monitor is active."""
        return self._is_active
