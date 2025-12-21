# Real-time Monitoring Dashboard
# Visualization for debugging and demonstration - REDESIGNED UI

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
    
    # Modern color scheme (BGR format)
    COLORS = {
        'bg_dark': (35, 25, 25),           # Deep dark blue-gray
        'bg_panel': (55, 40, 35),          # Panel background
        'bg_card': (70, 55, 50),           # Card background
        'accent_cyan': (255, 255, 0),      # Vibrant cyan
        'accent_green': (113, 204, 46),    # Modern green
        'accent_blue': (244, 133, 66),     # Primary blue
        'accent_orange': (0, 165, 255),    # Warm orange
        'accent_red': (82, 82, 255),       # Soft red
        'accent_purple': (180, 100, 200),  # Purple accent
        'text_primary': (255, 255, 255),   # White
        'text_secondary': (200, 180, 180), # Muted text
        'bar_bg': (80, 65, 60),            # Progress bar background
        'divider': (100, 85, 80),          # Divider lines
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
        
        # Panel dimensions (wider for better layout)
        self.panel_width = 340
        self.card_padding = 12
        self.card_radius = 10
        self.graph_height = 100
        self._last_snapshot: Optional[np.ndarray] = None
    
    def start(self) -> None:
        """Initialize display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_active = True
    
    def stop(self) -> None:
        """Close display window."""
        cv2.destroyWindow(self.window_name)
        self._is_active = False
    
    # ==================== HELPER DRAWING FUNCTIONS ====================
    
    def _draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], 
                           pt2: Tuple[int, int], color: Tuple[int, int, int],
                           radius: int = 10, thickness: int = -1) -> None:
        """Draw a rectangle with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Clamp radius
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        
        if thickness == -1:  # Filled
            # Draw main rectangles
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            # Draw corners
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        else:  # Outline
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def _draw_gradient_bar(self, img: np.ndarray, x: int, y: int, 
                           width: int, height: int, value: float,
                           color_start: Tuple[int, int, int],
                           color_end: Tuple[int, int, int]) -> None:
        """Draw a horizontal progress bar with gradient fill."""
        # Background
        self._draw_rounded_rect(img, (x, y), (x + width, y + height), 
                                self.COLORS['bar_bg'], radius=height // 2)
        
        # Filled portion
        fill_width = max(int(width * value), height)  # Minimum width for visibility
        if value > 0.01:
            # Create gradient
            for i in range(fill_width):
                ratio = i / max(fill_width - 1, 1)
                color = tuple(int(color_start[j] + ratio * (color_end[j] - color_start[j])) 
                             for j in range(3))
                cv2.line(img, (x + i, y + 2), (x + i, y + height - 2), color, 1)
            
            # Rounded end cap
            cv2.ellipse(img, (x + fill_width - height // 2, y + height // 2), 
                       (height // 2 - 2, height // 2 - 2), 0, -90, 90, color_end, -1)
            # Rounded start cap
            cv2.ellipse(img, (x + height // 2, y + height // 2), 
                       (height // 2 - 2, height // 2 - 2), 0, 90, 270, color_start, -1)

    def _draw_card(self, img: np.ndarray, x: int, y: int, 
                   width: int, height: int, title: str = "") -> int:
        """Draw a card container with optional title. Returns y position after card."""
        # Card background
        self._draw_rounded_rect(img, (x, y), (x + width, y + height),
                                self.COLORS['bg_card'], radius=self.card_radius)
        
        # Title
        if title:
            cv2.putText(img, title, (x + self.card_padding, y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['accent_cyan'], 1, cv2.LINE_AA)
        
        return y + height + 10  # Return next y position with margin

    # ==================== MAIN UPDATE FUNCTION ====================
    
    def update(self, 
               frame: np.ndarray,
               face_detection: Optional[FaceDetection] = None,
               pose_result: Optional[PoseResult] = None,
               prediction: Optional[MentalHealthPrediction] = None,
               alert: Optional[Alert] = None,
               emotion_result: Optional[EmotionResult] = None,
               additional_info: Optional[Dict] = None,
               snapshot_face: Optional[np.ndarray] = None) -> np.ndarray:
        """Update display with current frame and analysis results."""
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
        canvas[:, :, :] = self.COLORS['bg_dark']
        
        # Draw panel background
        cv2.rectangle(canvas, (w, 0), (w + self.panel_width, h), self.COLORS['bg_panel'], -1)
        
        # Store snapshot to display in panel
        if snapshot_face is not None:
            self._last_snapshot = snapshot_face
            
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
        panel_x = w + 15
        self._draw_panel(canvas, panel_x, h, prediction, alert, emotion_result, avg_fps, additional_info)
        
        # Display
        if self._is_active:
            cv2.imshow(self.window_name, canvas)
        
        return canvas
    
    def _draw_face(self, frame: np.ndarray, detection: FaceDetection,
                   emotion_result: Optional[EmotionResult] = None) -> np.ndarray:
        """Draw face detection overlay with modern styling."""
        x1, y1, x2, y2 = detection.bbox
        
        # Draw rounded bounding box
        self._draw_rounded_rect(frame, (x1, y1), (x2, y2), 
                                self.COLORS['accent_cyan'], radius=12, thickness=2)
        
        # Draw corner accents (tech look)
        corner_len = 20
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), self.COLORS['accent_cyan'], 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), self.COLORS['accent_cyan'], 3)
        
        # Draw landmarks with glow effect
        for lm in detection.landmarks:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(frame, (x, y), 5, self.COLORS['accent_green'], -1)
            cv2.circle(frame, (x, y), 3, self.COLORS['text_primary'], -1)
        
        # Emotion label with background
        if emotion_result:
            label = f"{emotion_result.emotion.upper()} ({emotion_result.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x, label_y = x1, y1 - 30
            
            # Label background
            self._draw_rounded_rect(frame, 
                                    (label_x - 5, label_y - 18),
                                    (label_x + label_size[0] + 10, label_y + 8),
                                    self.COLORS['bg_card'], radius=5)
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['accent_cyan'], 2, cv2.LINE_AA)
        
        return frame
    
    def _draw_pose(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Draw pose skeleton overlay with modern styling."""
        h, w = frame.shape[:2]
        
        # Draw connections with gradient
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
                    cv2.line(frame, start_pt, end_pt, self.COLORS['accent_green'], 3)
                    cv2.line(frame, start_pt, end_pt, self.COLORS['accent_cyan'], 1)  # Highlight
        
        # Draw key landmarks with glow
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        for name in key_landmarks:
            lm = result.landmarks.get(name)
            if lm and lm.visibility > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 8, self.COLORS['accent_orange'], -1)
                cv2.circle(frame, pt, 5, self.COLORS['text_primary'], -1)
        
        return frame
    
    def _draw_panel(self, canvas: np.ndarray, x: int, panel_height: int,
                    prediction: Optional[MentalHealthPrediction],
                    alert: Optional[Alert],
                    emotion_result: Optional[EmotionResult],
                    fps: float,
                    additional_info: Optional[Dict]) -> None:
        """Draw redesigned side panel."""
        card_width = self.panel_width - 30
        y = 20
        
        # ===== HEADER =====
        cv2.putText(canvas, "SENTRY AI", (x, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS['accent_cyan'], 2, cv2.LINE_AA)
        y += 15
        cv2.putText(canvas, "Mental Health Monitor", (x, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_secondary'], 1, cv2.LINE_AA)
        y += 35
        
        # FPS Badge
        fps_color = self.COLORS['accent_green'] if fps > 25 else self.COLORS['accent_orange']
        self._draw_rounded_rect(canvas, (x, y), (x + 80, y + 25), fps_color, radius=5)
        cv2.putText(canvas, f"{fps:.0f} FPS", (x + 12, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['bg_dark'], 1, cv2.LINE_AA)
        y += 40
        
        # Divider
        cv2.line(canvas, (x, y), (x + card_width, y), self.COLORS['divider'], 1)
        y += 15
        
        # ===== ASSESSMENT CARD =====
        if prediction is not None:
            card_height = 130
            self._draw_card(canvas, x, y, card_width, card_height, "ASSESSMENT")
            inner_y = y + 35
            
            # Stress
            cv2.putText(canvas, f"Stress: {prediction.stress_level}", (x + 12, inner_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_primary'], 1, cv2.LINE_AA)
            self._draw_gradient_bar(canvas, x + 140, inner_y - 10, 120, 14, 
                                   prediction.stress_confidence,
                                   self.COLORS['accent_green'], self.COLORS['accent_orange'])
            inner_y += 28
            
            # Neutral
            cv2.putText(canvas, f"Neutral: {prediction.neutral_level}", (x + 12, inner_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_primary'], 1, cv2.LINE_AA)
            self._draw_gradient_bar(canvas, x + 140, inner_y - 10, 120, 14,
                                   prediction.neutral_confidence,
                                   self.COLORS['accent_blue'], self.COLORS['accent_cyan'])
            inner_y += 28
            
            # Anxiety
            cv2.putText(canvas, f"Anxiety: {prediction.anxiety_level}", (x + 12, inner_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_primary'], 1, cv2.LINE_AA)
            self._draw_gradient_bar(canvas, x + 140, inner_y - 10, 120, 14,
                                   prediction.anxiety_confidence,
                                   self.COLORS['accent_orange'], self.COLORS['accent_red'])
            inner_y += 28
            
            # Primary concern
            cv2.putText(canvas, f"Primary: {prediction.primary_concern}", (x + 12, inner_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text_secondary'], 1, cv2.LINE_AA)
            
            y += card_height + 15
        
        # ===== EMOTION CARD =====
        if emotion_result is not None:
            card_height = 100
            self._draw_card(canvas, x, y, card_width, card_height, "FACIAL EMOTION")
            inner_y = y + 35
            
            sorted_probs = sorted(emotion_result.probabilities.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            
            for emo, prob in sorted_probs:
                # Choose color based on emotion
                if emo in ['happy', 'surprise']:
                    bar_color = (self.COLORS['accent_green'], self.COLORS['accent_cyan'])
                elif emo in ['neutral']:
                    bar_color = (self.COLORS['accent_blue'], self.COLORS['accent_purple'])
                else:
                    bar_color = (self.COLORS['accent_orange'], self.COLORS['accent_red'])
                
                cv2.putText(canvas, f"{emo.capitalize()}: {prob:.2f}", (x + 12, inner_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text_primary'], 1, cv2.LINE_AA)
                self._draw_gradient_bar(canvas, x + 140, inner_y - 8, 120, 12, prob,
                                       bar_color[0], bar_color[1])
                inner_y += 22
            
            y += card_height + 15
            
        # ===== SYSTEM RATINGS CARD =====
        card_height = 85
        self._draw_card(canvas, x, y, card_width, card_height, "SYSTEM RATINGS")
        inner_y = y + 35
        
        # Posture Rating
        posture_score = 0.5
        if additional_info:
            posture_score = float(additional_info.get('posture', 0.5)) / 30.0 # Curvature
            
        posture_rating, p_color = self._get_posture_rating(posture_score)
        cv2.putText(canvas, f"Posture: {posture_rating}", (x + 12, inner_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_color, 1, cv2.LINE_AA)
        inner_y += 25
        
        # Well-being Score
        wellbeing_score = 85
        if prediction:
            neu_idx = ['low', 'normal', 'high'].index(prediction.neutral_level)
            wellbeing_score = 40 + (neu_idx * 20)
            if prediction.stress_level == 'high': wellbeing_score -= 30
            
        cv2.putText(canvas, f"Well-being: {wellbeing_score}/100", (x + 12, inner_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['accent_green'], 1, cv2.LINE_AA)
        
        y += card_height + 15
        
        # ===== SNAPSHOT CARD =====
        if self._last_snapshot is not None:
            card_height = 160
            self._draw_card(canvas, x, y, card_width, card_height, "LATEST SNAPSHOT")
            
            # Resize snapshot to fit card
            size = 120
            try:
                snapshot_resized = cv2.resize(self._last_snapshot, (size, size))
                # Center it in card
                img_x = x + (card_width - size) // 2
                img_y = y + 30
                canvas[img_y:img_y+size, img_x:img_x+size] = snapshot_resized
            except:
                pass
                
            y += card_height + 15
            
        # ===== ALERT (if any) =====
        if alert is not None:
            alert_color = self.COLORS['accent_red'] if alert.alert_type == 'immediate' else \
                         self.COLORS['accent_orange']
            self._draw_rounded_rect(canvas, (x, y), (x + card_width, y + 35), alert_color, radius=5)
            cv2.putText(canvas, f"! ALERT: {alert.alert_type.upper()}", (x + 10, y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS['text_primary'], 2, cv2.LINE_AA)
            y += 50
            
        # ===== FOOTER =====
        footer_y = panel_height - 60
        cv2.line(canvas, (x, footer_y), (x + card_width, footer_y), self.COLORS['divider'], 1)
        footer_y += 20
        
        # Documentation link
        cv2.putText(canvas, "Docs: niksheyyadav.github.io/Sentry", (x, footer_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['accent_blue'], 1, cv2.LINE_AA)
        footer_y += 25
        
        # Monitoring indicator
        cv2.circle(canvas, (x + 8, footer_y - 5), 5, self.COLORS['accent_red'], -1)
        cv2.putText(canvas, "MONITORING ACTIVE", (x + 20, footer_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['accent_red'], 1, cv2.LINE_AA)
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press."""
        return cv2.waitKey(delay)
    
    def _get_posture_rating(self, score: float) -> Tuple[str, Tuple[int, int, int]]:
        """Convert posture score to rating and color."""
        if score < 0.2:
            return "Excellent", self.COLORS['accent_green']
        elif score < 0.4:
            return "Good", self.COLORS['accent_cyan']
        elif score < 0.7:
            return "Fair", self.COLORS['accent_orange']
        else:
            return "Poor", self.COLORS['accent_red']
    
    def is_active(self) -> bool:
        """Check if monitor is active."""
        return self._is_active
