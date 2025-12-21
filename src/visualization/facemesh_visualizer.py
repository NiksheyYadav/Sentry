# Face Mesh Visualization Module
# Renders MediaPipe FaceMesh 468 landmarks as a meshgrid overlay

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class MeshConfig:
    """Configuration for mesh visualization."""
    render_mode: str = 'full'          # 'full', 'minimal', 'contours'
    opacity: float = 0.7              # Line opacity (0-1)
    color_regions: bool = True        # Color-code facial regions
    show_contours: bool = True        # Show face outline contours
    show_irises: bool = True          # Show iris tracking
    line_thickness: int = 1           # Mesh line thickness
    

class FaceMeshVisualizer:
    """
    Visualize MediaPipe FaceMesh 468 landmarks as a meshgrid overlay.
    
    Features:
    - Full tesselation mesh (~2000 triangles)
    - Color-coded facial regions (eyes, mouth, nose, etc.)
    - Face contour highlighting
    - Iris tracking visualization
    """
    
    # Color scheme for different facial regions (BGR format)
    COLORS = {
        'mesh_default': (180, 180, 180),      # Light gray for general mesh
        'eyes': (0, 255, 0),                   # Green for eyes
        'eyebrows': (255, 136, 0),             # Blue for eyebrows  
        'nose': (255, 255, 0),                 # Cyan for nose
        'lips': (255, 0, 255),                 # Magenta for lips/mouth
        'face_contour': (0, 165, 255),         # Orange for face outline
        'irises': (255, 255, 255),             # White for iris center
        'forehead': (128, 128, 128),           # Gray for forehead
    }
    
    # MediaPipe landmark indices for facial regions
    # Left eye landmarks
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Eyebrow landmarks
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    
    # Lips landmarks
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    
    # Nose landmarks
    NOSE = [1, 2, 98, 327, 4, 5, 6, 168, 197, 195, 5, 4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 278, 344, 440, 275, 294]
    
    # Face oval/contour
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    def __init__(self, config: Optional[MeshConfig] = None):
        """
        Initialize FaceMesh visualizer.
        
        Args:
            config: Mesh visualization configuration
        """
        self.config = config or MeshConfig()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Get tesselation connections
        self.tesselation = self.mp_face_mesh.FACEMESH_TESSELATION
        self.contours = self.mp_face_mesh.FACEMESH_CONTOURS
        self.irises = self.mp_face_mesh.FACEMESH_IRISES
        
        # Create region sets for quick lookup
        self._left_eye_set = set(self.LEFT_EYE)
        self._right_eye_set = set(self.RIGHT_EYE)
        self._left_brow_set = set(self.LEFT_EYEBROW)
        self._right_brow_set = set(self.RIGHT_EYEBROW)
        self._lips_set = set(self.LIPS_OUTER + self.LIPS_INNER)
        self._nose_set = set(self.NOSE)
        self._oval_set = set(self.FACE_OVAL)
    
    def _get_connection_color(self, idx1: int, idx2: int) -> Tuple[int, int, int]:
        """Get color for a connection based on the facial region."""
        if not self.config.color_regions:
            return self.COLORS['mesh_default']
        
        # Check if connection is in a specific region
        if idx1 in self._left_eye_set or idx2 in self._left_eye_set:
            return self.COLORS['eyes']
        if idx1 in self._right_eye_set or idx2 in self._right_eye_set:
            return self.COLORS['eyes']
        if idx1 in self._left_brow_set or idx2 in self._left_brow_set:
            return self.COLORS['eyebrows']
        if idx1 in self._right_brow_set or idx2 in self._right_brow_set:
            return self.COLORS['eyebrows']
        if idx1 in self._lips_set or idx2 in self._lips_set:
            return self.COLORS['lips']
        if idx1 in self._nose_set or idx2 in self._nose_set:
            return self.COLORS['nose']
        if idx1 in self._oval_set or idx2 in self._oval_set:
            return self.COLORS['face_contour']
        
        return self.COLORS['mesh_default']
    
    def draw_mesh(self, frame: np.ndarray, 
                  face_landmarks, 
                  bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Draw face meshgrid overlay on frame.
        
        Args:
            frame: BGR frame to draw on
            face_landmarks: MediaPipe face landmarks object (from cropped face image)
            bbox: Face bounding box (x1, y1, x2, y2) for coordinate transformation.
                  If provided, landmarks are transformed from cropped face space to frame space.
            
        Returns:
            Frame with mesh overlay
        """
        if face_landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Convert landmarks to pixel coordinates
        landmarks_px = []
        
        if bbox is not None:
            # Transform from cropped face space to full frame space
            x1, y1, x2, y2 = bbox
            face_w = x2 - x1
            face_h = y2 - y1
            
            for lm in face_landmarks.landmark:
                # lm.x and lm.y are normalized 0-1 in the cropped face space
                x = int(x1 + lm.x * face_w)
                y = int(y1 + lm.y * face_h)
                landmarks_px.append((x, y))
        else:
            # Use full frame dimensions (when landmarks from full frame)
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks_px.append((x, y))
        
        # Create overlay for transparency
        overlay = annotated.copy()
        
        if self.config.render_mode == 'full':
            # Draw full tesselation mesh
            self._draw_tesselation(overlay, landmarks_px)
        elif self.config.render_mode == 'minimal':
            # Draw only key contours
            self._draw_minimal(overlay, landmarks_px)
        
        # Draw face contours (always on top)
        if self.config.show_contours:
            self._draw_contours(overlay, landmarks_px)
        
        # Draw iris tracking
        if self.config.show_irises:
            self._draw_irises(overlay, landmarks_px)
        
        # Apply transparency
        alpha = self.config.opacity
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        
        return annotated
    
    def _draw_tesselation(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
        """Draw full tesselation mesh with region coloring."""
        # Group connections by color for efficient batch drawing
        connections_by_color: Dict[Tuple[int, int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
        
        for connection in self.tesselation:
            idx1, idx2 = connection
            if idx1 < len(landmarks_px) and idx2 < len(landmarks_px):
                color = self._get_connection_color(idx1, idx2)
                pt1 = landmarks_px[idx1]
                pt2 = landmarks_px[idx2]
                
                if color not in connections_by_color:
                    connections_by_color[color] = []
                connections_by_color[color].append((pt1, pt2))
        
        # Draw connections by color (more efficient)
        for color, connections in connections_by_color.items():
            for pt1, pt2 in connections:
                cv2.line(frame, pt1, pt2, color, self.config.line_thickness, cv2.LINE_AA)
    
    def _draw_minimal(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
        """Draw minimal mesh with key regions only."""
        # Draw eye contours
        self._draw_region_polygon(frame, landmarks_px, self.LEFT_EYE, self.COLORS['eyes'])
        self._draw_region_polygon(frame, landmarks_px, self.RIGHT_EYE, self.COLORS['eyes'])
        
        # Draw eyebrow lines
        self._draw_region_polyline(frame, landmarks_px, self.LEFT_EYEBROW, self.COLORS['eyebrows'])
        self._draw_region_polyline(frame, landmarks_px, self.RIGHT_EYEBROW, self.COLORS['eyebrows'])
        
        # Draw lips
        self._draw_region_polygon(frame, landmarks_px, self.LIPS_OUTER, self.COLORS['lips'])
        self._draw_region_polygon(frame, landmarks_px, self.LIPS_INNER, self.COLORS['lips'])
        
        # Draw face oval
        self._draw_region_polygon(frame, landmarks_px, self.FACE_OVAL, self.COLORS['face_contour'])
    
    def _draw_contours(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
        """Draw face contour lines."""
        for connection in self.contours:
            idx1, idx2 = connection
            if idx1 < len(landmarks_px) and idx2 < len(landmarks_px):
                pt1 = landmarks_px[idx1]
                pt2 = landmarks_px[idx2]
                color = self._get_connection_color(idx1, idx2)
                cv2.line(frame, pt1, pt2, color, self.config.line_thickness + 1, cv2.LINE_AA)
    
    def _draw_irises(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]]) -> None:
        """Draw iris tracking circles."""
        # Left iris center (landmark 468)
        if len(landmarks_px) > 468:
            left_iris = landmarks_px[468]
            cv2.circle(frame, left_iris, 3, self.COLORS['irises'], -1, cv2.LINE_AA)
            cv2.circle(frame, left_iris, 6, self.COLORS['eyes'], 1, cv2.LINE_AA)
        
        # Right iris center (landmark 473)
        if len(landmarks_px) > 473:
            right_iris = landmarks_px[473]
            cv2.circle(frame, right_iris, 3, self.COLORS['irises'], -1, cv2.LINE_AA)
            cv2.circle(frame, right_iris, 6, self.COLORS['eyes'], 1, cv2.LINE_AA)
    
    def _draw_region_polygon(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]], 
                             indices: List[int], color: Tuple[int, int, int]) -> None:
        """Draw a closed polygon for a facial region."""
        points = []
        for idx in indices:
            if idx < len(landmarks_px):
                points.append(landmarks_px[idx])
        
        if len(points) >= 3:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, self.config.line_thickness, cv2.LINE_AA)
    
    def _draw_region_polyline(self, frame: np.ndarray, landmarks_px: List[Tuple[int, int]],
                              indices: List[int], color: Tuple[int, int, int]) -> None:
        """Draw an open polyline for a facial region."""
        points = []
        for idx in indices:
            if idx < len(landmarks_px):
                points.append(landmarks_px[idx])
        
        if len(points) >= 2:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, self.config.line_thickness, cv2.LINE_AA)


def create_mesh_visualizer(render_mode: str = 'full', **kwargs) -> FaceMeshVisualizer:
    """Create a FaceMesh visualizer with custom settings."""
    config = MeshConfig(render_mode=render_mode, **kwargs)
    return FaceMeshVisualizer(config)
