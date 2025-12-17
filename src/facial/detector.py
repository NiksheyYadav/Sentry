# Face Detection Module
# MTCNN-based face detection with landmark extraction

import torch
import numpy as np
from typing import Optional, List, Tuple, NamedTuple
from facenet_pytorch import MTCNN
import cv2

from ..config import FacialConfig


class FaceDetection(NamedTuple):
    """Detected face with metadata."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    landmarks: np.ndarray  # 5 facial landmarks
    face_image: np.ndarray  # Cropped face (RGB)
    padded_bbox: np.ndarray  # Padded bounding box


class FaceDetector:
    """
    MTCNN-based face detector with landmark extraction.
    
    Provides face detection with appropriate padding and
    facial landmarks for downstream analysis.
    """
    
    def __init__(self, config: Optional[FacialConfig] = None, device: str = "cuda"):
        """
        Initialize face detector.
        
        Args:
            config: Facial analysis configuration.
            device: Computation device ("cuda" or "cpu").
        """
        self.config = config or FacialConfig()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self._mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=self.config.min_face_size,
            thresholds=self.config.mtcnn_thresholds,
            factor=0.709,
            post_process=False,
            select_largest=False,  # Detect all faces
            keep_all=True,
            device=self.device
        )
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR or RGB).
            
        Returns:
            List of FaceDetection objects.
        """
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Detect faces
        boxes, probs, landmarks = self._mtcnn.detect(frame_rgb, landmarks=True)
        
        if boxes is None:
            return []
        
        detections = []
        for i, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
            if prob < self.config.mtcnn_thresholds[0]:
                continue
            
            # Calculate padded bounding box
            padded_box = self._add_padding(box, frame.shape)
            
            # Crop face with padding
            face_img = self._crop_face(frame_rgb, padded_box)
            
            detection = FaceDetection(
                bbox=box.astype(np.int32),
                confidence=float(prob),
                landmarks=lm,
                face_image=face_img,
                padded_bbox=padded_box.astype(np.int32)
            )
            detections.append(detection)
        
        return detections
    
    def _add_padding(self, bbox: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """Add padding around bounding box."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Calculate padding
        pad_w = width * self.config.face_padding
        pad_h = height * self.config.face_padding
        
        # Apply padding with bounds checking
        h, w = frame_shape[:2]
        padded = np.array([
            max(0, x1 - pad_w),
            max(0, y1 - pad_h),
            min(w, x2 + pad_w),
            min(h, y2 + pad_h)
        ])
        
        return padded
    
    def _crop_face(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop face region from frame."""
        x1, y1, x2, y2 = bbox.astype(int)
        face = frame[y1:y2, x1:x2]
        
        # Resize to standard size
        if face.size > 0:
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return face
    
    def detect_largest(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect the largest face in frame.
        
        Args:
            frame: Input frame.
            
        Returns:
            Largest FaceDetection or None if no face found.
        """
        detections = self.detect(frame)
        
        if not detections:
            return None
        
        # Find largest by area
        largest = max(detections, key=lambda d: 
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        )
        
        return largest
    
    def get_face_roi(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Get face region of interest resized for model input.
        
        Args:
            frame: Original frame.
            detection: Face detection result.
            
        Returns:
            224x224 RGB face image normalized for model input.
        """
        return detection.face_image
    
    def draw_detections(self, frame: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """
        Draw detection boxes and landmarks on frame.
        
        Args:
            frame: Frame to draw on (BGR).
            detections: List of detections.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        for det in detections:
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw landmarks
            for lm in det.landmarks:
                x, y = int(lm[0]), int(lm[1])
                cv2.circle(annotated, (x, y), 2, (255, 0, 0), -1)
        
        return annotated
