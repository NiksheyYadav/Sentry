import os
import torch
import numpy as np
from typing import Optional, List, Tuple, NamedTuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import urllib.request

from ..config import FacialConfig

# Model configuration
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "blaze_face_short_range.tflite"


class FaceDetection(NamedTuple):
    """Detected face with metadata."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    landmarks: np.ndarray  # facial landmarks
    face_image: np.ndarray  # Cropped face (RGB)
    padded_bbox: np.ndarray  # Padded bounding box


class FaceDetector:
    """
    MediaPipe Tasks API-based face detector.
    
    Provides high-speed face detection for real-time performance.
    """
    
    def __init__(self, config: Optional[FacialConfig] = None, device: str = "cuda"):
        """
        Initialize face detector.
        
        Args:
            config: Facial analysis configuration.
            device: Computation device.
        """
        self.config = config or FacialConfig()
        
        # Ensure model exists
        self._ensure_model_exists()
        
        # Initialize MediaPipe Face Detection
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        
        # Try to enable GPU delegate
        if torch.cuda.is_available() and device == "cuda":
            try:
                base_options.delegate = python.BaseOptions.Delegate.GPU
                print("  - Face Detector GPU delegate enabled")
            except Exception as e:
                print(f"  - Warning: Could not enable Face Detector GPU delegate: {e}")
        
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5
        )
        
        self._detector = vision.FaceDetector.create_from_options(options)
        self._frame_timestamp_ms = 0
    
    def _ensure_model_exists(self):
        """Download model if it doesn't exist."""
        if not MODEL_PATH.exists():
            print(f"Downloading face detection model to {MODEL_PATH}...")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download face detection model: {e}")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR or RGB).
            
        Returns:
            List of FaceDetection objects.
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        h, w = frame_rgb.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Increment timestamp
        self._frame_timestamp_ms += 33  # ~30fps
        
        # Detect faces
        results = self._detector.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        if not results.detections:
            return []
        
        detections = []
        for detection in results.detections:
            # Get bounding box
            bbox_data = detection.bounding_box
            x1, y1 = bbox_data.origin_x, bbox_data.origin_y
            width, height = bbox_data.width, bbox_data.height
            x2, y2 = x1 + width, y1 + height
            
            bbox = np.array([x1, y1, x2, y2])
            prob = detection.categories[0].score
            
            # Get landmarks (6 points usually)
            landmarks = []
            if detection.keypoints:
                for kp in detection.keypoints:
                    landmarks.append([kp.x * w, kp.y * h])
            landmarks = np.array(landmarks)
            
            # Calculate padded bounding box
            padded_box = self._add_padding(bbox, frame.shape)
            
            # Crop face with padding
            face_img = self._crop_face(frame_rgb, padded_box)
            
            face_det = FaceDetection(
                bbox=bbox.astype(np.int32),
                confidence=float(prob),
                landmarks=landmarks,
                face_image=face_img,
                padded_bbox=padded_box.astype(np.int32)
            )
            detections.append(face_det)
        
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
        else:
            # Fallback for empty crops
            face = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return face
    
    def detect_largest(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect the largest face in frame.
        """
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]))
    
    def get_face_roi(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Get face ROI."""
        return detection.face_image
    
    def draw_detections(self, frame: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """Draw detections."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det.confidence:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, '_detector'):
            self._detector.close()
