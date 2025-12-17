# Video Capture Module
# Handles webcam video acquisition with threading

import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np

from ..config import VideoConfig


class VideoCapture:
    """
    Thread-safe video capture from webcam.
    
    Captures at full FPS but yields frames at reduced rate
    for efficient processing.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize video capture.
        
        Args:
            config: Video configuration. Uses defaults if None.
        """
        self.config = config or VideoConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._skip_ratio = self.config.capture_fps // self.config.process_fps
        self._last_frame_time = 0.0
        
    def start(self) -> bool:
        """
        Start video capture in background thread.
        
        Returns:
            True if capture started successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self.config.camera_id)
        
        if not self._cap.isOpened():
            return False
        
        # Set capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.capture_fps)
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        return True
    
    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        while self._running:
            ret, frame = self._cap.read()
            
            if not ret:
                continue
            
            self._frame_count += 1
            
            # Only queue every Nth frame for processing
            if self._frame_count % self._skip_ratio == 0:
                timestamp = time.time()
                
                # Non-blocking put - drop frame if queue is full
                try:
                    self._frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    pass  # Drop frame to maintain real-time
                    
    def read(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], float]:
        """
        Read the next processed frame.
        
        Args:
            timeout: Maximum time to wait for a frame.
            
        Returns:
            Tuple of (frame, timestamp) or (None, 0) if no frame available.
        """
        try:
            frame, timestamp = self._frame_queue.get(timeout=timeout)
            return frame, timestamp
        except queue.Empty:
            return None, 0.0
    
    def get_frame_nowait(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get frame without waiting (non-blocking).
        
        Returns:
            Tuple of (frame, timestamp) or (None, 0) if no frame available.
        """
        try:
            frame, timestamp = self._frame_queue.get_nowait()
            return frame, timestamp
        except queue.Empty:
            return None, 0.0
    
    def stop(self) -> None:
        """Stop video capture and release resources."""
        self._running = False
        
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def is_running(self) -> bool:
        """Check if capture is active."""
        return self._running and self._cap is not None and self._cap.isOpened()
    
    def get_fps(self) -> float:
        """Get actual capture FPS."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current capture resolution (width, height)."""
        if self._cap is not None:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    def __iter__(self):
        """Iterate over frames."""
        return self
    
    def __next__(self) -> Tuple[np.ndarray, float]:
        """Get next frame for iteration."""
        frame, timestamp = self.read()
        if frame is None:
            raise StopIteration
        return frame, timestamp
