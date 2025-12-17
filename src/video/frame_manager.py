# Frame Buffer and Preprocessing Module
# Maintains sliding window and normalizes frames

import cv2
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Iterator
from dataclasses import dataclass
import threading

from ..config import VideoConfig


@dataclass
class TimestampedFrame:
    """Frame with associated metadata."""
    frame: np.ndarray
    timestamp: float
    index: int


class Preprocessor:
    """
    Frame preprocessing with adaptive histogram equalization.
    
    Normalizes lighting conditions across frames to ensure
    consistent input quality regardless of environment.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize preprocessor.
        
        Args:
            clip_limit: CLAHE clip limit for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
    
    def normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive histogram equalization to normalize lighting.
        
        Args:
            frame: BGR input frame.
            
        Returns:
            Lighting-normalized BGR frame.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_equalized = self._clahe.apply(l)
        
        # Merge back
        lab_equalized = cv2.merge([l_equalized, a, b])
        
        # Convert back to BGR
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    
    def resize(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize frame to specified dimensions."""
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for model input."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def process(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            frame: Input BGR frame.
            target_size: Optional (width, height) to resize to.
            
        Returns:
            Preprocessed frame (BGR, normalized lighting).
        """
        processed = self.normalize_lighting(frame)
        
        if target_size is not None:
            processed = self.resize(processed, target_size[0], target_size[1])
        
        return processed


class FrameBuffer:
    """
    Sliding window buffer for maintaining temporal context.
    
    Maintains the last N frames (default 100 = 10 seconds at 10 FPS)
    for temporal analysis of expressions and posture patterns.
    """
    
    def __init__(self, config: Optional[VideoConfig] = None, buffer_size: Optional[int] = None):
        """
        Initialize frame buffer.
        
        Args:
            config: Video configuration.
            buffer_size: Override buffer size (default from config).
        """
        self.config = config or VideoConfig()
        self._buffer_size = buffer_size or self.config.buffer_size
        self._buffer: deque = deque(maxlen=self._buffer_size)
        self._preprocessor = Preprocessor()
        self._frame_index = 0
        self._lock = threading.Lock()
    
    def add(self, frame: np.ndarray, timestamp: float, preprocess: bool = True) -> TimestampedFrame:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Input frame (BGR).
            timestamp: Frame timestamp.
            preprocess: Whether to apply preprocessing.
            
        Returns:
            The added TimestampedFrame.
        """
        if preprocess:
            frame = self._preprocessor.process(frame)
        
        timestamped = TimestampedFrame(
            frame=frame,
            timestamp=timestamp,
            index=self._frame_index
        )
        
        with self._lock:
            self._buffer.append(timestamped)
            self._frame_index += 1
        
        return timestamped
    
    def get_latest(self, n: int = 1) -> List[TimestampedFrame]:
        """
        Get the N most recent frames.
        
        Args:
            n: Number of frames to retrieve.
            
        Returns:
            List of most recent frames (newest last).
        """
        with self._lock:
            n = min(n, len(self._buffer))
            return list(self._buffer)[-n:]
    
    def get_window(self, seconds: float = 10.0) -> List[TimestampedFrame]:
        """
        Get frames from the last N seconds.
        
        Args:
            seconds: Time window in seconds.
            
        Returns:
            List of frames within the time window.
        """
        if len(self._buffer) == 0:
            return []
        
        with self._lock:
            latest_time = self._buffer[-1].timestamp
            cutoff_time = latest_time - seconds
            
            return [f for f in self._buffer if f.timestamp >= cutoff_time]
    
    def get_all(self) -> List[TimestampedFrame]:
        """Get all frames in buffer."""
        with self._lock:
            return list(self._buffer)
    
    def get_frames_array(self) -> np.ndarray:
        """
        Get all frames as a numpy array.
        
        Returns:
            Array of shape (N, H, W, C) where N is number of frames.
        """
        frames = self.get_all()
        if not frames:
            return np.array([])
        return np.stack([f.frame for f in frames])
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def __iter__(self) -> Iterator[TimestampedFrame]:
        """Iterate over buffered frames."""
        with self._lock:
            return iter(list(self._buffer))
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._buffer) >= self._buffer_size
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self._lock:
            self._buffer.clear()
    
    @property
    def preprocessor(self) -> Preprocessor:
        """Access the preprocessor for direct use."""
        return self._preprocessor
    
    def get_temporal_stats(self) -> dict:
        """
        Get statistics about buffered frames.
        
        Returns:
            Dictionary with buffer statistics.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return {
                    'count': 0,
                    'duration_seconds': 0.0,
                    'effective_fps': 0.0
                }
            
            first_time = self._buffer[0].timestamp
            last_time = self._buffer[-1].timestamp
            duration = last_time - first_time
            
            return {
                'count': len(self._buffer),
                'duration_seconds': duration,
                'effective_fps': len(self._buffer) / duration if duration > 0 else 0.0,
                'first_index': self._buffer[0].index,
                'last_index': self._buffer[-1].index
            }
