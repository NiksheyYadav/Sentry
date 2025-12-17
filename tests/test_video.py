# Video Module Tests

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, 'c:/sentry')

from src.config import VideoConfig
from src.video.frame_manager import FrameBuffer, Preprocessor, TimestampedFrame


class TestPreprocessor:
    """Tests for frame preprocessing."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = Preprocessor()
        assert preprocessor.clip_limit == 2.0
        assert preprocessor.tile_grid_size == (8, 8)
    
    def test_normalize_lighting(self):
        """Test CLAHE lighting normalization."""
        preprocessor = Preprocessor()
        
        # Create test frame (random colors)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = preprocessor.normalize_lighting(frame)
        
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
    
    def test_resize(self):
        """Test frame resizing."""
        preprocessor = Preprocessor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = preprocessor.resize(frame, 320, 240)
        
        assert result.shape == (240, 320, 3)
    
    def test_to_rgb(self):
        """Test BGR to RGB conversion."""
        preprocessor = Preprocessor()
        
        # Create BGR frame with known colors
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        
        result = preprocessor.to_rgb(frame)
        
        # After conversion, red channel should be 255
        assert result[:, :, 2].mean() == 255
        assert result[:, :, 0].mean() == 0
    
    def test_process_pipeline(self):
        """Test full preprocessing pipeline."""
        preprocessor = Preprocessor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = preprocessor.process(frame, target_size=(320, 240))
        
        assert result.shape == (240, 320, 3)


class TestFrameBuffer:
    """Tests for frame buffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        config = VideoConfig(buffer_size=50)
        buffer = FrameBuffer(config)
        
        assert buffer._buffer_size == 50
        assert len(buffer) == 0
    
    def test_add_frame(self):
        """Test adding frames to buffer."""
        buffer = FrameBuffer(buffer_size=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        timestamped = buffer.add(frame, time.time(), preprocess=False)
        
        assert len(buffer) == 1
        assert isinstance(timestamped, TimestampedFrame)
        assert timestamped.index == 0
    
    def test_buffer_sliding_window(self):
        """Test that buffer maintains sliding window."""
        buffer = FrameBuffer(buffer_size=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add more frames than buffer size
        for i in range(10):
            buffer.add(frame, float(i), preprocess=False)
        
        assert len(buffer) == 5
        
        # Check that oldest frames were dropped
        frames = buffer.get_all()
        assert frames[0].index == 5
        assert frames[-1].index == 9
    
    def test_get_latest(self):
        """Test getting most recent frames."""
        buffer = FrameBuffer(buffer_size=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for i in range(5):
            buffer.add(frame, float(i), preprocess=False)
        
        latest = buffer.get_latest(3)
        
        assert len(latest) == 3
        assert latest[-1].index == 4  # Most recent
    
    def test_get_window(self):
        """Test getting frames by time window."""
        buffer = FrameBuffer(buffer_size=100)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        base_time = time.time()
        for i in range(20):
            buffer.add(frame, base_time + i * 0.5, preprocess=False)
        
        # Get last 5 seconds (10 frames at 0.5s interval)
        window = buffer.get_window(5.0)
        
        assert len(window) >= 9  # Should be around 10 frames
    
    def test_is_full(self):
        """Test buffer full detection."""
        buffer = FrameBuffer(buffer_size=3)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        assert not buffer.is_full()
        
        for i in range(3):
            buffer.add(frame, float(i), preprocess=False)
        
        assert buffer.is_full()
    
    def test_clear(self):
        """Test buffer clearing."""
        buffer = FrameBuffer(buffer_size=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        for i in range(5):
            buffer.add(frame, float(i), preprocess=False)
        
        buffer.clear()
        
        assert len(buffer) == 0
    
    def test_temporal_stats(self):
        """Test temporal statistics calculation."""
        buffer = FrameBuffer(buffer_size=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        base_time = time.time()
        for i in range(5):
            buffer.add(frame, base_time + i * 0.1, preprocess=False)
        
        stats = buffer.get_temporal_stats()
        
        assert stats['count'] == 5
        assert 0.35 < stats['duration_seconds'] < 0.45
        assert stats['effective_fps'] > 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
