# Facial Analysis Module Tests

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, 'c:/sentry')

from src.config import FacialConfig
from src.facial.temporal import FacialTemporalAggregator


class TestFacialTemporalAggregator:
    """Tests for facial temporal aggregation."""
    
    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = FacialTemporalAggregator()
        
        assert aggregator.window_seconds == 10.0
        assert aggregator.sample_rate == 10.0
        assert aggregator.window_size == 100
    
    def test_update(self):
        """Test updating with new features."""
        aggregator = FacialTemporalAggregator()
        
        emotion_probs = {
            'happy': 0.7, 'sad': 0.1, 'neutral': 0.1,
            'angry': 0.05, 'fear': 0.02, 'disgust': 0.02, 'surprise': 0.01
        }
        au_intensities = {'AU1': 2.0, 'AU2': 1.5, 'AU4': 0.5}
        embedding = np.random.randn(1280).astype(np.float32)
        
        aggregator.update(emotion_probs, au_intensities, embedding, 1.0)
        
        assert len(aggregator._emotion_buffer) == 1
        assert aggregator._last_dominant_emotion == 'happy'
    
    def test_compute_features_insufficient_data(self):
        """Test that compute_features returns None with insufficient data."""
        aggregator = FacialTemporalAggregator()
        
        result = aggregator.compute_features()
        
        assert result is None
    
    def test_compute_features(self):
        """Test feature computation with sufficient data."""
        config = FacialConfig()
        aggregator = FacialTemporalAggregator(config)
        
        # Add enough data
        for i in range(20):
            emotion_probs = {e: np.random.random() for e in config.emotion_classes}
            # Normalize
            total = sum(emotion_probs.values())
            emotion_probs = {k: v/total for k, v in emotion_probs.items()}
            
            au_intensities = {au: np.random.random() * 3 for au in config.action_units}
            embedding = np.random.randn(1280).astype(np.float32)
            
            aggregator.update(emotion_probs, au_intensities, embedding, float(i) * 0.1)
        
        features = aggregator.compute_features()
        
        assert features is not None
        assert features.dominant_emotion in config.emotion_classes
        assert 0 <= features.emotion_stability <= 1
        assert features.mean_embedding.shape == (1280,)
    
    def test_emotion_change_tracking(self):
        """Test that emotion changes are tracked."""
        config = FacialConfig()
        aggregator = FacialTemporalAggregator(config)
        
        # Add frames with changing dominant emotion
        emotions = ['happy', 'neutral', 'sad', 'neutral', 'happy']
        
        for i, dominant in enumerate(emotions):
            probs = {e: 0.1 for e in config.emotion_classes}
            probs[dominant] = 0.5
            
            au_intensities = {au: 1.0 for au in config.action_units}
            embedding = np.random.randn(1280).astype(np.float32)
            
            aggregator.update(probs, au_intensities, embedding, float(i))
        
        # Should have recorded 4 changes (each transition)
        assert aggregator._emotion_changes == 4
    
    def test_get_feature_vector(self):
        """Test flattened feature vector generation."""
        config = FacialConfig()
        aggregator = FacialTemporalAggregator(config)
        
        for i in range(15):
            emotion_probs = {e: np.random.random() for e in config.emotion_classes}
            total = sum(emotion_probs.values())
            emotion_probs = {k: v/total for k, v in emotion_probs.items()}
            
            au_intensities = {au: np.random.random() * 3 for au in config.action_units}
            embedding = np.random.randn(1280).astype(np.float32)
            
            aggregator.update(emotion_probs, au_intensities, embedding, float(i) * 0.1)
        
        vector = aggregator.get_feature_vector()
        
        assert vector is not None
        assert len(vector) > 1280  # Scalars + embedding
    
    def test_reset(self):
        """Test resetting aggregator state."""
        config = FacialConfig()
        aggregator = FacialTemporalAggregator(config)
        
        # Add some data
        for i in range(5):
            probs = {e: 0.1 for e in config.emotion_classes}
            au = {au: 1.0 for au in config.action_units}
            aggregator.update(probs, au, np.zeros(1280), float(i))
        
        aggregator.reset()
        
        assert len(aggregator._emotion_buffer) == 0
        assert aggregator._emotion_changes == 0
    
    def test_is_ready(self):
        """Test readiness check."""
        aggregator = FacialTemporalAggregator(window_seconds=1.0, sample_rate=10.0)
        
        # window_size = 10, ready at half = 5
        assert not aggregator.is_ready()
        
        for i in range(5):
            aggregator._emotion_buffer.append({})
        
        assert aggregator.is_ready()


class TestEmotionClassifierArchitecture:
    """Tests for emotion classifier architecture (without full model)."""
    
    def test_config_defaults(self):
        """Test facial config defaults."""
        config = FacialConfig()
        
        assert len(config.emotion_classes) == 7
        assert config.embedding_dim == 1280
        assert 'happy' in config.emotion_classes
        assert 'sad' in config.emotion_classes


class TestActionUnitDetectorLogic:
    """Tests for AU detector logic."""
    
    def test_config_action_units(self):
        """Test AU configuration."""
        config = FacialConfig()
        
        assert 'AU4' in config.action_units
        assert 'AU12' in config.action_units
        assert 'AU15' in config.action_units
        assert len(config.action_units) >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
