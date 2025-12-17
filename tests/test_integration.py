# Integration Tests

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, 'c:/sentry')

from src.config import Config, PredictionConfig
from src.prediction.classifier import MentalHealthPrediction
from src.prediction.calibration import ConfidenceCalibrator, AlertSystem, Alert


class TestConfidenceCalibrator:
    """Tests for confidence calibration."""
    
    def test_initialization(self):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator(initial_temperature=1.5)
        
        assert calibrator.temperature == 1.5
    
    def test_calibrate(self):
        """Test probability calibration."""
        calibrator = ConfidenceCalibrator(initial_temperature=1.0)
        
        logits = np.array([2.0, 1.0, 0.0])
        probs = calibrator.calibrate(logits)
        
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > probs[1] > probs[2]
    
    def test_temperature_effect(self):
        """Test that higher temperature produces softer distributions."""
        calibrator_low = ConfidenceCalibrator(initial_temperature=0.5)
        calibrator_high = ConfidenceCalibrator(initial_temperature=2.0)
        
        logits = np.array([2.0, 1.0, 0.0])
        
        probs_low = calibrator_low.calibrate(logits)
        probs_high = calibrator_high.calibrate(logits)
        
        # Higher temperature should produce more uniform distribution
        assert probs_low[0] > probs_high[0]  # Less confident at high temp
        assert probs_low[2] < probs_high[2]  # More probability mass on low logits


class TestAlertSystem:
    """Tests for alert generation."""
    
    def test_initialization(self):
        """Test alert system initialization."""
        config = PredictionConfig()
        alert_system = AlertSystem(config)
        
        assert len(alert_system._alert_history) == 0
    
    def test_no_alert_for_low_severity(self):
        """Test that low severity doesn't trigger alert."""
        alert_system = AlertSystem()
        
        prediction = MentalHealthPrediction(
            stress_level='low',
            stress_confidence=0.9,
            stress_probabilities={'low': 0.9, 'moderate': 0.08, 'high': 0.02},
            depression_level='minimal',
            depression_confidence=0.85,
            depression_probabilities={'minimal': 0.85, 'mild': 0.1, 'moderate': 0.04, 'severe': 0.01},
            anxiety_level='minimal',
            anxiety_confidence=0.9,
            anxiety_probabilities={'minimal': 0.9, 'mild': 0.07, 'moderate': 0.02, 'severe': 0.01},
            primary_concern='none',
            overall_severity=0.0,
            requires_attention=False
        )
        
        alert = alert_system.evaluate(prediction)
        
        assert alert is None
    
    def test_alert_for_high_stress(self):
        """Test alert generation for high stress."""
        alert_system = AlertSystem()
        
        prediction = MentalHealthPrediction(
            stress_level='high',
            stress_confidence=0.85,
            stress_probabilities={'low': 0.05, 'moderate': 0.1, 'high': 0.85},
            depression_level='minimal',
            depression_confidence=0.7,
            depression_probabilities={'minimal': 0.7, 'mild': 0.2, 'moderate': 0.08, 'severe': 0.02},
            anxiety_level='minimal',
            anxiety_confidence=0.7,
            anxiety_probabilities={'minimal': 0.7, 'mild': 0.2, 'moderate': 0.08, 'severe': 0.02},
            primary_concern='stress',
            overall_severity=1.0,
            requires_attention=True
        )
        
        alert = alert_system.evaluate(prediction)
        
        assert alert is not None
        assert alert.primary_concern == 'stress'
    
    def test_alert_cooldown(self):
        """Test that cooldown prevents rapid alerts."""
        config = PredictionConfig(alert_cooldown_seconds=300)
        alert_system = AlertSystem(config)
        
        prediction = MentalHealthPrediction(
            stress_level='high',
            stress_confidence=0.85,
            stress_probabilities={'low': 0.05, 'moderate': 0.1, 'high': 0.85},
            depression_level='minimal',
            depression_confidence=0.7,
            depression_probabilities={'minimal': 0.7, 'mild': 0.2, 'moderate': 0.08, 'severe': 0.02},
            anxiety_level='minimal',
            anxiety_confidence=0.7,
            anxiety_probabilities={'minimal': 0.7, 'mild': 0.2, 'moderate': 0.08, 'severe': 0.02},
            primary_concern='stress',
            overall_severity=1.0,
            requires_attention=True
        )
        
        # First alert should succeed
        alert1 = alert_system.evaluate(prediction)
        assert alert1 is not None
        
        # Second immediate alert should be blocked by cooldown
        alert2 = alert_system.evaluate(prediction)
        assert alert2 is None
    
    def test_alert_history(self):
        """Test alert history tracking."""
        alert_system = AlertSystem()
        
        prediction = MentalHealthPrediction(
            stress_level='high',
            stress_confidence=0.85,
            stress_probabilities={'low': 0.05, 'moderate': 0.1, 'high': 0.85},
            depression_level='moderate',
            depression_confidence=0.75,
            depression_probabilities={'minimal': 0.1, 'mild': 0.1, 'moderate': 0.75, 'severe': 0.05},
            anxiety_level='minimal',
            anxiety_confidence=0.7,
            anxiety_probabilities={'minimal': 0.7, 'mild': 0.2, 'moderate': 0.08, 'severe': 0.02},
            primary_concern='stress',
            overall_severity=1.0,
            requires_attention=True
        )
        
        alert_system.evaluate(prediction)
        
        history = alert_system.get_history(hours=24)
        
        assert history.alert_count_24h >= 1
        assert len(history.alerts) >= 1


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.video.capture_fps == 30
        assert config.video.process_fps == 10
        assert config.video.buffer_size == 100
        assert config.fusion.fused_dim == 1024
    
    def test_config_relationships(self):
        """Test that config values are consistent."""
        config = Config()
        
        # Buffer should hold 10 seconds at process FPS
        expected_buffer = config.video.process_fps * 10
        assert config.video.buffer_size == expected_buffer
        
        # Fused dim should be sum of embed dims
        expected_fused = config.fusion.facial_embed_dim + config.fusion.posture_embed_dim
        assert config.fusion.fused_dim == expected_fused


class TestMentalHealthPrediction:
    """Tests for prediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating prediction object."""
        prediction = MentalHealthPrediction(
            stress_level='moderate',
            stress_confidence=0.7,
            stress_probabilities={'low': 0.2, 'moderate': 0.7, 'high': 0.1},
            depression_level='minimal',
            depression_confidence=0.8,
            depression_probabilities={'minimal': 0.8, 'mild': 0.15, 'moderate': 0.04, 'severe': 0.01},
            anxiety_level='mild',
            anxiety_confidence=0.65,
            anxiety_probabilities={'minimal': 0.2, 'mild': 0.65, 'moderate': 0.12, 'severe': 0.03},
            primary_concern='stress',
            overall_severity=0.5,
            requires_attention=False
        )
        
        assert prediction.stress_level == 'moderate'
        assert prediction.requires_attention == False
    
    def test_requires_attention_threshold(self):
        """Test requires_attention logic."""
        # Should require attention when any condition >= 2 (moderate/high)
        prediction_attention = MentalHealthPrediction(
            stress_level='high',
            stress_confidence=0.9,
            stress_probabilities={'low': 0.05, 'moderate': 0.05, 'high': 0.9},
            depression_level='minimal',
            depression_confidence=0.9,
            depression_probabilities={'minimal': 0.9, 'mild': 0.08, 'moderate': 0.015, 'severe': 0.005},
            anxiety_level='minimal',
            anxiety_confidence=0.9,
            anxiety_probabilities={'minimal': 0.9, 'mild': 0.08, 'moderate': 0.015, 'severe': 0.005},
            primary_concern='stress',
            overall_severity=1.0,
            requires_attention=True
        )
        
        assert prediction_attention.requires_attention == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
