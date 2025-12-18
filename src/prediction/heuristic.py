# Heuristic Mental Health Predictor
# Uses emotion and posture features to make sensible predictions
# Until proper training data is available

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class HeuristicPrediction:
    """Heuristic-based mental health prediction."""
    # Stress
    stress_level: str
    stress_confidence: float
    stress_probabilities: Dict[str, float]
    
    # Depression
    depression_level: str
    depression_confidence: float
    depression_probabilities: Dict[str, float]
    
    # Anxiety
    anxiety_level: str
    anxiety_confidence: float
    anxiety_probabilities: Dict[str, float]
    
    # Overall
    primary_concern: str
    overall_severity: float
    requires_attention: bool


class HeuristicPredictor:
    """
    Rule-based mental health assessment using emotion and posture.
    
    Uses research-backed correlations between emotions, posture,
    and mental health indicators.
    """
    
    # Emotion to mental health weight mappings
    # Based on clinical research correlations
    EMOTION_WEIGHTS = {
        'stress': {
            'angry': 0.8, 'fear': 0.7, 'disgust': 0.5,
            'sad': 0.4, 'surprise': 0.2, 'neutral': 0.1, 'happy': 0.0
        },
        'depression': {
            'sad': 0.9, 'neutral': 0.3, 'fear': 0.2,
            'angry': 0.3, 'disgust': 0.2, 'surprise': 0.0, 'happy': 0.0
        },
        'anxiety': {
            'fear': 0.9, 'surprise': 0.4, 'angry': 0.3,
            'sad': 0.3, 'disgust': 0.2, 'neutral': 0.1, 'happy': 0.0
        }
    }
    
    STRESS_LEVELS = ['low', 'moderate', 'high']
    DEPRESSION_LEVELS = ['minimal', 'mild', 'moderate', 'severe']
    ANXIETY_LEVELS = ['minimal', 'mild', 'moderate', 'severe']
    
    def __init__(self):
        # Temporal smoothing
        self._stress_history = []
        self._depression_history = []
        self._anxiety_history = []
        self._history_size = 30  # 1 second at 30 FPS
    
    def predict(
        self,
        emotion: str,
        emotion_probs: Optional[Dict[str, float]] = None,
        posture_score: float = 0.5,
        movement_score: float = 0.5
    ) -> HeuristicPrediction:
        """
        Make prediction based on emotion and posture.
        
        Args:
            emotion: Detected emotion (e.g., 'happy', 'sad')
            emotion_probs: Probability distribution over emotions
            posture_score: 0-1 where 0=good posture, 1=poor posture
            movement_score: 0-1 where 0=still, 1=fidgeting
            
        Returns:
            HeuristicPrediction
        """
        emotion = emotion.lower()
        
        # Calculate base scores from emotion
        stress_score = self.EMOTION_WEIGHTS['stress'].get(emotion, 0.2)
        depression_score = self.EMOTION_WEIGHTS['depression'].get(emotion, 0.1)
        anxiety_score = self.EMOTION_WEIGHTS['anxiety'].get(emotion, 0.1)
        
        # If we have full probability distribution, weighted average
        if emotion_probs:
            stress_score = sum(
                self.EMOTION_WEIGHTS['stress'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
            depression_score = sum(
                self.EMOTION_WEIGHTS['depression'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
            anxiety_score = sum(
                self.EMOTION_WEIGHTS['anxiety'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
        
        # Posture adjustments
        # Poor posture correlates with depression
        depression_score = depression_score * 0.7 + posture_score * 0.3
        
        # High movement/fidgeting correlates with anxiety
        anxiety_score = anxiety_score * 0.6 + movement_score * 0.4
        
        # Clamp scores
        stress_score = np.clip(stress_score, 0, 1)
        depression_score = np.clip(depression_score, 0, 1)
        anxiety_score = np.clip(anxiety_score, 0, 1)
        
        # Temporal smoothing
        self._stress_history.append(stress_score)
        self._depression_history.append(depression_score)
        self._anxiety_history.append(anxiety_score)
        
        if len(self._stress_history) > self._history_size:
            self._stress_history.pop(0)
            self._depression_history.pop(0)
            self._anxiety_history.pop(0)
        
        # Use smoothed scores
        stress_score = np.mean(self._stress_history)
        depression_score = np.mean(self._depression_history)
        anxiety_score = np.mean(self._anxiety_history)
        
        # Convert to levels
        stress_level, stress_probs = self._score_to_level(
            stress_score, self.STRESS_LEVELS
        )
        depression_level, depression_probs = self._score_to_level(
            depression_score, self.DEPRESSION_LEVELS
        )
        anxiety_level, anxiety_probs = self._score_to_level(
            anxiety_score, self.ANXIETY_LEVELS
        )
        
        # Primary concern
        scores = {'stress': stress_score, 'depression': depression_score, 'anxiety': anxiety_score}
        primary_concern = max(scores, key=scores.get)
        overall_severity = max(scores.values())
        
        # Requires attention if any score > 0.6
        requires_attention = overall_severity > 0.6
        
        return HeuristicPrediction(
            stress_level=stress_level,
            stress_confidence=max(stress_probs.values()),
            stress_probabilities=stress_probs,
            depression_level=depression_level,
            depression_confidence=max(depression_probs.values()),
            depression_probabilities=depression_probs,
            anxiety_level=anxiety_level,
            anxiety_confidence=max(anxiety_probs.values()),
            anxiety_probabilities=anxiety_probs,
            primary_concern=primary_concern,
            overall_severity=overall_severity,
            requires_attention=requires_attention
        )
    
    def _score_to_level(self, score: float, levels: list) -> tuple:
        """Convert 0-1 score to discrete level with probabilities."""
        n_levels = len(levels)
        
        # Create soft distribution around score
        thresholds = np.linspace(0, 1, n_levels + 1)
        probs = {}
        
        for i, level in enumerate(levels):
            center = (thresholds[i] + thresholds[i + 1]) / 2
            # Gaussian-like probability
            prob = np.exp(-((score - center) ** 2) / 0.1)
            probs[level] = prob
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        # Get level
        level = levels[min(int(score * n_levels), n_levels - 1)]
        
        return level, probs
    
    def reset(self):
        """Reset temporal history."""
        self._stress_history.clear()
        self._depression_history.clear()
        self._anxiety_history.clear()
