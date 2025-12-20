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
    
    # Neutral
    neutral_level: str
    neutral_confidence: float
    neutral_probabilities: Dict[str, float]
    
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
            'sad': 0.4, 'surprise': 0.2, 'neutral': 0.1, 'happy': -0.5
        },
        'neutral': {
            'happy': 0.9, 'neutral': 0.8, 'surprise': 0.4,
            'sad': 0.1, 'fear': 0.1, 'angry': 0.1, 'disgust': 0.1
        },
        'anxiety': {
            'fear': 0.9, 'surprise': 0.4, 'angry': 0.2,
            'sad': 0.2, 'disgust': 0.1, 'neutral': 0.1, 'happy': -1.0  # Strongly suppress
        }
    }
    
    STRESS_LEVELS = ['low', 'moderate', 'high']
    NEUTRAL_LEVELS = ['low', 'normal', 'high']
    ANXIETY_LEVELS = ['minimal', 'mild', 'moderate', 'severe']
    
    def __init__(self):
        # Temporal smoothing
        self._stress_history = []
        self._neutral_history = []
        self._anxiety_history = []
        self._history_size = 60  # 2 seconds at 30 FPS
    
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
        neutral_score = self.EMOTION_WEIGHTS['neutral'].get(emotion, 0.5)
        anxiety_score = self.EMOTION_WEIGHTS['anxiety'].get(emotion, 0.1)
        
        # If we have full probability distribution, weighted average
        if emotion_probs:
            stress_score = sum(
                self.EMOTION_WEIGHTS['stress'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
            neutral_score = sum(
                self.EMOTION_WEIGHTS['neutral'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
            anxiety_score = sum(
                self.EMOTION_WEIGHTS['anxiety'].get(e, 0) * p 
                for e, p in emotion_probs.items()
            )
        
        # Posture adjustments
        # Good posture correlates with high neutral/readiness
        neutral_score = neutral_score * 0.7 + (1 - posture_score) * 0.3
        
        # High movement/fidgeting correlates with anxiety
        # But only if not happy
        movement_factor = movement_score if emotion != 'happy' else movement_score * 0.2
        anxiety_score = anxiety_score * 0.6 + movement_factor * 0.4
        
        # Clamp scores
        stress_score = np.clip(stress_score, 0, 1)
        neutral_score = np.clip(neutral_score, 0, 1)
        anxiety_score = np.clip(anxiety_score, 0, 1)
        
        # Temporal smoothing
        self._stress_history.append(stress_score)
        self._neutral_history.append(neutral_score)
        self._anxiety_history.append(anxiety_score)
        
        if len(self._stress_history) > self._history_size:
            self._stress_history.pop(0)
            self._neutral_history.pop(0)
            self._anxiety_history.pop(0)
        
        # Happiness Lock: If happy in history, strictly cap stress/anxiety
        is_recently_happy = any(h > 0.7 for h in self._neutral_history[-10:])
        if is_recently_happy or emotion == 'happy':
            stress_score = min(stress_score, 0.2)
            anxiety_score = min(anxiety_score, 0.2)
            neutral_score = max(neutral_score, 0.7)
            
        # Convert to levels
        stress_level, stress_probs = self._score_to_level(
            stress_score, self.STRESS_LEVELS
        )
        neutral_level, neutral_probs = self._score_to_level(
            neutral_score, self.NEUTRAL_LEVELS
        )
        anxiety_level, anxiety_probs = self._score_to_level(
            anxiety_score, self.ANXIETY_LEVELS
        )
        
        # Primary concern
        scores = {'stress': stress_score, 'anxiety': anxiety_score, 'neutral': neutral_score}
        
        # Only set stress or anxiety as primary if they are significantly high
        if scores['stress'] > 0.4 or scores['anxiety'] > 0.4:
            primary_concern = max(['stress', 'anxiety'], key=lambda k: scores[k])
        else:
            primary_concern = 'neutral'
            
        overall_severity = max(scores['stress'], scores['anxiety'])
        
        # Requires attention if any score > 0.6
        requires_attention = overall_severity > 0.6
        
        return HeuristicPrediction(
            stress_level=stress_level,
            stress_confidence=max(stress_probs.values()),
            stress_probabilities=stress_probs,
            neutral_level=neutral_level,
            neutral_confidence=max(neutral_probs.values()),
            neutral_probabilities=neutral_probs,
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
        self._neutral_history.clear()
        self._anxiety_history.clear()
