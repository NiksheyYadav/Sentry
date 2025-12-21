# Facial Temporal Aggregation Module
# Tracks facial features over time for stability analysis

import numpy as np
from typing import Optional, Dict, List, Deque
from collections import deque
from dataclasses import dataclass
from scipy import stats

from ..config import FacialConfig


@dataclass 
class TemporalFacialFeatures:
    """Aggregated facial features over time window."""
    # Emotion statistics
    emotion_means: Dict[str, float]
    emotion_stds: Dict[str, float]
    dominant_emotion: str
    emotion_stability: float  # 0-1, higher = more stable
    
    # AU statistics
    au_means: Dict[str, float]
    au_stds: Dict[str, float]
    au_trends: Dict[str, float]  # Positive = increasing
    
    # Combined embedding
    mean_embedding: np.ndarray
    embedding_variance: float
    
    # Temporal patterns
    expression_change_rate: float  # Changes per second
    flat_affect_duration: float  # Seconds of low expressivity


class FacialTemporalAggregator:
    """
    Aggregates facial features over sliding time window.
    
    Calculates statistical moments and temporal patterns
    to distinguish momentary expressions from sustained states.
    """
    
    def __init__(self, config: Optional[FacialConfig] = None, 
                 window_seconds: float = 10.0,
                 sample_rate: float = 10.0):
        """
        Initialize temporal aggregator.
        
        Args:
            config: Facial analysis configuration.
            window_seconds: Analysis window in seconds.
            sample_rate: Expected samples per second.
        """
        self.config = config or FacialConfig()
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.window_size = int(window_seconds * sample_rate)
        
        # Circular buffers for each feature type
        self._emotion_buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self._au_buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self._embedding_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self._timestamp_buffer: Deque[float] = deque(maxlen=self.window_size)
        
        # Track expression changes
        self._last_dominant_emotion = None
        self._emotion_changes = 0
        self._flat_affect_frames = 0
    
    def update(self, 
               emotion_probs: Dict[str, float],
               au_intensities: Dict[str, float],
               embedding: np.ndarray,
               timestamp: float) -> None:
        """
        Add new frame features to the buffer.
        
        Args:
            emotion_probs: Emotion probability distribution.
            au_intensities: AU intensity values.
            embedding: Face embedding vector.
            timestamp: Frame timestamp.
        """
        self._emotion_buffer.append(emotion_probs)
        self._au_buffer.append(au_intensities)
        self._embedding_buffer.append(embedding)
        self._timestamp_buffer.append(timestamp)
        
        # Track expression changes
        current_emotion = max(emotion_probs, key=emotion_probs.get)
        if self._last_dominant_emotion is not None:
            if current_emotion != self._last_dominant_emotion:
                self._emotion_changes += 1
        self._last_dominant_emotion = current_emotion
        
        # Track flat affect
        expressivity = sum(abs(v - 1.0/len(emotion_probs)) for v in emotion_probs.values())
        if expressivity < 0.3:  # Near uniform = flat
            self._flat_affect_frames += 1
    
    def compute_features(self) -> Optional[TemporalFacialFeatures]:
        """
        Compute aggregated temporal features.
        
        Returns:
            TemporalFacialFeatures or None if insufficient data.
        """
        if len(self._emotion_buffer) < 10:  # Minimum 1 second
            return None
        
        # Compute emotion statistics
        emotion_arrays = {
            emotion: np.array([e.get(emotion, 0) for e in self._emotion_buffer])
            for emotion in self.config.emotion_classes
        }
        
        emotion_means = {k: float(np.mean(v)) for k, v in emotion_arrays.items()}
        emotion_stds = {k: float(np.std(v)) for k, v in emotion_arrays.items()}
        
        dominant_emotion = max(emotion_means, key=emotion_means.get)
        
        # Emotion stability (inverse of total variance)
        total_variance = sum(v ** 2 for v in emotion_stds.values())
        emotion_stability = 1.0 / (1.0 + total_variance * 10)
        
        # Compute AU statistics
        au_arrays = {
            au: np.array([a.get(au, 0) for a in self._au_buffer])
            for au in self.config.action_units
        }
        
        au_means = {k: float(np.mean(v)) for k, v in au_arrays.items()}
        au_stds = {k: float(np.std(v)) for k, v in au_arrays.items()}
        
        # Compute AU trends (linear regression slope)
        au_trends = {}
        x = np.arange(len(self._au_buffer))
        for au, values in au_arrays.items():
            if len(values) > 1:
                slope, _, _, _, _ = stats.linregress(x, values)
                au_trends[au] = float(slope)
            else:
                au_trends[au] = 0.0
        
        # Embedding statistics
        embeddings = np.array(list(self._embedding_buffer))
        mean_embedding = np.mean(embeddings, axis=0)
        embedding_variance = float(np.mean(np.var(embeddings, axis=0)))
        
        # Temporal patterns
        duration = max(self._timestamp_buffer) - min(self._timestamp_buffer)
        expression_change_rate = self._emotion_changes / max(duration, 0.1)
        flat_affect_duration = self._flat_affect_frames / self.sample_rate
        
        return TemporalFacialFeatures(
            emotion_means=emotion_means,
            emotion_stds=emotion_stds,
            dominant_emotion=dominant_emotion,
            emotion_stability=emotion_stability,
            au_means=au_means,
            au_stds=au_stds,
            au_trends=au_trends,
            mean_embedding=mean_embedding,
            embedding_variance=embedding_variance,
            expression_change_rate=expression_change_rate,
            flat_affect_duration=flat_affect_duration
        )
    
    def get_stable_emotion(self) -> str:
        """
        Get the stable emotion using weighted temporal voting and hysteresis.
        
        Uses a sliding window and weights votes by model confidence to 
        filter out low-confidence flickering.
        """
        if len(self._emotion_buffer) < 5:
            return self._last_dominant_emotion or 'neutral'
            
        # Use a LARGER window for MORE stability (20 frames ≈ 2 seconds)
        window_size = 20  # INCREASED from 15
        recent = list(self._emotion_buffer)[-window_size:]
        
        # Weighted voting: sum of probabilities across the window
        weighted_counts = {}
        for probs in recent:
            for emo, score in probs.items():
                weighted_counts[emo] = weighted_counts.get(emo, 0) + score
                
        # Get the winner of the weighted vote
        voted_emotion = max(weighted_counts, key=weighted_counts.get)
        
        # Calculate dominance (ratio of winner to total weight)
        total_weight = sum(weighted_counts.values())
        dominance = weighted_counts[voted_emotion] / total_weight if total_weight > 0 else 0
        
        current_stable = self._last_dominant_emotion or 'neutral'
        
        # Hysteresis Logic - INCREASED thresholds for more stability
        # Only switch if the new emotion is significantly dominant or persistent
        if voted_emotion != current_stable:
            # Different thresholds for different transitions
            if current_stable in ['neutral', 'happy']:
                threshold = 0.35  # INCREASED from 0.28 for more stability
            elif voted_emotion in ['anger', 'fear', 'sad', 'surprise'] and current_stable in ['anger', 'fear', 'sad', 'surprise']:
                # Intra-cluster switch (e.g., Sad to Anger)
                threshold = 0.45  # INCREASED from 0.40 for more stability
            else:
                threshold = 0.38  # INCREASED from 0.32 for more stability
                
            if dominance > threshold:
                # Require more than a single frame's worth of dominance
                self._last_dominant_emotion = voted_emotion
                return voted_emotion
                
        return current_stable

    def get_feature_vector(self) -> Optional[np.ndarray]:
        """
        Get flattened feature vector for fusion.
        
        Returns:
            Feature vector or None if insufficient data.
        """
        features = self.compute_features()
        if features is None:
            return None
        
        # Concatenate all scalar features
        scalars = []
        
        # Emotion means and stds
        for emotion in self.config.emotion_classes:
            scalars.append(features.emotion_means[emotion])
            scalars.append(features.emotion_stds[emotion])
        
        scalars.append(features.emotion_stability)
        
        # AU means, stds, trends
        for au in self.config.action_units:
            scalars.append(features.au_means[au])
            scalars.append(features.au_stds[au])
            scalars.append(features.au_trends[au])
        
        scalars.append(features.expression_change_rate)
        scalars.append(features.flat_affect_duration)
        scalars.append(features.embedding_variance)
        
        # Combine with mean embedding
        scalar_array = np.array(scalars, dtype=np.float32)
        
        return np.concatenate([scalar_array, features.mean_embedding])
    
    def reset(self) -> None:
        """Clear all buffers."""
        self._emotion_buffer.clear()
        self._au_buffer.clear()
        self._embedding_buffer.clear()
        self._timestamp_buffer.clear()
        self._last_dominant_emotion = None
        self._emotion_changes = 0
        self._flat_affect_frames = 0
    
    def is_ready(self) -> bool:
        """Check if aggregator has enough data."""
        return len(self._emotion_buffer) >= self.window_size // 2
    
    def get_buffer_stats(self) -> Dict[str, float]:
        """Get current buffer statistics."""
        return {
            'buffer_fill': len(self._emotion_buffer) / self.window_size,
            'duration_seconds': (
                max(self._timestamp_buffer) - min(self._timestamp_buffer)
                if self._timestamp_buffer else 0.0
            ),
            'emotion_changes': self._emotion_changes,
            'flat_affect_frames': self._flat_affect_frames
        }
