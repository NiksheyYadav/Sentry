# Confidence Calibration and Alert System Module
# Temperature scaling, uncertainty estimation, and alert generation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from ..config import PredictionConfig
from .classifier import MentalHealthPrediction


@dataclass
class Alert:
    """Mental health alert."""
    timestamp: datetime
    alert_type: str  # 'immediate', 'follow_up', 'monitor'
    primary_concern: str
    severity: float
    confidence: float
    description: str
    recommendations: List[str]


class AlertHistory(NamedTuple):
    """Historical alert data."""
    alerts: List[Alert]
    last_alert_time: Optional[datetime]
    alert_count_24h: int


class ConfidenceCalibrator:
    """
    Confidence calibration using temperature scaling.
    
    Ensures predicted probabilities reflect true likelihood
    of correct classification.
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        """
        Initialize calibrator.
        
        Args:
            initial_temperature: Initial temperature parameter.
        """
        self.temperature = initial_temperature
        self._calibration_data = []
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits.
            
        Returns:
            Calibrated probability distribution.
        """
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)
    
    def update_temperature(self, predictions: List[np.ndarray], 
                           ground_truth: List[int]) -> float:
        """
        Update temperature based on validation data.
        
        Uses NLL loss to find optimal temperature.
        
        Args:
            predictions: List of logit arrays.
            ground_truth: List of true labels.
            
        Returns:
            Updated temperature value.
        """
        if not predictions or not ground_truth:
            return self.temperature
        
        best_temp = self.temperature
        best_nll = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            nll = 0.0
            for logits, label in zip(predictions, ground_truth):
                probs = self.calibrate_with_temp(logits, temp)
                nll -= np.log(probs[label] + 1e-10)
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
    
    def calibrate_with_temp(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Calibrate with specific temperature."""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_reliability_diagram_data(self, 
                                      predictions: List[np.ndarray],
                                      ground_truth: List[int],
                                      num_bins: int = 10) -> Dict:
        """
        Compute data for reliability diagram (calibration visualization).
        
        Args:
            predictions: List of probability arrays.
            ground_truth: True labels.
            num_bins: Number of confidence bins.
            
        Returns:
            Dictionary with bin data for plotting.
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            
            in_bin = []
            for probs, label in zip(predictions, ground_truth):
                max_prob = np.max(probs)
                if lower <= max_prob < upper:
                    in_bin.append((probs, label))
            
            if in_bin:
                accuracy = np.mean([
                    np.argmax(p) == l for p, l in in_bin
                ])
                confidence = np.mean([np.max(p) for p, _ in in_bin])
                bin_accuracies.append(accuracy)
                bin_confidences.append(confidence)
                bin_counts.append(len(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((lower + upper) / 2)
                bin_counts.append(0)
        
        return {
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'ece': self._compute_ece(bin_accuracies, bin_confidences, bin_counts)
        }
    
    def _compute_ece(self, accuracies: List, confidences: List, counts: List) -> float:
        """Compute Expected Calibration Error."""
        total = sum(counts)
        if total == 0:
            return 0.0
        
        ece = sum(
            count * abs(acc - conf) 
            for acc, conf, count in zip(accuracies, confidences, counts)
        ) / total
        
        return float(ece)


class AlertSystem:
    """
    Alert generation and management system.
    
    Generates alerts based on severity and confidence,
    with cooldown to prevent alert fatigue.
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize alert system.
        
        Args:
            config: Prediction configuration.
        """
        self.config = config or PredictionConfig()
        
        self._alert_history: List[Alert] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def evaluate(self, prediction: MentalHealthPrediction) -> Optional[Alert]:
        """
        Evaluate prediction and generate alert if warranted.
        
        Args:
            prediction: Mental health prediction.
            
        Returns:
            Alert if generated, None otherwise.
        """
        now = datetime.now()
        
        # Check each concern type
        alerts = []
        
        # Stress alert
        if prediction.stress_level == 'high':
            alerts.append(self._create_alert(
                'stress', prediction.stress_confidence, 
                prediction.overall_severity, now
            ))
        
        # Depression alert
        if prediction.depression_level in ['moderate', 'severe']:
            severity_score = 0.7 if prediction.depression_level == 'moderate' else 1.0
            alerts.append(self._create_alert(
                'depression', prediction.depression_confidence,
                severity_score, now
            ))
        
        # Anxiety alert
        if prediction.anxiety_level in ['moderate', 'severe']:
            severity_score = 0.7 if prediction.anxiety_level == 'moderate' else 1.0
            alerts.append(self._create_alert(
                'anxiety', prediction.anxiety_confidence,
                severity_score, now
            ))
        
        if not alerts:
            return None
        
        # Return highest priority alert
        highest = max(alerts, key=lambda a: a.severity * a.confidence)
        
        # Check cooldown
        if self._is_in_cooldown(highest.primary_concern, now):
            return None
        
        # Record alert
        with self._lock:
            self._alert_history.append(highest)
            self._last_alert_time[highest.primary_concern] = now
        
        return highest
    
    def _create_alert(self, concern: str, confidence: float, 
                      severity: float, timestamp: datetime) -> Alert:
        """Create an alert object."""
        # Determine alert type
        if severity >= self.config.high_severity_threshold and \
           confidence >= self.config.high_confidence_threshold:
            alert_type = 'immediate'
        elif severity >= self.config.high_severity_threshold:
            alert_type = 'follow_up'
        else:
            alert_type = 'monitor'
        
        # Generate recommendations
        recommendations = self._get_recommendations(concern, severity, alert_type)
        
        # Generate description
        description = self._get_description(concern, severity, confidence)
        
        return Alert(
            timestamp=timestamp,
            alert_type=alert_type,
            primary_concern=concern,
            severity=severity,
            confidence=confidence,
            description=description,
            recommendations=recommendations
        )
    
    def _is_in_cooldown(self, concern: str, now: datetime) -> bool:
        """Check if alerting for this concern is in cooldown."""
        last_time = self._last_alert_time.get(concern)
        if last_time is None:
            return False
        
        cooldown = timedelta(seconds=self.config.alert_cooldown_seconds)
        return now - last_time < cooldown
    
    def _get_recommendations(self, concern: str, severity: float, 
                             alert_type: str) -> List[str]:
        """Get recommendations based on concern type."""
        recommendations = {
            'stress': [
                "Consider a brief break or breathing exercise",
                "Review workload and time management",
                "Connect with support resources if needed"
            ],
            'depression': [
                "Recommend speaking with a counselor",
                "Encourage social connection",
                "Monitor for persistent patterns"
            ],
            'anxiety': [
                "Provide grounding techniques",
                "Reduce immediate stressors if possible",
                "Consider professional support"
            ]
        }
        
        base = recommendations.get(concern, ["Monitor situation"])
        
        if alert_type == 'immediate':
            base.insert(0, "PRIORITY: Immediate counselor notification recommended")
        
        return base
    
    def _get_description(self, concern: str, severity: float, 
                         confidence: float) -> str:
        """Generate alert description."""
        severity_text = "elevated" if severity < 0.7 else "high" if severity < 0.9 else "severe"
        confidence_text = "moderate" if confidence < 0.7 else "high"
        
        return f"{severity_text.capitalize()} {concern} indicators detected with {confidence_text} confidence."
    
    def get_history(self, hours: int = 24) -> AlertHistory:
        """Get alert history for specified time period."""
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                a for a in self._alert_history 
                if a.timestamp >= cutoff
            ]
            
            last_time = max(
                (a.timestamp for a in self._alert_history), 
                default=None
            )
        
        return AlertHistory(
            alerts=recent_alerts,
            last_alert_time=last_time,
            alert_count_24h=len(recent_alerts)
        )
    
    def clear_history(self) -> None:
        """Clear alert history."""
        with self._lock:
            self._alert_history.clear()
            self._last_alert_time.clear()
