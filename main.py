# Mental Health Assessment Framework - Main Entry Point
# CLI interface and pipeline orchestration

import argparse
import sys
import time
import signal
from typing import Optional
import numpy as np

from src.config import Config, default_config
from src.video.capture import VideoCapture
from src.video.frame_manager import FrameBuffer
from src.facial.detector import FaceDetector
from src.facial.emotion import create_emotion_classifier
from src.facial.action_units import create_au_detector
from src.facial.temporal import FacialTemporalAggregator
from src.facial.postprocessor import EmotionPostProcessor
from src.posture.pose_estimator import PoseEstimator
from src.posture.features import PostureFeatureExtractor
from src.posture.temporal_model import create_temporal_model
from src.fusion.fusion_network import create_fusion_network
from src.prediction.classifier import create_classifier
from src.prediction.calibration import AlertSystem
from src.prediction.heuristic import HeuristicPredictor
from src.visualization.monitor import RealtimeMonitor


class MentalHealthPipeline:
    """
    Complete mental health assessment pipeline.
    
    Orchestrates video capture, facial analysis, posture analysis,
    fusion, and prediction in real-time.
    """
    
    def __init__(self, config: Optional[Config] = None, 
                 emotion_classifier: Optional['EmotionClassifier'] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Framework configuration.
            emotion_classifier: Pre-loaded/trained emotion classifier.
        """
        self.config = config or default_config
        device = self.config.device
        
        print("Initializing Mental Health Assessment Pipeline...")
        
        # Video capture
        print("  - Video capture...")
        self.video_capture = VideoCapture(self.config.video)
        self.frame_buffer = FrameBuffer(self.config.video)
        
        # Facial analysis
        print("  - Facial analysis modules...")
        self.face_detector = FaceDetector(self.config.facial, device)
        if emotion_classifier:
            print("  - Using provided trained emotion model")
            self.emotion_classifier = emotion_classifier
        else:
            print("  - Creating new (untrained) emotion model")
            self.emotion_classifier = create_emotion_classifier(self.config.facial, device)
        self.au_detector = create_au_detector(self.config.facial, device)
        self.facial_temporal = FacialTemporalAggregator(self.config.facial)
        # Post-processing for real-world corrections (optional if mediapipe unavailable)
        try:
            self.emotion_postprocessor = EmotionPostProcessor()
            self._use_postprocessor = True
            print("  - Emotion post-processor enabled (FaceMesh)")
        except Exception as e:
            print(f"  - Emotion post-processor disabled (mediapipe issue: {e})")
            self.emotion_postprocessor = None
            self._use_postprocessor = False
        
        # Posture analysis
        print("  - Posture analysis modules...")
        self.pose_estimator = PoseEstimator(self.config.posture)
        self.posture_features = PostureFeatureExtractor(self.config.posture)
        self.posture_temporal = create_temporal_model(self.config.posture, device)
        
        # Fusion and prediction
        print("  - Fusion and prediction modules...")
        self.fusion_network = create_fusion_network(self.config.fusion, device)
        self.classifier = create_classifier(self.config.prediction, self.config.fusion, device)
        self.alert_system = AlertSystem(self.config.prediction)
        self.heuristic_predictor = HeuristicPredictor()  # Rule-based predictor
        
        # Visualization
        print("  - Visualization...")
        self.monitor = RealtimeMonitor(self.config)
        
        # State
        self._running = False
        self._frame_count = 0
        self._posture_feature_buffer = []
        self._last_emotion = 'neutral'
        self._last_emotion_probs = {}
        
        print("Pipeline initialized successfully!")
    
    def start(self) -> bool:
        """Start the pipeline."""
        if not self.video_capture.start():
            print("ERROR: Failed to start video capture!")
            return False
        
        self.monitor.start()
        self._running = True
        return True
    
    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        self.video_capture.stop()
        self.face_detector.close()
        self.pose_estimator.close()
        self.monitor.stop()
    
    def run(self) -> None:
        """Run the main processing loop."""
        print("\nStarting mental health assessment...")
        print("Press 'q' to quit, 'r' to reset temporal state\n")
        
        # Reset temporal states
        self.fusion_network.reset_temporal_state()
        self.posture_temporal.reset_hidden()
        
        frame_skip_counter = 0
        last_result = {}
        
        while self._running:
            # Read frame
            frame, timestamp = self.video_capture.read(timeout=1.0)
            
            if frame is None:
                continue
            
            # Add to buffer
            timestamped = self.frame_buffer.add(frame, timestamp)
            self._frame_count += 1
            
            # Frame skipping logic for performance
            frame_skip_counter += 1
            should_process = (frame_skip_counter >= self.config.video.frame_skip)
            
            if should_process:
                frame_skip_counter = 0
                # Process frame
                result = self._process_frame(timestamped.frame, timestamp)
                last_result = result
            else:
                # Skip processing, use previous results for visualization
                result = last_result.copy()
                if 'info' not in result: result['info'] = {}
                result['info']['status'] = 'Frame skipped'
            
            # Update visualization
            self.monitor.update(
                frame=timestamped.frame,
                face_detection=result.get('face'),
                pose_result=result.get('pose'),
                prediction=result.get('prediction'),
                alert=result.get('alert'),
                emotion_result=result.get('emotion_result'),
                additional_info=result.get('info'),
                snapshot_face=getattr(self, '_current_snapshot', None),
                face_mesh_landmarks=result.get('face_mesh_landmarks')
            )
            
            # Reset one-time snapshot
            self._current_snapshot = None
            
            # Handle key press
            key = self.monitor.wait_key(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_temporal()
                print("Temporal state reset")
            elif key == ord('s'):
                # Take snapshot
                face = result.get('face')
                if face:
                    x1, y1, x2, y2 = face.bbox
                    # Ensure bbox is within frame
                    h, w = timestamped.frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    self._current_snapshot = timestamped.frame[y1:y2, x1:x2].copy()
                    print("Snapshot captured!")
        
        self.stop()
        print("\nAssessment stopped.")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        """Process a single frame through the pipeline."""
        result = {'info': {}}
        
        # Face detection
        face = self.face_detector.detect_largest(frame)
        result['face'] = face
        
        # Pose estimation
        pose = self.pose_estimator.estimate(frame)
        result['pose'] = pose
        
        if face is None and pose is None:
            result['info']['status'] = 'No detection'
            return result
        
        # Facial analysis
        facial_embedding = None
        if face is not None:
            emotion = self.emotion_classifier.predict(face.face_image)
            au_result = self.au_detector.predict(face.face_image)
            
            # Post-process emotion using smile detection and temporal smoothing (if available)
            if self._use_postprocessor:
                postprocessed = self.emotion_postprocessor.process(
                    raw_emotion=emotion.emotion,
                    raw_confidence=emotion.confidence,
                    raw_probabilities=emotion.probabilities,
                    face_image=face.face_image
                )
                
                # Update temporal aggregator with post-processed probabilities
                self.facial_temporal.update(
                    emotion_probs=postprocessed.final_probabilities,
                    au_intensities=au_result.au_intensities,
                    embedding=emotion.embedding,
                    timestamp=timestamp
                )
                
                # Use post-processed emotion as the stable emotion
                stable_emotion = postprocessed.final_emotion
            else:
                # Fallback: use original temporal aggregation
                self.facial_temporal.update(
                    emotion_probs=emotion.probabilities,
                    au_intensities=au_result.au_intensities,
                    embedding=emotion.embedding,
                    timestamp=timestamp
                )
                stable_emotion = self.facial_temporal.get_stable_emotion()
                postprocessed = None
            
            facial_embedding = emotion.embedding
            self._last_emotion = stable_emotion
            self._last_emotion = stable_emotion
            
            if self._use_postprocessor and postprocessed:
                self._last_emotion_probs = postprocessed.final_probabilities
                result['info']['emotion'] = stable_emotion
                result['info']['raw_emotion'] = emotion.emotion  # Keep raw for debugging
                if postprocessed.correction_applied:
                    result['info']['correction'] = postprocessed.correction_reason
                
                # Update the emotion result with post-processed data
                result['emotion_result'] = type(emotion)(
                    emotion=stable_emotion,
                    confidence=postprocessed.final_confidence,
                    probabilities=postprocessed.final_probabilities,
                    embedding=emotion.embedding
                )
                
                # Get FaceMesh landmarks for meshgrid visualization
                # This reuses the analyzer from the postprocessor, no extra overhead
                result['face_mesh_landmarks'] = self.emotion_postprocessor.get_face_landmarks(
                    face.face_image
                )
            else:
                # Fallback: sync probabilities with stable emotion
                updated_probs = emotion.probabilities.copy()
                if stable_emotion in updated_probs and updated_probs[stable_emotion] < 0.5:
                    updated_probs[stable_emotion] = 0.8
                    others_sum = sum(v for k, v in updated_probs.items() if k != stable_emotion)
                    if others_sum > 0:
                        scale = 0.2 / others_sum
                        for k in updated_probs:
                            if k != stable_emotion:
                                updated_probs[k] *= scale
                
                self._last_emotion_probs = updated_probs
                result['info']['emotion'] = stable_emotion
                result['emotion_result'] = emotion
                result['emotion_result'].emotion = stable_emotion
                result['emotion_result'].probabilities = updated_probs
        
        # Posture analysis
        posture_embedding = None
        posture_score = 0.5  # Default neutral
        movement_score = 0.3  # Default slight movement
        
        if pose is not None:
            # Extract features
            geo_features = self.posture_features.extract_geometric(pose)
            mov_features = self.posture_features.extract_movement(pose)
            feature_vec = self.posture_features.get_feature_vector(pose)
            
            # Calculate posture score (higher = worse posture)
            posture_score = min(1.0, abs(geo_features.spine_curvature) / 30.0)  # Normalize curvature
            movement_score = min(1.0, mov_features.total_movement * 2.0)  # Normalize movement
            
            # Buffer for temporal model
            self._posture_feature_buffer.append(feature_vec)
            if len(self._posture_feature_buffer) > 100:
                self._posture_feature_buffer.pop(0)
            
            # Temporal analysis
            if len(self._posture_feature_buffer) >= 10:
                features_array = np.stack(self._posture_feature_buffer[-30:])
                temporal_result = self.posture_temporal.process_sequence(features_array)
                posture_embedding = temporal_result.pattern_embedding
                result['info']['posture'] = f"{geo_features.spine_curvature:.1f}"
        
        # Fusion and Prediction
        prediction = None
        if facial_embedding is not None and posture_embedding is not None:
            # Full Multimodal Fusion
            fused = self.fusion_network.fuse(facial_embedding, posture_embedding)
            prediction = self.classifier.predict(
                fused.embedding, 
                emotion_hint=self._last_emotion
            )
            
            result['info']['facial_weight'] = f"{fused.facial_contribution:.2f}"
            result['info']['posture_weight'] = f"{fused.posture_contribution:.2f}"
            result['info']['prediction_type'] = 'Fusion'
        else:
            # Fallback to heuristic
            prediction = self.heuristic_predictor.predict(
                emotion=self._last_emotion,
                emotion_probs=self._last_emotion_probs,
                posture_score=posture_score,
                movement_score=movement_score
            )
            result['info']['prediction_type'] = 'Heuristic'
            
        result['prediction'] = prediction
        
        # Check for alerts
        alert = self.alert_system.evaluate(prediction)
        result['alert'] = alert
        
        return result
    
    def _reset_temporal(self) -> None:
        """Reset all temporal state."""
        self.facial_temporal.reset()
        if self._use_postprocessor:
            self.emotion_postprocessor.reset()  # Reset post-processor state
        self.posture_features.reset()
        self._posture_feature_buffer.clear()
        self.fusion_network.reset_temporal_state()
        self.posture_temporal.reset_hidden()


def run_demo(config: Config, emotion_model: Optional['EmotionClassifier'] = None) -> None:
    """Run demo mode with visualization."""
    pipeline = MentalHealthPipeline(config, emotion_classifier=emotion_model)
    
    if pipeline.start():
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nInterrupt received, stopping...")
            pipeline.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        pipeline.run()


def run_benchmark(config: Config, duration: int = 60, emotion_model: Optional['EmotionClassifier'] = None) -> None:
    """Run performance benchmark."""
    print(f"Running {duration}s performance benchmark...")
    
    pipeline = MentalHealthPipeline(config, emotion_classifier=emotion_model)
    
    if not pipeline.start():
        return
    
    start_time = time.time()
    frame_times = []
    
    while time.time() - start_time < duration:
        frame_start = time.time()
        
        frame, timestamp = pipeline.video_capture.read(timeout=1.0)
        if frame is not None:
            pipeline._process_frame(frame, timestamp)
        
        frame_times.append(time.time() - frame_start)
    
    pipeline.stop()
    
    # Report results
    avg_time = np.mean(frame_times) * 1000
    max_time = np.max(frame_times) * 1000
    min_time = np.min(frame_times) * 1000
    fps = 1.0 / np.mean(frame_times)
    
    print(f"\nBenchmark Results:")
    print(f"  Frames processed: {len(frame_times)}")
    print(f"  Average time: {avg_time:.1f}ms")
    print(f"  Min time: {min_time:.1f}ms")
    print(f"  Max time: {max_time:.1f}ms")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Target met: {'Yes' if avg_time < 500 else 'No'} (target: <500ms)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal Mental Health Assessment Framework"
    )
    
    parser.add_argument(
        '--demo', action='store_true',
        help='Run in demo mode with visualization'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--duration', type=int, default=60,
        help='Benchmark duration in seconds'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU mode (no GPU)'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--trained-model', type=str, default=None,
        help='Path to trained emotion model checkpoint to use instead of pretrained'
    )
    parser.add_argument(
        '--posture-model', type=str, default=None,
        help='Path to trained posture model checkpoint (auto-detected from models/posture_trained/)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = default_config
    
    # Apply CLI overrides
    if args.cpu:
        config.device = "cpu"
    config.video.camera_id = args.camera
    
    # Load trained model if specified
    trained_emotion_model = None
    if args.trained_model:
        from src.utils.model_loader import load_trained_emotion_model
        trained_emotion_model = load_trained_emotion_model(
            args.trained_model, config.device
        )
    
    # Run requested mode
    if args.benchmark:
        run_benchmark(config, args.duration, emotion_model=trained_emotion_model)
    else:
        run_demo(config, emotion_model=trained_emotion_model)


if __name__ == "__main__":
    main()
