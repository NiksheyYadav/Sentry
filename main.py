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
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Framework configuration.
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
        self.emotion_classifier = create_emotion_classifier(self.config.facial, device)
        self.au_detector = create_au_detector(self.config.facial, device)
        self.facial_temporal = FacialTemporalAggregator(self.config.facial)
        
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
        self.pose_estimator.close()
        self.monitor.stop()
    
    def run(self) -> None:
        """Run the main processing loop."""
        print("\nStarting mental health assessment...")
        print("Press 'q' to quit, 'r' to reset temporal state\n")
        
        # Reset temporal states
        self.fusion_network.reset_temporal_state()
        self.posture_temporal.reset_hidden()
        
        while self._running:
            # Read frame
            frame, timestamp = self.video_capture.read(timeout=1.0)
            
            if frame is None:
                continue
            
            # Add to buffer
            timestamped = self.frame_buffer.add(frame, timestamp)
            self._frame_count += 1
            
            # Process frame
            result = self._process_frame(timestamped.frame, timestamp)
            
            # Update visualization
            self.monitor.update(
                frame=timestamped.frame,
                face_detection=result.get('face'),
                pose_result=result.get('pose'),
                prediction=result.get('prediction'),
                alert=result.get('alert'),
                additional_info=result.get('info')
            )
            
            # Handle key press
            key = self.monitor.wait_key(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_temporal()
                print("Temporal state reset")
        
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
        # Facial analysis
        facial_embedding = None
        if face is not None:
            emotion = self.emotion_classifier.predict(face.face_image)
            au_result = self.au_detector.predict(face.face_image)
            
            # Update temporal
            self.facial_temporal.update(
                emotion_probs=emotion.probabilities,
                au_intensities=au_result.au_intensities,
                embedding=emotion.embedding,
                timestamp=timestamp
            )
            
            facial_embedding = emotion.embedding
            self._last_emotion = emotion.emotion
            self._last_emotion_probs = emotion.probabilities
            result['info']['emotion'] = emotion.emotion
        
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
        
        # Use heuristic predictor (emotion-based assessment)
        prediction = self.heuristic_predictor.predict(
            emotion=self._last_emotion,
            emotion_probs=self._last_emotion_probs,
            posture_score=posture_score,
            movement_score=movement_score
        )
        result['prediction'] = prediction
        
        # Check for alerts
        alert = self.alert_system.evaluate(prediction)
        result['alert'] = alert
        
        # Fusion info (if both modalities available)
        if facial_embedding is not None and posture_embedding is not None:
            fused = self.fusion_network.fuse(facial_embedding, posture_embedding)
            result['info']['facial_weight'] = f"{fused.facial_contribution:.2f}"
            result['info']['posture_weight'] = f"{fused.posture_contribution:.2f}"
        
        return result
    
    def _reset_temporal(self) -> None:
        """Reset all temporal state."""
        self.facial_temporal.reset()
        self.posture_features.reset()
        self._posture_feature_buffer.clear()
        self.fusion_network.reset_temporal_state()
        self.posture_temporal.reset_hidden()


def run_demo(config: Config) -> None:
    """Run demo mode with visualization."""
    pipeline = MentalHealthPipeline(config)
    
    if pipeline.start():
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nInterrupt received, stopping...")
            pipeline.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        pipeline.run()


def run_benchmark(config: Config, duration: int = 60) -> None:
    """Run performance benchmark."""
    print(f"Running {duration}s performance benchmark...")
    
    pipeline = MentalHealthPipeline(config)
    
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
        run_benchmark(config, args.duration)
    else:
        run_demo(config)


if __name__ == "__main__":
    main()
