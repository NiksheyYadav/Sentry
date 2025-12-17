# Mental Health Assessment Framework

A sophisticated Python-based deep learning framework that performs real-time mental health assessment by analyzing facial expressions and body posture through webcam input.

## Features

- **Multimodal Analysis**: Combines facial expression and body posture analysis
- **Real-time Processing**: Achieves 10 FPS on standard hardware
- **Temporal Modeling**: Tracks patterns over time using TCN-LSTM hybrid architecture
- **Cross-modal Fusion**: Bidirectional attention between facial and posture features
- **Calibrated Predictions**: Temperature scaling and Monte Carlo dropout for uncertainty
- **Alert System**: Severity-based alerts with cooldown to prevent fatigue
- **Privacy-First**: All processing done locally, no cloud transmission

## Installation

```bash
cd c:\sentry
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo with visualization
python main.py --demo

# Run performance benchmark
python main.py --benchmark --duration 60

# Use specific camera
python main.py --demo --camera 1

# Force CPU mode
python main.py --demo --cpu
```

## Project Structure

```
sentry/
├── src/
│   ├── config.py                # Configuration dataclasses
│   ├── video/                   # Video capture and preprocessing
│   ├── facial/                  # Face detection, emotion, AUs
│   ├── posture/                 # Pose estimation, features, temporal
│   ├── fusion/                  # Cross-attention, fusion network
│   ├── prediction/              # Classifier, calibration, alerts
│   └── visualization/           # Real-time monitoring dashboard
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── README.md
```

## Architecture

1. **Video Capture**: 30 FPS capture, 10 FPS processing (every 3rd frame)
2. **Facial Analysis**: MTCNN detection → MobileNetV3 emotion → AU detection
3. **Posture Analysis**: MediaPipe Pose → Geometric/movement features → TCN-LSTM
4. **Fusion**: Cross-attention mechanism → 1024D fused representation
5. **Prediction**: Three-headed classifier (stress/depression/anxiety)

## Controls

- `Q`: Quit
- `R`: Reset temporal state

## Configuration

Create a YAML config file:

```yaml
video:
  camera_id: 0
  process_fps: 10
  buffer_size: 100

prediction:
  high_severity_threshold: 0.7
  alert_cooldown_seconds: 300

device: "cuda"
```

Load with: `python main.py --demo --config my_config.yaml`

## Ethical Considerations

- All processing is local (no cloud transmission)
- Clear visual indicator when monitoring is active
- System serves as screening tool, not diagnostic instrument
- Alerts are recommendations for human follow-up
"# Sentry" 
