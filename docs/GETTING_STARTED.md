# Getting Started with Sentry

## Prerequisites

- Python 3.8+
- Webcam
- CUDA-compatible GPU (recommended for real-time performance)
- 4GB+ RAM

## 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sentry.git
cd sentry

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 2. Setup Models

The system requires MediaPipe task models.

1. Create a `models/` directory:
   ```bash
   mkdir models
   ```

2. Download the Pose Landmarker model:
   - download `pose_landmarker_heavy.task` from [Google MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)
   - Rename it to `pose_landmarker.task`
   - Place it in `c:\sentry\models\`

## 3. Running the Demo

Start the real-time monitoring dashboard:

```bash
python main.py --demo
```

### Controls

- **Q**: Quit application
- **R**: Reset temporal state (clears history)

### Troubleshooting

- **Low FPS**: 
  - Use the performance config: `python main.py --demo --config configs/performance_config.yaml`
  - Ensure GPU is being used (check console output for "MediaPipe GPU delegate enabled")
  - Adjust `frame_skip` in config (higher = faster but less frequent updates)
  - Reduce model complexity in `PostureConfig.model_complexity` (0=fastest)
- **"No detection"**: Ensure your face is well-lit and visible.

## 4. Performance Optimization

For maximum performance on GPU systems (RTX series recommended):

```bash
python main.py --demo --config configs/performance_config.yaml
```

This configuration enables:
- **GPU Delegate**: Hardware acceleration for MediaPipe pose estimation
- **Frame Skipping**: Process every 2nd frame for 2x speed boost
- **Lite Model**: Faster pose detection with minimal accuracy loss
- **Mixed Precision**: FP16 inference for faster neural network processing

### Manual Configuration

You can customize the system via `src/config.py` or by passing a YAML file:

```bash
python main.py --demo --config my_config.yaml
```

Example `my_config.yaml`:
```yaml
video:
  camera_id: 1        # Use external webcam
  process_fps: 30     # Processing speed
  frame_skip: 2       # Process every 2nd frame (1=all frames, 2=half, 3=third)
  
posture:
  model_complexity: 0           # 0=lite (fastest), 1=full, 2=heavy
  enable_gpu_delegate: true     # Enable GPU acceleration
  min_detection_confidence: 0.3 # Lower = faster detection
  
prediction:
  high_severity_threshold: 0.8

device: cuda          # Use GPU
use_fp16: true        # Mixed precision inference
```

### Performance Tuning Guide

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `frame_skip` | Higher = faster FPS, less frequent updates | 1-2 for real-time, 3-4 for slower systems |
| `model_complexity` | 0=fastest, 2=most accurate | 0 for RTX GPUs, 1 for high-end systems |
| `enable_gpu_delegate` | Enables GPU acceleration | Always `true` if GPU available |
| `min_detection_confidence` | Lower = faster detection | 0.3 for speed, 0.5 for accuracy |
