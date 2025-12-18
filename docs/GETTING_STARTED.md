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
  - Ensure you are running with `--demo` (uses GPU if available).
  - Use `--cpu` only if you don't have a GPU.
  - Reduce resolution in `src/config.py`.
- **"No detection"**: Ensure your face is well-lit and visible.

## 4. Configuration

You can customize the system via `src/config.py` or by passing a YAML file:

```bash
python main.py --demo --config my_config.yaml
```

Example `my_config.yaml`:
```yaml
video:
  camera_id: 1        # Use external webcam
  process_fps: 30     # Processing speed
  
prediction:
  high_severity_threshold: 0.8
```
