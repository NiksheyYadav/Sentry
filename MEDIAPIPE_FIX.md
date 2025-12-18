# MediaPipe Import Fix - Updated for MediaPipe 0.10+

## Problem
The error `AttributeError: module 'mediapipe' has no attribute 'solutions'` occurred when trying to run the application.

## Root Cause
MediaPipe versions >= 0.10 **removed** the `mp.solutions` API entirely. The code was using the old API that is no longer available.

### Old API (removed in 0.10+):
```python
import mediapipe as mp
mp_pose = mp.solutions.pose  # ❌ No longer exists
```

### New API (required for 0.10+):
```python
from mediapipe.tasks.python import vision
landmarker = vision.PoseLandmarker.create_from_options(options)  # ✅ New way
```

## Solution Applied

### Code Updated: `pose_estimator.py`
The file has been completely rewritten to use the new **MediaPipe Tasks API**:

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create options
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

# Create the landmarker
landmarker = vision.PoseLandmarker.create_from_options(options)

# Use it
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

### Model File Required
The new API requires a `.task` model file. Download it:

```bash
# Create models directory
mkdir models

# Download the model (Windows PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "models\pose_landmarker.task"

# Or with curl
curl -L -o models/pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

**Model variants:**
- `pose_landmarker_lite.task` (~6MB) - Fastest, less accurate
- `pose_landmarker_full.task` (~13MB) - Balanced
- `pose_landmarker_heavy.task` (~26MB) - Most accurate (recommended)

## Key Differences: Old vs New API

| Feature | Old API (< 0.10) | New API (>= 0.10) |
|---------|------------------|-------------------|
| Import | `mp.solutions.pose` | `mp.tasks.vision.PoseLandmarker` |
| Model | Built-in | Requires `.task` file download |
| Input | `process(rgb_frame)` | `detect_for_video(mp_image, timestamp)` |
| Image format | numpy array | `mp.Image` object |
| Running mode | Auto | `IMAGE`, `VIDEO`, `LIVE_STREAM` |

## Verification

After downloading the model, run:
```bash
python main.py
```

## Files Modified
- `c:\sentry\src\posture\pose_estimator.py` - Complete rewrite for new API
- `c:\sentry\models\pose_landmarker.task` - New model file (to download)

## Status
✅ Code updated for MediaPipe >= 0.10 Tasks API
⏳ Model file needs to be downloaded (~26MB)
