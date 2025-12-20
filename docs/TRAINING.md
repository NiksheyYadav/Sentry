# Training Guide

Sentry allows you to train models for emotion recognition, posture analysis, and mental health classification.

## Overview

| Model Type | Purpose | Dataset | Command |
|------------|---------|---------|---------|
| **Emotion** | Facial emotion recognition | AffectNet, FER2013 | `python train.py emotion` |
| **Posture** | Body language & stress detection | BoLD, MultiPosture | `python train.py posture` |
| **Classifier** | Mental health heads | Custom features | `python train.py classifier` |

---

## 1. Emotion Model Training

Fine-tune the DenseNet121 backbone for 6-class emotion recognition (neutral, happy, sad, surprise, fear, anger).

### Datasets

**AffectNet** (Recommended):
```bash
kaggle datasets download -d mstjebashazida/affectnet
# Supports: folder structure OR labels.csv file
```

**FER2013** (6 classes - disgust excluded):
```bash
kaggle datasets download -d msambare/fer2013
# Automatically excludes 'disgust' class for balanced training
```

> [!TIP]
> FER2013 now automatically excludes the 'disgust' class (poorly labeled, rarely useful) and remaps to 6 emotion classes matching AffectNet.

### Training Command

```bash
python train.py emotion --data data/affectnet --epochs 40 --batch-size 64
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to dataset directory |
| `--dataset` | `affectnet` | Dataset type (`affectnet` or `fer2013`) |
| `--output` | `models/emotion_trained` | Output directory |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--workers` | `4` | Data loading workers |
| `--cpu` | Flag | Force CPU training |

### Anti-Overfitting Features

The emotion trainer includes several regularization techniques:
- **Label smoothing**: 0.15 (reduces overconfidence)
- **Weight decay**: 0.05 (L2 regularization)
- **Dropout**: 0.4-0.5 in classifier heads
- **Data augmentation**: ColorJitter, RandomErasing, GaussianBlur, Perspective
- **Early stopping**: Stops after 7 epochs without improvement

### Balanced Training (for Underrepresented Classes)

Emotion datasets are typically imbalanced (e.g., "happy" is over-represented, "sad" is rare). Use balanced training to improve accuracy on minority classes:

```bash
# Balanced training with oversampling (recommended for sad/neutral improvement)
python -m training.trainers.emotion_trainer \
    --data data/affectnet \
    --balance \
    --epochs 30

# Balanced training with aggressive augmentation (best results)
python -m training.trainers.emotion_trainer \
    --data data/affectnet \
    --balance \
    --aggressive \
    --epochs 30

# Custom target samples per class
python -m training.trainers.emotion_trainer \
    --data data/affectnet \
    --balance \
    --target-samples 5000 \
    --aggressive
```

| Argument | Description |
|----------|-------------|
| `--balance` | Oversample minority classes to match the majority class |
| `--target-samples` | Target samples per class (default: match majority) |
| `--aggressive` | Use extra-strong augmentation for oversampled data |

> [!TIP]
> Balanced training significantly improves accuracy on "sad" and "neutral" emotions, which are often underrepresented in AffectNet.

---

## 2. Posture Model Training

Train the TCN-LSTM temporal model for body language analysis.

### Multi-Task Learning

The posture model learns three tasks simultaneously:

| Task | Classes | Description |
|------|---------|-------------|
| **Posture** | upright, slouched, open, closed | Body language state |
| **Stress** | calm, fidgeting, restless, stillness | Stress indicators |
| **Trajectory** | stable, deteriorating, improving | Temporal trends |

### Recommended Datasets

**Quick Download (Recommended):**
```bash
# Download all directly accessible datasets
python scripts/download_video_posture_datasets.py --dataset all

# List available datasets
python scripts/download_video_posture_datasets.py --list
```

**Available Datasets:**

| Dataset | Size | Access | Command |
|---------|------|--------|---------|
| **MultiPosture** | ~5MB | Direct | `--dataset multiposture` |
| **Figshare Sit/Stand** | ~10MB | Direct | `--dataset figshare` |
| **NTU RGB+D Skeleton** | ~5.8GB | Google Drive | `--dataset ntu_skeleton` |
| **CMU Panoptic** | Variable | Toolbox | `--dataset cmu_panoptic` |

**Dataset Details:**

| Dataset | Best For | Description |
|---------|----------|-------------|
| **MultiPosture** | Sitting posture | Upper/lower body posture labels |
| **Figshare** | Sit/stand | 50K OpenPose keypoints |
| **NTU RGB+D** | Action recognition | 60-120 action classes with skeleton |
| **CMU Panoptic** | 3D body pose | Multi-view social interactions |

**Additional Datasets (Manual Download):**

| Dataset | Link |
|---------|------|
| **BoLD** | https://cydar.ist.psu.edu/emotionchallenge/ |
| **DAiSEE** | https://people.iith.ac.in/vineethnb/resources/daisee/ |
| **MPII Pose** | http://human-pose.mpi-inf.mpg.de/ |

### Data Preparation

Structure your data as:
```
data/posture/
+-- train/
|   +-- sequences/
|   |   +-- seq_001.npy  # Shape: (T, 15)
|   |   +-- seq_002.npy
|   +-- labels.json
+-- val/
    +-- sequences/
    +-- labels.json
```

**labels.json format:**
```json
{
    "seq_001": {"posture": 0, "stress": 1, "trajectory": 0},
    "seq_002": {"posture": 2, "stress": 0, "trajectory": 1}
}
```

**Label encoding:**
- Posture: 0=upright, 1=slouched, 2=open, 3=closed
- Stress: 0=calm, 1=fidgeting, 2=restless, 3=excessive_stillness
- Trajectory: 0=stable, 1=deteriorating, 2=improving

### Training Command

```bash
python train.py posture --data data/posture --epochs 50 --batch-size 32
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to posture dataset |
| `--output` | `models/posture_trained` | Output directory |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--seq-length` | `30` | Sequence length (frames) |
| `--workers` | `4` | Data loading workers |
| `--cpu` | Flag | Force CPU training |

### Feature Extraction

To convert videos to feature sequences:

```python
from src.posture.pose_estimator import PoseEstimator
from src.posture.features import PostureFeatureExtractor
import numpy as np
import cv2

estimator = PoseEstimator()
extractor = PostureFeatureExtractor()

cap = cv2.VideoCapture("video.mp4")
features = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    pose_result = estimator.process_frame(frame)
    if pose_result:
        feature_vector = extractor.get_feature_vector(pose_result)
        features.append(feature_vector)

np.save("seq_001.npy", np.array(features))
```

---

## 3. Classifier Head Training

Train custom mental health classifier heads on extracted features.

```bash
python train.py classifier --features path/to/features --labels path/to/labels.json
```

---

## 4. Model Evaluation

Generate confusion matrices, training curves, and per-class metrics:

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/affectnet
```

Results are saved in `evaluation_results/`.

---

## 5. Using Trained Models

### Emotion Model
```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

### Loading in Code
```python
from src.facial.emotion import EmotionClassifier
import torch

model = EmotionClassifier()
checkpoint = torch.load("models/emotion_trained/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 6. Training Tips

### Reducing Overfitting
1. Use larger datasets (AffectNet > FER2013)
2. Enable early stopping (`--epochs 40` with built-in early stopping)
3. Use data augmentation (enabled by default)
4. Lower learning rate for fine-tuning

### GPU Optimization
```bash
# For RTX GPUs, TF32 is automatically enabled
python train.py emotion --data data/affectnet --batch-size 64 --workers 4
```

### Memory Issues
If you run out of GPU memory:
```bash
python train.py emotion --data data/affectnet --batch-size 32
```
