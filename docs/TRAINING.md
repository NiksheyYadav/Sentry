# Training Guide

Complete guide for training emotion recognition, posture analysis, and mental health prediction models.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Emotion Model Training](#2-emotion-model-training)
3. [Posture Model Training](#3-posture-model-training)
4. [Evaluation](#4-evaluation)
5. [Using Trained Models](#5-using-trained-models)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Quick Start

### Train Emotion Model (Recommended)

```bash
# FER2013 with balanced classes (5000 samples each, 6 classes)
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive
```

### Train Posture Model

```bash
python train.py posture --data data/posture --epochs 50
```

---

## 2. Emotion Model Training

The emotion model uses **DenseNet121** to classify 6 emotions from grayscale face images.

### Classes

| Index | Emotion | Description |
|-------|---------|-------------|
| 0 | Neutral | Baseline state |
| 1 | Happy | Positive affect |
| 2 | Sad | Sadness indicators (correlated with depression) |
| 3 | Surprise | Unexpected events |
| 4 | Fear | Anxiety marker |
| 5 | Anger | Stress indicator |

> **Note**: 'Disgust' is automatically excluded from FER2013 (poorly labeled, rarely useful).

---

### Datasets

#### FER2013 (Recommended for Quick Training)

```bash
# Download from Kaggle
kaggle datasets download -d msambare/fer2013

# Extract
unzip fer2013.zip -d data/fer2013/
```

**Structure:**
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```

#### AffectNet (Larger, More Diverse)

```bash
kaggle datasets download -d mstjebashazida/affectnet
```

**Supports multiple formats:**
- Folder structure: `train/0/`, `train/1/`, etc.
- CSV format: `labels.csv` with columns `image,label` or `subDirectory,image,expression`

---

### Training Commands

#### Basic Training

```bash
python train.py emotion --data data/fer2013 --epochs 40
```

#### Balanced Training (Recommended)

Class imbalance is common (e.g., "happy" is 7000+ samples, "surprise" is 3000). Use `--balance` to equalize:

```bash
# Balance to 5000 samples per class with strong augmentation
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive
```

**What `--balance` does:**
- Oversamples minority classes (fear, surprise, angry)
- Undersamples majority classes (happy)
- Creates equal 5000 samples per class

**What `--aggressive` does:**
- Stronger rotation (±30°)
- More color jitter
- Heavier random erasing
- Better for oversampled data diversity

#### Custom Samples Per Class

```bash
python train.py emotion --data data/fer2013 --balance --target-samples 3000
```

---

### All Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to dataset directory |
| `--output` | `models/emotion_trained` | Where to save trained model |
| `--dataset` | auto | `affectnet`, `fer2013`, or `auto` (auto-detected from path) |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--workers` | 4 | Data loading workers |
| `--balance` | False | Balance classes via oversampling |
| `--target-samples` | 5000 | Samples per class when `--balance` is used |
| `--aggressive` | False | Extra strong augmentation |
| `--cpu` | False | Force CPU training |

---

### Anti-Overfitting Features

The training automatically applies:

| Technique | Value | Purpose |
|-----------|-------|---------|
| Label Smoothing | 0.15 | Reduces overconfidence |
| Weight Decay | 0.05 | L2 regularization |
| Dropout | 0.4-0.5 | In classifier layers |
| Data Augmentation | Various | Rotation, color jitter, erasing |
| Early Stopping | 7 epochs | Stops if no improvement |

---

## 3. Posture Model Training

The posture model uses **TCN-LSTM** for temporal body language analysis.

### Classes (Multi-Task)

| Task | Classes | Description |
|------|---------|-------------|
| **Posture** | upright, slouched, open, closed | Body language state |
| **Stress** | calm, fidgeting, restless, stillness | Movement patterns |
| **Trajectory** | stable, deteriorating, improving | Temporal trends |

---

### Dataset Download

```bash
# Download all available posture datasets
python scripts/download_video_posture_datasets.py --dataset all

# Or specific datasets
python scripts/download_video_posture_datasets.py --dataset multiposture
python scripts/download_video_posture_datasets.py --dataset figshare
```

### Dataset Format

```
data/posture/
├── train/
│   ├── sequences/
│   │   ├── seq_001.npy  # Shape: (T, 15) - T frames, 15 features
│   │   └── seq_002.npy
│   └── labels.json
└── val/
    ├── sequences/
    └── labels.json
```

**labels.json format:**
```json
{
    "seq_001": {"posture": 0, "stress": 1, "trajectory": 0},
    "seq_002": {"posture": 2, "stress": 0, "trajectory": 1}
}
```

---

### Training Command

```bash
python train.py posture --data data/posture --epochs 50 --batch-size 32
```

### All Posture Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to posture dataset |
| `--output` | `models/posture_trained` | Output directory |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--seq-length` | 30 | Frames per sequence |
| `--workers` | 4 | Data loading workers |
| `--cpu` | False | Force CPU training |

---

## 4. Evaluation

Generate confusion matrices, training curves, and per-class metrics:

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/fer2013
```

Results are saved to `evaluation_results/` including:
- `confusion_matrix.png`
- `training_curves.png`
- `per_class_metrics.json`

---

## 5. Using Trained Models

### In Demo

```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

### In Code

```python
from src.facial.emotion import EmotionClassifier
from src.config import FacialConfig
import torch

# Load model
config = FacialConfig(emotion_classes=['neutral', 'happy', 'sad', 'surprise', 'fear', 'anger'])
model = EmotionClassifier(config)

checkpoint = torch.load('models/emotion_trained/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
result = model.predict(face_image)  # RGB image
print(f"Emotion: {result.emotion}, Confidence: {result.confidence:.2f}")
```

---

## 6. Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py emotion --data data/fer2013 --batch-size 32
```

### Slow Training

```bash
# Increase workers, use GPU
python train.py emotion --data data/fer2013 --workers 8
```

### Overfitting

Use balanced training with aggressive augmentation:
```bash
python train.py emotion --data data/fer2013 --balance --aggressive --epochs 30
```

### Channel Mismatch Error

The model expects 1-channel grayscale input. Make sure your dataset loader outputs grayscale images with `mean=[0.5], std=[0.5]` normalization.

### "No samples found"

Check your data directory structure matches the expected format (see Dataset sections above).

---

## Summary

| Task | Command |
|------|---------|
| **Train Emotion (Balanced)** | `python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive` |
| **Train Emotion (AffectNet)** | `python train.py emotion --data data/affectnet --epochs 40 --balance` |
| **Train Posture** | `python train.py posture --data data/posture --epochs 50` |
| **Evaluate** | `python train.py evaluate --model path/to/model.pth --data data/fer2013` |
| **Use in Demo** | `python main.py --demo --trained-model path/to/model.pth` |
