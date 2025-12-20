# 🧠 Sentry: Multimodal Mental Health Assessment Framework

A deep learning system for real-time mental health assessment using facial expressions and body posture analysis.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#-quick-start)
3. [Training Models](#-training-models)
4. [All Commands](#-all-commands)
5. [Project Structure](#-project-structure)
6. [Documentation](#-documentation)

---

## Overview

**Sentry** combines facial emotion recognition with body posture analysis to assess mental health indicators like stress, depression, and anxiety.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal AI** | DenseNet121 (Face) + TCN-LSTM (Posture) + Cross-Attention Fusion |
| **6 Emotions** | Neutral, Happy, Sad, Surprise, Fear, Anger |
| **6 Predictions** | Stress, Depression, Anxiety, Posture, Stress Indicators, Trajectory |
| **Real-time** | 20-30 FPS with GPU acceleration |
| **Privacy First** | 100% local processing - no data sent externally |

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                            │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │   FACE        │               │   BODY        │
    │   DenseNet121 │               │   MediaPipe   │
    │   → Emotion   │               │   → Pose      │
    │   → 512D      │               │   → Features  │
    └───────────────┘               └───────────────┘
            │                               │
            │                       ┌───────────────┐
            │                       │   TCN-LSTM    │
            │                       │   Temporal    │
            │                       │   → 512D      │
            │                       └───────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                ┌───────────────────────┐
                │   CROSS-ATTENTION     │
                │   FUSION (1024D)      │
                └───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │           6 PREDICTION HEADS              │
        ├───────────────────────────────────────────┤
        │ • Stress (low/moderate/high)              │
        │ • Depression (minimal/mild/moderate/severe)│
        │ • Anxiety (minimal/mild/moderate/severe)  │
        │ • Posture (upright/slouched/open/closed)  │
        │ • Stress Indicator (calm/fidgeting/...)   │
        │ • Trajectory (stable/deteriorating/...)   │
        └───────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Demo (Webcam)

```bash
python main.py --demo
```

### 3. Process a Video File

```bash
python main.py --video path/to/video.mp4
```

---

## 🎓 Training Models

### Emotion Model (FER2013 - Recommended)

```bash
# Balanced training - 5000 samples per class with augmentation
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive
```

### Emotion Model (AffectNet)

```bash
python train.py emotion --data data/affectnet --epochs 40 --balance
```

### Posture Model

```bash
python train.py posture --data data/posture --epochs 50
```

### Training Options

| Flag | Description |
|------|-------------|
| `--data` | Path to dataset (required) |
| `--epochs` | Number of training epochs (default: 20) |
| `--batch-size` | Batch size (default: 64) |
| `--balance` | Balance classes to 5000 samples each |
| `--aggressive` | Extra strong augmentation (use with --balance) |
| `--target-samples` | Custom samples per class when balancing |
| `--cpu` | Force CPU training |

---

## 📖 All Commands

### Demo & Inference

```bash
# Webcam demo
python main.py --demo

# Process video file
python main.py --video path/to/video.mp4

# Use trained emotion model
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

### Training

```bash
# Emotion training
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive

# Posture training
python train.py posture --data data/posture --epochs 50

# Classifier training (for mental health heads)
python train.py classifier --features path/to/features --labels path/to/labels.json
```

### Evaluation

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/fer2013
```

### Download Datasets

```bash
# Show download instructions
python train.py download --dataset fer2013
python train.py download --dataset affectnet
python train.py download --dataset posture

# Download posture datasets automatically
python scripts/download_video_posture_datasets.py --dataset all
```

---

## 📁 Project Structure

```
sentry/
├── main.py                 # Application entry point
├── train.py                # Training CLI
├── requirements.txt        # Dependencies
│
├── src/                    # Source code
│   ├── facial/             # Face detection & emotion
│   │   ├── emotion.py      # EmotionClassifier (DenseNet121)
│   │   └── detector.py     # MTCNN face detector
│   ├── posture/            # Pose estimation
│   │   ├── pose_estimator.py  # MediaPipe wrapper
│   │   ├── features.py     # Feature extraction
│   │   └── temporal_model.py  # TCN-LSTM model
│   ├── fusion/             # Multimodal fusion
│   │   └── fusion.py       # Cross-attention fusion
│   ├── prediction/         # Mental health prediction
│   │   └── classifier.py   # 6-head classifier
│   └── config.py           # Configuration
│
├── training/               # Training utilities
│   ├── datasets/           # Dataset loaders
│   │   ├── fer2013.py      # FER2013 loader
│   │   ├── affectnet.py    # AffectNet loader
│   │   └── transforms.py   # Data augmentation
│   └── trainers/           # Training loops
│       ├── emotion_trainer.py
│       └── posture_trainer.py
│
├── models/                 # Saved checkpoints
├── data/                   # Datasets
└── docs/                   # Documentation
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[TRAINING.md](docs/TRAINING.md)** | Complete training guide with all options |
| **[THEORY.md](docs/THEORY.md)** | Research theory and clinical correlations |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design and model details |
| **[COMMANDS.md](docs/COMMANDS.md)** | Full command reference |
| **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** | Installation and setup |
| **[CHANGELOG](docs/recent_changes_updates_byversion.md)** | Version history |

---

## 🎯 Datasets

### Emotion Recognition

| Dataset | Classes | Size | Download |
|---------|---------|------|----------|
| **FER2013** | 6 (disgust excluded) | ~28K train | `kaggle datasets download -d msambare/fer2013` |
| **AffectNet** | 6 | ~290K train | `kaggle datasets download -d mstjebashazida/affectnet` |

### Posture Analysis

```bash
# Download all posture datasets
python scripts/download_video_posture_datasets.py --dataset all
```

---

## ⚡ Performance Tips

- **GPU**: Use CUDA for 20-30 FPS real-time processing
- **Batch Size**: Reduce to 32 if out of memory
- **Workers**: Set `--workers 4` for faster data loading
- **FP16**: Automatically enabled on GPU

---

## 📄 License

MIT License - See LICENSE file for details.
