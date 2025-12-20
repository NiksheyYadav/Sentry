# Command Reference

Complete list of all CLI commands and options.

---

## Table of Contents

1. [Demo & Inference](#1-demo--inference)
2. [Training Commands](#2-training-commands)
3. [Dataset Management](#3-dataset-management)
4. [Evaluation](#4-evaluation)

---

## 1. Demo & Inference

### Run Webcam Demo

```bash
python main.py --demo
```

### Run on Video File

```bash
python main.py --video path/to/video.mp4
```

### With Custom Model

```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

### Main.py Options

| Option | Description |
|--------|-------------|
| `--demo` | Run webcam demo |
| `--video PATH` | Process video file |
| `--trained-model PATH` | Use custom trained model |
| `--camera ID` | Camera device ID (default: 0) |
| `--no-gpu` | Force CPU mode |

---

## 2. Training Commands

### Train Emotion Model

```bash
# Basic
python train.py emotion --data data/fer2013 --epochs 40

# Balanced (recommended)
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive

# Custom settings
python train.py emotion --data data/fer2013 --epochs 30 --batch-size 32 --balance --target-samples 3000
```

#### Emotion Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | (required) | Path to dataset |
| `--output` | `models/emotion_trained` | Output directory |
| `--dataset` | auto | `affectnet`, `fer2013`, or `auto` |
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--workers` | 4 | Data loading workers |
| `--balance` | - | Balance classes (oversample minorities) |
| `--target-samples` | 5000 | Samples per class when balanced |
| `--aggressive` | - | Extra strong augmentation |
| `--cpu` | - | Force CPU training |

---

### Train Posture Model

```bash
# Basic
python train.py posture --data data/posture --epochs 50

# Custom settings
python train.py posture --data data/posture --epochs 100 --batch-size 16 --seq-length 60
```

#### Posture Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | (required) | Path to dataset |
| `--output` | `models/posture_trained` | Output directory |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--seq-length` | 30 | Frames per sequence |
| `--workers` | 4 | Data loading workers |
| `--cpu` | - | Force CPU training |

---

### Train Classifier Heads

```bash
python train.py classifier --features path/to/features --labels path/to/labels.json
```

#### Classifier Options

| Option | Default | Description |
|--------|---------|-------------|
| `--features` | (required) | Path to feature files |
| `--labels` | (required) | Path to labels JSON |
| `--output` | `models/classifier_trained` | Output directory |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--cpu` | - | Force CPU training |

---

## 3. Dataset Management

### Show Download Instructions

```bash
python train.py download --dataset fer2013
python train.py download --dataset affectnet
python train.py download --dataset posture
```

### Download Posture Datasets

```bash
# List available datasets
python scripts/download_video_posture_datasets.py --list

# Download all
python scripts/download_video_posture_datasets.py --dataset all

# Download specific
python scripts/download_video_posture_datasets.py --dataset multiposture
python scripts/download_video_posture_datasets.py --dataset figshare
python scripts/download_video_posture_datasets.py --dataset ntu_skeleton
```

### Create Data Collection Session

```bash
python train.py create-session --output sessions/my_session --duration 120
```

---

## 4. Evaluation

### Evaluate Trained Model

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/fer2013
```

#### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to model checkpoint |
| `--data` | (required) | Path to evaluation dataset |
| `--output` | `evaluation_results` | Output directory |
| `--cpu` | - | Force CPU evaluation |

**Outputs:**
- `confusion_matrix.png` - Per-class accuracy visualization
- `training_curves.png` - Loss and accuracy over epochs
- `per_class_metrics.json` - Precision, recall, F1 per class

---

## Quick Reference

| Task | Command |
|------|---------|
| Webcam demo | `python main.py --demo` |
| Process video | `python main.py --video video.mp4` |
| Train emotion (balanced) | `python train.py emotion --data data/fer2013 --balance --aggressive` |
| Train posture | `python train.py posture --data data/posture --epochs 50` |
| Evaluate model | `python train.py evaluate --model path/to/model.pth --data data/fer2013` |
| Download datasets | `python scripts/download_video_posture_datasets.py --dataset all` |
