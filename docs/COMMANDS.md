
# Command Reference

This document provides a comprehensive list of all command-line interface (CLI) commands for the Sentry framework.

## Main Application (`main.py`)

The main entry point for running the real-time assessment system.

### Basic Usage

```bash
python main.py --demo
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--demo` | Flag | `False` | Run in demo mode with real-time visualization. |
| `--benchmark` | Flag | `False` | Run performance benchmark mode (no visualization). |
| `--duration` | Int | `60` | Duration in seconds for benchmark mode. |
| `--camera` | Int | `0` | Camera device ID (0=default, 1=external). |
| `--cpu` | Flag | `False` | Force CPU mode even if GPU is available. |
| `--config` | Path | `None` | Path to custom YAML configuration file. |
| `--trained-model` | Path | `None` | Path to a trained `.pth` emotion model checkpoint. |

### Examples

**Run with external camera:**
```bash
python main.py --demo --camera 1
```

**Use a custom trained model:**
```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

**Run a 30-second benchmark on CPU:**
```bash
python main.py --benchmark --duration 30 --cpu
```

**Use high-performance configuration (recommended for RTX GPUs):**
```bash
python main.py --demo --config configs/performance_config.yaml
```

---

## Training Utilities (`train.py`)

Utilities for training models, evaluating performance, and managing datasets.

### Available Commands

| Command | Description |
|---------|-------------|
| `emotion` | Train facial emotion classifier |
| `posture` | Train posture temporal model |
| `classifier` | Train mental health classifier heads |
| `evaluate` | Evaluate trained model performance |
| `download` | Show dataset download instructions |
| `create-session` | Create data collection session template |

---

### 1. Train Emotion Model

Fine-tune the facial emotion classifier on AffectNet or FER2013.

```bash
python train.py emotion --data data/affectnet --epochs 40
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | Yes | - | Path to dataset directory (e.g., `data/affectnet`). |
| `--dataset` | No | `affectnet` | Dataset type (`affectnet` or `fer2013`). |
| `--output` | No | `models/emotion_trained` | Directory to save trained models. |
| `--epochs` | No | `20` | Number of training epochs. |
| `--batch-size` | No | `64` | Training batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--workers` | No | `4` | Number of data loading workers. |
| `--cpu` | No | Flag | Force CPU training. |

**Features:**
- DenseNet121 backbone with transfer learning
- Label smoothing (0.15) and weight decay (0.05)
- Strong data augmentation (ColorJitter, GaussianBlur, RandomErasing)
- Early stopping after 7 epochs without improvement
- Class weighting for imbalanced data

---

### 2. Train Posture Model

Train the TCN-LSTM temporal model for body language and stress detection.

```bash
python train.py posture --data data/posture --epochs 50
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | Yes | - | Path to posture dataset directory. |
| `--output` | No | `models/posture_trained` | Directory to save trained models. |
| `--epochs` | No | `50` | Number of training epochs. |
| `--batch-size` | No | `32` | Training batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--seq-length` | No | `30` | Sequence length in frames. |
| `--workers` | No | `4` | Number of data loading workers. |
| `--cpu` | No | Flag | Force CPU training. |

**Multi-Task Learning:**
- Posture classification: upright, slouched, open, closed
- Stress indicators: calm, fidgeting, restless, excessive_stillness
- Trajectory prediction: stable, deteriorating, improving

---

### 3. Train Classifier Heads

Train custom mental health classifier heads on extracted features.

```bash
python train.py classifier --features path/to/features --labels path/to/labels.json
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--features` | Yes | - | Path to extracted feature directory. |
| `--labels` | Yes | - | Path to labels JSON file. |
| `--output` | No | `models/classifier_trained` | Output directory. |
| `--epochs` | No | `50` | Training epochs. |
| `--batch-size` | No | `64` | Batch size. |
| `--lr` | No | `1e-3` | Learning rate. |
| `--cpu` | No | Flag | Force CPU training. |

---

### 4. Evaluate Model

Generate metrics, confusion matrices, and training plots.

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/affectnet
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | - | Path to `.pth` model checkpoint. |
| `--data` | Yes | - | Path to validation dataset. |
| `--output` | No | `evaluation_results` | Directory for results. |
| `--cpu` | No | Flag | Force CPU evaluation. |

**Output Files:**
- `confusion_matrix.png` - Per-class accuracy visualization
- `training_curves.png` - Loss and accuracy over epochs
- `metrics.json` - Precision, recall, F1 per class

---

### 5. Dataset Download Instructions

Show download and preparation instructions for datasets.

```bash
python train.py download --dataset <name>
```

| Dataset | Description |
|---------|-------------|
| `affectnet` | Facial emotion dataset (7 classes) |
| `fer2013` | Facial expression dataset (7 classes) |
| `posture` | Body language datasets (BoLD, MultiPosture, etc.) |

**Examples:**
```bash
python train.py download --dataset affectnet
python train.py download --dataset posture
```

---

### 6. Create Session Template

Create a folder structure for recording custom video sessions.

```bash
python train.py create-session --output data/sessions/user_01 --duration 60
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--output` | Yes | - | Session output directory. |
| `--duration` | No | `60` | Expected video duration in seconds. |

---

## Quick Reference

```bash
# Run demo
python main.py --demo

# Train emotion model
python train.py emotion --data data/affectnet --epochs 40

# Train posture model  
python train.py posture --data data/posture --epochs 50

# Evaluate model
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/affectnet

# Get dataset info
python train.py download --dataset posture
```
