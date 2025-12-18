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

---

## Training Utilities (`train.py`)

Utilities for training models, evaluating performance, and managing datasets.

### 1. Train Emotion Model

Fine-tune the facial emotion classifier.

```bash
python train.py emotion [args]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data` | Yes | - | Path to dataset directory (e.g., `data/affectnet`). |
| `--dataset` | No | `affectnet` | Dataset type (`affectnet` or `fer2013`). |
| `--output` | No | `models/emotion_trained` | Directory to save trained models. |
| `--epochs` | No | `20` | Number of training epochs. |
| `--batch-size` | No | `32` | Training batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--cpu` | No | `False` | Force CPU training. |

**Example:**
```bash
python train.py emotion --data data/affectnet --epochs 20
```

### 2. Evaluate Model

Generate metrics, confusion matrices, and plots for a trained model.

```bash
python train.py evaluate [args]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | - | Path to `.pth` model checkpoint. |
| `--data` | Yes | - | Path to validation dataset. |
| `--output` | No | `evaluation_results` | Directory for results. |

**Example:**
```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/affectnet
```

### 3. Data Tools

**Download Information:**
Show instructions for downloading datasets.
```bash
python train.py download --dataset affectnet
```

**Create Session Template:**
Create a folder structure for recording custom video sessions.
```bash
python train.py create-session --output data/sessions/user_01 --duration 60
```
