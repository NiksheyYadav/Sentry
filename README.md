# Sentry: Multimodal Mental Health Assessment Framework

**Sentry** is a sophisticated deep learning system that performs real-time mental health assessment by analyzing facial expressions and body posture. It fuses visual cues to detect stress, depression, and anxiety indicators while prioritizing privacy through local processing.

## Key Features

- **Multimodal AI**: Combines DenseNet121 (Face) and MediaPipe (Pose) with Cross-Attention Fusion
- **Real-time Assessment**: 20-30 FPS processing with GPU acceleration
- **Multi-Task Learning**: Emotion, posture, stress, and trajectory prediction
- **Privacy First**: 100% local processing - no data leaves your machine
- **Anti-Overfitting**: Label smoothing, dropout, data augmentation, early stopping

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python main.py --demo
```

### 3. Train Models

**Emotion Model** (facial expressions):
```bash
python train.py emotion --data data/affectnet --epochs 40
```

**Posture Model** (body language):
```bash
python train.py posture --data data/posture --epochs 50
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Installation, setup, and running the demo |
| [Training Guide](docs/TRAINING.md) | Train emotion and posture models |
| [Command Reference](docs/COMMANDS.md) | Full CLI command documentation |
| [Architecture](docs/ARCHITECTURE.md) | System design and model architecture |
| [Performance](docs/PERFORMANCE.md) | Optimization for different hardware |

## Model Architecture

```
Input (Video Frame)
       |
       v
+------+------+
|             |
v             v
[Face]     [Pose]
DenseNet   MediaPipe
   |           |
   v           v
Emotion    TCN-LSTM
Classifier Temporal
   |           |
   +-----+-----+
         |
         v
   Cross-Attention
      Fusion
         |
         v
   Mental Health
    Prediction
```

## Training Capabilities

### Emotion Recognition (6 classes)
- neutral, happy, sad, surprise, fear, anger
- Dataset: AffectNet or FER2013
- Anti-overfitting: Label smoothing (0.15), Weight decay (0.05), Dropout (0.4-0.5)

### Posture Analysis (Multi-Task)
| Task | Classes |
|------|---------|
| **Body Language** | upright, slouched, open, closed |
| **Stress Indicators** | calm, fidgeting, restless, stillness |
| **Trajectory** | stable, deteriorating, improving |

Datasets: BoLD, MultiPosture, Stress Dataset

## Project Structure

```
sentry/
+-- docs/                 # Documentation
+-- src/
|   +-- facial/           # Face detection & emotion
|   +-- posture/          # Pose estimation & temporal
|   +-- fusion/           # Multimodal fusion
|   +-- prediction/       # Mental health prediction
|   +-- visualization/    # Real-time dashboard
+-- training/
|   +-- datasets/         # Dataset loaders
|   +-- trainers/         # Training utilities
+-- models/               # Saved checkpoints
+-- data/                 # Training datasets
+-- main.py               # Application entry
+-- train.py              # Training CLI
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (Recommended for real-time)
- MediaPipe

## License

MIT License - See LICENSE file for details.
