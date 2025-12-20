# Getting Started

Step-by-step guide to set up and run Sentry.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Download Datasets](#3-download-datasets)
4. [Run Demo](#4-run-demo)
5. [Train Models](#5-train-models)
6. [Next Steps](#6-next-steps)

---

## 1. Prerequisites

### Required

- **Python 3.8+** (3.10 recommended)
- **pip** package manager

### Recommended

- **CUDA GPU** (NVIDIA) for real-time processing
- **Webcam** for live demo
- **Kaggle account** for dataset downloads

---

## 2. Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/sentry.git
cd sentry
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x, CUDA: True
```

---

## 3. Download Datasets

### Emotion Dataset (FER2013)

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/

# Option 2: Manual
# Go to https://www.kaggle.com/datasets/msambare/fer2013
# Download and extract to data/fer2013/
```

### Posture Datasets

```bash
python scripts/download_video_posture_datasets.py --dataset all
```

---

## 4. Run Demo

### Webcam Demo

```bash
python main.py --demo
```

**Controls:**
- `Q` - Quit
- `S` - Screenshot
- `R` - Reset tracking

### Process Video File

```bash
python main.py --video path/to/video.mp4
```

### With Trained Model

```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

---

## 5. Train Models

### Quick Start - Emotion Model

```bash
# Download dataset first
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/

# Train with balanced classes
python train.py emotion --data data/fer2013 --epochs 40 --balance --aggressive
```

This will:
- Load FER2013 dataset (6 emotion classes)
- Balance to 5000 samples per class
- Apply strong augmentation
- Train for 40 epochs
- Save best model to `models/emotion_trained/`

### Check Training Progress

The training will show:
```
FER2013 train Loaded: 28273 total - {'angry': 3995, 'fear': 4097, ...}
Balancing to 5000 samples per class...
FER2013 train Balanced: 30000 total - {'angry': 5000, 'fear': 5000, ...}

Epoch 1: 100%|██████| 468/468 [02:30<00:00] loss=1.45 acc=35.2
Epoch 2: 100%|██████| 468/468 [02:28<00:00] loss=1.12 acc=52.1
...
```

---

## 6. Next Steps

| Goal | Documentation |
|------|---------------|
| Train better models | [TRAINING.md](TRAINING.md) |
| Understand the AI | [THEORY.md](THEORY.md) |
| All commands | [COMMANDS.md](COMMANDS.md) |
| System architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Optimize performance | [PERFORMANCE.md](PERFORMANCE.md) |

---

## Troubleshooting

### "CUDA not available"

1. Install NVIDIA drivers
2. Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "MediaPipe error"

```bash
pip install --upgrade mediapipe
```

### "No webcam found"

```bash
# Try different camera ID
python main.py --demo --camera 1
```

### "Out of GPU memory"

```bash
# Reduce batch size
python train.py emotion --data data/fer2013 --batch-size 32
```
