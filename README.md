# Sentry: Multimodal Mental Health Assessment Framework

**Sentry** is a sophisticated deep learning system that performs real-time mental health assessment by analyzing facial expressions and body posture. It fuses visual cues to detect stress, depression, and anxiety indicators while prioritizing privacy through local processing.

![Dashboard](https://via.placeholder.com/800x450?text=Sentry+Dashboard+Preview)

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Installation, setup, and running the demo.
- **[Performance Guide](docs/PERFORMANCE.md)**: Optimization tips for maximum FPS on your hardware.
- **[Command Reference](docs/COMMANDS.md)**: Full list of CLI commands and arguments.
- **[Training Guide](docs/TRAINING.md)**: How to train emotion models and custom classifiers.
- **[Architecture](docs/ARCHITECTURE.md)**: Deep dive into the fusion network, TCN-LSTM models, and heuristic predictors.

## 🚀 Quick Start

1. **Install**:
   ```bash
   pip install -r requirements.txt
   ```
   *(See [Getting Started](docs/GETTING_STARTED.md) for model setup)*

2. **Run Demo**:
   ```bash
   python main.py --demo
   ```

3. **Train Emotion Model**:
   ```bash
   python train.py emotion --data data/affectnet --epochs 20
   ```

## ✨ Key Features

- **Multimodal AI**: Combines MobileNetV3 (Face) and MediaPipe (Pose) with Cross-Attention Fusion.
- **Real-time Assessment**: 20-30 FPS processing with GPU acceleration (optimized for RTX GPUs).
- **GPU Accelerated**: MediaPipe GPU delegate support for faster pose estimation.
- **Smart Prediction**: Heuristic and Neural predictors for Stress, Depression, and Anxiety.
- **Privacy First**: 100% local processing; no video leaves your machine.
- **Performance Modes**: Pre-configured settings for different hardware capabilities.

## 📁 Project Structure

```
sentry/
├── docs/                # Comprehensive documentation
├── src/                 # Source code
│   ├── facial/          # Face detection & emotion recognition
│   ├── posture/         # Pose estimation & temporal analysis
│   ├── fusion/          # Multimodal fusion network
│   ├── prediction/      # Heuristic & Neural predictors
│   ├── visualization/   # Real-time dashboard
├── models/              # Saved model checkpoints
├── data/                # Training datasets
├── main.py              # Application entry point
└── train.py             # Training CLI
```

## 🛠️ Requirements

- Python 3.8+
- CUDA GPU (Recommended)
- MediaPipe Task Models (see installation guide)
