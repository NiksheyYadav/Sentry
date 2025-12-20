# System Architecture

Sentry is a multimodal mental health assessment framework that fuses facial and posture analysis.

## Core Pipeline

### 1. Video Acquisition
- **Capture**: 30 FPS via OpenCV
- **Buffering**: Circular buffer maintains recent history
- **Frame Management**: `src/video/`

### 2. Facial Analysis (`src/facial/`)
- **Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Emotion Recognition**: DenseNet121 (Pre-trained/Fine-tuned)
- **Action Units**: Detection of specific facial muscle movements
- **Temporal Aggregation**: Rolling window analysis of emotion stability

### 3. Posture Analysis (`src/posture/`)
- **Pose Estimation**: MediaPipe Pose (Tasks API)
- **Feature Extraction**: 
  - **Geometric**: Spine curvature, head tilt, shoulder symmetry
  - **Movement**: Fidgeting, restlessness, total kinetic energy
- **Temporal Model**: TCN (Temporal Convolutional Network) + LSTM for embedding extraction

### 4. Fusion Engine (`src/fusion/`)
- **Cross-Attention**: Bidirectional attention mechanism matches facial cues with body language
- **Feature Fusion**: Concatenates weighted features into a 1024D vector
- **Dynamic Weighting**: Assigns importance scores to each modality

### 5. Prediction & Assessment (`src/prediction/`)

#### Multi-Head Neural Classifier
Six-headed classifier operating on fused features (1024D):

| Head | Classes | Description |
|------|---------|-------------|
| **Stress** | Low, Moderate, High | Overall stress level |
| **Depression** | Minimal, Mild, Moderate, Severe | Depressive indicators |
| **Anxiety** | Minimal, Mild, Moderate, Severe | Anxiety indicators |
| **Posture Archetype** | Upright, Slouched, Open, Closed | Body language state |
| **Stress Indicator** | Calm, Fidgeting, Restless, Stillness | Movement patterns |
| **Trajectory** | Stable, Deteriorating, Improving | Temporal trend |

#### Heuristic Predictor (Fallback)
- Supports Monte Carlo Dropout for uncertainty estimation (Bayesian approximation)

## File Structure

- `main.py`: Entry point and pipeline orchestration
- `train.py`: Training and evaluation CLI
- `src/config.py`: Centralized configuration using dataclasses
- `src/utils/`: Helper utilities (model loading, math)

## Data Flow

```mermaid
graph LR
    Camera --> Video[Video Capture]
    Video --> Face[Face Detection]
    Video --> Pose[Pose Estimation]
    
    Face --> Emotion[Emotion Classifier]
    Pose --> Features[Posture Features]
    
    Emotion --> Fusion
    Features --> Fusion
    
    Fusion --> Predictor[Heuristic/Neural Predictor]
    Predictor --> Alert[Alert System]
    Predictor --> UI[Visualization]
```
