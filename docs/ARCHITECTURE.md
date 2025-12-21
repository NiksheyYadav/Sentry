# System Architecture

Technical overview of Sentry's multimodal mental health assessment pipeline.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Facial Analysis](#2-facial-analysis)
3. [Posture Analysis](#3-posture-analysis)
4. [Fusion Engine](#4-fusion-engine)
5. [Prediction Heads](#5-prediction-heads)
6. [File Structure](#6-file-structure)

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (30 FPS)                     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     FACIAL BRANCH       │     │     POSTURE BRANCH      │
├─────────────────────────┤     ├─────────────────────────┤
│ 1. MTCNN Face Detection │     │ 1. MediaPipe Pose       │
│ 2. Lighting Normalize   │     │ 2. Feature Extraction   │
│ 3. DenseNet121 Backbone │     │    (15 features)        │
│ 4. Emotion Classification│    │ 3. TCN Encoder          │
│ 5. 512D Embedding       │     │ 4. LSTM Temporal        │
└─────────────────────────┘     │ 5. 512D Embedding       │
              │                 └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
              ┌─────────────────────────────────┐
              │       FUSION ENGINE             │
              ├─────────────────────────────────┤
              │ • Cross-Attention (8 heads)     │
              │ • Feature Concatenation         │
              │ • 1024D Fused Embedding         │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │     MENTAL HEALTH CLASSIFIER    │
              ├─────────────────────────────────┤
              │ Shared Backbone (512D)          │
              │         │                       │
              │    ┌────┴────┬────┬────┬────┐   │
              │    ▼    ▼    ▼    ▼    ▼    ▼   │
              │ Stress Dep. Anx. Post.Stress Traj│
              └─────────────────────────────────┘
```

---

## 2. Facial Analysis

### Face Detection: MTCNN

Multi-task Cascaded Convolutional Networks for robust face detection.

| Stage | Purpose |
|-------|---------|
| P-Net | Proposal generation |
| R-Net | Refine bounding boxes |
| O-Net | Final detection + landmarks |

### Emotion Classification: DenseNet121

**Architecture:**
- Backbone: DenseNet121 (pre-trained on ImageNet)
- Modified: First conv layer accepts 1-channel grayscale
- Output: 6-class emotion + 512D embedding

**Preprocessing Pipeline:**
1. Convert to PIL Image
2. **Lighting Normalization** (CLAHE + adaptive gamma correction for low-light)
3. Resize to 224×224
4. Normalize (mean=0.5, std=0.5)

**Input:** 224×224 grayscale image (lighting-normalized)  
**Output:** `(emotion_logits, embedding)`

**Source:** `src/facial/emotion.py`

### FaceMesh Expression Analysis

MediaPipe FaceMesh provides 468 facial landmarks for precise expression detection.

**Used For:**
- Expression validation (smile, surprise, frown detection)
- Emotion post-processing corrections
- Real-time meshgrid visualization

**Expression Features:**
| Feature | Landmarks Used | Purpose |
|---------|----------------|---------|
| Eye Aspect Ratio (EAR) | 6 points per eye | Blink/openness detection |
| Eyebrow Raise | 10 points per brow | Surprise/fear detection |
| Mouth Openness | Lip landmarks | Surprise/speech detection |
| Smile Score | Mouth corners | Happy detection |

**Visualization Colors:**
- 🟢 **Eyes** (green) - Tracks openness
- 🔵 **Eyebrows** (blue) - Tracks raising
- 🟣 **Lips** (magenta) - Tracks smiles
- 🟠 **Face contour** (orange) - Tracks jaw

**Source:** `src/facial/facemesh_analyzer.py`, `src/visualization/facemesh_visualizer.py`

---

## 3. Posture Analysis

### Pose Estimation: MediaPipe

Extracts 33 body landmarks in real-time.

### Feature Extraction

From landmarks, we compute **15 features**:

| Category | Features |
|----------|----------|
| **Geometric** | Spine curvature, head tilt, shoulder slope |
| **Angular** | Arm angles, shoulder asymmetry |
| **Velocity** | Movement speed, acceleration |
| **Derived** | Kinetic energy, stillness duration |

**Source:** `src/posture/features.py`

### Temporal Model: TCN-LSTM

```
Input: (batch, 30 frames, 15 features)
           │
           ▼
    ┌──────────────┐
    │ TCN Encoder  │  Dilated convolutions
    │ 64→128→256   │  Multi-scale patterns
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │    LSTM      │  Temporal dependencies
    │ Hidden: 128  │  State: improving/deteriorating
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  Projection  │
    │    512D      │
    └──────────────┘
```

**Source:** `src/posture/temporal_model.py`

---

## 4. Fusion Engine

### Cross-Attention Mechanism

Allows each modality to attend to relevant parts of the other.

```python
# Facial features attend to posture
Q = FacialEmb @ W_query  
K = PostureEmb @ W_key
V = PostureEmb @ W_value
Attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**Configuration:**
- Attention heads: 8
- Facial embed dim: 512
- Posture embed dim: 512
- Fused dim: 1024

**Source:** `src/fusion/fusion.py`

---

## 5. Prediction Heads

The classifier has **6 prediction heads** operating on the 1024D fused embedding:

| Head | Classes | Purpose |
|------|---------|---------|
| **Stress** | low, moderate, high | Overall stress level |
| **Depression** | minimal, mild, moderate, severe | Depression indicators |
| **Anxiety** | minimal, mild, moderate, severe | Anxiety indicators |
| **Posture** | upright, slouched, open, closed | Body language state |
| **Stress Indicator** | calm, fidgeting, restless, stillness | Movement patterns |
| **Trajectory** | stable, deteriorating, improving | Temporal trend |

**Uncertainty Estimation:**
- Monte Carlo Dropout (10 samples)
- Temperature scaling (1.5)

**Source:** `src/prediction/classifier.py`

---

## 6. File Structure

```
src/
├── config.py                 # All configuration dataclasses
│
├── facial/
│   ├── emotion.py            # EmotionClassifier (DenseNet121)
│   ├── detector.py           # BlazeFace face detector
│   ├── facemesh_analyzer.py  # MediaPipe FaceMesh 468 landmarks
│   ├── postprocessor.py      # Emotion post-processing & temporal smoothing
│   └── action_units.py       # Action Unit detection
│
├── posture/
│   ├── pose_estimator.py     # MediaPipe wrapper
│   ├── features.py           # Feature extraction
│   └── temporal_model.py     # TCN-LSTM model
│
├── fusion/
│   └── fusion_network.py     # Cross-attention fusion
│
├── prediction/
│   ├── classifier.py         # 6-head mental health classifier
│   ├── heuristic.py          # Rule-based fallback
│   └── calibration.py        # Alert system
│
├── video/
│   └── capture.py            # Video capture utilities
│
└── visualization/
    ├── monitor.py            # Real-time monitoring dashboard
    └── facemesh_visualizer.py # Face meshgrid overlay (468 landmarks)
```

---

## Configuration

All model parameters are configured in `src/config.py`:

```python
@dataclass
class FacialConfig:
    emotion_classes: List[str]  # 6 emotions
    embedding_dim: int = 1280
    
@dataclass
class PostureConfig:
    input_dim: int = 15         # 15 posture features
    tcn_channels: List[int] = [64, 128, 256]
    lstm_hidden_size: int = 128
    
@dataclass
class FusionConfig:
    facial_embed_dim: int = 512
    posture_embed_dim: int = 512
    fused_dim: int = 1024
    attention_heads: int = 8
```
