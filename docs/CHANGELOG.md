---
layout: default
title: Changelog
---

# Recent Changes & Updates

Changelog for the Sentry Mental Health Assessment Framework.

---

## v0.3.4 (2025-12-21)

### New Features
- **Face Meshgrid Visualization** (`src/visualization/facemesh_visualizer.py`)
  - Replaced dot-based facial landmarks with full 468-point MediaPipe FaceMesh tesselation
  - ~2000 triangle connections for true mesh overlay
  - Color-coded facial regions:
    - 🟢 **Eyes** (green) - Tracks openness and blink rate
    - 🔵 **Eyebrows** (blue) - Tracks raising and frowning
    - 🔵 **Nose** (cyan) - Reference point
    - 🟣 **Lips/Mouth** (magenta) - Tracks smiles and opening
    - 🟠 **Face contour** (orange) - Tracks jaw movement
    - ⚪ **Irises** (white) - Eye tracking circles
  - Configurable render modes: `full`, `minimal`, `contours`
  - Adjustable opacity and line thickness

- **Landmark Caching** (`src/facial/facemesh_analyzer.py`)
  - FaceMesh landmarks are now cached after analysis
  - Avoids redundant processing (was running FaceMesh twice per frame)
  - `get_landmarks()` now returns cached landmarks from last `analyze()` call

### Improvements
- **Reduced Emotion Flickering** (`src/facial/postprocessor.py`)
  - Raised surprise detection threshold from 0.45 → 0.6
  - Increased temporal window from 10 → 15 frames
  - Increased hysteresis threshold from 3 → 5 frames
  - Less aggressive probability corrections (0.3 → 0.5)
  - Fast-switching only when confidence > 70%

### New Files
- `src/visualization/facemesh_visualizer.py` - FaceMeshVisualizer class with MeshConfig

### Files Changed
- `src/visualization/monitor.py` - Integrated meshgrid visualizer, fixed merge conflict
- `src/facial/facemesh_analyzer.py` - Added landmark caching, get_landmarks() method
- `src/facial/postprocessor.py` - Added get_face_landmarks(), improved thresholds
- `main.py` - Passes face_mesh_landmarks to monitor.update()

---

## v0.3.3 (2025-12-21)

### New Features
- **Monitor UI Redesign** (`src/visualization/monitor.py`)
  - Modern dark color palette with cyan/green/orange accents
  - Rounded corners on face bounding boxes and UI elements
  - Gradient-filled progress bars for metrics
  - Card-based layout for Assessment and Emotion sections
  - Tech-style corner accents on face detection box
  - Glow effects on pose landmarks
  - Wider sidebar (300px → 340px) for better readability

### New Helper Functions
- `_draw_rounded_rect()` - Rounded rectangles with fill or outline
- `_draw_gradient_bar()` - Horizontal gradient progress bars
- `_draw_card()` - Semi-transparent card containers

---

## v0.3.2 (2025-12-21)

### Bugfixes
- **Fixed low-light emotion recognition accuracy**
  - Added `LightingNormalization` class in `src/facial/emotion.py`
  - Applies adaptive **gamma correction** for dark images (brightness < 100)
  - Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for local contrast
  - Fixes issue where CK+ trained model misclassified emotions in low-light webcam conditions

### New Features
- **Emotion Post-Processing** (`src/facial/postprocessor.py`)
  - Added `EmotionPostProcessor` class to correct model predictions in real-world conditions
  - **Smile Detection**: Uses Haar cascade to detect smiles and override SAD predictions when smiling
  - **Mouth Curvature Analysis**: Analyzes mouth region to detect smile vs frown
  - **Temporal Smoothing**: Weighted moving average over 15 frames for stable predictions
  - **Hysteresis**: Requires 5 consistent frames before switching emotions (prevents flickering)
  - Automatically integrated into `MentalHealthPipeline`

### Technical Details
- Lighting: Detects mean image brightness and applies gamma 1.0–1.8 adaptively
- CLAHE with `clip_limit=3.0` and `tile_grid_size=(8, 8)` for enhanced facial feature contrast
- Preprocessing now: `ToPILImage → LightingNormalization → Resize → ToTensor → Normalize`
- Post-processing: `RawPrediction → SmileDetection → TemporalSmoothing → Hysteresis → FinalPrediction`

---

## v0.3.1 (2025-12-20)

### Bugfixes
- **Fixed train.py not using balanced training**
  - Added `--balance`, `--target-samples`, `--aggressive` flags
  - Auto-detects dataset type from path (fer2013, ck, or affectnet)
  - FER2013 now correctly uses 6 classes (disgust excluded)

- **Fixed channel mismatch error in FER2013 training**
  - All FER2013 transforms now output 1-channel grayscale (matching EmotionClassifier)
  - Fixed `get_fer2013_augmentation_transforms()` to use correct normalization

### New Features
- **Added CK+ (Cohn-Kanade) dataset support**
  - Download: `kaggle datasets download zhiguocui/ck-dataset`
  - Balances to 400 samples per class (via augmentation)
  - 6 emotion classes (contempt/disgust excluded)
  - Auto-detected from path containing "ck"

---

## v0.3.0 (2025-12-20)

### Major Changes

#### Prediction Architecture Refactor
- **Moved predictions from posture model to post-fusion**
  - `PostureTemporalModel` now returns only 512D embeddings (no longer classifies)
  - `MentalHealthClassifier` now has **6 prediction heads**:
    - Stress (low/moderate/high)
    - Depression (minimal/mild/moderate/severe)
    - Anxiety (minimal/mild/moderate/severe)
    - Posture Archetype (upright/slouched/open/closed)
    - Stress Indicator (calm/fidgeting/restless/stillness)
    - Trajectory (stable/deteriorating/improving)

#### Dataset Improvements
- **FER2013 Balanced Training**
  - Excluded 'disgust' class (poorly labeled)
  - Remapped to 6 classes matching AffectNet
  - Added `balance_classes` option (5000 samples/class)
  - Strong augmentation for oversampled data
  - Balanced validation set support
  
- **AffectNet CSV Support**
  - Added `labels.csv` loading in root directory
  - Multiple CSV formats supported:
    - `subDirectory,image,expression` (official)
    - `image,label` (simple)
    - `pth,label` (path-based)

#### Emotion Training Enhancements
- Added `--balance` flag for oversampling minority classes
- Added `--aggressive` flag for stronger augmentation
- Added `--target-samples` for custom sample count per class

### Files Changed
- `src/posture/temporal_model.py` - Removed classifier heads
- `src/prediction/classifier.py` - Added 6-head classifier
- `src/config.py` - Added posture prediction configs
- `training/datasets/fer2013.py` - Balanced training, no disgust
- `training/datasets/affectnet.py` - CSV support
- `training/datasets/transforms.py` - Aggressive transforms
- `training/trainers/emotion_trainer.py` - Balance flags
- `tests/test_integration.py` - Updated for new prediction fields

### Documentation
- Created `docs/THEORY.md` - Comprehensive research explanations
- Updated `docs/ARCHITECTURE.md` - 6-head classifier
- Updated `docs/TRAINING.md` - Balanced training options

---

## v0.2.0 (2025-12-19)

### Posture Model Training
- Added multi-task posture training (posture/stress/trajectory)
- Fixed training stability issues (gradient clipping, LayerNorm)
- Added video posture dataset download script
- Support for MultiPosture, NTU RGB+D, Figshare datasets

### Documentation
- Created comprehensive TRAINING.md guide
- Created GETTING_STARTED.md
- Created COMMANDS.md with all CLI commands

---

## v0.1.0 (2025-12-17)

### Initial Release
- Multimodal mental health assessment framework
- Facial emotion recognition (DenseNet121)
- Posture analysis (TCN-LSTM)
- Cross-attention fusion
- Heuristic and neural predictors
- Real-time visualization demo
