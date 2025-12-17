# Training Pipeline Implementation Plan

## Overview

Add training capabilities to the mental health assessment framework, enabling fine-tuning on labeled datasets and continuous improvement through active learning.

---

## Dataset Options

### Publicly Available (Immediate Access)

| Dataset | Size | Labels | How to Get |
|---------|------|--------|------------|
| **FER2013** | 35K images | 7 emotions | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) - Free download |
| **CK+** | 593 sequences | 7 emotions | [Request form](http://www.jeffcohn.net/Resources/) |
| **RAF-DB** | 30K images | 7 emotions | [Website](http://www.whdeng.cn/raf/model1.html) |

### Academic Request Required

| Dataset | Size | Labels | How to Get |
|---------|------|--------|------------|
| **AffectNet** | 400K images | 8 emotions + valence/arousal | [Request](http://mohammadmahoor.com/affectnet/) |
| **AVEC 2014** | 300 videos | Depression (BDI-II) | [Request](https://avec-db.sspnet.eu/) |
| **DAIC-WOZ** | 189 sessions | PHQ-8 depression scores | [Request](https://dcapswoz.ict.usc.edu/) |
| **EmoPain** | 18 subjects | Pain/emotion | [Request](https://www.ucl.ac.uk/uclic/research/emopain) |

### Create Your Own (Recommended for Domain Specificity)

Self-collected data with simple labeling:
- Record 10-minute sessions of students/volunteers
- Label stress/focus levels every minute (1-5 scale)
- ~50 sessions = good starting point

---

## Proposed Changes

### Training Module Structure

```
c:\sentry\
├── training/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── fer2013.py         # FER2013 emotion loader
│   │   ├── affectnet.py       # AffectNet loader
│   │   ├── video_dataset.py   # Custom video dataset
│   │   └── transforms.py      # Data augmentation
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── emotion_trainer.py # Fine-tune emotion model
│   │   ├── fusion_trainer.py  # Train fusion layers
│   │   └── classifier_trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Accuracy, F1, confusion
│   │   └── validation.py      # Cross-validation
│   └── active_learning/
│       ├── __init__.py
│       ├── uncertainty.py     # Sample uncertain cases
│       └── labeling_ui.py     # Simple labeling interface
```

---

### File Specifications

#### [NEW] [fer2013.py](file:///c:/sentry/training/datasets/fer2013.py)
- Load FER2013 CSV format
- 48x48 grayscale → 224x224 RGB conversion
- Train/val/test splits
- Emotion label mapping

#### [NEW] [video_dataset.py](file:///c:/sentry/training/datasets/video_dataset.py)
- Load video clips with per-second labels
- Extract frames at configurable FPS
- Return (facial_features, posture_features, labels)

#### [NEW] [emotion_trainer.py](file:///c:/sentry/training/trainers/emotion_trainer.py)
- Fine-tune MobileNetV3 on emotion data
- Freeze backbone, train classifier
- Learning rate scheduling
- Early stopping

#### [NEW] [classifier_trainer.py](file:///c:/sentry/training/trainers/classifier_trainer.py)
- Train mental health classification heads
- Multi-task loss (stress + depression + anxiety)
- Class weighting for imbalance

#### [NEW] [uncertainty.py](file:///c:/sentry/training/active_learning/uncertainty.py)
- Identify low-confidence predictions
- Rank by uncertainty score
- Export for human labeling

---

## Quick Start Path

### Week 1: FER2013 Emotion Training
```bash
# Download FER2013 from Kaggle
kaggle datasets download -d msambare/fer2013

# Train emotion classifier
python -m training.trainers.emotion_trainer --epochs 20
```

### Week 2: Self-Labeled Data Collection
- Record ~50 sessions (volunteers or self)
- Simple labels: "calm", "mildly stressed", "stressed"

### Week 3: Train Classification Heads
```bash
python -m training.trainers.classifier_trainer --data custom_sessions/
```

---

## Verification Plan

### Metrics to Track
- Emotion classification accuracy (target: >70% on FER2013)
- Mental health prediction correlation with self-reports
- False positive/negative rates for alerts

### Validation Approach
- 5-fold cross-validation on collected data
- Separate test set (20% holdout)
- Real-world testing with feedback
