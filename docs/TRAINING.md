# Training Guide

Sentry allows you to fine-tune the emotion recognition models and train custom classifiers.

## 1. Datasets

We support **AffectNet** and **FER2013** datasets.

### Downloading AffectNet

1. **Via Kaggle CLI** (Recommended):
   ```bash
   kaggle datasets download -d mstjebashazida/affectnet
   # Extract to data/affectnet
   ```

2. **Manual Download**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/mstjebashazida/affectnet)
   - Extract so that `data/affectnet/` contains `train/` and `val/` folders.

## 2. Training the Emotion Model

Fine-tune the MobileNetV3 backbone on AffectNet:

```bash
python train.py emotion --data data/affectnet --epochs 20 --batch-size 32
```

- **Output**: Trained models are saved to `models/emotion_trained/`
- **Metrics**: `best_model.pth` (best validation accuracy) and `final_model.pth`.

## 3. Evaluating Performance

Generate confusion matrices, training curves, and per-class metrics:

```bash
python train.py evaluate --model models/emotion_trained/best_model.pth --data data/affectnet
```

Results are saved in `evaluation_results/`.

## 4. Using Trained Models

To use your fine-tuned model in the live application:

```bash
python main.py --demo --trained-model models/emotion_trained/best_model.pth
```

## 5. Custom Data Collection

You can record your own sessions to train the mental health classifier heads.

1. **Create Session Template**:
   ```bash
   python train.py create-session --output data/sessions/user_01 --duration 60
   ```
   
2. **Record**:
   The system currently supports training on extracted feature vectors. (Future update: Integration with live recording tool).
