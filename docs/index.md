# Sentry Documentation

Welcome to the Sentry documentation.

## Core Guides

- **[Getting Started](GETTING_STARTED.md)**
  - Installation guide
  - Setting up the prerequisite models
  - Running the demo

- **[Training Guide](TRAINING.md)**
  - Dataset preparation (AffectNet/FER2013)
  - Training the emotion classifier
  - Evaluating model performance

- **[System Architecture](ARCHITECTURE.md)**
  - Understanding the pipeline
  - Fusion network details
  - Heuristic vs Neural prediction

## API Reference

The source code is organized into the following modules:

- `src.video`: Video capture and buffering
- `src.facial`: Face detection and expression analysis
- `src.posture`: Body pose estimation and feature extraction
- `src.fusion`: Multimodal data fusion
- `src.prediction`: Mental health assessment logic
