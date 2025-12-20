# Model Loader Utility
# Load trained models into the main pipeline

import torch
from pathlib import Path
from typing import Optional


def load_trained_emotion_model(
    checkpoint_path: str = "models/emotion_trained/best_model.pth",
    device: str = "cuda"
) -> "EmotionClassifier":
    """
    Load a trained emotion classifier for use in the main pipeline.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load model onto
        
    Returns:
        Loaded EmotionClassifier ready for inference
    """
    from src.facial.emotion import EmotionClassifier
    from src.config import FacialConfig
    
    # Check device
    device = device if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading trained emotion model from {checkpoint_path}")
    
    # Create model
    config = FacialConfig()
    model = EmotionClassifier(config)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded emotion model successfully!")
    
    return model


def get_training_info(checkpoint_path: str) -> dict:
    """
    Get information about a trained model.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary with training info
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_acc': checkpoint.get('best_val_acc', 'unknown'),
    }
    
    if 'history' in checkpoint:
        history = checkpoint['history']
        if history.get('val_acc'):
            info['final_val_acc'] = history['val_acc'][-1]
            info['best_val_acc'] = max(history['val_acc'])
        if history.get('train_acc'):
            info['final_train_acc'] = history['train_acc'][-1]
    
    return info
