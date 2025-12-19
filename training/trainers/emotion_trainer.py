# Emotion Classifier Trainer
# Fine-tune MobileNetV3 on emotion datasets

import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime


class EmotionTrainer:
    """
    Trainer for fine-tuning emotion classification model.
    
    Fine-tunes MobileNetV3 backbone on AffectNet/FER2013 datasets.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Emotion classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            learning_rate: Initial learning rate
            freeze_backbone: Freeze backbone, train only classifier
            class_weights: Class weights for imbalanced data
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Loss function with higher label smoothing to reduce overconfidence
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)  # Increased from 0.1
        
        # Optimizer - stronger weight decay for regularization
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.05)  # Increased from 0.01
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        
        # Unfreeze last few layers for fine-tuning
        if hasattr(self.model, 'backbone'):
            # Unfreeze last 2 blocks of MobileNetV3
            # for name, param in self.model.backbone.named_parameters():
            #     if 'features.16' in name or 'features.15' in name:
            #         param.requires_grad = True
            
            # Unfreeze last block of DenseNet121
            for name, param in self.model.backbone.named_parameters():
                if 'features.denseblock4' in name or 'features.norm5' in name:
                    param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize GradScaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('emotion', None))
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('emotion', None))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(
        self,
        epochs: int = 20,
        save_dir: Optional[str] = None,
        early_stopping: Optional[int] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            early_stopping: Stop if no improvement for N epochs
            
        Returns:
            Training history
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        no_improvement = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                no_improvement = 0
                
                if save_dir:
                    self.save_checkpoint(save_path / 'best_model.pth')
                    print(f"  -> New best model saved! (acc: {val_acc:.2f}%)")
            else:
                no_improvement += 1
            
            # Save periodic checkpoint
            if save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch+1}.pth')
                print(f"  -> Periodic checkpoint saved at epoch {epoch+1}")
            
            # Early stopping
            if early_stopping is not None and no_improvement >= early_stopping:
                print(f"Early stopping after {early_stopping} epochs without improvement")
                break
        
        # Save final model and history
        if save_dir:
            self.save_checkpoint(save_path / 'final_model.pth')
            with open(save_path / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']


def train_emotion_model(
    data_dir: str,
    output_dir: str = 'models/emotion_trained',
    dataset: str = 'affectnet',
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device: str = 'cuda',
    num_workers: int = 4
) -> Dict:
    """
    Main training function.
    
    Args:
        data_dir: Path to dataset
        output_dir: Path to save trained model
        dataset: 'affectnet' or 'fer2013'
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Training history
    """
    from ..datasets import AffectNetDataset, FER2013Dataset
    from ..datasets.affectnet import create_affectnet_loaders
    from ..datasets.fer2013 import create_fer2013_loaders
    
    # Create data loaders
    if dataset == 'affectnet':
        train_loader, val_loader = create_affectnet_loaders(
            data_dir, 
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=6
        )
        num_classes = 6
    else:
        train_loader, val_loader = create_fer2013_loaders(
            data_dir, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        num_classes = 7
    
    # Create model
    from src.facial.emotion import EmotionClassifier
    from src.config import FacialConfig
    
    # Update emotion classes based on num_classes
    # 0:neutral, 1:happy, 2:sad, 3:surprise, 4:fear, 5:anger
    emotion_classes = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'anger']
    if num_classes != 6:
        print(f"Warning: Expected 6 classes, got {num_classes}. Classes might be misaligned.")
    
    config = FacialConfig(emotion_classes=emotion_classes)
    model = EmotionClassifier(config)
    
    # Get class weights for imbalanced data
    class_weights = None
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device if torch.cuda.is_available() else 'cpu',
        learning_rate=learning_rate,
        freeze_backbone=True,
        class_weights=class_weights
    )
    
    # Train with early stopping to prevent overfitting
    history = trainer.train(
        epochs=epochs,
        save_dir=output_dir,
        early_stopping=7  # Stop if no improvement for 7 epochs
    )
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emotion classifier')
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--output', type=str, default='models/emotion_trained')
    parser.add_argument('--dataset', type=str, default='affectnet', 
                        choices=['affectnet', 'fer2013'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    history = train_emotion_model(
        data_dir=args.data,
        output_dir=args.output,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,

        learning_rate=args.lr,
        num_workers=4
    )
    
    print(f"\nTraining complete! Best validation accuracy: {max(history['val_acc']):.2f}%")
