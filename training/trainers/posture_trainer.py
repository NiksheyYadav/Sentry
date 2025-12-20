# Posture Temporal Model Trainer
# Train TCN-LSTM model for body language and stress detection

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime


class PostureTrainer:
    """
    Trainer for posture temporal model.
    
    Trains multi-task model for:
    - Posture classification (upright/slouched/open/closed)
    - Stress indicator detection (calm/fidgeting/restless/stillness)
    - Trajectory prediction (stable/deteriorating/improving)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        posture_weight: float = 1.0,
        stress_weight: float = 1.0,
        trajectory_weight: float = 1.0,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PostureTemporalModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            learning_rate: Initial learning rate
            posture_weight: Loss weight for posture task
            stress_weight: Loss weight for stress task
            trajectory_weight: Loss weight for trajectory task
            class_weights: Per-class weights for each task
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # Task weights
        self.posture_weight = posture_weight
        self.stress_weight = stress_weight
        self.trajectory_weight = trajectory_weight
        
        # Loss functions - don't use class weights if they cause numerical issues
        # (e.g., when all samples have the same label)
        self.posture_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.stress_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.trajectory_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.05
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'posture_acc': [], 'stress_acc': [], 'trajectory_acc': []
        }
        
        # Enable automatic mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = {'posture': 0, 'stress': 0, 'trajectory': 0}
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(self.device)
            posture_labels = labels['posture'].to(self.device)
            stress_labels = labels['stress'].to(self.device)
            trajectory_labels = labels['trajectory'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                # Get model outputs
                embedding, trajectory_logits, _ = self.model(sequences)
                
                # We need additional heads for posture and stress
                # These will be added to the model
                posture_logits = self.model.posture_classifier(embedding)
                stress_logits = self.model.stress_classifier(embedding)
                
                # Compute losses
                posture_loss = self.posture_criterion(posture_logits, posture_labels)
                stress_loss = self.stress_criterion(stress_logits, stress_labels)
                trajectory_loss = self.trajectory_criterion(trajectory_logits, trajectory_labels)
                
                # Combined loss
                loss = (self.posture_weight * posture_loss + 
                       self.stress_weight * stress_loss + 
                       self.trajectory_weight * trajectory_loss)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            total += posture_labels.size(0)
            
            _, pred_posture = posture_logits.max(1)
            _, pred_stress = stress_logits.max(1)
            _, pred_trajectory = trajectory_logits.max(1)
            
            correct['posture'] += pred_posture.eq(posture_labels).sum().item()
            correct['stress'] += pred_stress.eq(stress_labels).sum().item()
            correct['trajectory'] += pred_trajectory.eq(trajectory_labels).sum().item()
            
            # Update progress bar
            avg_acc = sum(correct.values()) / (3 * total) * 100
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': f'{avg_acc:.1f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = sum(correct.values()) / (3 * total) * 100
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[str, float]]:
        """Validate on validation set."""
        if self.val_loader is None:
            return 0.0, 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        correct = {'posture': 0, 'stress': 0, 'trajectory': 0}
        total = 0
        
        for sequences, labels in self.val_loader:
            sequences = sequences.to(self.device)
            posture_labels = labels['posture'].to(self.device)
            stress_labels = labels['stress'].to(self.device)
            trajectory_labels = labels['trajectory'].to(self.device)
            
            # Forward pass
            embedding, trajectory_logits, _ = self.model(sequences)
            posture_logits = self.model.posture_classifier(embedding)
            stress_logits = self.model.stress_classifier(embedding)
            
            # Compute losses
            posture_loss = self.posture_criterion(posture_logits, posture_labels)
            stress_loss = self.stress_criterion(stress_logits, stress_labels)
            trajectory_loss = self.trajectory_criterion(trajectory_logits, trajectory_labels)
            
            loss = (self.posture_weight * posture_loss + 
                   self.stress_weight * stress_loss + 
                   self.trajectory_weight * trajectory_loss)
            
            total_loss += loss.item()
            total += posture_labels.size(0)
            
            _, pred_posture = posture_logits.max(1)
            _, pred_stress = stress_logits.max(1)
            _, pred_trajectory = trajectory_logits.max(1)
            
            correct['posture'] += pred_posture.eq(posture_labels).sum().item()
            correct['stress'] += pred_stress.eq(stress_labels).sum().item()
            correct['trajectory'] += pred_trajectory.eq(trajectory_labels).sum().item()
        
        task_accs = {
            'posture': 100.0 * correct['posture'] / total,
            'stress': 100.0 * correct['stress'] / total,
            'trajectory': 100.0 * correct['trajectory'] / total
        }
        
        return (
            total_loss / len(self.val_loader),
            sum(correct.values()) / (3 * total) * 100,
            task_accs
        )
    
    def train(
        self,
        epochs: int = 50,
        save_dir: Optional[str] = None,
        early_stopping: int = 10
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
            val_loss, val_acc, task_accs = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['posture_acc'].append(task_accs.get('posture', 0))
            self.history['stress_acc'].append(task_accs.get('stress', 0))
            self.history['trajectory_acc'].append(task_accs.get('trajectory', 0))
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Posture: {task_accs.get('posture', 0):.1f}% | "
                  f"Stress: {task_accs.get('stress', 0):.1f}% | "
                  f"Trajectory: {task_accs.get('trajectory', 0):.1f}%")
            
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
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch_{epoch+1}.pth')
                print(f"  -> Periodic checkpoint saved at epoch {epoch+1}")
            
            # Early stopping
            if early_stopping and no_improvement >= early_stopping:
                print(f"\nEarly stopping after {early_stopping} epochs without improvement")
                break
        
        # Save final model and history
        if save_dir:
            self.save_checkpoint(save_path / 'final_model.pth')
            with open(save_path / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']


def add_classification_heads(model: nn.Module) -> nn.Module:
    """
    Add posture and stress classification heads to the model.
    
    Args:
        model: PostureTemporalModel instance
        
    Returns:
        Model with additional heads
    """
    # Posture classifier (4 classes: upright, slouched, open, closed)
    model.posture_classifier = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 4)
    )
    
    # Stress classifier (4 classes: calm, fidgeting, restless, stillness)
    model.stress_classifier = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 4)
    )
    
    return model


def train_posture_model(
    data_dir: str,
    output_dir: str = 'models/posture_trained',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    sequence_length: int = 30,
    device: str = 'cuda',
    num_workers: int = 4
) -> Dict:
    """
    Main training function for posture model.
    
    Args:
        data_dir: Path to posture dataset
        output_dir: Path to save trained model
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sequence_length: Sequence length for training
        device: Training device
        num_workers: Data loading workers
        
    Returns:
        Training history
    """
    from ..datasets.posture_dataset import create_posture_loaders
    from src.posture.temporal_model import PostureTemporalModel
    from src.config import PostureConfig
    
    print(f"Training posture model on {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create data loaders
    train_loader, val_loader = create_posture_loaders(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Auto-detect input dimension from dataset
    # The dataset standardizes all sequences to the max feature dimension
    if hasattr(train_loader.dataset, 'feature_dim'):
        input_dim = train_loader.dataset.feature_dim
    else:
        sample_seq, _ = train_loader.dataset[0]
        input_dim = sample_seq.shape[-1]
    print(f"Detected input dimension: {input_dim}")
    
    # Create model with correct input dimension
    config = PostureConfig()
    config.input_dim = input_dim
    model = PostureTemporalModel(config)
    print(f"Model input_dim: {model.input_dim}")
    
    # Classification heads are already in the model
    # (added in temporal_model.py)
    
    # Get class weights
    class_weights = {}
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights['posture'] = train_loader.dataset.get_class_weights('posture')
        class_weights['stress'] = train_loader.dataset.get_class_weights('stress')
        class_weights['trajectory'] = train_loader.dataset.get_class_weights('trajectory')
    
    # Create trainer
    trainer = PostureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device if torch.cuda.is_available() else 'cpu',
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Train
    history = trainer.train(
        epochs=epochs,
        save_dir=output_dir,
        early_stopping=10
    )
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train posture temporal model')
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--output', type=str, default='models/posture_trained')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-length', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    
    history = train_posture_model(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_length,
        num_workers=args.workers
    )
    
    print(f"\nTraining complete! Best validation accuracy: {max(history['val_acc']):.2f}%")
