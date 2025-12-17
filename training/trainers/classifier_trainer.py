# Mental Health Classifier Trainer
# Train the stress/depression/anxiety prediction heads

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json


class ClassifierTrainer:
    """
    Trainer for the mental health classification heads.
    
    Trains the three-headed classifier on extracted features
    with multi-task learning.
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3
    ):
        """
        Initialize trainer.
        
        Args:
            classifier: MentalHealthClassifier model
            device: Training device
            learning_rate: Learning rate
        """
        self.classifier = classifier.to(device)
        self.device = device
        
        # Multi-task loss
        self.stress_criterion = nn.CrossEntropyLoss()
        self.depression_criterion = nn.CrossEntropyLoss()
        self.anxiety_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
    
    def multi_task_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        stress_labels: torch.Tensor,
        depression_labels: torch.Tensor,
        anxiety_labels: torch.Tensor,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            outputs: Model outputs with 'stress', 'depression', 'anxiety' keys
            stress_labels: Ground truth stress labels
            depression_labels: Ground truth depression labels
            anxiety_labels: Ground truth anxiety labels
            task_weights: Weights for each task
            
        Returns:
            Combined loss
        """
        stress_loss = self.stress_criterion(outputs['stress'], stress_labels)
        depression_loss = self.depression_criterion(outputs['depression'], depression_labels)
        anxiety_loss = self.anxiety_criterion(outputs['anxiety'], anxiety_labels)
        
        total_loss = (
            task_weights[0] * stress_loss +
            task_weights[1] * depression_loss +
            task_weights[2] * anxiety_loss
        )
        
        return total_loss, {
            'stress': stress_loss.item(),
            'depression': depression_loss.item(),
            'anxiety': anxiety_loss.item()
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.classifier.train()
        total_loss = 0.0
        task_losses = {'stress': 0, 'depression': 0, 'anxiety': 0}
        
        for batch in tqdm(train_loader, desc="Training"):
            features, stress, depression, anxiety = [
                x.to(self.device) for x in batch
            ]
            
            self.optimizer.zero_grad()
            outputs = self.classifier(features)
            
            loss, losses = self.multi_task_loss(
                outputs, stress, depression, anxiety, task_weights
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for k in task_losses:
                task_losses[k] += losses[k]
        
        n_batches = len(train_loader)
        return total_loss / n_batches, {k: v/n_batches for k, v in task_losses.items()}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict]:
        """Validate model."""
        self.classifier.eval()
        total_loss = 0.0
        correct = {'stress': 0, 'depression': 0, 'anxiety': 0}
        total = 0
        
        for batch in val_loader:
            features, stress, depression, anxiety = [
                x.to(self.device) for x in batch
            ]
            
            outputs = self.classifier(features)
            loss, _ = self.multi_task_loss(outputs, stress, depression, anxiety)
            total_loss += loss.item()
            
            # Accuracy
            for key in ['stress', 'depression', 'anxiety']:
                labels = {'stress': stress, 'depression': depression, 'anxiety': anxiety}[key]
                _, predicted = outputs[key].max(1)
                correct[key] += predicted.eq(labels).sum().item()
            
            total += features.size(0)
        
        n_batches = len(val_loader)
        accuracies = {k: 100.0 * v / total for k, v in correct.items()}
        
        return total_loss / n_batches, accuracies
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_task_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['metrics'].append(val_acc)
                
                self.scheduler.step(val_loss)
                
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Acc - Stress: {val_acc['stress']:.1f}%, "
                      f"Depression: {val_acc['depression']:.1f}%, "
                      f"Anxiety: {val_acc['anxiety']:.1f}%")
                
                if val_loss < best_val_loss and save_dir:
                    best_val_loss = val_loss
                    self._save(Path(save_dir) / 'best_classifier.pth')
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")
        
        if save_dir:
            self._save(Path(save_dir) / 'final_classifier.pth')
            
        return self.history
    
    def _save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.classifier.state_dict(),
            'history': self.history
        }, path)
    
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])


def create_training_data_from_features(
    features_dir: str,
    labels_path: str
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create training dataset from pre-extracted features.
    
    Args:
        features_dir: Directory with .npy feature files
        labels_path: Path to labels JSON file
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    features_path = Path(features_dir)
    
    # Load labels
    with open(labels_path) as f:
        labels = json.load(f)
    
    all_features = []
    all_stress = []
    all_depression = []
    all_anxiety = []
    
    for item in labels:
        feature_file = features_path / f"{item['id']}.npy"
        if feature_file.exists():
            feature = np.load(feature_file)
            all_features.append(feature)
            all_stress.append(item['stress'])
            all_depression.append(item['depression'])
            all_anxiety.append(item['anxiety'])
    
    # Convert to tensors
    features = torch.FloatTensor(np.stack(all_features))
    stress = torch.LongTensor(all_stress)
    depression = torch.LongTensor(all_depression)
    anxiety = torch.LongTensor(all_anxiety)
    
    # Split 80/20
    n = len(features)
    indices = torch.randperm(n)
    train_idx = indices[:int(0.8 * n)]
    val_idx = indices[int(0.8 * n):]
    
    train_dataset = TensorDataset(
        features[train_idx], stress[train_idx],
        depression[train_idx], anxiety[train_idx]
    )
    val_dataset = TensorDataset(
        features[val_idx], stress[val_idx],
        depression[val_idx], anxiety[val_idx]
    )
    
    return train_dataset, val_dataset
