# Evaluation Module
# Metrics computation, visualization, and model assessment

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_recall_curve,
    roc_curve, auc
)
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive evaluation and visualization for trained models.
    """
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Computation device
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_classes: int = 7) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Test/validation data loader
            num_classes: Number of classes
            
        Returns:
            Dictionary with all metrics
        """
        self.predictions = []
        self.labels = []
        self.probabilities = []
        
        print("Evaluating model...")
        for images, labels in tqdm(dataloader):
            images = images.to(self.device)
            
            outputs = self.model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('emotion', None))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            self.predictions.extend(preds.cpu().numpy())
            self.labels.extend(labels.numpy())
            self.probabilities.extend(probs.cpu().numpy())
        
        self.predictions = np.array(self.predictions)
        self.labels = np.array(self.labels)
        self.probabilities = np.array(self.probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(num_classes)
        
        return metrics
    
    def _compute_metrics(self, num_classes: int) -> Dict:
        """Compute all evaluation metrics."""
        # Basic accuracy
        accuracy = accuracy_score(self.labels, self.predictions)
        
        # Per-class metrics
        f1_macro = f1_score(self.labels, self.predictions, average='macro')
        f1_weighted = f1_score(self.labels, self.predictions, average='weighted')
        f1_per_class = f1_score(self.labels, self.predictions, average=None)
        
        # Classification report
        class_labels = self.EMOTION_LABELS[:num_classes]
        report = classification_report(
            self.labels, self.predictions, 
            target_names=class_labels,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': {class_labels[i]: float(f1_per_class[i]) for i in range(num_classes)},
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(self.labels),
            'per_class_accuracy': {}
        }
        
        # Per-class accuracy
        for i, label in enumerate(class_labels):
            mask = self.labels == i
            if mask.sum() > 0:
                class_acc = (self.predictions[mask] == i).mean()
                metrics['per_class_accuracy'][label] = float(class_acc)
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                               num_classes: int = 7) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            save_path: Path to save the figure
            num_classes: Number of classes
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(self.labels, self.predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        class_labels = self.EMOTION_LABELS[:num_classes]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, history_path: str, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history curves.
        
        Args:
            history_path: Path to history.json file
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        with open(history_path) as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if history['val_loss']:
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        if history['val_acc']:
            axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_per_class_metrics(self, metrics: Dict, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class F1 scores and accuracy.
        
        Args:
            metrics: Metrics dictionary from evaluate()
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        classes = list(metrics['f1_per_class'].keys())
        f1_scores = list(metrics['f1_per_class'].values())
        accuracies = [metrics['per_class_accuracy'].get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='steelblue')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='coral')
        
        ax.set_xlabel('Emotion Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-class metrics saved to {save_path}")
        
        return fig
    
    def print_summary(self, metrics: Dict) -> None:
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"F1 Score (Macro): {metrics['f1_macro']*100:.2f}%")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']*100:.2f}%")
        print(f"Total Samples: {metrics['num_samples']}")
        
        print("\nPer-Class Performance:")
        print("-" * 40)
        for emotion, f1 in metrics['f1_per_class'].items():
            acc = metrics['per_class_accuracy'].get(emotion, 0)
            print(f"  {emotion:12s} | F1: {f1:.3f} | Acc: {acc:.3f}")
        print("="*60)


def evaluate_trained_model(
    model_path: str,
    data_dir: str,
    output_dir: str = 'evaluation_results',
    device: str = 'cuda'
) -> Dict:
    """
    Full evaluation of a trained emotion model.
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Path to dataset
        output_dir: Directory for evaluation outputs
        device: Computation device
        
    Returns:
        Metrics dictionary
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.facial.emotion import EmotionClassifier
    from src.config import FacialConfig
    from training.datasets.affectnet import create_affectnet_loaders
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    device = device if torch.cuda.is_available() else 'cpu'
    
    config = FacialConfig()
    model = EmotionClassifier(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    _, val_loader = create_affectnet_loaders(data_dir, batch_size=64)
    
    # Evaluate
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(val_loader)
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Generate plots
    evaluator.plot_confusion_matrix(output_path / 'confusion_matrix.png')
    evaluator.plot_per_class_metrics(metrics, output_path / 'per_class_metrics.png')
    
    # Check for history file
    history_path = Path(model_path).parent / 'history.json'
    if history_path.exists():
        evaluator.plot_training_history(str(history_path), output_path / 'training_curves.png')
    
    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        # Convert numpy types for JSON
        metrics_json = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    plt.show()
    
    return metrics


def evaluate_posture_model(
    model_path: str,
    data_dir: str,
    output_dir: str = 'evaluation_results',
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate trained posture model on multi-task metrics.
    
    Args:
        model_path: Path to posture model checkpoint
        data_dir: Path to posture dataset
        output_dir: Directory for evaluation outputs
        device: Computation device
        
    Returns:
        Metrics dictionary
    """
    from src.posture.temporal_model import PostureTemporalModel
    from src.config import PostureConfig
    from training.datasets.posture_dataset import create_posture_loaders
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = device if torch.cuda.is_available() else 'cpu'
    
    # Load data first to get input dimension
    _, val_loader = create_posture_loaders(data_dir, batch_size=32)
    
    # Get feature dimension from dataset
    if hasattr(val_loader.dataset, 'feature_dim'):
        input_dim = val_loader.dataset.feature_dim
    else:
        sample, _ = val_loader.dataset[0]
        input_dim = sample.shape[-1]
    
    print(f"Input dimension: {input_dim}")
    
    # Create model with correct input dimension
    config = PostureConfig()
    config.input_dim = input_dim
    model = PostureTemporalModel(config)
    
    # Load checkpoint
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    posture_preds, posture_labels = [], []
    stress_preds, stress_labels = [], []
    trajectory_preds, trajectory_labels = [], []
    
    print("Evaluating posture model...")
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader):
            sequences = sequences.to(device)
            
            embedding, trajectory_logits, _ = model(sequences)
            posture_logits = model.posture_classifier(embedding)
            stress_logits = model.stress_classifier(embedding)
            
            posture_preds.extend(posture_logits.argmax(dim=1).cpu().numpy())
            stress_preds.extend(stress_logits.argmax(dim=1).cpu().numpy())
            trajectory_preds.extend(trajectory_logits.argmax(dim=1).cpu().numpy())
            
            posture_labels.extend(labels['posture'].numpy())
            stress_labels.extend(labels['stress'].numpy())
            trajectory_labels.extend(labels['trajectory'].numpy())
    
    # Compute metrics
    posture_acc = accuracy_score(posture_labels, posture_preds)
    stress_acc = accuracy_score(stress_labels, stress_preds)
    trajectory_acc = accuracy_score(trajectory_labels, trajectory_preds)
    
    overall_acc = (posture_acc + stress_acc + trajectory_acc) / 3
    
    metrics = {
        'overall_accuracy': overall_acc,
        'posture_accuracy': posture_acc,
        'stress_accuracy': stress_acc,
        'trajectory_accuracy': trajectory_acc,
        'posture_f1': f1_score(posture_labels, posture_preds, average='weighted'),
        'stress_f1': f1_score(stress_labels, stress_preds, average='weighted'),
        'trajectory_f1': f1_score(trajectory_labels, trajectory_preds, average='weighted'),
        'num_samples': len(posture_labels)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("POSTURE MODEL EVALUATION")
    print("="*60)
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
    print(f"\nPer-Task Performance:")
    print(f"  Posture:    {posture_acc*100:.2f}% (F1: {metrics['posture_f1']:.3f})")
    print(f"  Stress:     {stress_acc*100:.2f}% (F1: {metrics['stress_f1']:.3f})")
    print(f"  Trajectory: {trajectory_acc*100:.2f}% (F1: {metrics['trajectory_f1']:.3f})")
    print(f"\nTotal samples: {metrics['num_samples']}")
    print("="*60)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    POSTURE_LABELS = ['upright', 'slouched', 'open', 'closed']
    STRESS_LABELS = ['calm', 'fidgeting', 'restless', 'stillness']
    TRAJECTORY_LABELS = ['stable', 'deteriorating', 'improving']
    
    # Posture confusion matrix
    cm_posture = confusion_matrix(posture_labels, posture_preds)
    sns.heatmap(cm_posture, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=POSTURE_LABELS[:len(set(posture_labels))],
                yticklabels=POSTURE_LABELS[:len(set(posture_labels))])
    axes[0].set_title(f'Posture ({posture_acc*100:.1f}%)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Stress confusion matrix
    cm_stress = confusion_matrix(stress_labels, stress_preds)
    sns.heatmap(cm_stress, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=STRESS_LABELS[:len(set(stress_labels))],
                yticklabels=STRESS_LABELS[:len(set(stress_labels))])
    axes[1].set_title(f'Stress ({stress_acc*100:.1f}%)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # Trajectory confusion matrix
    cm_trajectory = confusion_matrix(trajectory_labels, trajectory_preds)
    sns.heatmap(cm_trajectory, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                xticklabels=TRAJECTORY_LABELS[:len(set(trajectory_labels))],
                yticklabels=TRAJECTORY_LABELS[:len(set(trajectory_labels))])
    axes[2].set_title(f'Trajectory ({trajectory_acc*100:.1f}%)')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_path / 'posture_confusion_matrices.png', dpi=150)
    print(f"\nConfusion matrices saved to {output_path / 'posture_confusion_matrices.png'}")
    
    # Plot training history if available
    history_path = Path(model_path).parent / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes2[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes2[0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes2[0].set_xlabel('Epoch')
        axes2[0].set_ylabel('Loss')
        axes2[0].set_title('Loss Curves')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        axes2[1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes2[1].plot(epochs, history['val_acc'], 'r-', label='Val')
        axes2[1].set_xlabel('Epoch')
        axes2[1].set_ylabel('Accuracy (%)')
        axes2[1].set_title('Accuracy Curves')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'posture_training_curves.png', dpi=150)
        print(f"Training curves saved to {output_path / 'posture_training_curves.png'}")
    
    # Save metrics
    with open(output_path / 'posture_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_path / 'posture_metrics.json'}")
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output', type=str, default='evaluation_results')
    
    args = parser.parse_args()
    
    evaluate_trained_model(args.model, args.data, args.output)
