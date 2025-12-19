# Training CLI
# Main entry point for training models

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def train_emotion(args):
    """Train emotion classifier."""
    from training.trainers.emotion_trainer import train_emotion_model
    
    print(f"Training emotion classifier on {args.dataset}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    
    history = train_emotion_model(
        data_dir=args.data,
        output_dir=args.output,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,

        device='cuda',
        num_workers=args.workers
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {max(history.get('val_acc', [0])):.2f}%")


def train_classifier(args):
    """Train mental health classifier heads."""
    import torch
    from torch.utils.data import DataLoader
    from training.trainers.classifier_trainer import (
        ClassifierTrainer, 
        create_training_data_from_features
    )
    from src.prediction.classifier import create_classifier
    
    print(f"Training classifier heads")
    print(f"Features directory: {args.features}")
    print(f"Labels file: {args.labels}")
    
    # Load data
    train_dataset, val_dataset = create_training_data_from_features(
        args.features, args.labels
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    classifier = create_classifier(device='cuda' if not args.cpu else 'cpu')
    
    # Train
    trainer = ClassifierTrainer(
        classifier=classifier,
        device='cuda' if not args.cpu else 'cpu',
        learning_rate=args.lr
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.output
    )
    
    print(f"\nTraining complete!")


def download_info(args):
    """Show download instructions for datasets."""
    if args.dataset == 'affectnet':
        from training.datasets.affectnet import AffectNetDataset
        print(AffectNetDataset.download_instructions())
    elif args.dataset == 'fer2013':
        from training.datasets.fer2013 import FER2013Dataset
        print(FER2013Dataset.download_instructions())
    else:
        print("Unknown dataset. Available: affectnet, fer2013")


def create_session(args):
    """Create a template session for custom data collection."""
    from training.datasets.video_dataset import CustomVideoDataset
    
    CustomVideoDataset.create_session_template(
        args.output,
        num_seconds=args.duration
    )


def evaluate_model(args):
    """Evaluate trained model and generate visualizations."""
    from training.evaluation import evaluate_trained_model
    
    print(f"Evaluating model: {args.model}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    
    metrics = evaluate_trained_model(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device='cuda' if not args.cpu else 'cpu'
    )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Mental Health Assessment Framework - Training"
    )
    subparsers = parser.add_subparsers(dest='command', help='Training commands')
    
    # Emotion training
    emotion_parser = subparsers.add_parser('emotion', help='Train emotion classifier')
    emotion_parser.add_argument('--data', type=str, required=True,
                                help='Path to dataset (AffectNet or FER2013)')
    emotion_parser.add_argument('--output', type=str, default='models/emotion_trained',
                                help='Output directory')
    emotion_parser.add_argument('--dataset', type=str, default='affectnet',
                                choices=['affectnet', 'fer2013'])
    # Optimize for RTX GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere/Hopper/Blackwell
    
    emotion_parser.add_argument('--epochs', type=int, default=20)
    emotion_parser.add_argument('--batch-size', type=int, default=64)
    emotion_parser.add_argument('--lr', type=float, default=1e-4)  # Lower LR for better generalization
    emotion_parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    emotion_parser.add_argument('--cpu', action='store_true')
    emotion_parser.set_defaults(func=train_emotion)
    
    # Classifier training
    cls_parser = subparsers.add_parser('classifier', help='Train classifier heads')
    cls_parser.add_argument('--features', type=str, required=True,
                            help='Path to extracted features')
    cls_parser.add_argument('--labels', type=str, required=True,
                            help='Path to labels JSON')
    cls_parser.add_argument('--output', type=str, default='models/classifier_trained')
    cls_parser.add_argument('--epochs', type=int, default=50)
    cls_parser.add_argument('--batch-size', type=int, default=64)
    cls_parser.add_argument('--lr', type=float, default=1e-3)
    cls_parser.add_argument('--cpu', action='store_true')
    cls_parser.set_defaults(func=train_classifier)
    
    # Download info
    dl_parser = subparsers.add_parser('download', help='Show download instructions')
    dl_parser.add_argument('--dataset', type=str, required=True,
                           choices=['affectnet', 'fer2013'])
    dl_parser.set_defaults(func=download_info)
    
    # Create session template
    session_parser = subparsers.add_parser('create-session', 
                                           help='Create data collection session template')
    session_parser.add_argument('--output', type=str, required=True,
                                help='Session output directory')
    session_parser.add_argument('--duration', type=int, default=60,
                                help='Expected video duration in seconds')
    session_parser.set_defaults(func=create_session)
    
    # Evaluate model
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--data', type=str, required=True,
                            help='Path to dataset')
    eval_parser.add_argument('--output', type=str, default='evaluation_results',
                            help='Output directory for results')
    eval_parser.add_argument('--cpu', action='store_true')
    eval_parser.set_defaults(func=evaluate_model)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
