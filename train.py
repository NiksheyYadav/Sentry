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
    
    # Auto-detect dataset type from path if not explicitly set
    dataset = args.dataset
    if dataset == 'auto':
        data_lower = args.data.lower()
        if 'fer' in data_lower or 'fer2013' in data_lower:
            dataset = 'fer2013'
        else:
            dataset = 'affectnet'
    
    print(f"Training emotion classifier on {dataset}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    if args.balance:
        print(f"Balanced training: {args.target_samples} samples per class")
    
    history = train_emotion_model(
        data_dir=args.data,
        output_dir=args.output,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device='cuda' if not args.cpu else 'cpu',
        num_workers=args.workers,
        balance_classes=args.balance,
        target_samples_per_class=args.target_samples,
        use_aggressive_augmentation=args.aggressive
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


def train_posture(args):
    """Train posture temporal model."""
    from training.trainers.posture_trainer import train_posture_model
    
    print(f"Training posture model")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    
    history = train_posture_model(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_length,
        device='cuda' if not args.cpu else 'cpu',
        num_workers=args.workers
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {max(history.get('val_acc', [0])):.2f}%")


def download_info(args):
    """Show download instructions for datasets."""
    if args.dataset == 'affectnet':
        from training.datasets.affectnet import AffectNetDataset
        print(AffectNetDataset.download_instructions())
    elif args.dataset == 'fer2013':
        from training.datasets.fer2013 import FER2013Dataset
        print(FER2013Dataset.download_instructions())
    elif args.dataset == 'posture':
        from training.datasets.posture_dataset import download_instructions
        download_instructions()
    else:
        print("Unknown dataset. Available: affectnet, fer2013, posture")


def create_session(args):
    """Create a template session for custom data collection."""
    from training.datasets.video_dataset import CustomVideoDataset
    
    CustomVideoDataset.create_session_template(
        args.output,
        num_seconds=args.duration
    )


def evaluate_model(args):
    """Evaluate trained model and generate visualizations."""
    import torch
    
    print(f"Evaluating model: {args.model}")
    print(f"Data directory: {args.data}")
    print(f"Output directory: {args.output}")
    
    # Detect model type from checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    state_dict_keys = list(checkpoint.get('model_state_dict', checkpoint).keys())
    
    # Check if it's a posture model (has TCN/LSTM layers) or emotion model (has backbone)
    is_posture_model = any('tcn' in k or 'lstm' in k for k in state_dict_keys)
    
    if is_posture_model:
        print("Detected: Posture model")
        from training.evaluation import evaluate_posture_model
        metrics = evaluate_posture_model(
            model_path=args.model,
            data_dir=args.data,
            output_dir=args.output,
            device='cuda' if not args.cpu else 'cpu'
        )
    else:
        print("Detected: Emotion model")
        from training.evaluation import evaluate_trained_model
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
    emotion_parser.add_argument('--dataset', type=str, default='auto',
                                choices=['affectnet', 'fer2013', 'auto'],
                                help='Dataset type (auto-detected from path by default)')
    # Optimize for RTX GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere/Hopper/Blackwell
    
    emotion_parser.add_argument('--epochs', type=int, default=20)
    emotion_parser.add_argument('--batch-size', type=int, default=64)
    emotion_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    emotion_parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    emotion_parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    emotion_parser.add_argument('--balance', action='store_true',
                                help='Balance classes via oversampling (5000 samples/class by default)')
    emotion_parser.add_argument('--target-samples', type=int, default=5000,
                                dest='target_samples',
                                help='Target samples per class when --balance is used')
    emotion_parser.add_argument('--aggressive', action='store_true',
                                help='Use aggressive augmentation (recommended with --balance)')
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
    
    # Posture training
    posture_parser = subparsers.add_parser('posture', help='Train posture temporal model')
    posture_parser.add_argument('--data', type=str, required=True,
                                help='Path to posture dataset')
    posture_parser.add_argument('--output', type=str, default='models/posture_trained',
                                help='Output directory')
    posture_parser.add_argument('--epochs', type=int, default=50)
    posture_parser.add_argument('--batch-size', type=int, default=32)
    posture_parser.add_argument('--lr', type=float, default=1e-4)
    posture_parser.add_argument('--seq-length', type=int, default=30,
                                help='Sequence length (frames)')
    posture_parser.add_argument('--workers', type=int, default=4,
                                help='Number of data loading workers')
    posture_parser.add_argument('--cpu', action='store_true')
    posture_parser.set_defaults(func=train_posture)
    
    # Download info
    dl_parser = subparsers.add_parser('download', help='Show download instructions')
    dl_parser.add_argument('--dataset', type=str, required=True,
                           choices=['affectnet', 'fer2013', 'posture'])
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
