# FER2013 Dataset Loader
# Classic facial emotion recognition dataset with balanced training support

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset loader with balanced training support.
    
    The dataset contains 35,887 grayscale 48x48 face images.
    Original 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
    
    This version:
    - Excludes 'disgust' (rarely useful, often mislabeled)
    - Supports balanced training (5000 samples per class by default)
    - Uses augmentation for oversampled data
    
    Available from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
    """
    
    # 6 emotion classes (excluding disgust)
    # Remapped labels: 0=angry, 1=fear, 2=happy, 3=sad, 4=surprise, 5=neutral
    EMOTION_LABELS = {
        0: 'angry',
        1: 'fear',
        2: 'happy',
        3: 'sad',
        4: 'surprise',
        5: 'neutral'
    }
    
    # Mapping from original FER2013 labels to our labels
    ORIGINAL_TO_NEW = {
        0: 0,   # angry -> angry
        # 1: disgust (excluded)
        2: 1,   # fear -> fear
        3: 2,   # happy -> happy
        4: 3,   # sad -> sad
        5: 4,   # surprise -> surprise
        6: 5,   # neutral -> neutral
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        balance_classes: bool = False,
        target_samples_per_class: int = 5000,
        exclude_disgust: bool = True
    ):
        """
        Initialize FER2013 dataset.
        
        Args:
            root_dir: Path to extracted dataset (contains train/test folders)
            split: 'train' or 'test'
            transform: Image transforms
            balance_classes: If True, balance to target_samples_per_class
            target_samples_per_class: Target samples per class when balancing
            exclude_disgust: If True, exclude disgust class (recommended)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.balance_classes = balance_classes
        self.target_samples_per_class = target_samples_per_class
        self.exclude_disgust = exclude_disgust
        self.num_classes = 6 if exclude_disgust else 7
        
        # Load samples
        self.samples = self._load_samples()
        
        # Balance if requested
        if self.balance_classes:
            self.samples = self._balance_samples()
        
        # Default transform (1-channel grayscale for EmotionClassifier)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load sample paths and labels."""
        samples = []
        
        # Original FER2013 folder names
        ORIGINAL_LABELS = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        
        # FER2013 folder structure: train/angry, train/disgust, etc.
        split_dir = self.root_dir / self.split
        
        if split_dir.exists():
            for orig_label, emotion in ORIGINAL_LABELS.items():
                # Skip disgust if excluded
                if self.exclude_disgust and emotion == 'disgust':
                    continue
                
                emotion_dir = split_dir / emotion
                if emotion_dir.exists():
                    # Get new label
                    new_label = self.ORIGINAL_TO_NEW.get(orig_label, -1)
                    if new_label == -1:
                        continue
                    
                    for img_path in emotion_dir.glob('*.jpg'):
                        samples.append((img_path, new_label))
                    for img_path in emotion_dir.glob('*.png'):
                        samples.append((img_path, new_label))
        
        # Alternative: CSV format (original FER2013)
        csv_path = self.root_dir / 'fer2013.csv'
        if csv_path.exists() and not samples:
            df = pd.read_csv(csv_path)
            
            # Filter by split (Training, PublicTest, PrivateTest)
            usage_map = {'train': 'Training', 'test': 'PublicTest'}
            usage = usage_map.get(self.split, self.split)
            df = df[df['Usage'] == usage]
            
            # Convert pixel strings to images
            self._csv_data = []
            for _, row in df.iterrows():
                orig_label = int(row['emotion'])
                
                # Skip disgust if excluded
                if self.exclude_disgust and orig_label == 1:
                    continue
                
                new_label = self.ORIGINAL_TO_NEW.get(orig_label, -1)
                if new_label == -1:
                    continue
                
                pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                pixels = pixels.reshape(48, 48)
                self._csv_data.append((pixels, new_label))
            
            samples = [(i, d[1]) for i, d in enumerate(self._csv_data)]
        
        # Print distribution
        self._print_distribution(samples, "Loaded")
        return samples
    
    def _print_distribution(self, samples: List, prefix: str = ""):
        """Print class distribution."""
        if not samples:
            print(f"{prefix}: No samples found")
            return
        
        labels = [s[1] for s in samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        dist = {self.EMOTION_LABELS[i]: int(counts[i]) for i in range(self.num_classes)}
        print(f"FER2013 {self.split} {prefix}: {len(samples)} total - {dist}")
    
    def _balance_samples(self) -> List[Tuple]:
        """
        Balance samples by oversampling minority classes.
        
        Uses random oversampling to achieve target_samples_per_class for each class.
        """
        # Group samples by class
        class_samples: Dict[int, List] = {i: [] for i in range(self.num_classes)}
        for sample in self.samples:
            class_samples[sample[1]].append(sample)
        
        print(f"Balancing to {self.target_samples_per_class} samples per class...")
        
        balanced_samples = []
        for label, samples_list in class_samples.items():
            if len(samples_list) == 0:
                print(f"  Warning: No samples for class {label} ({self.EMOTION_LABELS.get(label, 'unknown')})")
                continue
            
            if len(samples_list) >= self.target_samples_per_class:
                # Undersample if we have too many
                balanced_samples.extend(random.sample(samples_list, self.target_samples_per_class))
            else:
                # Oversample with replacement
                balanced_samples.extend(samples_list)  # All original samples
                remaining = self.target_samples_per_class - len(samples_list)
                oversampled = random.choices(samples_list, k=remaining)
                balanced_samples.extend(oversampled)
        
        # Shuffle
        random.shuffle(balanced_samples)
        
        self._print_distribution(balanced_samples, "Balanced")
        return balanced_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_id, label = self.samples[idx]
        
        # Load image based on format
        if isinstance(sample_id, Path):
            image = Image.open(sample_id).convert('L')  # Grayscale
        else:
            # From CSV pixels
            pixels = self._csv_data[sample_id][0]
            image = Image.fromarray(pixels, mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_label_name(self, label: int) -> str:
        """Get emotion name for label."""
        return self.EMOTION_LABELS.get(label, 'unknown')
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)
    
    @staticmethod
    def download_instructions() -> str:
        return """
FER2013 Download Instructions:
==============================

From Kaggle:
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract to: c:\\sentry\\data\\fer2013\\

Using Kaggle CLI:
    kaggle datasets download -d msambare/fer2013
    unzip fer2013.zip -d data/fer2013/

Expected structure:
data/fer2013/
├── train/
│   ├── angry/ 4000
│   ├── disgust/ 450
│   ├── fear/ 4100
│   ├── happy/ 7215
│   ├── neutral/ 4965
│   ├── sad/ 4830
│   └── surprise/ 3171
| we will balance the dataset to 5000
└── test/
    ├── angry/ 950
    ├── disgust/ 111
    ├── fear/ 1024
    ├── happy/ 1774
    ├── neutral/ 1233
    ├── sad/ 1247
    └── surprise/ 831

Note: 'disgust' folder is automatically excluded.
"""


def get_fer2013_augmentation_transforms() -> transforms.Compose:
    """Get strong augmentation transforms for balanced FER2013 training.
    
    Note: EmotionClassifier expects 1-channel grayscale input.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # No saturation/hue for grayscale
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.Grayscale(num_output_channels=1),  # Convert to 1-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Single channel normalization
        transforms.RandomErasing(p=0.2)
    ])


def create_fer2013_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_classes: bool = False,
    target_samples_per_class: int = 5000
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for FER2013.
    
    Args:
        root_dir: Path to FER2013 dataset
        batch_size: Batch size
        num_workers: Data loading workers
        balance_classes: If True, balance train set to target_samples_per_class
        target_samples_per_class: Target samples per class when balancing
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Use strong augmentation for balanced training
    if balance_classes:
        train_transform = get_fer2013_augmentation_transforms()
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),  # 1-channel for EmotionClassifier
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # 1-channel for EmotionClassifier
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = FER2013Dataset(
        root_dir, 
        'train', 
        train_transform,
        balance_classes=balance_classes,
        target_samples_per_class=target_samples_per_class,
        exclude_disgust=True
    )
    
    # Validation set - also balance but with fewer samples
    val_target = min(target_samples_per_class // 5, 500) if balance_classes else 0
    test_dataset = FER2013Dataset(
        root_dir, 
        'test', 
        val_transform,
        balance_classes=balance_classes,
        target_samples_per_class=val_target if balance_classes else 5000,
        exclude_disgust=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

