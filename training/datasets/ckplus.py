# CK+ (Cohn-Kanade Extended) Dataset Loader
# Balanced to 400 samples per class with augmentation

import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CKPlusDataset(Dataset):
    """
    CK+ (Cohn-Kanade Extended) Dataset loader with balanced augmentation.
    
    The dataset contains posed facial expressions.
    Original 8 classes: anger, contempt, disgust, fear, happy, sadness, surprise, neutral
    
    This version:
    - Remaps to 6 classes (excluding contempt and disgust)
    - Balances to 400 samples per class via augmentation
    
    Download: kaggle datasets download zhiguocui/ck-dataset
    """
    
    # 6 emotion classes (excluding contempt and disgust to match FER2013/AffectNet)
    EMOTION_LABELS = {
        0: 'angry',
        1: 'fear',
        2: 'happy',
        3: 'sad',
        4: 'surprise',
        5: 'neutral'
    }
    
    # CK+ original folder names to our label mapping
    FOLDER_TO_LABEL = {
        'anger': 0,
        'angry': 0,
        'fear': 1,
        'happy': 2,
        'happiness': 2,
        'sad': 3,
        'sadness': 3,
        'surprise': 4,
        'neutral': 5,
        # Excluded: contempt, disgust
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        balance_classes: bool = True,
        target_samples_per_class: int = 400
    ):
        """
        Initialize CK+ dataset.
        
        Args:
            root_dir: Path to CK+ dataset (contains emotion folders)
            split: 'train' or 'test' (80/20 split applied)
            transform: Image transforms
            balance_classes: If True, balance to target_samples_per_class
            target_samples_per_class: Target samples per class (default: 400)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.balance_classes = balance_classes
        self.target_samples_per_class = target_samples_per_class
        self.num_classes = 6
        
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
        all_samples = []
        
        # Try different folder structures
        # Structure 1: root/emotion/images
        # Structure 2: root/train/emotion/images or root/test/emotion/images
        search_dirs = [self.root_dir]
        
        # Check for train/test subfolders
        for split_name in ['train', 'Train', 'test', 'Test']:
            split_dir = self.root_dir / split_name
            if split_dir.exists():
                search_dirs.append(split_dir)
        
        for search_dir in search_dirs:
            for subdir in search_dir.iterdir():
                if not subdir.is_dir():
                    continue
                
                folder_name = subdir.name.lower()
                
                # Skip contempt and disgust
                if folder_name in ['contempt', 'disgust']:
                    continue
                
                # Get label
                label = self.FOLDER_TO_LABEL.get(folder_name, -1)
                if label == -1:
                    continue
                
                # Find images
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in subdir.glob(ext):
                        all_samples.append((img_path, label))
        
        # Split into train/test (80/20)
        random.seed(42)  # Reproducible split
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * 0.8)
        if self.split == 'train':
            samples = all_samples[:split_idx]
        else:
            samples = all_samples[split_idx:]
        
        # Print distribution
        self._print_distribution(samples, "Loaded")
        return samples
    
    def _print_distribution(self, samples: List, prefix: str = ""):
        """Print class distribution."""
        if not samples:
            print(f"CK+ {self.split} {prefix}: No samples found")
            return
        
        labels = [s[1] for s in samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        dist = {self.EMOTION_LABELS[i]: int(counts[i]) for i in range(self.num_classes)}
        print(f"CK+ {self.split} {prefix}: {len(samples)} total - {dist}")
    
    def _balance_samples(self) -> List[Tuple]:
        """
        Balance samples by oversampling minority classes.
        
        Uses random oversampling to achieve target_samples_per_class.
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
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale
        
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
CK+ (Cohn-Kanade Extended) Download Instructions:
==================================================

From Kaggle:
1. Install Kaggle CLI: pip install kaggle
2. Download: kaggle datasets download zhiguocui/ck-dataset
3. Extract: unzip ck-dataset.zip -d data/ck/

Expected structure:
data/ck/
├── angry/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

Note: 'contempt' and 'disgust' folders are automatically excluded.
"""


def get_ck_augmentation_transforms() -> transforms.Compose:
    """Get strong augmentation transforms for balanced CK+ training."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomErasing(p=0.2)
    ])


def create_ck_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_classes: bool = True,
    target_samples_per_class: int = 400
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for CK+.
    
    Args:
        root_dir: Path to CK+ dataset
        batch_size: Batch size
        num_workers: Data loading workers
        balance_classes: If True, balance to target_samples_per_class
        target_samples_per_class: Target samples per class (default: 400)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Use strong augmentation for balanced training
    if balance_classes:
        train_transform = get_ck_augmentation_transforms()
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = CKPlusDataset(
        root_dir, 
        'train', 
        train_transform,
        balance_classes=balance_classes,
        target_samples_per_class=target_samples_per_class
    )
    
    # Validation set - smaller balanced set
    val_target = min(target_samples_per_class // 4, 100) if balance_classes else target_samples_per_class
    test_dataset = CKPlusDataset(
        root_dir, 
        'test', 
        val_transform,
        balance_classes=balance_classes,
        target_samples_per_class=val_target
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
