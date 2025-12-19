# AffectNet Dataset Loader
# Supports multiple Kaggle AffectNet versions

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


class AffectNetDataset(Dataset):
    """
    AffectNet Dataset loader for emotion classification.
    
    Supports multiple Kaggle versions:
    - mstjebashazida/affectnet (folder structure)
    - Young AffectNet HQ (high quality version)
    
    8 emotion classes: neutral, happy, sad, surprise, fear, disgust, anger, contempt
    """
    
    # Standard AffectNet emotion mapping
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'happy', 
        2: 'sad',
        3: 'surprise',
        4: 'fear',
        5: 'disgust',
        6: 'anger',
        7: 'contempt'
    }
    
    # Map to 7 classes (exclude contempt for compatibility with FER2013)
    EMOTION_LABELS_7 = {
        0: 'neutral',
        1: 'happy',
        2: 'sad', 
        3: 'surprise',
        4: 'fear',
        5: 'disgust',
        6: 'angry'  # 'anger' -> 'angry' for consistency
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        num_classes: int = 7,
        max_samples: Optional[int] = None
    ):
        """
        Initialize AffectNet dataset.
        
        Args:
            root_dir: Path to dataset root (extracted Kaggle folder)
            split: 'train' or 'val' 
            transform: Image transforms to apply
            num_classes: 7 or 8 emotion classes
            max_samples: Limit samples per class for faster training
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.max_samples = max_samples
        
        # Determine dataset structure
        self.samples = self._load_samples()
        
        # Default transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load sample paths and labels."""
        samples = []
        
        # Emotion name to label mapping (for named folders)
        EMOTION_NAME_TO_LABEL = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
            'fear': 4, 'disgust': 5, 'anger': 6, 'angry': 6, 
            'contempt': 7
        }
        
        # Try different folder structures
        split_map = {'train': ['train', 'Train'], 'val': ['val', 'Test', 'test']}
        split_names = split_map.get(self.split, [self.split])
        
        # Find the correct split directory
        split_dir = None
        for name in split_names:
            candidate = self.root_dir / name
            if candidate.exists():
                split_dir = candidate
                break
        
        if split_dir and split_dir.exists():
            # Check if folders are named (anger, happy) or numbered (0, 1, 2)
            subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for subdir in subdirs:
                folder_name = subdir.name.lower()
                
                # Try to get label from folder name
                if folder_name in EMOTION_NAME_TO_LABEL:
                    label = EMOTION_NAME_TO_LABEL[folder_name]
                elif folder_name.isdigit():
                    label = int(folder_name)
                else:
                    continue
                
                # Skip if label exceeds num_classes
                if label >= self.num_classes:
                    continue
                
                # Get all images
                images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')) + list(subdir.glob('*.jpeg'))
                
                if self.max_samples:
                    images = images[:self.max_samples]
                
                for img_path in images:
                    samples.append((img_path, label))
        
        # Structure 2: Images folder with CSV annotations
        if not samples:
            csv_path = self.root_dir / f'{self.split}_labels.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    img_path = self.root_dir / 'images' / row['image']
                    label = int(row['label'])
                    if label < self.num_classes and img_path.exists():
                        samples.append((img_path, label))
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)
    
    def get_label_name(self, label: int) -> str:
        """Get emotion name for label."""
        if self.num_classes == 7:
            return self.EMOTION_LABELS_7.get(label, 'unknown')
        return self.EMOTION_LABELS.get(label, 'unknown')
    
    @staticmethod
    def download_instructions() -> str:
        """Return download instructions."""
        return """
AffectNet Download Instructions:
================================

Option 1: Kaggle (Easiest)
--------------------------
1. Go to: https://www.kaggle.com/datasets/mstjebashazida/affectnet
2. Click "Download" (requires Kaggle account)
3. Extract to: c:\\sentry\\data\\affectnet\\

Or use Kaggle CLI:
    kaggle datasets download -d mstjebashazida/affectnet
    unzip affectnet.zip -d data/affectnet/

Option 2: Young AffectNet HQ
----------------------------
Higher quality version available at:
https://www.kaggle.com/code/arkhanzada/facial-emotions-classification-affectnet-dataset/input

Expected folder structure after extraction:
data/affectnet/
├── train/
│   ├── 0/  (neutral)
│   ├── 1/  (happy)
│   ├── 2/  (sad)
│   ├── 3/  (surprise)
│   ├── 4/  (fear)
│   ├── 5/  (disgust)
│   ├── 6/  (anger)
│   └── 7/  (contempt) - optional
└── val/
    ├── 0/
    ├── 1/
    ...
"""


def create_affectnet_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: int = 7,
    max_samples_per_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        root_dir: Path to AffectNet dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        num_classes: 7 or 8 emotion classes
        max_samples_per_class: Limit samples for faster training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    train_dataset = AffectNetDataset(
        root_dir=root_dir,
        split='train',
        transform=get_train_transforms(),
        num_classes=num_classes,
        max_samples=max_samples_per_class
    )
    
    val_dataset = AffectNetDataset(
        root_dir=root_dir,
        split='val',
        transform=get_val_transforms(),
        num_classes=num_classes
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, val_loader
