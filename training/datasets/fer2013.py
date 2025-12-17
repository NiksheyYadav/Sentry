# FER2013 Dataset Loader
# Classic facial emotion recognition dataset

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset loader.
    
    The dataset contains 35,887 grayscale 48x48 face images.
    7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
    
    Available from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
    """
    
    EMOTION_LABELS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize FER2013 dataset.
        
        Args:
            root_dir: Path to extracted dataset (contains train/test folders)
            split: 'train' or 'test'
            transform: Image transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load samples
        self.samples = self._load_samples()
        
        # Default transform (grayscale to RGB)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
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
        
        # FER2013 folder structure: train/angry, train/disgust, etc.
        split_dir = self.root_dir / self.split
        
        if split_dir.exists():
            for label, emotion in self.EMOTION_LABELS.items():
                emotion_dir = split_dir / emotion
                if emotion_dir.exists():
                    for img_path in emotion_dir.glob('*.jpg'):
                        samples.append((img_path, label))
                    for img_path in emotion_dir.glob('*.png'):
                        samples.append((img_path, label))
        
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
                pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                pixels = pixels.reshape(48, 48)
                label = int(row['emotion'])
                self._csv_data.append((pixels, label))
            
            samples = [(i, d[1]) for i, d in enumerate(self._csv_data)]
        
        print(f"FER2013: Loaded {len(samples)} samples for {self.split}")
        return samples
    
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
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ...
"""


def create_fer2013_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders for FER2013."""
    from .transforms import get_train_transforms, get_val_transforms
    
    # Modify transforms for grayscale input
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = FER2013Dataset(root_dir, 'train', train_transform)
    test_dataset = FER2013Dataset(root_dir, 'test', val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader
