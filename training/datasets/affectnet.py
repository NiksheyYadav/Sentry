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
    """
    
    # 6 emotion classes: neutral, happy, sad, surprise, fear, anger
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'happy', 
        2: 'sad',
        3: 'surprise',
        4: 'fear',
        5: 'anger'
    }
    
    EMOTION_LABELS_7 = EMOTION_LABELS
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        num_classes: int = 6,
        max_samples: Optional[int] = None,
        balance_classes: bool = False,
        target_samples_per_class: Optional[int] = None
    ):
        """
        Initialize AffectNet dataset.
        
        Args:
            root_dir: Path to dataset root directory
            split: 'train' or 'val'
            transform: Optional image transforms
            num_classes: Number of emotion classes (6 or 7)
            max_samples: Max samples per class before balancing
            balance_classes: If True, oversample minority classes to match majority
            target_samples_per_class: If set with balance_classes, sample this many per class
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.balance_classes = balance_classes
        self.target_samples_per_class = target_samples_per_class
        
        # Weak classes that need more augmentation (sad, surprise, fear, anger)
        self.weak_classes = {2, 3, 4, 5}  
        self.aggressive_transform = None # Set by trainer
        
        # Load and optionally balance samples
        self.samples = self._load_samples()
        
        if self.balance_classes and self.split == 'train':
            self.samples = self._balance_samples()
        
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
        samples = []
        
        # Emotion name to label mapping (remapped to 0-5)
        EMOTION_NAME_TO_LABEL = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3,
            'fear': 4, 'anger': 5, 'angry': 5
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
            subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
            
            for subdir in subdirs:
                folder_name = subdir.name.lower()
                
                label = -1
                if folder_name in EMOTION_NAME_TO_LABEL:
                    label = EMOTION_NAME_TO_LABEL[folder_name]
                elif folder_name.isdigit():
                    orig_label = int(folder_name)
                    if orig_label == 5 or orig_label == 7:
                        continue 
                    
                    if orig_label == 6:
                        label = 5 
                    elif orig_label < 5:
                        label = orig_label
                    else:
                        continue
                
                if label == -1:
                    continue
                
                images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')) + list(subdir.glob('*.jpeg'))
                
                if self.max_samples:
                    images = images[:self.max_samples]
                
                for img_path in images:
                    samples.append((img_path, label))
        
        # Structure 2: labels.csv in root directory
        # Supports formats: image,label OR subDirectory,image,expression
        if not samples:
            # Try various CSV names
            csv_candidates = [
                self.root_dir / 'labels.csv',
                self.root_dir / f'{self.split}_labels.csv',
                self.root_dir / f'{self.split}.csv'
            ]
            
            csv_path = None
            for candidate in csv_candidates:
                if candidate.exists():
                    csv_path = candidate
                    break
            
            if csv_path:
                print(f"Loading from CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                
                # Detect CSV format based on columns
                columns = df.columns.tolist()
                
                for _, row in df.iterrows():
                    # Format 1: subDirectory,image,expression (AffectNet official)
                    if 'subDirectory' in columns and 'expression' in columns:
                        subdir = str(row['subDirectory'])
                        img_name = row['image'] if 'image' in columns else row.get('face_id', '')
                        img_path = self.root_dir / self.split / subdir / str(img_name)
                        if not img_path.suffix:
                            img_path = img_path.with_suffix('.jpg')
                        orig_label = int(row['expression'])
                    
                    # Format 2: image,label (simple format)
                    elif 'image' in columns and 'label' in columns:
                        img_name = row['image']
                        img_path = self.root_dir / 'images' / img_name
                        if not img_path.exists():
                            img_path = self.root_dir / img_name
                        
                        raw_label = row['label']
                        if isinstance(raw_label, str):
                            orig_label = EMOTION_NAME_TO_LABEL.get(raw_label.lower(), -1)
                        else:
                            orig_label = int(raw_label)
                    
                    # Format 3: pth,label columns
                    elif 'pth' in columns:
                        img_path = Path(row['pth'])
                        if not img_path.is_absolute():
                            img_path = self.root_dir / img_path
                        
                        raw_label = row.get('label', row.get('expression', -1))
                        if isinstance(raw_label, str):
                            orig_label = EMOTION_NAME_TO_LABEL.get(raw_label.lower(), -1)
                        else:
                            orig_label = int(raw_label)
                    
                    else:
                        continue
                    
                    # Remap labels (skip disgust=5, contempt=7)
                    label = -1
                    if orig_label == 5 or orig_label == 7:
                        continue
                    elif orig_label == 6:  # anger -> 5
                        label = 5
                    elif orig_label < 5:
                        label = orig_label
                    
                    if label != -1 and img_path.exists():
                        samples.append((img_path, label))
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        
        # Print class distribution
        if samples:
            labels = [s[1] for s in samples]
            counts = np.bincount(labels, minlength=self.num_classes)
            print(f"  Class distribution: {dict(zip(self.EMOTION_LABELS.values(), counts))}")
        
        return samples
    
    def _balance_samples(self) -> List[Tuple[Path, int]]:
        """
        Balance samples by oversampling minority classes.
        
        Uses random oversampling to equalize class distribution,
        particularly helpful for underrepresented emotions like sad and neutral.
        """
        import random
        
        # Group samples by class
        class_samples = {i: [] for i in range(self.num_classes)}
        for sample in self.samples:
            class_samples[sample[1]].append(sample)
        
        # Determine target count per class
        class_counts = {k: len(v) for k, v in class_samples.items()}
        max_count = max(class_counts.values())
        
        if self.target_samples_per_class:
            target_count = self.target_samples_per_class
        else:
            target_count = max_count
        
        print(f"Balancing classes to {target_count} samples each...")
        print(f"  Original distribution: {class_counts}")
        
        # Oversample minority classes
        balanced_samples = []
        for label, samples_list in class_samples.items():
            if len(samples_list) == 0:
                print(f"  Warning: No samples for class {label} ({self.EMOTION_LABELS.get(label, 'unknown')})")
                continue
            
            if len(samples_list) >= target_count:
                # Undersample if we have too many
                balanced_samples.extend(random.sample(samples_list, target_count))
            else:
                # Oversample with replacement
                balanced_samples.extend(samples_list)  # All original samples
                remaining = target_count - len(samples_list)
                oversampled = random.choices(samples_list, k=remaining)
                balanced_samples.extend(oversampled)
        
        # Shuffle the balanced samples
        random.shuffle(balanced_samples)
        
        # Print new distribution
        new_labels = [s[1] for s in balanced_samples]
        new_counts = np.bincount(new_labels, minlength=self.num_classes)
        print(f"  Balanced distribution: {dict(zip(self.EMOTION_LABELS.values(), new_counts))}")
        print(f"  Total samples after balancing: {len(balanced_samples)}")
        
        return balanced_samples

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            # If it's a weak class, use aggressive transform if available
            current_transform = self.transform
            if label in self.weak_classes and self.aggressive_transform:
                current_transform = self.aggressive_transform
            
            image = current_transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)
    
    def get_label_name(self, label: int) -> str:
        if self.num_classes == 7:
            return self.EMOTION_LABELS_7.get(label, 'unknown')
        return self.EMOTION_LABELS.get(label, 'unknown')
    
    @staticmethod
    def download_instructions() -> str:
        return "Download instructions placeholder."


def create_affectnet_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: int = 6,
    max_samples_per_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
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
