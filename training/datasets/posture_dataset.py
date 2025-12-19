# Posture Dataset Loader
# Supports multiple body language and posture datasets

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import csv


class PostureState:
    """Posture state labels."""
    UPRIGHT = 0
    SLOUCHED = 1
    OPEN = 2
    CLOSED = 3
    
    LABELS = ['upright', 'slouched', 'open', 'closed']


class StressIndicator:
    """Stress indicator labels."""
    CALM = 0
    FIDGETING = 1
    RESTLESS = 2
    STILL = 3  # Excessive stillness (freeze response)
    
    LABELS = ['calm', 'fidgeting', 'restless', 'excessive_stillness']


class PostureSequenceDataset(Dataset):
    """
    Dataset for posture temporal sequences with body language labels.
    
    Expects data structure:
    root_dir/
        train/
            sequences/
                seq_001.npy  # Shape: (T, 15) - temporal features
            labels.json  # {seq_001: {posture: 0, stress: 1, trajectory: 2}}
        val/
            ...
    """
    
    # Posture archetypes
    POSTURE_LABELS = {
        0: 'upright',
        1: 'slouched', 
        2: 'open',
        3: 'closed'
    }
    
    # Stress indicators
    STRESS_LABELS = {
        0: 'calm',
        1: 'fidgeting',
        2: 'restless',
        3: 'excessive_stillness'
    }
    
    # Trajectory
    TRAJECTORY_LABELS = {
        0: 'stable',
        1: 'deteriorating',
        2: 'improving'
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sequence_length: int = 30,  # 1 second at 30 FPS
        stride: int = 15,  # 50% overlap
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize posture sequence dataset.
        
        Args:
            root_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            sequence_length: Fixed sequence length for training
            stride: Stride for sliding window
            augment: Apply data augmentation
            max_samples: Maximum samples to load
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment = augment and split == 'train'
        self.max_samples = max_samples
        
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[np.ndarray, Dict]]:
        """Load all samples from sequences directory."""
        samples = []
        
        split_dir = self.root_dir / self.split
        seq_dir = split_dir / 'sequences'
        labels_file = split_dir / 'labels.json'
        
        if not seq_dir.exists():
            print(f"Warning: Sequences directory not found: {seq_dir}")
            return samples
        
        # Load labels
        labels_dict = {}
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels_dict = json.load(f)
        
        # Load sequences
        seq_files = list(seq_dir.glob('*.npy'))
        
        for seq_file in seq_files:
            seq_name = seq_file.stem
            
            try:
                sequence = np.load(seq_file)
            except Exception as e:
                print(f"Error loading {seq_file}: {e}")
                continue
            
            # Get labels for this sequence
            seq_labels = labels_dict.get(seq_name, {
                'posture': 0,  # default: upright
                'stress': 0,   # default: calm
                'trajectory': 0  # default: stable
            })
            
            # Create sliding window samples
            T = sequence.shape[0]
            for start in range(0, max(1, T - self.sequence_length + 1), self.stride):
                end = start + self.sequence_length
                if end <= T:
                    sample_seq = sequence[start:end]
                    samples.append((sample_seq, seq_labels))
                    
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sequence, labels = self.samples[idx]
        
        # Apply augmentation
        if self.augment:
            sequence = self._augment(sequence)
        
        # Convert to tensors
        sequence_tensor = torch.from_numpy(sequence).float()
        
        labels_tensor = {
            'posture': torch.tensor(labels['posture'], dtype=torch.long),
            'stress': torch.tensor(labels['stress'], dtype=torch.long),
            'trajectory': torch.tensor(labels['trajectory'], dtype=torch.long)
        }
        
        return sequence_tensor, labels_tensor
    
    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """Apply temporal and feature augmentation."""
        seq = sequence.copy()
        
        # Temporal jitter (small random offsets in time)
        if np.random.random() < 0.3:
            jitter = np.random.randint(-2, 3, size=seq.shape[0])
            jitter = np.cumsum(jitter)
            jitter = jitter - jitter.min()
            jitter = (jitter / max(1, jitter.max()) * (seq.shape[0] - 1)).astype(int)
            seq = seq[jitter]
        
        # Add Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.02, seq.shape)
            seq = seq + noise
        
        # Random scaling
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            seq = seq * scale
        
        # Temporal scaling (speed variation)
        if np.random.random() < 0.2:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_len = int(seq.shape[0] * speed_factor)
            if new_len > 0:
                indices = np.linspace(0, seq.shape[0] - 1, new_len).astype(int)
                seq = seq[indices]
                # Pad or truncate to original length
                if seq.shape[0] < self.sequence_length:
                    pad_len = self.sequence_length - seq.shape[0]
                    seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
                elif seq.shape[0] > self.sequence_length:
                    start = np.random.randint(0, seq.shape[0] - self.sequence_length + 1)
                    seq = seq[start:start + self.sequence_length]
        
        # Random dropout (simulate missing landmarks)
        if np.random.random() < 0.1:
            mask = np.random.random(seq.shape) > 0.05
            seq = seq * mask
        
        return seq.astype(np.float32)
    
    def get_class_weights(self, label_type: str = 'posture') -> torch.Tensor:
        """Get class weights for imbalanced data."""
        labels = [s[1][label_type] for s in self.samples]
        
        if label_type == 'posture':
            num_classes = 4
        elif label_type == 'stress':
            num_classes = 4
        else:
            num_classes = 3
        
        counts = np.bincount(labels, minlength=num_classes)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        return torch.FloatTensor(weights)


class VideoPostureDataset(Dataset):
    """
    Dataset for extracting posture features from video files.
    
    Expects data structure:
    root_dir/
        train/
            videos/
                video_001.mp4
            labels.csv  # video_name, posture, stress, trajectory
        val/
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sequence_length: int = 30,
        feature_extractor = None,  # PostureFeatureExtractor instance
        transform = None
    ):
        """
        Initialize video posture dataset.
        
        Args:
            root_dir: Root directory
            split: 'train' or 'val'
            sequence_length: Number of frames per sequence
            feature_extractor: Instance of PostureFeatureExtractor
            transform: Optional video transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.feature_extractor = feature_extractor
        self.transform = transform
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[Path, Dict]]:
        """Load video paths and labels."""
        samples = []
        
        split_dir = self.root_dir / self.split
        video_dir = split_dir / 'videos'
        labels_file = split_dir / 'labels.csv'
        
        if not video_dir.exists():
            print(f"Warning: Videos directory not found: {video_dir}")
            return samples
        
        # Load labels from CSV
        labels_dict = {}
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_name = row['video_name']
                    labels_dict[video_name] = {
                        'posture': int(row.get('posture', 0)),
                        'stress': int(row.get('stress', 0)),
                        'trajectory': int(row.get('trajectory', 0))
                    }
        
        # Find video files
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        for video_file in video_files:
            video_name = video_file.stem
            labels = labels_dict.get(video_name, {
                'posture': 0, 'stress': 0, 'trajectory': 0
            })
            samples.append((video_file, labels))
        
        print(f"Loaded {len(samples)} videos for {self.split} split")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from video.
        
        Note: This is slow - prefer pre-extracting features to .npy files
        and using PostureSequenceDataset.
        """
        import cv2
        
        video_path, labels = self.samples[idx]
        
        cap = cv2.VideoCapture(str(video_path))
        features_list = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.feature_extractor:
                # Extract pose and features
                feature_vector = self._extract_features(frame)
                features_list.append(feature_vector)
            
            frame_count += 1
        
        cap.release()
        
        # Pad if necessary
        while len(features_list) < self.sequence_length:
            if features_list:
                features_list.append(features_list[-1])
            else:
                features_list.append(np.zeros(15))
        
        sequence = np.array(features_list[:self.sequence_length])
        
        return (
            torch.from_numpy(sequence).float(),
            {
                'posture': torch.tensor(labels['posture'], dtype=torch.long),
                'stress': torch.tensor(labels['stress'], dtype=torch.long),
                'trajectory': torch.tensor(labels['trajectory'], dtype=torch.long)
            }
        )
    
    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract posture features from a single frame."""
        # This would use PoseEstimator and PostureFeatureExtractor
        # Returning placeholder for now
        return np.zeros(15)


def create_posture_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sequence_length: int = 30,
    stride: int = 15
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        root_dir: Root directory with train/val splits
        batch_size: Batch size
        num_workers: Number of data loading workers
        sequence_length: Sequence length
        stride: Sliding window stride
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = PostureSequenceDataset(
        root_dir=root_dir,
        split='train',
        sequence_length=sequence_length,
        stride=stride,
        augment=True
    )
    
    val_dataset = PostureSequenceDataset(
        root_dir=root_dir,
        split='val',
        sequence_length=sequence_length,
        stride=sequence_length,  # No overlap for validation
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Dataset download instructions
DATASET_INFO = """
# Recommended Posture Datasets

## 1. BoLD (Body Language Dataset) - BEST CHOICE
   - 32 body language categories (arm crossed, finger tapping, etc.)
   - Emotion and psychiatric symptom annotations
   - Download: https://cydar.ist.psu.edu/emotionchallenge/
   
## 2. MultiPosture Dataset
   - 3D skeletal coordinates with posture labels
   - Upright/slouched/leaning classifications
   - Sitting posture focus
   - Download: https://zenodo.org/record/7155660
   
## 3. Stress Detection Dataset (ggian/stress_dataset)
   - Multimodal: biosignals + facial video + body pose
   - 58 participants with stress/neutral labels
   - Download: https://github.com/ggian/stress_dataset
   
## 4. URMC Body Language Dataset
   - Conversations with body language annotations
   - OpenPose landmark extraction
   - Psychiatric symptom labels
   - Paper: arxiv.org/abs/2204.02037
   
## 5. Human Activity Recognition (OpenPose/Kaggle)
   - Key point coordinates for standing, walking, etc.
   - Good for fidgeting detection
   - Download: kaggle.com/datasets/uciml/human-activity-recognition

## Data Preparation
After downloading, structure your data as:

data/posture/
+-- train/
|   +-- sequences/
|   |   +-- seq_001.npy  # Shape: (T, 15)
|   |   +-- ...
|   +-- labels.json
+-- val/
    +-- sequences/
    +-- labels.json

labels.json format:
{
    "seq_001": {"posture": 0, "stress": 1, "trajectory": 0},
    "seq_002": {"posture": 2, "stress": 0, "trajectory": 1}
}

Labels:
- posture: 0=upright, 1=slouched, 2=open, 3=closed
- stress: 0=calm, 1=fidgeting, 2=restless, 3=excessive_stillness
- trajectory: 0=stable, 1=deteriorating, 2=improving
"""


def download_instructions():
    """Print dataset download instructions."""
    print(DATASET_INFO)
    return DATASET_INFO
