"""
Dataset Preparation Script
Organizes emotion datasets into train/val/test splits (70:15:15)
Supports both AffectNet and FER2013 folder structures
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


# Emotion mappings
AFFECTNET_EMOTIONS = {
    '0': 'neutral', '1': 'happy', '2': 'sad', '3': 'surprise',
    '4': 'fear', '5': 'disgust', '6': 'anger', '7': 'contempt',
    'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad',
    'surprise': 'surprise', 'fear': 'fear', 'disgust': 'disgust',
    'anger': 'anger', 'angry': 'anger', 'contempt': 'contempt'
}

FER2013_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def detect_dataset_type(source_dir: Path) -> str:
    """Auto-detect dataset type based on folder structure."""
    subdirs = [d.name.lower() for d in source_dir.iterdir() if d.is_dir()]
    
    # Check for train/test structure
    if 'train' in subdirs or 'test' in subdirs:
        # Look inside train folder
        train_dir = source_dir / 'train' if (source_dir / 'train').exists() else source_dir / 'Train'
        if train_dir.exists():
            inner_dirs = [d.name.lower() for d in train_dir.iterdir() if d.is_dir()]
            if any(d.isdigit() for d in inner_dirs):
                return 'affectnet'
            elif any(d in FER2013_EMOTIONS for d in inner_dirs):
                return 'fer2013'
    
    # Direct emotion folders
    if any(d.isdigit() for d in subdirs):
        return 'affectnet_flat'
    elif any(d in FER2013_EMOTIONS for d in subdirs):
        return 'fer2013_flat'
    
    return 'unknown'


def get_all_images(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = []
    for ext in extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    return images


def collect_images_by_class(source_dir: Path, dataset_type: str) -> Dict[str, List[Path]]:
    """Collect all images organized by emotion class."""
    images_by_class = defaultdict(list)
    
    if dataset_type == 'affectnet':
        # Has train/val structure with numbered folders
        for split in ['train', 'Train', 'val', 'Val', 'test', 'Test']:
            split_dir = source_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        images = get_all_images(class_dir)
                        images_by_class[class_name].extend(images)
    
    elif dataset_type == 'fer2013':
        # Has train/test structure with named folders
        for split in ['train', 'Train', 'test', 'Test']:
            split_dir = source_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name.lower()
                        images = get_all_images(class_dir)
                        images_by_class[class_name].extend(images)
    
    elif dataset_type in ['affectnet_flat', 'fer2013_flat']:
        # Flat structure with emotion folders directly
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower() if dataset_type == 'fer2013_flat' else class_dir.name
                images = get_all_images(class_dir)
                images_by_class[class_name].extend(images)
    
    return dict(images_by_class)


def split_data(
    images: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split images into train/val/test sets."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def prepare_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_files: bool = True
):
    """
    Prepare dataset with proper train/val/test splits.
    
    Args:
        source_dir: Path to source dataset (e.g., Downloads/archive (1))
        output_dir: Path to output directory (e.g., c:/sentry/data/affectnet)
        train_ratio: Training set ratio (default 0.70)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        copy_files: If True, copy files; if False, move files
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    # Detect dataset type
    dataset_type = detect_dataset_type(source_path)
    print(f"Detected dataset type: {dataset_type}")
    
    if dataset_type == 'unknown':
        print("ERROR: Could not detect dataset type. Please specify manually.")
        print("Expected structure: folders named 0-6 (AffectNet) or angry/happy/etc. (FER2013)")
        return
    
    # Collect all images by class
    print("\nCollecting images...")
    images_by_class = collect_images_by_class(source_path, dataset_type)
    
    if not images_by_class:
        print("ERROR: No images found in source directory")
        return
    
    # Print class distribution
    print(f"\nFound {len(images_by_class)} classes:")
    total_images = 0
    for class_name, images in sorted(images_by_class.items()):
        print(f"  {class_name}: {len(images)} images")
        total_images += len(images)
    print(f"Total: {total_images} images")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in images_by_class.keys():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Split and copy/move images
    print(f"\nSplitting data ({train_ratio*100:.0f}:{val_ratio*100:.0f}:{test_ratio*100:.0f})...")
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for class_name, images in images_by_class.items():
        train_imgs, val_imgs, test_imgs = split_data(
            images, train_ratio, val_ratio, test_ratio
        )
        
        splits = [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]
        
        for split_name, split_images in splits:
            dest_dir = output_path / split_name / class_name
            
            for img_path in split_images:
                dest_path = dest_dir / img_path.name
                
                # Handle duplicate filenames
                if dest_path.exists():
                    base = img_path.stem
                    ext = img_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{base}_{counter}{ext}"
                        counter += 1
                
                if copy_files:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(img_path, dest_path)
                
                stats[split_name] += 1
        
        print(f"  {class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    print(f"\nComplete!")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare emotion dataset with train/val/test splits')
    parser.add_argument('--source', type=str, required=True,
                        help='Source dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                        help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying')
    
    args = parser.parse_args()
    
    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_files=not args.move
    )
