"""
Download and prepare video posture datasets for Sentry mental state training.

Supported Datasets:
1. MultiPosture (Zenodo) - Body keypoints with posture labels
2. Figshare Sit/Stand - OpenPose keypoints with sit/stand classification
3. NTU RGB+D Skeleton - 25-joint skeleton data for action recognition
4. CMU Panoptic - Multi-view 3D body pose (toolbox setup)

Usage:
    python download_video_posture_datasets.py --dataset all
    python download_video_posture_datasets.py --dataset figshare
    python download_video_posture_datasets.py --dataset ntu
    python download_video_posture_datasets.py --dataset multiposture
    python download_video_posture_datasets.py --dataset cmu --setup-only
"""
import os
import sys
import argparse
import requests
import zipfile
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# DATASET URLs AND CONFIGURATIONS
# ============================================================================

DATASETS = {
    'multiposture': {
        'name': 'MultiPosture Dataset',
        'url': 'https://zenodo.org/records/14230872/files/data.csv?download=1',
        'description': 'Body keypoints with upper/lower body posture labels',
        'size': '~5 MB',
        'format': 'csv'
    },
    'figshare': {
        'name': 'Figshare Sit/Stand Pose Dataset',
        'url': 'https://figshare.com/ndownloader/files/26582851',  # Human pose dataset
        'description': '50K OpenPose keypoints with sit/stand classification',
        'size': '~10 MB',
        'format': 'csv'
    },
    'ntu_skeleton': {
        'name': 'NTU RGB+D 60 Skeleton (Sample)',
        'url': None,  # Requires gdown for Google Drive
        'gdrive_id': '1CUZnBtYwifVXS21yVg62T-vrPVayso5H',
        'description': '25-joint skeleton data, 60 action classes',
        'size': '~5.8 GB',
        'format': 'skeleton'
    },
    'cmu_panoptic': {
        'name': 'CMU Panoptic Studio',
        'url': 'https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox',
        'description': 'Multi-view 3D body pose with social interactions',
        'size': 'Variable',
        'format': 'toolbox'
    }
}


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress tracking."""
    print(f"Downloading from {url}...")
    
    try:
        response = requests.get(url, stream=True, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading: {e}")
        return False
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                bar_length = 40
                filled = int(bar_length * downloaded / total_size)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='', flush=True)
    
    print(f"\n✓ Saved to {output_path}")
    return True


def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    print(f"Downloading from Google Drive (ID: {file_id})...")
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a zip file."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


# ============================================================================
# DATASET PROCESSORS
# ============================================================================

def prepare_multiposture_dataset(csv_path: str, output_dir: str) -> Dict:
    """
    Convert MultiPosture CSV to Sentry training format.
    
    Labels:
    - Upper body: TUP (upright), TLB/TLF (slouched), TLR/TLL (leaning)
    - Lower body: LAP/LWA (calm), LCS (stress), LCR/LCL (fidgeting)
    """
    print(f"\n📊 Processing MultiPosture dataset...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} samples")
    
    # Posture label mapping
    upper_body_map = {
        'TUP': 0,  # Upright
        'TLB': 1,  # Leaning backward (slouched)
        'TLF': 1,  # Leaning forward (slouched)
        'TLR': 2,  # Leaning right
        'TLL': 2,  # Leaning left
    }
    
    # Stress label mapping
    lower_body_map = {
        'LAP': 0,  # Legs apart - calm
        'LWA': 0,  # Legs wide apart - calm
        'LCS': 1,  # Legs closed - moderate stress
        'LCR': 2,  # Legs crossed right - restless
        'LCL': 2,  # Legs crossed left - restless
        'LLR': 1,  # Legs lateral right
        'LLL': 1,  # Legs lateral left
    }
    
    # Find columns
    upper_col, lower_col, subject_col = None, None, None
    feature_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if 'upperbody' in col_lower or 'upper_body' in col_lower:
            upper_col = col
        elif 'lowerbody' in col_lower or 'lower_body' in col_lower:
            lower_col = col
        elif col_lower in ['subject', 'participant', 'user', 'id']:
            subject_col = col
        elif '_x' in col_lower or '_y' in col_lower or '_z' in col_lower:
            feature_cols.append(col)
    
    if not feature_cols:
        for col in df.columns:
            if col not in [upper_col, lower_col, subject_col] and df[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
    
    # Process features
    all_features = df[feature_cols].values.astype(np.float32)
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    global_std[global_std < 1e-6] = 1.0
    
    return _save_dataset(df, feature_cols, upper_col, lower_col, subject_col,
                         upper_body_map, lower_body_map, global_mean, global_std,
                         output_dir, 'multiposture')


def prepare_figshare_sitstand_dataset(csv_path: str, output_dir: str) -> Dict:
    """
    Convert Figshare Sit/Stand dataset to Sentry training format.
    
    Format: 18 OpenPose keypoints (x, y, confidence) + sit/stand label
    
    Mapping:
    - sit → closed/defensive posture (1)
    - stand → open/upright posture (0)
    """
    print(f"\n📊 Processing Figshare Sit/Stand dataset...")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns)[:5]}...")
    
    # Find label column and feature columns
    label_col = None
    for col in df.columns:
        if 'class' in col.lower() or 'label' in col.lower() or 'pose' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        label_col = df.columns[-1]  # Assume last column is label
    
    feature_cols = [col for col in df.columns if col != label_col]
    
    # Map sit/stand to posture labels
    posture_map = {
        'sit': 1,      # Closed/defensive
        'stand': 0,    # Open/upright
        0: 0,
        1: 1
    }
    
    # Normalize features
    all_features = df[feature_cols].values.astype(np.float32)
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    global_std[global_std < 1e-6] = 1.0
    
    # Create output directories
    stats_dir = Path(output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / 'mean.npy', global_mean)
    np.save(stats_dir / 'std.npy', global_std)
    
    train_dir = Path(output_dir) / 'train' / 'sequences'
    val_dir = Path(output_dir) / 'val' / 'sequences'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sequences from chunks of data
    train_labels = {}
    val_labels = {}
    
    # Group consecutive samples into sequences
    seq_length = 30  # frames per sequence
    n_sequences = len(df) // seq_length
    
    np.random.seed(42)
    indices = np.random.permutation(n_sequences)
    n_train = int(len(indices) * 0.8)
    
    for i, idx in enumerate(indices):
        start = idx * seq_length
        end = start + seq_length
        
        chunk = df.iloc[start:end]
        features = chunk[feature_cols].values.astype(np.float32)
        features_normalized = (features - global_mean) / global_std
        
        # Get majority label for sequence
        labels_raw = chunk[label_col].values
        if isinstance(labels_raw[0], str):
            labels_mapped = [posture_map.get(l.lower().strip(), 0) for l in labels_raw]
        else:
            labels_mapped = [posture_map.get(l, 0) for l in labels_raw]
        
        posture = int(np.bincount(labels_mapped).argmax())
        
        # Infer stress from movement variance
        movement_var = np.var(np.diff(features, axis=0))
        if movement_var < 0.01:
            stress = 0  # Calm
        elif movement_var < 0.05:
            stress = 1  # Moderate
        else:
            stress = 2  # Restless
        
        seq_name = f"seq_{i:05d}"
        
        if i < n_train:
            np.save(train_dir / f"{seq_name}.npy", features_normalized)
            train_labels[seq_name] = {
                'posture': posture,
                'stress': stress,
                'trajectory': 0
            }
        else:
            np.save(val_dir / f"{seq_name}.npy", features_normalized)
            val_labels[seq_name] = {
                'posture': posture,
                'stress': stress,
                'trajectory': 0
            }
    
    # Save labels
    with open(Path(output_dir) / 'train' / 'labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    with open(Path(output_dir) / 'val' / 'labels.json', 'w') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"   ✓ Train: {len(train_labels)} sequences")
    print(f"   ✓ Val: {len(val_labels)} sequences")
    
    return {'train': len(train_labels), 'val': len(val_labels)}


def prepare_ntu_skeleton_dataset(skeleton_dir: str, output_dir: str) -> Dict:
    """
    Convert NTU RGB+D skeleton data to Sentry training format.
    
    NTU Format: 25 joints × 3 coordinates per frame
    Action classes mapped to posture/stress:
    - Standing actions → upright (0)
    - Sitting actions → closed (1)
    - Active actions → movement/stress indicators
    """
    print(f"\n📊 Processing NTU RGB+D skeleton dataset...")
    
    # Action to posture/stress mapping
    # NTU RGB+D action IDs (A001-A060)
    upright_actions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 27, 28, 29, 30}  # Standing, walking
    sitting_actions = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}  # Sitting related
    active_actions = {21, 22, 23, 24, 25, 26, 31, 32, 33, 34}  # Active movement
    
    skeleton_files = list(Path(skeleton_dir).glob('*.skeleton'))
    
    if not skeleton_files:
        print("   ⚠ No skeleton files found. Trying .npy format...")
        skeleton_files = list(Path(skeleton_dir).glob('*.npy'))
    
    print(f"   Found {len(skeleton_files)} skeleton files")
    
    if len(skeleton_files) == 0:
        print("   ✗ No skeleton files found in the directory")
        return {'train': 0, 'val': 0}
    
    # Process skeleton files
    train_labels = {}
    val_labels = {}
    
    train_dir = Path(output_dir) / 'train' / 'sequences'
    val_dir = Path(output_dir) / 'val' / 'sequences'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    np.random.shuffle(skeleton_files)
    n_train = int(len(skeleton_files) * 0.8)
    
    all_features = []
    
    for i, skeleton_file in enumerate(skeleton_files):
        try:
            # Parse filename for action ID (NTU format: SsssCcccPpppRrrrAaaa)
            fname = skeleton_file.stem
            if len(fname) >= 20 and fname[16] == 'A':
                action_id = int(fname[17:20])
            else:
                action_id = i % 60 + 1  # Default action
            
            # Load skeleton data
            if skeleton_file.suffix == '.npy':
                data = np.load(skeleton_file)
            else:
                data = _load_ntu_skeleton(skeleton_file)
            
            if data is None or len(data) < 10:
                continue
            
            all_features.append(data)
            
            # Determine labels
            if action_id in upright_actions:
                posture = 0
            elif action_id in sitting_actions:
                posture = 1
            else:
                posture = 2
            
            if action_id in active_actions:
                stress = 2  # Restless
            else:
                # Compute from movement variance
                movement = np.var(np.diff(data, axis=0))
                stress = 0 if movement < 0.1 else (1 if movement < 0.5 else 2)
            
            seq_name = skeleton_file.stem
            
            if i < n_train:
                np.save(train_dir / f"{seq_name}.npy", data.astype(np.float32))
                train_labels[seq_name] = {
                    'posture': posture,
                    'stress': stress,
                    'trajectory': 0
                }
            else:
                np.save(val_dir / f"{seq_name}.npy", data.astype(np.float32))
                val_labels[seq_name] = {
                    'posture': posture,
                    'stress': stress,
                    'trajectory': 0
                }
                
        except Exception as e:
            print(f"   ⚠ Error processing {skeleton_file.name}: {e}")
            continue
    
    # Compute normalization stats
    if all_features:
        all_data = np.vstack([f.reshape(-1, f.shape[-1]) for f in all_features])
        global_mean = np.mean(all_data, axis=0)
        global_std = np.std(all_data, axis=0)
        global_std[global_std < 1e-6] = 1.0
        
        np.save(Path(output_dir) / 'mean.npy', global_mean)
        np.save(Path(output_dir) / 'std.npy', global_std)
    
    # Save labels
    with open(Path(output_dir) / 'train' / 'labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    with open(Path(output_dir) / 'val' / 'labels.json', 'w') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"   ✓ Train: {len(train_labels)} sequences")
    print(f"   ✓ Val: {len(val_labels)} sequences")
    
    return {'train': len(train_labels), 'val': len(val_labels)}


def _load_ntu_skeleton(filepath: Path) -> Optional[np.ndarray]:
    """Load NTU RGB+D skeleton file format."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_frames = int(lines[0].strip())
        data = []
        line_idx = 1
        
        for frame in range(n_frames):
            n_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            frame_joints = []
            for body in range(n_bodies):
                # Skip body info line
                line_idx += 1
                n_joints = int(lines[line_idx].strip())
                line_idx += 1
                
                for joint in range(n_joints):
                    joint_data = lines[line_idx].strip().split()
                    x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                    frame_joints.extend([x, y, z])
                    line_idx += 1
            
            if frame_joints:
                data.append(frame_joints[:75])  # 25 joints × 3 coords
        
        return np.array(data) if data else None
        
    except Exception as e:
        return None


def setup_cmu_panoptic_toolbox(output_dir: str) -> bool:
    """Setup CMU Panoptic Studio toolbox."""
    print("\n🔧 Setting up CMU Panoptic Studio Toolbox...")
    
    toolbox_dir = Path(output_dir) / 'panoptic-toolbox'
    
    if toolbox_dir.exists():
        print(f"   ✓ Toolbox already exists at {toolbox_dir}")
    else:
        print("   Cloning repository...")
        result = os.system(f'git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox "{toolbox_dir}"')
        if result != 0:
            print("   ✗ Failed to clone repository")
            return False
    
    print("\n   📝 Instructions for downloading CMU Panoptic data:")
    print("   " + "=" * 50)
    print(f"   1. cd {toolbox_dir}")
    print("   2. ./scripts/getData.sh 171204_pose1_sample")
    print("   3. ./scripts/extractAll.sh 171204_pose1_sample")
    print("   " + "=" * 50)
    print("\n   Available sequences: https://domedb.perception.cs.cmu.edu/")
    
    return True


def _save_dataset(df, feature_cols, upper_col, lower_col, subject_col,
                  upper_body_map, lower_body_map, global_mean, global_std,
                  output_dir, dataset_name) -> Dict:
    """Common dataset saving logic."""
    
    stats_dir = Path(output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / 'mean.npy', global_mean)
    np.save(stats_dir / 'std.npy', global_std)
    
    train_dir = Path(output_dir) / 'train' / 'sequences'
    val_dir = Path(output_dir) / 'val' / 'sequences'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic subjects if needed
    if subject_col is None:
        df['_subject'] = (df.index // 300) % 13
        subject_col = '_subject'
    
    subjects = np.array(df[subject_col].unique())
    np.random.seed(42)
    np.random.shuffle(subjects)
    n_train = max(1, int(len(subjects) * 0.8))
    train_subjects = set(subjects[:n_train])
    
    train_labels = {}
    val_labels = {}
    
    for subject in subjects:
        subject_data = df[df[subject_col] == subject]
        features = subject_data[feature_cols].values.astype(np.float32)
        features_normalized = (features - global_mean) / global_std
        
        # Get labels
        posture = 0
        if upper_col:
            upper_label = subject_data[upper_col].mode().iloc[0] if len(subject_data[upper_col].mode()) > 0 else 'TUP'
            posture = upper_body_map.get(str(upper_label), 0)
        
        stress = 0
        if lower_col:
            lower_label = subject_data[lower_col].mode().iloc[0] if len(subject_data[lower_col].mode()) > 0 else 'LAP'
            stress = lower_body_map.get(str(lower_label), 0)
        
        seq_name = f"{dataset_name}_{subject}" if isinstance(subject, (int, np.integer)) else f"{dataset_name}_{str(subject).replace(' ', '_')}"
        
        if subject in train_subjects:
            np.save(train_dir / f"{seq_name}.npy", features_normalized)
            train_labels[seq_name] = {'posture': int(posture), 'stress': int(stress), 'trajectory': 0}
        else:
            np.save(val_dir / f"{seq_name}.npy", features_normalized)
            val_labels[seq_name] = {'posture': int(posture), 'stress': int(stress), 'trajectory': 0}
    
    with open(Path(output_dir) / 'train' / 'labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    with open(Path(output_dir) / 'val' / 'labels.json', 'w') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"   ✓ Train: {len(train_labels)} sequences")
    print(f"   ✓ Val: {len(val_labels)} sequences")
    
    return {'train': len(train_labels), 'val': len(val_labels)}


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def download_dataset(dataset_key: str, data_dir: Path) -> bool:
    """Download and process a specific dataset."""
    
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    ds = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"📦 {ds['name']}")
    print(f"   {ds['description']}")
    print(f"   Size: {ds['size']}")
    print(f"{'='*60}")
    
    raw_dir = data_dir / 'raw' / dataset_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    success = False
    
    if dataset_key == 'multiposture':
        csv_path = raw_dir / 'multiposture_data.csv'
        if not csv_path.exists():
            success = download_file(ds['url'], str(csv_path))
        else:
            print(f"   ✓ Already downloaded: {csv_path}")
            success = True
        
        if success:
            prepare_multiposture_dataset(str(csv_path), str(data_dir))
    
    elif dataset_key == 'figshare':
        csv_path = raw_dir / 'sitstand_pose.csv'
        if not csv_path.exists():
            success = download_file(ds['url'], str(csv_path))
        else:
            print(f"   ✓ Already downloaded: {csv_path}")
            success = True
        
        if success:
            prepare_figshare_sitstand_dataset(str(csv_path), str(data_dir))
    
    elif dataset_key == 'ntu_skeleton':
        skeleton_dir = raw_dir / 'nturgb+d_skeletons'
        
        if not skeleton_dir.exists() or len(list(skeleton_dir.glob('*'))) == 0:
            print("\n   ⚠ NTU RGB+D requires manual download due to size (~5.8 GB)")
            print("\n   Options:")
            print("   1. Google Drive: https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H")
            print("   2. Official site: https://rose1.ntu.edu.sg/dataset/actionRecognition/")
            print(f"\n   After downloading, extract to: {skeleton_dir}")
            
            # Try with gdown if user wants
            choice = input("\n   Try automatic download with gdown? [y/N]: ").strip().lower()
            if choice == 'y':
                zip_path = raw_dir / 'nturgbd_skeletons.zip'
                success = download_from_gdrive(ds['gdrive_id'], str(zip_path))
                if success:
                    extract_zip(str(zip_path), str(raw_dir))
        else:
            print(f"   ✓ Skeleton data exists: {skeleton_dir}")
            success = True
        
        if success or skeleton_dir.exists():
            prepare_ntu_skeleton_dataset(str(skeleton_dir), str(data_dir))
    
    elif dataset_key == 'cmu_panoptic':
        success = setup_cmu_panoptic_toolbox(str(raw_dir))
    
    return success


def download_all(data_dir: Path):
    """Download all directly accessible datasets."""
    print("\n" + "="*60)
    print("🚀 Downloading All Video Posture Datasets")
    print("="*60)
    
    # Download datasets that don't require manual intervention
    for key in ['multiposture', 'figshare']:
        download_dataset(key, data_dir)
    
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    print("✓ Downloaded: MultiPosture, Figshare Sit/Stand")
    print("\n⚠ Manual download required:")
    print("  - NTU RGB+D: Run with --dataset ntu_skeleton")
    print("  - CMU Panoptic: Run with --dataset cmu_panoptic --setup-only")


def list_datasets():
    """List all available datasets."""
    print("\n📋 Available Video Posture Datasets:")
    print("="*70)
    for key, ds in DATASETS.items():
        print(f"\n  {key}:")
        print(f"    Name: {ds['name']}")
        print(f"    Description: {ds['description']}")
        print(f"    Size: {ds['size']}")
        print(f"    Format: {ds['format']}")


def main():
    parser = argparse.ArgumentParser(
        description='Download video posture datasets for Sentry training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_video_posture_datasets.py --list
  python download_video_posture_datasets.py --dataset all
  python download_video_posture_datasets.py --dataset figshare
  python download_video_posture_datasets.py --dataset ntu_skeleton
  python download_video_posture_datasets.py --dataset cmu_panoptic --setup-only
        """
    )
    
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'multiposture', 'figshare', 'ntu_skeleton', 'cmu_panoptic'],
                        help='Dataset to download (default: all)')
    parser.add_argument('--data-dir', type=str, default='data/posture',
                        help='Output directory (default: data/posture)')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only setup toolbox, do not download data')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'all':
        download_all(data_dir)
    else:
        download_dataset(args.dataset, data_dir)
    
    print("\n" + "="*60)
    print("✅ Dataset download complete!")
    print("="*60)
    print(f"\nTo train with this data:")
    print(f"  python train.py posture --data {data_dir} --epochs 50")


if __name__ == "__main__":
    main()
