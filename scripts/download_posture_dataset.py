"""
Download and prepare the MultiPosture dataset for Sentry posture training.
"""
import os
import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path

# URLs
MULTIPOSTURE_URL = "https://zenodo.org/records/14230872/files/data.csv?download=1"

def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\nSaved to {output_path}")


def prepare_multiposture_dataset(csv_path, output_dir):
    """
    Convert MultiPosture CSV to Sentry training format.
    
    The dataset has:
    - 100 input dimensions (body keypoints x, y, z)
    - Upper body labels: TUP, TLB, TLF, TLR, TLL
    - Lower body labels: LAP, LWA, LCS, LCR, LCL, LLR, LLL
    """
    print(f"\nPreparing dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)[:10]}... (total: {len(df.columns)})")
    
    # Map upper body labels to our posture categories
    upper_body_map = {
        'TUP': 0,  # Upright
        'TLB': 1,  # Leaning backward (slouched)
        'TLF': 1,  # Leaning forward (slouched)
        'TLR': 2,  # Leaning right (open/relaxed)
        'TLL': 2,  # Leaning left (open/relaxed)
    }
    
    # Map lower body labels for stress indicators
    lower_body_map = {
        'LAP': 0,  # Legs apart - calm
        'LWA': 0,  # Legs wide apart - calm
        'LCS': 1,  # Legs closed - potential stress
        'LCR': 2,  # Legs crossed right - fidgeting/restless
        'LCL': 2,  # Legs crossed left - fidgeting/restless
        'LLR': 1,  # Legs lateral right
        'LLL': 1,  # Legs lateral left
    }
    
    # Find the label and feature columns
    upper_col = None
    lower_col = None
    subject_col = None
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
    
    # If no specific feature columns found, use all numeric except labels
    if not feature_cols:
        for col in df.columns:
            if col not in [upper_col, lower_col, subject_col] and df[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
    
    print(f"Upper body label column: {upper_col}")
    print(f"Lower body label column: {lower_col}")
    print(f"Subject column: {subject_col}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Extract all features for normalization
    all_features = df[feature_cols].values.astype(np.float32)
    
    # Compute global mean and std for normalization
    global_mean = np.mean(all_features, axis=0)
    global_std = np.std(all_features, axis=0)
    global_std[global_std < 1e-6] = 1.0  # Avoid division by zero
    
    print(f"Feature stats - Mean range: [{global_mean.min():.3f}, {global_mean.max():.3f}]")
    print(f"Feature stats - Std range: [{global_std.min():.3f}, {global_std.max():.3f}]")
    
    # Save normalization stats
    stats_dir = Path(output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / 'mean.npy', global_mean)
    np.save(stats_dir / 'std.npy', global_std)
    print(f"Saved normalization stats to {stats_dir}")
    
    # Create output directories
    train_dir = Path(output_dir) / 'train' / 'sequences'
    val_dir = Path(output_dir) / 'val' / 'sequences'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique subjects
    if subject_col:
        subjects = df[subject_col].unique()
    else:
        # Create synthetic subject IDs based on index
        df['_subject'] = (df.index // 300) % 13
        subject_col = '_subject'
        subjects = df[subject_col].unique()
    
    print(f"Found {len(subjects)} subjects")
    
    # 80/20 train/val split by subject
    np.random.seed(42)
    subjects = np.array(subjects)
    np.random.shuffle(subjects)
    n_train = max(1, int(len(subjects) * 0.8))
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:])
    
    train_labels = {}
    val_labels = {}
    
    # Process each subject's data as a sequence
    for subject in subjects:
        subject_data = df[df[subject_col] == subject]
        
        # Extract and normalize features
        features = subject_data[feature_cols].values.astype(np.float32)
        features_normalized = (features - global_mean) / global_std
        
        # Determine labels from mode (most common value)
        if upper_col:
            upper_label = subject_data[upper_col].mode().iloc[0] if len(subject_data[upper_col].mode()) > 0 else 'TUP'
            posture = upper_body_map.get(str(upper_label), 0)
        else:
            posture = 0
        
        if lower_col:
            lower_label = subject_data[lower_col].mode().iloc[0] if len(subject_data[lower_col].mode()) > 0 else 'LAP'
            stress = lower_body_map.get(str(lower_label), 0)
        else:
            stress = 0
        
        # Trajectory is 0 (stable) for static dataset
        trajectory = 0
        
        # Handle different subject ID types
        if isinstance(subject, (int, np.integer)):
            seq_name = f"subject_{int(subject):03d}"
        else:
            seq_name = f"subject_{str(subject).replace(' ', '_')}"
        
        if subject in train_subjects:
            seq_path = train_dir / f"{seq_name}.npy"
            np.save(seq_path, features_normalized)
            train_labels[seq_name] = {
                'posture': int(posture),
                'stress': int(stress),
                'trajectory': int(trajectory)
            }
            print(f"Train: {seq_name} ({len(features)} frames) - posture={posture}, stress={stress}")
        else:
            seq_path = val_dir / f"{seq_name}.npy"
            np.save(seq_path, features_normalized)
            val_labels[seq_name] = {
                'posture': int(posture),
                'stress': int(stress),
                'trajectory': int(trajectory)
            }
            print(f"Val: {seq_name} ({len(features)} frames) - posture={posture}, stress={stress}")
    
    # Save labels
    with open(Path(output_dir) / 'train' / 'labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    
    with open(Path(output_dir) / 'val' / 'labels.json', 'w') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"\nDataset prepared!")
    print(f"Train: {len(train_labels)} sequences")
    print(f"Val: {len(val_labels)} sequences")
    
    # Print label distribution
    print("\nLabel distribution:")
    posture_dist = {}
    stress_dist = {}
    for labels in list(train_labels.values()) + list(val_labels.values()):
        posture_dist[labels['posture']] = posture_dist.get(labels['posture'], 0) + 1
        stress_dist[labels['stress']] = stress_dist.get(labels['stress'], 0) + 1
    print(f"  Posture: {posture_dist}")
    print(f"  Stress: {stress_dist}")


def main():
    # Setup paths
    data_dir = Path("data/posture")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download MultiPosture dataset
    csv_path = raw_dir / "multiposture_data.csv"
    if not csv_path.exists():
        download_file(MULTIPOSTURE_URL, str(csv_path))
    else:
        print(f"Dataset already exists at {csv_path}")
    
    # Clear old processed data
    train_seq_dir = data_dir / 'train' / 'sequences'
    val_seq_dir = data_dir / 'val' / 'sequences'
    if train_seq_dir.exists():
        for f in train_seq_dir.glob('*.npy'):
            f.unlink()
    if val_seq_dir.exists():
        for f in val_seq_dir.glob('*.npy'):
            f.unlink()
    
    # Prepare for training with normalization
    prepare_multiposture_dataset(csv_path, data_dir)
    
    print("\n" + "="*50)
    print("Dataset ready for training!")
    print("Run: python train.py posture --data data/posture --epochs 50")
    print("="*50)


if __name__ == "__main__":
    main()
