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
    response = requests.get(url, stream=True)
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
    - 33 input dimensions (11 joints x 3 coordinates)
    - Upper body labels: TUP, TLB, TLF, TLR, TLL
    - Lower body labels: LAP, LWA, LCS, LCR, LCL, LLR, LLL
    """
    print(f"\nPreparing dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Map upper body labels to our posture categories
    # TUP (upright) -> 0, TLF/TLB (slouched/leaning) -> 1, 
    # Open posture -> 2, Closed posture -> 3
    upper_body_map = {
        'TUP': 0,  # Upright
        'TLB': 1,  # Leaning backward (slouched)
        'TLF': 1,  # Leaning forward (slouched)
        'TLR': 1,  # Leaning right (slouched)
        'TLL': 1,  # Leaning left (slouched)
    }
    
    # Map lower body labels for stress indicators
    # Wide/apart legs = open/calm -> 0
    # Closed/crossed = closed/fidgeting -> 1-3
    lower_body_map = {
        'LAP': 0,  # Legs apart - calm
        'LWA': 0,  # Legs wide apart - calm
        'LCS': 1,  # Legs closed - potential stress
        'LCR': 2,  # Legs crossed right - fidgeting/restless
        'LCL': 2,  # Legs crossed left - fidgeting/restless
        'LLR': 1,  # Legs lateral right
        'LLL': 1,  # Legs lateral left
    }
    
    # Find the label columns
    upper_col = None
    lower_col = None
    feature_cols = []
    
    for col in df.columns:
        if 'upper' in col.lower() or 'trunk' in col.lower():
            upper_col = col
        elif 'lower' in col.lower() or 'leg' in col.lower():
            lower_col = col
        elif col not in ['participant', 'frame', 'label', 'Upper', 'Lower']:
            feature_cols.append(col)
    
    print(f"Upper body label column: {upper_col}")
    print(f"Lower body label column: {lower_col}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Create output directories
    train_dir = Path(output_dir) / 'train' / 'sequences'
    val_dir = Path(output_dir) / 'val' / 'sequences'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by participant for train/val split
    if 'participant' in df.columns:
        participants = df['participant'].unique()
    else:
        # Create synthetic participant IDs
        df['participant'] = (df.index // 300) % 13  # Assume ~300 frames per participant
        participants = df['participant'].unique()
    
    # 80/20 train/val split by participant
    np.random.seed(42)
    np.random.shuffle(participants)
    n_train = int(len(participants) * 0.8)
    train_participants = set(participants[:n_train])
    val_participants = set(participants[n_train:])
    
    train_labels = {}
    val_labels = {}
    
    # Process each participant's data as a sequence
    for participant in participants:
        participant_data = df[df['participant'] == participant]
        
        # Extract features (use all numeric columns if feature_cols is empty)
        if feature_cols:
            features = participant_data[feature_cols].values
        else:
            numeric_cols = participant_data.select_dtypes(include=[np.number]).columns
            features = participant_data[numeric_cols].values
        
        # Determine labels
        if upper_col and upper_col in participant_data.columns:
            upper_label = participant_data[upper_col].mode().iloc[0]
            posture = upper_body_map.get(str(upper_label), 0)
        else:
            posture = 0
        
        if lower_col and lower_col in participant_data.columns:
            lower_label = participant_data[lower_col].mode().iloc[0]
            stress = lower_body_map.get(str(lower_label), 0)
        else:
            stress = 0
        
        # Trajectory is 0 (stable) for static dataset
        trajectory = 0
        
        seq_name = f"participant_{participant:02d}"
        
        if participant in train_participants:
            seq_path = train_dir / f"{seq_name}.npy"
            np.save(seq_path, features.astype(np.float32))
            train_labels[seq_name] = {
                'posture': int(posture),
                'stress': int(stress),
                'trajectory': trajectory
            }
            print(f"Saved train sequence: {seq_name} ({len(features)} frames)")
        else:
            seq_path = val_dir / f"{seq_name}.npy"
            np.save(seq_path, features.astype(np.float32))
            val_labels[seq_name] = {
                'posture': int(posture),
                'stress': int(stress),
                'trajectory': trajectory
            }
            print(f"Saved val sequence: {seq_name} ({len(features)} frames)")
    
    # Save labels
    with open(Path(output_dir) / 'train' / 'labels.json', 'w') as f:
        json.dump(train_labels, f, indent=2)
    
    with open(Path(output_dir) / 'val' / 'labels.json', 'w') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"\nDataset prepared!")
    print(f"Train: {len(train_labels)} sequences")
    print(f"Val: {len(val_labels)} sequences")


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
    
    # Prepare for training
    prepare_multiposture_dataset(csv_path, data_dir)
    
    print("\n" + "="*50)
    print("Dataset ready for training!")
    print("Run: python train.py posture --data data/posture --epochs 50")
    print("="*50)


if __name__ == "__main__":
    main()
