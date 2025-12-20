import os
import shutil
import random
from pathlib import Path

def create_val_split(data_dir: str, val_split: float = 0.1):
    """
    Create a 'val' directory by moving a percentage of files from 'train'.
    """
    data_path = Path(data_dir)
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if not train_path.exists():
        print(f"Error: Train directory {train_path} not found.")
        return

    if val_path.exists():
        print(f"Notice: Val directory {val_path} already exists. Skipping manual split.")
        # Optional: check if it's empty or has files
        return

    val_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate through emotion classes
    for emotion_dir in train_path.iterdir():
        if not emotion_dir.is_dir():
            continue
            
        emotion = emotion_dir.name
        files = list(emotion_dir.glob('*.*'))
        num_val = int(len(files) * val_split)
        
        print(f"Processing {emotion}: {len(files)} files, moving {num_val} to val...")
        
        # Create corresponding emotion dir in val
        (val_path / emotion).mkdir(parents=True, exist_ok=True)
        
        # Randomly select files to move
        val_files = random.sample(files, num_val)
        for f in val_files:
            shutil.move(str(f), str(val_path / emotion / f.name))

    print("Validation split created successfully!")

if __name__ == "__main__":
    create_val_split("C:/sentry/data/FER2013")
