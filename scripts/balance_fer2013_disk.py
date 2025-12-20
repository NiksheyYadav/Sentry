import os
import shutil
import random
from pathlib import Path

def balance_dataset_on_disk(data_dir: str, target_count: int = 5000):
    """
    Physically balance classes in the training directory to target_count.
    Undersamples by deleting random files.
    Oversamples by duplicating files with new names.
    """
    train_path = Path(data_dir) / 'train'
    
    if not train_path.exists():
        print(f"Error: Train directory {train_path} not found.")
        return

    # Categories to balance (everything currently in train)
    categories = [d.name for d in train_path.iterdir() if d.is_dir()]
    
    print(f"Balancing categories to {target_count} files each: {categories}")

    for category in categories:
        cat_path = train_path / category
        files = list(cat_path.glob('*.*'))
        current_count = len(files)
        
        if current_count == 0:
            print(f"Skipping {category}: No files found.")
            continue

        if current_count > target_count:
            # Undersample: delete surplus
            surplus_count = current_count - target_count
            files_to_delete = random.sample(files, surplus_count)
            print(f"  {category}: {current_count} files. Deleting {surplus_count} surplus files...")
            for f in files_to_delete:
                f.unlink()
        
        elif current_count < target_count:
            # Oversample: duplicate existing files
            needed_count = target_count - current_count
            print(f"  {category}: {current_count} files. Duplicating to add {needed_count} files...")
            
            # Simple oversampling: repeat existing files until we reach target
            i = 0
            while i < needed_count:
                source_file = random.choice(files)
                new_name = f"aug_{i}_{source_file.name}"
                dest_path = cat_path / new_name
                shutil.copy2(str(source_file), str(dest_path))
                i += 1
        
        else:
            print(f"  {category}: Already has {target_count} files. Skipping.")

    print("\nPhysical balancing complete!")
    # Verify
    for category in categories:
        count = len(list((train_path / category).glob('*.*')))
        print(f"Category {category}: {count} files")

if __name__ == "__main__":
    balance_dataset_on_disk("C:/sentry/data/FER2013", 5000)
    # Verification also for val (optional, but user asked for 5000 usually implies train)
    # The val split was 10%, so it should be around 300-700. We keep it as is unless specified.
