"""
Dataset Organization and Splitting Script
==========================================
Organizes dataset into train/val/test splits (70/15/15)
Creates proper directory structure for model training.
"""

import os
import shutil
import glob
import random
from pathlib import Path

# Configuration
DATASET_DIR = "dataset"
OUTPUT_DIR = "organized_dataset"
SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

random.seed(42)  # For reproducibility

def find_all_images(directory):
    """Recursively find all image files"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, '**', ext)
        all_images.extend(glob.glob(pattern, recursive=True))
    
    return all_images

def categorize_images():
    """Categorize all images into cancer/non-cancer"""
    
    all_images = find_all_images(DATASET_DIR)
    
    cancer_images = []
    non_cancer_images = []
    
    for img_path in all_images:
        path_upper = img_path.upper()
        
        # Check if path contains 'CANCER' folder but NOT 'NON'
        if '\\CANCER\\' in path_upper or '/CANCER/' in path_upper:
            if 'NON' not in path_upper:
                cancer_images.append(img_path)
            else:
                non_cancer_images.append(img_path)
        # Check for non-cancer keywords
        elif any(keyword in path_upper for keyword in ['NON-CANCER', 'NONCANCER', 'NORMAL']):
            non_cancer_images.append(img_path)
        else:
            # Assume remaining are non-cancer (based on folder structure)
            non_cancer_images.append(img_path)
    
    return cancer_images, non_cancer_images

def create_splits(images, class_name):
    """Split images into train/val/test"""
    
    # Shuffle images
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * SPLIT_RATIOS['train'])
    val_end = train_end + int(total * SPLIT_RATIOS['val'])
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"\n{class_name}:")
    print(f"  Total: {total}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
    
    return splits

def organize_dataset():
    """Main function to organize dataset"""
    
    print("=" * 70)
    print("DATASET ORGANIZATION & SPLITTING")
    print("=" * 70)
    
    # Categorize images
    print("\n[1/3] Categorizing images...")
    cancer_images, non_cancer_images = categorize_images()
    
    print(f"  Cancer images: {len(cancer_images)}")
    print(f"  Non-cancer images: {len(non_cancer_images)}")
    
    # Create splits
    print("\n[2/3] Creating train/val/test splits (70/15/15)...")
    cancer_splits = create_splits(cancer_images, 'CANCER')
    non_cancer_splits = create_splits(non_cancer_images, 'NON_CANCER')
    
    # Create directory structure
    print("\n[3/3] Creating organized directory structure...")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"  [WARNING] {OUTPUT_DIR} already exists. Removing...")
        shutil.rmtree(OUTPUT_DIR)
    
    # Create folders
    for split in ['train', 'val', 'test']:
        for class_name in ['cancer', 'non_cancer']:
            path = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(path, exist_ok=True)
    
    # Copy files
    print("\n  Copying files...")
    
    for split in ['train', 'val', 'test']:
        # Cancer images
        dest_dir = os.path.join(OUTPUT_DIR, split, 'cancer')
        for i, src in enumerate(cancer_splits[split], 1):
            ext = os.path.splitext(src)[1]
            dest = os.path.join(dest_dir, f"cancer_{split}_{i:05d}{ext}")
            shutil.copy2(src, dest)
        
        # Non-cancer images
        dest_dir = os.path.join(OUTPUT_DIR, split, 'non_cancer')
        for i, src in enumerate(non_cancer_splits[split], 1):
            ext = os.path.splitext(src)[1]
            dest = os.path.join(dest_dir, f"non_cancer_{split}_{i:05d}{ext}")
            shutil.copy2(src, dest)
    
    print("\n" + "=" * 70)
    print("ORGANIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nOrganized dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("\nDirectory structure:")
    print(f"{OUTPUT_DIR}/")
    print("  ├── train/")
    print("  │   ├── cancer/")
    print("  │   └── non_cancer/")
    print("  ├── val/")
    print("  │   ├── cancer/")
    print("  │   └── non_cancer/")
    print("  └── test/")
    print("      ├── cancer/")
    print("      └── non_cancer/")
    
    # Verify
    print("\n" + "=" * 70)
    print("VERIFICATION:")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        cancer_count = len(os.listdir(os.path.join(OUTPUT_DIR, split, 'cancer')))
        non_cancer_count = len(os.listdir(os.path.join(OUTPUT_DIR, split, 'non_cancer')))
        print(f"\n{split.upper()}:")
        print(f"  Cancer:     {cancer_count:6d} images")
        print(f"  Non-cancer: {non_cancer_count:6d} images")
        print(f"  Total:      {cancer_count + non_cancer_count:6d} images")
    
    print("\n" + "=" * 70)
    print("[OK] Dataset is ready for training!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        organize_dataset()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
