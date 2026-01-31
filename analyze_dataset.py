"""
Complete Dataset Analysis and Organization Script
==================================================
Finds all images in nested directories and organizes them by class.
"""

import os
import glob
from collections import defaultdict

# Dataset directory
DATASET_DIR = "dataset"

def find_all_images(directory):
    """Recursively find all images in directory"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, '**', ext)
        all_images.extend(glob.glob(pattern, recursive=True))
    
    return all_images

def analyze_nested_dataset():
    """Analyze dataset with nested structure"""
    
    print("=" * 70)
    print("ORAL CANCER DATASET - COMPLETE ANALYSIS")
    print("=" * 70)
    
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR} folder not found!")
        return
    
    # Find all images recursively
    all_images = find_all_images(DATASET_DIR)
    
    print(f"\n[OK] Found {len(all_images)} total images in dataset")
    print("-" * 70)
    
    # Categorize by folder structure
    categorized = defaultdict(list)
    
    for img_path in all_images:
        # Normalize path
        norm_path = os.path.normpath(img_path)
        parts = norm_path.split(os.sep)
        
        # Look for class indicators in path
        path_str = norm_path.upper()
        
        if 'CANCER' in path_str and 'NON' not in path_str:
            categorized['CANCER'].append(img_path)
        elif 'NON CANCER' in path_str or 'NONCANCER' in path_str or 'NORMAL' in path_str:
            categorized['NON_CANCER'].append(img_path)
        else:
            categorized['UNKNOWN'].append(img_path)
    
    print("\nCLASS DISTRIBUTION:")
    print("-" * 70)
    
    for class_name, images in sorted(categorized.items()):
        print(f"{class_name:20s}: {len(images):6d} images")
        if images:
            print(f"  Sample: {os.path.basename(images[0])}")
            print(f"  Path:   {os.path.dirname(images[0])}")
    
    print("-" * 70)
    
    # Check for unknown categorization
    if categorized['UNKNOWN']:
        print(f"\n[WARNING] {len(categorized['UNKNOWN'])} images couldn't be auto-categorized")
        print("  Sample paths:")
        for img in categorized['UNKNOWN'][:3]:
            print(f"    - {img}")
    
    print("\n" + "=" * 70)
    print("PROPOSED TRAIN/VAL/TEST SPLIT (70/15/15):")
    print("=" * 70)
    
    for class_name in ['CANCER', 'NON_CANCER']:
        if class_name in categorized:
            total = len(categorized[class_name])
            train = int(total * 0.70)
            val = int(total * 0.15)
            test = total - train - val
            
            print(f"\n{class_name}:")
            print(f"  Total:      {total:6d}")
            print(f"  Train:      {train:6d} (70%)")
            print(f"  Validation: {val:6d} (15%)")
            print(f"  Test:       {test:6d} (15%)")
    
    print("\n" + "=" * 70)
    
    return categorized

if __name__ == "__main__":
    categorized = analyze_nested_dataset()
    
    # Summary
    cancer_count = len(categorized.get('CANCER', []))
    non_cancer_count = len(categorized.get('NON_CANCER', []))
    
    print("\n[OK] ANALYSIS COMPLETE")
    print(f"  - Cancer images: {cancer_count}")
    print(f"  - Non-cancer images: {non_cancer_count}")
    print(f"  - Total: {cancer_count + non_cancer_count}")
    print("\nReady for data splitting!")
    print("=" * 70)
