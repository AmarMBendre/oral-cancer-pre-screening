"""
Dataset Cleaner
===============
Verifies all images in the organized_dataset directory.
Deletes any file that cannot be opened by PIL.
"""

from PIL import Image
import os
import glob

DATASET_DIR = "organized_dataset"

def verify_images():
    print(f"Verifying images in {DATASET_DIR}...")
    
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.*"), recursive=True)
    bad_files = 0
    
    print(f"Checking {len(files)} files...")
    
    for file_path in files:
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        try:
            with Image.open(file_path) as img:
                img.verify() # Verify it's an image
        except (IOError, SyntaxError) as e:
            print(f"BAD FILE: {file_path} - Removing...")
            try:
                os.remove(file_path)
                bad_files += 1
            except Exception as del_err:
                print(f"Could not delete: {del_err}")

    print("-" * 50)
    print(f"Verification Complete.")
    print(f"Removed {bad_files} corrupted images.")

if __name__ == "__main__":
    verify_images()
