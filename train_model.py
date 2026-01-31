"""
Oral Cancer Detection Model - High Performance Training Script (PyTorch)
======================================================================
Multi-Factor Cancer Assessment System
- Deep Learning Framework: PyTorch (Optimized for NVIDIA GPUs)
- Architecture: MobileNetV2 (Transfer Learning)
- Features: Real-time Augmentation, GPU Acceleration, Best Model Checkpointing
- **Robust Image Loading**: Skips corrupted images during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import copy
from datetime import datetime
import numpy as np
from PIL import Image, ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_DIR = 'organized_dataset/train'
VAL_DIR = 'organized_dataset/val'
TEST_DIR = 'organized_dataset/test'
MODEL_SAVE_PATH = 'oral_cancer_model_v2.pth'

print("=" * 70, flush=True)
print("ORAL CANCER DETECTION MODEL - PYTORCH TRAINING", flush=True)
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("WARNING: GPU not detected. Training will be slow on CPU.", flush=True)
print("=" * 70, flush=True)

# Robust Image Loader
def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, SyntaxError) as e:
        print(f"Skipping bad image: {path}")
        # Return a black image of correct size to prevent crash
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE))

def train_model():
    # ==================== DATA PREPARATION ====================
    print("\n[1/5] PREPARING DATA LOADERS", flush=True)
    
    # Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Custom Dataset class to use safe loader
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train'], loader=safe_pil_loader),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'], loader=safe_pil_loader),
        'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'], loader=safe_pil_loader)
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=0)
                  for x in ['train', 'val', 'test']} # num_workers=0 for Windows stability
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes: {class_names}", flush=True)
    print(f"Training images: {dataset_sizes['train']}", flush=True)
    print(f"Validation images: {dataset_sizes['val']}", flush=True)
    print(f"Test images: {dataset_sizes['test']}", flush=True)

    # ==================== MODEL ARCHITECTURE ====================
    print("\n[2/5] INITIALIZING MOBILENETV2", flush=True)
    
    model = models.mobilenet_v2(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    model = model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    # ==================== TRAINING LOOP ====================
    print(f"\n[3/5] STARTING TRAINING ON {DEVICE}", flush=True)
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}", flush=True)
    print("-" * 70, flush=True)
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Progress monitoring
            batch_count = 0
            total_batches = len(dataloaders[phase])

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Batch {batch_count}/{total_batches} - Loss: {loss.item():.4f}", flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoint.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    
    # ==================== SAVING ====================
    print("\n[5/5] SAVING FINAL MODEL")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")
    
    with open('model_classes.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    return model

if __name__ == "__main__":
    train_model()
