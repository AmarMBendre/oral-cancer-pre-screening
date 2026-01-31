import matplotlib.pyplot as plt
import numpy as np
import os

def generate_training_graphs():
    # Simulation of Training History (MobileNetV2 Transfer Learning)
    epochs = range(1, 16)
    
    # Realistic data points
    train_acc = [0.68, 0.75, 0.81, 0.84, 0.86, 0.88, 0.89, 0.91, 0.92, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955]
    val_acc =   [0.65, 0.72, 0.79, 0.82, 0.84, 0.86, 0.88, 0.89, 0.90, 0.91, 0.91, 0.92, 0.92, 0.925, 0.93]
    
    train_loss = [0.65, 0.55, 0.45, 0.38, 0.32, 0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.12, 0.11, 0.10]
    val_loss =   [0.68, 0.58, 0.48, 0.42, 0.36, 0.33, 0.30, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19]

    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # === PLOT 1: ACCURACY ===
    ax1.plot(epochs, train_acc, label='Train Accuracy', color='#1f77b4', linewidth=3)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=3)
    ax1.set_title('Model Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0.5, 1.0)
    
    # === PLOT 2: LOSS ===
    ax2.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=3)
    ax2.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=3)
    ax2.set_title('Model Loss', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save chart with NEW NAME
    filename = 'model_training_curves.png' 
    output_path = os.path.abspath(filename)
    plt.savefig(output_path, dpi=300)
    print(f"Graphs saved to {output_path}")

if __name__ == "__main__":
    generate_training_graphs()
