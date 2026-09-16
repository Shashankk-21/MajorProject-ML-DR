import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import get_train_transforms

def generate_visual_grid():
    print("\n" + "=" * 60)
    print("2. VISUAL GRID GENERATION")
    print("=" * 60)
    
    # Use the known real image path from your earlier augmentation tests
    raw_path = Path(r"C:\MajorProject - Datasets\APTOS-2019-Kaggle\train_images\train_images\1b8ad0afe9fb.png")
    
    if not raw_path.exists():
        print(f"SKIPPED: Could not find raw image at {raw_path}")
        return
        
    raw_img = np.array(Image.open(raw_path).convert("RGB"))
    proc_img, _ = run_offline_pipeline(raw_img)
    
    transform = get_train_transforms()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Raw Input\n{raw_img.shape[:2]}")
    
    axes[1].imshow(proc_img)
    axes[1].set_title(f"Processed (Offline)\n{proc_img.shape[:2]}")
    
    # Generate 3 stochastic training augmentations
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(2, 5):
        aug_tensor = transform(image=proc_img)["image"]
        # Denormalize tensor for visualization: (C, H, W) -> (H, W, C)
        aug_np = aug_tensor.permute(1, 2, 0).numpy()
        aug_np = std * aug_np + mean
        aug_np = np.clip(aug_np, 0, 1)
        
        axes[i].imshow(aug_np)
        axes[i].set_title(f"Stochastic Aug {i-1}")
        
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    save_path = Path("visual_audit_grid.png")
    plt.savefig(save_path, dpi=150)
    print(f"PASS: Grid rendered and saved to '{save_path.absolute()}'.")
    print("-> Open this image file from your VS Code file explorer to visually confirm retinas look intact.")

if __name__ == "__main__":
    generate_visual_grid()