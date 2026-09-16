import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import get_train_transforms

DATA_ROOT = Path(r"C:\MajorProject - Datasets")

def check_true_crop_rates():
    print("=" * 60)
    print("1. TRUE CROP RATES (FROM FINAL MANIFESTS)")
    print("=" * 60)
    for client in ["APTOS", "IDRiD", "Messidor-2", "EyePACS"]:
        manifest_path = Path(f"data/splits/{client}_manifest.csv")
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            n_cropped = df["was_cropped"].sum()
            total = len(df)
            print(f"{client:12s}: {n_cropped}/{total} images cropped ({n_cropped/total*100:.1f}%)")
        else:
            print(f"{client:12s}: Manifest not found.")

def generate_client_grid(client, raw_path_str):
    raw_path = Path(raw_path_str)
    if not raw_path.exists():
        print(f"SKIPPED: {client} raw image missing at {raw_path}")
        return
        
    raw_img = np.array(Image.open(raw_path).convert("RGB"))
    proc_img, _ = run_offline_pipeline(raw_img)
    transform = get_train_transforms()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"{client} Raw\n{raw_img.shape[:2]}")
    
    axes[1].imshow(proc_img)
    axes[1].set_title(f"Processed\n{proc_img.shape[:2]}")
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(2, 4):
        aug_tensor = transform(image=proc_img)["image"]
        aug_np = aug_tensor.permute(1, 2, 0).numpy()
        aug_np = np.clip(std * aug_np + mean, 0, 1)
        axes[i].imshow(aug_np)
        axes[i].set_title(f"Stochastic Aug {i-1}")
        
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    save_path = f"visual_audit_{client}.jpg"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PASS: {client} visual grid saved to {save_path}")

def generate_50_aug_stress_test():
    print("\n" + "=" * 60)
    print("3. AUGMENTATION STACKING STRESS TEST (50 DRAWS)")
    print("=" * 60)
    
    raw_path = DATA_ROOT / "APTOS-2019-Kaggle/train_images/train_images/1b8ad0afe9fb.png"
    if not raw_path.exists():
        print("SKIPPED: APTOS raw image missing.")
        return
        
    raw_img = np.array(Image.open(raw_path).convert("RGB"))
    proc_img, _ = run_offline_pipeline(raw_img)
    transform = get_train_transforms()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(50):
        aug_tensor = transform(image=proc_img)["image"]
        aug_np = aug_tensor.permute(1, 2, 0).numpy()
        aug_np = np.clip(std * aug_np + mean, 0, 1)
        axes[i].imshow(aug_np)
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = "visual_audit_50_augs.jpg"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PASS: 50-draw stress test saved to {save_path}")
    print("-> Review this image. If only ~5-10 out of 50 look overly degraded, the probabilities are perfect.")

if __name__ == "__main__":
    check_true_crop_rates()
    
    print("\n" + "=" * 60)
    print("2. VISUAL GRIDS FOR IDRID & MESSIDOR-2")
    print("=" * 60)
    # Using known valid image paths based on standard dataset structures
    generate_client_grid("IDRiD", DATA_ROOT / "Disease Grading Images - IDRiD/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg")
    generate_client_grid("Messidor-2", DATA_ROOT / "Messidor-2-Dataset/Messidor-2-Dataset/IMAGES/20051019_38557_0100_PP.png")
    
    generate_50_aug_stress_test()