import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import albumentations as A
from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import RandomClahe

def run_isolation_test():
    raw_path = Path(r"C:\MajorProject - Datasets\APTOS-2019-Kaggle\train_images\train_images\1b8ad0afe9fb.png")
    if not raw_path.exists():
        print("Raw image missing.")
        return
        
    raw_img = np.array(Image.open(raw_path).convert("RGB"))
    proc_img, _ = run_offline_pipeline(raw_img)
    
    # Isolate the two suspects: 
    # Force custom CLAHE to its absolute maximum clip limit
    test_clahe = A.Compose([RandomClahe(clip_limit_range=(4.0, 4.0), p=1.0)])
    
    # Force GaussNoise (using the legacy var_limit matching older albumentations)
    test_noise = A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)])
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Top row: Only CLAHE
        clahe_out = test_clahe(image=proc_img)["image"]
        axes[0, i].imshow(clahe_out)
        axes[0, i].set_title("Only Custom CLAHE")
        axes[0, i].axis('off')
        
        # Bottom row: Only GaussNoise
        noise_out = test_noise(image=proc_img)["image"]
        axes[1, i].imshow(noise_out)
        axes[1, i].set_title("Only GaussNoise")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = "visual_audit_isolated.jpg"
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}. Open it and see which row looks 'deep-fried'.")

if __name__ == "__main__":
    run_isolation_test()