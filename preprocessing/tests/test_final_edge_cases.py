import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path

from preprocessing.dataset import build_manifest
from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import get_train_transforms, get_eval_transforms

DATA_ROOT = Path(r"C:\MajorProject - Datasets")

def test_crop_rates():
    print("=" * 60)
    print("1. CROP-RATE CHECK (PER-CLIENT)")
    print("=" * 60)
    
    # EyePACS is excluded because its raw `.zip` files were successfully 
    # deleted to save disk space after the full pipeline ran.
    clients = ["APTOS", "IDRiD", "Messidor-2"]
    samples_per_client = 50
    
    for client in clients:
        try:
            # Build the source manifest to get the raw_image_locator paths
            source_df = build_manifest(client, data_root=DATA_ROOT)
            source_df = source_df.dropna(subset=["raw_image_locator"])
            
            sampled = source_df.sample(n=min(samples_per_client, len(source_df)), random_state=42)
            
            cropped_count = 0
            total_processed = 0
            
            for _, row in sampled.iterrows():
                raw_path = Path(row["raw_image_locator"])
                if not raw_path.exists():
                    continue
                    
                raw_img = np.array(Image.open(raw_path).convert("RGB"))
                _, meta = run_offline_pipeline(raw_img)
                
                if meta.get("was_cropped", False):
                    cropped_count += 1
                total_processed += 1
                
            if total_processed > 0:
                crop_rate = (cropped_count / total_processed) * 100
                print(f"{client:12s}: {cropped_count:2d}/{total_processed} images cropped ({crop_rate:.1f}%)")
            else:
                print(f"{client:12s}: No valid raw images found on disk for sampling.")
                
        except Exception as e:
            print(f"Error checking {client}: {e}")


def test_multi_image_stochasticity():
    print("\n" + "=" * 60)
    print("2. MULTI-IMAGE STOCHASTICITY & DETERMINISM")
    print("=" * 60)
    
    manifest_path = Path("data/splits/APTOS_manifest.csv")
    if not manifest_path.exists():
        print("SKIPPED: APTOS_manifest.csv not found.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Grab 5 random cached (offline processed) images
    sample_paths = df["cached_image_path"].dropna().sample(5, random_state=123).tolist()
    
    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()
    
    all_passed = True
    
    for i, path in enumerate(sample_paths, 1):
        if not Path(path).exists():
            print(f"  Image {i} missing from cache. Skipping.")
            continue
            
        img = np.array(Image.open(path).convert("RGB"))
        
        # Train Stochasticity: Run twice, expect a difference
        t1 = train_tf(image=img)["image"]
        t2 = train_tf(image=img)["image"]
        train_diff = torch.abs(t1 - t2).mean().item()
        
        # Eval Determinism: Run twice, expect absolute identity
        e1 = eval_tf(image=img)["image"]
        e2 = eval_tf(image=img)["image"]
        eval_diff = torch.abs(e1 - e2).mean().item()
        
        is_stoch = train_diff > 0.0
        is_det = eval_diff == 0.0
        
        status = "PASS" if (is_stoch and is_det) else "FAIL"
        print(f"  Image {i:d}: {status} (Train Diff: {train_diff:.4f} | Eval Diff: {eval_diff:.4f})")
        
        if not (is_stoch and is_det):
            all_passed = False
            
    if all_passed:
        print("\nRESULT: PASS. Stochasticity and determinism hold consistently across multiple images.")
    else:
        print("\nRESULT: FAIL on one or more images.")


def test_extreme_brightness():
    print("\n" + "=" * 60)
    print("3. EXTREME BRIGHTNESS & DEGENERATE AUGMENTATION")
    print("=" * 60)
    
    # Create synthetic arrays representing extreme photo conditions
    dark_img = np.full((224, 224, 3), 10, dtype=np.uint8)  # Pure dark gray/black
    bright_img = np.full((224, 224, 3), 245, dtype=np.uint8) # Blown-out white
    
    train_tf = get_train_transforms()
    
    try:
        # Pass them through the aggressive train augmentations (Gaussian noise, brightness jitter, etc.)
        out_dark = train_tf(image=dark_img)["image"]
        dark_finite = torch.isfinite(out_dark).all().item()
        
        out_bright = train_tf(image=bright_img)["image"]
        bright_finite = torch.isfinite(out_bright).all().item()
        
        if dark_finite and bright_finite:
            print("PASS: Train augmentation handled extreme dark and bright inputs without NaN/Inf failures.")
            print(f"  Dark Image Output Range:   [{out_dark.min():.2f}, {out_dark.max():.2f}]")
            print(f"  Bright Image Output Range: [{out_bright.min():.2f}, {out_bright.max():.2f}]")
        else:
            print("FAIL: Train augmentation produced non-finite NaN/Inf values on extreme inputs.")
            
    except Exception as e:
        print(f"FAIL: Exception thrown during extreme brightness test: {e}")


if __name__ == "__main__":
    test_crop_rates()
    test_multi_image_stochasticity()
    test_extreme_brightness()