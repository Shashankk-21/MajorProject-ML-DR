import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from preprocessing.image_processing import run_offline_pipeline
from preprocessing.augmentation import get_train_transforms

DATA_ROOT = Path(r"C:\MajorProject - Datasets")
raw_path = DATA_ROOT / "APTOS-2019-Kaggle/train_images/train_images/1b8ad0afe9fb.png"
raw_img = np.array(Image.open(raw_path).convert("RGB"))
proc_img, _ = run_offline_pipeline(raw_img)

# Pull the real transform objects straight out of your own pipeline --
# no need to reconstruct or guess constructor args.
composed = get_train_transforms()
by_name = {type(t).__name__: t for t in composed.transforms}

candidates = ["GaussNoise", "RandomClahe"]
fig, axes = plt.subplots(len(candidates), 6, figsize=(18, 3 * len(candidates)))

for row, name in enumerate(candidates):
    t = by_name[name]
    t.p = 1.0  # force it on every draw to isolate this transform alone
    for col in range(6):
        out = t(image=proc_img)["image"]
        axes[row, col].imshow(out)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("isolation_test.jpg", dpi=150)
print("Saved isolation_test.jpg -- compare each row to the noisy cells from the 50-draw grid.")