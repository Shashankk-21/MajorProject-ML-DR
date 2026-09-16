import albumentations as A
import warnings

def verify_gauss_noise():
    print("=" * 60)
    print("GAUSS NOISE PARAMETER VERIFICATION")
    print("=" * 60)
    
    print(f"Installed Albumentations version: {A.__version__}")
    
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Instantiate exactly as it currently exists in augmentation.py
        noise_transform = A.GaussNoise(std_range=(0.01, 0.03), p=1.0)
        
    print("\nAttempted to set: std_range=(0.01, 0.03)")
    print(f"Actual saved var_limit inside object: {getattr(noise_transform, 'var_limit', 'Attribute Missing')}")
    
    if caught:
        print("\nWarnings caught:")
        for w in caught:
            print(f" - {w.message}")
    else:
        print("\nWarnings caught: NONE. (The library silently ignored the parameter)")

if __name__ == "__main__":
    verify_gauss_noise()