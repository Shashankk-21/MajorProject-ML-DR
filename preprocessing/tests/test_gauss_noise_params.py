import albumentations as A
import warnings

def verify_gauss_noise():
    print("=" * 60)
    print("GAUSS NOISE PARAMETER VERIFICATION")
    print("=" * 60)
    
    print(f"Installed Albumentations version: {A.__version__}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Instantiate exactly as it currently exists in augmentation.py
        noise_transform = A.GaussNoise(std_range=(0.01, 0.03), p=1.0)
        
    print("\nAttempted to set: std_range=(0.01, 0.03)")
    print(f"Actual saved std_range inside object: {noise_transform.std_range}")
    assert noise_transform.std_range == (0.01, 0.03)
    print("\nWarnings caught: NONE.")

if __name__ == "__main__":
    verify_gauss_noise()