"""
Dataset Download & Setup Script
=================================
Downloads a pancreatic cancer CT scan dataset from Kaggle
and organizes it into train/val/test splits.
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def download_dataset():
    """Download pancreatic CT dataset from Kaggle using kagglehub."""
    import kagglehub

    print("=" * 60)
    print("  Downloading Pancreatic CT Dataset from Kaggle...")
    print("=" * 60)

    # Try multiple dataset sources
    datasets_to_try = [
        "tahsinmostafiz/pancreas-ct-dataset",
        "salihayesilyurt/pancreatic-ct-images",
        "andrewmvd/medical-mnist",
    ]

    download_path = None
    dataset_used = None

    for dataset_slug in datasets_to_try:
        try:
            print(f"\n  Trying: {dataset_slug}...")
            download_path = kagglehub.dataset_download(dataset_slug)
            dataset_used = dataset_slug
            print(f"  ✓ Downloaded to: {download_path}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    if download_path is None:
        print("\n  ⚠ Could not download from Kaggle.")
        print("  Generating synthetic demo dataset instead...")
        generate_synthetic_dataset()
        return

    # Explore what was downloaded
    print(f"\n  Exploring downloaded data...")
    downloaded = Path(download_path)
    for item in sorted(downloaded.rglob("*"))[:30]:
        rel = item.relative_to(downloaded)
        prefix = "📁" if item.is_dir() else "📄"
        print(f"    {prefix} {rel}")

    # Organize into our project structure
    organize_dataset(downloaded, dataset_used)


def organize_dataset(source_path: Path, dataset_slug: str):
    """
    Organize downloaded dataset into train/val/test splits.
    Handles various folder structures from Kaggle datasets.
    """
    print(f"\n  Organizing dataset into project structure...")

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    all_images = [
        f for f in source_path.rglob("*")
        if f.suffix.lower() in image_extensions
    ]
    print(f"  Found {len(all_images)} images total")

    if len(all_images) == 0:
        print("  ⚠ No images found. Generating synthetic dataset...")
        generate_synthetic_dataset()
        return

    # Try to detect class structure
    # Look for folders named like: normal, tumor, cancer, benign, malignant, etc.
    normal_keywords = {"normal", "healthy", "benign", "negative", "0", "non"}
    cancer_keywords = {"cancer", "tumor", "malignant", "positive", "1", "cancerous", "abnormal"}

    normal_images = []
    cancer_images = []
    unclassified = []

    for img in all_images:
        path_lower = str(img).lower()
        parent_name = img.parent.name.lower()

        if any(kw in parent_name for kw in cancer_keywords):
            cancer_images.append(img)
        elif any(kw in parent_name for kw in normal_keywords):
            normal_images.append(img)
        else:
            unclassified.append(img)

    print(f"  Classified: Normal={len(normal_images)}, Cancerous={len(cancer_images)}, Unclassified={len(unclassified)}")

    # If no clear class structure, split unclassified 50/50 for demo
    if len(normal_images) == 0 and len(cancer_images) == 0 and len(unclassified) > 0:
        print("  No class folders detected. Splitting images 50/50 for demonstration...")
        random.seed(config.RANDOM_SEED)
        random.shuffle(unclassified)
        mid = len(unclassified) // 2
        normal_images = unclassified[:mid]
        cancer_images = unclassified[mid:]
    elif len(normal_images) == 0 and len(cancer_images) > 0:
        # Only cancer images found, use unclassified as normal
        normal_images = unclassified
    elif len(cancer_images) == 0 and len(normal_images) > 0:
        # Only normal images found, use unclassified as cancer
        cancer_images = unclassified

    if len(normal_images) == 0 or len(cancer_images) == 0:
        print("  ⚠ Could not create balanced classes. Generating synthetic dataset...")
        generate_synthetic_dataset()
        return

    # Limit to reasonable size for CPU training
    max_per_class = 500
    if len(normal_images) > max_per_class:
        random.seed(config.RANDOM_SEED)
        normal_images = random.sample(normal_images, max_per_class)
    if len(cancer_images) > max_per_class:
        random.seed(config.RANDOM_SEED)
        cancer_images = random.sample(cancer_images, max_per_class)

    print(f"  Using: Normal={len(normal_images)}, Cancerous={len(cancer_images)}")

    # Split into train/val/test
    _split_and_copy(normal_images, "Normal")
    _split_and_copy(cancer_images, "Cancerous")

    print(f"\n  ✓ Dataset organized successfully!")
    _print_dataset_summary()


def _split_and_copy(images: list, class_name: str):
    """Split images into train/val/test and copy to project directories."""
    random.seed(config.RANDOM_SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        dest_dir = config.PROCESSED_DATA_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(split_images):
            ext = img_path.suffix
            dest = dest_dir / f"{class_name.lower()}_{split_name}_{i:04d}{ext}"
            shutil.copy2(str(img_path), str(dest))

        print(f"    {split_name}/{class_name}: {len(split_images)} images")


def generate_synthetic_dataset():
    """
    Generate a synthetic demo dataset for testing the pipeline.
    Creates realistic-looking grayscale CT-like images with
    distinguishable patterns for Normal vs Cancerous.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    print("\n" + "=" * 60)
    print("  Generating Synthetic Demo Dataset")
    print("  (For testing the pipeline — replace with real data later)")
    print("=" * 60)

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    n_per_class = 200  # 200 Normal + 200 Cancerous
    img_size = 224

    def create_normal_image():
        """Create a synthetic 'normal' CT-like image."""
        # Dark background with smooth tissue-like texture
        base = np.random.normal(80, 15, (img_size, img_size)).clip(0, 255)

        # Add smooth organ-like region in center
        y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
        r = np.sqrt(x*x + y*y)
        organ_mask = (r < img_size * 0.3).astype(float)

        # Smooth the mask
        from scipy.ndimage import gaussian_filter
        organ_mask = gaussian_filter(organ_mask, sigma=10)

        # Slightly brighter organ region
        organ_intensity = np.random.normal(120, 10)
        base = base * (1 - organ_mask) + organ_intensity * organ_mask

        # Add subtle noise
        noise = np.random.normal(0, 5, base.shape)
        base = (base + noise).clip(0, 255).astype(np.uint8)

        # Convert to RGB (grayscale-like)
        img = np.stack([base, base, base], axis=-1)
        return img

    def create_cancerous_image():
        """Create a synthetic 'cancerous' CT-like image with visible lesion."""
        # Start with normal-looking base
        base = np.random.normal(80, 15, (img_size, img_size)).clip(0, 255)

        y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
        r = np.sqrt(x*x + y*y)
        organ_mask = (r < img_size * 0.3).astype(float)

        from scipy.ndimage import gaussian_filter
        organ_mask = gaussian_filter(organ_mask, sigma=10)
        organ_intensity = np.random.normal(120, 10)
        base = base * (1 - organ_mask) + organ_intensity * organ_mask

        # Add a bright irregular "tumor" region
        tumor_x = img_size // 2 + np.random.randint(-30, 30)
        tumor_y = img_size // 2 + np.random.randint(-30, 30)
        tumor_r = np.random.randint(15, 35)

        ty, tx = np.ogrid[:img_size, :img_size]
        tumor_dist = np.sqrt((tx - tumor_x)**2 + (ty - tumor_y)**2)

        # Irregular tumor shape
        angles = np.arctan2(ty - tumor_y, tx - tumor_x)
        irregularity = tumor_r + 5 * np.sin(3 * angles) + 3 * np.cos(5 * angles)
        tumor_mask = (tumor_dist < irregularity).astype(float)
        tumor_mask = gaussian_filter(tumor_mask, sigma=3)

        # Tumor is brighter with heterogeneous texture
        tumor_intensity = np.random.normal(180, 20, base.shape)
        base = base * (1 - tumor_mask) + tumor_intensity * tumor_mask

        # Add noise
        noise = np.random.normal(0, 8, base.shape)
        base = (base + noise).clip(0, 255).astype(np.uint8)

        img = np.stack([base, base, base], axis=-1)
        return img

    # Check if scipy is available
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("  Installing scipy for image generation...")
        os.system("pip install scipy -q")
        from scipy.ndimage import gaussian_filter

    classes = {
        "Normal": create_normal_image,
        "Cancerous": create_cancerous_image,
    }

    for class_name, generator_fn in classes.items():
        all_images = []
        print(f"\n  Generating {n_per_class} {class_name} images...")

        for i in range(n_per_class):
            img_array = generator_fn()
            all_images.append(img_array)

        # Split
        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)

        splits = {
            "train": all_images[:n_train],
            "val": all_images[n_train:n_train + n_val],
            "test": all_images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dest_dir = config.PROCESSED_DATA_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for j, img_arr in enumerate(split_imgs):
                pil_img = Image.fromarray(img_arr)
                pil_img.save(str(dest_dir / f"{class_name.lower()}_{j:04d}.png"))

            print(f"    {split_name}/{class_name}: {len(split_imgs)} images")

    print(f"\n  ✓ Synthetic dataset generated!")
    _print_dataset_summary()


def _print_dataset_summary():
    """Print summary of the organized dataset."""
    print(f"\n  {'─' * 50}")
    print(f"  Dataset Summary:")
    print(f"  {'─' * 50}")

    total = 0
    for split in ["train", "val", "test"]:
        split_dir = config.PROCESSED_DATA_DIR / split
        if split_dir.exists():
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*")))
                    total += count
                    print(f"    {split:5s} / {class_dir.name:10s} : {count:4d} images")

    print(f"  {'─' * 50}")
    print(f"    Total: {total} images")
    print(f"\n  Location: {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    config.setup_directories()
    download_dataset()
