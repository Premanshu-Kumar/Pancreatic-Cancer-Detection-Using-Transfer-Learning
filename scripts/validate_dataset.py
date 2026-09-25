"""
Dataset Sanity, Duplicate & Corruption Checking Tool
===================================================
Scans medical CT imaging directories to detect corrupted images,
truncated byte streams, blank/low-entropy slices, exact duplicates,
and reports class balance with sanity summaries.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash to detect exact duplicate files."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_single_image(file_path: Path) -> Dict[str, Any]:
    """
    Validate a single CT slice for corruption, dimensions, and entropy.
    """
    result = {
        "path": str(file_path),
        "valid": False,
        "error": None,
        "shape": None,
        "mean_intensity": None,
        "std_intensity": None,
        "is_blank": False,
    }

    try:
        # Check integrity via Pillow
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        result["error"] = f"Pillow verification failed: {e}"
        return result

    try:
        # Check readable array and channels via OpenCV
        arr = cv2.imread(str(file_path))
        if arr is None:
            result["error"] = "OpenCV decode returned None"
            return result

        result["shape"] = list(arr.shape)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        result["mean_intensity"] = round(mean_val, 2)
        result["std_intensity"] = round(std_val, 2)

        # Flag blank slices (zero variance or near pure black/white)
        if std_val < 1.0 or mean_val < 0.5:
            result["is_blank"] = True

        result["valid"] = True
    except Exception as e:
        result["error"] = f"Decode processing error: {e}"

    return result


def validate_dataset(
    data_dir: Path,
    output_report: Path = None,
    create_preview: bool = True,
) -> Dict[str, Any]:
    """
    Scan data_dir, validate all images, check duplicates and class balance.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]

    stats = {
        "scanned_directory": str(data_dir),
        "total_files": len(image_files),
        "valid_files": 0,
        "corrupted_files": 0,
        "blank_files": 0,
        "duplicate_count": 0,
        "class_distribution": {},
        "issues": [],
    }

    seen_hashes = {}
    class_images: Dict[str, List[Path]] = {}

    for img_path in image_files:
        # Determine class from folder
        parent_name = img_path.parent.name
        class_name = parent_name if parent_name in ["Normal", "Cancerous"] else "Other"
        stats["class_distribution"][class_name] = stats["class_distribution"].get(class_name, 0) + 1

        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(img_path)

        # Check duplicate
        try:
            fhash = compute_file_hash(img_path)
            if fhash in seen_hashes:
                stats["duplicate_count"] += 1
                stats["issues"].append({
                    "type": "duplicate",
                    "file": str(img_path),
                    "original": seen_hashes[fhash],
                })
            else:
                seen_hashes[fhash] = str(img_path)
        except Exception as e:
            pass

        # Check image validity
        check = check_single_image(img_path)
        if not check["valid"]:
            stats["corrupted_files"] += 1
            stats["issues"].append({
                "type": "corrupted",
                "file": str(img_path),
                "error": check["error"],
            })
        elif check["is_blank"]:
            stats["blank_files"] += 1
            stats["issues"].append({
                "type": "blank_slice",
                "file": str(img_path),
                "mean": check["mean_intensity"],
                "std": check["std_intensity"],
            })
        else:
            stats["valid_files"] += 1

    # Optional preview generation
    if create_preview and config.RESULTS_DIR.exists():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            samples_to_show = []
            labels_to_show = []
            for cname in ["Normal", "Cancerous"]:
                imgs = class_images.get(cname, [])
                for p in imgs[:4]:
                    loaded = cv2.imread(str(p))
                    if loaded is not None:
                        samples_to_show.append(cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB))
                        labels_to_show.append(f"{cname}\n({p.name})")

            if samples_to_show:
                n = len(samples_to_show)
                cols = min(4, n)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                axes = np.array(axes).reshape(-1)
                for i in range(len(axes)):
                    if i < n:
                        axes[i].imshow(samples_to_show[i])
                        axes[i].set_title(labels_to_show[i], fontsize=9)
                    axes[i].axis("off")
                plt.tight_layout()
                preview_path = config.RESULTS_DIR / "dataset_validation_preview.png"
                plt.savefig(preview_path, dpi=120)
                plt.close()
                stats["preview_path"] = str(preview_path)
        except Exception as e:
            stats["preview_error"] = str(e)

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Sanity & Corruption Checker")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    parser.add_argument("--output", type=str, default=str(config.RESULTS_DIR / "dataset_validation_report.json"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    results = validate_dataset(
        data_dir=Path(args.data_dir),
        output_report=Path(args.output),
        create_preview=not args.no_preview,
    )
    print("=" * 60)
    print("  Dataset Validation Summary:")
    print(f"  Total Files Scanned : {results['total_files']}")
    print(f"  Valid Files         : {results['valid_files']}")
    print(f"  Corrupted Files     : {results['corrupted_files']}")
    print(f"  Blank Slices        : {results['blank_files']}")
    print(f"  Duplicate Files     : {results['duplicate_count']}")
    print(f"  Class Distribution  : {results['class_distribution']}")
    print("=" * 60)
