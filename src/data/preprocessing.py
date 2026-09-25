"""
Image Preprocessing Pipeline
=============================
Handles image resizing, normalization, noise reduction,
format conversion, and train/val/test splitting.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


def apply_pancreatic_window(
    image: np.ndarray,
    window_width: int = 350,
    window_level: int = 45,
) -> np.ndarray:
    """
    Apply pancreatic CT soft-tissue Hounsfield Unit (HU) windowing.
    Standard pancreas window: Window Level (L) = 45 HU, Window Width (W) = 350 HU.
    Window bounds: [L - W/2, L + W/2] -> [-130, 220] HU.

    Parameters
    ----------
    image : np.ndarray
        Input CT image as floating point or uint8 array.
    window_width : int
        Window width (W), defaults to 350.
    window_level : int
        Window center/level (L), defaults to 45.

    Returns
    -------
    np.ndarray
        Windowed image normalized to [0.0, 1.0] range.
    """
    img = image.astype(np.float32)
    win_min = window_level - (window_width / 2.0)
    win_max = window_level + (window_width / 2.0)

    # Clip to window bounds
    img_windowed = np.clip(img, win_min, win_max)
    # Rescale to [0, 1]
    if win_max > win_min:
        img_windowed = (img_windowed - win_min) / (win_max - win_min)
    else:
        img_windowed = np.zeros_like(img_windowed)
    return img_windowed


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    contrast in subtle pancreatic lesions.

    Parameters
    ----------
    image : np.ndarray
        Input image (float [0, 1] or uint8 [0, 255]).
    clip_limit : float
        Threshold for contrast limiting. Defaults to 2.0.
    tile_grid_size : tuple of int
        Size of grid for histogram equalization. Defaults to (8, 8).

    Returns
    -------
    np.ndarray
        Contrast-enhanced image in the same value range and format as input.
    """
    is_float = np.issubdtype(image.dtype, np.floating) or image.max() <= 1.0
    if is_float:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img_uint8.shape) == 2:
        enhanced = clahe.apply(img_uint8)
    elif img_uint8.shape[2] == 1:
        enhanced = clahe.apply(img_uint8[:, :, 0])[:, :, np.newaxis]
    else:
        # Convert to LAB color space, apply CLAHE to L-channel
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    if is_float:
        return enhanced.astype(np.float32) / 255.0
    return enhanced


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = None,
    normalize: bool = True,
    denoise: bool = False,
    window: bool = False,
    clahe: bool = False,
    window_width: int = 350,
    window_level: int = 45,
    as_array: bool = True,
) -> np.ndarray:
    """
    Preprocess a single medical image.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    target_size : tuple of int, optional
        (height, width) to resize the image. Defaults to config.DEFAULT_IMG_SIZE.
    normalize : bool
        If True, scale pixel values to [0, 1].
    denoise : bool
        If True, apply Gaussian blur for noise reduction.
    window : bool
        If True, apply pancreatic CT HU windowing.
    clahe : bool
        If True, apply CLAHE adaptive contrast enhancement.
    window_width : int
        Window width for CT windowing (defaults to 350).
    window_level : int
        Window level for CT windowing (defaults to 45).
    as_array : bool
        If True, return a NumPy array. Otherwise, return a PIL Image.

    Returns
    -------
    np.ndarray or PIL.Image
        Preprocessed image.
    """
    if target_size is None:
        target_size = config.DEFAULT_IMG_SIZE

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional CT Windowing
    if window:
        img = (apply_pancreatic_window(img, window_width, window_level) * 255.0).astype(np.uint8)

    # Optional CLAHE contrast enhancement
    if clahe:
        img = apply_clahe(img)

    # Noise reduction
    if denoise:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    if as_array:
        img = img.astype(np.float32)
        if normalize and not window:
            img = img / 255.0
        elif window:
            img = img / 255.0
        return img
    else:
        return Image.fromarray(img if not np.issubdtype(img.dtype, np.floating) else (img * 255).astype(np.uint8))


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = None,
    denoise: bool = False,
    output_format: str = "png",
) -> int:
    """
    Preprocess all images in a directory and save to output directory.

    Parameters
    ----------
    input_dir : str
        Source directory containing images.
    output_dir : str
        Destination directory for preprocessed images.
    target_size : tuple of int, optional
        (height, width) for resizing.
    denoise : bool
        Apply noise reduction.
    output_format : str
        Output image format ('png', 'jpg').

    Returns
    -------
    int
        Number of images processed.
    """
    if target_size is None:
        target_size = config.DEFAULT_IMG_SIZE

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in valid_extensions
    ]

    count = 0
    for img_file in tqdm(image_files, desc=f"Preprocessing {input_path.name}"):
        try:
            img = preprocess_image(
                str(img_file),
                target_size=target_size,
                normalize=False,  # Save as uint8
                denoise=denoise,
                as_array=True,
            )
            # Convert back to uint8 for saving
            img_uint8 = img.astype(np.uint8)
            out_file = output_path / f"{img_file.stem}.{output_format}"
            pil_img = Image.fromarray(img_uint8)
            pil_img.save(str(out_file))
            count += 1
        except Exception as e:
            print(f"  ⚠ Skipping {img_file.name}: {e}")

    return count


def split_dataset(
    source_dir: str,
    output_base_dir: str,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
) -> dict:
    """
    Split a folder-structured dataset into train/val/test sets.

    Expects source_dir to contain class subdirectories:
        source_dir/
        ├── Normal/
        └── Cancerous/

    Parameters
    ----------
    source_dir : str
        Root directory with class subfolders.
    output_base_dir : str
        Where to create train/, val/, test/ splits.
    train_ratio, val_ratio, test_ratio : float
        Split proportions. Defaults from config.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Split statistics per class.
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_SPLIT
    if val_ratio is None:
        val_ratio = config.VAL_SPLIT
    if test_ratio is None:
        test_ratio = config.TEST_SPLIT
    if seed is None:
        seed = config.RANDOM_SEED

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_base_dir)

    stats = {}

    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        class_stats = {}
        for split_name, split_images in splits.items():
            dest_dir = output_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in tqdm(
                split_images,
                desc=f"  {split_name}/{class_name}",
                leave=False,
            ):
                shutil.copy2(str(img_file), str(dest_dir / img_file.name))

            class_stats[split_name] = len(split_images)

        stats[class_name] = class_stats
        print(f"  ✓ {class_name}: train={class_stats['train']}, "
              f"val={class_stats['val']}, test={class_stats['test']}")

    return stats


def validate_image(image_path: str) -> bool:
    """Check if a file is a valid, loadable image."""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def get_image_stats(directory: str) -> dict:
    """
    Compute statistics about images in a directory.

    Returns
    -------
    dict
        Keys: 'count', 'sizes', 'mean_size', 'formats'.
    """
    dir_path = Path(directory)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = [f for f in dir_path.rglob("*") if f.suffix.lower() in valid_extensions]

    sizes = []
    formats = {}
    for img_file in images:
        try:
            with Image.open(img_file) as img:
                sizes.append(img.size)
                fmt = img.format or "Unknown"
                formats[fmt] = formats.get(fmt, 0) + 1
        except Exception:
            pass

    if sizes:
        widths, heights = zip(*sizes)
        mean_size = (int(np.mean(widths)), int(np.mean(heights)))
    else:
        mean_size = (0, 0)

    return {
        "count": len(images),
        "sizes": sizes,
        "mean_size": mean_size,
        "formats": formats,
    }
