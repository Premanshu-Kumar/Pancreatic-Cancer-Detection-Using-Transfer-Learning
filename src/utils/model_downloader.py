"""
Model Weight Downloader & Hugging Face Hub / Release Manager
============================================================
Utility module to verify and automatically download trained deep learning
model checkpoints (.keras, .h5) from Hugging Face Hub or GitHub Releases
when running on clean environments, CI/CD pipelines, or cloud deployments.
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Dict

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from config import MODELS_DIR
except ImportError:
    MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


# Default Hugging Face repository or mirror endpoints
DEFAULT_HF_REPO = os.getenv("HF_MODEL_REPO", "premanshu/pancreatic-cancer-resnet50")
HF_BASE_URL = f"https://huggingface.co/{DEFAULT_HF_REPO}/resolve/main"

# Registry of known model checkpoints and their fallback remote endpoints
KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    "resnet50_best.keras": {
        "description": "Fine-tuned ResNet50 for Pancreatic Cancer Detection (Phase 2 Best)",
        "url": f"{HF_BASE_URL}/resnet50_best.keras",
        "size_mb": 229,
    },
    "resnet50_phase1_best.keras": {
        "description": "Feature-extraction ResNet50 checkpoint (Phase 1 Best)",
        "url": f"{HF_BASE_URL}/resnet50_phase1_best.keras",
        "size_mb": 109,
    },
    "resnet50_phase2_best.keras": {
        "description": "Fine-tuned ResNet50 checkpoint (Phase 2 Best)",
        "url": f"{HF_BASE_URL}/resnet50_phase2_best.keras",
        "size_mb": 229,
    },
}


class DownloadProgressBar:
    """Progress hook for urllib download."""

    def __init__(self, filename: str):
        self.filename = filename
        self.pbar = None
        try:
            from tqdm import tqdm
            self.tqdm_class = tqdm
        except ImportError:
            self.tqdm_class = None

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.tqdm_class:
            if self.pbar is None:
                self.pbar = self.tqdm_class(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {self.filename}",
                )
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()
        else:
            downloaded_mb = (block_num * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            percent = (downloaded_mb / total_mb * 100) if total_mb > 0 else 0
            sys.stdout.write(f"\rDownloading {self.filename}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
            sys.stdout.flush()
            if downloaded_mb >= total_mb:
                sys.stdout.write("\n")


def list_available_models() -> Dict[str, Dict[str, str]]:
    """Return dictionary of known remote model checkpoints."""
    return KNOWN_MODELS


def download_model(
    filename: str,
    dest_dir: Optional[Path] = None,
    custom_url: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download a model checkpoint to the target directory.

    Parameters
    ----------
    filename : str
        Model filename (e.g., 'resnet50_best.keras')
    dest_dir : Path, optional
        Destination directory. Defaults to config.MODELS_DIR.
    custom_url : str, optional
        Custom URL if downloading from an alternate mirror.
    force : bool, optional
        If True, overwrite existing local file.

    Returns
    -------
    Path
        Path to the local downloaded model file.
    """
    target_dir = Path(dest_dir) if dest_dir else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.exists() and not force:
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model '{filename}' already exists locally ({size_mb:.1f} MB) at: {target_path}")
        return target_path

    # Determine URL
    url = custom_url
    if not url:
        if filename in KNOWN_MODELS:
            url = KNOWN_MODELS[filename]["url"]
        else:
            url = f"{HF_BASE_URL}/{filename}"

    print(f"📥 Downloading '{filename}' from {url}...")
    temp_path = target_dir / f"{filename}.downloading"

    try:
        progress_hook = DownloadProgressBar(filename)
        urllib.request.urlretrieve(url, temp_path, reporthook=progress_hook)
        shutil.move(str(temp_path), str(target_path))
        print(f"✓ Successfully downloaded and saved to: {target_path}")
        return target_path
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"Failed to download model checkpoint '{filename}' from '{url}'. Error: {e}"
        ) from e


def ensure_model_exists(model_path_or_name: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Ensure that a specified model exists locally, or attempt download if missing.

    Parameters
    ----------
    model_path_or_name : str
        Either a relative/absolute filepath or a filename in MODELS_DIR.
    dest_dir : Path, optional
        Destination directory to search/download to.

    Returns
    -------
    Path
        Verified Path to the model file.
    """
    target_dir = Path(dest_dir) if dest_dir else MODELS_DIR
    path_obj = Path(model_path_or_name)

    # Check if absolute/direct path exists
    if path_obj.exists():
        return path_obj

    # Check inside target directory
    candidate = target_dir / path_obj.name
    if candidate.exists():
        return candidate

    # If missing, attempt automatic download
    print(f"⚠️ Model not found at '{model_path_or_name}'. Attempting auto-download...")
    return download_model(path_obj.name, dest_dir=target_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Pancreatic Cancer Model Checkpoints from Hugging Face / Remote")
    parser.add_argument("--model", type=str, default="resnet50_best.keras", help="Model filename to download")
    parser.add_argument("--all", action="store_true", help="Download all available model checkpoints")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--dest", type=str, default=None, help="Custom destination directory")

    args = parser.parse_args()
    dest = Path(args.dest) if args.dest else None

    if args.all:
        print(f"Downloading all known models to: {dest or MODELS_DIR}")
        for model_file in KNOWN_MODELS:
            download_model(model_file, dest_dir=dest, force=args.force)
    else:
        download_model(args.model, dest_dir=dest, force=args.force)
