"""
Quick Training CLI Script
=========================
Trains cancer detection models with customizable epochs and batch sizes
optimized for local CPU and GPU environments.
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment optimization defaults
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils.helpers import set_seed, get_device_info
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Cancer Detection Model (Phase 1 Transfer Learning + Phase 2 Fine-Tuning)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=["resnet50", "vgg16", "inceptionv3"],
        help="Architecture backbone to train (default: resnet50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation (default: 16)",
    )
    parser.add_argument(
        "--epochs-p1",
        type=int,
        default=5,
        help="Phase 1 (Feature Extraction) epochs (default: 5)",
    )
    parser.add_argument(
        "--epochs-p2",
        type=int,
        default=5,
        help="Phase 2 (Fine-Tuning) epochs (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    info = get_device_info()
    print("=" * 60)
    print(f" Pancreatic Cancer Detection: Training Pipeline")
    print("=" * 60)
    print(f"TensorFlow {info.get('tf_version', 'N/A')} | GPUs: {info.get('gpus', 0)}")
    print(f"Model: {args.model.upper()} | Batch Size: {args.batch_size}")
    print(f"Epochs: Phase 1 = {args.epochs_p1}, Phase 2 = {args.epochs_p2}")
    print("=" * 60)

    trainer = Trainer(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_p1,
        epochs_phase2=args.epochs_p2,
    )

    results = trainer.train()
    trainer.save_results(results)
    print("\n✓ Training complete! Checkpoints and metrics saved to models/ and results/.")


if __name__ == "__main__":
    main()
