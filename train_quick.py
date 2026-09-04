"""
Quick Training Script (CPU-optimized)
=======================================
Trains models with reduced epochs for CPU machines.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.utils.helpers import set_seed, get_device_info
from src.training.trainer import Trainer

set_seed(42)
info = get_device_info()
print(f"TensorFlow {info['tf_version']} | GPUs: {info['gpus']}")

# CPU-optimized settings
BATCH_SIZE = 16
EPOCHS_P1 = 5    # Feature extraction
EPOCHS_P2 = 5    # Fine-tuning

model_name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"

print(f"\n Training {model_name.upper()} (CPU mode: {EPOCHS_P1}+{EPOCHS_P2} epochs)")

trainer = Trainer(
    model_name=model_name,
    batch_size=BATCH_SIZE,
    epochs_phase1=EPOCHS_P1,
    epochs_phase2=EPOCHS_P2,
)

results = trainer.train()
trainer.save_results(results)
print("\n✓ Training complete!")
