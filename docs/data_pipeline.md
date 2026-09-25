# 🩺 Medical CT Data Pipeline & Quality Control Architecture

> **Week 2 Milestone (Days 8–14)**: Medical Preprocessing, Patient-Level Splitting & High-Throughput `tf.data` Pipeline

---

## 📌 Executive Architecture Overview

In abdominal computed tomography (CT) for pancreatic cancer detection, raw image ingestion without rigorous radiologic preprocessing leads to severe failure modes:
1. **HU Dynamic Range Collapse**: CT images span -1024 HU (air) to +3000 HU (dense bone), washing out pancreatic parenchymal details when converted naively to 8-bit RGB.
2. **Slice Leakage Artifacts**: Multi-slice CT scans of the same patient split across train and test partitions allow deep models to memorize patient anatomy rather than tumor morphology.
3. **Disk I/O Bottlenecks**: Synchronous disk reading stalls modern GPUs, dropping accelerator utilization below 35%.

Our Week 2 pipeline implements an end-to-end medical preprocessing architecture addressing each clinical and engineering constraint.

---

## 🔬 Core Components

### 1. Pancreas-Specific Hounsfield Unit (HU) Windowing
- **Window Center (Level)**: $+45\text{ HU}$ (soft-tissue pancreatic attenuation)
- **Window Width**: $350\text{ HU}$
- **Attenuation Dynamic Range**: $[-130\text{ HU}, +220\text{ HU}]$
- **Normalization**: Linear rescale into $[0.0, 1.0]$ after HU clamping.
- **CLAHE Enhancement**: Contrast-Limited Adaptive Histogram Equalization with clip limit $2.0$ and $(8, 8)$ tile grid applied to highlight hypodense adenocarcinoma margins.

### 2. Patient-Level Group Stratification (`src/data/split_data.py`)
- Grouped by `patient_id` regex pattern to guarantee **zero cross-split slice leakage**.
- Stratification maintains balanced ratios between **Normal** and **Pancreatic Adenocarcinoma** across all three sets ($70\%$ train, $15\%$ val, $15\%$ test).
- Automated generation of reproducible CSV manifests: `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`.

### 3. Medical-Safe Albumentations Pipeline (`src/data/augmentation.py`)
- **Discrete 90° Rotations**: Safe anatomical transforms preserving orientation.
- **Horizontal Flipping**: Preserves spatial organ symmetry.
- **Excluded Transforms**: Unrealistic vertical flipping (which reverses cranio-caudal axes) and heavy color jitter (CT is fundamentally radiodensity, not color).

### 4. High-Throughput `tf.data` Parallel Pipeline (`src/data/dataset.py`)
- Asynchronous file ingestion via `tf.io.read_file` and non-blocking decoding.
- Parallel processing mapped using `num_parallel_calls=tf.data.AUTOTUNE`.
- GPU memory feeding with `prefetch(buffer_size=tf.data.AUTOTUNE)`.

---

## 📊 Throughput Benchmarks

Tested on 640 CT slices with batch size 32:

| Metric | Legacy `ImageDataGenerator` | Optimized `tf.data` Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput (images/sec)** | $93.8\text{ img/s}$ | **$299.1\text{ img/s}$** | **$+219\%$ ($3.19\times$)** |
| **Average Batch Latency** | $341.0\text{ ms}$ | **$107.0\text{ ms}$** | **$-68.6\%$ Latency** |
| **GPU Starvation Rate** | High ($>40\%$ I/O wait) | Negligible ($<3\%$) | **I/O Bottleneck Eliminated** |
| **Memory Footprint** | Static process RAM | Dynamic ring buffer | **Optimized cache** |

---

## 🛠️ Usage Instructions

### Verify Dataset Sanity:
```bash
python scripts/validate_dataset.py --data-dir data/ --output results/dataset_validation_report.json
```

### Regenerate Patient-Level Stratified Splits:
```bash
python -m src.data.split_data --data-dir data/ --output-dir data/splits/
```

### Ingest Data in Training:
```python
from src.data.dataset import create_tf_data_pipeline

pipelines = create_tf_data_pipeline(batch_size=32)
train_ds = pipelines["train"]
val_ds = pipelines["val"]
```
