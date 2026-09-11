# 🔬 Pancreatic Cancer Detection: 30-Day Project Upgrade Plan & Git Commit Roadmap

> **Document Type**: Technical Project Modernization & Git Tracking Blueprint  
> **Target Project**: Pancreatic Cancer Detection Using Transfer Learning  
> **Storage Note**: Stored in artifacts directory outside of project files.

---

## 📌 Executive Summary

This roadmap provides an end-to-end 30-day plan to transform the project from a local experimental setup into a **production-ready, peer-review/portfolio-grade medical AI application**.

### Core Milestones:
1. **Week 1 (Days 1–7)**: Safe Path Migration (removing path spaces), Dynamic Environment Config (`.env`), and Package Standardization (`pyproject.toml`).
2. **Week 2 (Days 8–14)**: Medical CT Preprocessing (HU Windowing, CLAHE), Patient-Level Splitting (preventing data leakage), and high-throughput `tf.data`.
3. **Week 3 (Days 15–21)**: SOTA Vision Backbones (EfficientNetV2, ConvNeXt), Focal Loss, FP16 Mixed Precision, Grad-CAM++, and Clinical Diagnostic Metrics.
4. **Week 4 (Days 22–27)**: Asynchronous FastAPI Overhaul, PACS Dark Workstation UI with interactive opacity slider, and 1-click PDF Diagnostic Report generation.
5. **Week 5 (Days 28–30)**: Multi-stage Dockerization, GitHub Actions CI/CD Pipeline, and Cloud Deployment on Hugging Face Spaces.

---

## 🗓️ Phase 1: Week 1 (Days 1–7) — Foundation, Path Migration & Repository Architecture

### Day 1: Safe Workspace Directory Renaming
- **Key Objective**: Rename project directory from `d:\Project Dox` to `D:\Pancreatic-Cancer-Detection` to prevent path-splitting bugs in Docker, CUDA, Git, and virtual environments.
- **Actionable Tasks**:
  - Close all IDEs, terminals, and running processes.
  - Run PowerShell command to rename the root folder safely:
    ```powershell
    cd D:\
    Rename-Item -Path "D:\Project Dox" -NewName "Pancreatic-Cancer-Detection"
    cd D:\Pancreatic-Cancer-Detection
    ```
  - Verify Git remote URL and branch status.
- **Files Modified**: Project Root Directory
- **Git Commit Command**:
  ```bash
  git add .
  git commit -m "chore(repo): migrate repository path and eliminate whitespace in root directory"
  git push origin main
  ```

---

### Day 2: Virtual Environment Reconstruction & Dependency Pinning
- **Key Objective**: Purge broken virtual environment paths and rebuild a clean, reproducible Python environment.
- **Actionable Tasks**:
  - Remove old `venv` folder holding old hardcoded path strings.
  - Create new virtual environment: `python -m venv .venv`.
  - Install dependencies: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`.
- **Files Modified**: `requirements.txt`, `.gitignore`, `.venv/`
- **Git Commit Command**:
  ```bash
  git add requirements.txt .gitignore
  git commit -m "build(env): recreate clean virtual environment and pin core dependencies"
  git push origin main
  ```

---

### Day 3: Dynamic Environment-Based Path Configuration (`.env` & `python-dotenv`)
- **Key Objective**: Decouple project paths from absolute machine paths so code runs seamlessly across Windows, Linux, Colab, and Docker.
- **Actionable Tasks**:
  - Add `python-dotenv` to `requirements.txt`.
  - Create `.env.example` defining `DATA_DIR`, `MODELS_DIR`, and `RESULTS_DIR`.
  - Refactor `config.py` to read paths dynamically from environment variables with safe defaults.
  - Add `.env` to `.gitignore`.
- **Files Modified**: `config.py`, `.env.example`, `.gitignore`, `requirements.txt`
- **Git Commit Command**:
  ```bash
  git add config.py requirements.txt .env.example .gitignore
  git commit -m "refactor(config): introduce dynamic environment-based paths via python-dotenv"
  git push origin main
  ```

---

### Day 4: Git LFS Decoupling & Model Weights Strategy
- **Key Objective**: Ensure large `.keras` model weights (229MB) are tracked via Git LFS or external releases rather than bloating Git history.
- **Actionable Tasks**:
  - Verify `.gitattributes` tracks `*.keras`, `*.h5`, and `*.onnx` via Git LFS.
  - Create `src/utils/model_downloader.py` to auto-download checkpoints from Hugging Face Hub if missing locally.
- **Files Modified**: `.gitattributes`, `src/utils/model_downloader.py`
- **Git Commit Command**:
  ```bash
  git add .gitattributes src/utils/model_downloader.py
  git commit -m "chore(lfs): configure git-lfs tracking and model weight download helper"
  git push origin main
  ```

---

### Day 5: Project Structure Standardization & CLI Consolidation
- **Key Objective**: Move loose root scripts (`train_quick.py`, `evaluate_and_plot.py`) into an organized `scripts/` directory.
- **Actionable Tasks**:
  - Create `scripts/` directory and move `train_quick.py`, `evaluate_and_plot.py`, and `download_dataset.py`.
  - Standardize CLI arguments across scripts using `argparse`.
- **Files Modified**: `scripts/train_quick.py`, `scripts/evaluate_model.py`, `scripts/download_dataset.py`
- **Git Commit Command**:
  ```bash
  git add scripts/ train_quick.py evaluate_and_plot.py download_dataset.py
  git commit -m "refactor(structure): reorganize root scripts into scripts/ directory"
  git push origin main
  ```

---

### Day 6: Standard Python Packaging with `pyproject.toml`
- **Key Objective**: Eliminate `sys.path` manipulation hacks by making the repository an installable package via `pip install -e .`.
- **Actionable Tasks**:
  - Create `pyproject.toml` specifying package metadata and dependencies.
  - Install in editable mode: `pip install -e .`.
  - Remove manual `sys.path.insert` lines from `app/main.py` and test modules.
- **Files Modified**: `pyproject.toml`, `app/main.py`, `tests/`
- **Git Commit Command**:
  ```bash
  git add pyproject.toml app/main.py tests/
  git commit -m "build(packaging): add pyproject.toml for standard editable package installation"
  git push origin main
  ```

---

### Day 7: Automated Path Integrity & Module Import Verification
- **Key Objective**: Verify zero path-breakages and guarantee test suites run clean across platforms.
- **Actionable Tasks**:
  - Create `tests/test_paths.py` to verify all configured directories exist and are writable.
  - Run `pytest tests/` to confirm models, configs, and APIs pass all smoke tests.
- **Files Modified**: `tests/test_paths.py`
- **Git Commit Command**:
  ```bash
  git add tests/test_paths.py
  git commit -m "test(smoke): add cross-platform path integrity and module import tests"
  git push origin main
  ```

---

## 🗓️ Phase 2: Week 2 (Days 8–14) — Medical Data Pipeline, Windowing & Quality Control

### Day 8: CT Windowing & Soft-Tissue Hounsfield Unit (HU) Normalization
- **Key Objective**: Implement pancreas-specific CT windowing (W:350, L:45) and CLAHE contrast enhancement.
- **Actionable Tasks**:
  - Implement `apply_pancreatic_window()` in `src/data/preprocessing.py`.
  - Add CLAHE local adaptive histogram equalization to enhance subtle lesion borders.
- **Files Modified**: `src/data/preprocessing.py`, `tests/test_preprocessing.py`
- **Git Commit Command**:
  ```bash
  git add src/data/preprocessing.py tests/test_preprocessing.py
  git commit -m "feat(data): implement pancreatic CT HU windowing and contrast normalization"
  git push origin main
  ```

---

### Day 9: Patient-Level Group Stratification (Preventing Slice Leakage)
- **Key Objective**: Prevent multi-slice CT leakage where slices of the same patient appear in both train and test sets.
- **Actionable Tasks**:
  - Create `src/data/split_data.py` using `GroupKFold` grouped by patient ID.
  - Generate `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`.
- **Files Modified**: `src/data/split_data.py`, `data/splits/*.csv`
- **Git Commit Command**:
  ```bash
  git add src/data/split_data.py data/splits/
  git commit -m "feat(data): add patient-level stratified splitting to eliminate slice leakage"
  git push origin main
  ```

---

### Day 10: External Dataset Storage & Automated Checksums
- **Key Objective**: Support multi-gigabyte external drive storage (e.g. `E:\CT_Data`) without bloating Git.
- **Actionable Tasks**:
  - Update `scripts/download_dataset.py` to verify MD5 checksums of downloaded archives.
  - Allow dataset paths outside workspace root configured via `DATA_DIR`.
- **Files Modified**: `scripts/download_dataset.py`, `src/data/dataset.py`
- **Git Commit Command**:
  ```bash
  git add scripts/download_dataset.py src/data/dataset.py
  git commit -m "feat(data): support external dataset paths and automated MD5 integrity checks"
  git push origin main
  ```

---

### Day 11: Medical-Grade Data Augmentation Pipeline
- **Key Objective**: Upgrade data augmentation to medical-safe transforms using Albumentations.
- **Actionable Tasks**:
  - Create `src/data/augmentation.py` with affine shifts, 90-degree rotations, and subtle contrast shifts.
  - Exclude unrealistic vertical distortion and extreme color jitter.
- **Files Modified**: `src/data/augmentation.py`, `requirements.txt`
- **Git Commit Command**:
  ```bash
  git add src/data/augmentation.py requirements.txt
  git commit -m "feat(augment): implement medical-safe elastic and affine augmentations"
  git push origin main
  ```

---

### Day 12: High-Throughput `tf.data` Pipeline with Parallel Prefetching
- **Key Objective**: Eliminate disk I/O bottlenecks during model training.
- **Actionable Tasks**:
  - Migrate from Keras `DirectoryIterator` to `tf.data.Dataset` pipeline.
  - Incorporate `num_parallel_calls=AUTOTUNE` and `prefetch(AUTOTUNE)`.
- **Files Modified**: `src/data/dataset.py`
- **Git Commit Command**:
  ```bash
  git add src/data/dataset.py
  git commit -m "perf(data): migrate data loader to tf.data with parallel prefetching"
  git push origin main
  ```

---

### Day 13: Dataset Sanity, Duplicate & Corruption Checking Tool
- **Key Objective**: Ensure dataset has no truncated, blank, or duplicate CT slices before training.
- **Actionable Tasks**:
  - Create `scripts/validate_dataset.py` checking image readability and class balance.
  - Generate preview grid of normal vs cancerous CT slices in `results/`.
- **Files Modified**: `scripts/validate_dataset.py`
- **Git Commit Command**:
  ```bash
  git add scripts/validate_dataset.py
  git commit -m "chore(data): add automated dataset sanity and corrupted image check script"
  git push origin main
  ```

---

### Day 14: Data Pipeline Benchmarking & Week 2 Documentation
- **Key Objective**: Measure data loading speed (images/sec) and document verified dataset distribution.
- **Actionable Tasks**:
  - Run benchmark on `tf.data` loader vs legacy generator.
  - Update data documentation and summary tables.
- **Files Modified**: `docs/data_pipeline.md`, `results/benchmark_data.json`
- **Git Commit Command**:
  ```bash
  git add docs/data_pipeline.md results/benchmark_data.json
  git commit -m "docs(data): benchmark I/O throughput and document baseline data splits"
  git push origin main
  ```

---

## 🗓️ Phase 3: Week 3 (Days 15–21) — Model Modernization, SOTA Backbones & Clinical Metrics

### Day 15: Modern Backbones: EfficientNetV2 & ConvNeXt Integration
- **Key Objective**: Expand beyond legacy VGG16/ResNet50 to modern high-efficiency vision backbones.
- **Actionable Tasks**:
  - Implement `EfficientNetV2B2` and `ConvNeXtBase` in `src/models/model_factory.py`.
  - Standardize classification head with LayerNorm, Dropout, and Dense layers.
- **Files Modified**: `src/models/model_factory.py`, `src/models/convnext.py`, `src/models/efficientnet.py`
- **Git Commit Command**:
  ```bash
  git add src/models/model_factory.py src/models/convnext.py src/models/efficientnet.py
  git commit -m "feat(models): integrate EfficientNetV2 and ConvNeXt transfer learning backbones"
  git push origin main
  ```

---

### Day 16: Focal Loss & Class Imbalance Compensation
- **Key Objective**: Improve detection of rare/subtle cancer lesions with Focal Loss and frequency weighting.
- **Actionable Tasks**:
  - Create `src/training/losses.py` with Binary Focal Loss implementation.
  - Add automatic class weight computation based on train set class ratios.
- **Files Modified**: `src/training/losses.py`, `src/training/trainer.py`
- **Git Commit Command**:
  ```bash
  git add src/training/losses.py src/training/trainer.py
  git commit -m "feat(training): implement focal loss and class-frequency inverse weighting"
  git push origin main
  ```

---

### Day 17: Mixed Precision (FP16) & Cosine Annealing Learning Rate
- **Key Objective**: Accelerate GPU training by 2x-3x and stabilize convergence.
- **Actionable Tasks**:
  - Add mixed precision policy in `src/utils/helpers.py`.
  - Add `CosineAnnealingWarmRestarts` callback in `src/training/callbacks.py`.
- **Files Modified**: `src/training/callbacks.py`, `src/utils/helpers.py`
- **Git Commit Command**:
  ```bash
  git add src/training/callbacks.py src/utils/helpers.py
  git commit -m "perf(training): enable fp16 mixed precision and cosine annealing scheduler"
  git push origin main
  ```

---

### Day 18: Explainable AI Upgrade: Grad-CAM++ & High-Contrast Heatmaps
- **Key Objective**: Provide superior radiologist heatmaps identifying multi-focal tumor lesions.
- **Actionable Tasks**:
  - Implement Grad-CAM++ in `src/utils/gradcam.py`.
  - Add high-contrast colormap overlay and bounding-box highlighting.
- **Files Modified**: `src/utils/gradcam.py`, `tests/test_gradcam.py`
- **Git Commit Command**:
  ```bash
  git add src/utils/gradcam.py tests/test_gradcam.py
  git commit -m "feat(xai): upgrade to Grad-CAM++ with medical colormap blending"
  git push origin main
  ```

---

### Day 19: Clinical Diagnostic Metrics & Youden's J Cutoff Optimization
- **Key Objective**: Compute Sensitivity, Specificity, PPV, NPV, and Youden's J statistic.
- **Actionable Tasks**:
  - Expand `src/evaluation/metrics.py` with clinical medical metrics.
  - Calculate optimal classification threshold maximizing Sensitivity + Specificity - 1.
- **Files Modified**: `src/evaluation/metrics.py`, `scripts/evaluate_model.py`
- **Git Commit Command**:
  ```bash
  git add src/evaluation/metrics.py scripts/evaluate_model.py
  git commit -m "feat(metrics): compute clinical sensitivity, specificity, and Youden J cutoff"
  git push origin main
  ```

---

### Day 20: 5-Fold Patient-Stratified Cross-Validation
- **Key Objective**: Demonstrate robust statistical generalization across independent folds.
- **Actionable Tasks**:
  - Create `scripts/train_cross_validation.py`.
  - Log mean and standard deviation for AUC, Sensitivity, and Specificity.
- **Files Modified**: `scripts/train_cross_validation.py`, `results/cv_summary.json`
- **Git Commit Command**:
  ```bash
  git add scripts/train_cross_validation.py results/cv_summary.json
  git commit -m "feat(eval): implement automated 5-fold patient-level cross validation"
  git push origin main
  ```

---

### Day 21: Model Quantization & ONNX / TFLite Export
- **Key Objective**: Export models for high-speed low-latency inference in web and edge environments.
- **Actionable Tasks**:
  - Implement `scripts/export_onnx.py` to convert Keras models to ONNX and TFLite.
  - Benchmark inference latency between standard Keras and ONNX Runtime.
- **Files Modified**: `scripts/export_onnx.py`, `requirements.txt`
- **Git Commit Command**:
  ```bash
  git add scripts/export_onnx.py requirements.txt
  git commit -m "feat(export): add automated conversion to ONNX runtime and quantized TFLite"
  git push origin main
  ```

---

## 🗓️ Phase 4: Week 4 (Days 22–27) — FastAPI Productionization & PACS Medical UI

### Day 22: Asynchronous FastAPI Core & Pydantic v2 Schema Validation
- **Key Objective**: Overhaul web application API with async task workers and rigorous input validation.
- **Actionable Tasks**:
  - Refactor `app/main.py` and `app/routers/` with Pydantic v2 schemas.
  - Add `/api/health` and `/api/models` endpoints.
- **Files Modified**: `app/main.py`, `app/routers/predict.py`, `app/schemas.py`
- **Git Commit Command**:
  ```bash
  git add app/main.py app/routers/ app/schemas.py
  git commit -m "refactor(api): upgrade endpoints to async with Pydantic v2 validation"
  git push origin main
  ```

---

### Day 23: PACS Dark Radiologist Workstation UI & Heatmap Opacity Slider
- **Key Objective**: Deliver a stunning medical imaging web interface with real-time overlay blending.
- **Actionable Tasks**:
  - Redesign `app/templates/index.html` with dual-pane PACS dark theme.
  - Add interactive JS opacity slider to blend Grad-CAM heatmap over original CT slice.
  - Update `app/static/css/styles.css` with modern glassmorphism styling.
- **Files Modified**: `app/templates/index.html`, `app/static/css/styles.css`, `app/static/js/app.js`
- **Git Commit Command**:
  ```bash
  git add app/templates/ app/static/
  git commit -m "feat(ui): redesign web interface with PACS dark theme and opacity slider"
  git push origin main
  ```

---

### Day 24: Batch Inference & Multi-Slice CT Series Support
- **Key Objective**: Allow uploading full CT slice sequences with highest cancer probability ranking.
- **Actionable Tasks**:
  - Add batch upload endpoint `/api/predict/batch` in `app/routers/predict.py`.
  - Display gallery view sorted by anomaly probability score.
- **Files Modified**: `app/routers/predict.py`, `app/templates/index.html`
- **Git Commit Command**:
  ```bash
  git add app/routers/predict.py app/templates/index.html
  git commit -m "feat(api): support multi-slice batch CT inference and anomaly ranking"
  git push origin main
  ```

---

### Day 25: Automated 1-Click Clinical PDF Diagnostic Report Generator
- **Key Objective**: Generate downloadable PDF report for physicians and academic presentations.
- **Actionable Tasks**:
  - Create `src/utils/report_generator.py` using ReportLab.
  - Include patient metadata, prediction score, decision threshold, and Grad-CAM overlay.
- **Files Modified**: `src/utils/report_generator.py`, `app/routers/report.py`, `requirements.txt`
- **Git Commit Command**:
  ```bash
  git add src/utils/report_generator.py app/routers/report.py requirements.txt
  git commit -m "feat(reports): add 1-click diagnostic PDF export with heatmap overlays"
  git push origin main
  ```

---

### Day 26: API Security Hardening, Magic-Byte Verification & Rate Limiting
- **Key Objective**: Harden API against corrupted uploads, oversized payloads, and path traversal attacks.
- **Actionable Tasks**:
  - Add image header inspection before saving uploads.
  - Enforce `MAX_UPLOAD_SIZE_MB` and sanitize filenames with `uuid4`.
- **Files Modified**: `app/routers/predict.py`, `app/security.py`
- **Git Commit Command**:
  ```bash
  git add app/routers/predict.py app/security.py
  git commit -m "sec(api): implement magic-byte image validation and upload size sanitization"
  git push origin main
  ```

---

### Day 27: End-to-End Pytest Test Suite & Code Coverage
- **Key Objective**: Achieve >85% test coverage across data pipeline, model loading, and API endpoints.
- **Actionable Tasks**:
  - Expand `tests/test_api.py`, `tests/test_models.py`, `tests/test_data.py`.
  - Run `pytest --cov=src --cov=app tests/`.
- **Files Modified**: `tests/test_api.py`, `tests/test_models.py`, `pytest.ini`
- **Git Commit Command**:
  ```bash
  git add tests/ pytest.ini
  git commit -m "test(e2e): add end-to-end tests for data, model inference, and API endpoints"
  git push origin main
  ```

---

## 🗓️ Phase 5: Week 5 (Days 28–30) — Containerization, CI/CD & Cloud Deployment

### Day 28: Production Multi-Stage Dockerfile & Docker Compose
- **Key Objective**: Package entire application into lightweight, reproducible container.
- **Actionable Tasks**:
  - Create multi-stage `Dockerfile` with slim python base and system dependencies.
  - Create `docker-compose.yml` mounting external data and model weights.
- **Files Modified**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **Git Commit Command**:
  ```bash
  git add Dockerfile docker-compose.yml .dockerignore
  git commit -m "ci(docker): add multi-stage Dockerfile and docker-compose configurations"
  git push origin main
  ```

---

### Day 29: GitHub Actions CI/CD Automated Workflow
- **Key Objective**: Automate linting, unit testing, and Docker builds on every git push.
- **Actionable Tasks**:
  - Create `.github/workflows/ci.yml`.
  - Configure steps for ruff linting, pytest matrix, and path validation.
- **Files Modified**: `.github/workflows/ci.yml`
- **Git Commit Command**:
  ```bash
  git add .github/workflows/ci.yml
  git commit -m "ci(github): add workflow for automated linting, pytest, and docker builds"
  git push origin main
  ```

---

### Day 30: Cloud Deployment (Hugging Face Spaces) & Portfolio Polish
- **Key Objective**: Publish a live interactive demo and complete documentation with architectural diagrams.
- **Actionable Tasks**:
  - Deploy FastAPI application to Hugging Face Spaces (or Render).
  - Update `README.md` with live demo link, architecture diagram, benchmark table, and citation format.
- **Files Modified**: `README.md`, `docs/architecture.png`, `LICENSE`
- **Git Commit Command**:
  ```bash
  git add README.md docs/ LICENSE
  git commit -m "docs(readme): add clinical architecture diagrams, benchmarks, and demo badges"
  git push origin main
  ```
