# Pancreatic-Cancer-Detection-Using-Transfer-Learning
Build an end-to-end system that classifies pancreatic CT scan images as Normal or Cancerous using pre-trained deep learning models (VGG16, ResNet50, InceptionV3), with a FastAPI web app for deployment.

Comprehensive project documentation with setup instructions, usage guide, architecture diagram, and academic references.

[NEW] 
requirements.txt

tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
opencv-python>=4.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
Pillow>=9.4.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
jinja2>=3.1.0
tqdm>=4.65.0
[NEW] 
config.py
Centralized configuration with:

Image dimensions per model (224×224 for VGG16/ResNet50, 299×299 for InceptionV3)
Batch size, epochs, learning rate
Data paths and split ratios
Class names and labels
Model save paths
Component 2: Data Pipeline
[NEW] 
src/data/preprocessing.py
Image resizing to target dimensions
Pixel normalization (0-1 range)
Optional noise reduction (Gaussian blur)
Image format conversion (DICOM/NIfTI → PNG/JPEG)
Train/val/test split (70/15/15)
[NEW] 
src/data/augmentation.py
ImageDataGenerator configuration with:
Rotation (±20°)
Width/height shift (±10%)
Horizontal flip
Zoom (0.8–1.2)
Brightness adjustment
Shear transformation
[NEW] 
src/data/dataset.py
create_generators() — returns train/val/test data generators
load_dataset_info() — scans directories, returns class distribution
get_sample_images() — loads sample images for visualization
Component 3: Transfer Learning Models
Each model follows the same architecture pattern:


Pre-trained Base (frozen) → GlobalAveragePooling2D → Dense(512, relu) → 
BatchNorm → Dropout(0.5) → Dense(256, relu) → BatchNorm → 
Dropout(0.3) → Dense(1, sigmoid)
[NEW] 
src/models/base_model.py
Abstract base class defining:

build() — constructs the model
compile_model() — sets optimizer, loss, metrics
freeze_base() / unfreeze_base() — for fine-tuning
get_summary() — model architecture summary
save() / load() — model persistence
[NEW] 
src/models/vgg16_model.py
Base: VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
Fine-tuning: Unfreeze last 4 conv layers after initial training
Optimizer: Adam with lr=1e-4 (initial), 1e-5 (fine-tuning)
[NEW] 
src/models/resnet50_model.py
Base: ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
Fine-tuning: Unfreeze from conv5_block1 onwards
Optimizer: Adam with lr=1e-4
[NEW] 
src/models/inception_model.py
Base: InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
Fine-tuning: Unfreeze from mixed7 onwards
Optimizer: Adam with lr=1e-4
[NEW] 
src/models/model_factory.py
Factory pattern: ModelFactory.create("resnet50") returns the configured model instance.

Component 4: Training Pipeline
[NEW] 
src/training/trainer.py
Two-phase training strategy:

Phase 1 — Feature Extraction (10-15 epochs): Frozen base, train only custom layers
Phase 2 — Fine-Tuning (10-20 epochs): Unfreeze top base layers, very low learning rate
Features:

Training history logging
Checkpoint saving (best val_accuracy)
Progress tracking with tqdm
[NEW] 
src/training/callbacks.py
EarlyStopping(patience=7, restore_best_weights=True)
ReduceLROnPlateau(factor=0.2, patience=3)
ModelCheckpoint — saves best model
TensorBoard — optional training visualization
Component 5: Evaluation & Visualization
[NEW] 
src/evaluation/metrics.py
Computes and returns:

Accuracy, Precision, Recall, F1-Score
Confusion matrix (TP, TN, FP, FN)
ROC curve data and AUC score
Classification report (per-class)
[NEW] 
src/evaluation/visualizations.py
Publication-quality plots:

Training/validation accuracy & loss curves
Confusion matrix heatmap (seaborn)
ROC-AUC curve
Precision-Recall curve
Sample predictions with confidence scores
Model comparison bar charts
Component 6: Grad-CAM Explainability
[NEW] 
src/utils/gradcam.py
Generates heatmap overlays showing which image regions influenced the prediction
Supports all three model architectures
Outputs: original image + heatmap overlay + superimposed visualization
Component 7: Jupyter Notebooks
[NEW] 
notebooks/01_data_exploration.ipynb
Dataset statistics (image count per class, resolution distribution)
Sample image visualization grid
Class distribution bar chart
Image intensity histograms
[NEW] 
notebooks/02_preprocessing.ipynb
Step-by-step preprocessing demonstration
Before/after augmentation visualization
Data pipeline validation
[NEW] 
notebooks/03_model_training.ipynb
Train all three models sequentially
Side-by-side training curve comparison
Save trained models and history
[NEW] 
notebooks/04_model_evaluation.ipynb
Load trained models and evaluate on test set
Generate all metrics and plots
Model comparison summary table
[NEW] 
notebooks/05_gradcam_visualization.ipynb
Grad-CAM heatmaps for sample predictions
Comparison of attention regions across models
Component 8: FastAPI Web Application
[NEW] 
app/main.py
FastAPI app with Jinja2 templates
Static file serving
CORS middleware
Model loading at startup
Routes: / (upload page), /predict (classification endpoint)
[NEW] 
app/routers/prediction.py
POST /predict — accepts uploaded image, returns prediction
POST /predict/api — JSON API endpoint for programmatic access
Image preprocessing before prediction
Confidence score calculation
Optional Grad-CAM generation
[NEW] 
app/templates/index.html
Premium dark-themed UI with:

Drag-and-drop image upload zone
Model selector (VGG16 / ResNet50 / InceptionV3)
Real-time image preview
Animated prediction results
Confidence gauge visualization
Grad-CAM overlay display
Medical disclaimer banner
[NEW] 
app/static/css/styles.css
Modern glassmorphism design:

Dark gradient background (#0a0a1a → #1a1a3e)
Frosted glass cards with backdrop-filter
Animated gradient borders
Custom file upload styling
Responsive layout (mobile-friendly)
Smooth transitions and micro-animations
Confidence meter with gradient fill (green → yellow → red)
[NEW] 
app/static/js/app.js
Drag-and-drop file handling
AJAX prediction request
Animated results rendering
Image preview with zoom
Loading spinner animation
Error handling and validation
Implementation Order
Phase	Components	Est. Files
1	Foundation (config, requirements, README)	3
2	Data pipeline (preprocessing, augmentation, dataset)	5
3	Transfer learning models (base + 3 architectures + factory)	6
4	Training pipeline (trainer, callbacks)	3
5	Evaluation & visualization	3
6	Grad-CAM explainability	1
7	Jupyter notebooks (all 5)	5
8	FastAPI web application	7
9	Tests & final integration	4
Verification Plan
Automated Tests
bash

# Run unit tests
python -m pytest tests/ -v
# Verify model builds correctly
python -c "from src.models.model_factory import ModelFactory; m = ModelFactory.create('resnet50'); print(m.model.summary())"
# Start FastAPI server
uvicorn app.main:app --reload --port 8000
Manual Verification
Data Pipeline: Run notebook 01 & 02 to verify data loading and augmentation
Model Training: Train on a small sample to verify the pipeline works end-to-end
Web App: Upload a test image through the browser and verify prediction output
Grad-CAM: Verify heatmap overlays are generated correctly
Responsive UI: Test the web interface on different screen sizes
Key Metrics Targets
Metric	Target
Accuracy	> 85%
Precision	> 80%
Recall	> 85%
F1-Score	> 82%
AUC-ROC	> 0.90
NOTE

Actual performance depends heavily on dataset quality, size, and class balance. These targets assume a reasonably balanced dataset with 1000+ images per class.
