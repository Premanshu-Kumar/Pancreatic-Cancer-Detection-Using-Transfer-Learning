# 🔬 Cancer Detection Using Transfer Learning

> An automated **Pancreatic Cancer Detection System** that classifies medical CT scan images as **Normal** or **Cancerous** using pre-trained deep learning models (VGG16, ResNet50, InceptionV3).

⚠️ **Medical Disclaimer**: This system is a research/educational tool and must NOT be used as a replacement for professional medical diagnosis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Web Application](#web-application)
- [Results](#results)
- [Future Scope](#future-scope)
- [References](#references)

---

## 🧬 Overview

Cancer is one of the leading causes of death globally, and early detection plays a crucial role in improving survival rates. This project applies **Transfer Learning** to classify pancreatic CT scan images into two categories:

| Class | Description |
|-------|-------------|
| **Normal** | Healthy pancreatic tissue |
| **Cancerous** | Tissue showing cancerous abnormalities |

### Workflow

```
Medical Image → Preprocessing → Transfer Learning Model → Feature Extraction → Classification → Normal / Cancerous
```

---

## ✨ Features

- 🧠 **Three pre-trained models**: VGG16, ResNet50, InceptionV3
- 🔄 **Two-phase training**: Feature extraction → Fine-tuning
- 📊 **Comprehensive evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
- 🔍 **Grad-CAM explainability**: Visualize which regions influence predictions
- 🌐 **FastAPI web app**: Upload images and get predictions
- 📈 **Data augmentation**: Rotation, zoom, flip, shift, brightness
- 📓 **Jupyter notebooks**: Interactive exploration and training

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Web App | FastAPI, Uvicorn, Jinja2 |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
Project Dox/
├── config.py                  # Centralized configuration
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── data/                      # Dataset directory
│   ├── processed/             # Train / Val / Test splits
│   └── sample/                # Quick-test samples
├── notebooks/                 # Jupyter notebooks (01–05)
├── src/                       # Source modules
│   ├── data/                  # Preprocessing, augmentation, loaders
│   ├── models/                # VGG16, ResNet50, InceptionV3
│   ├── training/              # Trainer, callbacks
│   ├── evaluation/            # Metrics, visualizations
│   └── utils/                 # Grad-CAM, helpers
├── models/                    # Saved trained models
├── results/                   # Plots and evaluation results
├── app/                       # FastAPI web application
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JS, images
│   └── routers/               # API endpoints
└── tests/                     # Unit tests
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cancer-detection-transfer-learning.git
cd cancer-detection-transfer-learning
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Directories
```bash
python config.py
```

### 5. Prepare Dataset
Place your dataset images in the following structure:
```
data/processed/
├── train/
│   ├── Normal/        # Normal images
│   └── Cancerous/     # Cancerous images
├── val/
│   ├── Normal/
│   └── Cancerous/
└── test/
    ├── Normal/
    └── Cancerous/
```

---

## 🚀 Usage

### Training (Python Script)
```bash
# Train a specific model
python -m src.training.trainer --model resnet50

# Train all models
python -m src.training.trainer --model all
```

### Training (Jupyter Notebook)
Open `notebooks/03_model_training.ipynb` in Jupyter or Google Colab.

### Prediction
```python
from src.models.model_factory import ModelFactory

model = ModelFactory.create("resnet50")
model.load("models/resnet50_best.keras")
prediction = model.predict("path/to/image.jpg")
print(f"Prediction: {prediction}")
```

### Web Application
```bash
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

---

## 🧠 Model Architectures

| Model | Input Size | Parameters | Fine-Tune From |
|-------|-----------|------------|----------------|
| VGG16 | 224×224 | ~138M | Last 4 layers |
| ResNet50 | 224×224 | ~25.6M | conv5_block1 |
| InceptionV3 | 299×299 | ~23.9M | mixed7 |

### Custom Classification Head
```
Base Model (frozen) → GlobalAveragePooling2D → Dense(512, ReLU) →
BatchNorm → Dropout(0.5) → Dense(256, ReLU) → BatchNorm →
Dropout(0.3) → Dense(1, Sigmoid)
```

---

## 🌐 Web Application

The FastAPI web application provides:
- **Drag-and-drop** image upload
- **Model selection** (VGG16 / ResNet50 / InceptionV3)
- **Real-time predictions** with confidence scores
- **Grad-CAM visualizations** of prediction regions
- **Premium dark-themed UI** with glassmorphism design

---

## 📊 Results

Results are saved in the `results/` directory after training:
- Training/validation accuracy & loss curves
- Confusion matrix heatmap
- ROC-AUC curve
- Model comparison charts

---

## 🔮 Future Scope

- Larger, more diverse datasets
- Multi-class cancer classification
- Advanced architectures (EfficientNet, Vision Transformers)
- Mobile application deployment
- Cloud platform deployment (AWS, GCP)
- Integration with healthcare systems
- Federated learning for privacy-preserving training

---

## 📚 References

1. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556
2. He, K., et al. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385
3. Szegedy, C., et al. (2015). *Rethinking the Inception Architecture for Computer Vision*. arXiv:1512.00567
4. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. arXiv:1610.02391

---

# 📜 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

*Built with ❤️ using TensorFlow and Transfer Learning*
