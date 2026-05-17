# 🩻 CXR-Net: Thoracic Disease Classification from Chest X-Rays
### CNN · DenseNet121 · Multi-Label Classification · TensorFlow · OpenCV · Self-Project

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Dataset](https://img.shields.io/badge/NIH-ChestXray14_(5K_subset)-blue?style=flat-square)
![Status](https://img.shields.io/badge/Val_Accuracy-75.16%25-brightgreen?style=flat-square)

---

## 🧩 Problem Statement

Chest X-ray interpretation requires expert radiologists — scarce, expensive, and slow. This project builds a **multi-label CNN classifier** that detects the presence of **14 thoracic pathologies** from a single CXR image, trained on the NIH ChestX-ray14 dataset.

> *"Given a chest X-ray image, detect which of 14 diseases (Pneumonia, Effusion, Cardiomegaly, etc.) are present — a true multi-label medical imaging problem."*

---

## 🎯 14 Disease Classes

`Atelectasis` · `Cardiomegaly` · `Consolidation` · `Edema` · `Effusion` · `Emphysema` · `Fibrosis` · `Hernia` · `Infiltration` · `Mass` · `No Finding` · `Nodule` · `Pleural Thickening` · `Pneumonia` · `Pneumothorax`

---

## ⚙️ Pipeline Overview

```
NIH ChestX-ray14 Dataset (5,000 image subset)
        │
        ▼
Patient-Aware Data Splitting
  └── Patients strictly separated across train/val/test
      (prevents data leakage)
        │
        ▼
Image Preprocessing (OpenCV)
  ├── Resize to 128×128 pixels
  ├── Grayscale normalization (mean=0.5127, std=0.2527)
  └── Multi-hot label encoding for 14 classes
        │
        ▼
Class Imbalance Handling
  └── Custom per-sample weighting (w_pos / w_neg per class)
        │
        ▼
Data Augmentation (ImageDataGenerator)
  ├── Rotation (±10°), horizontal flip
  ├── Width/height shift, zoom, shear
  └── Memory-efficient batch loading
        │
        ▼
Model Training — Two Architectures:
  ├── Custom CNN (Conv2D × 3 + GAP + Dropout + Sigmoid)
  └── DenseNet121 (Transfer Learning — ImageNet weights)
        │
        ▼
Evaluation: Accuracy + AUC (multi-label)
```

---

## 📊 Results

| Model | Val Accuracy | Val AUC | Epochs |
|---|---|---|---|
| **Custom CNN** | **75.16%** | 0.57 | 30 |
| DenseNet121 (fine-tuned) | In progress | — | 12 |

**Custom CNN Architecture:**
```python
Input (128×128×1)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + GlobalAveragePooling
→ Dropout(0.3)
→ Dense(14, activation='sigmoid')   # Multi-label output
```

---

## 🔑 Key Engineering Decisions

**1. Patient-Aware Splitting**
Patients strictly separated between train/val/test — prevents data leakage where the same patient's images appear in both training and evaluation sets.

**2. Class Imbalance Handling**
Rare diseases (Hernia, Pneumonia) severely underrepresented — custom per-sample weighting strategy ensures the model learns from minority classes.

**3. Memory-Efficient Pipeline**
Full dataset too large for RAM — used `ImageDataGenerator.flow_from_dataframe()` for batch-wise loading without loading all images into memory.

**4. Multi-Label Sigmoid Output**
Unlike softmax (single class), sigmoid on each output neuron allows **multiple diseases to be detected simultaneously** from one image.

---

## 📁 Repository Structure

```
📦 CXR-Net-Thoracic-Disease-Classification/
├── training.ipynb                    # Full pipeline: preprocessing + CNN + DenseNet121
├── Data_Entry_2017.csv               # NIH dataset metadata (labels + patient IDs)
├── train_log.csv                     # Training history log
├── requirements.txt                  # Python dependencies
├── README.md
└── .gitignore
```

> **Note:** Model weights (`.keras`) and image files not included due to size. Dataset available via [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data).

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
```

Download dataset via Kaggle API:
```bash
kaggle datasets download -d nih-chest-xrays/data
```

Then open and run `training.ipynb` sequentially.

---

## 🧠 Concepts Covered

`Medical Image Classification` · `Multi-Label Classification` · `CNN` · `Transfer Learning (DenseNet121)` · `Data Augmentation` · `Class Imbalance` · `OpenCV Preprocessing` · `Patient-Aware Splitting` · `NIH ChestX-ray14`

---

## 👤 Author

**Dhruv Kumar Sahu**
M.Tech, Industrial & Management Engineering — IIT Kanpur
GATE 2024 AIR 33 | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-sahu-157ab9193/) · [GitHub](https://github.com/dhruvkumar24-ai)
