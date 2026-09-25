# CancerNet: Invasive Ductal Carcinoma (IDC) Classifier

## 📌 Objective & Problem Statement
Breast cancer (BC) is one of the most common cancers among women worldwide, where Invasive Ductal Carcinoma (IDC) accounts for the vast majority of cases. Early detection enables timely clinical intervention and significantly improves patient survival rates. 

This project implements **CancerNet**, an end-to-end Deep Learning Convolutional Neural Network (CNN) built with TensorFlow/Keras to analyze biological microscopic histology images (50x50 pixel patches) and accurately classify them into **Benign (0)** or **Malignant (1)**.

---

## 📊 Project Questions & Scientific Justification

* **Training and Testing Split**: 
  * 80% Training set (3,200 images) and 20% Validation/Testing set (800 images), sampled from the public Kaggle IDC dataset.
* **Epochs & Iterations**: 
  * Trained for 5 epochs using the Adam optimizer and Binary Crossentropy loss.
* **Accuracy Progression**:
  * The model achieved **83.25% Validation Accuracy** with a validation loss of 0.7842 after 5 epochs.
* **Is CNN Best or are there better alternatives?**:
  * CNNs are the benchmark for histological image patches due to their spatial feature extraction (cellular density, nuclear pleomorphism, tissue architecture). 
  * **Alternatives**: Vision Transformers (ViTs) and Transfer Learning architectures (ResNet-50, EfficientNet) provide superior performance on complex whole-slide images (WSIs) by capturing long-range contextual dependencies across larger tissue areas.
* **Overfitting / Underfitting Justification**:
  * The model maintains an **optimal balance**. Overfitting was mitigated by integrating **Batch Normalization** layers and progressive **Dropout** (0.25, 0.3, 0.5) across feature extraction blocks and dense classification heads.
* **Clinical Real-World Application**:
  * Can be deployed as a **"Second-Reader AI Screening Assistant"** in digital pathology laboratories. When whole mount slides are digitized, the model generates heatmaps over suspected IDC regions, reducing manual scanning time for pathologists and minimizing fatigue-related diagnostic errors.

---

## 🧠 CancerNet Architecture
* **Input Layer**: 50x50 RGB Histology Patches with `Rescaling(1./255)`
* **Conv Block 1**: `Conv2D(32, 3x3)` + `BatchNormalization` + `MaxPooling2D(2x2)`
* **Conv Block 2**: `Conv2D(64, 3x3)` + `BatchNormalization` + `MaxPooling2D(2x2)` + `Dropout(0.25)`
* **Conv Block 3**: `Conv2D(128, 3x3)` + `BatchNormalization` + `MaxPooling2D(2x2)` + `Dropout(0.3)`
* **Classification Head**: `Flatten` + `Dense(128, ReLU)` + `Dropout(0.5)` + `Dense(1, Sigmoid)`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SandeshGawali04/breast-cancer-cancernet.git](https://github.com/SandeshGawali04/breast-cancer-cancernet.git)
   cd breast-cancer-cancernet
