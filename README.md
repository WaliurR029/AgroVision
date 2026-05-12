# AgroVision — AI-Powered Plant Disease Detection & Crop Advisory System

AgroVision is an AI-driven smart agriculture system that uses Deep Learning and Computer Vision to detect plant diseases from crop leaf images and provide treatment recommendations.

The project was developed to support precision agriculture by helping farmers, researchers, and agricultural enthusiasts identify crop diseases at an early stage and reduce crop loss through AI-assisted diagnosis.

AgroVision combines:

* Computer Vision
* Transfer Learning
* Deep Learning
* Agricultural Knowledge Systems
* Web-based Deployment

into a single practical smart farming solution.

---

# Project Highlights

✅ Deep Learning-based plant disease detection
✅ MobileNetV2 Transfer Learning architecture
✅ Real-time prediction pipeline
✅ Treatment recommendation system
✅ Streamlit-powered web application
✅ Image preprocessing & augmentation
✅ Multi-class disease classification
✅ Confusion matrix & training visualization
✅ Scalable architecture for future AI agriculture systems

---

# Problem Statement

Plant diseases are one of the major causes of agricultural productivity loss worldwide. Many farmers struggle to identify diseases accurately during early stages, which often results in:

* Reduced crop yield
* Economic loss
* Excessive pesticide usage
* Delayed treatment
* Poor disease management

AgroVision aims to solve this issue by leveraging Artificial Intelligence to automate disease detection and provide quick treatment guidance.

---

# System Workflow

```text
Leaf Image Upload
        ↓
Image Preprocessing
        ↓
MobileNetV2 Deep Learning Model
        ↓
Disease Classification
        ↓
Treatment Recommendation System
        ↓
Farmer Guidance Output
```

---

# AI/ML Architecture

The model is developed using Transfer Learning with MobileNetV2.

### Why MobileNetV2?

MobileNetV2 was selected because:

* Lightweight architecture
* Fast inference speed
* High accuracy on image classification tasks
* Efficient deployment capability
* Suitable for edge/mobile applications

---

# Model Pipeline

## Data Preprocessing

The dataset images were processed using:

* Rescaling
* Rotation augmentation
* Brightness adjustment
* Zoom augmentation
* Horizontal flipping
* Validation splitting

This improves model generalization and reduces overfitting.

---

## Transfer Learning

The pretrained ImageNet weights from MobileNetV2 were fine-tuned on the PlantVillage dataset.

### Architecture:

```text
MobileNetV2
      ↓
GlobalAveragePooling2D
      ↓
Dropout Layer
      ↓
Dense Layer
      ↓
Softmax Output Layer
```

---

## Optimization Techniques

The project uses several modern optimization techniques:

| Technique              | Purpose                          |
| ---------------------- | -------------------------------- |
| AdamW Optimizer        | Stable convergence               |
| Label Smoothing        | Better generalization            |
| Dropout                | Reduce overfitting               |
| EarlyStopping          | Prevent unnecessary training     |
| ReduceLROnPlateau      | Dynamic learning rate adjustment |
| Class Weight Balancing | Handle class imbalance           |

---

# Dataset

### Dataset Used:

PlantVillage Dataset

The dataset contains multiple crop disease categories with thousands of labeled leaf images.

### Supported Crops

## Tomato

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Target Spot
* Tomato Mosaic Virus
* Yellow Leaf Curl Virus
* Spider Mites
* Healthy

## Potato

* Early Blight
* Late Blight
* Healthy

## Pepper Bell

* Bacterial Spot
* Healthy

---

# 🛠️ Technology Stack

| Technology   | Usage                          |
| ------------ | ------------------------------ |
| Python       | Core Programming               |
| TensorFlow   | Deep Learning                  |
| Keras        | Model Building                 |
| MobileNetV2  | Transfer Learning              |
| NumPy        | Numerical Operations           |
| OpenCV       | Image Processing               |
| Matplotlib   | Visualization                  |
| Seaborn      | Confusion Matrix Visualization |
| Scikit-learn | Evaluation Metrics             |
| Streamlit    | Web Deployment                 |
| Git & GitHub | Version Control                |

---

# Model Performance

| Metric              | Result       |
| ------------------- | ------------ |
| Validation Accuracy | ~95%+        |
| Architecture        | MobileNetV2  |
| Dataset             | PlantVillage |
| Classification Type | Multi-Class  |

---

# Evaluation Metrics

The system performance was evaluated using:

* Accuracy
* Validation Loss
* Confusion Matrix
* Classification Report

These evaluation metrics help measure:

* Prediction reliability
* Generalization capability
* Disease classification effectiveness

---

# Application Features

## Disease Detection

Users can upload leaf images and receive disease predictions instantly.

## Treatment Recommendation

After prediction, AgroVision provides:

* Suggested treatments
* Crop management advice
* Preventive measures
* Agricultural best practices

## Visualization Support

The system also provides:

* Accuracy curves
* Loss curves
* Confusion matrix visualization

---

# Recommended README Images

YES — you absolutely SHOULD add images.

Projects with screenshots perform significantly better on:

* LinkedIn
* GitHub
* Portfolio reviews
* Internship applications

---
# Recommended Repository Structure

```bash
AgroVision/
│
├── app.py
├── predict.py
├── best_agrovision_model.keras
├── class_indices.json
├── requirements.txt
├── README.md
│
├── PlantVillage/
│   ├── Tomato_Late_blight
│   ├── Tomato_healthy
│   ├── Potato_Early_blight
│   └── ...
│
├── screenshots/
│   ├── app_ui.png
│   ├── prediction_demo.png
│   └── treatment_output.png
│
├── confusion_matrix.png
├── accuracy_curve.png
├── loss_curve.png
│
└── streamlit/
```

---

# ▶️ Installation Guide

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AgroVision.git
cd AgroVision
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---
# Real-World Impact

AgroVision was designed with real agricultural impact in mind.

The system can help:

* Farmers detect diseases early
* Reduce crop damage
* Improve productivity
* Minimize unnecessary pesticide use
* Support smart farming initiatives

This project demonstrates how Artificial Intelligence can be applied to solve real-world agricultural challenges.

---

# Author

## Waliur Rahman

Software Engineering Student
AI/ML Enthusiast | Computer Vision | Smart Agriculture Systems

---

# Support

If you found this project useful, consider:

Starring the repository
Forking the project
Sharing it on LinkedIn

---
