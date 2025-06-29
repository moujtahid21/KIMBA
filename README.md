# Advanced Tumor Recognition and Prediction in Medical Imaging

A sophisticated deep learning system for detecting tumor types in IRMs using a VGG19-ResNet50 Architecture.

## 🎯 Overview
This project implements a state-of-the-art tumor detection system designed for IRM scans of lung and breast tumors. The model combines the power of a pre-trained very large CNN (VGG19) and a Deep Residual-Learning framework (ResNet50) to achieve high accuracy in identifying benign, malignant tumor types.

## Key Features
- **Domain specific**: Uses very large CNNs and Deep Learning models specifically trained on all sorts of images
- **Hybrid Architecture**: Combines CNN and Residual Deep Learning model for ...
- **Multiscale Feature Extraction**: Uses multiple layers composed of different kernel sizes to capture different patterns.
- **Production Ready**: Complete training pipeline with evaluation metrics and visualization

## 🏗️ Architecture 
```
Input image --> Transformers --> Multi-Scale VGG19 + ResNet50 --> Malignan / Benign / Normal
```

### Model components

## 📊 Dataset
The model is trained on the **Lung cancer, breast cancer and brain tumor datasets** which contains:
- IRM scans of different patient's breast, lungs and brains.
- Multi-class labels for lung and cancer data (benign, malignant and normal) and binary labels for brain data (yes, no)

### Data split
**Training**: 80%, **Test**: 10% and **Validation**: 10%.

## 🚀 Quick start
```bash
pip install requirement.txt
```
