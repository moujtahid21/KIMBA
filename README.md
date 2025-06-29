# Advanced Tumor Recognition and Prediction in Medical Imaging

-----

🎯 **Overview**

This project introduces a sophisticated deep learning system for **detecting and classifying tumor types in Medical Resonance Imaging (MRI) scans**. Our model utilizes a powerful **VGG19-ResNet50 hybrid architecture** to accurately identify benign and malignant tumors, with a primary focus on lung and breast cancers.

-----

✨ **Key Features**

  * **Domain-Specific Excellence**: Built upon advanced Convolutional Neural Networks (CNNs) and deep learning models, our system is meticulously tuned for precise medical image analysis.
  * **Hybrid Architecture**: We've combined the robust feature extraction capabilities of **VGG19** with the deep residual learning of **ResNet50**. This fusion enables enhanced pattern recognition and superior classification performance.
  * **Multi-Scale Feature Extraction**: The architecture integrates multiple convolutional layers with varying kernel sizes, allowing the capture of diverse patterns and details at different scales within MRI scans.
  * **Production-Ready Pipeline**: The system includes a comprehensive training pipeline, complete with evaluation metrics and visualization tools, making it ready for real-world application.

-----

🏗️ **Architecture**

Our streamlined architecture processes input images through a series of advanced components:

```
Input MRI Scan
      ↓
Transformers (for preprocessing)
      ↓
Multi-Scale VGG19 + ResNet50 (for feature extraction and classification)
      ↓
Malignant / Benign / Normal Classification
```

-----

📊 **Dataset**

The model is rigorously trained on a diverse collection of **Lung, Breast, and Brain Tumor datasets**, comprising:

  * MRI scans from a variety of patients' breasts, lungs, and brains.
  * **Multi-class labels** for lung and breast cancer data (distinguishing between benign, malignant, and normal cases).
  * **Binary labels** for brain tumor data (indicating the presence or absence of a tumor).

**Data Split**:

To ensure robust training and evaluation, the dataset is strategically split as follows:

  * **Training**: 80%
  * **Test**: 10%
  * **Validation**: 10%

-----

🚀 **Quick Start**

Get started with our tumor detection system in just a few steps:

### Prerequisites

First, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Then, download the pre-trained models:

```bash
python download_models.py
```

### Basic Usage

Run the Streamlit application to interact with the system:

```bash
streamlit run streamlit_app.py
```
