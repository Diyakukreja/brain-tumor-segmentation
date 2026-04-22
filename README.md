# 🧠 Brain Tumor Segmentation using Attention-Based Residual U-Net with FPN and Pyramid Pooling

A deep learning project for automatic **brain tumor segmentation from MRI scans** using an improved U-Net architecture enhanced with:

- ⭐ Residual Blocks  
- ⭐ Attention Gates  
- ⭐ Feature Pyramid Network (FPN)  
- ⭐ Pyramid Pooling Module (PPM)  
- ⭐ Hybrid Loss Function  

This project was developed as a final-year Computer Vision / Deep Learning project and focuses on accurate tumor localization, reduced false positives, and improved boundary prediction.

---

# 📌 Overview

Brain tumor segmentation is a critical task in medical image analysis. Manual annotation of MRI scans is time-consuming and requires expert radiologists. This project automates pixel-level tumor segmentation using convolutional neural networks.

The proposed model improves upon standard U-Net and Attention U-Net by combining multi-scale context extraction and feature fusion techniques.

---

# 🚀 Key Features

✅ Automatic segmentation of tumor regions from MRI images  
✅ Improved localization of small tumors  
✅ Reduced false positives in healthy regions  
✅ Cleaner and sharper tumor boundaries  
✅ Quantitative evaluation using multiple metrics  
✅ Visualization of predictions and performance graphs  

---

# 🏗️ Model Architecture

The final architecture is based on U-Net with multiple enhancements:

### Encoder
- Convolution layers
- Residual Blocks
- Max Pooling

### Bottleneck
- Pyramid Pooling Module
- Multi-scale context extraction

### Feature Fusion
- Basic Feature Pyramid Network (FPN)
- Multi-level feature aggregation

### Decoder
- Upsampling
- Attention-guided skip connections
- Residual refinement blocks

### Output
- `1x1 Conv + Sigmoid`
- Binary tumor mask

---

# 📂 Project Structure

```bash id="8l7t1c"
.
├── CV_report_1727.pdf        # Final project report
├── Colab notebook.ipynb      # Training + testing notebook (base + improved models)
├── app.py                    # Inference web app
├── architecture.png          # Model architecture diagram
├── fnr.png                   # False prediction rate graph
├── matrix.png                # Confusion matrix
├── metrics.png               # Evaluation metrics graph
├── results.png               # Sample prediction results
├── segmentation.png          # Segmentation outputs
├── requirements.txt          # Python dependencies
├── .gitignore
└── .gitattributes
