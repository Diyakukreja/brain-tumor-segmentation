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

The proposed model improves upon standard U-Net and Attention U-Net by combining multi-scale context extraction, feature fusion, and attention-guided learning techniques.

---

# 🚀 Key Features

✅ Automatic segmentation of tumor regions from MRI images  
✅ Improved localization of small tumors  
✅ Reduced false positives in healthy regions  
✅ Cleaner and sharper tumor boundaries  
✅ Quantitative evaluation using multiple metrics  
✅ Visualization of predictions and performance graphs  
✅ Pretrained model weights available for direct testing  

---

# 🏗️ Model Architecture

The final architecture is based on U-Net with multiple enhancements:

## Encoder
- Convolution layers
- Residual Blocks
- Max Pooling

## Bottleneck
- Pyramid Pooling Module
- Multi-scale context extraction

## Feature Fusion
- Basic Feature Pyramid Network (FPN)
- Multi-level feature aggregation

## Decoder
- Upsampling
- Attention-guided skip connections
- Residual refinement blocks

## Output
- `1×1 Conv + Sigmoid`
- Binary tumor mask

---

# 📂 Project Structure

```bash
.
├── CV_report_1727.pdf        # Final project report
├── Colab notebook.ipynb      # Training + testing notebook (base + improved models)
├── app.py                    # Inference app
├── architecture.png          # Model architecture diagram
├── fnr.png                   # False prediction rate graph
├── matrix.png                # Confusion matrix
├── metrics.png               # Evaluation metrics graph
├── results.png               # Sample prediction outputs
├── segmentation.png          # Segmentation comparison visuals
├── requirements.txt          # Python dependencies
├── .gitignore
└── .gitattributes
```
# Pretrained ML Models
Trained model files are available in Google Drive:

🔗 Download Models Folder:
https://drive.google.com/drive/folders/1-KDrt0HqDDVuS-DMTjw8JkL8uwPgx37T?usp=sharing

Includes:
Base U-Net model weights
Improved Attention Residual U-Net weights
Saved checkpoints
Ready-to-use .pth files for inference/testing

📌 Place downloaded model files in the project root directory before running the app.

# Dataset
LGG MRI Segmentation Dataset

Low-Grade Glioma MRI images with pixel-wise tumor masks.

Used for:

-Training
-Validation
-Testing

# Training Details
Framework: PyTorch
Platform:	Google Colab
Optimizer:	Adam
Epochs:	30
Batch Size:	8
Image Size:	256 × 256
Loss Function:	Dice + BCE + Focal Loss

# 📈 Evaluation Metrics

The model was evaluated using the following metrics:

- Accuracy  
- Dice Score  
- IoU  
- Precision  
- Recall  
- Sensitivity  
- Specificity  
- F1 Score  
- Confusion Matrix  
- False Positive / False Negative Rates  

---

# 🏆 Final Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.995 |
| Dice Score | 0.839 |
| Sensitivity | 0.781 |
| Specificity | 0.997 |

---

# 🖼️ Results

## Sample Predictions

**MRI Input → Ground Truth → Predicted Mask**

---

## 📊 Performance Graphs Included

- Metrics Bar Chart  
- Confusion Matrix  
- False Prediction Rates  
- Qualitative Segmentation Outputs  

---

# ▶️ How to Run

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

```
## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```
## 3️⃣ Download Models

Download the pretrained .pth files from the Google Drive link above and place them in the project folder.

## 4️⃣ Run App.py

# Author
Diya Kukreja
B.Tech CSE
Netaji Subhas University of Technology (NSUT), Delhi

