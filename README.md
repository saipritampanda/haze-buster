# 🌫️ Haze Buster - Lightweight Hybrid CNN for Real-Time Image Dehazing with Perceptual Loss Optimization

A lightweight and robust image dehazing web application powered by a refined hybrid CNN model inspired by AOD-Net.


## Project Overview

**Haze Buster** is an end-to-end AI-based system that removes haze from real-world images using a custom-built deep learning model. The model enhances visibility in hazy images, trained on RESIDE's dataset and deployed via a FastAPI backend for real-time image dehazing.

<a href="https://image-dehaze.vercel.app/" target="_blank">
  <img src="https://img.shields.io/badge/Image%20Dehazing%20Website-Visit-blue?style=for-the-badge&logo=github" alt="Image Dehazing Website"/>
</a>


## 🚀 Key Features

- 🔧 **Custom Lightweight CNN Model**: Designed for real-time image dehazing using dynamic input dimensions.
- 📊 **Refined Architecture**: Combines multi-scale feature extraction, AOD-Net formulation, and refinement blocks.
- 🌐 **REST API Backend**: Built using FastAPI and integrated with TensorFlow for serving predictions.
- 🧪 **High Accuracy Metrics**: Achieved excellent **PSNR** and **SSIM** values with optimized loss curves.
- 📸 **End-to-End Workflow**: From dataset preprocessing to live inference through web interface.



## Model Architecture

### 1. Full Pipeline Overview

![Implementation Pipeline](Images/Implementation.png)

### 2. Methodology

<p align="center">
  <img src="Images/Methodology.png" alt="Methodology" width="600"/>
</p>

### 2. Detailed CNN Architecture

![Architecture](Images/Architecture.png)

- **Input Layer**: Accepts RGB images of any size `(None, None, 3)`.
- **Multi-Scale Feature Extraction Block**: Series of 1×1, 3×3, 5×5, and 7×7 convolutions.
- **AOD-Net Inspired Formula**: Applies the formula `J(x) = K(x) * I(x) - K(x) + 1` for transmission map estimation.
- **Refinement Block**: 2-layer convolution with batch normalization.
- **Output Layer**: Sigmoid activation for image reconstruction in the 0–1 range.



## Training Performance

### 1. Loss (MSE) vs Epoch Graph

<!--
🔁 How to URL-encode a path (like for mse graph): Replace space ( ) → %20
-->
<img src="Training%20Files%20and%20Logs/AOD-Net-Modified/History-and-Log-Files/Last%20Model%20and%20Training%20History%20Files%20(2025-05-13%20,%2002-39%20PM)/mse_vs_epoch_graph.png" alt="Loss Curve" width="550">

- **Training Loss**: Steadily decreasing trend, indicating proper learning.
- **Validation Loss**: Shows generalization with acceptable fluctuation.
- **Number of Epochs Trained**: 48

### 2. Evaluation Metrics
- **Loss Function**: Mean Squared Error (MSE)
- **Peak Signal-to-Noise Ratio (PSNR)**: `> 60 dB` on validation set
- **Structural Similarity Index (SSIM)**: `Around 0.76 - 0.78` (on test samples)



## 🖥️ Tech Stack

| Layer         | Technology                   |
|---------------|------------------------------|
| Frontend      | React, Tailwind CSS          |
| Backend API   | FastAPI, Python, TensorFlow  |
| Model         | AOD-Net (Refined Keras)      |
| Deep Learning | TensorFlow + Keras |
| Utilities  | Python, NumPy, Pillow, OpenCV |
| Frontend Deployment   | Vercel              |
| Backend Deployment    | Render               |


## API Endpoint
- **POST** /dehaze
- **Accepts**: .png, .jpg, .jpeg
- **Returns**: Dehazed image in image/png format

## Training Info
- **Dataset**: `RESIDE - ITS` (Indoor Training Set) || `Reside - SOTS` ,  `I-HAZE` & `D-HAZE` are removed for now
- **Training Platform**: `Google Colab` (T4 GPU)
- **Batch Size**: `1` (to preserve image quality)
- **Dynamic Image Size Support**: Accepts RGB images of any size `(None, None, 3)`.
- **Model saved as**: `aod_net_refined.keras`

## Team Contributions

| Name                                  | Contribution Areas                                          |
|-------------------------------------- | ---------------------------------------------------------- |
| **Sai Pritam Panda** *(Group Leader)* | Data cleaning, preprocessing, model architecture design, training, backend integration |
| **Debi Prasad Mahakud**               | Model implementation, backend setup, error handling        |
| **Prabhat Sharma**                    | Research analysis, debugging, frontend planning            |
| **Hrishikesh Swain**                  | Data collection, frontend prototyping                      |

## 📈 Project Results

- ✅ **Model Performance:**
  - Final Training Loss: **Consistently low MSE**
  - PSNR: **Above 60 dB** on validation set
  - SSIM: **Approximately 0.76 to 0.78**, indicating high structural similarity with ground truth

- 📊 **Training Insights:**
  - Smooth convergence with minimal overfitting
  - Training graph available: `mse_vs_epoch_graph.png`
  - Detailed training logs available in `training_log.txt`
  
- 📁 **Output Samples:**
  - Real image and dehazed image comparisons
  - Dehazed Images can be downloaded, if viewed in the website
  
- ⚡ **Deployment:**
  - Fast, real-time inference using FastAPI backend
  - Supports dynamic image input sizes for flexible usage

### 1. After AOD-NET Implementation

![](Images/Results/AOD-Net-Modifications/image1.png)

### 2. After AOD-NET Modification

![](Images/Results/Frontend/Comparations/img2.png)

### 3. Website

![](Images/Results/Frontend/image3.png)



## 📄 License
This project is released under the MIT License.
