The application will automatically open in your default web browser at `http://localhost:8501`.

## 🛠️ Technical Implementation Details
* **Framework:** PyTorch & `timm`
* **Optimizer:** AdamW
* **Scheduler:** Cosine Annealing
* **Explainability:** `pytorch-grad-cam` library targeting the final convolutional blocks (`model.layer4[-1]` for ResNet).
* **Evaluation:** Quadratic Weighted Kappa (QWK) was heavily prioritized over raw accuracy to account for the ordinal nature of disease progression.

## ⚠️ Disclaimer
This application is designed for educational and research purposes only. It is not intended for use inI can certainly help you write a README for your Diabetic Retinopathy Classification project! A strong README is crucial for making your project look professional and helping others (or future you) understand how to run it.

Based on the code and structures we've built, here is a comprehensive template for your `README.md` file. You can copy this directly into your project folder and adjust any specific details.

***

# 👁️ Diabetic Retinopathy Severity Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview
This project provides an end-to-end deep learning pipeline and interactive web application for classifying the severity of Diabetic Retinopathy (DR) from retinal fundus images.

The application utilizes state-of-the-art convolutional neural networks (**ResNet50** and **EfficientNet-B4**) to categorize images into one of five severity grades (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative). It features a robust Streamlit dashboard for real-time predictions, Grad-CAM explainability, and comprehensive model performance metrics.

## ✨ Key Features
* **Dual-Model Support:** Compares performance between PyTorch `timm` implementations of ResNet50 and EfficientNet-B4.
* **Advanced Preprocessing:** Utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) applied specifically to the Green Channel to enhance the visibility of blood vessels and microaneurysms.
* **Ordinal Regression:** Employs Test Time Augmentation (TTA) and threshold optimization to calculate a continuous severity score, significantly improving the Quadratic Weighted Kappa (QWK).
* **Explainable AI (XAI):** Integrates Grad-CAM to generate heatmaps, highlighting the specific retinal regions that most influenced the model's prediction.
* **Interactive Dashboard:** Built with Streamlit, providing a clean UI for image upload, prediction visualization, and side-by-side model metric comparisons.

## 📊 Model Performance (Test Set: 550 Images)

| Metric | ResNet50 | EfficientNet-B4 |
| :--- | :--- | :--- |
| **Accuracy** | 81.09% | 78.73% |
| **QWK** | **0.9116** | 0.9088 |
| **Weighted F1-Score** | 0.8027 | 0.7757 |

*Note: ResNet50 was selected as the primary deployment model due to its marginally superior QWK and general accuracy on the test set.*

## 📁 Project Structure
```text
DR_Classification_Project/
├── app.py                     # Main Streamlit application file
├── resnet_best_final.pth      # Trained PyTorch ResNet50 weights
├── class_names.py             # Dictionary mapping class indices to severity names
├── requirements.txt           # Python dependencies
├── assets/                    # Static assets for the dashboard
│   ├── confusion matrix(ResNet).jpeg
│   ├── confusion matrix(Effnet).jpeg
│   └── training_curve.png
└── utils/                     
    ├── preprocess.py          # CLAHE and image transformation logic
    ├── predict.py             # PyTorch inference, TTA, and thresholding logic
    └── gradCAM.py             # Grad-CAM heatmap generation
🚀 Installation & Setup
1. Clone the repository
(If applicable, insert your git clone command here)

2. Create a virtual environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Add the Model Weights
Ensure your trained model weights file (resnet_best_final.pth) is placed in the root directory.

💻 Running the Application
To launch the interactive dashboard, run the following command in your terminal from the project root:

Bash
streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

🛠️ Technical Implementation Details
Framework: PyTorch & timm

Optimizer: AdamW

Scheduler: Cosine Annealing

Explainability: pytorch-grad-cam library targeting the final convolutional blocks (model.layer4[-1] for ResNet).

Evaluation: Quadratic Weighted Kappa (QWK) was heavily prioritized over raw accuracy to account for the ordinal nature of disease progression.
