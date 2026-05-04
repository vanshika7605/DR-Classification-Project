Diabetic Retinopathy Severity Classification using Deep Learning and Explainable AI
Overview

This project presents an automated Diabetic Retinopathy (DR) severity classification system using deep learning techniques on retinal fundus images. The proposed framework leverages transfer learning-based CNN architectures along with explainable AI methods to improve both diagnostic accuracy and clinical interpretability.

A comparative analysis was performed between ResNet50 and EfficientNet-B4, with ResNet50 achieving superior performance in DR severity grading.

Objectives
Develop an automated DR severity classification system
Improve retinal lesion visibility using advanced preprocessing
Compare deep learning architectures for DR grading
Integrate explainable AI for model transparency
Deploy the model through a Streamlit web application
Dataset

Dataset Used:
APTOS 2019 Blindness Detection Dataset

The dataset contains retinal fundus images categorized into five severity levels of Diabetic Retinopathy.

| Class | Severity Level   |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild             |
| 2     | Moderate         |
| 3     | Severe           |
| 4     | Proliferative DR |

Methodology
1. Preprocessing
Green Channel Extraction
CLAHE Enhancement
Image Resizing and Normalization
2. Data Augmentation
Rotation
Flipping
Brightness and Contrast Adjustment
Test Time Augmentation (TTA)
3. Model Training
ResNet50
EfficientNet-B4
Transfer Learning
AdamW Optimizer
Cosine Annealing Scheduler
4. Explainability
Grad-CAM Heatmap Visualization
5. Deployment
Streamlit-based Web Application
Results

| Model           | Accuracy | QWK Score |
| --------------- | -------- | --------- |
| ResNet50        | 81.09%   | 0.9116    |
| EfficientNet-B4 | 78.73%   | 0.9088    |

Best Performing Model

ResNet50 demonstrated:

better classification stability,
improved minority-class learning,
and superior overall diagnostic performance.

Features
Automated DR Severity Classification
Explainable AI using Grad-CAM
Retinal Image Enhancement using CLAHE
Real-time Prediction Interface
Confusion Matrix and Classification Report Visualization
Lightweight Streamlit Deployment
Tech Stack
Python
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Streamlit

Project Structure

DR_CLASSIFICATION_PROJECT/
│
├── app.py
├── class_names.py
├── README.md
├── Project_Report.pdf
├── .gitignore
│
├── assets/
│   ├── confusion matrix(ResNet).jpeg
│   ├── confusion matrix(Effnet).jpeg
│   ├── Final ResNetscore.jpeg
│   └── Final.jpeg
│
├── utils/
│   ├── preprocess.py
│   ├── predict.py
│   └── gradCAM.py
│
└── resnet_best_final.pth

Streamlit Deployment
Run the application locally using:
streamlit run app.py

Grad-CAM Explainability
Grad-CAM is used to visualize lesion-focused retinal regions influencing model predictions, improving transparency and clinical interpretability.

Future Scope
Vision Transformer-based DR grading
Lesion segmentation integration
Mobile and cloud deployment
Real-time hospital screening systems

## Project Report
[View Project Report](Project_Report.pdf)

Author
Vanshika Aggarwal
B.Tech — Electronics and Communication Engineering
Jaypee Institute of Information Technology
