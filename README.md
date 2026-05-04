Diabetic Retinopathy Severity Classification using Deep Learning and Explainable AI
Overview

This project presents an automated Diabetic Retinopathy (DR) severity classification system using deep learning techniques on retinal fundus images. The system utilizes transfer learning-based CNN architectures along with explainable AI methods to improve both diagnostic accuracy and interpretability.

The project compares the performance of ResNet50 and EfficientNet-B4 for multi-class DR severity grading and integrates Grad-CAM visualization for clinical explainability.

Features
Multi-class Diabetic Retinopathy Severity Classification
Transfer Learning using ResNet50 and EfficientNet-B4
Green Channel Extraction and CLAHE preprocessing
Test Time Augmentation (TTA)
AdamW Optimizer with Cosine Annealing Scheduler
Grad-CAM based Explainable AI
Confusion Matrix and Classification Report Visualization
Streamlit Web Application Deployment
Dataset

Dataset used:
APTOS 2019 Blindness Detection Dataset

The dataset contains retinal fundus images categorized into five DR severity classes:

Class	Severity Level
0	No DR
1	Mild
2	Moderate
3	Severe
4	Proliferative DR
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
Model	Accuracy	QWK Score
ResNet50	81.09%	0.9116
EfficientNet-B4	78.73%	0.9088
Best Performing Model

ResNet50 demonstrated superior classification stability and better minority-class performance.

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
├── .gitignore
│
├── assets/
│   ├── confusion matrix(ResNet).jpeg
│   ├── confusion matrix(Effnet).jpeg
│   ├── Final ResNetscore.jpeg
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