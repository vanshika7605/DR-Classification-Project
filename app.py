import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from utils.gradCAM import generate_gradcam
# Import from your custom modules
from utils.predict import predict_image
from class_names import CLASS_NAMES

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DR Severity Classifier", 
    
    layout="wide"
)

# --- SIDEBAR: TECHNICAL DETAILS ---
with st.sidebar:
    st.header(" Technical Highlights")
    st.markdown("""
    * **Model Backbone:** ResNet / EfficientNet
    * **Framework:** PyTorch & `timm`
    * **Preprocessing:** CLAHE, Green Channel Enhancement
    * **Optimizer:** AdamW
    * **Scheduler:** Cosine Annealing
    * **Advanced Techniques:** * Transfer Learning
        * Threshold Optimization
        * Test Time Augmentation (TTA)
    """)
    st.markdown("---")
    st.info("Upload a retinal fundus image in the main panel to begin diagnosis.")

# --- MAIN UI ---
st.title(" Diabetic Retinopathy Severity Classification")

# Create tabs to organize the Predict and Dashboard sections
tab_predict, tab_performance = st.tabs([" Diagnosis & Explainability", "Model Performance Dashboard"])

with tab_predict:
    st.header("Image Upload & Prediction")
    uploaded_file = st.file_uploader("Upload a retinal fundus image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Fundus Image", use_container_width=True)

        with st.spinner("Analyzing fundus image..."):
            # Call your backend PyTorch prediction function
            results = predict_image(image)

        with col2:
            st.subheader("Prediction Results")
            
            # Dynamic text coloring based on severity index
            colors = ["green", "blue", "orange", "red", "darkred"]
            st.markdown(f"### Predicted Grade: **<span style='color:{colors[results['class_index']]}'>{results['class_name']}</span>**", unsafe_allow_html=True)
            
            # Display the continuous severity score
            st.info(f"**Calculated Severity Score:** {results['continuous_score']:.2f} / 4.0")
            
            with st.expander("How was this calculated?"):
                st.markdown(f"Because DR is progressive, this model uses a weighted average of all probabilities. The continuous score of **{results['continuous_score']:.2f}** falls into the threshold for **{results['class_name']}**.")

            st.subheader("Class Probabilities")
            probs = results['probabilities']
            
            # --- THIS WAS THE MISSING LINE ---
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Plotting the probability distribution
            ax.bar(list(CLASS_NAMES.values()), probs, color='#4A90E2')
            ax.set_ylabel("Probability")
            ax.set_xlabel("DR Severity Grade")
            ax.set_ylim(0, 1) # Lock Y-axis to 0-1 (0-100%)
            st.pyplot(fig)
        
        st.markdown("---")
        st.header("Grad-CAM Explainability")
        st.markdown("Grad-CAM highlights retinal regions (like hemorrhages, microaneurysms, or exudates) that most influenced the model's prediction.")
        
        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                # Generate heatmap specifically for the predicted class
                heatmap_image = generate_gradcam(image, target_class_index=results['class_index'])
                
                # Display side-by-side comparison
                cam_col1, cam_col2 = st.columns(2)
                with cam_col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with cam_col2:
                    st.image(heatmap_image, caption=f"Grad-CAM (Focusing on: {results['class_name']})", use_container_width=True)
                    
            except Exception as e:
                st.error(f"Failed to generate Grad-CAM visualization. Error: {e}")

with tab_performance:
    st.header("Model Performance Dashboard")
    st.markdown("Evaluation metrics evaluated on the 3662-image test set.")

    # Create sub-tabs to compare models side-by-side
    sub_tab_resnet, sub_tab_effnet = st.tabs(["ResNet Results", "EfficientNet Results"])

    with sub_tab_resnet:
        st.subheader("ResNet Test Metrics")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Accuracy", "81.09%")
        metric2.metric("QWK", "0.9116")
        metric3.metric("Weighted F1-Score", "0.8027")
        metric4.metric("Weighted Precision", "0.8217")

        st.code("""
                      precision    recall  f1-score   support

                   0     0.9673    0.9815    0.9744       271
                   1     0.7500    0.3214    0.4500        56
                   2     0.7299    0.8467    0.7840       150
                   3     0.3469    0.5862    0.4359        29
                   4     0.6429    0.4091    0.5000        44

            accuracy                         0.8109       550
           macro avg     0.6874    0.6290    0.6288       550
        weighted avg     0.8217    0.8109    0.8027       550
        """)

    with sub_tab_effnet:
        st.subheader("EfficientNet Test Metrics")
        metric1_e, metric2_e, metric3_e, metric4_e = st.columns(4)
        metric1_e.metric("Accuracy", "78.73%")
        metric2_e.metric("QWK", "0.9088")
        metric3_e.metric("Weighted F1-Score", "0.7757")
        metric4_e.metric("Weighted Precision", "0.8188")

        st.code("""
                      precision    recall  f1-score   support

                   0     0.9711    0.9926    0.9818       271
                   1     0.7857    0.1964    0.3143        56
                   2     0.7126    0.7933    0.7508       150
                   3     0.2857    0.6897    0.4040        29
                   4     0.6364    0.3182    0.4242        44

            accuracy                         0.7873       550
           macro avg     0.6783    0.5980    0.5750       550
        weighted avg     0.8188    0.7873    0.7757       550
        """)

    st.markdown("---")
    st.subheader("Model Comparison: Confusion Matrices")
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.markdown("#### **ResNet**")
        # Safely check if file exists before telling Streamlit to draw it
        if os.path.exists("assets/confusion matrix(ResNet).jpeg"):
            st.image("assets/confusion matrix(ResNet).jpeg", use_container_width=True)
        else:
            st.warning("⚠️ ResNet confusion matrix not found in 'assets/'.")

    with col_metrics2:
        st.markdown("#### **EfficientNet**")
        # Safely check if file exists before telling Streamlit to draw it
        if os.path.exists("assets/confusion matrix(Effnet).jpeg"):
            st.image("assets/confusion matrix(Effnet).jpeg", use_container_width=True)
        else:
            st.warning("⚠️ EfficientNet confusion matrix not found in 'assets/'.")