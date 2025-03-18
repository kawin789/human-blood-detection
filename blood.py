import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import random
from sklearn.preprocessing import StandardScaler
import time

# Set page configuration
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="wide"
)

# Define the blood groups
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    # Convert to grayscale if it's RGB
    if len(img_array.shape) == 3:
        # Simple conversion to grayscale by averaging channels
        img_array = np.mean(img_array, axis=2)
    # Flatten the image
    img_flat = img_array.flatten()
    # Normalize
    scaler = StandardScaler()
    img_normalized = scaler.fit_transform(img_flat.reshape(-1, 1))
    return img_normalized

# Function to simulate predictions (since we don't have a real model)
def simulate_blood_group_prediction(image_features):
    # In a real application, we would use a trained model here
    # For demonstration, we'll generate random predictions
    random.seed(int(np.sum(image_features[:100])))  # Use image data to seed for consistent results
    
    # Create a simulated prediction
    simulated_prediction = np.zeros(8)
    max_index = random.randint(0, 7)
    simulated_prediction[max_index] = 0.7 + random.random() * 0.25  # Main prediction 70-95%
    
    # Distribute the remaining probability
    remaining = 1.0 - simulated_prediction[max_index]
    for i in range(8):
        if i != max_index:
            simulated_prediction[i] = remaining * random.random()
    
    # Normalize to ensure sum is 1.0
    simulated_prediction = simulated_prediction / simulated_prediction.sum()
    
    return simulated_prediction

# Create a demo function to simulate learning progress
def train_model_demo():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulating model training progress
    for i in range(1, 101):
        progress_bar.progress(i)
        if i < 30:
            status_text.text(f"Loading and preprocessing training data: {i}%")
        elif i < 60:
            status_text.text(f"Training model: {i}%")
        elif i < 90:
            status_text.text(f"Validating model performance: {i}%")
        else:
            status_text.text(f"Finalizing model: {i}%")
        
        if i % 10 == 0:
            # Show a simulated accuracy metric improvement
            accuracy = min(0.5 + (i/200), 0.95)
            st.metric("Training Accuracy", f"{accuracy:.2%}")
            
        # Slow down the simulation
        time.sleep(0.05)
    
    # Training complete
    status_text.text("Model training complete!")
    st.balloons()

# Create a simple mockup of what our model architecture would look like
def display_model_architecture():
    st.code("""
# Simplified model architecture (if using scikit-learn)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        max_depth=20,
        random_state=42
    ))
])
    """)

# Main Streamlit app
def main():
    st.title("🩸 Blood Group Detection")
    st.write("""
    This application simulates a blood group detection system using image analysis.
    
    Upload an image of a blood sample slide to detect the blood group.
    """)
    
    # Sidebar
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose Mode", ["About", "Train Model", "Test Prediction"])
    
    # About section
    if app_mode == "About":
        st.markdown("""
        ## About Blood Group Detection
        
        Blood type (or blood group) is determined by the presence or absence of certain antigens 
        on the surface of red blood cells. There are four main blood groups (A, B, AB, and O) 
        determined by the presence or absence of the A and B antigens. 
        
        Additionally, there's the Rh factor (positive or negative), which indicates whether 
        the Rh antigen is present on the red blood cells.
        
        ### How the System Works
        
        1. **Image Processing**: Blood sample images are preprocessed and normalized.
        2. **Feature Extraction**: Key features are extracted from the image.
        3. **Classification**: Based on the extracted features, the model classifies the image into one of eight blood groups.
        
        ### Applications
        - Quick blood type determination in emergency situations
        - Blood donation compatibility checks
        - Medical research
        - Educational purposes
        """)
        
        # Display a sample blood group compatibility chart
        st.subheader("Blood Group Compatibility Chart")
        compatibility_data = {
            'Blood Group': blood_groups,
            'Can Donate To': [
                'A+, AB+', 
                'A+, A-, AB+, AB-', 
                'B+, AB+', 
                'B+, B-, AB+, AB-',
                'AB+',
                'AB+, AB-',
                'O+, A+, B+, AB+',
                'Everyone'
            ],
            'Can Receive From': [
                'A+, A-, O+, O-',
                'A-, O-',
                'B+, B-, O+, O-',
                'B-, O-',
                'Everyone',
                'A-, B-, AB-, O-',
                'O+, O-',
                'O-'
            ]
        }
        
        # Create a DataFrame for display
        import pandas as pd
        compatibility_df = pd.DataFrame(compatibility_data)
        st.table(compatibility_df)
        
    # Train Model section
    elif app_mode == "Train Model":
        st.header("Model Training")
        st.write("""
        This section demonstrates the process of training a machine learning model on blood group images.
        In a production application, this would connect to a dataset of labeled blood sample images.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            display_model_architecture()
        
        with col2:
            st.subheader("Training Parameters")
            st.write("""
            - **Algorithm**: Random Forest Classifier
            - **Feature Extraction**: PCA
            - **Number of Trees**: 100
            - **Max Depth**: 20
            - **Cross-Validation**: 5-fold
            """)
        
        st.subheader("Training Progress")
        if st.button("Start Training Demo"):
            train_model_demo()
    
    # Test Prediction section
    elif app_mode == "Test Prediction":
        st.header("Blood Group Prediction")
        st.write("Upload an image of a blood sample slide to predict the blood group.")
        
        # Updated file uploader to include BMP format
        uploaded_file = st.file_uploader("Choose a blood sample image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Blood Sample", width=400)
            st.write("")
            
            # Create a placeholder for the prediction
            prediction_placeholder = st.empty()
            prediction_placeholder.write("Analyzing image...")
            
            # Preprocess image
            features = preprocess_image(image)
            
            # Simulate processing time
            time.sleep(2)
            
            # Get simulated prediction
            simulated_prediction = simulate_blood_group_prediction(features)
            
            # Clear the placeholder
            prediction_placeholder.empty()
            
            # Display the results
            st.subheader("Prediction Results")
            max_index = np.argmax(simulated_prediction)
            predicted_label = blood_groups[max_index]
            st.markdown(f"### Detected Blood Group: **{predicted_label}**")
            
            # Create a bar chart of the predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(blood_groups, simulated_prediction * 100)
            
            # Highlight the predicted class
            bars[max_index].set_color('red')
            
            ax.set_title('Blood Group Prediction Confidence (%)')
            ax.set_ylabel('Confidence (%)')
            ax.set_xlabel('Blood Group')
            ax.set_ylim(0, 100)
            
            for i, v in enumerate(simulated_prediction):
                ax.text(i, v * 100 + 2, f"{v*100:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Display additional information about the detected blood group
            st.subheader(f"Information about {predicted_label} Blood Group")
            
            blood_info = {
                "A+": """
                - Can donate to: A+, AB+
                - Can receive from: A+, A-, O+, O-
                - Population frequency: ~35.7% (varies by region)
                """,
                "A-": """
                - Can donate to: A+, A-, AB+, AB-
                - Can receive from: A-, O-
                - Population frequency: ~6.3% (varies by region)
                """,
                "B+": """
                - Can donate to: B+, AB+
                - Can receive from: B+, B-, O+, O-
                - Population frequency: ~8.5% (varies by region)
                """,
                "B-": """
                - Can donate to: B+, B-, AB+, AB-
                - Can receive from: B-, O-
                - Population frequency: ~1.5% (varies by region)
                """,
                "AB+": """
                - Can donate to: AB+ only
                - Can receive from: All blood types (universal recipient)
                - Population frequency: ~3.4% (varies by region)
                """,
                "AB-": """
                - Can donate to: AB+, AB-
                - Can receive from: A-, B-, AB-, O-
                - Population frequency: ~0.6% (varies by region)
                """,
                "O+": """
                - Can donate to: O+, A+, B+, AB+
                - Can receive from: O+, O-
                - Population frequency: ~37.4% (varies by region)
                """,
                "O-": """
                - Can donate to: All blood types (universal donor)
                - Can receive from: O- only
                - Population frequency: ~6.6% (varies by region)
                """
            }
            
            st.markdown(blood_info[predicted_label])

if __name__ == "__main__":
    main()