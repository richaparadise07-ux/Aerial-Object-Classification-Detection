# --- 1. CORE SYSTEM IMPORTS FIRST ---
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import streamlit as st  # Import streamlit here alongside other dependencies

# --- 2. INITIALIZE PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT CODE CALLED) ---
st.set_page_config(page_title="Sky Surveillance - ResNet50", page_icon="🦅", layout="centered")

# --- 3. DASHBOARD TITLES ---
st.title("🦅 ResNet50 Space Entity Classifier")
st.markdown("Upload sky imagery to classify airspace entities using your custom-trained **ResNet50 Transfer Learning Model**.")

# --- 4. CLASS DEFINITIONS ---
# Make sure these match the exact folder order of your training dataset!
CLASS_NAMES = ["Bird", "Drone"] 

# --- 5. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "deployment_best_model.pth")

# --- 6. MODEL LOADING ENGINE ---
@st.cache_resource
def load_resnet_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 CRITICAL ERROR: Weight file not found at: {MODEL_PATH}")
        st.stop()
        
    # Recreate the precise ResNet50 structure layout
    model = models.resnet50(weights=None)
    
    # Match the final dense layer to your exact number of classes (2: Bird vs Drone)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))
    
    # Inject your saved mathematical weights dictionary matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Lock model into Evaluation Mode
    model.eval()
    return model, device

model, device = load_resnet_model()

# --- 7. PRE-PROCESSING TRANSFORM PIPELINE ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# --- 8. FILE UPLOADER ENGINE ---
uploaded_file = st.file_uploader("Select an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Input Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Processing telemetry matrix via ResNet50..."):
        # Apply transforms and simulate a batch dimension
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Calculate across the accurate batch dimension (dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Extract highest scoring class mapping parameters (using dim=1 for the batch output matrix)
            confidence_score, class_id = torch.max(probabilities, dim=1)
            
            predicted_class = CLASS_NAMES[class_id.item()]
            confidence_percentage = confidence_score.item() * 100

    with col2:
        st.subheader("Model Diagnostic Output")
        
        if "drone" in predicted_class.lower():
            st.error(f"🛸 **DRONE DETECTED**\n\n**{predicted_class.upper()}** Intrusion Detected!")
            st.metric(label="Target Match Confidence", value=f"{confidence_percentage:.2f}%")
        else:
            st.success(f"🕊️ **ENVIRONMENTAL MONITOR**\n\n**{predicted_class.upper()}** Tracked in Airspace.")
            st.metric(label="Target Match Confidence", value=f"{confidence_percentage:.2f}%")

    # --- ADVANCED RAW PROBABILITIES CHART ---
    st.subheader("📊 Class Confidence Breakdown Log")
    for idx, name in enumerate(CLASS_NAMES):
        prob = probabilities[0][idx].item() * 100  # Extract from batch 0 cleanly
        st.write(f"**{name}**: {prob:.2f}%")
        st.progress(int(prob))