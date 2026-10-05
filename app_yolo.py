import os
import sys
import tempfile
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- PAGE SETUP ---
st.set_page_config(page_title="YOLO Detection Hub", page_icon="🎯", layout="wide")
st.title("🎯 YOLO Object Detection Dashboard")
st.write("Upload an image or video to run predictions using your custom weights.")

# --- DYNAMIC PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

# Define structural fallback paths to your best.pt file
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
    os.path.join(BASE_DIR, "Project_5", "runs", "detect", "train", "weights", "best.pt"),
    os.path.join(BASE_DIR, "best.pt")
]

# Find the first path that actually exists
WEIGHT_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        WEIGHT_PATH = path
        break

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔧 Model Configurations")

if WEIGHT_PATH:
    st.sidebar.success(f"Loaded weights from:\n`...{WEIGHT_PATH[-40:]}`")
else:
    st.sidebar.error("Could not auto-detect `best.pt` file.")
    # Allow user to type/override path manually if auto-detect fails
    WEIGHT_PATH = st.sidebar.text_input("Enter exact absolute path to best.pt:", "")

# Confidence Threshold Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Minimum confidence score required to display a bounding box."
)

# File Type Selector
file_type = st.sidebar.radio("Select Media Type:", ["Image", "Video"])

# --- CORE LOGIC ---
if WEIGHT_PATH and os.path.exists(WEIGHT_PATH):
    # Cache the model loading step so it doesn't reload on every UI click
    @st.cache_resource
    def load_yolo_model(path):
        return YOLO(path)
    
    model = load_yolo_model(WEIGHT_PATH)

    # --- IMAGE HANDLING ---
    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("YOLO Prediction")
                with st.spinner("Analyzing image..."):
                    # Run prediction
                    results = model.predict(source=image, conf=conf_threshold)
                    
                    # Plot bounding boxes on the image array (BGR format)
                    res_plotted = results[0].plot()
                    
                    # Convert BGR back to RGB for Streamlit rendering
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, use_container_width=True)

    # --- VIDEO HANDLING ---
    elif file_type == "Video":
        uploaded_video = st.file_uploader("Choose a test video...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            # Streamlit needs a physical file path to read video frames using OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_video.read())
            
            st.subheader("Processing Video Feed")
            video_frame_placeholder = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Read and process frame-by-frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run prediction on single frame
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Continuously update the single image placeholder block
                video_frame_placeholder.image(res_rgb, use_container_width=True)
                
            cap.release()
            st.success("Video processing complete!")

else:
    st.warning("Please provide or fix your model weight path in the sidebar to start.")