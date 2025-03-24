import streamlit as st
from PIL import Image
import numpy as np
import os
from face_recognition import FaceRecognition

# Initialize system
fr = FaceRecognition()

# App UI
st.title("Face Recognition System")
st.write("Upload images to recognize faces")

# Configuration
st.sidebar.header("Settings")
fr.threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.4, 0.05)
fr.model_name = st.sidebar.selectbox(
    "Model",
    ["Facenet", "VGG-Face", "OpenFace"],
    index=0
)

# Known faces management
st.sidebar.header("Known Faces")
known_dir = "known_faces"
os.makedirs(known_dir, exist_ok=True)

uploaded_known = st.sidebar.file_uploader(
    "Upload known faces",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

for file in uploaded_known:
    with open(os.path.join(known_dir, file.name), "wb") as f:
        f.write(file.getbuffer())
    st.sidebar.success(f"Saved {file.name}")

if st.sidebar.button("Build Database"):
    fr.load_faces(known_dir)
    fr.build_db()
    st.sidebar.success("Database built!")

# Main image processing
uploaded_img = st.file_uploader(
    "Upload image to analyze",
    type=["jpg", "jpeg", "png"]
)

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Original Image", use_column_width=True)
    
    if st.button("Recognize Faces"):
        processed_img, faces = fr.recognize(img)
        
        if processed_img is not None:
            st.image(processed_img, caption="Processed Image", use_column_width=True)
            
            if faces:
                st.success("Found faces:")
                for face in faces:
                    st.write(f"- {face['name']} (confidence: {face['confidence']:.2f})")
            else:
                st.warning("No faces recognized")
            
            metrics = fr.get_metrics()
            st.subheader("Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{metrics['precision']:.2f}")
            col2.metric("Recall", f"{metrics['recall']:.2f}")
            col3.metric("F1 Score", f"{metrics['f1']:.2f}")
            
            st.write(f"Average Confidence: {metrics['avg_conf']:.2f}")
            st.write(f"True Positives: {metrics['true_pos']}")
            st.write(f"False Positives: {metrics['false_pos']}")
