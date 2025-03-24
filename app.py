import streamlit as st
import numpy as np
from PIL import Image
import os
import time
from face_recognition_system import FaceRecognitionSystem

# Initialize the face recognition system
fr_system = FaceRecognitionSystem()

# Streamlit app layout
st.title("Face Recognition System")
st.write("Upload images to recognize faces using DeepFace")

# Sidebar controls
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Recognition Threshold", 0.0, 1.0, 0.4, 0.05)
fr_system.threshold = threshold

model_options = ["Facenet", "VGG-Face", "OpenFace", "DeepFace"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)
fr_system.model_name = selected_model

# Initialize known faces
known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

# Upload new known faces
st.sidebar.header("Add Known Faces")
uploaded_known_faces = st.sidebar.file_uploader(
    "Upload known face images (jpg, png)", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png'],
    key="known_faces"
)

for uploaded_file in uploaded_known_faces:
    with open(os.path.join(known_faces_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {uploaded_file.name}")

if st.sidebar.button("Rebuild Face Database"):
    fr_system.load_known_faces(known_faces_dir)
    fr_system.build_face_database()
    st.sidebar.success("Face database rebuilt!")

# Main image upload
st.header("Upload Image for Recognition")
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    key="test_image"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    if st.button("Recognize Faces"):
        with st.spinner("Processing image..."):
            # Recognize faces
            processed_image, recognized_faces = fr_system.recognize_faces_in_image(image)
            
            if processed_image is not None:
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                # Display recognition results
                if recognized_faces:
                    st.success("Faces recognized successfully!")
                    for face in recognized_faces:
                        st.write(f"- {face['name']} (confidence: {face['confidence']:.2f})")
                else:
                    st.warning("No faces recognized or no matches found")
                
                # Display metrics
                metrics = fr_system.calculate_metrics()
                st.subheader("Performance Metrics")
                cols = st.columns(3)
                cols[0].metric("Precision", f"{metrics['precision']:.2f}")
                cols[1].metric("Recall", f"{metrics['recall']:.2f}")
                cols[2].metric("F1 Score", f"{metrics['f1_score']:.2f}")
                
                st.write(f"Average Confidence: {metrics['average_confidence']:.2f}")
                st.write(f"True Positives: {metrics['true_positives']}")
                st.write(f"False Positives: {metrics['false_positives']}")
            else:
                st.error("Failed to process the image")
