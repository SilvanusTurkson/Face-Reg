import streamlit as st
import numpy as np
from PIL import Image
import os
import tempfile
import logging

# First try to import with headless OpenCV
try:
    import cv2
    from deepface import DeepFace
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "deepface"])
    import cv2
    from deepface import DeepFace

# Configuration
KNOWN_FACES_DB = "known_faces_db"  # Local directory for Streamlit Cloud
RECOGNITION_THRESHOLD = 0.4
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"

# Initialize session state
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}

def setup():
    """Create directories if they don't exist"""
    if not os.path.exists(KNOWN_FACES_DB):
        os.makedirs(KNOWN_FACES_DB)

def add_known_face(image, identity):
    """Add a new face to the known faces database"""
    try:
        img_path = os.path.join(KNOWN_FACES_DB, f"{identity}.jpg")
        image.save(img_path)
        
        # Verify the image contains a face
        face = DeepFace.detectFace(img_path, detector_backend=DETECTOR_BACKEND)
        st.session_state.known_faces[identity] = img_path
        st.success(f"Successfully added {identity}!")
        return True
    except Exception as e:
        st.error(f"Error adding face: {str(e)}")
        return False

def recognize_face(img_array):
    """Recognize a face from the known faces database"""
    try:
        # Save temporary image for recognition
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            Image.fromarray(img_array).save(tmp_file.name)
            
            # Find the most similar face
            dfs = DeepFace.find(
                img_path=tmp_file.name,
                db_path=KNOWN_FACES_DB,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
            
            if len(dfs) > 0 and len(dfs[0]) > 0:
                best_match = dfs[0].iloc[0]
                if best_match['distance'] < RECOGNITION_THRESHOLD:
                    identity = os.path.splitext(os.path.basename(best_match['identity']))[0]
                    return identity, best_match['distance']
        
        return "Unknown", 1.0
    except Exception as e:
        return "No face detected", 1.0

def main():
    st.title("Face Recognition System")
    setup()
    
    # Sidebar for management
    st.sidebar.header("Face Database")
    
    with st.sidebar.expander("Add New Face"):
        new_face_name = st.text_input("Name")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file and new_face_name:
            image = Image.open(uploaded_file)
            if st.button("Add Face"):
                add_known_face(image, new_face_name)
    
    if st.session_state.known_faces:
        st.sidebar.subheader("Registered Faces")
        for identity in sorted(st.session_state.known_faces.keys()):
            st.sidebar.text(f"• {identity}")
    
    # Image recognition option
    st.header("Image Recognition")
    uploaded_img = st.file_uploader("Upload an image to recognize", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_img:
        try:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Analyzing..."):
                identity, confidence = recognize_face(np.array(image))
                st.success(f"Recognized as: {identity} (Confidence: {1-confidence:.2f})")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
