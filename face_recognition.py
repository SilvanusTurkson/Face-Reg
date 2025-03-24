import streamlit as st
import numpy as np
from PIL import Image
import os
import tempfile
import logging
from deepface import DeepFace

# Ensure OpenCV-Headless is installed
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(['apt-get', 'update'])
    subprocess.run(['apt-get', 'install', '-y', 'libgl1-mesa-glx'])
    subprocess.run(['pip', 'install', 'opencv-python-headless'])
    import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Using your Google Drive path
KNOWN_FACES_DB = "/content/drive/MyDrive/images"  # Google Drive path
RECOGNITION_THRESHOLD = 0.4
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
FRAME_SKIP = 3  # Process every 3rd frame to reduce load

# Initialize session state
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}
if 'capture' not in st.session_state:
    st.session_state.capture = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def setup():
    """Initialize the known faces database from Google Drive"""
    try:
        # Mount Google Drive if in Colab
        if os.path.exists("/content"):
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")

        if not os.path.exists(KNOWN_FACES_DB):
            os.makedirs(KNOWN_FACES_DB)
            st.info(f"Created directory for known faces at: {KNOWN_FACES_DB}")
        else:
            for file in os.listdir(KNOWN_FACES_DB):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    identity = os.path.splitext(file)[0]
                    st.session_state.known_faces[identity] = os.path.join(KNOWN_FACES_DB, file)
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        st.error(f"Failed to initialize database: {str(e)}")

def add_known_face(image, identity):
    """Add a new face to the known faces database in Google Drive"""
    try:
        img_path = os.path.join(KNOWN_FACES_DB, f"{identity}.jpg")
        image.save(img_path)

        # Verify the image contains a face
        face = DeepFace.detectFace(img_path, detector_backend=DETECTOR_BACKEND)
        st.session_state.known_faces[identity] = img_path
        st.success(f"Successfully added {identity} to known faces!")
        return True
    except Exception as e:
        logger.error(f"Add face error: {str(e)}")
        st.error(f"Error adding face: {str(e)}")
        if 'img_path' in locals() and os.path.exists(img_path):
            os.remove(img_path)
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
        logger.error(f"Recognition error: {str(e)}")
        return "No face detected", 1.0

def main():
    st.title("🔍 Google Drive Facial Recognition System")
    setup()

    # Sidebar for management
    st.sidebar.header("Face Database")

    # Add new face
    with st.sidebar.expander("Add New Face"):
        new_face_name = st.text_input("Name")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file and new_face_name:
            try:
                image = Image.open(uploaded_file)
                if st.button("Add Face"):
                    add_known_face(image, new_face_name)
            except Exception as e:
                st.error(f"Invalid image: {str(e)}")

    # Display known faces
    if st.session_state.known_faces:
        st.sidebar.subheader("Registered Faces")
        for identity in sorted(st.session_state.known_faces.keys()):
            st.sidebar.text(f"• {identity}")

    # Main content - Image Upload mode
    st.header("Image Recognition")
    uploaded_img = st.file_uploader("Upload an image to recognize", type=['jpg', 'png', 'jpeg'])

    if uploaded_img:
        try:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing..."):
                identity, confidence = recognize_face(np.array(image))
                st.success(f"Recognized as: {identity} (Confidence: {1 - confidence:.2f})")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
