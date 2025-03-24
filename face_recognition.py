import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import logging
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Using Google Drive path
KNOWN_FACES_DB = "/content/drive/MyDrive/images"
RECOGNITION_THRESHOLD = 0.4
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
FRAME_SKIP = 3

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
            st.info(f"Created directory: {KNOWN_FACES_DB}")
        
        # Load existing faces
        for file in os.listdir(KNOWN_FACES_DB):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                identity = os.path.splitext(file)[0]
                st.session_state.known_faces[identity] = os.path.join(KNOWN_FACES_DB, file)
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        st.error(f"Database error: {str(e)}")

def add_known_face(image, identity):
    """Add a new face to Google Drive database"""
    try:
        img_path = os.path.join(KNOWN_FACES_DB, f"{identity}.jpg")
        image.save(img_path)
        
        # Verify the image contains a face
        face = DeepFace.detectFace(img_path, detector_backend=DETECTOR_BACKEND)
        st.session_state.known_faces[identity] = img_path
        st.success(f"Added {identity} successfully!")
        return True
    except Exception as e:
        logger.error(f"Add face error: {str(e)}")
        st.error(f"Error: {str(e)}")
        if 'img_path' in locals() and os.path.exists(img_path):
            os.remove(img_path)
        return False

def recognize_face(img_array):
    """Recognize face from database"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            Image.fromarray(img_array).save(tmp_file.name)
            
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
    st.title("🔍 Face Recognition System")
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
    
    # Check environment
    try:
        from google.colab import drive
        IS_COLAB = True
    except:
        IS_COLAB = False
    
    if IS_COLAB:
        st.warning("Colab detected - using image upload mode")
        st.header("Image Recognition")
        uploaded_img = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_img:
            try:
                image = Image.open(uploaded_img)
                st.image(image, use_column_width=True)
                
                with st.spinner("Analyzing..."):
                    identity, confidence = recognize_face(np.array(image))
                    st.success(f"Result: {identity} ({1-confidence:.2f} confidence)")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.header("Live Camera")
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
        status_text = st.empty()
        
        if run:
            if st.session_state.capture is None:
                st.session_state.capture = cv2.VideoCapture(0)
                st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while run:
                ret, frame = st.session_state.capture.read()
                if not ret:
                    st.error("Camera error")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                st.session_state.frame_count += 1
                if st.session_state.frame_count % FRAME_SKIP == 0:
                    identity, confidence = recognize_face(frame_rgb)
                    status_text.text(f"Status: {identity} ({1-confidence:.2f})")
                    
                    cv2.putText(frame_rgb, f"{identity}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"{1-confidence:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                FRAME_WINDOW.image(frame_rgb)
        else:
            if st.session_state.capture is not None:
                st.session_state.capture.release()
                st.session_state.capture = None

if __name__ == "__main__":
    # Colab dependency setup
    if os.path.exists("/content"):
        import subprocess
        subprocess.run(['apt-get', 'update'])
        subprocess.run(['apt-get', 'install', '-y', 'libgl1-mesa-glx'])
        subprocess.run(['pip', 'install', 'opencv-python-headless'])
    
    main()
