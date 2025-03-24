import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# Configuration
KNOWN_FACES_DB = "known_faces_db"
RECOGNITION_THRESHOLD = 0.4  # Lower is more strict
MODEL_NAME = "VGG-Face"  # Other options: OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib, ArcFace
DETECTOR_BACKEND = "opencv"  # Other options: ssd, mtcnn, retinaface, mediapipe

# Initialize session state
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}
if 'capture' not in st.session_state:
    st.session_state.capture = None

def setup():
    """Create directories if they don't exist"""
    if not os.path.exists(KNOWN_FACES_DB):
        os.makedirs(KNOWN_FACES_DB)
        st.info(f"Created directory for known faces at: {KNOWN_FACES_DB}")

def add_known_face(image, identity):
    """Add a new face to the known faces database"""
    try:
        # Save the image
        img_path = os.path.join(KNOWN_FACES_DB, f"{identity}.jpg")
        image.save(img_path)
        
        # Verify the image contains a face
        face = DeepFace.detectFace(img_path, detector_backend=DETECTOR_BACKEND)
        st.session_state.known_faces[identity] = img_path
        st.success(f"Successfully added {identity} to known faces!")
        return True
    except Exception as e:
        st.error(f"Error adding face: {str(e)}")
        return False

def recognize_face(img_array):
    """Recognize a face from the known faces database"""
    try:
        # Save temporary image for recognition
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            Image.fromarray(img_array).save(temp_path)
        
        for identity, known_img_path in st.session_state.known_faces.items():
            result = DeepFace.verify(temp_path, known_img_path, 
                                    model_name=MODEL_NAME,
                                    detector_backend=DETECTOR_BACKEND)
            
            if result['verified'] and result['distance'] < RECOGNITION_THRESHOLD:
                os.unlink(temp_path)  # Clean up temp file
                return identity, result['distance']
        
        os.unlink(temp_path)  # Clean up temp file
        return "Unknown", 1.0
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return "No face detected", 1.0

def main():
    st.title("Real-Time Facial Recognition System")
    setup()
    
    # Sidebar for adding new faces
    st.sidebar.header("Face Database Management")
    new_face_name = st.sidebar.text_input("Enter name for new face")
    uploaded_file = st.sidebar.file_uploader("Upload face image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file and new_face_name:
        image = Image.open(uploaded_file)
        add_known_face(image, new_face_name)
    
    # Display known faces
    if st.session_state.known_faces:
        st.sidebar.subheader("Known Faces")
        for identity in st.session_state.known_faces.keys():
            st.sidebar.text(identity)
    
    # Real-time recognition
    st.header("Real-Time Face Recognition")
    run = st.checkbox('Start Recognition')
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    if run:
        if st.session_state.capture is None:
            st.session_state.capture = cv2.VideoCapture(0)
        
        while run:
            ret, frame = st.session_state.capture.read()
            if not ret:
                st.error("Failed to capture video")
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Recognize face
                identity, confidence = recognize_face(frame_rgb)
                status_text.text(f"Status: Recognizing... Current: {identity} ({confidence:.2f})")
                
                # Display result on frame
                cv2.putText(frame_rgb, f"{identity} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                FRAME_WINDOW.image(frame_rgb)
                
            except Exception as e:
                status_text.text(f"Status: Error - {str(e)}")
            
            # Check if user stopped the recognition
            if not run:
                break
    else:
        if st.session_state.capture is not None:
            st.session_state.capture.release()
            st.session_state.capture = None
        status_text.text("Status: Ready")

if __name__ == "__main__":
    main()
