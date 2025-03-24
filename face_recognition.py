import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# Configuration
KNOWN_FACES_DB = "/content/drive/MyDrive/images"
RECOGNITION_THRESHOLD = 0.4  # Lower is more strict
MODEL_NAME = "VGG-Face"  # Other options: OpenFace, Facenet, Facenet512, DeepFace, DeepID, Dlib, ArcFace
DETECTOR_BACKEND = "opencv"  # Other options: ssd, mtcnn, retinaface, mediapipe

# Initialize session state
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}
if 'capture' not in st.session_state:
    st.session_state.capture = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def setup():
    """Initialize the known faces database"""
    if not os.path.exists(KNOWN_FACES_DB):
        os.makedirs(KNOWN_FACES_DB)
        st.info(f"Created directory for known faces at: {KNOWN_FACES_DB}")
    else:
        # Load existing faces
        for file in os.listdir(KNOWN_FACES_DB):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                identity = os.path.splitext(file)[0]
                st.session_state.known_faces[identity] = os.path.join(KNOWN_FACES_DB, file)

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
        if os.path.exists(img_path):
            os.remove(img_path)
        return False

def recognize_face(img_array):
    """Recognize a face from the known faces database"""
    try:
        # Save temporary image for recognition
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            Image.fromarray(img_array).save(temp_path)
        
        # Find the most similar face
        dfs = DeepFace.find(img_path=temp_path, 
                           db_path=KNOWN_FACES_DB,
                           model_name=MODEL_NAME,
                           detector_backend=DETECTOR_BACKEND,
                           enforce_detection=False,
                           silent=True)
        
        if len(dfs) > 0 and len(dfs[0]) > 0:
            best_match = dfs[0].iloc[0]
            if best_match['distance'] < RECOGNITION_THRESHOLD:
                identity = os.path.splitext(os.path.basename(best_match['identity']))[0]
                os.unlink(temp_path)
                return identity, best_match['distance']
        
        os.unlink(temp_path)
        return "Unknown", 1.0
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return "No face detected", 1.0

def main():
    st.title("🔍 Real-Time Facial Recognition System")
    setup()
    
    # Sidebar for management
    st.sidebar.header("Face Database")
    
    # Add new face
    with st.sidebar.expander("Add New Face"):
        new_face_name = st.text_input("Name")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file and new_face_name:
            image = Image.open(uploaded_file)
            if st.button("Add Face"):
                add_known_face(image, new_face_name)
    
    # Display known faces
    if st.session_state.known_faces:
        st.sidebar.subheader("Registered Faces")
        for identity in sorted(st.session_state.known_faces.keys()):
            st.sidebar.text(f"• {identity}")
    
    # Main content
    st.header("Live Recognition")
    
    # Camera settings
    col1, col2 = st.columns(2)
    with col1:
        run = st.checkbox('Start Camera', key='run_recognition')
    with col2:
        show_fps = st.checkbox('Show FPS', value=True)
    
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    fps_text = st.empty()
    
    if run:
        if st.session_state.capture is None:
            st.session_state.capture = cv2.VideoCapture(0)
            st.session_state.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_time = 0
        while run and st.session_state.capture.isOpened():
            ret, frame = st.session_state.capture.read()
            if not ret:
                st.error("Failed to capture video")
                break
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.CAP_PROP_CONVERT_RGB)
            
            # Process every 3rd frame to reduce computation
            if not st.session_state.processing:
                st.session_state.processing = True
                
                try:
                    # Recognize face
                    identity, confidence = recognize_face(frame_rgb)
                    status_text.text(f"Status: {identity} (confidence: {confidence:.2f})")
                    
                    # Display result on frame
                    cv2.putText(frame_rgb, f"{identity}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"Confidence: {confidence:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                except Exception as e:
                    status_text.text(f"Status: Error - {str(e)}")
                
                st.session_state.processing = False
            
            if show_fps:
                cv2.putText(frame_rgb, f"FPS: {int(fps)}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                fps_text.text(f"Current FPS: {int(fps)}")
            
            FRAME_WINDOW.image(frame_rgb)
            
            # Check if user stopped the recognition
            if not run:
                break
    else:
        if st.session_state.capture is not None:
            st.session_state.capture.release()
            st.session_state.capture = None
        status_text.text("Status: Ready")
        fps_text.empty()

if __name__ == "__main__":
    main()
