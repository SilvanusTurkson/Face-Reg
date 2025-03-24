import streamlit as st
import cv2
import numpy as np
from face_recognition_system import FaceRecognitionSystem
import time
import os
from PIL import Image

# Initialize the face recognition system
fr_system = FaceRecognitionSystem()

# Streamlit app layout
st.title("Real-Time Face Recognition System")
st.write("This system recognizes faces in real-time using DeepFace and displays performance metrics.")

# Sidebar controls
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Recognition Threshold", 0.0, 1.0, 0.4, 0.05)
fr_system.threshold = threshold

model_options = ["Facenet", "VGG-Face", "OpenFace", "DeepFace"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)
fr_system.model_name = selected_model

# Upload new faces
st.sidebar.header("Add New Faces")
uploaded_files = st.sidebar.file_uploader(
    "Upload face images (jpg, png)", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

for uploaded_file in uploaded_files:
    with open(os.path.join(known_faces_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {uploaded_file.name}")

if st.sidebar.button("Rebuild Face Database"):
    fr_system.load_known_faces(known_faces_dir)
    fr_system.build_face_database()
    st.sidebar.success("Face database rebuilt!")

# Video processing
st.header("Live Face Recognition")
run_recognition = st.checkbox("Start Recognition", value=False)

# Placeholders for the video feed and metrics
video_placeholder = st.empty()
metrics_placeholder = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0
curr_time = 0

while run_recognition:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video")
        break
    
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Recognize faces
    recognized_faces = fr_system.recognize_face(frame_rgb)
    
    # Draw rectangles and labels
    for face in recognized_faces:
        x, y, w, h = face["location"]
        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{face['name']} ({face['confidence']:.2f})"
        cv2.putText(
            frame_rgb, 
            label, 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(
        frame_rgb, 
        f"FPS: {int(fps)}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (255, 0, 0), 
        2
    )
    
    # Update metrics
    fr_system.update_metrics()
    metrics = fr_system.calculate_metrics()
    
    # Display video feed
    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    # Display metrics
    metrics_text = f"""
    ### Performance Metrics
    - **Precision**: {metrics['precision']:.2f}
    - **Recall**: {metrics['recall']:.2f}
    - **F1 Score**: {metrics['f1_score']:.2f}
    - **Average Confidence**: {metrics['average_confidence']:.2f}
    - **True Positives**: {metrics['true_positives']}
    - **False Positives**: {metrics['false_positives']}
    """
    metrics_placeholder.markdown(metrics_text)

# Release resources when stopped
cap.release()
cv2.destroyAllWindows()

if not run_recognition:
    st.write("Click the checkbox above to start recognition")
