import logging
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import DeepFace safely
try:
    from deepface import DeepFace
except ImportError as e:
    logging.error(f"DeepFace import failed: {e}")
    st.error("DeepFace library not installed. Please install it using `pip install deepface`.")

# Configuration - Use local files instead of Google Drive
CONFIG = {
    "database_path": "face_embeddings.csv",
    "model_name": "ArcFace",
    "threshold": 0.55,
    "detector": "retinaface"
}

# Load or create database
@st.cache_data
def load_database():
    try:
        df = pd.read_csv(CONFIG['database_path'])
        df['embedding'] = df['embedding'].apply(eval)  # Convert string back to list
        return df
    except Exception as e:
        logging.warning(f"Failed to load database: {e}")
        return pd.DataFrame(columns=['name', 'embedding'])

# Face recognition function
def recognize_face(image_path, df):
    try:
        # Get face embedding
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name=CONFIG['model_name'],
            detector_backend=CONFIG['detector'],
            enforce_detection=True
        )[0]['embedding']

        # Convert to NumPy array for calculations
        query = np.array(embedding)

        # Compute similarity with stored embeddings
        df['similarity'] = df['embedding'].apply(
            lambda x: np.dot(query, np.array(x)) /
            (np.linalg.norm(query) * np.linalg.norm(np.array(x)))
        )

        # Get best match
        best_match = df.loc[df['similarity'].idxmax()]
        return best_match

    except Exception as e:
        logging.error(f"Recognition error: {e}")
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI
st.title("Face Recognition System")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Recognize button
    if st.button("Recognize Face"):
        df = load_database()

        if df.empty:
            st.warning("No faces in database! Please add faces first.")
        else:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                result = recognize_face(tmp.name, df)
                os.unlink(tmp.name)

            if result is not None:
                st.subheader("Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Best Match", result['name'])

                with col2:
                    st.metric("Confidence", f"{result['similarity']:.2f}")

                # Show progress bar
                st.progress(result['similarity'])

                # Verification result
                if result['similarity'] > CONFIG['threshold']:
                    st.success("✅ Verified Match")
                else:
                    st.warning("⚠️ Low Confidence Match")

                # Show all matches
                with st.expander("Show all matches"):
                    for _, row in df.sort_values('similarity', ascending=False).iterrows():
                        st.write(f"{row['name']}: {row['similarity']:.4f}")

# Database management
st.sidebar.header("Database Info")
df = load_database()
if not df.empty:
    st.sidebar.write(f"Registered faces: {len(df)}")
    with st.sidebar.expander("View all"):
        st.table(df[['name']])
else:
    st.sidebar.warning("Database is empty")
