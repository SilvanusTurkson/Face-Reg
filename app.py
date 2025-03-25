import logging
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from PIL import Image
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configuration
CONFIG = {
    "database_path": "face_embeddings.csv",
    "model_name": "Facenet",  # Change to "Facenet" if "ArcFace" is unavailable
    "threshold": 0.55,
    "detector": "retinaface"
}

# Load Database
@st.cache(allow_output_mutation=True)
def load_database():
    try:
        df = pd.read_csv(CONFIG['database_path'])
        df['embedding'] = df['embedding'].apply(eval)  # Convert string back to list
        return df
    except Exception as e:
        logging.warning(f"Failed to load database: {e}")
        return pd.DataFrame(columns=['name', 'embedding'])

# Face Recognition Function
def recognize_face(image_path, df):
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name=CONFIG['model_name'],
            detector_backend=CONFIG['detector'],
            enforce_detection=True
        )[0]['embedding']

        query = np.array(embedding)

        df['similarity'] = df['embedding'].apply(
            lambda x: np.dot(query, np.array(x)) /
            (np.linalg.norm(query) * np.linalg.norm(np.array(x)))
        )

        best_match = df.loc[df['similarity'].idxmax()]
        return best_match

    except Exception as e:
        logging.error(f"Recognition error: {e}")
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI
st.title("Face Recognition System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Recognize Face"):
        df = load_database()

        if df.empty:
            st.warning("No faces in database! Please add faces first.")
        else:
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

                st.progress(result['similarity'])

                if result['similarity'] > CONFIG['threshold']:
                    st.success("✅ Verified Match")
                else:
                    st.warning("⚠️ Low Confidence Match")

                with st.expander("Show all matches"):
                    for _, row in df.sort_values('similarity', ascending=False).iterrows():
                        st.write(f"{row['name']}: {row['similarity']:.4f}")

# Sidebar
st.sidebar.header("Database Info")
df = load_database()
if not df.empty:
    st.sidebar.write(f"Registered faces: {len(df)}")
    with st.sidebar.expander("View all"):
        st.table(df[['name']])
else:
    st.sidebar.warning("Database is empty")
