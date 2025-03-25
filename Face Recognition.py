# Install Dependencies (Colab Only)
# !pip install deepface streamlit opencv-python-headless

import streamlit as st
import os
import pandas as pd
import numpy as np
from deepface import DeepFace

# Configuration
CONFIG = {
    "known_faces_dir": "/content/drive/MyDrive/known_faces",
    "unknown_faces_dir": "/content/drive/MyDrive/unknown_faces",
    "database_path": "/content/drive/MyDrive/face_embeddings.csv",
    "model_name": "ArcFace",
    "threshold": 0.55,
    "image_extensions": ('.png', '.jpg', '.jpeg'),
    "detector": "retinaface"
}

def is_image_file(filename):
    return filename.lower().endswith(CONFIG['image_extensions'])

def clean_directory_listing(path):
    return [f for f in os.listdir(path) if is_image_file(f)]

# Extract Embeddings
def extract_embeddings():
    data = []
    st.info("⏳ Extracting face embeddings...")

    for person_name in os.listdir(CONFIG['known_faces_dir']):
        person_path = os.path.join(CONFIG['known_faces_dir'], person_name)

        if not os.path.isdir(person_path):
            continue

        for img_name in clean_directory_listing(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=CONFIG['model_name'],
                    detector_backend=CONFIG['detector'],
                    enforce_detection=True
                )[0]['embedding']

                data.append({
                    "name": person_name,
                    "embedding": np.array(embedding).tolist(),
                    "image_path": img_path
                })
                st.success(f"✅ Processed: {person_name}/{img_name}")

            except Exception as e:
                st.error(f"❌ Failed on {img_path}: {str(e)}")

    if data:
        pd.DataFrame(data).to_csv(CONFIG['database_path'], index=False)
        st.success(f"💾 Saved {len(data)} embeddings to {CONFIG['database_path']}")
    else:
        st.error("❌ No valid faces found!")

# Recognize Faces
def recognize_faces():
    try:
        df = pd.read_csv(CONFIG['database_path'])
        df['embedding'] = df['embedding'].apply(eval)
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return

    st.info("🔍 Starting recognition...")

    for img_name in clean_directory_listing(CONFIG['unknown_faces_dir']):
        img_path = os.path.join(CONFIG['unknown_faces_dir'], img_name)

        try:
            query_embedding = DeepFace.represent(
                img_path=img_path,
                model_name=CONFIG['model_name'],
                detector_backend=CONFIG['detector'],
                enforce_detection=True
            )[0]['embedding']
            query_embedding = np.array(query_embedding)

            similarities = []
            for _, row in df.iterrows():
                known_embedding = np.array(row['embedding'])
                similarity = np.dot(query_embedding, known_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(known_embedding)
                )
                similarities.append(similarity)

            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            match_name = df.iloc[best_match_idx]['name']

            st.image(img_path, caption=f"Processing: {img_name}")
            st.write(f"🔎 Best Match: {match_name}")
            st.write(f"📏 Similarity: {best_similarity:.4f} (Threshold: {CONFIG['threshold']})")

            for idx, similarity in enumerate(similarities):
                st.write(f"ℹ️ {df.iloc[idx]['name']}: {similarity:.4f}")

            if best_similarity > CONFIG['threshold']:
                st.success(f"✅ VERIFIED: {match_name}")
            else:
                st.error("❌ UNKNOWN FACE")

        except Exception as e:
            st.error(f"⚠️ Error processing {img_name}: {str(e)}")

# Streamlit UI
st.title("⭐ Face Recognition System ⭐")

if st.button("Extract Face Embeddings"):
    extract_embeddings()

if st.button("Recognize Faces"):
    recognize_faces()

st.info("📊 Summary")
st.write("Known Faces:", os.listdir(CONFIG['known_faces_dir']))
st.write("Tested Unknown Faces:", clean_directory_listing(CONFIG['unknown_faces_dir']))
