import streamlit as st
import pandas as pd
import numpy as np
import os
from deepface import DeepFace
from PIL import Image
import tempfile

# Configuration
CONFIG = {
    "known_faces_dir": "known_faces",
    "database_path": "face_embeddings.csv",
    "model_name": "ArcFace",  # Model options: VGG-Face, Facenet, DeepFace
    "threshold": 0.55,
    "detector": "retinaface"
}

# Ensure directory exists
os.makedirs(CONFIG["known_faces_dir"], exist_ok=True)

# Load database
@st.cache_data
def load_database():
    try:
        df = pd.read_csv(CONFIG["database_path"])
        df["embedding"] = df["embedding"].apply(eval)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "embedding"])

# Save new embeddings
def save_embedding(name, image_path):
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name=CONFIG["model_name"],
            detector_backend=CONFIG["detector"],
            enforce_detection=True
        )[0]["embedding"]

        df = load_database()
        new_entry = pd.DataFrame([{"name": name, "embedding": embedding}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(CONFIG["database_path"], index=False)
        st.success(f"✅ {name} added to the database!")
    except Exception as e:
        st.error(f"Error during embedding extraction: {e}")

# Recognize face
def recognize_face(image_path):
    df = load_database()
    if df.empty:
        st.warning("⚠️ No faces found in the database. Add some first.")
        return None
    
    try:
        query_embedding = DeepFace.represent(
            img_path=image_path,
            model_name=CONFIG["model_name"],
            detector_backend=CONFIG["detector"],
            enforce_detection=True
        )[0]["embedding"]

        similarities = df["embedding"].apply(lambda x: np.dot(query_embedding, np.array(x)) / 
                                             (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(x))))
        
        best_match = df.iloc[similarities.idxmax()]
        confidence = similarities.max()

        if confidence > CONFIG["threshold"]:
            return best_match["name"], confidence
        else:
            return "Unknown", confidence

    except Exception as e:
        st.error(f"Error during recognition: {e}")
        return None

# Streamlit UI
st.title("🔍 Simple Face Recognition System")

tab1, tab2 = st.tabs(["Recognize Face", "Add New Face"])

# Recognize Face Tab
with tab1:
    uploaded_file = st.file_uploader("Upload an image to recognize", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

        if st.button("Recognize Face"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                img.save(temp_file.name)
                result = recognize_face(temp_file.name)
                os.unlink(temp_file.name)

                if result:
                    name, confidence = result
                    if name != "Unknown":
                        st.success(f"✅ Match: {name} (Confidence: {confidence:.2f})")
                    else:
                        st.warning(f"❌ No match found (Confidence: {confidence:.2f})")

# Add New Face Tab
with tab2:
    new_name = st.text_input("Enter person's name")
    new_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="add_face")

    if new_name and new_image:
        if st.button("Add Face"):
            person_dir = os.path.join(CONFIG["known_faces_dir"], new_name)
            os.makedirs(person_dir, exist_ok=True)

            img_path = os.path.join(person_dir, new_image.name)
            Image.open(new_image).save(img_path)

            save_embedding(new_name, img_path)

    st.header("📊 Database Overview")
    df = load_database()
    if not df.empty:
        st.write(f"Total faces in database: {len(df)}")
        st.dataframe(df[["name"]])
    else:
        st.info("No faces available in the database.")
