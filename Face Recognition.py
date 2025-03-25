import streamlit as st
import pandas as pd
import numpy as np
from deepface import DeepFace
import os
from PIL import Image
import tempfile

# Configuration
CONFIG = {
    "known_faces_dir": "known_faces",
    "database_path": "face_embeddings.csv",
    "model_name": "Facenet",
    "threshold": 0.55,
    "detector": "opencv"
}

# Initialize directories
os.makedirs(CONFIG['known_faces_dir'], exist_ok=True)

@st.cache_data
def load_database():
    try:
        df = pd.read_csv(CONFIG['database_path'])
        df['embedding'] = df['embedding'].apply(eval)
        return df
    except:
        return pd.DataFrame(columns=['name', 'embedding'])

def recognize_face(img_path, df):
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=CONFIG['model_name'],
            detector_backend=CONFIG['detector'],
            enforce_detection=False
        )[0]['embedding']
        query = np.array(embedding)
        
        # Fixed this line - properly closed all parentheses
        df['similarity'] = df['embedding'].apply(
            lambda x: np.dot(query, np.array(x)) / 
            (np.linalg.norm(query) * np.linalg.norm(np.array(x)))
        )
        
        best_match = df.loc[df['similarity'].idxmax()]
        return best_match
    except Exception as e:
        st.error(f"Recognition error: {str(e)}")
        return None

# Streamlit UI
st.title("Face Recognition System")

tab1, tab2 = st.tabs(["Recognize Faces", "Manage Database"])

with tab1:
    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)
        
        if st.button("Recognize"):
            df = load_database()
            if not df.empty:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    img.save(tmp.name)
                    result = recognize_face(tmp.name, df)
                    os.unlink(tmp.name)
                    
                    if result is not None:
                        if result['similarity'] > CONFIG['threshold']:
                            st.success(f"✅ Match: {result['name']} (Confidence: {result['similarity']:.2f})")
                        else:
                            st.warning(f"⚠️ Possible Match: {result['name']} (Low Confidence: {result['similarity']:.2f})")
            else:
                st.error("No faces in database! Add some faces first.")

with tab2:
    st.header("Add New Face")
    new_name = st.text_input("Person's Name")
    new_face = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"], key="enrollment")
    
    if new_name and new_face:
        if st.button("Add to Database"):
            person_dir = os.path.join(CONFIG['known_faces_dir'], new_name)
            os.makedirs(person_dir, exist_ok=True)
            
            img_path = os.path.join(person_dir, new_face.name)
            Image.open(new_face).save(img_path)
            
            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=CONFIG['model_name'],
                    detector_backend=CONFIG['detector'],
                    enforce_detection=False
                )[0]['embedding']
                
                df = load_database()
                new_entry = pd.DataFrame([{
                    "name": new_name,
                    "embedding": embedding  # Fixed typo here
                }])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(CONFIG['database_path'], index=False)
                
                st.success(f"✅ Added {new_name} to database!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.header("Database Info")
    df = load_database()
    if not df.empty:
        st.write(f"Total registered faces: {len(df)}")
        st.dataframe(df[['name']])
    else:
        st.warning("No faces in database")
