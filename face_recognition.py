# Install Dependencies (Colab Only)
!pip install -q -U deepface streamlit tensorflow keras numpy opencv-python-headless pandas Pillow
!pip install streamlit
!apt-get install -y libgl1-mesa-glx libglib2.0-0
!pip install pyngrok

# Import Libraries
import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Configuration
CONFIG = {
    "known_faces_dir": "/content/drive/MyDrive/known_faces",
    "unknown_faces_dir": "/content/drive/MyDrive/unknown_faces",
    "database_path": "/content/drive/MyDrive/face_embeddings.csv",
    "model_name": "Facenet",  # Change to "Facenet" if "ArcFace" is not available
    "threshold": 0.55,
    "image_extensions": ('.png', '.jpg', '.jpeg'),
    "detector": "retinaface"
}

# Helper Functions
def is_image_file(filename):
    return filename.lower().endswith(CONFIG['image_extensions'])

def clean_directory_listing(path):
    return [f for f in os.listdir(path) if is_image_file(f)]

# Extract Face Embeddings
def extract_embeddings():
    data = []
    print("⏳ Extracting face embeddings...")

    for person_name in os.listdir(CONFIG['known_faces_dir']):
        person_path = os.path.join(CONFIG['known_faces_dir'], person_name)

        if not os.path.isdir(person_path):
            continue

        for img_name in clean_directory_listing(person_path):
            img_path = os.path.join(person_path, img_name)
            print(f"Processing image: {img_path}")

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
                print(f"✅ Processed: {person_name}/{img_name}")

            except Exception as e:
                print(f"❌ Failed on {img_path}: {str(e)}")
                continue

    if data:
        pd.DataFrame(data).to_csv(CONFIG['database_path'], index=False)
        print(f"💾 Saved {len(data)} embeddings to {CONFIG['database_path']}")
    else:
        print("❌ No valid faces found!")

# Recognize Faces
def recognize_faces():
    try:
        df = pd.read_csv(CONFIG['database_path'])
        df['embedding'] = df['embedding'].apply(eval)  # Convert string to list
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    print("\n🔍 Starting recognition...")

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

            similarities = [
                np.dot(query_embedding, np.array(row['embedding'])) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(row['embedding'])))
                for _, row in df.iterrows()
            ]

            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            match_name = df.iloc[best_match_idx]['name']

            print(f"\n📸 Processing: {img_name}")
            print(f"🔎 Best Match: {match_name}")
            print(f"📏 Similarity: {best_similarity:.4f} (Threshold: {CONFIG['threshold']})")

            if best_similarity > CONFIG['threshold']:
                print(f"✅ VERIFIED: {match_name}")
            else:
                print("❌ UNKNOWN FACE")

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {str(e)}")

# Execute
print("⭐ Face Recognition System ⭐")
extract_embeddings()
recognize_faces()

# Summary
print("\n🔬 Summary")
print("Known faces:", os.listdir(CONFIG['known_faces_dir']))
print("Tested unknown faces:", clean_directory_listing(CONFIG['unknown_faces_dir']))
