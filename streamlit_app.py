import os
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Set page title and layout
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("🔍 Face Recognition System")

# Function to load known faces and extract embeddings
def load_known_faces(known_faces_dir):
    known_embeddings = {}
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                    known_embeddings[person_name] = embedding
                except Exception as e:
                    st.warning(f"Skipped {img_path} (error: {str(e)})")
    return known_embeddings

# Function to recognize a face
def recognize_face(unknown_img_path, known_embeddings, threshold=0.65):
    try:
        unknown_embedding = DeepFace.represent(
            img_path=unknown_img_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
        
        best_match = None
        best_score = 0

        for name, known_embedding in known_embeddings.items():
            result = DeepFace.verify(
                img1_path=unknown_embedding,
                img2_path=known_embedding,
                model_name="Facenet",
                distance_metric="cosine",
                enforce_detection=False
            )
            similarity = 1 - result["distance"]  # Convert to similarity score (0-1)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = name

        return best_match, best_score if best_match else ("Unknown", 0)
    except Exception as e:
        st.error(f"Recognition error: {str(e)}")
        return "Error", 0

# Main Streamlit UI
def main():
    # Sidebar options
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Similarity Threshold", 0.1, 1.0, 0.65, 0.05)
    model = st.sidebar.selectbox("Model", ["Facenet", "VGG-Face", "OpenFace"])

    # Load known faces (cache to avoid recomputing)
    KNOWN_FACES_DIR = "dataset/known_faces"
    if not os.path.exists(KNOWN_FACES_DIR):
        st.error("Error: 'dataset/known_faces' folder not found!")
        return
    
    known_embeddings = load_known_faces(KNOWN_FACES_DIR)
    if not known_embeddings:
        st.error("No valid faces found in 'known_faces'. Add images in subfolders.")
        return

    # Main options
    option = st.radio("Select mode:", ("Upload Image", "Test Unknown Faces Folder"))

    if option == "Upload Image":
        st.subheader("Upload a Face to Recognize")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)
            
            # Save temp file and recognize
            temp_path = "temp_upload.jpg"
            img.save(temp_path)
            
            name, score = recognize_face(temp_path, known_embeddings, threshold)
            st.write("---")
            if name != "Unknown":
                st.success(f"✅ **Match Found:** {name} (Confidence: {score:.2f})")
            else:
                st.warning("❌ No match found.")

    elif option == "Test Unknown Faces Folder":
        st.subheader("Testing Images from 'unknown_faces' Folder")
        UNKNOWN_FACES_DIR = "dataset/unknown_faces"
        
        if not os.path.exists(UNKNOWN_FACES_DIR):
            st.error("Error: 'dataset/unknown_faces' folder not found!")
            return
        
        for img_file in os.listdir(UNKNOWN_FACES_DIR):
            img_path = os.path.join(UNKNOWN_FACES_DIR, img_file)
            try:
                img = Image.open(img_path)
                st.image(img, caption=f"Image: {img_file}", width=200)
                
                name, score = recognize_face(img_path, known_embeddings, threshold)
                if name != "Unknown":
                    st.success(f"✅ **Match:** {name} (Confidence: {score:.2f})")
                else:
                    st.warning("❌ No match found.")
                st.write("---")
            except Exception as e:
                st.error(f"Failed to process {img_file}: {str(e)}")

if __name__ == "__main__":
    main()
