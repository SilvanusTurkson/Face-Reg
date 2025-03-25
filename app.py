import streamlit as st
import tempfile
from PIL import Image
from face_recognition import extract_embeddings, recognize_face

# Title
st.title("🔍 Face Recognition System")

# Extract Embeddings Button
if st.button("📊 Extract Face Embeddings"):
    extract_embeddings()
    st.success("✅ Face embeddings extracted successfully!")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an image for recognition", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # Recognize Face
    if st.button("🔎 Recognize Face"):
        match_name, confidence = recognize_face(image_path)

        if match_name:
            st.success(f"✅ Match Found: {match_name}")
            st.write(f"🔢 Confidence: {confidence:.4f}")
        else:
            st.error("❌ No face detected or database error.")
