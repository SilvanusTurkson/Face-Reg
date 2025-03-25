# app.py - Streamlit Interface for Face Recognition

import streamlit as st
from face_recognition import extract_embeddings, recognize_faces

# Streamlit Page Configuration
st.set_page_config(page_title="Face Recognition System", layout="wide")

# Title
st.title("⭐ Face Recognition System")

# Menu
menu = st.sidebar.radio("Select Option", ["Extract Embeddings", "Recognize Faces"])

# Display Instructions
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("""
1. Ensure known faces are in the `known_faces` directory.
2. Place unknown faces in the `unknown_faces` directory.
3. Choose an option to run.
""")

# Run Extract Embeddings
if menu == "Extract Embeddings":
    if st.button("Extract Face Embeddings"):
        st.info("Processing images from the `known_faces` directory...")
        extract_embeddings()
        st.success("✅ Embeddings extracted successfully!")

# Run Recognize Faces
elif menu == "Recognize Faces":
    if st.button("Recognize Faces"):
        st.info("Recognizing faces from the `unknown_faces` directory...")
        recognize_faces()
        st.success("✅ Face recognition completed!")

# Footer
st.sidebar.markdown("Developed with ❤️ using DeepFace and Streamlit")
