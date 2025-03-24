# deepface_streamlit.py

import streamlit as st
from deepface import DeepFace
import tempfile
import os

def verify_faces(img1_path, img2_path):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name='Facenet', enforce_detection=False)
        return result
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Facial Recognition System")
    st.write("Upload two images to compare using DeepFace.")

    uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"], key="1")
    uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="2")

    if uploaded_file1 and uploaded_file2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1:
            temp1.write(uploaded_file1.read())
            img1_path = temp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
            temp2.write(uploaded_file2.read())
            img2_path = temp2.name

        st.image([img1_path, img2_path], caption=["Image 1", "Image 2"], width=300)

        if st.button("Compare Faces"):
            st.write("Processing... Please wait.")

            result = verify_faces(img1_path, img2_path)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Comparison Result")
                st.json(result)

        # Clean up temporary files
        os.remove(img1_path)
        os.remove(img2_path)

if __name__ == "__main__":
    main()
