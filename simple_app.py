import streamlit as st
from deepface import DeepFace

st.title("Facial Recognition System")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    analysis = DeepFace.analyze(uploaded_file)
    st.write(analysis)
