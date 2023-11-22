import streamlit as st
import subprocess
import get_face as gf
import Training as tn
from PIL import Image

def solve():
    st.title("Face Detections")

    st.markdown(
        """
        <style>
        .stButton button {
            width: 700px;
            height: 70px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    button_face_detection = st.button("Bắt đầu nhận dạng khuôn mặt")
    st.write(" ")
    st.write("Hoặc...")
    st.write(" ")
    input_text = st.text_input("Nhập tên của bạn")
    button_face_scanning = st.button("Bắt đầu quét khuôn mặt")

    if button_face_detection:
        face_dection()

    if button_face_scanning:
        face_scanning(input_text)
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
def face_dection():
    subprocess.run(['python', 'predict.py'], shell=True, check=True)

def face_scanning(folderName):
    gf.face_scanning(folderName)
    subprocess.run(['python', 'Training.py'], shell=True, check=True)