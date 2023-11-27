import subprocess
import streamlit as st
import get_face as gf
import predict as pr
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
        temp_image_path = "./image_test_face_detection_with_image.jpg"
        with open(temp_image_path, "wb") as temp_image_file:
            temp_image_file.write(uploaded_image.getvalue())

        image = Image.open(temp_image_path)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        pr.face_detection_with_image(temp_image_path)

def face_dection():
    pr.face_detection_with_camera()

def face_scanning(folderName):
    gf.face_scanning(folderName)
    with st.spinner('Đang tiến hành training...'):
        subprocess.run(['python', 'Training.py'], shell=True, check=True)
    st.success('Hoàn thành!')