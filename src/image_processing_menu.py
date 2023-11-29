import cv2
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu
import io
import Chapter03
import Chapter04
import Chapter05 
import Chapter09

def do_chapter3(image):
    selected_option = st.selectbox("Choose an operation", [
        "Negative",
        "Logarithm",
        "Piecewise Linear",
        "Histogram",
        "Histogram Equalization",
        "Histogram Equalization (Color)",
        "Local Histogram",
        "Histogram Statistics",
        "Box Filter",
        "Low-pass Gauss Filter",
        "Thresholding",
        "Median Filter",
        "Sharpen",
        "Gradient"
    ],   index=None,
    placeholder="Select method...",)
    if (image is not None and selected_option is not None):
        # Implement the corresponding image processing operations based on the selected_option
        if selected_option == "Negative":
            imgout = Chapter03.Negative(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Logarithm":
            imgout = Chapter03.Logarit(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Piecewise Linear":
            imgout = Chapter03.PiecewiseLinear(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        
        elif selected_option == "Histogram":
            imgout = Chapter03.Histogram(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Histogram Equalization":
            imgout = Chapter03.HistEqual(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Histogram Equalization (Color)":
            imgout = Chapter03.HistEqualColor(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR))

        elif selected_option == "Local Histogram":
            imgout = Chapter03.LocalHist(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Histogram Statistics":
            imgout = Chapter03.HistStat(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Box Filter":
            imgout = Chapter03.BoxFilter(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Low-pass Gauss Filter":
            frame=cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            imgout = cv2.GaussianBlur(frame,(43,43),7.0)

        elif selected_option == "Thresholding":
            imgout = Chapter03.Threshold(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        elif selected_option == "Median Filter":
            imgout = Chapter03.Threshold(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        
        elif selected_option == "Sharpen":
            imgout = Chapter03.Sharpen(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        elif selected_option == "Gradient":
            imgout = Chapter03.Gradient(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        if imgout.shape[-1] != 3:
            st.image(imgout, caption=None, channels="GRAY")
        
        else:
            st.image(imgout, caption=None, channels="BGR")

def do_chapter4(image):
    selected_option = st.selectbox("Choose an operation", [
        "Spectrum",
        "Frequency Filter",
        "Draw Notch Reject Filter",
        "Remove Moire"
    ],   index=None,
    placeholder="Select method...",)
    
    if (image is not None and selected_option is not None and selected_option != "Draw Notch Reject Filter"):
        # Implement the corresponding image processing operations based on the selected_option
        if selected_option == "Spectrum":
            imgout = Chapter04.Spectrum(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        elif selected_option == "Frequency Filter":
            imgout = Chapter04.FrequencyFilter(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        elif selected_option == "Remove Moire":
            imgout = Chapter04.RemoveMoire(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))

        if imgout.shape[-1] != 3:
            st.image(imgout, caption=None, channels="GRAY")
        
        else:
            st.image(imgout, caption=None, channels="BGR")

    elif selected_option == "Draw Notch Reject Filter":
            st.image(Chapter04.CreateNotchRejectFilter(), caption=None, channels="GRAY")

def do_chapter5(image):
    selected_option = st.selectbox("Choose an operation", [
        "Create Motion Noise",
        "Denoise Motion",
        "Denoisest Motion"
    ],   index=None,
    placeholder="Select method...",)
    if (image is not None and selected_option is not None):
        # Implement the corresponding image processing operations based on the selected_option
        if selected_option == "Create Motion Noise":
            imgout = Chapter05.CreateMotionNoise(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        elif selected_option == "Denoise Motion":
            imgout = Chapter05.DenoiseMotion(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        elif selected_option == "Denoisest Motion":
            temp = cv2.medianBlur(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE), 7)
            imgout = Chapter05.DenoiseMotion(temp)

        if imgout.shape[-1] != 3:
            st.image(imgout, caption=None, channels="GRAY")
        
        else:
            st.image(imgout, caption=None, channels="BGR")
def do_chapter9(image):
    selected_option = st.selectbox("Choose an operation", [
        "Connected Component",
        "Count Rice"
    ],   index=None,
    placeholder="Select method...",)
    if (image is not None and selected_option is not None):
        if (selected_option == "Connected Component"):
            imgout = Chapter09.ConnectedComponent(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        if (selected_option == "Count Rice"):
            imgout = Chapter09.CountRice(cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
            
        if imgout.shape[-1] != 3:
            st.image(imgout, caption=None, channels="GRAY")
        
        else:
            st.image(imgout, caption=None, channels="BGR")

     
menu_dict = {
    "Chapter 3" : {"fn": do_chapter3},
    "Chapter 4" : {"fn": do_chapter4},
    "Chapter 5" : {"fn": do_chapter5},
    "Chapter 9" : {"fn": do_chapter9}}
def do_image_processing():
    col1, col2 = st.columns(2)
    with col2:
        frame = open_image()
    with col1:
        selected = option_menu('Select Chapter', ["Chapter 3", "Chapter 4", "Chapter 5", "Chapter 9"], 
        default_index=0)
        selected

        if selected in menu_dict.keys():
            menu_dict[selected]["fn"](frame)


def save_image(imgout):
    img_pil = Image.fromarray(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    buffer.seek(0)
    file_dl_name = "ImageProcessing.jpg"
    st.download_button("Download Image", buffer, file_name=file_dl_name, mime="image/jpeg")

def open_image():
    image_file = st.file_uploader("Upload Images", type=["bmp", "png","jpg","jpeg"])
    if image_file is not None:
        st.image(image_file, caption=None, channels="BGR")
        return image_file
    return None