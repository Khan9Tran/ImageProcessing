import streamlit as st
from streamlit_option_menu import option_menu
import solving_quadratic_equations as sqe
import handwritten_digit_recognition as hdr
# import object_detection as obd
import qr_scanner as qrsc
import image_processing_menu as menuImgProcessing
import face_detection as fd
import home
import PredictCharacters as pcs
import nhandangtraicay 

def do_home():
    home.show_info()

def do_solving_quadratic_equations():
    sqe.solve()

def do_handwritten_digit_recognition():
    hdr.solve()

def do_object_detection():
    obd.solve()

def do_qr_scanner():
    qrsc.solve()
    
def do_image_processing():
    menuImgProcessing.do_image_processing()

def do_fruit_processing():
    nhandangtraicay.solve()

def do_face_detections():
    fd.solve()

def do_license_plate_detections():
    pcs.solve()


menu_dict = {
    "Home" : {"fn": do_home},
    "Face detections": {"fn": do_face_detections},
    "Solving quadratic equations" : {"fn": do_solving_quadratic_equations},
    "Handwritten digit recognition using MNIST": {"fn": do_handwritten_digit_recognition},
    "Object detection using YOLOv4" : {"fn": do_object_detection},
    "Fruit recognition of 5 types" : {"fn": do_fruit_processing},
    "Image processing" : {"fn": do_image_processing},
    "QR Scanner" : {"fn": do_qr_scanner},
    "License plates detections" : {"fn": do_license_plate_detections},
}
with st.sidebar:
    selected = option_menu(None, 
        [
            "Home", 
            "Solving quadratic equations",
            "Face detections",
            "Object detection using YOLOv4",
            "Handwritten digit recognition using MNIST",
            "Fruit recognition of 5 types",
            "Image processing",
            "QR Scanner",
            "License plates detections"
            ]
        ,
        default_index=0, icons=['house', '1-square-fill', '2-square-fill', '3-square-fill', '4-square-fill', '5-square-fill', '6-square-fill', '7-square-fill', '8-square-fill'])

if selected in menu_dict.keys():
        menu_dict[selected]["fn"]()
        