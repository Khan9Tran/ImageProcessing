import streamlit as st

from streamlit_option_menu import option_menu
import solving_quadratic_equations as sqe
import handwritten_digit_recognition as hdr
import face_detection as fd

def do_home():
    st.title("Informations")
    st.write("Họ và tên: Trần Lâm Nhựt Khang")
    st.write("Mã số sinh viên: 21110497")

    st.write("Họ và tên: Nguyễn Thanh Huy")
    st.write("Mã số sinh viên: 21110473")

def do_solving_quadratic_equations():
    sqe.solve()

def do_handwritten_digit_recognition():
    hdr.solve()

def do_face_detections():
    fd.solve()

menu_dict = {
    "Home" : {"fn": do_home},
    "Face detections": {"fn": do_face_detections},
    "Solving quadratic equations" : {"fn": do_solving_quadratic_equations},
    "Handwritten digit recognition using MNIST": {"fn": do_handwritten_digit_recognition}
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
            "Image processing"]
        ,
        default_index=0, icons=['house', '1-square-fill', '2-square-fill', '3-square-fill', '4-square-fill', '5-square-fill', '6-square-fill'])

if selected in menu_dict.keys():
        menu_dict[selected]["fn"]()