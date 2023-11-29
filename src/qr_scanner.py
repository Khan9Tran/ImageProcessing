import cv2
import numpy as np
import streamlit as st

def solve():
    image_file = st.file_uploader("Upload Images", type=["bmp", "png", "jpg", "jpeg"])
    if image_file is not None:
        # Read the image directly with OpenCV
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

        st.image(img, channels="BGR", caption=None)  # Display the image in BGR format

        umat_img = cv2.UMat(img)  # Convert to UMat

        det = cv2.QRCodeDetector()
        info, box_coordinates, _ = det.detectAndDecode(umat_img)

        if box_coordinates is None:
            st.write("Error")
        else:
            st.success(info)

        if box_coordinates is not None:
            box_coordinates_np = box_coordinates.get()  # Convert to NumPy array
            box_coordinates_np = [box_coordinates_np[0].astype(int)]
            n = len(box_coordinates_np[0])
            for i in range(n):
                cv2.line(
                    img,
                    tuple(box_coordinates_np[0][i]),
                    tuple(box_coordinates_np[0][(i + 1) % n]),
                    (0, 255, 0),
                    3,
                )
        
        st.image(img, channels="BGR", caption= "QR Decode")