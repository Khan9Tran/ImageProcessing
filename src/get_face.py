import os
import sys
import shutil
import time
import argparse
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='./face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='./face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            
            center = ((coords[0] + coords[0] + coords[2]) // 2, (coords[1] + coords[1] + coords[3]) // 2)
            axes = ((coords[2] // 2), (coords[3] // 2))
            
            cv.ellipse(input, center, axes, 0, 0, 360, (0, 255, 0), thickness)
            
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (40, 35, 37), 2)

def face_scanning(folderName):

    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,"")

    tm = cv.TickMeter()
    cap = cv.VideoCapture(0)
    frameWidth = 777
    frameHeight = 617
    detector.setInputSize([frameWidth, frameHeight])
    dem = 0
    startTime = 0
    currentTime = 0

    if os.path.exists("./image/" + folderName + "/"):
        shutil.rmtree("./image/" + folderName + "/")
        os.makedirs("./image/" + folderName + "/")
    else:
        os.makedirs("./image/" + folderName + "/")

    while True:
        hasFrame, frame = cap.read()
        frame = cv.resize(frame, (777, 617))
        if not hasFrame:
            print('No frames grabbed!')
            break


        if currentTime == 0:
            currentTime = int(time.time())
            startTime = int(time.time())
        else:
            currentTime = currentTime + 1
        

        if currentTime - startTime >= 50 and currentTime - startTime < 130:
            cv.putText(frame, "Look straight", (310, 150), cv.FONT_HERSHEY_DUPLEX, 0.7, (40, 35, 37), 2)
            if (currentTime - startTime)%2 == 0:
                if faces[1] is not None:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    file_name = './image/' + folderName + '/' + folderName + '_%04d.bmp' % dem
                    cv.imwrite(file_name, face_align)
                    dem = dem + 1

        if currentTime - startTime >= 140 and currentTime - startTime < 220:
            cv.putText(frame, "Look left", (30, 400), cv.FONT_HERSHEY_DUPLEX, 0.7, (40, 35, 37), 2)
            if (currentTime - startTime)%2 == 0:
                if faces[1] is not None:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    file_name = './image/' + folderName + '/' + folderName + '_%04d.bmp' % dem
                    cv.imwrite(file_name, face_align)
                    dem = dem + 1

        if currentTime - startTime >= 230 and currentTime - startTime < 310:
            cv.putText(frame, "Look right", (550, 400), cv.FONT_HERSHEY_DUPLEX, 0.7, (40, 35, 37), 2)
            if (currentTime - startTime)%2 == 0:
                if faces[1] is not None:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    file_name = './image/' + folderName + '/' + folderName + '_%04d.bmp' % dem
                    cv.imwrite(file_name, face_align)
                    dem = dem + 1

        if currentTime - startTime >= 320 and currentTime - startTime < 400:
            cv.putText(frame, "Look up", (310, 150), cv.FONT_HERSHEY_DUPLEX, 0.7, (40, 35, 37), 2)
            if (currentTime - startTime)%2 == 0:
                if faces[1] is not None:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    file_name = './image/' + folderName + '/' + folderName + '_%04d.bmp' % dem
                    cv.imwrite(file_name, face_align)
                    dem = dem + 1

        if currentTime - startTime >= 410 and currentTime - startTime < 490:
            cv.putText(frame, "Look down", (310, 610), cv.FONT_HERSHEY_DUPLEX, 0.7, (40, 35, 37), 2)
            if (currentTime - startTime)%2 == 0:
                if faces[1] is not None:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    file_name = './image/' + folderName + '/' + folderName + '_%04d.bmp' % dem
                    cv.imwrite(file_name, face_align)
                    dem = dem + 1

        if currentTime - startTime >= 490 and currentTime - startTime < 520:
            cv.putText(frame, "Complete!!!", (310, 150), cv.FONT_HERSHEY_DUPLEX, 0.7, (10, 255, 10), 2)

        if currentTime - startTime == 520:
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        key = cv.waitKey(1)
        if key == 27:
            break

        if key == ord('s') or key == ord('S'):
            if faces[1] is not None:
                face_align = recognizer.alignCrop(frame, faces[1][0])
                file_name = './image/ThanhHuy/ThanhHuy_%04d.bmp' % dem
                cv.imwrite(file_name, face_align)
                dem = dem + 1
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        cv.imshow('Live', frame)
    cv.destroyAllWindows()

if __name__ == "__main__":
    face_scanning("THANHHUY")
