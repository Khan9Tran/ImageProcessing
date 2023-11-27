import argparse

import numpy as np
import cv2 as cv
import joblib
import os            

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
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()


def visualize(names, input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            center = ((coords[0] + coords[0] + coords[2]) // 2, (coords[1] + coords[1] + coords[3]) // 2)
            axes = ((coords[2] // 2), (coords[3] // 2))
            cv.ellipse(input, center, axes, 0, 0, 360, (0, 255, 0), thickness)
            
            name = names[idx]
            cv.putText(input, name, (coords[4] - 25, coords[5] - 170 - idx * 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (40, 35, 37), 2)

    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (40, 35, 37), 2)

def visualize_image(names, input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            center = ((coords[0] + coords[0] + coords[2]) // 2, (coords[1] + coords[1] + coords[3]) // 2)
            axes = ((coords[2] // 2), (coords[3] // 2))
            cv.ellipse(input, center, axes, 0, 0, 360, (0, 255, 0), thickness)
            
            name = names[idx]
            cv.putText(input, name, (coords[4] - 25, coords[5] - 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (40, 35, 37), 2)

def get_subdirectory_names(root_directory):
    def get_subdirectories(directory):
        subdirectories = []
        for entry in os.scandir(directory):
            if entry.is_dir():
                subdirectories.append(entry.name)
                subdirectories.extend(get_subdirectories(entry.path))
        return subdirectories
    
    # Kiểm tra xem thư mục gốc có tồn tại hay không
    if not os.path.exists(root_directory):
        print("Thư mục gốc không tồn tại.")
        return []
    
    # Gọi hàm để lấy tất cả các thư mục con và lưu vào một mảng
    subdirectories = get_subdirectories(root_directory)
    
    return subdirectories

svc = joblib.load('./svc.pkl')
mydict = get_subdirectory_names("image");

def face_detection_with_camera():
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(
        args.face_recognition_model,""
    )

    tm = cv.TickMeter()
    cap = cv.VideoCapture(0)
    frameWidth = 1320
    frameHeight = 780
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        frame = cv.resize(frame, (1320, 780))
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        key = cv.waitKey(1)
        if key == 27:
            break

        if faces[1] is not None:
            names = []
            for face in faces[1]:
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                names.append(result)

            visualize(names, frame, faces, tm.getFPS())

        # Visualize results
        cv.imshow('Live', frame)
        
        cv.moveWindow('Live', 300, 110)
    cv.destroyAllWindows()


def face_detection_with_image(image_path):
    svc = joblib.load('./svc.pkl')
    mydict = get_subdirectory_names("image")

    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (1320, 680),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    
    recognizer = cv.FaceRecognizerSF.create(
        args.face_recognition_model, ""
    )

    frame = cv.imread(image_path)
    frame = cv.resize(frame, (1320, 680))
    faces = detector.detect(frame) 

    if faces[1] is not None:
        names = []
        for face in faces[1]:
            face_align = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            names.append(result)

        visualize_image(names, frame, faces, 0.0)  

    cv.imshow('Image', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()