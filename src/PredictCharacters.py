import streamlit as st
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import numpy as np
from skimage.transform import resize
import pickle

def process_car_image(image):
    # Chuyển đổi ảnh xám và tăng cường độ tương phản
    gray_car_image = image * 255

    # Áp dụng phương pháp ngưỡng Otsu để chuyển thành ảnh nhị phân
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value

    # Liên kết các vùng nhị phân
    label_image = measure.label(binary_car_image)

    # Kích thước ước lượng của biển số xe
    plate_dimensions = (0.03 * label_image.shape[0], 0.08 * label_image.shape[0],
                        0.15 * label_image.shape[1], 0.3 * label_image.shape[1])
    plate_dimensions2 = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0],
                         0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []

    flag = 0

    # regionprops tạo một danh sách các thuộc tính của tất cả các vùng được gắn nhãn
    for region in regionprops(label_image):
        if region.area < 50:
            # Nếu vùng quá nhỏ thì nó có thể không phải là biển số
            continue

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        # Đảm bảo rằng vùng được xác định đáp ứng điều kiện của một biển số xe thông thường
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            flag = 1
            plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

    if flag == 1:
        return plate_like_objects, plate_objects_cordinates
    else:
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []

        # regionprops tạo một danh sách các thuộc tính của tất cả các vùng được gắn nhãn
        for region in regionprops(label_image):
            if region.area < 50:
                # Nếu vùng quá nhỏ thì nó có thể không phải là biển số
                continue

            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col

            # Đảm bảo rằng vùng được xác định đáp ứng điều kiện của một biển số xe thông thường
            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

        return plate_like_objects, plate_objects_cordinates

def predict_license_plate(column_list, characters):
    print("Đang tải mô hình")
    filename = './finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    print('Mô hình đã được tải. Đang dự đoán ký tự của biển số')
    classification_result = []
    for each_character in characters:
        # chuyển đổi thành mảng 1D
        each_character = each_character.reshape(1, -1)
        result = model.predict(each_character)
        classification_result.append(result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    # Có thể ký tự được sắp xếp sai
    # do đó, column_list sẽ được sử dụng để sắp xếp các chữ cái theo đúng thứ tự
    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]
    return rightplate_string

def segment_characters(plate_like_objects):
    # The invert was done so as to convert the black pixel to white pixel and vice versa
    license_plate = np.invert(plate_like_objects[0])
    labelled_plate = measure.label(license_plate)

    character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    column_list = []

    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)

    return column_list, characters

def solve():
    st.title("License Plates Detection")

    uploaded_file = st.file_uploader("Select Car image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        car_image = imread(uploaded_file, as_gray=True)
        st.image(car_image, caption="Car Image", use_column_width=True)

        if st.button("Submit"):
            plate_like_objects, x = process_car_image(car_image)
            column_list, characters = segment_characters(plate_like_objects)
            st.success("License Plates: " + predict_license_plate(column_list, characters))
