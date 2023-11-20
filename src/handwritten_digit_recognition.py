import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json
import numpy as np
import random
import cv2

def tao_anh_ngau_nhien():
    image = np.zeros((10 * 28, 10 * 28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28] = sample[:, :, 0]
    return image, data

def nhan_dang_chu_so(data):
    data = data / 255.0
    data = data.astype('float32')
    ket_qua = st.session_state.model.predict(data)
    dem = 0
    s = ''
    for x in ket_qua:
        s = s + '%d ' % (np.argmax(x))
        dem = dem + 1
        if (dem % 10 == 0) and (dem < 100):
            s = s + '\n'
    return s

if 'is_load' not in st.session_state:
    # Load model
    model_architecture = 'digit_config.json'
    model_weights = 'digit_weight.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    OPTIMIZER = tf.keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
                  metrics=["accuracy"])
    st.session_state.model = model

    # Load data
    (_, _), (X_test, _) = datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

    st.session_state.is_load = True
    st.session_state.image = None  # Initialize image in session state

    print('Lần đầu load model và data')
else:
    print('Đã load model và data rồi')

def solve():
    st.title("Digit Recognition")
    
    if st.button('Tạo ảnh'):
        image, data = tao_anh_ngau_nhien()
        st.session_state.image = image
        st.session_state.data = data
    
    if st.session_state.image is not None:
        image = st.session_state.image
        st.image(image)
        
        if st.button('Nhận dạng'):
            data = st.session_state.data
            ket_qua = nhan_dang_chu_so(data)
            st.text(ket_qua)


