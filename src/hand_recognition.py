import streamlit as st
import app as handr

def solve():
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

    st.title("Hand recognition")
    button_hand_recognition = st.button("Bắt đầu nhận dạng bàn tay")
    if button_hand_recognition:
        handr.main()