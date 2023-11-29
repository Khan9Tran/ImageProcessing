import streamlit as st

def show_info():

    # Thông tin về thành viên 1
    name_1 = "Trần Lâm Nhựt Khang"
    student_id_1 = "21110497"
    image_path_1 = "TranLamNhutKhang.jpg"  # Thay đổi đường dẫn ảnh của thành viên 1

    # Thông tin về thành viên 2
    name_2 = "Nguyễn Thanh Huy"
    student_id_2 = "21110473"
    image_path_2 = "NguyenThanhHuy.jpg"  # Thay đổi đường dẫn ảnh của thành viên 2

    # Tạo trang web bằng Streamlit
    st.title("Thông Tin Nhóm")
    st.balloons()
    st.info("Chào thầy! Đây là trang web của nhóm chúng em.", icon="🧑🏽‍💻")

    # Chia thành 2 cột
    col1, col2 = st.columns(2)

    # Hiển thị thông tin thành viên 1
    with col1:
        st.header(name_1)
        st.image(image_path_1, caption='Ảnh của {}'.format(name_1), use_column_width=True)
        st.text('Mã số sinh viên: {}'.format(student_id_1))

    # Hiển thị thông tin thành viên 2
    with col2:
        st.header(name_2)
        st.image(image_path_2, caption='Ảnh của {}'.format(name_2), use_column_width=True)
        st.text('Mã số sinh viên: {}'.format(student_id_2))
