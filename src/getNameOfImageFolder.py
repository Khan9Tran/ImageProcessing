import os

def get_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.name)
            subdirectories.extend(get_subdirectories(entry.path))
    return subdirectories

def main():
    # Nhập thư mục gốc từ người dùng
    root_directory = input("Nhập đường dẫn thư mục gốc: ")

    # Kiểm tra xem thư mục gốc có tồn tại hay không
    if not os.path.exists(root_directory):
        print("Thư mục gốc không tồn tại.")
        return

    # Gọi hàm để lấy tất cả các thư mục con và lưu vào một mảng
    subdirectories = get_subdirectories(root_directory)

    # Kiểm tra xem có thư mục con nào hay không
    if len(subdirectories) == 0:
        print("Không có thư mục con trong thư mục gốc.")
    else:
        # In kết quả
        print("Các thư mục con:")
        print(subdirectories)
if __name__ == '__main__':
    main()