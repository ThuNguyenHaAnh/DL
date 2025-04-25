import cv2
import os
import numpy as np
"""
    Kiểm tra xem ảnh có phải là ảnh đen không.
    - threshold: Ngưỡng giá trị pixel được coi là đen (0-255, mặc định 10).
    - black_ratio: Tỷ lệ phần trăm pixel đen để coi là ảnh đen (mặc định 95%).
    """
def is_black_image(image_path, threshold=10, black_ratio=0.95):
    
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
    if image is None:
        return False, 0  # Nếu không mở được ảnh, không tính là ảnh đen
    
    total_pixels = image.size
    black_pixels = np.sum(image < threshold)  # Đếm số pixel có giá trị nhỏ hơn threshold
    black_percent = black_pixels / total_pixels  # Tính tỷ lệ pixel đen

    return black_percent >= black_ratio, black_percent  # Trả về True nếu ảnh gần như toàn màu đen

def remove_black_images(folder_path):
    """Xóa các ảnh đen trong thư mục và in tỷ lệ pixel đen của từng ảnh."""
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.abspath(os.path.join(folder_path, filename))
            is_black, black_percent = is_black_image(image_path)

            print(f"Ảnh: {filename}, Tỷ lệ pixel đen: {black_percent:.2%}")

            if is_black:
                print(f"==> Xóa ảnh đen: {filename}")
                os.remove(image_path)

# Thư mục chứa ảnh cần kiểm tra
image_folder = r"D:\CT\CT551\extracted_images"
remove_black_images(image_folder)
