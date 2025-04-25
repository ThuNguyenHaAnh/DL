import tensorflow as tf
import numpy as np
from keras.api.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from keras.api.applications.inception_v3 import preprocess_input
# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("D:/Downloads/InV3_2.h5")

# Danh sách các lớp (Thay bằng tên lớp của bạn)
class_names = ["Charts", "Maps", "Others", "Photographs", "System_Model"]

# Đọc và xử lý ảnh
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title="Chọn file ảnh",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.jfif")])  # Đường dẫn đến ảnh cần test

img = image.load_img(img_path, target_size=(299, 299))  # Resize ảnh về đúng kích thước mô hình
img_array = image.img_to_array(img)  # Chuẩn hóa về khoảng [0,1]
img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

img_array = preprocess_input(img_array) 
# Dự đoán
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)  # Lấy index lớp có xác suất cao nhất
class_name = class_names[predicted_class]  # Tên lớp dự đoán


# Hiển thị ảnh và kết quả
plt.imshow(img)
plt.axis("off")
plt.title(f"Dự đoán: {class_name} (Xác suất: {np.max(prediction):.2f})")
plt.show()
