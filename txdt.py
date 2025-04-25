import os
import numpy as np
import pickle
from keras.api.preprocessing import image
from keras.api.applications.inception_v3 import InceptionV3
from keras.api.models import Model
import tensorflow as tf
# Load mô hình InceptionV3 đã huấn luyện
model = tf.keras.models.load_model("D:/Downloads/inceptionv3_best3.h5")
feature_model = Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)

class_names = ["Charts", "Maps", "Others", "Photographs", "System_Model"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(img_path):
    img_array = preprocess_image(img_path)
    features = feature_model.predict(img_array)
    return features  # Chuyển về vector 1D

def save_features(data_folder, output_file):
    feature_dict = {class_name: [] for class_name in class_names}
    image_paths = {class_name: [] for class_name in class_names}

    for class_name in class_names:
        class_path = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_path):
            continue  # Bỏ qua nếu không phải thư mục
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(class_path, img_name)
                feature_vector = extract_features(img_path)
                
                feature_dict[class_name].append(feature_vector)  # Lưu feature dạng (1, 2048)
                image_paths[class_name].append(img_name)  # Lưu tên ảnh
    
    # Chuyển danh sách về dạng ma trận NumPy
    for class_name in class_names:
        if feature_dict[class_name]:
            feature_dict[class_name] = np.vstack(feature_dict[class_name])  # Chuyển (N, 1, 2048) → (N, 2048)
    
    
    # Lưu dữ liệu vào file pickle
    with open(output_file, "wb") as f:
        pickle.dump(feature_dict, f)

    print(f"✅ Đã lưu đặc trưng vào {output_file}")

# Chạy script lưu vector đặc trưng của toàn bộ dataset
save_features("data", "features.pkl")
