from flask import Flask, render_template, request, send_from_directory
import fitz  # PyMuPDF
import os
import tensorflow as tf
from keras.api.preprocessing import image
import numpy as np
from keras.api.layers import Input
from keras.api.models import Model
from keras.api.applications import VGG16
from keras.api.applications.inception_v3 import preprocess_input
from keras.api.applications.vgg16 import preprocess_input
from keras.api.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle
app = Flask(__name__)

with open("features_v3.pkl", "rb") as f:
    stored_features = pickle.load(f)
with open("features_16.pkl", "rb") as f:
    stored_features1 = pickle.load(f)
with open("features_50.pkl", "rb") as f:
    stored_features2 = pickle.load(f)
    
UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "extracted_images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("D:/Downloads/InV3_2.h5")

class_names = ["Charts", "Maps", "Others", "Photographs", "System_Model"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize ảnh về đúng kích thước mô hình
    img_array = image.img_to_array(img) # Chuẩn hóa về khoảng [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
    
def predict_class(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Lấy index lớp có xác suất cao nhất
    return class_names[predicted_class], predicted_class  # Trả về tên lớp và index

layer_name = 'global_average_pooling2d'
feature_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

def get_feature_vector_fromPIL(img):
    img_array = preprocess_image(img)
    feature_vector = feature_model.predict(img_array)
    assert(feature_vector.shape == (1,2048))
    return feature_vector

 

class SimiImage:
    def __init__(self):
        self.imagesimilar = []
    
    def update(self, similar, name, index, path):
        if similar >= 0.5:
            self.imagesimilar.append({
                "name" : name,
                "similar" : similar,
                "index" : index,
                "path" : path
            })
            
    def sort_by_similarity(self):
        self.imagesimilar.sort(key=lambda x: x["similar"], reverse=True)
        
    def get_image(self):
        return self.imagesimilar    
    
    def __len__(self):
        return len(self.imagesimilar)
    
    def __str__(self):
        """Hiển thị danh sách các ảnh vượt ngưỡng similarity."""
        result = "Các hình có similarity > 90%:\n"
        for image in self.imagesimilar:
            result += f'Image: {image["name"]}, Similarity: {image["similar"].item() * 100:.2f}%, Index: {image["index"]}, Path:{image["path"]}\n'
            
        return result if self.imagesimilar else "Không có hình nào vượt qua ngưỡng."      


def calculate_similarity_cosine(vector1, vector2):
 #return 1- distance.cosine(vector1, vector2)
    return cosine_similarity(vector1, vector2)



# image_similar1 = SimiImage()

# 



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "Không có file nào được tải lên!", 400

        files = request.files["image"]
        extracted_images = []

        if files:           
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], files.filename)
            files.save(img_path)

        # Chuẩn hóa ảnh
        images = preprocess_image(img_path)
        # Chuẩn đoán nhãn
        prediction = model.predict(images)
        predicted_class = np.argmax(prediction)  # Lấy index lớp có xác suất cao nhất
        class_name = class_names[predicted_class]  # Tên lớp dự đoán
        
        # class_folder = f"data/{class_name}"
        
        # img_paths=[]
        # for dataset in os.listdir(class_folder):
        #     full_path = os.path.join(class_folder, dataset)  # Đường dẫn đầy đủ
        #     img_paths.append(full_path)
        #     #img_data_list.append(preprocess_image(img_paths)) 
        stored_vectors = stored_features.get(class_name, [])  # Lấy đặc trưng của lớp dự đoán

        # file_count = len(img_paths)
        uploaded_feature = get_feature_vector_fromPIL(img_path)
        image_similar = SimiImage()
        file_count = len(stored_vectors)
        # for i in range(file_count):
        #     img_name = os.path.basename(img_paths[i])
        #     image_similarity_cosine = calculate_similarity_cosine(get_feature_vector_fromPIL(img_path), get_feature_vector_fromPIL(img_paths[i]))
        #     # print(f"Đường dẫn ảnh tương đồng: {img_paths[i]}")
        #     image_similar.update(image_similarity_cosine,img_name,i,img_paths[i])
        for i in range(file_count):
            img_name, feature_vector = stored_vectors[i]  # Lấy tên ảnh + feature vector
            similarity = calculate_similarity_cosine(uploaded_feature, feature_vector)
            image_similar.update(similarity, img_name, i, f"data/{class_name}/{img_name}")
        image_similar.sort_by_similarity()
        
        similar_images = image_similar.get_image()
        
        return render_template("index2.html", image=files.filename, result=class_name, confidence=np.max(prediction), similar_images=similar_images)

    return render_template("index2.html", image=None, result=None, confidence=None, similar_images=[])

@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
@app.route("/data/<path:filename>")
def get_similar_image(filename):
    return send_from_directory("data", filename)

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001/")  # Mở trình duyệt với địa chỉ Flask

if __name__ == "__main__":
    threading.Timer(1.25, open_browser).start()  # Đợi 1.25 giây rồi mở trình duyệt
    app.run(debug=True, port=5001)


