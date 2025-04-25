from flask import Flask, render_template, request, send_from_directory
import fitz  # PyMuPDF
import os
import sys
import cv2
import numpy as np
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "extracted_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_black_image(image_path, threshold=10, black_ratio=0.95):
    
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
    if image is None:
        return False, 0  # Nếu không mở được ảnh, không tính là ảnh đen
    
    total_pixels = image.size
    black_pixels = np.sum(image < threshold)  # Đếm số pixel có giá trị nhỏ hơn threshold
    black_percent = black_pixels / total_pixels  # Tính tỷ lệ pixel đen

    return black_percent >= black_ratio, black_percent

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    # Mở file PDF
    doc = fitz.open(pdf_path)

    # Tạo thư mục để lưu ảnh nếu chưa có
    #os.makedirs(output_folder, exist_ok=True)

    image_paths = []  # Danh sách lưu đường dẫn ảnh đã trích xuất
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Duyệt qua từng trang trong PDF
    for page_number in range(len(doc)):
        page = doc[page_number]  # Lấy trang hiện tại

        # Lấy danh sách ảnh trong trang
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            
            if page_number == 0 and img_index == 0:
                # print(f"Bỏ ảnh đầu tiên của {pdf_name} - Trang {page_number}")
                continue
            xref = img[0]  # ID của ảnh trong tài liệu PDF
            base_image = doc.extract_image(xref)  # Trích xuất ảnh
            img_bytes = base_image["image"]  # Dữ liệu ảnh
            img_ext = base_image["ext"]  # Định dạng ảnh (png, jpeg,...)

            
            # Tạo tên file ảnh
            img_filename = f"{output_folder}/{pdf_name}_page_{page_number}_img_{img_index}.{img_ext}"
            
            # Lưu ảnh ra file
            with open(img_filename, "wb") as img_file:
                img_file.write(img_bytes)
                
            is_black, percent = is_black_image(img_filename)
            if is_black:
                print(f"❌ Ảnh đen ({percent*100:.2f}%) - đã xóa: {img_filename}")
                os.remove(img_filename)
                continue
            
            image_paths.append(img_filename)

    return image_paths

def extract_images_from_folder(pdf_folder, output_folder="extracted_images"):
    """Trích xuất ảnh từ tất cả các file PDF trong thư mục."""
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]  # Lọc danh sách file PDF
    all_images = {}  # Dictionary lưu danh sách ảnh từ mỗi file PDF
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Đang xử lý: {pdf_file}...")  # In ra file đang xử lý
        images = extract_images_from_pdf(pdf_path, output_folder)
        all_images[pdf_file] = images  # Lưu danh sách ảnh của file PDF đó
    return all_images

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdfs" not in request.files:
            return "Không có file nào được tải lên!", 400

        files = request.files.getlist("pdfs")
        extracted_images = []

        for file in files:
            if file.filename.endswith(".pdf"):
                safe_filename = os.path.basename(file.filename)
                pdf_path = os.path.join(UPLOAD_FOLDER, safe_filename)
                file.save(pdf_path)

                # Trích xuất ảnh từ thư mục
        images = extract_images_from_folder(UPLOAD_FOLDER, OUTPUT_FOLDER)
        extracted_images.extend(sum(images.values(), []))  # Gộp danh sách ảnh

        return render_template("index.html", images=extracted_images)

    return render_template("index.html", images=[])

@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")  # Mở trình duyệt với địa chỉ Flask

if __name__ == "__main__":
    
    app.run(debug=True, port=5003)


