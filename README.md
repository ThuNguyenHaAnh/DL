# Dự án: Xây dựng hệ thống lưu trữ, phân loại và tìm kiếm hình ảnh được trích xuất từ tài liệu PDF dựa trên học sâu

## Mô tả
Dự án này nhằm mục đích xây dựng một hệ thống có khả năng lưu trữ, phân loại và tìm kiếm hình ảnh được trích xuất từ các tài liệu PDF. Hệ thống sử dụng các mô hình học sâu như InceptionV3, VGG16, và ResNet50 để phân loại hình ảnh và tìm kiếm sự tương đồng giữa các hình ảnh.

## Các tính năng chính
- **Phân loại hình ảnh**: Dựa trên các mô hình học sâu, hệ thống có thể phân loại các hình ảnh vào các nhóm khác nhau như: ảnh chụp, bản đồ, mô hình hệ thống, đồ thị,...
- **Tìm kiếm hình ảnh**: Người dùng có thể tìm kiếm các hình ảnh tương tự dựa trên mô hình học sâu.

## Công nghệ sử dụng
- Python
- TensorFlow, Keras
- OpenCV
- Flask (cho giao diện web)

## Sử dụng
Để sử dụng hệ thống, bạn có thể tải lên các tài liệu PDF chứa hình ảnh và hệ thống sẽ tự động trích xuất hình ảnh và phân loại chúng.

## Giới thiệu mô hình học sâu
Dự án sử dụng các mô hình học sâu (InceptionV3, VGG16, ResNet50) để phân loại và tìm kiếm hình ảnh. Các mô hình này đã được huấn luyện trên bộ dữ liệu hình ảnh lớn và được tối ưu hóa để hoạt động hiệu quả trong việc phân loại và tìm kiếm hình ảnh.
## Tải Mô Hình
Bạn có thể tải mô hình đã huấn luyện từ Google Drive bằng cách nhấp vào liên kết dưới đây:
- [Tải mô hình InceptionV3](https://drive.google.com/file/d/1PmkEcH8cuZu1f9JfG9JXRrHCOLKEi2nu/view?usp=drive_link)
- [Tải mô hình VGG16](https://drive.google.com/file/d/1--Z470IXfFd2qTillqGOjk50QYgNqMjT/view?usp=drive_link)
- [Tải mô hình ResNet50](https://drive.google.com/file/d/11XjvKPgj_JaWxslU9NHkB6KRcDW8Ay6z/view?usp=drive_link)
