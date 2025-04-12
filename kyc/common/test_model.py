import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
from datetime import datetime
import sys
from kyc.manager.face_network import FaceRecognitionNet  # Sửa import model

import cv2
import mediapipe as mp


def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'testing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load và tiền xử lý ảnh với Mediapipe"""

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")

    # Chuyển sang RGB vì Mediapipe yêu cầu
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Dùng Mediapipe để phát hiện khuôn mặt
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if not results.detections:
            raise ValueError(f"Không tìm thấy khuôn mặt trong ảnh {image_path}")

        # Lấy khuôn mặt đầu tiên (có thể chọn cái lớn nhất nếu nhiều)
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        h, w, _ = image.shape
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        bw = int(bboxC.width * w)
        bh = int(bboxC.height * h)

        # Mở rộng vùng cắt
        margin = 0.2
        x = max(0, int(x - bw * margin))
        y = max(0, int(y - bh * margin))
        bw = min(w - x, int(bw * (1 + 2 * margin)))
        bh = min(h - y, int(bh * (1 + 2 * margin)))

        face_image = image[y:y+bh, x:x+bw]

    # Resize
    face_image = cv2.resize(face_image, target_size)

    # Lọc nhiễu
    face_image = cv2.fastNlMeansDenoisingColored(face_image, None, 10, 10, 7, 21)

    # Chuyển sang RGB (PIL yêu cầu)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)

    # Chuyển đổi thành tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    face_tensor = transform(face_image).unsqueeze(0)
    return face_tensor, face_image


def extract_embedding(model, image_tensor, device):
    """Trích xuất embedding từ ảnh"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model(image_tensor)
        return embedding.cpu().numpy()


def print_embedding_stats(embedding, image_name):
    """In thống kê về embedding"""
    print(f"\nThống kê embedding cho ảnh {image_name}:")
    print(f"Shape: {embedding.shape}")
    print(f"Min value: {np.min(embedding):.4f}")
    print(f"Max value: {np.max(embedding):.4f}")
    print(f"Mean value: {np.mean(embedding):.4f}")
    print(f"Std value: {np.std(embedding):.4f}")
    print(f"First 10 values: {embedding[0][:10]}")


def compare_faces(embedding1, embedding2, threshold=0.7):
    """So sánh hai embedding và trả về kết quả"""
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    is_match = similarity > threshold
    return is_match, similarity


def main():
    logger = setup_logging()
    logger.info("Bắt đầu test model với ảnh thực tế...")

    try:
        # Thiết lập device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Sử dụng device: {device}")

        # Khởi tạo model
        logger.info("Đang khởi tạo model...")
        model = FaceRecognitionNet(embedding_size=512).to(device)

        # Load state dict
        models_dir = os.path.abspath("../models")
        logger.info("Đang load weights...")
        # state_dict = torch.load(models_dir + '/model.pt', map_location=device, weights_only=False)
        # model.load_state_dict(state_dict)
        model.eval()

        # Đường dẫn ảnh test
        # test_dir = os.path.abspath("../res/images")
        test_dir = os.path.abspath("C:/Users/Innotech_mobile13/Documents/Huit/kyc_be/debug_images")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            logger.error(f"Thư mục {test_dir} không tồn tại. Vui lòng thêm ảnh test vào thư mục này.")
            return

        # Lấy danh sách ảnh
        image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) < 2:
            logger.error("Cần ít nhất 2 ảnh để so sánh")
            return

        logger.info(f"Đường dẫn thư mục test: {test_dir}")
        logger.info(f"Số lượng ảnh tìm thấy: {len(image_files)}")

        # Tạo thư mục lưu kết quả
        results_dir = 'test_results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Xử lý từng cặp ảnh
        for i in range(0, len(image_files), 2):
            if i + 1 >= len(image_files):
                break

            image1_path = os.path.join(test_dir, image_files[i])
            image2_path = os.path.join(test_dir, image_files[i + 1])

            logger.info(f"\nSo sánh ảnh {image_files[i]} và {image_files[i + 1]}")

            try:
                # Xử lý ảnh 1
                image1_tensor, face1_image = load_and_preprocess_image(image1_path)
                embedding1 = extract_embedding(model, image1_tensor, device)
                print_embedding_stats(embedding1, image_files[i])

                # Xử lý ảnh 2
                image2_tensor, face2_image = load_and_preprocess_image(image2_path)
                embedding2 = extract_embedding(model, image2_tensor, device)
                print_embedding_stats(embedding2, image_files[i + 1])

                # So sánh embeddings
                predicted_match, similarity = compare_faces(embedding1, embedding2)

                # Lưu ảnh đã xử lý
                face1_image.save(os.path.join(results_dir, f'processed_{image_files[i]}'))
                face2_image.save(os.path.join(results_dir, f'processed_{image_files[i + 1]}'))

                # In kết quả chi tiết
                logger.info(f"\nKết quả so sánh:")
                logger.info(f"Độ tương đồng: {similarity:.4f}")
                logger.info(f"Kết quả: {'Cùng một người' if predicted_match else 'Khác người'}")

                # In khoảng cách Euclidean
                euclidean_dist = np.linalg.norm(embedding1 - embedding2)
                logger.info(f"Khoảng cách Euclidean: {euclidean_dist:.4f}")

                # In góc giữa hai vector
                angle = np.arccos(similarity) * 180 / np.pi
                logger.info(f"Góc giữa hai vector: {angle:.2f} độ")

            except Exception as e:
                logger.error(f"Lỗi khi xử lý cặp ảnh {image_files.__ge__(i)} và {image_files.__ge__(i+1)}: {str(e)}")
                continue

        logger.info("\nHoàn thành test model!")

    except Exception as e:
        logger.error(f"Lỗi chính trong chương trình: {str(e)}")
        raise e


if __name__ == '__main__':
    main()
