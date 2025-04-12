import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_face(image):
    """Phát hiện và cắt khuôn mặt từ ảnh"""
    try:
        # Khởi tạo detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Chuyển ảnh sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        elif len(faces) > 1:
            logger.warning("Multiple faces detected, using the first one")

        # Lấy khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]

        # Thêm padding xung quanh khuôn mặt
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # Cắt khuôn mặt
        face_img = image[y1:y2, x1:x2]

        # Vẽ rectangle để debug
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return face_img, debug_img

    except Exception as e:
        logger.error(f"Error detecting face: {str(e)}")
        raise


def load_image(image_path, target_size=(96, 96), should_detect_face=True):
    """Load và preprocess ảnh"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Phát hiện và cắt khuôn mặt nếu cần
        if should_detect_face:
            face_img, debug_img = detect_face(img)
            img = face_img

        # Resize ảnh
        img = cv2.resize(img, target_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise


def get_embedding(model, image1, image2):
    """Trích xuất embedding từ cặp ảnh"""
    try:
        # Thêm batch dimension
        image1 = np.expand_dims(image1, axis=0)
        image2 = np.expand_dims(image2, axis=0)

        # Lấy embedding từ mô hình
        embedding = model.predict([image1, image2], verbose=0)

        return embedding[0][0]  # Lấy giá trị đầu tiên của kết quả dự đoán

    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise


def calculate_distance(embedding1, embedding2, distance_type='euclidean'):
    """Tính khoảng cách giữa hai vector embedding"""
    try:
        if distance_type == 'euclidean':
            # Khoảng cách Euclidean
            distance = np.linalg.norm(embedding1 - embedding2)
        elif distance_type == 'cosine':
            # Khoảng cách Cosine (1 - cosine similarity)
            distance = 1 - np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

        return distance
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        raise


def create_test_results_dir():
    """Tạo thư mục test_results nếu chưa tồn tại"""
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def create_combined_image(img1_path, img2_path, similarity, threshold, is_same_person):
    """Tạo ảnh kết hợp từ hai ảnh đầu vào và thêm thông tin kết quả"""
    try:
        # Load ảnh gốc
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)

        # Phát hiện khuôn mặt và lấy ảnh debug
        _, debug_img1 = detect_face(img1_orig)
        _, debug_img2 = detect_face(img2_orig)

        # Resize ảnh về cùng kích thước
        target_size = (300, 300)
        debug_img1 = cv2.resize(debug_img1, target_size)
        debug_img2 = cv2.resize(debug_img2, target_size)

        # Tạo ảnh kết hợp
        combined = np.hstack((debug_img1, debug_img2))

        # Thêm thông tin vào ảnh
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (0, 255, 0) if is_same_person else (0, 0, 255)

        # Thêm similarity score
        similarity_text = f"Similarity: {similarity:.4f}"
        cv2.putText(combined, similarity_text, (10, 30), font, font_scale, text_color, font_thickness)

        # Thêm threshold
        threshold_text = f"Threshold: {threshold}"
        cv2.putText(combined, threshold_text, (10, 60), font, font_scale, text_color, font_thickness)

        # Thêm kết quả
        result_text = "Same Person: Yes" if is_same_person else "Same Person: No"
        cv2.putText(combined, result_text, (10, 90), font, font_scale, text_color, font_thickness)

        # Thêm timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_text = f"Test Time: {timestamp}"
        cv2.putText(combined, timestamp_text, (10, 120), font, font_scale, (255, 255, 255), font_thickness)

        return combined
    except Exception as e:
        logger.error(f"Error creating combined image: {str(e)}")
        raise


def save_test_result(results_dir, combined_img, img1_path, img2_path):
    """Lưu kết quả test"""
    try:
        # Tạo tên file từ tên của hai ảnh đầu vào
        img1_name = Path(img1_path).stem
        img2_name = Path(img2_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"result_{img1_name}_{img2_name}_{timestamp}.jpg")

        # Lưu ảnh
        cv2.imwrite(output_path, combined_img)
        logger.info(f"Saved test result to: {output_path}")

        return output_path
    except Exception as e:
        logger.error(f"Error saving test result: {str(e)}")
        raise


def compare_faces(model_path, image1_path, image2_path, threshold=0.5, distance_type='euclidean'):
    """So sánh hai ảnh khuôn mặt"""
    try:
        # Load mô hình
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load và preprocess ảnh (có phát hiện khuôn mặt)
        img1 = load_image(image1_path, should_detect_face=True)
        img2 = load_image(image2_path, should_detect_face=True)

        # Dự đoán trực tiếp từ mô hình
        similarity = model.predict([np.expand_dims(img1, axis=0),
                                    np.expand_dims(img2, axis=0)],
                                   verbose=0)[0][0]

        # Đưa ra quyết định
        is_same_person = similarity > threshold

        # Tạo thư mục kết quả
        results_dir = create_test_results_dir()

        # Tạo và lưu ảnh kết hợp
        combined_img = create_combined_image(image1_path, image2_path,
                                             similarity, threshold, is_same_person)
        result_path = save_test_result(results_dir, combined_img,
                                       image1_path, image2_path)

        # In kết quả
        logger.info("\nComparison Results:")
        logger.info(f"Image 1: {image1_path}")
        logger.info(f"Image 2: {image2_path}")
        logger.info(f"Similarity score: {similarity:.4f}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Same person: {'Yes' if is_same_person else 'No'}")
        logger.info(f"Result saved to: {result_path}")

        return {
            'similarity': similarity,
            'is_same_person': is_same_person,
            'threshold': threshold,
            'result_path': result_path
        }

    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        raise


def main():
    try:
        # Tạo parser cho command line arguments
        parser = argparse.ArgumentParser(description='Face comparison using Siamese Network')
        parser.add_argument('--model', type=str, required=True, help='Path to the model file')
        parser.add_argument('--image1', type=str, required=True, help='Path to the first image')
        parser.add_argument('--image2', type=str, required=True, help='Path to the second image')
        parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for comparison')
        parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                            help='Distance metric to use')

        args = parser.parse_args()

        # Kiểm tra đường dẫn
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")
        if not os.path.exists(args.image1):
            raise FileNotFoundError(f"Image 1 not found: {args.image1}")
        if not os.path.exists(args.image2):
            raise FileNotFoundError(f"Image 2 not found: {args.image2}")

        # So sánh hai ảnh
        results = compare_faces(args.model, args.image1, args.image2,
                                args.threshold, args.distance)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()