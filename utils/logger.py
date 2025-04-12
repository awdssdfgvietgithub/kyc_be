import base64
import logging
from io import BytesIO

import cv2
from PIL import Image


def get_logger(name: str, file_path: str = None):
    logger = logging.getLogger(name)

    # Xoá tất cả handler cũ để đảm bảo setup lại hoàn toàn
    if logger.hasHandlers():
        logger.handlers.clear()

    # Tạo format chuẩn
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Log ra terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Log ra file nếu có
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def convert_image_to_base64(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str
