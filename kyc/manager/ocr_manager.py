import os

import cv2
import numpy as np
import pytesseract
from pytesseract import pytesseract
from ultralytics import YOLO

from utils.logger import convert_image_to_base64


# tesseract_path = os.path.abspath("../kyc_be/kyc/res/Tesseract-OCR/tesseract.exe")
# model_ocr_path = os.path.abspath("../kyc_be/kyc/models/yolov11.pt")
# model_face_check_path = os.path.abspath("../kyc_be/kyc/models/model.pt")
#
# print(tesseract_path)
# print(model_ocr_path)
# print(model_face_check_path)
#
# pytesseract.tesseract_cmd = tesseract_path


def load_model(model_path):
    """Load YOLO model from the given path"""
    model = YOLO(model_path)
    return model


def detect_objects(model, image_path):
    """Run inference on an image and return results"""
    results = model(image_path)
    return results


def extract_objects(image, results):
    """Extract detected objects, convert to binary image, apply OCR, and return as a dictionary"""
    extracted_data = {}

    # image là đối tượng PIL, chuyển thành numpy.ndarray
    image_np = np.array(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            if label == "face":
                cropped_obj = image[y1:y2, x1:x2]
                if "face" not in extracted_data:
                    extracted_data["face"] = []
                # extracted_data["face"].append([x1, y1, x2, y2])
                face_b64 = convert_image_to_base64(cropped_obj)
                extracted_data["face"].append(face_b64)
            else:
                cropped_obj = image[y1:y2, x1:x2]

                # Convert to grayscale
                gray_obj = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2GRAY)

                # Convert to binary (thresholding)
                _, binary_obj = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Apply OCR
                text = pytesseract.image_to_string(binary_obj, lang='vie', config='--psm 6').strip()

                if label not in extracted_data:
                    extracted_data[label] = []
                extracted_data[label].append(text)

    return extracted_data
