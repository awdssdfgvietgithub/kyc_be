# import thư viện
import base64
import os
from io import BytesIO

import torch
import cv2
import matplotlib.pyplot as plt

import json

import pytesseract
from PIL import Image

from pytesseract import pytesseract
from ultralytics import YOLO
import re

import pytesseract

# tesseract_path = os.path.abspath("../res/Tesseract-OCR/tesseract.exe")
# print(tesseract_path)
# pytesseract.pytesseract.tesseract_cmd = tesseract_path
# print(pytesseract.get_tesseract_version())

class CCCDDataset:
    def __init__(self):
        self.data = {
            "current_place": "",
            "dob": "",
            "expire_date": "",
            "face": None,
            "gender": "",
            "id": "",
            "name": "",
            "nationality": "",
            "origin_place": ""
        }

    def update(self, extracted_data):
        self.data["current_place"] = " ".join(
            self.clean_texts(extracted_data.get("current_place1", [])) + self.clean_texts(
                extracted_data.get("current_place2", []))).strip()
        self.data["dob"] = " ".join(self.clean_texts(extracted_data.get("dob", []))).strip()
        self.data["expire_date"] = " ".join(self.clean_texts(extracted_data.get("expire_date", []))).strip()
        self.data["face"] = extracted_data.get("face", [])
        self.data["gender"] = " ".join(self.clean_texts(extracted_data.get("gender", []))).strip()
        self.data["id"] = " ".join(self.clean_texts(extracted_data.get("id", []))).strip()
        self.data["name"] = " ".join(self.clean_texts(extracted_data.get("name", []))).strip()
        self.data["nationality"] = " ".join(self.clean_texts(extracted_data.get("nationality", []))).strip()
        self.data["origin_place"] = " ".join(self.clean_texts(extracted_data.get("origin_place", []))).strip()

    def clean_texts(self, texts):
        """Remove unwanted characters from OCR results"""
        s = [re.sub(r'[:_\'"~`“”-]', '', text) for text in texts]

        return s

    def to_json(self):
        return json.dumps(self.data, indent=4, ensure_ascii=False)


def load_model(model_path):
    """Load YOLO model from the given path"""
    model = YOLO(model_path)
    return model


def detect_objects(model, image_path):
    """Run inference on an image and return results"""
    results = model(image_path)
    return results


def extract_objects(image_path, results):
    """Extract detected objects, convert to binary image, apply OCR, and return as a dictionary"""
    image = cv2.imread(image_path)
    extracted_data = {}

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

def convert_image_to_base64(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str

def display_json(json_data):
    """Display JSON data in a readable format"""
    print(json.dumps(json.loads(json_data), indent=4, ensure_ascii=False))


def display_detected_objects(image_path, results):
    """Display the detected objects on the image"""
    image = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# # Path to your trained model
# models_path = os.path.abspath("../models")
# model = load_model(models_path + "/yolov11.pt")
#
# # Path to an example image
# image_path = os.path.abspath("../res/images/id_card_1.jpg")
#
# # Run detection
# results = detect_objects(model, image_path)
#
# # Extract data and store in CCCDDataset
# extracted_data = extract_objects(image_path, results)
# cccd_dataset = CCCDDataset()
# cccd_dataset.update(extracted_data)
# cccd_dataset.data["id"]
#
# display_json(cccd_dataset.to_json())
