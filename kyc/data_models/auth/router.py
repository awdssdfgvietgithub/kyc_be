import base64
import io
import os
import time
import uuid

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import APIRouter, HTTPException, status, Depends
from pytesseract import pytesseract
from requests import Session
from starlette.responses import JSONResponse

from kyc.manager.face_network import FaceRecognitionNet  # Sửa import model
from . import schema
from .. import db
from ..user import validator
from ..user.validator import get_all_users
from ...common.test_model import setup_logging, extract_embedding, print_embedding_stats, compare_faces
from ...common.test_ocr_model import CCCDDataset
from ...manager.face_recognition import load_and_preprocess_image
from ...manager.ocr_manager import load_model, detect_objects, extract_objects
from utils.logger import get_logger

import logging
from datetime import datetime

# logger = logging.getLogger("ocr_logon_logger")
# logger.setLevel(logging.DEBUG)
#
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

tesseract_path = os.path.abspath("../kyc_be/kyc/res/Tesseract-OCR/tesseract.exe")
model_ocr_path = os.path.abspath("../kyc_be/kyc/models/yolov11.pt")
model_face_check_path = os.path.abspath("../kyc_be/kyc/models/best_model.pt")

print(tesseract_path)
print(model_ocr_path)
print(model_face_check_path)

pytesseract.tesseract_cmd = tesseract_path

router = APIRouter(tags=['Auth'], prefix='/auth')

def fix_base64_padding(b64_string: str) -> str:
    """Ensure base64 string is padded properly to avoid decode errors."""
    return b64_string + '=' * ((4 - len(b64_string) % 4) % 4)


@router.post('/ocr_id_card_enrollment', status_code=status.HTTP_200_OK)
async def execute_ocr_id_card_enrollment(request: schema.OCRIDCardEnrollment, database: Session = Depends(db.get_db)):
    function_name = "ocr_id_card_enrollment"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/{timestamp}-{function_name}"
    os.makedirs(log_dir, exist_ok=True)

    request_id = datetime.utcnow().isoformat()
    start_time = time.time()
    log_file_path = os.path.join(log_dir, "log.txt")
    logger = get_logger("ocr_id_card_enrollment", file_path=log_file_path)

    logger.info(f"[{request_id}] ⚡ START execute_ocr_id_card_enrollment")

    try:
        if not request.front_id_card_base64:
            logger.warning(f"[{request_id}] ❌ front_id_card_base64 is empty.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "front_id_card_base64 is empty"}
            )

        # Load model
        logger.info(f"[{request_id}] 📦 Loading OCR model from: {model_ocr_path}")
        model = load_model(model_ocr_path)

        try:
            img_data = base64.b64decode(request.front_id_card_base64)
            image = Image.open(io.BytesIO(img_data))
            image.save(f"{log_dir}/image.png")
            logger.info(f"[{request_id}] 🖼️ Successfully decoded image from base64.")
        except Exception as e:
            logger.warning(f"[{request_id}] ❌ Invalid front image: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Ảnh trước không hợp lệ hoặc bị lỗi định dạng."}
            )

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{log_dir}/image_cv.png", image_cv)
        logger.info(f"[{request_id}] 🤖 Running OCR detection...")

        results = detect_objects(model, image_cv)
        extracted_data = extract_objects(image_cv, results)
        logger.info(f"[{request_id}] 📄 Extracted data: {extracted_data}")

        cccd_dataset = CCCDDataset()
        cccd_dataset.update(extracted_data)

        if not cccd_dataset.data["id"]:
            logger.warning(f"[{request_id}] ❌ Unable to extract ID from image.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Unable to extract information from the image."}
            )

        if not cccd_dataset.data["face"]:
            logger.warning(f"[{request_id}] ❌ Unable to extract Face from image.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Unable to extract Face from the image."}
            )

        logger.info(f"[{request_id}] 🔍 Verifying if ID {cccd_dataset.data['id']} exists...")
        user = await validator.verify_id_card_exist(cccd_dataset.data["id"], database)

        if user:
            logger.warning(f"[{request_id}] ❌ ID already exists in system: {user}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "The user with this id card already exists in the system."}
            )

        face_base64 = cccd_dataset.data["face"][0]
        img_data_one = base64.b64decode(face_base64)
        image_one = Image.open(io.BytesIO(img_data_one))
        image_one.save(f"{log_dir}/image_one.png")

        logger.info(f"[{request_id}] 🧠 Initializing face recognition model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceRecognitionNet(embedding_size=512).to(device)

        try:
            state_dict = torch.load(model_face_check_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Cannot set model to eval mode: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Không thể tải model nhận diện khuôn mặt."}
            )

        users = await get_all_users(database)
        match_found = False

        logger.info(f"[{request_id}] 👥 Comparing with {len(users)} users in DB...")

        for user2 in users:
            try:
                img_data_two = base64.b64decode(user2.front_face_base64)
                image_two = Image.open(io.BytesIO(img_data_two))

                # image1_tensor, face1_image = load_and_preprocess_image(image, id="origin")
                image1_tensor, face1_image = load_and_preprocess_image(image_one, id="origin")
                embedding1 = extract_embedding(model, image1_tensor, device)
                print_embedding_stats(embedding1, "One")

                image2_tensor, face2_image = load_and_preprocess_image(image_two, id=user2.id)
                embedding2 = extract_embedding(model, image2_tensor, device)
                print_embedding_stats(embedding2, "Two")

                predicted_match, similarity = compare_faces(embedding1, embedding2, 0.8)
                logger.info(
                    f"[{request_id}] 🧬 Compared with user {user2.id}, similarity: {similarity:.4f}, match: {predicted_match}")

                if predicted_match:
                    match_found = True
                    break

            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Error comparing with user {user2.id}: {str(e)}")
                continue

        if match_found:
            logger.warning(f"[{request_id}] ❌ Found duplicate face in system.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Đã có thông tin trong hệ thống"}
            )

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] ✅ END execute_ocr_id_card_enrollment (Success) ⏱️ {elapsed:.2f}s")
        return cccd_dataset

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{request_id}] ❗ Internal Server Error: {str(e)}")
        logger.info(f"[{request_id}] ❌ END execute_ocr_id_card_enrollment (Exception) ⏱️ {elapsed:.2f}s")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Interal Server Error {str(e)}"}
        )


@router.post('/ocr_id_card_logon', status_code=status.HTTP_200_OK)
async def execute_ocr_id_card_logon(request: schema.OCRIDCardLogon, database: Session = Depends(db.get_db)):
    function_name = "ocr_id_card_logon"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/{timestamp}-{function_name}"
    os.makedirs(log_dir, exist_ok=True)

    request_id = datetime.utcnow().isoformat()
    start_time = time.time()
    log_file_path = os.path.join(log_dir, "log.txt")
    logger = get_logger("ocr_id_card_logon", file_path=log_file_path)

    logger.info(f"[{request_id}] ⚡ START execute_ocr_id_card_logon")

    try:
        if not request.front_id_card_base64:
            logger.warning(f"[{request_id}] ❌ front_id_card_base64 is empty.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "front_id_card_base64 is empty"}
            )

        logger.info(f"[{request_id}] 📦 Loading OCR model from: {model_ocr_path}")
        model = load_model(model_ocr_path)

        try:
            # fixed_data = fix_base64_padding(request.front_id_card_base64)
            # img_data = base64.b64decode(fixed_data)
            img_data = base64.b64decode(request.front_id_card_base64)
            image = Image.open(io.BytesIO(img_data))
            image.save(f"{log_dir}/image.png")
            logger.info(f"[{request_id}] 🖼️ Successfully decoded image from base64.")
        except Exception as e:
            logger.warning(f"[{request_id}] ❌ Invalid image format: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Ảnh không hợp lệ hoặc bị lỗi định dạng."}
            )

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{log_dir}/image_cv.png", image_cv)
        logger.info(f"[{request_id}] 🤖 Running OCR detection...")

        results = detect_objects(model, image_cv)
        extracted_data = extract_objects(image_cv, results)
        logger.info(f"[{request_id}] 📄 Extracted data: {extracted_data}")

        cccd_dataset = CCCDDataset()
        cccd_dataset.update(extracted_data)

        if not cccd_dataset.data["id"]:
            logger.warning(f"[{request_id}] ❌ Unable to extract ID from image.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Unable to extract information from the image."}
            )

        logger.info(f"[{request_id}] 🔍 Verifying user by ID: {cccd_dataset.data['id']}")
        user = await validator.verify_id_card_exist(cccd_dataset.data["id"], database)

        if user:
            logger.info(f"[{request_id}] ✅ Found matching user in database: {user.id}")
            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] ✅ END execute_ocr_id_card_logon (Success) ⏱️ {elapsed:.2f}s")
            return user
        else:
            logger.warning(f"[{request_id}] ❌ No user found with provided ID.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "User info is not in System."}
            )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{request_id}] ❗ Internal Server Error: {str(e)}")
        logger.info(f"[{request_id}] ❌ END execute_ocr_id_card_logon (Exception) ⏱️ {elapsed:.2f}s")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Interal Server Error {str(e)}"}
        )


@router.post('/liveness_enrollment', status_code=status.HTTP_200_OK)
async def execute_check_face_enrollment(request: schema.CheckFaceEnrollment):
    function_name = "liveness_enrollment"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/{timestamp}-{function_name}"
    os.makedirs(log_dir, exist_ok=True)

    request_id = datetime.utcnow().isoformat()
    start_time = time.time()
    log_file_path = os.path.join(log_dir, "log.txt")
    logger = get_logger("liveness_enrollment", file_path=log_file_path)

    logger.info(f"[{request_id}] 🚀 START liveness enrollment check")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[{request_id}] 📦 Using device: {device}")

        logger.info(f"[{request_id}] ⚙️ Initializing model...")
        model = FaceRecognitionNet(embedding_size=512).to(device)

        logger.info(f"[{request_id}] 📥 Loading model weights...")
        try:
            state_dict = torch.load(model_face_check_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error loading model: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Không thể tải model nhận diện khuôn mặt."}
            )

        image_req_one = request.front_id_card_base64
        image_req_two = request.front_face_base64

        if not image_req_one or not image_req_two:
            logger.error(f"[{request_id}] ❌ Thiếu ảnh CMND hoặc ảnh khuôn mặt.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Thiếu ảnh CMND hoặc ảnh khuôn mặt."}
            )

        try:
            img_data_one = base64.b64decode(image_req_one)
            img_data_two = base64.b64decode(image_req_two)
            image_one = Image.open(io.BytesIO(img_data_one))
            image_two = Image.open(io.BytesIO(img_data_two))
            image_one.save(f"{log_dir}/image_one.png")
            image_two.save(f"{log_dir}/image_two.png")
            logger.info(f"[{request_id}] 🖼️ Successfully decoded both images.")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error decoding/opening images: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Ảnh không hợp lệ hoặc bị lỗi định dạng."}
            )

        try:
            logger.info(f"[{request_id}] 🔍 Extracting embedding from ID card image...")
            image1_tensor, _ = load_and_preprocess_image(image_one, id="id_card")
            embedding1 = extract_embedding(model, image1_tensor, device)

            logger.info(f"[{request_id}] 🔍 Extracting embedding from face image...")
            image2_tensor, _ = load_and_preprocess_image(image_two, id="face")
            embedding2 = extract_embedding(model, image2_tensor, device)

            logger.info(f"[{request_id}] 🔄 Comparing embeddings...")
            predicted_match, similarity = compare_faces(embedding1, embedding2, 0.4)

            euclidean_dist = np.linalg.norm(embedding1 - embedding2)
            angle = np.arccos(similarity) * 180 / np.pi

            logger.info(f"[{request_id}] 📊 Similarity: {similarity:.4f}")
            logger.info(f"[{request_id}] 📏 Euclidean Distance: {euclidean_dist:.4f}")
            logger.info(f"[{request_id}] 📐 Angle (degree): {angle:.2f}")
            logger.info(f"[{request_id}] ✅ Match result: {'Cùng một người' if predicted_match else 'Không cùng người'}")

            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] ✅ END liveness enrollment (Success) ⏱️ {elapsed:.2f}s")

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "is_same_person": int(predicted_match),
                    "similarity": float(similarity*100),
                    "euclidean_distance": float(euclidean_dist),
                    "angle_degree": float(angle),
                    "message": "Cùng một người" if predicted_match else "Không cùng một người"
                }
            )

        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error during image processing or comparison: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Lỗi khi xử lý ảnh và so sánh khuôn mặt."}
            )

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Lỗi chính trong chương trình: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Lỗi nội bộ trong hệ thống."}
        )


@router.post('/liveness_logon', status_code=status.HTTP_200_OK)
async def execute_check_face_logon(request: schema.CheckFaceLogon, database: Session = Depends(db.get_db)):
    function_name = "liveness_logon"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/{timestamp}-{function_name}"
    os.makedirs(log_dir, exist_ok=True)

    request_id = datetime.utcnow().isoformat()
    start_time = time.time()
    log_file_path = os.path.join(log_dir, "log.txt")
    logger = get_logger("liveness_logon", file_path=log_file_path)

    logger.info(f"[{request_id}] ⚡ START execute_check_face_logon")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[{request_id}] 📍 Sử dụng device: {device}")

        logger.info(f"[{request_id}] 📦 Khởi tạo model nhận diện khuôn mặt...")
        model = FaceRecognitionNet(embedding_size=512).to(device)

        logger.info(f"[{request_id}] 🧠 Đang load weights...")
        try:
            state_dict = torch.load(model_face_check_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Lỗi khi load model: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Không thể tải model nhận diện khuôn mặt."}
            )

        image_req = request.front_face_base64
        if not image_req:
            logger.warning(f"[{request_id}] ⚠️ Yêu cầu thiếu dữ liệu ảnh.")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Thiếu ảnh CMND hoặc ảnh khuôn mặt."}
            )

        try:
            img_data_one = base64.b64decode(image_req)
            image_one = Image.open(io.BytesIO(img_data_one))
            image_one.save(f"{log_dir}/image_one.png")
            logger.info(f"[{request_id}] 🖼️ Ảnh được giải mã thành công.")
        except Exception as e:
            logger.warning(f"[{request_id}] ❌ Lỗi khi giải mã hoặc mở ảnh: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Ảnh không hợp lệ hoặc bị lỗi định dạng."}
            )

        logger.info(f"[{request_id}] 📥 Truy vấn danh sách người dùng từ database...")
        users = await get_all_users(database)
        logger.info(f"[{request_id}] 🔁 Tổng số người dùng: {len(users)}")

        match_found = False

        for user in users:
            logger.info(f"[{request_id}] 🧪 So sánh với user ID: {user.id}")
            try:
                img_data_two = base64.b64decode(user.front_face_base64)
                image_two = Image.open(io.BytesIO(img_data_two))
                image_two.save(f"{log_dir}/image_two_{user.id_card}.png")

                image1_tensor, face1_image = load_and_preprocess_image(image_one, id="origin")
                embedding1 = extract_embedding(model, image1_tensor, device)

                image2_tensor, face2_image = load_and_preprocess_image(image_two, id=user.id)
                embedding2 = extract_embedding(model, image2_tensor, device)

                predicted_match, similarity = compare_faces(embedding1, embedding2)

                euclidean_dist = np.linalg.norm(embedding1 - embedding2)
                angle = np.arccos(similarity) * 180 / np.pi

                logger.info(f"[{request_id}] 🔗 Độ tương đồng: {similarity:.4f}")
                logger.info(f"[{request_id}] 📏 Khoảng cách Euclidean: {euclidean_dist:.4f}")
                logger.info(f"[{request_id}] 📐 Góc giữa hai vector: {angle:.2f}°")
                logger.info(f"[{request_id}] 📊 Kết quả: {'✅ MATCH' if predicted_match else '❌ NO MATCH'}")

                if predicted_match:
                    match_found = True
                    elapsed = time.time() - start_time
                    logger.info(f"[{request_id}] ✅ Người dùng khớp: {user.id}")
                    logger.info(f"[{request_id}] ✅ END execute_check_face_logon (Matched) ⏱️ {elapsed:.2f}s")
                    return user

            except Exception as e:
                logger.error(f"[{request_id}] ❌ Lỗi xử lý ảnh user ID {user.id}: {str(e)}")
                continue

        if not match_found:
            elapsed = time.time() - start_time
            logger.warning(f"[{request_id}] ❌ Không tìm thấy người dùng phù hợp.")
            logger.info(f"[{request_id}] ❌ END execute_check_face_logon (Not Found) ⏱️ {elapsed:.2f}s")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "Không tìm thấy người dùng phù hợp."}
            )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{request_id}] ❗ Internal Server Error: {str(e)}")
        logger.info(f"[{request_id}] ❌ END execute_check_face_logon (Exception) ⏱️ {elapsed:.2f}s")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Lỗi nội bộ trong hệ thống."}
        )