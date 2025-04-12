import base64
import io
import os
import time
from datetime import datetime

import torch
from PIL import Image
from fastapi import APIRouter, status, Depends
from requests import Session
from starlette.responses import JSONResponse

from kyc.manager.face_network import FaceRecognitionNet  # Sửa import model
from utils.logger import get_logger
from . import schema
from . import services
from .. import db
from ..user import validator
from ..user.validator import get_all_users
from ...common.test_model import extract_embedding, print_embedding_stats, compare_faces
from ...manager.face_recognition import load_and_preprocess_image

model_face_check_path = os.path.abspath("../kyc_be/kyc/models/best_model.pt")

router = APIRouter(tags=['Users'], prefix='/user')


@router.post('/confirm_enrollment', status_code=status.HTTP_201_CREATED)
async def create_user_registration(request: schema.User, database: Session = Depends(db.get_db)):
    function_name = "confirm_enrollment"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/{timestamp}-{function_name}"
    os.makedirs(log_dir, exist_ok=True)

    request_id = datetime.utcnow().isoformat()
    start_time = time.time()
    log_file_path = os.path.join(log_dir, "log.txt")
    logger = get_logger("confirm_enrollment", file_path=log_file_path)

    logger.info(f"[{request_id}] ⚡ START create_user_registration")

    try:
        if not request.id_card:
            logger.warning(f"[{request_id}] ⚠️ Thiếu ID Card")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "id_card is empty"}
            )

        image_req = request.front_face_base64
        if not image_req:
            logger.warning(f"[{request_id}] ⚠️ Thiếu khuôn mặt trước")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "front_face_base64 is empty"}
            )

        logger.info(f"[{request_id}] 🔍 Verifying user by ID")
        user = await validator.verify_id_card_exist(request.id_card, database)

        if user:
            logger.warning(f"[{request_id}] ⚠️ Đã tồn tại ID card này trong hệ thống")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "The user with this id card already exists in the system."}
            )

        """
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

        try:
            img_data_one = base64.b64decode(image_req)
            image_one = Image.open(io.BytesIO(img_data_one))
            image_one.save(f"{log_dir}/image_one.png")
            logger.info(f"[{request_id}] 🖼️ Ảnh được giải mã thành công.")
        except Exception as e:
            logger.warning(f"[{request_id}] ❌ Lỗi khi giải mã hoặc mở ảnh: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Ảnh trước không hợp lệ hoặc bị lỗi định dạng."}
            )

        logger.info(f"[{request_id}] 📥 Truy vấn danh sách người dùng từ database...")
        users = await get_all_users(database)
        logger.info(f"[{request_id}] 🔁 Tổng số người dùng: {len(users)}")
        
        match_found = False

        for user2 in users:
            logger.info(f"[{request_id}] 🧪 So sánh với user ID: {user.id}")
            try:
                img_data_two = base64.b64decode(user2.front_face_base64)
                image_two = Image.open(io.BytesIO(img_data_two))
                image_two.save(f"{log_dir}/image_two_{user.id_card}.png")

                image1_tensor, face1_image = load_and_preprocess_image(image_one, id="origin")
                embedding1 = extract_embedding(model, image1_tensor, device)
                print_embedding_stats(embedding1, "One")

                image2_tensor, face2_image = load_and_preprocess_image(image_two, id=user2.id)
                embedding2 = extract_embedding(model, image2_tensor, device)
                print_embedding_stats(embedding2, "Two")

                predicted_match, similarity = compare_faces(embedding1, embedding2, 0.4)

                logger.info(f"[{request_id}] 🔗 Độ tương đồng: {similarity:.4f}")
                logger.info(f"[{request_id}] 📊 Kết quả: {'✅ MATCH' if predicted_match else '❌ NO MATCH'}")

                if predicted_match:
                    match_found = True
                    break

            except Exception as e:
                continue

        if match_found:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Đã có thông tin trong hệ thống"}
            )
        """

        new_user = await services.new_user_register(request, database)

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] ✅ END create_user_registration (Success) ⏱️ {elapsed:.2f}s")
        return new_user

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{request_id}] ❗ Internal Server Error: {str(e)}")
        logger.info(f"[{request_id}] ❌ END create_user_registration (Exception) ⏱️ {elapsed:.2f}s")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Interal Server Error {str(e)}"}
        )
