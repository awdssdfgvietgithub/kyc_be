import os
import uuid
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FaceNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super(FaceNet, self).__init__()
        # Sử dụng MobileNetV2 làm backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Thay đổi layer cuối để output embedding size
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_features, embedding_size)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def extract_embedding(self, x):
        """Trích xuất embedding và chuẩn hóa"""
        embedding = self.forward(x)
        return F.normalize(embedding, p=2, dim=1)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Tính triplet loss
        Args:
            anchor: embedding của ảnh anchor
            positive: embedding của ảnh positive
            negative: embedding của ảnh negative
        """
        # Tính khoảng cách giữa anchor và positive
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        
        # Tính khoảng cách giữa anchor và negative
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Tính triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return torch.mean(loss)

def save_model(model, path, format='pt'):
    """Lưu mô hình theo định dạng yêu cầu"""
    if format == 'pt':
        torch.save(model.state_dict(), path)
    elif format == 'onnx':
        # Tạo dummy input
        dummy_input = torch.randn(1, 3, 160, 160)
        # Export sang ONNX
        torch.onnx.export(model, dummy_input, path, verbose=True)
    else:
        raise ValueError(f"Format {format} không được hỗ trợ")

def load_model(path, format='pt'):
    """Load mô hình từ file"""
    model = FaceNet()
    if format == 'pt':
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError(f"Format {format} không được hỗ trợ")
    return model

def load_and_preprocess_image(image, target_size=(224, 224), id=""):
    """Load và tiền xử lý ảnh với Mediapipe"""
    # Tạo thư mục debug nếu chưa có
    debug_dir = os.path.join("debug_images")
    os.makedirs(debug_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    original_image_np = np.array(image)
    original_image_path = os.path.join(debug_dir, f"{id}_{timestamp}.jpg")
    cv2.imwrite(original_image_path,
                cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))  # Chuyển về BGR để OpenCV lưu được

    # Chuyển sang RGB vì Mediapipe yêu cầu
    # image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    # rgb_image_path = os.path.join(debug_dir, f"{type}_rgb_{random_id}.jpg")
    # cv2.imwrite(rgb_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # OpenCV yêu cầu BGR khi lưu

    # Dùng Mediapipe để phát hiện khuôn mặt
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(original_image_np)

        if not results.detections:
            raise ValueError(f"Không tìm thấy khuôn mặt trong ảnh {type}")

        # Lấy khuôn mặt đầu tiên (có thể chọn cái lớn nhất nếu nhiều)
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        image_np = np.array(image)
        h, w, _ = image_np.shape
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

        face_image = image_np[y:y + bh, x:x + bw]

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