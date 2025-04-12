import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import logging
import os
from datetime import datetime
import numpy as np

class FaceRecognitionNet(nn.Module):
    def __init__(self, embedding_size=256, pretrained=True):
        super(FaceRecognitionNet, self).__init__()
        
        # Sử dụng ResNet18 thay vì ResNet50 cho CPU
        self.model = models.resnet18(pretrained=pretrained)
        
        # Thay đổi kích thước đầu vào
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Thay đổi kích thước đầu ra
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.model(x)

class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=0.3, alpha=0.1):
        super(ImprovedTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        
    def forward(self, anchor, positive, negative):
        """
        Tính triplet loss với cải tiến và hard mining
        Args:
            anchor: embedding của ảnh anchor
            positive: embedding của ảnh positive
            negative: embedding của ảnh negative
        """
        # Chuẩn hóa embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Tính cosine similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        
        # Hard mining: chọn negative có similarity cao nhất (gần nhất với anchor)
        hardest_neg_sim = neg_sim
        
        # Semi-hard mining: chọn negative có similarity > pos_sim nhưng < pos_sim + margin
        semi_hard_mask = (neg_sim > pos_sim) & (neg_sim < pos_sim + self.margin)
        if torch.any(semi_hard_mask):
            semi_hard_neg_sim = neg_sim[semi_hard_mask]
            hardest_neg_sim[semi_hard_mask] = semi_hard_neg_sim
        
        # Tính triplet loss với margin động
        loss = torch.clamp(hardest_neg_sim - pos_sim + self.margin, min=0.0)
        
        # Thêm regularization term với trọng số động
        pos_reg = torch.mean(pos_sim)
        neg_reg = torch.mean(hardest_neg_sim)
        reg_term = self.alpha * (pos_reg + neg_reg)
        
        # Thêm contrastive loss term với mục tiêu rõ ràng hơn
        contrastive_term = torch.mean(torch.clamp(0.8 - pos_sim, min=0.0))  # Đẩy positive pairs gần nhau hơn
        
        # Thêm term để đẩy negative pairs xa nhau hơn
        negative_term = torch.mean(torch.clamp(hardest_neg_sim - 0.2, min=0.0))
        
        return torch.mean(loss) + reg_term + contrastive_term + negative_term

class FaceRecognitionTrainer:
    def __init__(self, model, device, learning_rate=0.0001):
        self.model = model.to(device)
        self.device = device
        
        # Thiết lập logging
        self.setup_logging()
        
        # Sử dụng AdamW optimizer với learning rate nhỏ hơn cho CPU
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,  # Sử dụng learning rate được truyền vào
            weight_decay=0.01
        )
        
        # Sử dụng ReduceLROnPlateau với patience lớn hơn cho CPU
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Sử dụng ImprovedTripletLoss với margin 0.5 và hard mining
        self.criterion = ImprovedTripletLoss(margin=0.5, alpha=0.2)
        
    def setup_logging(self):
        """Thiết lập logging cho trainer"""
        # Tạo thư mục logs nếu chưa tồn tại
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Tạo tên file log với timestamp
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Thiết lập logger
        self.logger = logging.getLogger('FaceRecognitionTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Handler cho file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler cho console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format cho log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Thêm handlers vào logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def train_step(self, anchor, positive, negative):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        anchor_embedding = self.model(anchor)
        positive_embedding = self.model(positive)
        negative_embedding = self.model(negative)
        
        # Tính loss
        loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
        
        return loss
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        embeddings_list = []
        labels_list = []
        
        self.logger.info("Bắt đầu đánh giá model...")
        self.logger.info("Đang tải dữ liệu validation...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                # Lưu embeddings và labels
                embeddings_list.append(embeddings)
                labels_list.append(labels)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Đã xử lý {batch_idx + 1} batches")
        
        self.logger.info("Đang kết hợp embeddings và labels...")
        # Kết hợp tất cả embeddings và labels
        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        self.logger.info(f"Tổng số ảnh: {len(embeddings)}")
        self.logger.info(f"Số lượng labels duy nhất: {len(torch.unique(labels))}")
        
        # Chuẩn hóa embeddings
        self.logger.info("Chuẩn hóa embeddings...")
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Tính accuracy và các metrics khác
        n = len(labels)
        unique_labels = torch.unique(labels)
        
        # Lưu trữ similarities cho positive và negative pairs
        pos_similarities = []
        neg_similarities = []
        
        self.logger.info("Bắt đầu tính metrics...")
        # Xử lý từng label một
        for idx, label in enumerate(unique_labels):
            if (idx + 1) % 10 == 0:
                print(f"Đang xử lý label {idx + 1}/{len(unique_labels)}")
            
            # Lấy indices của các ảnh có cùng label
            pos_indices = (labels == label).nonzero().squeeze()
            if len(pos_indices) > 1:
                # Tính similarity giữa các ảnh cùng label
                pos_embeddings = embeddings_norm[pos_indices]
                pos_sim = torch.mm(pos_embeddings, pos_embeddings.t())
                
                # Lưu similarities của positive pairs
                pos_similarities.extend(pos_sim.triu(diagonal=1).flatten().tolist())
        
        self.logger.info("Tính similarity với các ảnh khác label...")
        # Tính similarity với các ảnh khác label theo batch
        batch_size = 100  # Giảm batch size để tiết kiệm bộ nhớ
        for i in range(0, n, batch_size):
            if (i + 1) % 100 == 0:
                print(f"Đang xử lý ảnh {i + 1}/{n}")
            
            # Lấy batch embeddings
            batch_embeddings = embeddings_norm[i:i+batch_size]
            
            # Tính similarity với tất cả các ảnh khác
            for j in range(0, n, batch_size):
                other_embeddings = embeddings_norm[j:j+batch_size]
                
                # Tính similarity giữa hai batch
                batch_sim = torch.mm(batch_embeddings, other_embeddings.t())
                
                # Lọc các cặp negative
                batch_labels = labels[i:i+batch_size]
                other_labels = labels[j:j+batch_size]
                neg_mask = batch_labels.unsqueeze(1) != other_labels.unsqueeze(0)
                
                # Lưu similarities của negative pairs
                neg_similarities.extend(batch_sim[neg_mask].tolist())
                
                # Giải phóng bộ nhớ
                del batch_sim
                del neg_mask
                torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        
        # Tính statistics cho similarities
        pos_similarities = torch.tensor(pos_similarities)
        neg_similarities = torch.tensor(neg_similarities)
        pos_mean = pos_similarities.mean().item() if len(pos_similarities) > 0 else 0
        neg_mean = neg_similarities.mean().item() if len(neg_similarities) > 0 else 0
        pos_std = pos_similarities.std().item() if len(pos_similarities) > 0 else 0
        neg_std = neg_similarities.std().item() if len(neg_similarities) > 0 else 0
        
        # Tìm threshold tối ưu dựa trên distribution của similarities
        threshold = (pos_mean + neg_mean) / 2  # Bắt đầu với threshold ở giữa
        best_threshold = threshold
        best_f1 = 0
        
        # Thử các threshold khác nhau
        for t in np.linspace(neg_mean, pos_mean, 100):
            true_positives = (pos_similarities > t).sum().item()
            false_positives = (neg_similarities > t).sum().item()
            false_negatives = (pos_similarities <= t).sum().item()
            true_negatives = (neg_similarities <= t).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        threshold = best_threshold
        self.logger.info(f"Threshold tối ưu: {threshold:.4f}")
        
        # Tính metrics cuối cùng với threshold tối ưu
        true_positives = (pos_similarities > threshold).sum().item()
        false_positives = (neg_similarities > threshold).sum().item()
        false_negatives = (pos_similarities <= threshold).sum().item()
        true_negatives = (neg_similarities <= threshold).sum().item()
        
        total_pairs = len(pos_similarities) + len(neg_similarities)
        accuracy = (true_positives + true_negatives) / total_pairs if total_pairs > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.logger.info("\n=== Kết quả đánh giá chi tiết ===")
        self.logger.info(f"Tổng số cặp: {total_pairs}")
        self.logger.info(f"Số cặp positive: {len(pos_similarities)}")
        self.logger.info(f"Số cặp negative: {len(neg_similarities)}")
        self.logger.info(f"True Positives: {true_positives}")
        self.logger.info(f"True Negatives: {true_negatives}")
        self.logger.info(f"False Positives: {false_positives}")
        self.logger.info(f"False Negatives: {false_negatives}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1_score:.4f}")
        self.logger.info(f"Positive Similarity Mean: {pos_mean:.4f}")
        self.logger.info(f"Negative Similarity Mean: {neg_mean:.4f}")
        self.logger.info(f"Positive Similarity Std: {pos_std:.4f}")
        self.logger.info(f"Negative Similarity Std: {neg_std:.4f}")
        
        self.logger.info("\nTính validation loss...")
        # Tính validation loss với regularization
        loss = 0
        valid_labels = 0
        l2_reg = 0.0  # L2 regularization
        
        for idx, label in enumerate(unique_labels):
            if (idx + 1) % 10 == 0:
                print(f"Đang tính loss cho label {idx + 1}/{len(unique_labels)}")
            
            pos_indices = (labels == label).nonzero().squeeze()
            if len(pos_indices) > 1:
                pos_embeddings = embeddings_norm[pos_indices]
                pos_sim = torch.mm(pos_embeddings, pos_embeddings.t())
                
                # Lấy các cặp negative
                neg_indices = (labels != label).nonzero().squeeze()
                if len(neg_indices) > 0:
                    # Tính loss theo batch
                    batch_size = 100
                    batch_loss = 0
                    for j in range(0, len(neg_indices), batch_size):
                        batch_indices = neg_indices[j:j+batch_size]
                        neg_sim = torch.mm(pos_embeddings, embeddings_norm[batch_indices].t())
                        # Tính loss cho từng cặp positive-negative
                        for k in range(len(pos_embeddings)):
                            batch_loss += torch.clamp(neg_sim[k] - pos_sim[k,k] + self.criterion.margin, min=0.0).mean()
                    loss += batch_loss / (len(pos_embeddings) * (len(neg_indices) // batch_size))
                    valid_labels += 1
        
        if valid_labels > 0:
            # Thêm L2 regularization
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss = loss / valid_labels + 0.01 * l2_reg
            self.logger.info(f"Số label hợp lệ: {valid_labels}")
            self.logger.info(f"Validation Loss: {loss:.4f}")
        else:
            self.logger.warning("Không tìm thấy cặp positive/negative hợp lệ!")
            loss = 0.0
        
        self.logger.info("=====================\n")
        
        return accuracy, loss.item(), pos_mean, neg_mean, pos_std, neg_std
    
    def save_checkpoint(self, filename, epoch, accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy
        }
        torch.save(checkpoint, filename)
        self.logger.info(f"Đã lưu checkpoint tại epoch {epoch} với accuracy {accuracy:.4f}")

def save_model(model, path, format='pt'):
    """Lưu mô hình theo định dạng yêu cầu"""
    if format == 'pt':
        torch.save(model.state_dict(), path)
    elif format == 'onnx':
        # Tạo dummy input
        dummy_input = torch.randn(1, 3, 224, 224)  # EfficientNet input size
        # Export sang ONNX
        torch.onnx.export(model, dummy_input, path, verbose=True)
    else:
        raise ValueError(f"Format {format} không được hỗ trợ")

def load_model(path, format='pt'):
    """Load mô hình từ file"""
    model = FaceRecognitionNet()
    if format == 'pt':
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError(f"Format {format} không được hỗ trợ")
    return model 