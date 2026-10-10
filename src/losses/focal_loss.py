import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        # alpha: cân bằng tỷ lệ giữa 2 lớp 
        # gamma: tốc độ giảm phạt cho các ca đã đoán đúng
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs là logits (chưa qua sigmoid)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Xác suất mô hình đoán đúng
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()