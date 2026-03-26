import torch
import torch.nn as nn
from .PointNet import PointNetfeat   # 假设 PointNet.py 中已定义 PointNetfeat

class PointNetSequenceClassifier(nn.Module):
    def __init__(self, num_classes, point_dim=3, feat_dim=1024, hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.pointnet = PointNetfeat(global_feat=True)  # 输出 (B, feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, N, 3)
        B, T, N, C = x.shape
        # 合并 B 和 T，转为 PointNet 需要的 (B*T, 3, N)
        x = x.view(B * T, N, C).transpose(1, 2)  # (B*T, 3, N)
        # 调用 PointNetfeat，取第一个返回值（全局特征）
        feat, _, _ = self.pointnet(x)            # (B*T, 1024)
        feat = feat.view(B, T, -1)               # (B, T, 1024)
        lstm_out, _ = self.lstm(feat)            # (B, T, hidden*2)
        out = lstm_out[:, -1, :]                 # 取最后一个时间步
        out = self.dropout(out)
        logits = self.fc(out)                    # (B, num_classes)
        return logits