import torch
import torch.nn as nn
from .PointNet import * # 假设 PointNet.py 中已定义 PointNetfeat


class PointNetfeat(nn.Module):
    """基于PointNet重写的的特征提取模"""

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(5, 64, 1) # 原为3
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x shape: (B, C, N)  其中 C=5（坐标3 + 速度/功率2）
        B, C, N = x.size()

        # 分离坐标（前3维）和其他特征（后2维）
        xyz = x[:, :3, :]   # (B, 3, N)
        other = x[:, 3:, :] # (B, 2, N)

        # 空间变换网络只对坐标进行变换
        trans = self.stn(xyz)       # 输出 (B, 3, 3)
        xyz = xyz.transpose(2, 1)   # (B, N, 3)
        xyz = torch.bmm(xyz, trans) # (B, N, 3)
        xyz = xyz.transpose(2, 1)   # (B, 3, N)

        # 拼接回其他特征
        x = torch.cat([xyz, other], dim=1) # (B, 5, N)

        # 继续原有流程
        x = F.relu(self.bn1(self.conv1(x))) # conv1 输入5通道，输出64
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetSequenceClassifier(nn.Module):

    def __init__(self, num_classes, point_dim=3, feat_dim=1024, hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.pointnet = PointNetfeat(global_feat=True) # 输出 (B, feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, N, 3)
        B, T, N, C = x.shape
        # 合并 B 和 T，转为 PointNet 需要的 (B*T, 3, N)
        x = x.view(B * T, N, C).transpose(1, 2) # (B*T, 3, N)
                                                # 调用 PointNetfeat，取第一个返回值（全局特征）
        feat, _, _ = self.pointnet(x)           # (B*T, 1024)
        feat = feat.view(B, T, -1)              # (B, T, 1024)
        lstm_out, _ = self.lstm(feat)           # (B, T, hidden*2)
        out = lstm_out[:, -1, :]                # 取最后一个时间步
        out = self.dropout(out)
        logits = self.fc(out)                   # (B, num_classes)
        return logits
