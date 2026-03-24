import sys
import os
import torch
import numpy as np

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.pointnet_seq import PointNetSequenceClassifier
from src.datasets.parse_point_cloud_data import RadarBinSequenceDataset

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = PointNetSequenceClassifier(num_classes=11).to(device)
    checkpoint = torch.load("./results/radar_identification/checkpoints/best_ckpt.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 加载测试集（注意路径）
    test_dataset = RadarBinSequenceDataset(
        data_source="F:/radar_data/test",  # 你的测试集路径
        phase='test'
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels, _ in test_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    test()