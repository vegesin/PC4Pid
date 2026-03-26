#计算训练集的均值和标准差,将它们保存到文件
import sys
import os
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from datasets.parse_point_cloud_data import RadarBinSequenceDataset

def compute_stats(dataset):
    all_points = []
    print("正在收集所有点云...")
    for i in range(len(dataset)):
        frames, _, _ = dataset[i]
        # 打印形状信息
        print(f"样本 {i}: frames shape = {frames.shape}")
        if frames.shape[1] != dataset.target_point_num:
            print(f"  警告：第 {i} 个样本的帧点数 {frames.shape[1]} 与目标 {dataset.target_point_num} 不一致")
        all_points.append(frames.numpy().reshape(-1, 5))
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{len(dataset)} 个样本")
    all_points = np.concatenate(all_points, axis=0)
    mean = np.mean(all_points, axis=0)
    std = np.std(all_points, axis=0)
    return mean, std

# 创建训练集 dataset（注意：此处 phase 必须为 'train'，且不传 mean/std）
train_dataset = RadarBinSequenceDataset(
    data_source="F:/radar_data/train",
    phase='train',
    # 不传 mean/std，避免干扰统计
)

mean, std = compute_stats(train_dataset)
print(f"Mean: {mean}")
print(f"Std: {std}")

# 保存到文件，供后续训练使用
stats_path = "F:/radar_data/train_stats.pkl"
with open(stats_path, 'wb') as f:
    pickle.dump({'mean': mean, 'std': std}, f)
print(f"统计量已保存至 {stats_path}")