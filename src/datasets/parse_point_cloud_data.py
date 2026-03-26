# -*- coding: utf-8 -*-
"""
毫米波雷达点云解析与数据集加载工具
包含：
- parse_radar_bin: 解析单个 .bin 文件为点云坐标
- RadarBinSequenceDataset: PyTorch Dataset，用于加载文件夹/文件列表的点云序列
- 原有的可视化/播放功能保持不变
单帧可视化命令：python parse_point_cloud_data.py F:/DATA_20260320/walk_obj1_2_20260320_160137/pointcloud/xxx.bin --angle 25 --height 2.0
"""

import os
import sys
import math
import struct
import argparse
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# ========== 雷达参数（可被配置文件覆盖）==========
DEFAULT_ANGLE = 25.0          # 雷达俯仰角（与竖直方向夹角，度）
DEFAULT_HEIGHT = 2.0          # 雷达安装高度（米）
DEFAULT_FRAME_NUM = 200       # 统一序列帧数
DEFAULT_POINT_NUM = 128       # 统一每帧点数


# ========== 点云解析函数 ==========
def parse_radar_bin(bin_path: str, install_angle: float = DEFAULT_ANGLE, radar_height: float = DEFAULT_HEIGHT):
    """
    解析 .bin 文件，返回 (N, 5) 点云数组，包含 [x, y, z, velocity, power]
    """
    try:
        with open(bin_path, 'rb') as f:
            frame_data = f.read()
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return None

    if len(frame_data) < 12:
        return None
    if frame_data[0] != 0x55 or frame_data[1] != 0xAA:
        return None

    frame_length = struct.unpack('<I', frame_data[2:6])[0]
    if len(frame_data) < 6 + frame_length:
        return None

    pos = 6
    pos += 2                      # 跳过时间戳
    pos += 1                      # 跳过 numTLVs
    tlv_type = frame_data[pos]    # 应该是 0x01（点云）
    pos += 1
    if tlv_type != 0x01:
        print(f"[WARN] TLV type {tlv_type} not point cloud, skip")
        return None

    if pos + 2 > len(frame_data):
        return None
    target_num = struct.unpack('<H', frame_data[pos:pos+2])[0]
    pos += 2

    points = []
    for _ in range(target_num):
        if pos + 9 > len(frame_data):
            break
        idx1 = struct.unpack('<H', frame_data[pos:pos+2])[0]          # 距离索引
        idx2 = frame_data[pos+2]                                      # 速度索引
        idx3 = frame_data[pos+3]                                      # 水平角索引
        idx4 = frame_data[pos+4]                                      # 俯仰角索引
        pow_abs = struct.unpack('<I', frame_data[pos+5:pos+9])[0]    # 功率（绝对值）
        pos += 9

        # 1. 距离（米）
        range_val = idx1 * 0.05

        # 2. 速度（m/s），远离雷达为正，靠近为负
        vel_idx = idx2 - 32
        velocity = vel_idx * 0.104167   # 范围约 -3.33 ~ +3.33

        # 3. 水平角（度）
        if idx3 <= 63:
            az_deg = math.asin(idx3 / 64.0) * 180 / math.pi
        else:
            az_deg = math.asin((idx3 - 128) / 64.0) * 180 / math.pi

        # 4. 俯仰角（度）
        if idx4 <= 63:
            el_deg = math.asin(idx4 / 64.0) * 180 / math.pi
        else:
            el_deg = math.asin((idx4 - 128) / 64.0) * 180 / math.pi

        az_rad = math.radians(az_deg)
        el_rad = math.radians(el_deg)

        # 球坐标转雷达自身直角坐标（X前，Y左，Z上）
        x_radar = range_val * math.sin(az_rad) * math.cos(el_rad)
        y_radar = range_val * math.cos(el_rad) * math.cos(az_rad)
        z_radar = range_val * math.sin(el_rad)

        # 世界坐标转换（安装角度与高度）
        vertical_angle_rad = math.radians(-install_angle)
        z_world = z_radar * math.cos(vertical_angle_rad) + y_radar * math.sin(vertical_angle_rad)
        y_world = -z_radar * math.sin(vertical_angle_rad) + y_radar * math.cos(vertical_angle_rad)
        z_world += radar_height

        # 功率处理：可保留线性值，或取对数 dB（推荐）
        power_db = 10 * math.log10(pow_abs + 1e-6) if pow_abs > 0 else -100

        points.append([x_radar, y_world, z_world, velocity, power_db])

    return np.array(points, dtype=np.float32) if points else None
    
# ========== PyTorch Dataset 类 ==========
class RadarBinSequenceDataset(Dataset):
    """
    毫米波雷达点云序列数据集，用于身份识别。

    每个样本是一个文件夹，包含连续的 .bin 文件（按文件名排序），构成一个时间序列。
    数据集可从以下方式构建：
        1. 指定根目录（root_dir），自动递归查找所有包含 pointcloud 子目录的文件夹，
           每个 pointcloud 子目录视为一个样本，根据目录结构提取标签。
        2. 指定一个文件夹路径（data_folder），将其视为单个样本（标签需要另外提供）。
        3. 指定一个文件列表（file_list），每个元素为 (bin_folder_path, label) 元组。

    Args:
        data_source (str or list): 
            - 若为 str：可以是根目录（自动扫描）或单个样本文件夹路径。
            - 若为 list：每个元素为 (folder_path, label) 元组，folder_path 为 pointcloud 文件夹路径。
        label_map (dict, optional): 当 data_source 为根目录且自动扫描时，可通过此字典将目录名映射为标签。
        target_frame_num (int): 统一序列长度（帧数）。超过则随机裁剪（训练时）或取前 N 帧（测试时），不足则填充零帧。
        target_point_num (int): 统一每帧点数。超过则随机下采样，不足则重复采样补足。
        install_angle (float): 雷达安装俯仰角（度），若每个样本不同，可从配置文件读取。
        radar_height (float): 雷达安装高度（米），若每个样本不同，可从配置文件读取。
        load_config (bool): 是否从每个样本的父目录中读取 radar_config.json 覆盖 install_angle 和 radar_height。
        transform (callable, optional): 应用于每一帧点云的额外变换（如归一化）。
        phase (str): 'train', 'val', 'test' 之一，影响裁剪策略（训练时随机裁剪，否则取前 N 帧）。
    """

    def __init__(self, data_source, label_map=None,
                 target_frame_num=DEFAULT_FRAME_NUM,
                 target_point_num=DEFAULT_POINT_NUM,
                 install_angle=DEFAULT_ANGLE,
                 radar_height=DEFAULT_HEIGHT,
                 load_config=True,
                 transform=None,
                 mean=None, 
                 std=None,
                 phase='train'):
        self.target_frame_num = target_frame_num
        self.target_point_num = target_point_num
        self.install_angle = install_angle
        self.radar_height = radar_height
        self.load_config = load_config
        self.transform = transform
        self.mean = mean
        self.std = std
        self.phase = phase

        # 构建样本列表：每个元素为 (pointcloud_dir, label, config_path)
        self.samples = []

        if isinstance(data_source, str):
            # 判断是根目录还是单个文件夹
            if os.path.isdir(data_source):
                # 尝试判断是否为 pointcloud 子目录（包含 .bin 文件）
                if any(f.endswith('.bin') for f in os.listdir(data_source)):
                    # 单个样本文件夹，需要标签
                    self.samples.append((data_source, None, None))
                else:
                    # 根目录，递归查找所有 pointcloud 子目录
                    self._scan_root(data_source)
            else:
                raise ValueError(f"路径不存在: {data_source}")
        elif isinstance(data_source, list):
            # 直接提供列表，每个元素应为 (folder, label)
            for item in data_source:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError("列表元素应为 (folder_path, label)")
                folder, label = item
                if not os.path.isdir(folder):
                    raise ValueError(f"文件夹不存在: {folder}")
                self.samples.append((folder, label, None))
        else:
            raise TypeError("data_source 必须是 str 或 list")

        # 如果样本列表中的标签有 None，且未提供 label_map，则根据文件夹名自动生成
        if any(label is None for _, label, _ in self.samples):
            if label_map is None:
                # 自动生成标签：根据文件夹名的前缀或父目录名
                # 假设文件夹名格式为 walk_obj1_1_...，提取 obj1 部分，或者使用父目录名
                self._auto_label()
            else:
                # 使用提供的映射
                for i, (folder, label, cfg) in enumerate(self.samples):
                    if label is None:
                        # 从映射中查找，映射的键可以是文件夹名或父目录名
                        # 简单处理：取文件夹名作为键，若找不到则取父目录名
                        base = os.path.basename(folder)
                        if base in label_map:
                            self.samples[i] = (folder, label_map[base], cfg)
                        else:
                            parent = os.path.basename(os.path.dirname(folder))
                            if parent in label_map:
                                self.samples[i] = (folder, label_map[parent], cfg)
                            else:
                                raise KeyError(f"找不到 {base} 或 {parent} 在 label_map 中")
        # 最终确保所有样本都有标签
        for _, label, _ in self.samples:
            if label is None:
                raise ValueError("存在未指定标签的样本")

    def _scan_root(self, root_dir):
        """递归查找所有包含 .bin 文件的 pointcloud 子目录"""
        for root, dirs, files in os.walk(root_dir):
            # 仅当存在 'pointcloud' 子目录且其中有 .bin 文件时添加
            if 'pointcloud' in dirs:
                pc_dir = os.path.join(root, 'pointcloud')
                if any(f.endswith('.bin') for f in os.listdir(pc_dir)):
                    config_path = os.path.join(root, 'radar_config.json') if os.path.exists(os.path.join(root, 'radar_config.json')) else None
                    self.samples.append((pc_dir, None, config_path))

    def _auto_label(self):
        """自动生成标签：从父目录名（行走文件夹名）中提取标识，例如 walk_obj1_1_... -> 'obj1'"""
        labels_set = {}
        for i, (folder, _, cfg) in enumerate(self.samples):
            # 获取父目录名（行走文件夹名）
            parent_dir = os.path.basename(os.path.dirname(folder))
            import re
            m = re.search(r'(obj\d+)', parent_dir)
            if m:
                ident = m.group(1)          # 如 'obj1'
            else:
                ident = parent_dir           # 后备方案
            if ident not in labels_set:
                labels_set[ident] = len(labels_set)
            self.samples[i] = (folder, labels_set[ident], cfg)

    def __len__(self):
        return len(self.samples)

    def _sample_points(self, points, num_points):
        """采样/填充点云至固定点数"""
        current = points.shape[0]
        if current == num_points:
            return points
        elif current > num_points:
            idx = np.random.choice(current, num_points, replace=False)
            return points[idx]
        else:
            idx = np.random.choice(current, num_points, replace=True)
            return points[idx]

    def _get_frame_points(self, bin_path, install_angle, radar_height):
        """解析单个 .bin 文件，返回点云坐标数组 (N, 3) 或 None"""
        return parse_radar_bin(bin_path, install_angle, radar_height)

    def _load_config(self, config_path):
        """读取 radar_config.json，返回 (install_angle, radar_height)"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            angle = cfg.get('install_angle', self.install_angle)
            height = cfg.get('install_height', self.radar_height)
            return angle, height
        except Exception:
            return self.install_angle, self.radar_height
        
    def _normalize_points(self, pts):
        """标准化点云特征： (pts - mean) / (std + eps)"""
        if self.mean is None or self.std is None:
            return pts
        eps = 1e-8
        return (pts - self.mean) / (self.std + eps)

    def _augment_points(self, pts):
        """数据增强：随机旋转、添加高斯噪声、随机丢弃点"""
        # 1. 随机旋转（绕 Z 轴，仅对坐标）
        if np.random.rand() > 0.7:  # 70% 概率做旋转
            theta = np.random.uniform(-np.pi/18, np.pi/18)  # ±10度
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            pts[:, :3] = pts[:, :3] @ rot_mat.T

        # 2. 添加高斯噪声（仅对坐标和速度，功率不加噪声）
        if np.random.rand() > 0.5:  # 50% 概率加噪声
            noise = np.random.normal(0, 0.02, size=pts[:, :4].shape)
            pts[:, :4] += noise

        # 3. 随机丢弃点（模拟遮挡）
        if np.random.rand() > 0.6:  # 40% 概率丢弃
            keep_ratio = np.random.uniform(0.7, 1.0)
            num_points = pts.shape[0]
            keep_idx = np.random.choice(num_points, int(num_points * keep_ratio), replace=False)
            pts = pts[keep_idx, :]

        return pts

    def __getitem__(self, idx):
        folder, label, config_path = self.samples[idx]

        if self.load_config and config_path and os.path.exists(config_path):
            angle, height = self._load_config(config_path)
        else:
            angle, height = self.install_angle, self.radar_height

        bin_files = sorted([f for f in os.listdir(folder) if f.endswith('.bin')])
        if not bin_files:
            # 空样本，返回全零张量（维度改为5）
            frames = torch.zeros(self.target_frame_num, self.target_point_num, 5)
            valid_len = 0
            info = f"{folder} (empty)"
            return frames, label, info

        frames = []
        for bin_file in bin_files:
            bin_path = os.path.join(folder, bin_file)
            pts = self._get_frame_points(bin_path, angle, height)
            if pts is None or len(pts) == 0:
                continue
            # 采样至固定点数
            pts = self._sample_points(pts, self.target_point_num)
            # 确保形状正确
            if pts.shape[0] != self.target_point_num:
                print(f"警告：采样后点数 {pts.shape[0]} != {self.target_point_num}")
                continue
            if self.transform:
                pts = self.transform(pts)
            frames.append(pts)

        if not frames:
            # 所有帧解析失败，返回全零（维度改为5）
            frames_tensor = torch.zeros(self.target_frame_num, self.target_point_num, 5)
            valid_len = 0
        else:
            # 堆叠成 (T_orig, N, 5) 的张量
            frames_tensor = torch.from_numpy(np.stack(frames, axis=0)).float()

            T = frames_tensor.shape[0]
            if T >= self.target_frame_num:
                if self.phase == 'train':
                    start = np.random.randint(0, T - self.target_frame_num + 1)
                    frames_tensor = frames_tensor[start:start+self.target_frame_num]
                    valid_len = self.target_frame_num
                else:
                    frames_tensor = frames_tensor[:self.target_frame_num]
                    valid_len = self.target_frame_num
            else:
                # 填充零帧（维度改为5）
                pad = torch.zeros(self.target_frame_num - T, self.target_point_num, 5)
                frames_tensor = torch.cat([frames_tensor, pad], dim=0)
                valid_len = T

        info = folder
        return frames_tensor, label, info


# ========== 辅助函数：创建 DataLoader 的 collate_fn ==========
def radar_bin_collate(batch):
    """
    自定义 collate 函数，用于 DataLoader。
    输入 batch 是 list of (frames, label, info)
    返回 (batch_frames, batch_labels, batch_infos)
    """
    frames = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    infos = [item[2] for item in batch]
    return frames, labels, infos

if __name__ == "__main__":
    def visualize_points(points, title="Point Cloud"):
        if points is None or len(points) == 0:
            print("无可视化的点")
            return

        # 输出统计信息
        print(f"点云统计信息 ({title}):")
        print(f"  点数: {len(points)}")
        print(f"  X: min={np.min(points[:,0]):.3f}, max={np.max(points[:,0]):.3f}, mean={np.mean(points[:,0]):.3f}")
        print(f"  Y: min={np.min(points[:,1]):.3f}, max={np.max(points[:,1]):.3f}, mean={np.mean(points[:,1]):.3f}")
        print(f"  Z: min={np.min(points[:,2]):.3f}, max={np.max(points[:,2]):.3f}, mean={np.mean(points[:,2]):.3f}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴颜色（与Qt界面一致）
        ax.xaxis.label.set_color('red')
        ax.yaxis.label.set_color('green')
        ax.zaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='red')
        ax.tick_params(axis='y', colors='green')
        ax.tick_params(axis='z', colors='blue')
        
        # 绘制点云，按Z高度着色
        sc = ax.scatter(points[:,0], points[:,1], points[:,2],
                        c=points[:,2], cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(sc, label='Z (m)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 设置合理的坐标范围（根据点云实际范围动态调整，保留一些余量）
        x_range = np.max(points[:,0]) - np.min(points[:,0])
        y_range = np.max(points[:,1]) - np.min(points[:,1])
        z_range = np.max(points[:,2]) - np.min(points[:,2])
        x_mid = (np.max(points[:,0]) + np.min(points[:,0])) / 2
        y_mid = (np.max(points[:,1]) + np.min(points[:,1])) / 2
        z_mid = (np.max(points[:,2]) + np.min(points[:,2])) / 2
        max_range = max(x_range, y_range, z_range) * 1.2
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
        
        plt.tight_layout()
        plt.show()


    def process_folder(folder_path, install_angle, radar_height, interval=0.1):
        """
        连续播放 pointcloud 文件夹中的所有帧，使用固定坐标轴、彩色轴线/网格，
        点云按 Z 高度着色，支持按 Q 键退出。
        
        Args:
            folder_path (str): 包含 .bin 文件的文件夹路径
            install_angle (float): 雷达安装俯仰角（度）
            radar_height (float): 雷达安装高度（米）
            interval (float): 帧间延迟（秒），默认 0.1
        """
        if not os.path.isdir(folder_path):
            print(f"[ERROR] 文件夹不存在: {folder_path}")
            return

        bin_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])
        if not bin_files:
            print(f"[ERROR] 文件夹中无 .bin 文件: {folder_path}")
            return

        print(f"找到 {len(bin_files)} 个 .bin 文件，将连续播放（间隔 {interval} 秒）。按 Q 键退出...")

        # ---------- 固定坐标轴范围（雷达为原点） ----------
        # 根据实际场景设定：X 前（0~5），Y 左右（-1~1），Z 高度（1~3，雷达高2米）
        x_lim = (-1.0, 5.0)
        y_lim = (-1.0, 1.0)
        z_lim = (1.0, 3.0)

        # ---------- 交互模式开启 ----------
        plt.ion()
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # ---------- 设置坐标轴外观 ----------
        # 轴标签颜色
        ax.xaxis.label.set_color('red')
        ax.yaxis.label.set_color('green')
        ax.zaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='red')
        ax.tick_params(axis='y', colors='green')
        ax.tick_params(axis='z', colors='blue')
        ax.set_xlabel('X (m) - Front')
        ax.set_ylabel('Y (m) - Left')
        ax.set_zlabel('Z (m) - Height')

        # 固定范围
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        # ---------- 绘制显式的坐标轴线条（从原点出发到范围边界） ----------
        # X 轴（红色）
        ax.plot([0, x_lim[1]], [0, 0], [0, 0], color='red', linewidth=3, alpha=0.8)
        ax.plot([0, x_lim[0]], [0, 0], [0, 0], color='red', linewidth=3, alpha=0.8)  # 负方向
        # Y 轴（绿色）
        ax.plot([0, 0], [0, y_lim[1]], [0, 0], color='green', linewidth=3, alpha=0.8)
        ax.plot([0, 0], [0, y_lim[0]], [0, 0], color='green', linewidth=3, alpha=0.8)
        # Z 轴（蓝色）
        ax.plot([0, 0], [0, 0], [0, z_lim[1]], color='blue', linewidth=3, alpha=0.8)
        ax.plot([0, 0], [0, 0], [0, z_lim[0]], color='blue', linewidth=3, alpha=0.8)

        # 在原点添加一个小球体表示雷达位置
        ax.scatter(0, 0, 0, color='black', s=50, marker='o', alpha=0.7)

        # ---------- 添加辅助网格线（增强空间感） ----------
        # 在三个主要平面上绘制半透明网格线
        # XY 平面 (z=0)
        grid_xy_color = (0.5, 0.5, 0.5, 0.3)
        for x in np.arange(x_lim[0], x_lim[1] + 0.5, 0.5):
            ax.plot([x, x], [y_lim[0], y_lim[1]], [0, 0], color=grid_xy_color, linewidth=0.8)
        for y in np.arange(y_lim[0], y_lim[1] + 0.5, 0.5):
            ax.plot([x_lim[0], x_lim[1]], [y, y], [0, 0], color=grid_xy_color, linewidth=0.8)

        # XZ 平面 (y=0)
        for x in np.arange(x_lim[0], x_lim[1] + 0.5, 0.5):
            ax.plot([x, x], [0, 0], [z_lim[0], z_lim[1]], color=grid_xy_color, linewidth=0.8)
        for z in np.arange(z_lim[0], z_lim[1] + 0.5, 0.5):
            ax.plot([x_lim[0], x_lim[1]], [0, 0], [z, z], color=grid_xy_color, linewidth=0.8)

        # YZ 平面 (x=0)
        for y in np.arange(y_lim[0], y_lim[1] + 0.5, 0.5):
            ax.plot([0, 0], [y, y], [z_lim[0], z_lim[1]], color=grid_xy_color, linewidth=0.8)
        for z in np.arange(z_lim[0], z_lim[1] + 0.5, 0.5):
            ax.plot([0, 0], [y_lim[0], y_lim[1]], [z, z], color=grid_xy_color, linewidth=0.8)

        # 添加一个简单的半透明框体（可选）
        # 绘制立方体边框（轻量级）
        edges = [
            [x_lim[0], y_lim[0], z_lim[0]], [x_lim[1], y_lim[0], z_lim[0]],
            [x_lim[1], y_lim[1], z_lim[0]], [x_lim[0], y_lim[1], z_lim[0]],
            [x_lim[0], y_lim[0], z_lim[1]], [x_lim[1], y_lim[0], z_lim[1]],
            [x_lim[1], y_lim[1], z_lim[1]], [x_lim[0], y_lim[1], z_lim[1]]
        ]
        edge_connections = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        for start, end in edge_connections:
            ax.plot([edges[start][0], edges[end][0]],
                    [edges[start][1], edges[end][1]],
                    [edges[start][2], edges[end][2]],
                    color='gray', linewidth=0.5, alpha=0.3)

        # ---------- 初始化空散点图（稍后更新） ----------
        sc = ax.scatter([], [], [], c=[], cmap='viridis', s=10, alpha=0.7, vmin=z_lim[0], vmax=z_lim[1])
        cbar = plt.colorbar(sc, ax=ax, label='Height (m)')
        title = ax.set_title('')

        # ---------- 键盘事件（按 Q 退出） ----------
        running = True
        def on_key(event):
            nonlocal running
            if event.key == 'q':
                running = False
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key)

        # ---------- 播放循环 ----------
        for i, fname in enumerate(bin_files):
            if not running:
                break

            bin_path = os.path.join(folder_path, fname)
            points = parse_radar_bin(bin_path, install_angle, radar_height)
            if points is None or len(points) == 0:
                print(f"[WARN] 第 {i+1} 帧解析失败: {fname}")
                continue

            # 更新散点图数据
            sc._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            sc.set_array(points[:, 2])          # 按 Z 高度着色
            # 更新颜色条范围（避免超出）
            sc.set_clim(vmin=points[:, 2].min(), vmax=points[:, 2].max())
            cbar.update_normal(sc)

            title.set_text(f"Frame {i+1}/{len(bin_files)}: {fname} (points: {len(points)})")
            plt.draw()
            plt.pause(interval)

        plt.ioff()
        if running:
            print("播放完成，按任意键关闭窗口...")
            input()
            plt.close(fig)
        else:
            print("用户中断")

    # def process_folder(folder_path, install_angle, radar_height):
    #     """处理 pointcloud 文件夹，逐帧显示点云（按任意键继续）"""
    #     if not os.path.isdir(folder_path):
    #         print(f"[ERROR] 文件夹不存在: {folder_path}")
    #         return

    #     bin_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])
    #     if not bin_files:
    #         print(f"[ERROR] 文件夹中无 .bin 文件: {folder_path}")
    #         return

    #     print(f"找到 {len(bin_files)} 个 .bin 文件，将逐帧显示。按任意键查看下一帧，按 Q 退出...")
    #     for i, fname in enumerate(bin_files):
    #         bin_path = os.path.join(folder_path, fname)
    #         points = parse_radar_bin(bin_path, install_angle, radar_height)
    #         if points is None:
    #             print(f"[WARN] 第 {i+1} 帧解析失败: {fname}")
    #             continue
    #         print(f"第 {i+1}/{len(bin_files)} 帧: {fname}, 点数={len(points)}")
    #         visualize_points(points, title=f"Frame {i+1}: {fname}")

    #         # 等待用户按键继续
    #         ans = input("按 Enter 继续，Q 退出：").strip().lower()
    #         if ans == 'q':
    #             break
    #         plt.close('all')


    def main():
        parser = argparse.ArgumentParser(description="解析毫米波雷达点云 bin 文件并可视化")
        parser.add_argument("input", help="单个 .bin 文件路径或包含 pointcloud 的文件夹路径")
        parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE,
                            help=f"雷达俯仰角（与竖直方向夹角），默认 {DEFAULT_ANGLE}°")
        parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT,
                            help=f"雷达安装高度（米），默认 {DEFAULT_HEIGHT}")
        parser.add_argument("--config", action="store_true",
                            help="如果输入为文件夹，尝试从 radar_config.json 读取安装参数（覆盖命令行）")
        args = parser.parse_args()

        install_angle = args.angle
        radar_height = args.height

        # 如果输入是文件夹且指定了 --config，尝试加载 radar_config.json
        if args.config and os.path.isdir(args.input):
            config_path = os.path.join(args.input, "radar_config.json")
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    if 'install_angle' in cfg:
                        install_angle = cfg['install_angle']
                        print(f"从配置文件读取安装角度: {install_angle}°")
                    if 'install_height' in cfg:
                        radar_height = cfg['install_height']
                        print(f"从配置文件读取安装高度: {radar_height}m")
                except Exception as e:
                    print(f"[WARN] 读取配置文件失败: {e}，使用命令行参数")
            else:
                print("[WARN] 未找到 radar_config.json，使用命令行参数")

        # 判断输入是文件还是文件夹
        if os.path.isfile(args.input):
            points = parse_radar_bin(args.input, install_angle, radar_height)
            if points is not None:
                print(f"解析成功，点数: {len(points)}")
                visualize_points(points, title=os.path.basename(args.input))
            else:
                print("解析失败")
        elif os.path.isdir(args.input):
            process_folder(args.input, install_angle, radar_height)
        else:
            print(f"无效的输入: {args.input}")

    print("Script started")
    main()