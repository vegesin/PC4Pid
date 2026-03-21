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
    解析 TI IWR6843 雷达原始 bin 文件，返回点云坐标 (N, 3)。

    参数：
        bin_path: .bin 文件路径
        install_angle: 雷达安装俯仰角（度，与竖直方向夹角）
        radar_height: 雷达安装高度（米）

    返回：
        np.ndarray, shape (N, 3), 每个点为 [x, y, z]（米）
        坐标系：X 前（雷达正前方），Y 左，Z 上（世界坐标系）。
        解析失败返回 None。
    """
    try:
        with open(bin_path, 'rb') as f:
            frame_data = f.read()
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return None

    if len(frame_data) < 12:
        print("[ERROR] 文件过小，无有效数据")
        return None

    # 检查帧头 0x55AA
    if frame_data[0] != 0x55 or frame_data[1] != 0xAA:
        print("[ERROR] 帧头不正确，不是有效的雷达数据包")
        return None

    # 读取帧长度（4字节小端）
    frame_length = struct.unpack('<I', frame_data[2:6])[0]
    if len(frame_data) < 6 + frame_length:
        print(f"[ERROR] 数据不足，需要 {6+frame_length} 字节，实际 {len(frame_data)}")
        return None

    pos = 6
    pos += 2                      # 跳过时间戳（2字节）
    pos += 1                      # 跳过 num_tlvs
    pos += 1                      # 跳过 cloud_type

    if pos + 2 > len(frame_data):
        print("[ERROR] 无法读取目标数量")
        return None

    target_num = struct.unpack('<H', frame_data[pos:pos+2])[0]
    pos += 2

    points = []
    for _ in range(target_num):
        if pos + 9 > len(frame_data):
            break

        # 9字节数据结构：距离(2)+reserved(1)+方位角(1)+俯仰角(1)+reserved(4)
        range_byte1 = frame_data[pos]
        range_byte2 = frame_data[pos+1]
        azimuth_byte = frame_data[pos+3]
        elevation_byte = frame_data[pos+4]
        pos += 9

        # 距离（米），缩放因子 0.05004 为常见配置
        range_val = (range_byte1 + range_byte2 * 256) * 0.05004

        # 方位角（度），编码范围 -90° ~ +90°
        if azimuth_byte <= 63:
            azimuth_angle = math.asin(azimuth_byte / 64.0) * 180 / math.pi
        else:
            azimuth_angle = math.asin((azimuth_byte - 128) / 64.0) * 180 / math.pi

        # 俯仰角（度），编码范围 -90° ~ +90°
        if elevation_byte <= 63:
            elevation_angle = math.asin(elevation_byte / 64.0) * 180 / math.pi
        else:
            elevation_angle = math.asin((elevation_byte - 128) / 64.0) * 180 / math.pi

        azimuth_rad = math.radians(azimuth_angle)
        elevation_rad = math.radians(elevation_angle)

        # 球坐标转雷达自身直角坐标（X前，Y左，Z上）
        x_radar = range_val * math.sin(azimuth_rad) * math.cos(elevation_rad)
        y_radar = range_val * math.cos(elevation_rad) * math.cos(azimuth_rad)
        z_radar = range_val * math.sin(elevation_rad)

        # 根据安装角度和高度转换到世界坐标系
        # 假设雷达安装在 Z=0 平面，向下俯仰 install_angle 度（与竖直方向夹角）
        vertical_angle_rad = math.radians(-install_angle)
        z_world = z_radar * math.cos(vertical_angle_rad) + y_radar * math.sin(vertical_angle_rad)
        y_world = -z_radar * math.sin(vertical_angle_rad) + y_radar * math.cos(vertical_angle_rad)
        z_world += radar_height

        points.append([x_radar, y_world, z_world])

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
                 phase='train'):
        self.target_frame_num = target_frame_num
        self.target_point_num = target_point_num
        self.install_angle = install_angle
        self.radar_height = radar_height
        self.load_config = load_config
        self.transform = transform
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
            if 'pointcloud' in dirs:
                pc_dir = os.path.join(root, 'pointcloud')
                # 检查该目录下是否有 .bin 文件
                if any(f.endswith('.bin') for f in os.listdir(pc_dir)):
                    config_path = os.path.join(root, 'radar_config.json') if os.path.exists(os.path.join(root, 'radar_config.json')) else None
                    self.samples.append((pc_dir, None, config_path))
            # 如果当前目录已有 .bin 文件，也视为一个样本（可能是直接存储 bin 的文件夹）
            elif any(f.endswith('.bin') for f in files):
                config_path = os.path.join(root, 'radar_config.json') if os.path.exists(os.path.join(root, 'radar_config.json')) else None
                self.samples.append((root, None, config_path))

    def _auto_label(self):
        """自动生成标签：从文件夹名提取标识，例如 walk_obj1_1_... -> 'obj1'"""
        labels_set = {}
        for i, (folder, _, cfg) in enumerate(self.samples):
            base = os.path.basename(folder)
            # 假设命名规则为 walk_obj1_1_...，提取 obj1 部分
            import re
            m = re.search(r'(obj\d+)', base)
            if m:
                ident = m.group(1)
            else:
                # 取父目录名
                ident = os.path.basename(os.path.dirname(folder))
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

    def __getitem__(self, idx):
        folder, label, config_path = self.samples[idx]

        # 获取该样本的雷达安装参数
        if self.load_config and config_path and os.path.exists(config_path):
            angle, height = self._load_config(config_path)
        else:
            angle, height = self.install_angle, self.radar_height

        # 获取所有 bin 文件并排序
        bin_files = sorted([f for f in os.listdir(folder) if f.endswith('.bin')])
        if not bin_files:
            # 空样本，返回全零张量，并给出警告（可打印）
            frames = torch.zeros(self.target_frame_num, self.target_point_num, 3)
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
            if self.transform:
                pts = self.transform(pts)
            frames.append(pts)

        if not frames:
            # 所有帧解析失败，返回全零
            frames_tensor = torch.zeros(self.target_frame_num, self.target_point_num, 3)
            valid_len = 0
        else:
            # 堆叠成 (T_orig, N, 3) 的张量
            frames_tensor = torch.from_numpy(np.stack(frames, axis=0)).float()

            T = frames_tensor.shape[0]
            if T >= self.target_frame_num:
                if self.phase == 'train':
                    # 随机裁剪
                    start = np.random.randint(0, T - self.target_frame_num + 1)
                    frames_tensor = frames_tensor[start:start+self.target_frame_num]
                    valid_len = self.target_frame_num
                else:
                    # 取前 target_frame_num 帧
                    frames_tensor = frames_tensor[:self.target_frame_num]
                    valid_len = self.target_frame_num
            else:
                # 填充零帧
                pad = torch.zeros(self.target_frame_num - T, self.target_point_num, 3)
                frames_tensor = torch.cat([frames_tensor, pad], dim=0)
                valid_len = T

        # 返回样本、标签、定位字符串（文件夹路径）
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


    def process_folder(folder_path, install_angle, radar_height):
        """处理 pointcloud 文件夹，逐帧显示点云（按任意键继续）"""
        if not os.path.isdir(folder_path):
            print(f"[ERROR] 文件夹不存在: {folder_path}")
            return

        bin_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bin')])
        if not bin_files:
            print(f"[ERROR] 文件夹中无 .bin 文件: {folder_path}")
            return

        print(f"找到 {len(bin_files)} 个 .bin 文件，将逐帧显示。按任意键查看下一帧，按 Q 退出...")
        for i, fname in enumerate(bin_files):
            bin_path = os.path.join(folder_path, fname)
            points = parse_radar_bin(bin_path, install_angle, radar_height)
            if points is None:
                print(f"[WARN] 第 {i+1} 帧解析失败: {fname}")
                continue
            print(f"第 {i+1}/{len(bin_files)} 帧: {fname}, 点数={len(points)}")
            visualize_points(points, title=f"Frame {i+1}: {fname}")

            # 等待用户按键继续
            ans = input("按 Enter 继续，Q 退出：").strip().lower()
            if ans == 'q':
                break
            plt.close('all')


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