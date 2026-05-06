# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-30
# @FilePath: \SNN\src\datasets\mmradar_pc.py
# @Description: 华为毫米波点云 人员ID识别相关数据集解析处理
# -------------------------------------------------------

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .ipix_dataset import register_dataset
except ImportError:
    from src.datasets.ipix_dataset import register_dataset

# ========== 雷达参数（可被配置文件覆盖）==========
DEFAULT_ANGLE = 25.0    # 雷达俯仰角（与竖直方向夹角，度）
DEFAULT_HEIGHT = 2.0    # 雷达安装高度（米）
DEFAULT_FRAME_NUM = 500 # 统一序列帧数
DEFAULT_POINT_NUM = 128 # 统一每帧点数


# -------------------------------------------------------
# 点云数据解析函数
# -------------------------------------------------------
def parse_radar_bin(bin_path: str, install_angle: float = DEFAULT_ANGLE, radar_height: float = DEFAULT_HEIGHT):
    """
    解析雷达 .bin 文件，提取点云数据并转换到世界坐标系。

    读取符合特定帧格式的二进制文件，解析点云 TLV 块，将每个点的球坐标转换为雷达自身直角坐标，
    再根据雷达安装角度和高度转换到世界坐标系。最终返回包含 x, y, z, 速度, 功率（dB）的数组。

    Args:
        bin_path (str): .bin 文件的路径。
        install_angle (float, optional): 雷达俯仰角（与竖直方向夹角），单位为度。默认使用 DEFAULT_ANGLE。
        radar_height (float, optional): 雷达安装高度，单位为米。默认使用 DEFAULT_HEIGHT。

    Returns:
        np.ndarray: 形状为 (N, 5) 的 float32 数组，每行包含 [x, y, z, velocity, power_db]。
                    若无有效点云或解析失败，返回形状为 (0, 5) 的空数组；若文件读取失败或格式严重错误，返回 None。

    注意：
        - 函数依赖 struct, math, numpy 模块。
        - 世界坐标系：X 向前（雷达原水平方向），Y 向左，Z 向上。安装角度使雷达向下倾斜时，install_angle 为正。
        - 速度正值表示远离雷达，负值表示靠近雷达。
        - 功率经过 10*log10(power_abs+1e-6) 转换为 dB，最小值为 -100 dB。
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
    pos += 2                   # 跳过时间戳
    pos += 1                   # 跳过 numTLVs
    tlv_type = frame_data[pos] # 应该是 0x01（点云）
    pos += 1
    if tlv_type != 0x01:
        print(f"[WARN] TLV type {tlv_type} not point cloud, skip")
        return None

    if pos + 2 > len(frame_data):
        return None
    target_num = struct.unpack('<H', frame_data[pos:pos + 2])[0]
    pos += 2

    points = []
    for _ in range(target_num):
        if pos + 9 > len(frame_data):
            break
        idx1 = struct.unpack('<H', frame_data[pos:pos + 2])[0]        # 距离索引
        idx2 = frame_data[pos + 2]                                    # 速度索引
        idx3 = frame_data[pos + 3]                                    # 水平角索引
        idx4 = frame_data[pos + 4]                                    # 俯仰角索引
        pow_abs = struct.unpack('<I', frame_data[pos + 5:pos + 9])[0] # 功率（绝对值）
        pos += 9

        # 1. 距离（米）
        range_val = idx1 * 0.05

        # 2. 速度（m/s），远离雷达为正，靠近为负
        vel_idx = idx2 - 32
        velocity = vel_idx * 0.104167 # 范围约 -3.33 ~ +3.33

        # 3. 水平角（度）
        if idx3 <= 63:
            az_arg = idx3 / 64.0
        else:
            az_arg = (idx3 - 128) / 64.0
        # 限制在 [-1, 1] 范围内
        az_arg = max(-1.0, min(1.0, az_arg))
        az_deg = math.asin(az_arg) * 180 / math.pi

        # 4. 俯仰角（度）
        if idx4 <= 63:
            el_arg = idx4 / 64.0
        else:
            el_arg = (idx4 - 128) / 64.0
        el_arg = max(-1.0, min(1.0, el_arg))
        el_deg = math.asin(el_arg) * 180 / math.pi

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

    points_array = np.array(points, dtype=np.float32) if points else None
    if points_array is None or len(points_array) == 0:
        # 返回空数组（0行，5列）
        return np.zeros((0, 5), dtype=np.float32)
    # 确保有5列（x,y,z,velocity,power）
    if points_array.shape[1] != 5:
        # 如果列数不对，补零
        pad = np.zeros((points_array.shape[0], 5 - points_array.shape[1]), dtype=np.float32)
        points_array = np.hstack([points_array, pad])
    return points_array


# -------------------------------------------------------
# 点云数据集
# -------------------------------------------------------
@dataclass(frozen=True)
class SequenceInfo:
    """一次毫米波雷达采集序列的元数据。

    Args:
        sequence_dir: 包含 ``pointcloud`` 文件夹的目录。
        bin_paths: 有序的雷达帧路径列表。
        person_id: 从序列名称中解析出的原始人员 ID。
        label: 从 ``person_id`` 映射得到的连续类别标签。
        sequence_name: 采集序列的文件夹名称。
    """

    sequence_dir: Path
    bin_paths: list[Path]
    person_id: int
    label: int
    sequence_name: str


@dataclass(frozen=True)
class SampleIndex:
    """一个时序点云样本的索引记录。

    Args:
        sequence_idx: 源序列在 ``self.sequences`` 中的索引。
        start_frame: 起始帧索引（包含）。
        end_frame: 结束帧索引（不包含）。
        label: 连续的身份标签。
    """

    sequence_idx: int
    start_frame: int
    end_frame: int
    label: int


def _ensure_list(file_paths):
    """将数据集路径输入转换为列表。

    Args:
        file_paths: 单个路径、路径的可迭代对象或 ``None``。

    Returns:
        类路径（path-like）值的列表。
    """
    if file_paths is None:
        return []
    if isinstance(file_paths, (str, Path)):
        return [file_paths]
    return list(file_paths)


def _parse_person_id(sequence_name: str) -> int:
    """从序列文件夹名称中解析出原始人员 ID。

    Args:
        sequence_name: 文件夹名称，例如 ``walk_obj1_1_20260320_155806``。

    Returns:
        ``obj`` 之后的原始人员 ID。

    Raises:
        ValueError: 如果无法解析出 ``obj`` ID。
    """
    match = re.search(r"obj(\d+)", sequence_name)
    if match is None:
        raise ValueError(f"Cannot parse person id from sequence name: {sequence_name}")
    return int(match.group(1))


def _is_sequence_dir(path: Path) -> bool:
    """检查路径是否为一个采集序列目录。

    Args:
        path: 候选目录。

    Returns:
        该目录是否包含 ``pointcloud/*.bin`` 文件。
    """
    pointcloud_dir = path / "pointcloud"
    return pointcloud_dir.is_dir() and any(pointcloud_dir.glob("*.bin"))


def _discover_sequence_dirs(file_paths) -> list[Path]:
    """从用户输入中发现采集序列目录。

    Args:
        file_paths: 数据集根目录、单个序列目录，或两者的列表。

    Returns:
        排序后去重的序列目录列表。

    Raises:
        FileNotFoundError: 如果提供的路径不存在。
        ValueError: 如果未找到有效的序列目录。
    """
    sequence_dirs: dict[Path, Path] = {}
    for raw_path in _ensure_list(file_paths):
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        candidates = [path] if _is_sequence_dir(path) else sorted(item for item in path.iterdir() if item.is_dir())
        for candidate in candidates:
            if _is_sequence_dir(candidate):
                resolved = candidate.resolve()
                sequence_dirs[resolved] = candidate

    if not sequence_dirs:
        raise ValueError(f"No mmWave point-cloud sequences found in: {file_paths}")
    return [sequence_dirs[key] for key in sorted(sequence_dirs)]


def _build_sequences(file_paths) -> list[SequenceInfo]:
    """构建带有连续身份标签的序列元数据。

    Args:
        file_paths: 数据集根目录、单个序列目录，或两者的列表。

    Returns:
        序列元数据的列表。
    """
    sequence_dirs = _discover_sequence_dirs(file_paths)
    raw_infos = []
    for sequence_dir in sequence_dirs:
        person_id = _parse_person_id(sequence_dir.name)
        bin_paths = sorted((sequence_dir / "pointcloud").glob("*.bin"))
        raw_infos.append((sequence_dir, bin_paths, person_id))

    label_map = {person_id: idx for idx, person_id in enumerate(sorted({item[2] for item in raw_infos}))}
    return [
        SequenceInfo(
            sequence_dir=sequence_dir,
            bin_paths=bin_paths,
            person_id=person_id,
            label=label_map[person_id],
            sequence_name=sequence_dir.name,
        ) for sequence_dir, bin_paths, person_id in raw_infos
    ]


@register_dataset("pcid_dataset")
class MMRadarPointCloudIDDataset(Dataset):
    """使用毫米波点云进行身份识别的数据集。

    每个样本都是雷达点云的一个时序窗口。返回信号的形状为 ``[C, T, N]``，其中 ``C`` 是所选的特征通道数，``T`` 是帧数，``N`` 是每帧固定的点数。
    """

    def __init__(self, file_paths, **kwargs):
        """初始化点云身份数据集。

        Args:
            file_paths: 数据集根目录、单个序列目录，或序列/根目录的列表。
            **kwargs: 数据集选项。支持的键包括 ``num_frames``, ``stride``, ``num_points``, ``features``, ``install_angle``, ``radar_height`` 和 ``drop_last``。

        Raises:
            ValueError: 如果选项无效或未找到训练样本。
        """
        # 设定 **kwargs 中的默认值
        self.num_frames = int(kwargs.get("num_frames", 16))
        self.stride = int(kwargs.get("stride", 8))
        self.num_points = int(kwargs.get("num_points", DEFAULT_POINT_NUM))
        self.features = self._prepare_features(kwargs.get("features", None))
        self.install_angle = float(kwargs.get("install_angle", DEFAULT_ANGLE))
        self.radar_height = float(kwargs.get("radar_height", DEFAULT_HEIGHT))
        self.drop_last = bool(kwargs.get("drop_last", True))

        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.num_points <= 0:
            raise ValueError(f"num_points must be positive, got {self.num_points}")

        self.sequences = _build_sequences(file_paths)
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError("No samples were built. Check num_frames, stride, drop_last, and input sequences.")

    def __len__(self):
        """返回时序窗口的数量。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """加载一个时序点云样本。

        Args:
            idx: 样本索引。

        Returns:
            包含 ``signal, label, info_str`` 的元组。``signal`` 的形状为 ``[C, T, N]``，``label`` 是一个 ``torch.long`` 类型的标量。
        """
        sample = self.samples[idx]
        sequence = self.sequences[sample.sequence_idx]
        window = []

        for frame_idx in range(sample.start_frame, sample.end_frame):
            points = parse_radar_bin(
                str(sequence.bin_paths[frame_idx]),
                install_angle=self.install_angle,
                radar_height=self.radar_height,
            )
            window.append(self._fix_points(points))

        while len(window) < self.num_frames:
            window.append(np.zeros((self.num_points, 5), dtype=np.float32))

        signal_np = np.stack(window, axis=0)
        if self.features is not None:
            signal_np = signal_np[:, :, self.features]
        signal_np = np.transpose(signal_np, (2, 0, 1)).copy()

        signal = torch.from_numpy(signal_np).float()
        label = torch.as_tensor(sample.label, dtype=torch.long)
        info_str = f"{sequence.sequence_name}:start={sample.start_frame}:end={sample.end_frame}"
        return signal, label, info_str

    @staticmethod
    def _prepare_features(features):
        """验证特征列索引。

        Args:
            features: 若为 ``None`` 则表示所有解析器列，或为一个包含整数列索引的可迭代对象。

        Returns:
            ``None`` 或一个 numpy 整数索引数组。

        Raises:
            ValueError: 如果特征索引无效。
        """
        if features is None:
            return None
        feature_indices = np.asarray(list(features), dtype=np.int64)
        if feature_indices.ndim != 1 or len(feature_indices) == 0:
            raise ValueError(f"features must be None or a non-empty 1D index list, got {features}")
        if np.any(feature_indices < 0) or np.any(feature_indices >= 5):
            raise ValueError(f"features indices must be in [0, 4], got {features}")
        return feature_indices

    def _build_samples(self) -> list[SampleIndex]:
        """构建时序滑动窗口样本索引。

        Returns:
            样本索引列表。
        """
        samples = []
        for sequence_idx, sequence in enumerate(self.sequences):
            frame_num = len(sequence.bin_paths)
            if frame_num == 0:
                continue

            if self.drop_last:
                if frame_num < self.num_frames:
                    continue
                starts = range(0, frame_num - self.num_frames + 1, self.stride)
            else:
                starts = range(0, frame_num, self.stride)

            for start_frame in starts:
                end_frame = min(start_frame + self.num_frames, frame_num)
                samples.append(SampleIndex(
                    sequence_idx=sequence_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    label=sequence.label,
                ))
        return samples

    def _fix_points(self, points: np.ndarray | None) -> np.ndarray:
        """使用 SNR TopK 和填充（padding）将单帧固定为 ``[num_points, 5]``。
        
        # TODO : 点云数据的增强策略

        Args:
            points: 原始解析器输出，形状为 ``[Ni, 5]``。

        Returns:
            固定大小的点矩阵，形状为 ``[num_points, 5]``。
        """
        fixed = np.zeros((self.num_points, 5), dtype=np.float32)
        if points is None or len(points) == 0:
            return fixed

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 5:
            raise ValueError(f"Expected points with shape [N, 5], got {points.shape}")

        keep_num = min(len(points), self.num_points)
        if len(points) > self.num_points:
            top_indices = np.argpartition(points[:, 4], -keep_num)[-keep_num:]
            top_indices = top_indices[np.argsort(points[top_indices, 4])[::-1]]
            selected = points[top_indices]
        else:
            selected = points

        fixed[:keep_num] = selected[:keep_num]
        return fixed
