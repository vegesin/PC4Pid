# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-20
# @FilePath: /SNN/src/datasets/ipix_dataset.py
# @Description: 不同模型训练需要的不同dataset实现
# 当前的dataset类self.__getitem__(self.idx)需要返回signal, label, sample_str
# signal : 输入模型的样本，当模型需要多个样本时，拼接为一个数据矩阵，或者打包为字典，模型内部逻辑进行处理
# label : 标签
# sample_str : 样本唯一标识符，用于后续数据分析定位异常样本
# -------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from ..main.utils.base import *


class ipix_dataset(Dataset):
    """
    ### @ /src/scripts/gen_ipixmat2npy.py 脚本导出的npy文件数据集
    signal : 1024脉冲的原始回波切片，维度2*1024，数值经过归一化[0,1]
    label : 0杂波单元 1目标单元 （移除掉了ipix的保护单元）

    Args:
        - data_path :  ipix一个采集场景对应的一个npy文件 该数据集对应原始回波
    """

    def __init__(self, data_path, **kwargs):

        # 验证路径有效性
        if not data_path:
            raise ValueError("数据路径不能为空")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        self.data_path = data_path
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        signal = self.data[idx]['echo']
        #  label 是一个数 (0 或 1)
        label_raw = self.data[idx]['label']

        sample_str = f'{self.data_path[:-3]}_{idx}'

        signal = torch.from_numpy(signal).float()

        label = torch.as_tensor(label_raw)

        return signal, label, sample_str


class ipix_tfg_dataset(Dataset):
    """
    ### @ 经过/SNN/src/scripts/TFG_dataset_gen.py 转换后的npz数据集
    
    Args:
        - data_path : npz 文件路径
    """

    def __init__(self, data_path, **kwargs):

        # 验证路径有效性
        if not data_path:
            raise ValueError("数据路径不能为空")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 加载 npz 数据
        # 使用 with 语句确保读取后正确释放文件句柄
        with np.load(data_path) as data:
            # 直接读取数组到内存中
            self.signals = data['signals']
            self.labels = data['labels']

        self.data_path = data_path

        print(f"成功加载数据集: {data_path}, 样本数: {len(self.labels)}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 获取 numpy 格式的信号
        signal = self.signals[idx]

        # 转换为 PyTorch Tensor 并指定为 float32
        signal = torch.from_numpy(signal).float()

        # ======== 信号处理 ========
        # 判断当前信号的维度
        if signal.dim() == 2:
            # 单通道情况: [H, W] -> [1, H, W]
            # 适配普通 2D CNN 需要的通道维度
            signal = signal.unsqueeze(0)
        elif signal.dim() == 3:
            # 双通道/多通道情况: 已经是 [C, H, W]，例如 [2, H, W]
            # 无需增加维度，直接保留即可
            pass
        else:
            raise ValueError(f"意外的信号维度，期望 2D 或 3D，但得到了形状: {signal.shape}")

        # ======== 标签处理 ========
        # 如果使用 BCEWithLogitsLoss (二分类)，float32
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)

        sample_str = f'{self.data_path[:-3]}_{idx}'

        return signal, label, sample_str


class ipix_tfg_all_dataset(Dataset):
    """
    ### @ 选取一个ipix 极化模式下的所有npz数据集（给入文件list，控制选取的数据集，使用已经划分好的train，val，test）
    
    用于加载 IPIX 离线时频图 (TFG) 数据的 Dataset 类。
    支持从多个 .npz 文件合并数据，并提供目标与杂波比例的随机下采样控制。
    参考 IPIX 雷达参数，PRF 为 1kHz，观测时间 1.024s 对应 1024 个脉冲 。

    Attributes:
        signals (np.ndarray): 内存中合并后的时频图信号。
        labels (np.ndarray): 样本标签 (1: 目标, 0: 杂波)。
        sample_str (list): 样本追踪信息。
    """

    def __init__(self, file_paths: list | str, **kwargs):
        """初始化数据集。

        Args:
            file_paths (list): .npz 文件路径列表。
            target_clutter_ratio (float, optional): 期望的目标/杂波比例。
                如 1.0 代表 1:1 [cite: 922]，0.33 代表约 1:3 [cite: 181]。
                原始数据中杂波目标分布不均衡，通过这个参数开控制杂波目标比例，方便训练，为None是保持原始比例

            seed (int): 随机种子，确保实验可重复性。
            
            兼容框架：
            arget_clutter_ratio=None, seed=42, 从kargs中传递
            
        """
        # 从 kwargs 获取参数，设置默认值
        target_clutter_ratio = kwargs.get('target_clutter_ratio', None)
        seed = kwargs.get('seed', 42)

        only_positive = kwargs.get('only_positive', False)

        # 统一处理为列表
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not file_paths:
            raise ValueError("文件列表为空，请检查路径。")

        all_signals = []
        all_labels = []
        all_info = []

        # 加载并合并数据
        for path in file_paths:
            base_name = os.path.basename(path).replace(".npz", "")
            with np.load(path) as data:
                log.info(f"成功载入文件{path}")

                signals = data['signals']
                labels = data['labels']
                all_signals.append(signals)
                all_labels.append(labels)
                for i in range(len(labels)):
                    all_info.append(f"{base_name}_{i}")

        # 数据合并
        # 使用 concatenate axis=0 兼容 (N, H, W) 和 (N, C, H, W)
        self.signals = np.concatenate(all_signals, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.sample_info = all_info

        log.info(f"数据合并完成，总样本数: {len(self.labels)}, 数据形状: {self.signals.shape}")

        # 只保留正类
        if only_positive:
            self._filter_by_label(label=1)
            log.info("已启用 only_positive 模式，只保留正类样本(label==1)")

        # 比例控制逻辑：当原始比例 (通常 1:10) 与设定不符时进行下采样
        if target_clutter_ratio is not None:
            self._resample(target_clutter_ratio, seed)
            log.info(f"数据下采样比例{target_clutter_ratio}")

    def _resample(self, ratio, seed):
        """执行随机下采样以匹配目标比例。"""
        np.random.seed(seed)
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        log.info(f"当前载入的数据中目标总数{n_pos}，杂波总数{n_neg}")

        # 根据 ratio 决定保留多少杂波或目标样本
        if (n_pos / n_neg) < ratio:
            # 杂波过多，下采样杂波 (Clutter)
            new_n_neg = int(n_pos / ratio)
            keep_neg = np.random.choice(neg_indices, min(new_n_neg, n_neg), replace=False)
            keep_pos = pos_indices
        else:
            # 目标过多，下采样目标 (Target)
            new_n_pos = int(n_neg * ratio)
            keep_pos = np.random.choice(pos_indices, min(new_n_pos, n_pos), replace=False)
            keep_neg = neg_indices

        final_idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        log.info(f"降采样之后，数据中目标总数{len(keep_pos)}，杂波总数{len(keep_neg)}")
        self.signals = self.signals[final_idx]
        self.labels = self.labels[final_idx]
        self.sample_info = [self.sample_info[i] for i in final_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取信号
        signal_np = self.signals[idx]
        signal = torch.from_numpy(signal_np).float()

        # ======== 信号处理 ========

        # fix [信号幅度处理临时补丁] :  之前导出的时候归一化到[-1,1] 导致神经元无法激活，现在将幅度归一化到[0,1]
        signal = (signal + 1.0) # [0,2]

        # 判断当前信号的维度
        if signal.dim() == 2:
            # 单通道情况: [H, W] -> [1, H, W]
            # 适配普通 2D CNN 需要的通道维度
            signal = signal.unsqueeze(0)
        elif signal.dim() == 3:
            # 双通道/多通道情况: 已经是 [C, H, W]，例如 [2, H, W]
            # 无需增加维度，直接保留即可
            pass
        else:
            raise ValueError(f"意外的信号维度，期望 2D 或 3D，但得到了形状: {signal.shape}")

        # ======== 标签处理 ========
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        sample_str = self.sample_info[idx]

        return signal, label, sample_str

    def _filter_by_label(self, label=1):
        """只保留指定标签的样本。"""
        keep_idx = np.where(self.labels == label)[0]

        if len(keep_idx) == 0:
            raise ValueError(f"过滤后没有样本，当前要求保留 label == {label}")

        self.signals = self.signals[keep_idx]
        self.labels = self.labels[keep_idx]
        self.sample_info = [self.sample_info[i] for i in keep_idx]

        log.info(f"标签过滤完成，保留 label == {label} 的样本数: {len(self.labels)}")


class ipix_mdccnn_dataset(Dataset):
    """
    ### @ 用于加载 IPIX 原始npy回波数据 (MDCCNN 专用) 的 Dataset 类。
    
    # 输入 ： 和ipix_dataset一样的原始回波npy文件[参考ipix_tfg_all_dataset实现传入文件list载入多个文件，以及目标杂波上下采样功能].
    # tesnsor上进行，是在dataset里面进行STFT RGB 运算还是在模型forward里面进行，希望能利用cuda的并行计算
    
    支持从多个 .npy 文件合并数据，并提供目标与杂波比例的随机下采样控制。
    输出的 signal 维度为 [2, 1024] (实部和虚部通道)。

    Attributes:
        signals (np.ndarray): 内存中合并后的原始信号 [N, 2, 1024]。
        labels (np.ndarray): 样本标签 (1: 目标, 0: 杂波)。
        sample_info (list): 样本追踪信息。
    """

    def __init__(self, file_paths: list | str, **kwargs):
        """初始化 MDCCNN 数据集。

        Args:
            file_paths (list | str): .npy 文件路径列表或单个路径字符串。
            **kwargs: 可选参数。
                target_clutter_ratio (float, optional): 期望的目标/杂波比例。
                    例如 1.0 代表 1:1，0.33 代表 1:3。为 None 时保持原始比例。
                seed (int, optional): 随机种子，默认为 42。
        
        Raises:
            ValueError: 如果文件列表为空或路径不存在。
        """
        target_clutter_ratio = kwargs.get('target_clutter_ratio', None)
        seed = kwargs.get('seed', 42)

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not file_paths:
            raise ValueError("文件列表为空，请检查路径。")

        all_signals = []
        all_labels = []
        all_info = []

        # 载入并合并数据
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件不存在: {path}")

            base_name = os.path.basename(path).replace(".npy", "")
            data_array = np.load(path, allow_pickle=True)
            log.info(f"成功载入文件: {path}")

            for i in range(len(data_array)):
                # 读取信号并确保形状为 [2, 1024]
                signal = data_array[i]['echo']
                label = data_array[i]['label']

                all_signals.append(np.expand_dims(signal, axis=0))
                all_labels.append(label)
                all_info.append(f"{base_name}_{i}")

        # 数据合并
        self.signals = np.concatenate(all_signals, axis=0)
        self.labels = np.array(all_labels)
        self.sample_info = all_info

        log.info(f"数据合并完成，总样本数: {len(self.labels)}, 数据形状: {self.signals.shape}")

        # 执行降采样逻辑
        if target_clutter_ratio is not None:
            self._resample(target_clutter_ratio, seed)
            log.info(f"数据已按目标杂波比例 {target_clutter_ratio} 进行下采样")

    def _resample(self, ratio: float, seed: int):
        """执行随机下采样以匹配目标杂波比例。"""
        np.random.seed(seed)
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        log.info(f"当前载入的数据中目标总数: {n_pos}，杂波总数: {n_neg}")

        if n_neg == 0 or n_pos == 0:
            log.warning("目标或杂波数量为 0，无法进行比例调整。")
            return

        if (n_pos / n_neg) < ratio:
            # 杂波过多，下采样杂波
            new_n_neg = int(n_pos / ratio)
            keep_neg = np.random.choice(neg_indices, min(new_n_neg, n_neg), replace=False)
            keep_pos = pos_indices
        else:
            # 目标过多，下采样目标
            new_n_pos = int(n_neg * ratio)
            keep_pos = np.random.choice(pos_indices, min(new_n_pos, n_pos), replace=False)
            keep_neg = neg_indices

        final_idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        log.info(f"降采样之后，数据中目标总数: {len(keep_pos)}，杂波总数: {len(keep_neg)}")

        self.signals = self.signals[final_idx]
        self.labels = self.labels[final_idx]
        self.sample_info = [self.sample_info[i] for i in final_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label_raw = self.labels[idx]

        # 转换为 PyTorch Tensor 并指定为 float32
        signal = torch.from_numpy(signal).float()
        label = torch.as_tensor(label_raw, dtype=torch.float32)
        sample_str = self.sample_info[idx]

        return signal, label, sample_str
