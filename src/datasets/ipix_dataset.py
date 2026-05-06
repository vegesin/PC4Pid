# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-20
# @FilePath: \SNN\src\datasets\ipix_dataset.py
# @Description:
#   不同模型训练所需的数据集实现。
#   当前各个 Dataset 的 __getitem__ 统一返回：
#   signal, label, sample_str
# -------------------------------------------------------

import copy
import os
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..main.base import log

DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name: str):
    """注册一个内置的数据集构建器。

    Args:
        name: 数据集注册名称。

    Returns:
        Callable: 注册该数据集构建器的装饰器。
    """

    def decorator(builder):
        DATASET_REGISTRY[name] = builder
        return builder

    return decorator


def _ensure_list(file_paths):
    """将路径输入标准化为列表。

    Args:
        file_paths: 路径字符串、路径可迭代对象或 ``None``。

    Returns:
        list[str]: 标准化后的路径列表。
    """
    if isinstance(file_paths, str):
        return [file_paths]
    if file_paths is None:
        return []
    return list(file_paths)


def _prepare_signal(signal_np: np.ndarray) -> torch.Tensor:
    """将 numpy 信号数组转换为标准的模型输入张量形状。

    中文说明：
        约定 2D 信号会自动补成 `[1, H, W]`，
        3D 信号默认已经是 `[C, H, W]`，直接透传。

    Args:
        signal_np: 原始的 numpy 信号数组。

    Returns:
        torch.Tensor: 准备好输入模型的张量。
    """
    signal = torch.from_numpy(signal_np).float()
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)
    elif signal.dim() != 3:
        raise ValueError(f"不支持的信号形状: {tuple(signal.shape)}")
    return signal


def _prepare_binary_label(label) -> torch.Tensor:
    """Converts a binary label to a ``[1]`` float tensor.

    Args:
        label: Scalar binary label.

    Returns:
        torch.Tensor: Label tensor with shape ``[1]`` for BCE losses.
    """
    return torch.as_tensor([label], dtype=torch.float32)


def _concat_split_files(file_paths: list[str], suffix: str):
    """加载并拼接多个拆分文件到内存中。

    Args:
        file_paths: 数据集文件路径列表。
        suffix: 用于提取样本名称的文件后缀。

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]: 信号、标签和样本标识符。
    """
    all_signals = []
    all_labels = []
    all_info = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到数据集文件: {path}")

        base_name = os.path.basename(path).replace(suffix, "")
        with np.load(path, allow_pickle=True) as data:
            signals = data["signals"]
            labels = data["labels"]

        all_signals.append(signals)
        all_labels.append(labels)
        all_info.extend(f"{base_name}_{idx}" for idx in range(len(labels)))
        log.info(f"已加载数据集文件: {path}")

    return (
        np.concatenate(all_signals, axis=0),
        np.concatenate(all_labels, axis=0),
        all_info,
    )


@register_dataset("ipix")
class ipix_dataset(Dataset):
    """用于原始 IPIX `.npy` 格式的数据集。"""

    def __init__(self, file_paths, **kwargs):
        """初始化原始 IPIX 数据集。

        Args:
            file_paths: 单个 `.npy` 数据集文件的路径。
            **kwargs: 保留的数据集选项。
        """
        del kwargs

        file_paths = _ensure_list(file_paths)
        if len(file_paths) != 1:
            raise ValueError(f"ipix 需要恰好一个文件路径，当前收到: {file_paths}")
        data_path = file_paths[0]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到数据集文件: {data_path}")

        self.data_path = data_path
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.data[idx]["echo"]).float()
        label = _prepare_binary_label(self.data[idx]["label"])
        sample_str = f"{self.data_path[:-3]}_{idx}"
        return signal, label, sample_str


@register_dataset("ipix_tfg")
class ipix_tfg_dataset(Dataset):
    """用于单个 TFG `.npz` 文件的数据集。"""

    def __init__(self, file_paths, **kwargs):
        """初始化 TFG 数据集。

        Args:
            file_paths: 单个 `.npz` 数据集文件的路径。
            **kwargs: 保留的数据集选项。
        """
        del kwargs

        file_paths = _ensure_list(file_paths)
        if len(file_paths) != 1:
            raise ValueError(f"ipix_tfg 需要恰好一个文件路径，当前收到: {file_paths}")
        file_path = file_paths[0]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到数据集文件: {file_path}")

        with np.load(file_path) as data:
            self.signals = data["signals"]
            self.labels = data["labels"]

        self.file_paths = file_path
        log.info(f"已加载数据集文件: {file_path}, 样本数={len(self.labels)}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = _prepare_signal(self.signals[idx])
        label = _prepare_binary_label(self.labels[idx])
        sample_str = f"{self.file_paths[:-3]}_{idx}"
        return signal, label, sample_str


@register_dataset("ipix_tfg_all")
class ipix_tfg_all_dataset(Dataset):
    """支持可选重采样逻辑的合并 TFG 数据集。

    该数据集支持：
    - 多个 `.npz` 文件合并；
    - 仅保留正样本；
    - 按目标/杂波比例做随机下采样；
    - 额外保存“被下采样出去的评估子集”，供后续 CFAR 门限估计使用。
    """

    def __init__(self, file_paths: list | str, **kwargs):
        """初始化合并的 TFG 数据集。

        Args:
            file_paths: 单个路径或路径列表。
            **kwargs: 数据集选项，包括：
                tc_ratio: 可选的目标/杂波比例。
                seed: 用于重采样的随机种子。
                only_positive: 是否仅保留正样本。
                save_rest_clutter: 是否保留丢弃的评估子集。
        """
        tc_ratio = kwargs.get("tc_ratio")
        seed = kwargs.get("seed", 42)
        only_positive = kwargs.get("only_positive", False)
        self.save_rest_clutter = kwargs.get("save_rest_clutter", False)

        file_paths = _ensure_list(file_paths)
        if not file_paths:
            raise ValueError("file_paths 不能为空")

        self.signals, self.labels, self.sample_info = _concat_split_files(file_paths, ".npz")
        log.info(f"数据集合并完成: 样本数={len(self.labels)}, 形状={self.signals.shape}")

        if only_positive:
            self._filter_by_label(label=1)
            log.info("已将数据集过滤为仅包含正样本")

        if tc_ratio is not None:
            self._resample(tc_ratio, seed)
            log.info(f"已使用 tc_ratio={tc_ratio} 对数据集进行重采样")

    def _filter_by_label(self, label: int):
        """仅保留指定标签的样本。"""
        keep_idx = np.where(self.labels == label)[0]
        self.signals = self.signals[keep_idx]
        self.labels = self.labels[keep_idx]
        self.sample_info = [self.sample_info[idx] for idx in keep_idx]

    def _resample(self, ratio, seed):
        """将数据集重采样至所需的目标/杂波比例。

        Args:
            ratio: 期望的目标/杂波比例。
            seed: 随机种子。
        """
        np.random.seed(seed)
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        log.info(f"重采样前: 目标={n_pos}, 杂波={n_neg}")

        if n_pos == 0 or n_neg == 0:
            log.warning("目标或杂波数量为 0，跳过重采样")
            return

        if (n_pos / n_neg) < ratio:
            new_n_neg = int(n_pos / ratio)
            keep_neg = np.random.choice(neg_indices, min(new_n_neg, n_neg), replace=False)
            keep_pos = pos_indices

            if self.save_rest_clutter:
                discarded_neg = np.setdiff1d(neg_indices, keep_neg)
                eval_indices = np.sort(np.concatenate([pos_indices, discarded_neg]))
                self.rest_eval_signals = self.signals[eval_indices]
                self.rest_eval_labels = self.labels[eval_indices]
                self.rest_eval_info = [self.sample_info[idx] for idx in eval_indices]
                log.info(f"已保存剩余的杂波评估子集: 总数={len(eval_indices)}")
        else:
            new_n_pos = int(n_neg * ratio)
            keep_pos = np.random.choice(pos_indices, min(new_n_pos, n_pos), replace=False)
            keep_neg = neg_indices

            if self.save_rest_clutter:
                discarded_pos = np.setdiff1d(pos_indices, keep_pos)
                eval_indices = np.sort(np.concatenate([discarded_pos, neg_indices]))
                self.rest_eval_signals = self.signals[eval_indices]
                self.rest_eval_labels = self.labels[eval_indices]
                self.rest_eval_info = [self.sample_info[idx] for idx in eval_indices]
                log.info(f"已保存剩余的目标评估子集: 总数={len(eval_indices)}")

        final_idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        self.signals = self.signals[final_idx]
        self.labels = self.labels[final_idx]
        self.sample_info = [self.sample_info[idx] for idx in final_idx]
        log.info(f"重采样后: 目标={len(keep_pos)}, 杂波={len(keep_neg)}")

    def get_rest_clutter_dataset(self):
        """返回重采样后保存的评估子集。

        Returns:
            Optional[ipix_tfg_all_dataset]: 用于阈值估计的辅助评估数据集。
        """
        if not getattr(self, "save_rest_clutter", False):
            raise ValueError("调用 get_rest_clutter_dataset 之前必须启用 save_rest_clutter")

        if not hasattr(self, "rest_eval_labels") or len(self.rest_eval_labels) == 0:
            log.warning("没有可用的剩余杂波数据集")
            return None

        rest_dataset = copy.copy(self)
        rest_dataset.signals = self.rest_eval_signals
        rest_dataset.labels = self.rest_eval_labels
        rest_dataset.sample_info = self.rest_eval_info
        rest_dataset.save_rest_clutter = False
        return rest_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = _prepare_signal(self.signals[idx])
        label = _prepare_binary_label(self.labels[idx])
        sample_str = self.sample_info[idx]
        return signal, label, sample_str


@register_dataset("ipix_tfg_all_pos")
def build_ipix_tfg_all_pos_dataset(file_paths, **kwargs):
    """构建仅包含正样本的合并 TFG 数据集。"""
    kwargs = dict(kwargs)
    kwargs.setdefault("only_positive", True)
    return ipix_tfg_all_dataset(file_paths=file_paths, **kwargs)


class ipix_tfg_auto_split_dataset(ipix_tfg_all_dataset):
    """支持内存中动态划分训练/验证/测试集的 TFG 数据集。"""

    def __init__(self, file_paths: list | str = None, memory_data: dict = None, **kwargs):
        """初始化自动划分的数据集。

        Args:
            file_paths: 原始数据集文件路径。
            memory_data: ``create_splits`` 使用的内存划分数据。
            **kwargs: 继承自 ``ipix_tfg_all_dataset`` 的数据集选项。
        """
        kwargs = dict(kwargs)
        if memory_data is not None:
            self.signals = memory_data["signals"]
            self.labels = memory_data["labels"]
            self.sample_info = memory_data["sample_info"]
            self.save_rest_clutter = kwargs.get("save_rest_clutter", False)
            tc_ratio = kwargs.get("tc_ratio")
            seed = kwargs.get("seed", 42)
            if tc_ratio is not None:
                self._resample(tc_ratio, seed)
        else:
            if file_paths is None:
                raise ValueError("必须提供 file_paths 或 memory_data 之一")
            super().__init__(file_paths=file_paths, **kwargs)

    @classmethod
    def create_splits(
            cls,
            file_paths: list | str,
            *,
            split_ratios=(0.7, 0.15, 0.15),
            seed: int = 42,
            tc_ratio=None,
            get_rest_clutter_dataset: bool = False,
    ):
        """通过分层拆分创建训练/验证/测试数据集。

        中文说明：
            这里先把多个输入文件全部读入内存，再分别对目标样本和杂波样本
            做独立打乱与分层切分，最后重新组装成三个 Dataset，保证类别分布
            在各个 split 中尽量稳定。

        Args:
            file_paths: 单个路径或路径列表。
            split_ratios: 训练/验证/测试拆分比例。
            seed: 随机种子。
            tc_ratio: 可选的共享目标/杂波比例。
            get_rest_clutter_dataset: 是否在训练集中保留辅助阈值数据。

        Returns:
            tuple: ``(train_ds, val_ds, test_ds)``。
        """
        log.note(f"====== 开始动态拆分: train/val/test -> {split_ratios} ======")

        file_paths = _ensure_list(file_paths)
        if not file_paths:
            raise ValueError("file_paths 不能为空")

        signals, labels, sample_info = _concat_split_files(file_paths, ".npz")
        memory_data = {"signals": signals, "labels": labels, "sample_info": sample_info}

        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        log.info(f"完整数据集已准备就绪: 总数={len(labels)} | 目标={len(pos_indices)} | 杂波={len(neg_indices)}")

        if len(split_ratios) != 3:
            raise ValueError(f"split_ratios 必须恰好包含 3 个值，当前收到: {split_ratios}")

        ratios = np.array(split_ratios, dtype=float)
        if ratios.sum() == 0:
            raise ValueError(f"split_ratios 的总和必须大于 0，当前收到: {split_ratios}")
        ratios /= ratios.sum()

        rng = np.random.default_rng(seed)
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        def get_split_sizes(total_len, ratio_values):
            sizes = [int(total_len * ratio) for ratio in ratio_values]
            sizes[np.argmax(sizes)] += total_len - sum(sizes)
            return sizes

        def slice_indices(indices, sizes):
            return [
                indices[0:sizes[0]],
                indices[sizes[0]:sizes[0] + sizes[1]],
                indices[sizes[0] + sizes[1]:sizes[0] + sizes[1] + sizes[2]],
            ]

        pos_sizes = get_split_sizes(len(pos_indices), ratios)
        neg_sizes = get_split_sizes(len(neg_indices), ratios)
        pos_splits = slice_indices(pos_indices, pos_sizes)
        neg_splits = slice_indices(neg_indices, neg_sizes)

        dataset_names = ["Train", "Val", "Test"]
        tc_ratios = [tc_ratio, tc_ratio, tc_ratio]
        save_clutters = [get_rest_clutter_dataset, False, False]
        datasets = []

        for idx in range(3):
            if pos_sizes[idx] + neg_sizes[idx] == 0:
                log.info(f"跳过空拆分: {dataset_names[idx]}")
                datasets.append(None)
                continue

            subset_idx = np.concatenate([pos_splits[idx], neg_splits[idx]])
            rng.shuffle(subset_idx)

            subset_mem = {
                "signals": memory_data["signals"][subset_idx],
                "labels": memory_data["labels"][subset_idx],
                "sample_info": [memory_data["sample_info"][item] for item in subset_idx],
            }
            log.info(f"构建拆分 [{dataset_names[idx]}]: 目标={pos_sizes[idx]}, 杂波={neg_sizes[idx]}, tc_ratio={tc_ratios[idx]}")
            datasets.append(cls(
                memory_data=subset_mem,
                tc_ratio=tc_ratios[idx],
                seed=seed,
                save_rest_clutter=save_clutters[idx],
            ))

        log.note("====== 动态拆分完成 ======")
        return tuple(datasets)


@register_dataset("ipix_mdccnn")
class ipix_mdccnn_dataset(Dataset):
    """用于 MDCCNN 输入格式的数据集。"""

    def __init__(self, file_paths: list | str, **kwargs):
        """从一个或多个 `.npy` 文件初始化 MDCCNN 数据集。

        Args:
            file_paths: 单个路径或路径列表。
            **kwargs: 数据集选项，包括 ``tc_ratio`` 和 ``seed``。
        """
        tc_ratio = kwargs.get("tc_ratio")
        seed = kwargs.get("seed", 42)

        file_paths = _ensure_list(file_paths)
        if not file_paths:
            raise ValueError("file_paths 不能为空")

        all_signals = []
        all_labels = []
        all_info = []

        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"未找到数据集文件: {path}")

            base_name = os.path.basename(path).replace(".npy", "")
            data_array = np.load(path, allow_pickle=True)
            log.info(f"已加载数据集文件: {path}")

            for idx in range(len(data_array)):
                signal = data_array[idx]["echo"]
                label = data_array[idx]["label"]
                all_signals.append(np.expand_dims(signal, axis=0))
                all_labels.append(label)
                all_info.append(f"{base_name}_{idx}")

        self.signals = np.concatenate(all_signals, axis=0)
        self.labels = np.array(all_labels)
        self.sample_info = all_info

        log.info(f"数据集合并完成: 样本数={len(self.labels)}, 形状={self.signals.shape}")

        if tc_ratio is not None:
            self._resample(tc_ratio, seed)
            log.info(f"已使用 tc_ratio={tc_ratio} 对数据集进行重采样")

    def _resample(self, ratio: float, seed: int):
        """将 MDCCNN 数据集重采样至所需的目标/杂波比例。"""
        np.random.seed(seed)
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        log.info(f"重采样前: 目标={n_pos}, 杂波={n_neg}")

        if n_neg == 0 or n_pos == 0:
            log.warning("目标或杂波数量为 0，跳过重采样")
            return

        if (n_pos / n_neg) < ratio:
            new_n_neg = int(n_pos / ratio)
            keep_neg = np.random.choice(neg_indices, min(new_n_neg, n_neg), replace=False)
            keep_pos = pos_indices
        else:
            new_n_pos = int(n_neg * ratio)
            keep_pos = np.random.choice(pos_indices, min(new_n_pos, n_pos), replace=False)
            keep_neg = neg_indices

            final_idx = np.sort(np.concatenate([keep_pos, keep_neg]))
        self.signals = self.signals[final_idx]
        self.labels = self.labels[final_idx]
        self.sample_info = [self.sample_info[idx] for idx in final_idx]
        log.info(f"重采样后: 目标={len(keep_pos)}, 杂波={len(keep_neg)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).float()
        label = _prepare_binary_label(self.labels[idx])
        sample_str = self.sample_info[idx]
        return signal, label, sample_str
