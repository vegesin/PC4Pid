# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-27
# @FilePath: \SNN\src\main\data_factory.py
# @Description:
#   数据工厂模块。
#   负责 dataset / dataloader 的创建和数据划分策略。
#   数据集内部参数由 dataset 自己解析，工厂层只做透传和模式调度。
# -------------------------------------------------------

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Generator
from torch.utils.data import DataLoader, random_split

from src.datasets.ipix_dataset import DATASET_REGISTRY, ipix_tfg_auto_split_dataset
from src.datasets.mmradar_pc import *
from src.main.base import DataConfig
from src.main.logger import log


@dataclass
class DatasetBundle:
    """按角色存储数据集对象。"""

    train: Any = None
    val: Any = None
    test: Any = None
    get_th: Any = None


@dataclass
class DataLoaderBundle:
    """按角色存储数据加载器对象。"""

    train: DataLoader | None = None
    val: DataLoader | None = None
    test: DataLoader | None = None
    get_th: DataLoader | None = None

    def to_dict(self) -> dict[str, DataLoader]:
        """将非空的数据加载器转换为供训练器 (Trainer) 使用的字典。

        Returns:
            dict[str, DataLoader]: 以角色为键 (key) 的数据加载器字典。
        """
        return {key: value for key, value in vars(self).items() if value is not None}


def build_dataset(dataset_name: str | type | Callable, file_paths=None, dataset_extra: dict[str, Any] | None = None):
    """构建或返回一个数据集对象。

    中文说明：
        - ``dataset_name`` 是已经实例化的 Dataset 时，直接返回；
        - ``dataset_name`` 是字符串时，从注册表中查找 dataset；
        - ``dataset_name`` 是类或构造函数时，传入 ``file_paths, **kwargs`` 这两个参数进行创建。

    Args:
        dataset_name: 已注册的数据集名称、数据集类、构建器或数据集实例。
        file_paths: 数据集源路径或路径列表。
        dataset_extra: 数据集特定的额外参数。

    Returns:
        Dataset: 构建完成或已提供的数据集对象。

    Raises:
        ValueError: 如果提供了未注册的字符串类型的数据集名称。
    """
    dataset_extra = dict(dataset_extra or {})

    if isinstance(dataset_name, Dataset):
        if file_paths is not None or dataset_extra:
            log.warning(f"传入了已实例化的数据集 {dataset_name.__class__.__name__},传入的 file_paths 和 dataset_extra 将被忽略！")

        log.info(f"直接使用已实例化的数据集: {dataset_name.__class__.__name__}")
        return dataset_name

    if isinstance(dataset_name, str):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"不支持的数据集名称: {dataset_name}")
        log.info(f"通过注册名称构建数据集: {dataset_name}, 额外参数: {dataset_extra}")
        return DATASET_REGISTRY[dataset_name](file_paths=file_paths, **dataset_extra)

    if isinstance(dataset_name, type) or callable(dataset_name):
        log.info(f"通过类构建数据集: {dataset_name.__name__}, 额外参数: {dataset_extra}")
        return dataset_name(file_paths=file_paths, **dataset_extra)

    return dataset_name


def _plan_explicit_datasets(
    dataset_name: str | type | Callable,
    train_path,
    val_path,
    test_path,
    data_extra: dict[str, Any],
) -> DatasetBundle:
    """根据显式的训练/验证/测试路径规划数据集。

    Args:
        dataset_name: 已注册的数据集名称、构建器或数据集实例。
        train_path: 训练数据集路径。
        val_path: 验证数据集路径。
        test_path: 测试数据集路径。
        data_extra: 数据集额外参数，此处不做修改直接透传。

    Returns:
        DatasetBundle: 数据集绑定包，缺失的路径对应的值保留为 ``None``。
    """
    dataset_extra = dict(data_extra)
    bundle = DatasetBundle()

    if train_path is not None:
        bundle.train = build_dataset(dataset_name=dataset_name, file_paths=train_path, dataset_extra=dataset_extra)
    if val_path is not None:
        bundle.val = build_dataset(dataset_name=dataset_name, file_paths=val_path, dataset_extra=dataset_extra)
    if test_path is not None:
        bundle.test = build_dataset(dataset_name=dataset_name, file_paths=test_path, dataset_extra=dataset_extra)

    return bundle


def _split_lengths(total_len: int, split_ratios) -> list[int]:
    """将训练/验证/测试比例转换为整数拆分长度。

    Args:
        total_len: 完整数据集的长度。
        split_ratios: 训练/验证/测试的拆分比例。

    Returns:
        list[int]: 三个具体的拆分长度。
    """
    if len(split_ratios) != 3:
        raise ValueError(f"train_val_test_split_ratios 必须恰好包含 3 个值，当前收到: {split_ratios}")

    ratios = [float(item) for item in split_ratios]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError(f"train_val_test_split_ratios 的和必须大于 0，当前收到: {split_ratios}")

    lengths = [int(total_len * ratio / ratio_sum) for ratio in ratios]
    lengths[0] += total_len - sum(lengths)
    return lengths


def _plan_auto_split_datasets(
    dataset_name: str | type | Callable,
    train_path,
    data_extra: dict[str, Any],
) -> DatasetBundle:
    """加载单个数据集并按比例拆分，以规划数据集角色。

    中文说明：
        这是面向后续数据集的通用自动划分逻辑，只要求 dataset 能通过
        ``file_paths, **kwargs`` 创建，并能被 PyTorch ``random_split`` 使用。

    Args:
        dataset_name: 已注册的数据集名称、构建器或数据集实例。
        train_path: 完整数据集的路径或路径列表。
        data_extra: 额外参数。在此处消费 ``train_val_test_split_ratios``；
            其余参数透传给数据集。

    Returns:
        DatasetBundle: 拆分后的训练/验证/测试数据集包。
    """
    if train_path is None:
        raise ValueError("auto_split_train_val_test 模式必须提供 train_path")

    dataset_extra = dict(data_extra)
    split_ratios = dataset_extra.get("train_val_test_split_ratios", (8, 1, 1))
    seed = dataset_extra.get("seed", 42)

    full_dataset = build_dataset(dataset_name=dataset_name, file_paths=train_path, dataset_extra=dataset_extra)
    train_set, val_set, test_set = random_split(
        full_dataset,
        lengths=_split_lengths(total_len=len(full_dataset), split_ratios=split_ratios),
        generator=Generator().manual_seed(seed),
    )
    return DatasetBundle(train=train_set, val=val_set, test=test_set)


def _plan_ipix_tfg_auto_split_datasets(train_path, data_extra: dict[str, Any]) -> DatasetBundle:
    """使用当前的 IPIX TFG 实验性拆分逻辑规划数据集。

    中文说明：
        这是当前实验需要保留的特例。它依赖
        ``ipix_tfg_auto_split_dataset.create_splits``，不作为通用自动划分路径。

    Args:
        train_path: 完整 IPIX TFG 数据集的路径或路径列表。
        data_extra: IPIX TFG 拆分相关的额外参数。

    Returns:
        DatasetBundle: 拆分后的训练/验证/测试数据集包。
    """
    if train_path is None:
        raise ValueError("ipix_tfg_auto_split_train_val_test 模式必须提供 train_path")

    train_dataset, val_dataset, test_dataset = ipix_tfg_auto_split_dataset.create_splits(
        file_paths=train_path,
        split_ratios=data_extra.get("train_val_test_split_ratios", (8, 1, 1)),
        seed=data_extra.get("seed", 42),
        tc_ratio=data_extra.get("tc_ratio"),
        get_rest_clutter_dataset=data_extra.get("get_rest_clutter_dataset", False),
    )
    return DatasetBundle(train=train_dataset, val=val_dataset, test=test_dataset)


def plan_base_datasets(
    loader_mode: str,
    dataset_name: str | type | Callable,
    train_path,
    val_path,
    test_path,
    data_extra: dict[str, Any] | None = None,
) -> DatasetBundle:
    """根据 ``loader_mode`` 规划基础的训练/验证/测试数据集。

    Args:
        loader_mode: 数据集加载模式。
        dataset_name: 已注册的数据集名称、构建器或数据集实例。
        train_path: 训练或完整数据集的路径。
        val_path: 验证数据集路径。
        test_path: 测试数据集路径。
        data_extra: 数据的额外参数。

    Returns:
        DatasetBundle: 规划好的训练/验证/测试数据集包。
    """
    data_extra = dict(data_extra or {})

    match loader_mode:
        case "train_val_test":
            return _plan_explicit_datasets(
                dataset_name=dataset_name,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                data_extra=data_extra,
            )
        case "auto_split_train_val_test":
            return _plan_auto_split_datasets(
                dataset_name=dataset_name,
                train_path=train_path,
                data_extra=data_extra,
            )
        case "ipix_tfg_auto_split_train_val_test":
            return _plan_ipix_tfg_auto_split_datasets(
                train_path=train_path,
                data_extra=data_extra,
            )
        case _:
            raise ValueError(f"不支持的数据加载模式 (loader_mode): {loader_mode}")


def attach_threshold_dataset(dataset_bundle: DatasetBundle, data_extra: dict[str, Any] | None = None) -> DatasetBundle:
    """在启用时，将用于阈值估计的数据集附加到配置中。

    Args:
        dataset_bundle: 已规划好的基础数据集包。
        data_extra: 数据的额外参数。

    Returns:
        DatasetBundle: 附加了可选的 ``get_th`` 的数据集包。
    """
    data_extra = dict(data_extra or {})
    if not data_extra.get("get_rest_clutter_dataset", False) or dataset_bundle.train is None:
        return dataset_bundle

    train_dataset = dataset_bundle.train
    if hasattr(train_dataset, "get_rest_clutter_dataset"):
        dataset_bundle.get_th = train_dataset.get_rest_clutter_dataset()

    return dataset_bundle


def _build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    """将数据集包装到 PyTorch 数据加载器 (DataLoader) 中。
    
    Args:
        dataset: 数据集实例。
        batch_size: 批处理大小。
        num_workers: 数据加载的工作线程数。
        shuffle: 是否打乱数据。
        
    Returns:
        DataLoader: 构建完成的数据加载器。
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def build_dataloaders(dataset_bundle: DatasetBundle, batch_size: int, num_workers: int) -> DataLoaderBundle:
    """从已规划的数据集包中构建数据加载器。
    
    Args:
        dataset_bundle: 数据集绑定包。
        batch_size: 批处理大小。
        num_workers: 工作线程数。
        
    Returns:
        DataLoaderBundle: 构建完成的数据加载器包。
    """
    bundle = DataLoaderBundle()

    if dataset_bundle.train is not None:
        bundle.train = _build_loader(dataset=dataset_bundle.train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    if dataset_bundle.val is not None:
        bundle.val = _build_loader(dataset=dataset_bundle.val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    if dataset_bundle.test is not None:
        bundle.test = _build_loader(dataset=dataset_bundle.test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    if dataset_bundle.get_th is not None:
        bundle.get_th = _build_loader(dataset=dataset_bundle.get_th, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return bundle


def build_data_components(data_cfg: DataConfig) -> dict[str, DataLoader]:
    """为训练器 (Trainer类) 初始化构建数据组件。

    Args:
        data_cfg: 数据配置对象。

    Returns:
        dict[str, DataLoader]: 以角色为键 (key) 的数据加载器字典。
    """
    log.note(f"正在以 '{data_cfg.loader_mode}' 模式构建数据加载器")

    data_extra = data_cfg.get_extra()
    dataset_bundle = plan_base_datasets(
        loader_mode=data_cfg.loader_mode,
        dataset_name=data_cfg.dataset_name,
        train_path=data_cfg.train_path,
        val_path=data_cfg.val_path,
        test_path=data_cfg.test_path,
        data_extra=data_extra,
    )
    dataset_bundle = attach_threshold_dataset(dataset_bundle=dataset_bundle, data_extra=data_extra)
    loader_bundle = build_dataloaders(
        dataset_bundle=dataset_bundle,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
    )
    return loader_bundle.to_dict()
