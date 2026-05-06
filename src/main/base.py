# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-27
# @FilePath: \SNN\src\main\base.py
# @Description:
#   配置边界层。
#   这里只保留分层配置定义和最小解析逻辑。
# -------------------------------------------------------

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from collections.abc import Callable
from typing import Any

from .logger import log
from src.utils.func import extract_public_attrs, read_json_with_comments


class ExtraConfigMixin:
    """暴露标准化额外参数的 Mixin。"""

    def get_extra(self) -> dict[str, Any]:
        """返回额外参数字典的浅拷贝。

        Returns:
            dict[str, Any]: 额外参数字典。
        """
        return dict(self.extra)


@dataclass
class RunConfig(ExtraConfigMixin):
    """运行环境配置。

    Attributes:
        experiment_name (str): 用于日志记录和输出的实验名称。
        mode (str): 运行模式，例如 ``run_train`` 或 ``run_test``。
        device (str): 设备字符串，例如 ``cuda:0`` 或 ``cpu``。
        save_dir (str | None): 实验输出目录。
        load_model_path (str | None): 加载模型权重时使用的检查点路径。
        model_save_interval (int): 定期保存检查点的 epoch 间隔。
        extra (dict[str, Any]): 额外的运行参数。
    """

    experiment_name: str
    mode: str
    device: str = "cuda:0"
    save_dir: str | None = None
    load_model_path: str | None = None
    model_save_interval: int = 20
    allow_overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """在初始化后填充派生的运行默认值。"""
        if self.save_dir is None:
            self.save_dir = os.path.join("./results", self.experiment_name)
        if self.load_model_path is None:
            self.load_model_path = os.path.join(self.save_dir, "checkpoints", "best_ckpt.pth")


@dataclass
class ModelConfig(ExtraConfigMixin):
    """模型配置。

    Attributes:
        model_name (str): 已注册的模型名称。
        num_steps (int): 时序或 SNN 模型使用的时间步数。
        extra (dict[str, Any]): 额外的模型参数。
    """

    model_name: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig(ExtraConfigMixin):
    """数据与数据加载器配置。

    中文说明：
        这里只保留所有数据流程共享的公共字段，像 target/clutter ratio、
        auto split ratio、seed 这类只对部分数据流程有意义的参数统一放到
        ``extra`` 中。

    Attributes:
        dataset_name (str | type | Callable): 已注册的数据集名称或自定义数据集构建器。
        loader_mode (str): 加载器构建模式。
        train_path (str | list[str] | None): 训练数据集路径或路径列表。
        val_path (str | list[str] | None): 验证数据集路径或路径列表。
        test_path (str | list[str] | None): 测试数据集路径或路径列表。
        batch_size (int): 批大小。
        num_workers (int): 数据加载器工作线程数。
        extra (dict[str, Any]): 额外的数据参数。
    """

    dataset_name: str | type | Callable
    loader_mode: str
    train_path: str | list[str] | None = None
    val_path: str | list[str] | None = None
    test_path: str | list[str] | None = None
    batch_size: int = 32
    num_workers: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig(ExtraConfigMixin):
    """训练超参数配置。

    Attributes:
        num_epochs (int): 总 epoch 数。
        optimizer_type (str): 优化器类型名称。
        scheduler_type (str | None): 调度器类型名称。
        learning_rate (float): 初始学习率。
        min_lr (float): 最小学习率。
        weight_decay (float): 权重衰减系数。
        loss_func (str): 已注册的损失函数名称。
        extra (dict[str, Any]): 额外的训练参数。
    """

    num_epochs: int = 50
    optimizer_type: str = "adam"
    scheduler_type: str | None = None
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    loss_func: str = "bce_with_logits"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig(ExtraConfigMixin):
    """评估配置。

    Attributes:
        pfa (float): 目标虚警概率。
        logits_process (str): 分数转换模式。
        metrics_func (Any): Callable metric function. If omitted, Trainer
            falls back to the binary classification metric.
        extra (dict[str, Any]): 额外的评估参数。
    """

    pfa: float = 1e-3
    metrics_func: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """顶层实验配置。"""

    run: RunConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        """将实验配置序列化为普通字典。

        Returns:
            dict[str, Any]: 整个实验配置的字典快照。
        """
        return asdict(self)

    def show(self):
        """打印当前结构化的配置快照。"""
        log.note("Experiment config:")
        log.info(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str))


def _build_config(config_cls, raw_value: Any):
    """从字典或配置对象构建单个配置对象。

    Args:
        config_cls: 目标数据类 (dataclass) 类型。
        raw_value: 数据类实例、普通字典或配置对象。

    Returns:
        Any: 构建完成的数据类实例。
    """
    if isinstance(raw_value, config_cls):
        return raw_value

    raw_dict = extract_public_attrs(raw_value) if not isinstance(raw_value, dict) else dict(raw_value)
    known_fields = {item.name for item in fields(config_cls)}

    kwargs = {}
    extra = dict(raw_dict.get("extra", {}))
    for key, value in raw_dict.items():
        if key == "extra":
            continue
        if key in known_fields:
            kwargs[key] = value
        else:
            extra[key] = value

    kwargs["extra"] = extra
    return config_cls(**kwargs)


def _extract_settings_sections(settings_cls) -> dict[str, Any]:
    """从运行器 (runner) 设置中提取嵌套的配置部分。

    Args:
        settings_cls: 运行器端的设置类。

    Returns:
        dict[str, Any]: 嵌套的原始配置字典。

    Raises:
        ValueError: 如果缺少任何必需的顶层部分。
    """
    attrs = extract_public_attrs(settings_cls)
    required_sections = ("run", "model", "data", "train", "eval")
    missing_sections = [section for section in required_sections if section not in attrs]
    if missing_sections:
        raise ValueError(f"设置类 (Settings) 必须包含嵌套的配置层级 {required_sections}，当前缺失: {tuple(missing_sections)}")

    return {section: extract_public_attrs(attrs[section]) for section in required_sections}


def load_raw_config(settings_cls=None, config_path: str = "") -> dict[str, Any]:
    """从运行器设置或 JSON 文件加载原始分层配置。

    Args:
        settings_cls: 启动脚本的设置类。
        config_path:  启动 JSON 配置文件路径。

    Returns:
        dict[str, Any]: 原始嵌套配置字典。
    """
    if bool(settings_cls) == bool(config_path):
        raise ValueError("必须且只能提供 settings_cls 或 config_path 两者之一。")

    if settings_cls is not None:
        return _extract_settings_sections(settings_cls)

    return read_json_with_comments(config_path)


def parse_experiment_config(raw_dict: dict[str, Any]) -> ExperimentConfig:
    """将嵌套的原始配置解析为结构化的实验配置。

    Args:
        raw_dict: 嵌套的原始配置字典。

    Returns:
        ExperimentConfig: 结构化的配置对象。
    """
    return ExperimentConfig(
        run=_build_config(RunConfig, raw_dict["run"]),
        model=_build_config(ModelConfig, raw_dict["model"]),
        data=_build_config(DataConfig, raw_dict["data"]),
        train=_build_config(TrainConfig, raw_dict["train"]),
        eval=_build_config(EvalConfig, raw_dict["eval"]),
    )


def load_experiment_config(settings_cls=None, config_path: str = "") -> ExperimentConfig:
    """从支持的输入源加载结构化的实验配置。

    Args:
        settings_cls: 运行器端的设置类。
        config_path: JSON 配置文件路径。

    Returns:
        ExperimentConfig: 结构化的实验配置。
    """
    raw_dict = load_raw_config(settings_cls=settings_cls, config_path=config_path)
    return parse_experiment_config(raw_dict)
