# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-27
# @FilePath: \SNN\src\main\train_factory.py
# @Description:
#   训练组件构建工厂。
# -------------------------------------------------------

import os

import torch

from src.main.logger import log
from src.main.loss import FocalLoss


def build_loss_func(loss_name: str):
    """根据注册的损失函数名称构建损失函数。

    Args:
        loss_name: 注册的损失函数名称。

    Returns:
        torch.nn.Module: 损失函数实例。

    Raises:
        ValueError: 如果不支持该 ``loss_name``。
    """
    match loss_name:
        case "mse":
            return torch.nn.MSELoss()
        case "bce":
            return torch.nn.BCELoss()
        case "bce_with_logits":
            return torch.nn.BCEWithLogitsLoss()
        case "cross_entropy" | "ce":
            return torch.nn.CrossEntropyLoss()
        case "focal":
            return FocalLoss()
        case _:
            raise ValueError(f"不支持的损失函数 (loss_func): {loss_name}")


def build_optimizer(model, optimizer_type: str, learning_rate: float, weight_decay: float):
    """为给定模型构建优化器。

    Args:
        model: 参数将被优化的模型。
        optimizer_type: 优化器类型名称。
        learning_rate: 优化器学习率。
        weight_decay: 权重衰减系数。

    Returns:
        torch.optim.Optimizer: 优化器实例。
        
    Raises:
        ValueError: 如果不支持该 ``optimizer_type``。
    """
    match optimizer_type:
        case "adamw":
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        case _:
            raise ValueError(f"不支持的优化器类型 (optimizer_type): {optimizer_type}")


def build_scheduler(optimizer, scheduler_type: str | None, num_epochs: int, min_lr: float):
    """构建学习率调度器。

    Args:
        optimizer: 优化器实例。
        scheduler_type: 调度器类型名称，若不使用则为 ``None``。
        num_epochs: 基于 epoch 的调度器使用的总 epoch 数。
        min_lr: 衰减调度器使用的最小学习率。

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: 调度器实例，禁用时返回 ``None``。
        
    Raises:
        ValueError: 如果不支持该 ``scheduler_type``。
    """
    if scheduler_type is None:
        return None

    match scheduler_type:
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
        case _:
            raise ValueError(f"不支持的调度器类型 (scheduler_type): {scheduler_type}")


def build_save_path(save_dir: str, allow_overwrite: bool = False):
    """创建并返回实验输出的目录树。

    Args:
        save_dir: 实验输出的根目录。

    Returns:
        str: 确保各项子目录存在后，返回相同的保存目录路径。
    """
    if os.path.isdir(save_dir) and os.listdir(save_dir) and not allow_overwrite:
        log.error("Experiment save_dir already exists and is not empty: "
                  f"{save_dir}. Set RunConfig.allow_overwrite=True or use a new experiment_name.")
        assert allow_overwrite, f"当前实验结果保存文件夹 {save_dir} 已经存在,运行会覆盖写之前的实验结果,修改结果保存目录"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "inference"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "show"), exist_ok=True)
    return save_dir
