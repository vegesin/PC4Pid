# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-27
# @FilePath: \SNN\src\main\model_factory.py
# @Description: 模型构建
# -------------------------------------------------------

from typing import Any

import timm
import torch

from src.main.logger import log

from src.models.MDCCNN import MDCCNN
from src.models.ResNet import ResNet18
from src.models.rsn import RSN
from src.models.spikeformer import SpikeDrivenTransformer
from src.models.my_spike_net import *


def build_model(model_name, model_extra: dict[str, Any] | None = None):
    """通过对象、类或注册的模型名称构建模型实例。

    Args:
        model_name: 模型实例、模型类或注册的模型名称。
        model_extra: 模型特定的额外参数。

    Returns:
        torch.nn.Module: 构建完毕的模型实例。

    Raises:
        ValueError: 如果提供的 ``model_name`` 不受支持。
    """
    model_extra = dict(model_extra or {})

    if isinstance(model_name, torch.nn.Module):
        log.info(f"传入创建好的模型实例: {model_name.__class__.__name__}")
        return model_name

    if isinstance(model_name, type) and issubclass(model_name, torch.nn.Module):
        log.info(f"传入创建模型类: {model_name.__name__}, 以及模型配置参数: {model_extra}, 正在创建模型实例")
        return model_name(**model_extra)

    match model_name:
        case "resnet":
            model = ResNet18(num_classes=1)
        case "mdccnn":
            model = MDCCNN()
        case "rsn":
            model = RSN(num_steps=5) # 针对RSN的对比实验，固定num_steps=5
        case "spkformer":
            model = SpikeDrivenTransformer()
        case "spikingresformer":
            model = timm.create_model("spikingresformer_ti", pretrained=False)
        case "my_spike_net":
            model = timm.create_model("my_spike_net", pretrained=False)
        case "my_spike_net_ssa1":
            model = timm.create_model("my_spike_net_ssa1", pretrained=False)
        case _:
            raise ValueError(f"Unsupported model: {model_name}")

    if model_extra:
        raise ValueError(f"Unused model_extra for {model_name}: {tuple(model_extra.keys())}")

    return model
