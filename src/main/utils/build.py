# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-12
# @FilePath: \radar_pc\src\main\utils\build.py
# @Description: 根据cfg加载不同配置
# -------------------------------------------------------

import os
import logging

import torch
from torch.utils.data import DataLoader
import datetime

import timm

from .base import *
from .loss import *



from src.datasets.ipix_dataset import *



def build_model(cfg: Config):
    """
    ### @ 根据cfg 文件中的 model_name 参数切换模型

    # TODO：
    # 1.cfg中传入一个初始化好的model对象直接返回这个对象
    # 2.模型单独py文件中使用timm修饰器注册模型，这个build切换模型
    """
    
    match cfg.model_name:

        case "model_name":
            model = None
            


        case _:
            raise ValueError(f"Unsupported model: {cfg.model_name}")


    return model


def build_loss_func(cfg: Config):
    match cfg.loss_func:
        case "mse":
            return torch.nn.MSELoss()
        case "bce":
            return torch.nn.BCELoss()
        case "bce_with_logits":
            return torch.nn.BCEWithLogitsLoss()

        # TODO：cfg中传入一个初始化好的损失函数对象直接返回这个对象
        case _:
            raise ValueError(f"Unsupported loss_func: {cfg.loss_func}")


def build_optimizer(cfg: Config, model):
    """
    构建优化器和学习率调度器
    返回: (optimizer, scheduler)
    """

    # 1. 获取必要的超参数，如果cfg中没有定义则使用推荐的默认值
    # weight_decay 是提升泛化性、防止过拟合的关键（推荐 1e-4 或 1e-3）
    weight_decay = getattr(cfg, 'weight_decay', 1e-4)

    match cfg.optimizer_type:
        case "adamw":
            # AdamW 
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)
        case "adam":
            # 传统的 Adam
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=weight_decay)
        case _:
            # 默认 fallback 到 AdamW
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)

    return optimizer


def build_scheduler(cfg: Config, optimizer):
    """
    构建学习率调度器,默认为None
    返回: scheduler
    """
    # 获取最小学习率，防止后期学习率过低导致无法更新（推荐 1e-6）
    min_lr = getattr(cfg, 'min_lr', 1e-6)
    # 获取总 Epoch 数用于计算余弦周期
    epochs = getattr(cfg, 'num_epochs')

    # 根据配置选择不同的调度器
    scheduler_type = cfg.scheduler_type

    if scheduler_type is not None:
        match scheduler_type:
            case "cosine":
                # 余弦退火，不需要调参且效果稳定
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,                                       # 调度周期对应总 Epoch 数
                    eta_min=min_lr                                      # 最终衰减到的最小学习率
                )

            case _:
                raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")
    else:
        return None

    return scheduler


# --- 路径管理 ---
def build_save_path(cfg: Config):
    """
    生成格式如：results/snn_ipix/
    """

    # 根据实验名称创建
    save_dir = cfg.save_dir

    # * 后续需要其他指标结果、推理结果保存路径从这里创建 注意覆盖写问题

    os.makedirs(save_dir, exist_ok=True)                              # 一次实验结果保存总文件夹
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True) # 检查点保存路径
    os.makedirs(os.path.join(save_dir, "inference"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "show"), exist_ok=True)

    return save_dir


# --- 数据集和数据加载器 ---
def build_dataset(cfg: Config, data_path, **kwargs):
    """
    ### @ 根据配置构建底层数据集实例。
    
    根据数据集名称选择对应的数据集类并初始化，支持不同的数据格式和处理方式。
    数据集的具体定义在 datasets包下面实现，import 导入
    
    Args:
        cfg (Config): 配置对象，包含数据集相关参数
        data_path (str): 数据文件或目录的路径
        
    Returns:
        Dataset: 初始化后的PyTorch Dataset实例
        
    Raises:
        ValueError: 当数据集名称不被支持时抛出
        
    # TODO： 兼容使用cfg类传参
        
    """
    match cfg.dataset_name:
        case "ipix":
            return ipix_dataset(data_path=data_path, **kwargs)
        case "ipix_tfg":
            return ipix_tfg_dataset(data_path=data_path, **kwargs)
        case "ipix_tfg_all":
            return ipix_tfg_all_dataset(file_paths=data_path, **kwargs)
        case "ipix_tfg_all_pos":
            return ipix_tfg_all_dataset(file_paths=data_path, only_positive=True, **kwargs)
        case "ipix_mdccnn":
            return ipix_mdccnn_dataset(file_paths=data_path, **kwargs)

        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")


def build_loaders(cfg: Config):
    """
    ### @ 根据cfg.loader模式返回不同的 Loader 字典
    
    下面的其他多种模式是为了兼容ipix一个极化方式下的多个文件编号label进行适配的。

    Args:
    cfg (Config): 配置对象，包含以下关键参数：
        - mode (str): 执行模式（如'train', 'test'等）
        - loader (str): 加载器配置模式
        - train_path (str): 训练数据路径
        - val_path (str): 验证数据路径
        - test_path (str): 测试数据路径
        - batch_size (int): 批次大小
        - num_workers (int): 数据加载工作进程数
        - pin_memory (bool): 是否使用固定内存（GPU加速）
        
    Returns:
    dict: 数据加载器字典，键为数据集标识（如'train', 'val'或文件名），
            值为对应的DataLoader实例
            
    Notes:
    - 当前实现没有对mode和loader的匹配进行校验，需要用户自行确保一致性
    """

    log.note(f"执行模式为{cfg.mode},数据加载器模式为{cfg.loader},注意检查两个是否匹配,当前没有校验检查")

    loaders = {}

    match cfg.loader:

        case "train":
            # 训练模式 (返回 单个train 无校验)
            train_ds = build_dataset(cfg, cfg.train_path)
            loaders['train'] = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

        case "train_val":
            # 从cfg中获取kwargs参数 | 这个参数是一个极化模式下面载入所有子数据需要的 all
            train_tc_ratio = getattr(cfg, 'train_tc_ratio', None)
            val_tc_ratio = getattr(cfg, 'val_tc_ratio', None)
            test_tc_ratio = getattr(cfg, 'test_tc_ratio', None)

            seed = getattr(cfg, 'seed')

            # 训练模式 (返回 单个train 和 val)
            train_ds = build_dataset(cfg, cfg.train_path, target_clutter_ratio=train_tc_ratio, seed=seed)
            val_ds = build_dataset(cfg, cfg.val_path, target_clutter_ratio=val_tc_ratio, seed=seed)

            loaders['train'] = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            loaders['val'] = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        case "train_test":
            # 测试模式 (返回 单个train 和 test,需要使用train计算虚警门限，使用test计算Pd)

            # 从cfg中获取kwargs参数 | 这个参数是一个极化模式下面载入所有子数据需要的 all
            train_tc_ratio = getattr(cfg, 'train_tc_ratio', None)
            val_tc_ratio = getattr(cfg, 'val_tc_ratio', None)
            test_tc_ratio = getattr(cfg, 'test_tc_ratio', None)
            seed = getattr(cfg, 'seed')

            train_ds = build_dataset(cfg, cfg.train_path, target_clutter_ratio=train_tc_ratio, seed=seed)
            test_ds = build_dataset(cfg, cfg.test_path, target_clutter_ratio=test_tc_ratio, seed=seed)

            # 传入list 之后 basename 报错
            # train_name = os.path.basename(cfg.train_path)
            # test_name = os.path.basename(cfg.test_path)

            loaders['train'] = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            loaders['test'] = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        case "train_val_test":
            # 训练验证测试模式 (一起执行训练、验证、测试）
            train_tc_ratio = getattr(cfg, 'train_tc_ratio', None)
            val_tc_ratio = getattr(cfg, 'val_tc_ratio', None)
            test_tc_ratio = getattr(cfg, 'test_tc_ratio', None)
            seed = getattr(cfg, 'seed')

            train_ds = build_dataset(cfg, cfg.train_path, target_clutter_ratio=train_tc_ratio, seed=seed)
            val_ds = build_dataset(cfg, cfg.val_path, target_clutter_ratio=val_tc_ratio, seed=seed)
            test_ds = build_dataset(cfg, cfg.test_path, target_clutter_ratio=test_tc_ratio, seed=seed)

            loaders['train'] = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            loaders['val'] = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            loaders['test'] = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        case "test_ipix_single_with_cfar":
            # 传入单独的train和test路径，使用train计算虚警门限，在test上面进行验证Pd

            train_ds = build_dataset(cfg, cfg.train_path)
            test_ds = build_dataset(cfg, cfg.test_path)

            train_f_name = os.path.basename(cfg.train_path)
            test_f_name = os.path.basename(cfg.test_path)

            loaders[train_f_name] = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            loaders[test_f_name] = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)



    return loaders
