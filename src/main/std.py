# -------------------------------------------------------
# @ Author: vegesin
# @ Date: 2026-01-10
# @FilePath: \radar_pc\src\main\std.py
# @ Description: std启动脚本
# -------------------------------------------------------
"""
>================================================================================
>                                使用说明
>================================================================================
 1. 脚本定位:
    本脚本 (std.py) 是项目的统一启动入口，负责环境初始化、配置加载及 执行模式 调度。
    后续其他实验复制这个脚本进行。

 2. 配置方式 (优先级: 命令行传入 JSON  脚本内 Settings):

    [方式 A] 脚本内直接配置 (调试/开发推荐)
    - 方法: 直接修改脚本下方的 `class Settings` 类中的变量值。
    - 优点: 简单直观，无需创建额外文件，利用 IDE 补全快速微调参数。
    - 适用: 日常代码调试、新模型测试。

    [方式 B] 加载 JSON 配置文件 (正式实验推荐)
    - 方法: 通过命令行参数 `--config` 传入 json 文件路径。
    - 指令: python src/main/std.py --config ./configs/exp_001.json
    - 机制: 一旦传入 config 路径，脚本将忽略下方的 Settings 类，完全使用 JSON 参数。
    - 适用: 需要固化实验参数、批量运行实验、复现结果。
    
>================================================================================
>                               参数配置详解
>================================================================================
[基础运行参数]
# @ 不同的执行模式详细解释（待补充）
    experiment_name (str): 实验名称，用于生成日志和结果保存目录。
    mode (str):            运行模式。
                           - "run_train": 执行训练流程。
                           - "run_test":  执行测试流程。
    device (str):          计算设备，例如 "cuda:0" 或 "cpu"。
    save_dir (str):        [自动生成] 结果保存根路径，默认为 ./results/{experiment_name}/。
    load_mode_path (str):  预训练模型加载路径。若为 None，默认加载当前实验 best_ckpt.pth。

[模型配置]
# @ 后续兼容实现用户在启动脚本实现一个自定义模型传入，更多模型详细参考build.py以及models包下面的定义
    model_name (str):      配置载入模型。 
   

[ 数据集配置 ]
# @ 数据集更多详细参考dataset包下面的数据集定义 | 更多详细参考build.py下面不同的loaders创建模式
    dataset_name (str):    数据集
    loader (str):          数据加载器构建模式。
                           - "train":       仅返回 train loader。
                           - "train_val":   返回 train 和 val loader。
                           - "train_test":  返回 train 和 test loader。
                           - "test_ipix_single_with_cfar": 单文件测试 (Train算门限, Test测Pd)。
                           - "test_ipix_all_with_cfar":    文件夹遍历测试 (Train算门限, Test测Pd)。
    train_path (str):      训练数据路径 (文件或文件夹)。
    val_path (str):        验证数据路径。
    test_path (str):       测试数据路径。

[训练评估相关组件]
    batch_size (int):      批大小。
    num_epochs (int):      训练总轮数。
    optimizer_type (str):  优化器，可选 "adamw", "adam", "sgd"。
    scheduler_type (str):  学习率调度，可选 "cosine" 或 None。
    loss_func (str):       损失函数。
                           - "bce_with_logits", "bce", "ce", "mse"
                           - "ce_count", "ce_rate" (SNN专用)
    learning_rate (float): 初始学习率。
    min_lr (float):        最小学习率 (用于 cosine 调度)。
    pfa (float):           虚警率指标 (用于评估)。


"""

# > ------------------------  环境导入 ------------------------

import os
import sys
import argparse
from dataclasses import dataclass, asdict

# 将项目根目录加入 sys.path SNN（src上一级文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 获取当前文件所在目录的父目录 (即 src 目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.dirname(current_dir)

if project_src not in sys.path:
    sys.path.append(project_src)

from main.utils.Train import Trainer

from main.utils.base import *

# logger 配置
log: myLogger = logger_init(log_level_idx=0, save_log=False, save_dir=None)


# > ------------------------ 环境导入结束，开始用户配置 ------------------------
class Settings:
    # --- 基础配置 ---
    experiment_name = "std" # 实验名，决定结果保存路径
    mode = "run_train_test"                      # 模式: "run_train" | "run_test" | run_train_test |run_profile
    device = "cuda:3"                            # 设备: "cuda:x" | "cpu"

    # 输出结果路径配置 (None表示使用默认规则，config类中定义的方法进行默认相关依赖处理)
    save_dir = None
    load_mode_path = None # 加载权重路径，None则加载best_ckpt

    # --- 模型配置 ---

    model_name = "model_name"



    # --- 数据配置  ---
    # 正常训练推理使用数据集
    dataset_name = "ipix_tfg_all"

    # SNN 仅使用正类，统计正类的神经元发射率
    # dataset_name = "ipix_tfg_all_pos"

    loader = "train_val_test"

    # 数据路径
    # TODO: 针对雷达点云数据的加载



    # --- 训练参数 ---
    batch_size = 16
    num_epochs = 100
    num_workers = 8

    optimizer_type = "adam"
    learning_rate = 1e-3

    scheduler_type = "cosine" # 允许为 None

    min_lr = 1e-6

    loss_func = "bce_with_logits"

    pfa = 1e-3            # 虚警率
    logits_process = "z1" # 兼容参数


def main():

    # ----------------------------------------------------
    # 加载配置
    # ----------------------------------------------------
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='配置文件路径')
    parser.add_argument('--config', type=str, default='', help='Path to JSON config file (optional)')
    args = parser.parse_args()

    # 实例化一个默认的 Config 对象 (此时只包含 base.py 中的默认值)
    cfg = Config()

    # 命令行传入JSON覆盖脚本中的配置
    if args.config and os.path.exists(args.config):
        log.note(f"传入配置文件路径: {args.config}")
        cfg = Config.from_json(args.config)

    # 将 Settings 类中的变量赋值给 cfg 对象
    else:
        log.warning(f"没有传入配置文件，使用脚本 [Settings] 中的参数")

        # 遍历 Settings 中的所有变量，只要不是内置变量(__开头)，就赋值给 cfg
        for key, value in vars(Settings).items():
            if not key.startswith("__"):
                # 覆盖 cfg 中的默认值
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                else:
                    log.warning(f"注意: Config 类中不存在参数 '{key}'，已跳过赋值。")

        # 手动触发初始化后处理 (处理路径依赖等逻辑)
        if hasattr(cfg, '__post_init__'):
            cfg.__post_init__()

    # ----------------------------------------------------
    # 执行不同模式
    # ----------------------------------------------------
    cfg.show()     # 打印配置信息

    executor = Trainer(cfg)

    log.note(f"========================= 当前模式 {cfg.mode} =========================")
    match cfg.mode:
        case "run_train":
            executor.run_train()
        case "run_test":
            executor.run_test()

        case "run_train_test":
            executor.run_train()
            executor.run_test()
            executor.profile_model() # PD推理完成之后统计模型参数量计算量

        case "run_profile":
            # 统计模型参数量计算量/ 这个运行模式需要载入模型/测试集
            executor.profile_model()

        case _:
            print(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------
# 常用运行指令参考 (Usage Examples):
# ---------------------------------------------------------------------------------
# 1. 使用脚本默认配置直接运行 (推荐调试用):
#    python src/main/std.py
#
# 2. 使用 JSON 配置文件运行 (推荐实验记录用):
#    python src/main/std.py --config configs/exp001_snn.json
#
# 3. 后台运行并保存日志 (Linux):
#    nohup python3 ./src/main/std.py > ./logs/std.log 2>&1 &
