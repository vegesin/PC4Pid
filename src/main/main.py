# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-13
# @FilePath: \SNN\src\main\main.py
# @Description:框架统一主入口。启动脚本只负责提供 Settings；
# -------------------------------------------------------

import argparse

from .base import *
from .logger import log
from .train import Trainer


def parse_args():
    """解析命令行Args以供 CLI 入口使用。

    Returns:
        argparse.Namespace: 解析后的命令行Args。
    """
    parser = argparse.ArgumentParser(description="深度学习训练框架")
    parser.add_argument("--config", type=str, default="", help="JSON 配置文件路径")
    return parser.parse_args()


def execute(experiment: ExperimentConfig):
    """将结构化的实验配置分发给训练器 (Trainer)。

    这里是运行时边界：
    - main 层只处理配置与模式分发；
    - 真正的训练、测试、profile 逻辑全部交给 Trainer。

    Args:
        experiment (ExperimentConfig): 结构化的实验配置对象。
    
    Raises:
        ValueError: 如果提供的运行模式 (run_cfg.mode) 不受支持。
    """

    experiment.show()

    run_cfg = experiment.run
    trainer = Trainer(experiment)

    log.note(f"========================= 运行模式: {run_cfg.mode} =========================")

    match run_cfg.mode:
        case "run_train":
            trainer.run_train()
        case "run_test":
            trainer.run_test()
            trainer.profile_model()
        case "run_train_test":
            trainer.run_train()
            trainer.run_test()
            trainer.profile_model()
        case "run_profile":
            trainer.profile_model()
        case _:
            raise ValueError(f"未知的运行模式: {run_cfg.mode}")


def launch(settings_cls=None, config_path: str = ""):
    """加载配置并启动所选的实验模式。

    Args:
        settings_cls (可调用对象 | None, 可选): 运行器 (Runner) 提供的设置类。默认为 None。
        config_path (str, 可选): JSON 配置文件路径。默认为空字符串。
    """
    experiment = load_experiment_config(settings_cls=settings_cls, config_path=config_path)
    execute(experiment)


def run():
    """用于 ``python -m`` 或直接脚本启动的命令行 (CLI) 入口。"""
    args = parse_args()
    launch(config_path=args.config)
