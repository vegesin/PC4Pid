# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-26
# @FilePath: \SNN\src\runner\std.py
# @Description:
#   标准训练入口示例。
#   runner 侧直接组织层级配置，再交给 main 启动。
# -------------------------------------------------------

from src.main.base import DataConfig, EvalConfig, ModelConfig, RunConfig, TrainConfig
from src.main.main import launch


class Settings:
    """标准实验配置示例。"""

    # 运行配置
    run = RunConfig(
        experiment_name="std",
        mode="run_train",
        device="cuda:0",
    )

    # 模型配置
    # bias/beta/learn_beta 当前不属于稳定公共字段，先放到 extra。
    model = ModelConfig(
        model_name="spkformer",
        num_steps=8,
        extra={
            "bias": True,
            "beta": 0.9,
            "learn_beta": False,
        },
    )

    # 数据配置
    data = DataConfig(
        dataset_name="ipix_tfg",
        loader_mode="train_val_test",
        train_path="../dataset/ipix_tfg/ipix_train_tfg/train_#017_19931107_135603_starea_tfg.npz",
        val_path="../dataset/ipix_tfg/ipix_val_tfg/val_#017_19931107_135603_starea_tfg.npz",
        test_path=None,
        batch_size=32,
        num_workers=10,
    )

    # 训练配置
    train = TrainConfig(
        num_epochs=50,
        optimizer_type="adam",
        learning_rate=1e-3,
        loss_func="bce_with_logits",
    )

    # 评估配置
    eval = EvalConfig(
        pfa=1e-3,
        logits_process="z1",
    )


if __name__ == "__main__":
    launch(settings_cls=Settings)
