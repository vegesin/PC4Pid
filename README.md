# 深度学习训练测试框架

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行一个实验启动脚本：

```bash
python src/runner/ours_train.py
```

也可以运行标准示例：

```bash
python src/runner/std.py
```

如果使用 JSON 配置，可以通过主入口传入：

```bash
python -m src.main.main --config configs/exp.json
```

## 项目结构

```text
SNN/
├── README.md
├── src/
│   ├── datasets/
│   │   └── ipix_dataset.py        # IPIX 相关 Dataset 与内置数据集注册
│   ├── main/
│   │   ├── base.py                # 分层配置 dataclass
│   │   ├── main.py                # 框架主入口与 mode 分发
│   │   ├── train.py               # Trainer 训练/测试/profile 流程
│   │   ├── model_factory.py       # 模型创建
│   │   ├── train_factory.py       # loss / optimizer / scheduler / save path 创建
│   │   ├── data_factory.py        # dataset / dataloader 创建与数据划分
│   │   ├── metrics.py             # 指标函数
│   │   ├── loss.py                # loss 函数
│   │   └── logger.py              # 日志模块
│   ├── models/                    # 模型定义
│   ├── runner/                    # 实验启动脚本
│   ├── scripts/                   # 数据处理和调试脚本
│   └── utils/
│       └── func.py                # 通用工具函数
```

当前设计中：

- `src/runner/` 只负责配置实验参数并调用 `launch()`。
- `src/main/main.py` 负责读取配置、创建 `Trainer`、按 `mode` 分发任务。
- `src/main/base.py` 定义分层配置对象。
- `src/main/*_factory.py` 负责按功能创建组件。
- `src/datasets/` 负责具体数据集读取、采样和数据格式处理。

## 配置结构

配置采用分层 dataclass，而不是一个扁平 `cfg` 传到底。入口脚本一般写成：

```python
from src.main.base import DataConfig, EvalConfig, ModelConfig, RunConfig, TrainConfig
from src.main.main import launch


class Settings:
    run = RunConfig(
        experiment_name="exp_name",
        mode="run_train",
        device="cuda:0",
    )

    model = ModelConfig(
        model_name="my_spike_net",
        num_steps=4,
        extra={},
    )

    data = DataConfig(
        dataset_name="ipix_tfg_all",
        loader_mode="train_val_test",
        train_path="path/to/train.npz",
        val_path="path/to/val.npz",
        test_path="path/to/test.npz",
        batch_size=16,
        num_workers=8,
        extra={
            "tc_ratio": 1 / 3,
            "seed": 42,
        },
    )

    train = TrainConfig(
        num_epochs=100,
        optimizer_type="adam",
        scheduler_type="cosine",
        learning_rate=1e-3,
        min_lr=1e-6,
        loss_func="bce_with_logits",
    )

    eval = EvalConfig(
        pfa=1e-3,
        logits_process="z1",
    )


if __name__ == "__main__":
    launch(settings_cls=Settings)
```

### 配置分组

- `RunConfig`：实验名、运行模式、设备、保存路径、加载权重路径。
- `ModelConfig`：模型名、时间步数、模型额外参数。
- `DataConfig`：数据集名、loader 模式、数据路径、batch size、worker 数。
- `TrainConfig`：epoch、优化器、学习率、scheduler、loss。
- `EvalConfig`：虚警率、logits 后处理方式、指标函数。

每个子配置都有 `extra` 字段，用来放当前模块的扩展参数。原则是：

- 通用字段放在主配置字段中。
- 模型、数据集、训练策略的专用参数放进对应的 `extra`。
- 底层函数只接收自己需要的参数，不再依赖一个全局 `cfg`。

## 运行模式

`RunConfig.mode` 由 `Trainer` 分发，目前常用模式包括：

- `run_train`：训练模型。
- `run_test`：加载模型并测试。
- `run_train_test`：训练后执行测试。
- `run_profile`：统计模型参数量、推理耗时等 profile 信息。

具体支持的模式以 [src/main/train.py](src/main/train.py) 中 `Trainer` 方法为准。

## Factory 分层

框架将原来的单个 `build.py` 拆成三个工厂文件：

### model_factory.py

负责模型创建：

```python
build_model(model_name, num_steps, model_extra)
```

模型专用参数从 `ModelConfig.extra` 进入，并在模型工厂内部按需要解包。

### train_factory.py

负责训练组件创建：

```python
build_loss_func(loss_name)
build_optimizer(model, optimizer_type, learning_rate, weight_decay)
build_scheduler(optimizer, scheduler_type, num_epochs, min_lr)
build_save_path(save_dir)
```

### data_factory.py

负责数据链路：

```python
build_dataset(...)
plan_base_datasets(...)
attach_threshold_dataset(...)
build_dataloaders(...)
build_data_components(...)
```

`Trainer` 只调用 `build_data_components(data_cfg)`，最终拿到一个 loader 字典：

```python
{
    "train": train_loader,
    "val": val_loader,
    "test": test_loader,
    "get_th": threshold_loader,
}
```

其中不存在的 loader 不会出现在字典中。

## 数据集说明

### Dataset 基本接口

框架推荐所有 dataset 使用统一入口：

```python
DatasetClass(file_paths, **kwargs)
```

其中：

- `file_paths`：数据文件路径，支持字符串或路径列表。
- `**kwargs`：数据集内部参数，例如 `tc_ratio`、`seed`、`only_positive`、`save_rest_clutter`。

数据集内部自行决定需要读取哪些参数。多传但未使用的参数默认不影响运行。

### 内置 IPIX 数据集

当前内置数据集定义在 [src/datasets/ipix_dataset.py](src/datasets/ipix_dataset.py)：

- `ipix`：原始 `.npy` IPIX 数据。
- `ipix_tfg`：单个 `.npz` TFG 数据。
- `ipix_tfg_all`：合并多个 `.npz` TFG 数据，并支持重采样。
- `ipix_tfg_all_pos`：只保留正样本的 TFG 数据。
- `ipix_mdccnn`：MDCCNN 输入格式数据。

这些内置数据集通过注册表注册：

```python
@register_dataset("ipix_tfg_all")
class ipix_tfg_all_dataset(Dataset):
    ...
```

`data_factory.py` 使用字符串 `dataset_name` 查找注册表并创建 dataset。

### 自定义数据集

自定义数据集有两种接入方式。

方式一：直接在 runner 中传入 dataset 类或构造函数：

```python
data = DataConfig(
    dataset_name=MyDataset,
    loader_mode="train_val_test",
    train_path="train.xxx",
    val_path="val.xxx",
    test_path="test.xxx",
)
```

要求 `MyDataset` 支持：

```python
MyDataset(file_paths, **kwargs)
```

方式二：注册为内置数据集：

```python
from src.datasets.ipix_dataset import register_dataset


@register_dataset("my_dataset")
class MyDataset(Dataset):
    ...
```

然后 runner 中使用：

```python
dataset_name="my_dataset"
```

注意：当前注册表定义在 `ipix_dataset.py` 中，后续如果数据集变多，建议迁移到独立的 `src/datasets/registry.py`。

## Loader 模式

`DataConfig.loader_mode` 控制数据集如何创建。

### train_val_test

显式给出 train / val / test 路径：

```python
data = DataConfig(
    dataset_name="ipix_tfg_all",
    loader_mode="train_val_test",
    train_path="train.npz",
    val_path="val.npz",
    test_path="test.npz",
    extra={
        "tc_ratio": 1 / 3,
        "seed": 42,
    },
)
```

`data_factory.py` 会分别创建 train、val、test dataset。缺少某个路径时，该角色为 `None`。

### auto_split_train_val_test

通用自动划分模式。流程是：

1. 根据 `train_path` 创建一个完整 dataset。
2. 使用 PyTorch `random_split` 按比例切分。
3. 返回 train / val / test 三个子集。

示例：

```python
data = DataConfig(
    dataset_name="ipix_tfg_all",
    loader_mode="auto_split_train_val_test",
    train_path=["all_data_1.npz", "all_data_2.npz"],
    extra={
        "tc_ratio": 1 / 3,
        "seed": 42,
        "train_val_test_split_ratios": (8, 1, 1),
    },
)
```

这个模式是通用逻辑，不使用 IPIX 实验里的特殊 `create_splits()`。

### ipix_tfg_auto_split_train_val_test

这是当前 IPIX TFG 实验使用的特殊划分模式，会调用：

```python
ipix_tfg_auto_split_dataset.create_splits(...)
```

适用于当前实验中需要：

- 按标签分层划分 target / clutter。
- 对每个 split 使用同一个 `tc_ratio`。
- 可选保留 train 重采样后剩余的 clutter 数据，用于虚警阈值估计。

示例见 [src/runner/ours_train.py](src/runner/ours_train.py)：

```python
data = DataConfig(
    dataset_name="ipix_tfg_all",
    loader_mode="ipix_tfg_auto_split_train_val_test",
    train_path=[
        "../dataset/ipix_tfg_random_hh_s128_complex/train/train_xxx_tfg.npz",
    ],
    extra={
        "tc_ratio": 1 / 3,
        "train_val_test_split_ratios": (8, 0, 2),
        "get_rest_clutter_dataset": True,
    },
)
```



## 输出目录

默认输出目录由 `RunConfig` 自动生成：

```text
./results/{experiment_name}
```

其中 checkpoint 默认路径为：

```text
./results/{experiment_name}/checkpoints/best_ckpt.pth
```

也可以在 `RunConfig` 中显式指定：

```python
RunConfig(
    experiment_name="exp",
    save_dir="./results/custom_exp",
    load_model_path="./results/custom_exp/checkpoints/best_ckpt.pth",
)
```

## 开发约定

- runner 不写训练逻辑，只写配置。
- main 不创建具体组件，只负责加载配置和分发。
- Trainer 负责组织训练/测试流程，但不直接关心 dataset 如何构造。
- factory 负责组件创建。
- dataset 自己解析自己的 `**kwargs`。
- 数据集主入口统一为 `file_paths, **kwargs`。

## 当前注意事项

- 一些旧 runner 仍可能使用旧的扁平配置或 `train_tc_ratio / val_tc_ratio / test_tc_ratio` 字段，需要逐步迁移到分层配置和统一 `tc_ratio`。
- 当前数据集注册表还在 `ipix_dataset.py` 中，后续数据集类型变多后建议拆到 `src/datasets/registry.py`。
- 本仓库当前没有随附数据文件，训练和测试需要本地数据路径正确配置后才能运行。
