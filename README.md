# DeepLearning  训练框架说明文档

这是一个模块化、易扩展的深度学习训练框架。框架集成了配置管理、模型构建、自动日志、可视化及标准化的训练/测试流程。

---

## 🚀 快速启动 (Quick Start)

### 1. 环境准备

确保已安装 Python 3.8+ 及 PyTorch 环境。
当前缺少requirements.txt
脉冲神经网络需要使用部分相关库

```bash
pip install -r requirements.txt

```

### 2. 运行默认演示

框架提供了标准启动脚本 `std.py`，默认配置下即可运行。

```bash
# 确保在项目根目录下
python src/main/std.py

```

### 3. 使用 GPU 运行

指定设备参数：

```bash
python src/main/std.py --device cuda:0

```

---

## 📂 框架结构说明

```text
SNN/
├── configs/                 # JSON 配置文件存储目录
├── logs/                    # 运行日志
├── results/                 # 实验结果 (Checkpoints, Visualizations)
├── src/
│   ├── datasets/            # 数据集定义 (IPIX, etc.)
│   ├── models/              # 模型定义 (SNN, CNN, Transformers)
│   └── main/
│       ├── std.py           # [核心] 标准启动脚本 (入口)
│       └── utils/
│           ├── base.py      # 配置类 (Config) 及基础工具
│           ├── build.py     # 工厂模式构建器 (Model, Loader, Optimizer)
│           ├── Train.py     # 训练器 (Trainer) 执行逻辑
│           └── loss.py      # 自定义损失函数
└── README.md                # 说明文档

```

---

## 📖 使用说明

本框架支持 **"脚本内直接配置"** 和 **"JSON 外部配置"** 两种模式，优先级：`JSON > 脚本内 Settings`。

### 1. 配置参数详解

核心配置位于启动脚本 (`src/main/std.py`) 或 JSON 文件中，主要参数如下：

| 模块 | 参数名 | 说明 |
| --- | --- | --- |
| **基础** | `experiment_name` | 实验名称，自动生成结果保存目录 `results/{name}/` |
|  | `mode` | 运行模式：`run_train` (训练+验证) / `run_test` (测试) |
|  | `device` | 运行设备，如 `cuda:0` 或 `cpu` |
| **模型** | `model_name` | 模型架构名称 (如 `skpformer`, `snn_ipix`) |
|  | `beta` | (SNN专用) 膜电位衰减因子 |
| **数据** | `dataset_name` | 数据集名称 (如 `ipix_tfg`) |
|  | `loader` | 加载模式 (如 `train_val`, `test_ipix_single_with_cfar`) |
| **训练** | `save_dir` | 设为 `None` 时自动根据实验名生成，否则使用指定路径 |

### 2. 开发调试模式 (推荐)

直接修改 `src/main/std.py` 中的 `Settings` 类变量。

```python
class Settings:
    experiment_name = "debug_exp_01"
    mode = "run_train"
    model_name = "skpformer"
    # ... 修改变量即可生效

```

### 3. 正式实验模式

编写 JSON 配置文件（例如 `configs/exp_v1.json`），通过命令行加载：

```bash
python src/main/std.py --config configs/exp_v1.json

```

### 4. 后台挂起训练

```bash
nohup python3 -u src/main/std.py > logs/exp_run.log 2>&1 &

```

---

## 📅 未来开发计划 (Roadmap)

### 🛠️ 框架优化

* [x] **配置系统重构**：实现 `Config` 数据类与 JSON 的解耦加载。
* [x] **标准化启动脚本**：完成 `std.py`，支持变量赋值式配置。
* [ ] **参数灵活覆盖**：允许用户通过外部脚本自定义组件（如 `net`, `dataset`）直接传入 Trainer，覆盖默认 `build` 逻辑。
* [ ] **参数校验防呆**：增加配置参数的类型检查和依赖检查。

### 🚀 训练与推理

* [x] **可视化**：集成进度条 (tqdm) 显示。
* [x] **模型管理**：支持 Checkpoint 保存、最优指标 (`Best Acc/Lowest Loss`) 自动保存。
* [ ] **断点续训**：支持从 Checkpoint 恢复训练状态（需固定 Random Seed）。
* [x] **推理流程**：完成模型载入、推理测试、指标计算。
* [ ] **高级可视化**：集成 TensorBoard 或 WandB 记录 Loss 曲线。

### 📦 数据处理

* [x] **Dataloader 封装**：支持通过 `loader` 参数切换不同的数据加载策略 (Train/Val/Test/CFAR)。

### 📝 日志系统

* [x] **彩色日志**：自定义 Logger，支持不同级别的颜色高亮。
* [x] **文件分流**：支持控制台输出与日志文件保存同步进行。

---

## 📝 版本更新记录

### v0.2.0

* **Refactor**: 引入 `std.py` 标准启动脚本，通过 `Settings` 类实现“变量即配置”。
* **Fix**: 修复了 Logger 在 Root Logger 开启传播时导致日志重复打印的问题。
* **Feat**: 完善了 `Config` 类的 `__post_init__` 逻辑，支持自动路径生成。

### v0.1.0

* 初始化项目结构。

---

