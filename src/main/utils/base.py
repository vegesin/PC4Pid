# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-13
# @FilePath: \radar_pc\src\main\utils\base.py
# @Description: 基础通用组件初始化，包含一些通用功能的类、函数等等
# -------------------------------------------------------

import os
import logging
import re
import json
from typing import List, Optional
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------
# logger 配置
# --------------------------------------------------------

# 定义级别数值
NOTE_LEVEL_NUM = 25
logging.addLevelName(NOTE_LEVEL_NUM, "NOTE")


# 创建自定义 Logger 类，显式定义 note 方法
class myLogger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def note(self, msg, *args, **kwargs):
        """
        自定义 NOTE 级别日志，IDE 现在可以识别并补全此方法
        """
        if self.isEnabledFor(NOTE_LEVEL_NUM):
            self._log(NOTE_LEVEL_NUM, msg, args, **kwargs)


# 告诉 logging 模块使用自定义类
logging.setLoggerClass(myLogger)


class ColoredFormatter(logging.Formatter):
    # ANSI 颜色码
    GREY = "\x1b[38;20m"    # 灰色
    CYAN = "\x1b[36m"       # 青色 (用于 NOTE)
    YELLOW = "\x1b[33m"     # 黄色 (用于 WARN)
    RED = "\x1b[31m"        # 红色 (用于 ERROR)
    BOLD_RED = "\x1b[31;1m" # 加粗红
    RESET = "\x1b[0m"       # 重置颜色

    # 0: INFO (不带颜色)
    # 1: NOTE (青色)
    # 2: WARN (黄色)
    # 3: ERROR (红色)
    FORMATS = {
        logging.INFO: "%(levelname)s - %(message)s",                    # 0级：无颜色
        NOTE_LEVEL_NUM: f"{CYAN}%(levelname)s - %(message)s{RESET}",    # 1级：全行青色
        logging.WARNING: f"{YELLOW}%(levelname)s - %(message)s{RESET}", # 2级：全行黄色
        logging.ERROR: f"{RED}%(levelname)s - %(message)s{RESET}",      # 3级：全行红色
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logger_init(logger_name="mylogger", log_level_idx=0, save_log=False, save_dir=None):
    """
    配置带有彩色输出和自定义等级的日志系统
    log_level_idx: 0=INFO, 1=NOTE, 2=WARN, 3=ERROR
    """

    logger = logging.getLogger(logger_name)
    logger.propagate = False # 防止日志向上传播到 Root Logger 导致重复打印

    # 映射等级索引到 logging 内部级别
    level_map = {0: logging.INFO, 1: NOTE_LEVEL_NUM, 2: logging.WARNING, 3: logging.ERROR}
    logger.setLevel(level_map.get(log_level_idx, logging.INFO))

    if not logger.handlers:
        # 控制台 Handler (带颜色)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(ColoredFormatter())
        logger.addHandler(stream_handler)

        # 文件 Handler (不带颜色，避免文本出现乱码字符)
        if save_log:
            if save_dir is None:
                print("[err]: 缺少日志保存目录，日志仅在终端输出")
            else:
                os.makedirs(save_dir, exist_ok=True)
                file_formatter = logging.Formatter('%(levelname)s - %(message)s')
                logger_savepath = os.path.join(save_dir, f"{logger_name}.log")

                file_handler = logging.FileHandler(logger_savepath, encoding='utf-8')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

    return logger


# fix base里面创建logger,后续所有的东西都从base导入logger进行输出
log: myLogger = logger_init(log_level_idx=0, save_log=False, save_dir=None)



# --------------------------------------------------------
# config类 定义
# --------------------------------------------------------
@dataclass
class Config:
    """
    ### @ 全局配置类
    详细配置等待补充
    # TODO  后续组件以及当前组件配置针对None缺失参数进行配置兼容
    # TODO  框架兼容配置中这些参数后续是要修复的
    """

    # 新增以下字段（可放在最后）
    num_classes: int = None
    target_frame_num: int = 200
    target_point_num: int = 128
    install_angle: float = 25.0
    install_height: float = 2.0
    load_config: bool = True
    use_normalization = True                      # 是否启用特征归一化
    stats_path = "F:/radar_data/train_stats.pkl"  # 统计量文件路径（根据你实际保存的位置）
    # --------------------------------------------------------
    # 基础配置
    # --------------------------------------------------------
    experiment_name: str = None
    mode: str = None
    device: str = "cuda:0"

    # 结果保存路径依赖 experiment_name
    save_dir: str = field(default=None)
    load_mode_path: str = field(default=None)
    model_save_interval: int = 20 # 模型ckpt保存周期

    # 兼容配置 | 下面这些配置是为了兼容框架的冗余或者bug，没用但是缺少可能报错
    logits_process = "z1"

    # --------------------------------------------------------
    # 模型配置
    # --------------------------------------------------------
    model_name: str = None

    bias: bool = True # 模型内部的 bn/fc 偏置是否启动

    # 其他模型专属超参数
    # SNN 专用超参数
    beta: float = 0.9        # 神经元膜电位衰减
    learn_beta: bool = False # 是否学习 beta
    num_steps: int = 8       # 时间步

    # --------------------------------------------------------
    # 数据集配置
    # --------------------------------------------------------
    dataset_name: str = None
    loader: str = None

    # 建议尽量使用绝对路径或基于项目根目录的路径
    train_path: str = None
    val_path: str = None
    test_path: str = None

    # 特殊数据加载需要的参数
    train_tc_ratio: str = None
    val_tc_ratio: str = None
    test_tc_ratio: str = None

    seed: int = 42

    # --------------------------------------------------------
    # 训练超参数
    # --------------------------------------------------------
    batch_size: int = 16
    num_epochs: int = 50
    num_workers: int = 1

    # 优化器与学习率
    optimizer_type: str = "adam"
    scheduler_type: Optional[str] = "cosine" # 允许为 None
    learning_rate: float = 1e-3
    min_lr: float = 1e-6

    # 损失与指标
    loss_func: str = None
    metrics_func: Optional[str] = None
    pfa: float = 1e-3

    def __post_init__(self):
        """
        初始化后处理：
        用于处理依赖于其他字段的默认值（如路径依赖实验名）
        """
        if self.save_dir is None:
            self.save_dir = f"./results/{self.experiment_name}/"

        if self.load_mode_path is None:
            self.load_mode_path = f"./results/{self.experiment_name}/checkpoints/best_ckpt.pth"

    @classmethod
    def from_json(cls, json_path: str):
        """从带有注释的 JSON 文件加载配置"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config file not found: {json_path}")

        config_dict = read_json_withcomment(json_path)

        # 过滤掉 JSON 中存在但 Config 类中未定义的键（防止报错）
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

    def show(self):
        """
        打印关键参数，格式化输出
        """
        print("*" * 80)
        log.note(f"Configuration: {self.experiment_name}")
        print("-" * 80)

        # 将配置转为字典
        cfg_dict = asdict(self)

        # 定义分组打印
        # yapf:disable
        groups = {
            "Base": ["mode", "device", "save_dir", "load_mode_path"],
            # "Model": ["model_name", "beta", "num_steps", "bias", "learn_beta"],
            "Model" :["model_name"],
            "Data": ["dataset_name", "batch_size", "num_workers", "loader"],
            "Training": ["optimizer_type", "learning_rate", "num_epochs", "loss_func"]
        }
        # yapf:enable

        printed_keys = set()

        # 按组打印
        for group_name, keys in groups.items():
            print(f"[{group_name}]")
            for key in keys:
                if key in cfg_dict:
                    print(f"{key:<20} : {cfg_dict[key]}")
                    printed_keys.add(key)
            print("") # 空行

        # 打印剩余未归类的参数
        others = [k for k in cfg_dict.keys() if k not in printed_keys]
        if others:
            print("[Others]")
            for key in others:
                # 对于很长的路径，可以截断显示
                val = str(cfg_dict[key])
                if len(val) > 50:
                    val = "..." + val[-47:]
                print(f"{key:<20} : {val}")

        print("*" * 80)


# --------------------------------------------------------
# 基础功能函数
# --------------------------------------------------------
def read_json_withcomment(json_path: str) -> dict:
    """读取并解析带有注释（//）的 JSON 配置文件。

    Args:
        json_path (str): 读取的json文件路径

    Returns:
        config(dict): 读取到的参数字典
    """
    # 读取配置文件
    if not os.path.exists(json_path):
        print(f"Error: Config file not found at {json_path}\n 使用 --config './config/test.json' 传入配置文件 ")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 移除每行的注释
        processed_lines = []
        for line in lines:
            cleaned_line = line.split('//')[0]
            processed_lines.append(cleaned_line)

        # 将处理后的行组合成字符串
        json_str = ''.join(processed_lines)

        # 解析 JSON
        config = json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"Error: 当前传入的json存在语法错误 {json_path}")
        print(f"  Error details: {e}")
        print(f"  Line {e.lineno}, column {e.colno}: {e.msg}")
        print("根据上述报错信息检查json文件,程序退出")
        exit(1)
    except Exception as e:
        print(f"Error: 不能正常读取json，检查文件 {json_path}")
        print(f"  Error details: {e}")
        print("根据上述报错信息检查json文件,程序退出")
        exit(1)

    return config


def match_dictkey(key_list: List[str], npy_prefix: Optional[str] = None, npy_number: Optional[str] = None) -> List[str]:
    """根据指定的模式匹配字符串列表中的键。

    支持三种匹配场景：
    1. 仅匹配开头前缀（train/test）；
    2. 仅匹配#后跟3位数字的序号（如#017）；
    3. 同时匹配前缀和序号，实现精准匹配。
    
    ### info 可以扩展兼容匹配不同的字典键名

    Args:
        key_list: 待匹配的字符串列表（字典的所有键）。
        npy_prefix: 可选，要匹配的开头前缀，仅支持"train"或"test"；
                不传则不限制前缀。
        npy_number: 可选，要匹配的序号标识（如"#017"），格式必须是#后跟3位数字；
                不传则不限制序号。

    Returns:
        符合所有匹配条件的字符串列表。


    """

    # 验证输入参数的合法性
    if npy_prefix is not None and npy_prefix not in ["train", "test", "val"]:
        raise ValueError("npy_prefix参数仅支持'train','val','test'")

    if npy_number is not None:
        if not re.match(r'^#\d{3}$', npy_number):
            raise ValueError("npy_number格式必须为#后跟3位数字，例如#017")

    # 构建正则表达式模式
    pattern_parts = []
    # 1. 处理前缀匹配（开头必须是指定前缀）
    if npy_prefix:
        pattern_parts.append(rf'^{npy_prefix}')
    # 2. 处理序号匹配（包含指定的#xxx标识）
    if npy_number:
        pattern_parts.append(re.escape(npy_number))

    # 拼接正则模式：多个条件用非捕获组的AND逻辑（?=...）实现
    if pattern_parts:
        # 组合成正向预查，确保所有条件都满足
        regex_pattern = r''.join([rf'(?=.*{part})' for part in pattern_parts]) + r'.*'
    else:
        # 无匹配条件时返回空列表（也可根据需求改为返回原列表）
        return None

    # 编译正则表达式，提升匹配效率
    regex = re.compile(regex_pattern)

    # 筛选符合条件的键
    matched_keys = [key for key in key_list if regex.match(key)]

    return matched_keys


def match_npynum(f_name: str) -> Optional[str]:
    """从文件名/键名中提取#后跟3位数字的序号（如#017）。
    
    Args:
        f_name: 待提取的字符串（如"val_#017_19931107_starea.npy"）
    
    Returns:
        提取到的序号（如#017），无匹配则返回None
    """
    # 正则匹配#后跟3位数字的模式
    match = re.search(r'#\d{3}', f_name)
    if match:
        return match.group()
    return None
