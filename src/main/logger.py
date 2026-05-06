# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-26
# @FilePath: \SNN\src\main\logger.py
# @Description: 自定义 logger 模块。
# -------------------------------------------------------

import logging
import os
from typing import Optional

SUCC_LEVEL_NUM = 21
NOTE_LEVEL_NUM = 25
logging.addLevelName(SUCC_LEVEL_NUM, "SUCC")
logging.addLevelName(NOTE_LEVEL_NUM, "NOTE")


class myLogger(logging.Logger):
    """带有额外 NOTE 和 SUCC 级别的自定义日志记录器。"""

    def succ(self, msg, *args, **kwargs):
        """记录 SUCC (成功) 级别的日志消息。

        Args:
            msg: 日志消息内容。
            *args: 日志记录的位置参数。
            **kwargs: 日志记录的关键字参数。
        """
        if self.isEnabledFor(SUCC_LEVEL_NUM):
            self._log(SUCC_LEVEL_NUM, msg, args, **kwargs)

    def note(self, msg, *args, **kwargs):
        """记录 NOTE (提示) 级别的日志消息。

        Args:
            msg: 日志消息内容。
            *args: 日志记录的位置参数。
            **kwargs: 日志记录的关键字参数。
        """
        if self.isEnabledFor(NOTE_LEVEL_NUM):
            self._log(NOTE_LEVEL_NUM, msg, args, **kwargs)


logging.setLoggerClass(myLogger)


class ColoredFormatter(logging.Formatter):
    """根据日志级别为控制台输出着色的格式化器。"""

    BRIGHT_GREEN = "\x1b[92m" # 亮绿色 (SUCC)
    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36m"         # 青色 (NOTE)
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.INFO: "%(levelname)s - %(message)s",
        SUCC_LEVEL_NUM: f"{BRIGHT_GREEN}%(levelname)s - %(message)s{RESET}",
        NOTE_LEVEL_NUM: f"{CYAN}%(levelname)s - %(message)s{RESET}",
        logging.WARNING: f"{YELLOW}%(levelname)s - %(message)s{RESET}",
        logging.ERROR: f"{RED}%(levelname)s - %(message)s{RESET}",
    }

    def format(self, record):
        """使用特定级别的颜色样式格式化单条日志记录。

        Args:
            record: 待格式化的日志记录对象。

        Returns:
            str: 格式化后的日志文本。
        """
        formatter = logging.Formatter(self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO]))
        return formatter.format(record)


def logger_init(logger_name: str = "mylogger", log_level_idx: int = 0, save_log: bool = False, save_dir: Optional[str] = None):
    """创建并初始化项目共享的日志记录器。

    Args:
        logger_name: 在 Python logging 系统中注册的日志记录器名称。
        log_level_idx: 整数型的日志级别选择器。
            0 -> INFO, 1 -> SUCC, 2 -> NOTE, 3 -> WARNING, 4 -> ERROR。
        save_log: 是否同时将日志写入文件。
        save_dir: 当 ``save_log`` 为 True 时，日志文件的输出目录。

    Returns:
        myLogger: 初始化后的日志记录器实例。

    Raises:
        ValueError: 如果启用了文件日志记录但缺少 ``save_dir``。
    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # 更新了映射表，加入了 SUCC_LEVEL_NUM
    level_map = {0: logging.INFO, 1: SUCC_LEVEL_NUM, 2: NOTE_LEVEL_NUM, 3: logging.WARNING, 4: logging.ERROR}
    logger.setLevel(level_map.get(log_level_idx, logging.INFO))

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(ColoredFormatter())
        logger.addHandler(stream_handler)

        if save_log:
            if save_dir is None:
                raise ValueError("当 save_log=True 时，必须提供 save_dir 路径。")

            os.makedirs(save_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(save_dir, f"{logger_name}.log"), encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

    return logger


# 默认实例化
log: myLogger = logger_init(log_level_idx=0, save_log=False, save_dir=None)
