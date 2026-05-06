# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-04-26
# @FilePath: \SNN\src\utils\func.py
# @Description:
#   项目通用工具函数。
#   这里放和具体训练流程无关、可以被多个模块复用的基础能力。
# -------------------------------------------------------

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def read_json_with_comments(json_path: str) -> dict[str, Any]:
    """Load a JSON file while allowing line comments.

    中文说明：
        主要给配置文件使用，忽略 ``//`` 和 ``#`` 开头的整行注释。

    Args:
        json_path: JSON file path.

    Returns:
        dict[str, Any]: Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        json.JSONDecodeError: If the JSON content is invalid.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()

    cleaned_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        cleaned_lines.append(line)

    return json.loads("".join(cleaned_lines))


def to_json_safe(value: Any) -> Any:
    """Convert a value to JSON-serializable Python objects.

    Args:
        value: Value to convert. Tensor-like objects, array-like objects,
            dataclasses, dictionaries, tuples, and lists are converted
            recursively.

    Returns:
        Any: JSON-serializable value.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return to_json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return to_json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def save_dict_as_json(data: dict[str, Any], json_path: str | os.PathLike[str], indent: int = 4, ensure_ascii: bool = False) -> str:
    """Save a dictionary as a JSON file.

    Args:
        data: Dictionary to save. Values are converted with
            :func:`to_json_safe` before writing.
        json_path: Output JSON file path.
        indent: Indentation width passed to ``json.dump``.
        ensure_ascii: Whether non-ASCII characters should be escaped.

    Returns:
        str: Saved JSON file path.
    """
    json_path = os.fspath(json_path)
    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    safe_data = to_json_safe(data)
    with open(json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(safe_data, jsonfile, indent=indent, ensure_ascii=ensure_ascii)

    return json_path


def extract_public_attrs(obj: Any) -> dict[str, Any]:
    """Extract public attributes from a config object or dataclass.

    Args:
        obj: Dataclass instance, plain object, or config class.

    Returns:
        dict[str, Any]: Public non-callable attributes.
    """
    if is_dataclass(obj):
        return asdict(obj)

    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        result[key] = value
    return result


def match_dictkey(key_list: list[str], npy_prefix: Optional[str] = None, npy_number: Optional[str] = None) -> list[str]:
    """Filter keys by optional split prefix and radar sample number.

    Args:
        key_list: Candidate key list.
        npy_prefix: Optional split prefix such as ``train`` or ``test``.
        npy_number: Optional radar sample id such as ``#017``.

    Returns:
        list[str]: Matched keys.

    Raises:
        ValueError: If prefix or number format is invalid.
    """
    if npy_prefix is not None and npy_prefix not in ["train", "test", "val"]:
        raise ValueError("npy_prefix must be one of 'train', 'val', or 'test'.")

    if npy_number is not None and not re.match(r"^#\d{3}$", npy_number):
        raise ValueError("npy_number must match the format '#017'.")

    pattern_parts = []
    if npy_prefix:
        pattern_parts.append(rf"^{npy_prefix}")
    if npy_number:
        pattern_parts.append(re.escape(npy_number))

    if not pattern_parts:
        return key_list

    regex_pattern = "".join(rf"(?=.*{part})" for part in pattern_parts) + r".*"
    regex = re.compile(regex_pattern)
    return [key for key in key_list if regex.match(key)]


def match_npynum(f_name: str) -> Optional[str]:
    """Extract the ``#xxx`` radar sample number from a filename.

    Args:
        f_name: File name or sample string.

    Returns:
        Optional[str]: Matched number like ``#017`` or ``None``.
    """
    match = re.search(r"#\d{3}", f_name)
    if match:
        return match.group()
    return None
