# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-18
# @FilePath: \radar_pc\src\main\utils\metrics.py
# @Description: 不同任务验证评估指标
# -------------------------------------------------------

import numpy as np
from sklearn import metrics
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from typing import Union
import torch.nn.functional as F
# >--------------------------------------------------------------
# > 指标计算函数
# >--------------------------------------------------------------


def metrics_binary_class(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    针对二分类任务的指标封装。
    
    Args:
        y_true: 真实标签，形状为 [N] 的 numpy 数组或 PyTorch Tensor
        y_pred: 预测标签，形状为 [N] 的 numpy 数组或 PyTorch Tensor
    
    Returns:
        dict: 包含以下指标的字典：
            - acc: 准确率
            - pd: 检测概率（等同于召回率）
            - f1: F1分数
            - precision: 精确率
            - 混淆矩阵
    
    Note:
        Pd (Detection Probability) 在信号处理/二分类中通常等同于 Recall
        如果传入Tensor，函数会自动转换为numpy数组
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # 1. 基础分类指标计算
    acc = accuracy_score(y_true, y_pred)
    pd = recall_score(y_true, y_pred, zero_division=0) # 等同于 Recall
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    # 2. 计算混淆矩阵
    # 对于二分类 [0, 1]，cm 的结构为 [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 展开混淆矩阵以提取雷达专用指标
    # 注意：如果数据集中只存在单一样本类别，ravel() 可能会报错，需确保 labels=[0, 1]
    tn, fp, fn, tp = cm.ravel()

    # 3. 计算实际虚警率 Pfa (False Alarm Probability)
    # Pfa = FP / (FP + TN)
    pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "acc": acc,
        "pd": pd,
        "f1": f1,
        "precision": prec,
        "pfa": pfa,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,          # 展开混淆矩阵
                           # "cm": cm.tolist()  # 返回原始矩阵列表格式
    }


def metrics_multi_class(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    针对多分类任务的指标封装。
    
    Args:
        y_true: 真实标签，形状为 [N] 的 numpy 数组或 PyTorch Tensor
        y_pred: 预测标签，形状为 [N] 的 numpy 数组或 PyTorch Tensor
    
    Returns:
        dict: 包含准确率的字典
    
    Note:
        如果传入Tensor，函数会自动转换为numpy数组
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    return {"acc": acc}


def prinit_confusion_matrix(f_name, tn, fp, fn, tp):
    """
    格式化打印混淆矩阵 ASCII 表格
    """
    # 计算总数
    total_clutter = tn + fp
    total_target = fn + tp

    # 构建表格字符串
    header = f"Confusion Matrix ".center(40, "-")
    table = (f"{header}\n"
             f"{'':<15} | {'Pred Clutter':<12} | {'Pred Target':<12}\n"
             f"{'-'*40}\n"
             f"{'Actual Clutter':<15} | {tn:<12} | {fp:<12} \n"
             f"{'Actual Target':<15} | {fn:<12} | {tp:<12} \n"
             f"{'-'*40}")
    print(table)


def logits_process(
    logits: torch.Tensor,
    mode: str = "softmax",
) -> torch.Tensor:
    """
    模型的逻辑输出处理，传入的tensor维度需要是[B,2]

    Args:
        logits (Tensor): Model output logits with shape [B, 2].
        mode (str): Score type.
            - "z1_z0": log-likelihood ratio (z1 - z0)
            - "z1" : logtis z1
            - "softmax": softmax probability of target class

    Returns:
        Tensor: Detection score with shape [B].
    """
    assert logits.shape[1] == 2, f"Expected logits shape [B, 2], got {logits.shape}"

    match mode:
        case "z1_z0":
            return logits[:, 1] - logits[:, 0]
        case "z1":
            return logits[:, 1]
        case "softmax":
            return F.softmax(logits, dim=1)[:, 1]
        case _:
            raise ValueError(f"Unsupported score mode: {mode}")


# >--------------------------------------------------------------
# > 绘图函数
# >--------------------------------------------------------------
import matplotlib.pyplot as plt
import os


def plot_score_distribution(clutter_scores, target_scores, threshold, pfa, f_name, save_dir):
    """绘制并保存杂波与目标的得分分布直方图.

    使用 Matplotlib 绘制模型输出概率的分布情况，并标记出当前的 CFAR 门限。
    该函数有助于直观分析模型对两类样本的区分能力及 Softmax 饱和情况。

    Args:
        clutter_scores (list[float]): 杂波样本（Label 0）的目标概率得分列表.
        target_scores (list[float]): 目标样本（Label 1）的目标概率得分列表.
        threshold (float): 当前计算得到的 CFAR 决策门限.
        pfa (float): 设定的虚警概率.
        f_name (str): 数据源名称，用于生成标题和文件名.
        save_dir (str): 图像保存的根目录.
    """

    plt.figure(figsize=(12, 7))

    # 设置绘图风格
    bins = 100
    alpha = 0.6

    # 1. 绘制杂波分布 (使用蓝色系)
    plt.hist(clutter_scores, bins=bins, alpha=alpha, label='Clutter (Label 0)', color='steelblue', density=True, edgecolor='white', linewidth=0.5)

    # 2. 绘制目标分布 (使用红色系) - 仅当存在目标样本时
    if len(target_scores) > 0:
        plt.hist(target_scores, bins=bins, alpha=alpha, label='Target (Label 1)', color='salmon', density=True, edgecolor='white', linewidth=0.5)

    # 3. 绘制 CFAR 门限垂线
    plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2.5, label=f'CFAR Threshold: {threshold:.4f}\n(Pfa={pfa})')

    # 4. 图表细节配置
    plt.title(f"{f_name}", fontsize=14)
    plt.xlabel("Target Probability ($\hat{y}_{target}$)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(-0.05, 1.05) # 强制显示 0-1 完整范围
    plt.legend(loc='upper center', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    # 5. 保存图像
    show_dir = os.path.join(save_dir, "show")
    os.makedirs(show_dir, exist_ok=True)

    file_tag = f_name
    plot_path = os.path.join(show_dir, f"{file_tag}_pfa{pfa}.png")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # log.info(f"目标杂波分布直方图已保存至: {plot_path}")


# yapf:disable
def plot_logit_distribution(clutter_logits, target_logits, threshold, pfa, f_name, save_dir):
    """绘制原始逻辑输出分布直方图 (已修正维度问题)."""

    # 【关键修复】强制转换为一维 NumPy 数组，防止 Matplotlib 误判数据集数量
    clutter_logits = np.array(clutter_logits).ravel()
    target_logits = np.array(target_logits).ravel() if target_logits else np.array([])

    plt.figure(figsize=(10, 6))

    # 统计信息计算
    c_mean, c_std = np.mean(clutter_logits), np.std(clutter_logits)

    # 设置动态范围，排除异常值干扰
    all_vals = np.concatenate([clutter_logits, target_logits]) if target_logits.size > 0 else clutter_logits
    low, high = np.percentile(all_vals, [0.5, 99.5])

    # 1. 绘制杂波分布
    plt.hist(clutter_logits, bins=100, range=(low, high), alpha=0.6,
             label=f'Clutter (μ={c_mean:.2f}, σ={c_std:.2f})',
             color='steelblue', density=True, edgecolor='white', linewidth=0.3)

    # 2. 绘制目标分布
    if target_logits.size > 0:
        t_mean, t_std = np.mean(target_logits), np.std(target_logits)
        plt.hist(target_logits, bins=100, range=(low, high), alpha=0.6,
                 label=f'Target (μ={t_mean:.2f}, σ={t_std:.2f})',
                 color='salmon', density=True, edgecolor='white', linewidth=0.3)

    # 3. 绘制门限线
    plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.4f}')

    plt.title(f"{f_name}\n(Pfa={pfa})")
    plt.xlabel("Logit Value (Target Channel)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 保存路径逻辑
    show_dir = os.path.join(save_dir, "show")
    os.makedirs(show_dir, exist_ok=True)
    file_tag = f_name
    save_path = os.path.join(show_dir, f"{file_tag}_pfa{pfa}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
# yapf:enable
