# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-18
# @FilePath: \SNN\src\main\metrics.py
# @Description: Metric helpers for training and evaluation.
# -------------------------------------------------------

from __future__ import annotations

import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert tensors or arrays to detached NumPy arrays."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _flatten_labels(labels: ArrayLike) -> np.ndarray:
    """Convert labels to a one-dimensional integer array."""
    return _to_numpy(labels).reshape(-1).astype(np.int64)


def _binary_predictions(outputs: ArrayLike) -> np.ndarray:
    """Convert binary logits, probabilities, or predicted labels to class ids."""
    outputs_np = _to_numpy(outputs)

    if outputs_np.ndim >= 2 and outputs_np.shape[-1] == 2:
        return np.argmax(outputs_np, axis=-1).reshape(-1).astype(np.int64)

    scores = outputs_np.reshape(-1)
    unique_values = np.unique(scores)
    if np.all(np.isin(unique_values, [0, 1])):
        return scores.astype(np.int64)

    threshold = 0.5 if scores.size > 0 and scores.min() >= 0.0 and scores.max() <= 1.0 else 0.0
    return (scores >= threshold).astype(np.int64)


def _multi_predictions(outputs: ArrayLike) -> np.ndarray:
    """Convert multiclass logits/probabilities or labels to class ids."""
    outputs_np = _to_numpy(outputs)
    if outputs_np.ndim >= 2 and outputs_np.shape[-1] > 1:
        return np.argmax(outputs_np, axis=-1).reshape(-1).astype(np.int64)
    return outputs_np.reshape(-1).astype(np.int64)


def _top_k_accuracy(outputs: ArrayLike, labels: ArrayLike, k: int) -> float | None:
    """Compute top-k accuracy for two-dimensional class scores."""
    scores = _to_numpy(outputs)
    y_true = _flatten_labels(labels)
    if scores.ndim != 2 or scores.shape[1] <= 1 or k > scores.shape[1]:
        return None

    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean([label in candidates for label, candidates in zip(y_true, topk)]))


def metrics_binary_class(outputs: ArrayLike, labels: ArrayLike) -> dict:
    """Compute binary classification metrics from model outputs and labels."""
    y_true = _flatten_labels(labels)
    y_pred = _binary_predictions(outputs)

    acc = accuracy_score(y_true, y_pred)
    pd = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pfa_hat = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "acc": float(acc),
        "pd": float(pd),
        "f1": float(f1),
        "precision": float(prec),
        "pfa_hat": float(pfa_hat),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_multi_class(outputs: ArrayLike, labels: ArrayLike) -> dict:
    """Compute closed-set multiclass classification metrics.

    Args:
        outputs: Multiclass logits/probabilities with shape ``[B, num_classes]``
            or predicted class labels with shape ``[B]``.
        labels: Ground-truth class labels.

    Returns:
        Dictionary containing aggregate metrics, optional top-k accuracy,
        confusion matrix, and per-class precision/recall/F1/support.
    """
    y_true = _flatten_labels(labels)
    y_pred = _multi_predictions(outputs)

    outputs_np = _to_numpy(outputs)
    if outputs_np.ndim >= 2 and outputs_np.shape[-1] > 1:
        class_ids = list(range(outputs_np.shape[-1]))
    else:
        class_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=class_ids)

    results = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "num_samples": int(y_true.size),
        "num_classes": int(len(class_ids)),
        "class_ids": [int(class_id) for class_id in class_ids],
        "confusion_matrix": conf_mat.astype(int).tolist(),
        "per_class": {
            int(class_id): {
                "precision": float(precision_per_class[idx]),
                "recall": float(recall_per_class[idx]),
                "f1": float(f1_per_class[idx]),
                "support": int(support_per_class[idx]),
            }
            for idx, class_id in enumerate(class_ids)
        },
    }

    top2_acc = _top_k_accuracy(outputs, labels, k=2)
    top3_acc = _top_k_accuracy(outputs, labels, k=3)
    top5_acc = _top_k_accuracy(outputs, labels, k=5)
    if top2_acc is not None:
        results["top2_acc"] = top2_acc
    if top3_acc is not None:
        results["top3_acc"] = top3_acc
    if top5_acc is not None:
        results["top5_acc"] = top5_acc

    return results


def print_confusion_matrix(f_name, tn, fp, fn, tp):
    """Print a binary confusion matrix as an ASCII table."""
    del f_name

    header = "Confusion Matrix ".center(40, "-")
    table = (f"{header}\n"
             f"{'':<15} | {'Pred Clutter':<12} | {'Pred Target':<12}\n"
             f"{'-' * 40}\n"
             f"{'Actual Clutter':<15} | {tn:<12} | {fp:<12} \n"
             f"{'Actual Target':<15} | {fn:<12} | {tp:<12} \n"
             f"{'-' * 40}")
    print(table)


def prinit_confusion_matrix(f_name, tn, fp, fn, tp):
    """Backward-compatible alias for the misspelled function name."""
    print_confusion_matrix(f_name, tn, fp, fn, tp)


def logits_process(logits: torch.Tensor, mode: str = "softmax") -> torch.Tensor:
    """Convert two-class logits to one-dimensional detection scores."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits shape [B, 2], got {tuple(logits.shape)}")

    match mode:
        case "z1_z0":
            return logits[:, 1] - logits[:, 0]
        case "z1":
            return logits[:, 1]
        case "softmax":
            return F.softmax(logits, dim=1)[:, 1]
        case _:
            raise ValueError(f"Unsupported score mode: {mode}")


def plot_score_distribution(clutter_scores, target_scores, threshold, pfa, f_name, save_dir):
    """Plot score distributions for clutter and target samples."""
    plt.figure(figsize=(12, 7))

    plt.hist(
        clutter_scores,
        bins=100,
        alpha=0.6,
        label="Clutter (Label 0)",
        color="steelblue",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )

    if len(target_scores) > 0:
        plt.hist(
            target_scores,
            bins=100,
            alpha=0.6,
            label="Target (Label 1)",
            color="salmon",
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

    plt.axvline(
        threshold,
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label=f"CFAR Threshold: {threshold:.4f}\n(Pfa={pfa})",
    )
    plt.title(f"{f_name}", fontsize=14)
    plt.xlabel("Target Probability")
    plt.ylabel("Probability Density")
    plt.xlim(-0.05, 1.05)
    plt.legend(loc="upper center", frameon=True, shadow=True)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    show_dir = os.path.join(save_dir, "show")
    os.makedirs(show_dir, exist_ok=True)
    plot_path = os.path.join(show_dir, f"{f_name}_pfa{pfa}.png")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_logit_distribution(clutter_logits, target_logits, threshold, pfa, f_name, save_dir):
    """Plot logit distributions for CFAR threshold inspection."""
    clutter_logits = np.array(clutter_logits).ravel()
    target_logits = np.array(target_logits).ravel() if target_logits else np.array([])

    plt.figure(figsize=(10, 6))

    c_mean, c_std = np.mean(clutter_logits), np.std(clutter_logits)
    all_vals = np.concatenate([clutter_logits, target_logits]) if target_logits.size > 0 else clutter_logits
    low, high = np.percentile(all_vals, [0.5, 99.5])

    plt.hist(
        clutter_logits,
        bins=100,
        range=(low, high),
        alpha=0.6,
        label=f"Clutter (mean={c_mean:.2f}, std={c_std:.2f})",
        color="steelblue",
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )

    if target_logits.size > 0:
        t_mean, t_std = np.mean(target_logits), np.std(target_logits)
        plt.hist(
            target_logits,
            bins=100,
            range=(low, high),
            alpha=0.6,
            label=f"Target (mean={t_mean:.2f}, std={t_std:.2f})",
            color="salmon",
            density=True,
            edgecolor="white",
            linewidth=0.3,
        )

    plt.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.4f}")
    plt.title(f"{f_name}\n(Pfa={pfa})")
    plt.xlabel("Logit Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    show_dir = os.path.join(save_dir, "show")
    os.makedirs(show_dir, exist_ok=True)
    save_path = os.path.join(show_dir, f"{f_name}_pfa{pfa}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
