# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-01-10
# @FilePath: \src\main\utils\loss.py
# @Description: 自定义loss_func,根据cfg.loss_func,在build中配置不同loss函数
# -------------------------------------------------------

import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if alpha is not None and not (0 < alpha < 1):
            raise ValueError("Focal Loss 的 Alpha 值必须在 0 和 1 之间。")

    def forward(self, logits, labels):
        # 1. 计算基础的 CrossEntropyLoss
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)

        # 2. 获取 ground truth 类的概率
        pt = torch.exp(-ce_loss)

        # 3. 计算 Focal Loss: (1 - pt)^gamma * CE_loss
        focal_loss = ((1 - pt)**self.gamma) * ce_loss

        if self.alpha is not None:
            # self.alpha 是 class 1 (目标) 的权重
            # (1 - self.alpha) 是 class 0 (杂波) 的权重
            alpha_t = torch.where(labels == 1, self.alpha, 1.0 - self.alpha).to(logits.device)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
