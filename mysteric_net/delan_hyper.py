"""
按关节自由度 n_dof 与样本量推荐 DeLaN 超参（L-Net 训练用）。
"""

from __future__ import annotations

import math
from typing import Any


def suggest_hyper(n_dof: int, n_train_samples: int, *, base: str = "delan_model") -> dict[str, Any]:
    """
    返回 hyper 字典，供 ``build_lnet`` / ``train_delan_loop`` 使用。

    base:
      - ``delan_model``: 以 2-DoF BAK 上验证过的偏小网络为基准，按 dof 放大宽度
      - ``example``: 官方 example_DeLaN 的 128×8（高 dof 时易欠拟合/难训，慎用）
    """
    if base == "example":
        h = {
            "n_width": 128,
            "n_depth": 8,
            "diagonal_epsilon": 0.01,
            "activation": "SoftPlus",
            "b_init": 1.0e-3,
            "b_diag_init": 0.01,
            "learning_rate": 5.0e-3,
            "weight_decay": 1.0e-5,
            "max_epoch": 10000,
        }
    else:
        h = {
            "n_width": 64,
            "n_depth": 2,
            "diagonal_epsilon": 0.01,
            "activation": "SoftPlus",
            "b_init": 1.0e-4,
            "b_diag_init": 0.001,
            "learning_rate": 5.0e-4,
            "weight_decay": 1.0e-5,
            "max_epoch": 10000,
        }

    # Cholesky 参数个数 m = n(n+1)/2；略增宽度以覆盖更大惯性矩阵
    m_chol = n_dof * (n_dof + 1) // 2
    width_scale = math.sqrt(max(m_chol, 3) / 3.0)
    h["n_width"] = int(min(256, max(h["n_width"], round(h["n_width"] * width_scale))))

    if n_dof >= 12:
        h["n_depth"] = max(h["n_depth"], 4)
    if n_dof >= 20:
        h["n_depth"] = max(h["n_depth"], 6)
        h["learning_rate"] = min(h["learning_rate"], 3.0e-4)

    # 参数量与 dH/dq 张量随 dof 增长，适当减小 batch
    h["n_minibatch"] = int(min(512, max(32, n_train_samples // 30)))
    h["w_init"] = "xavier_normal"
    h["gain_hidden"] = float(2**0.5)
    h["gain_output"] = 0.1
    return h
