"""
摩擦 PINN 损失（Hu 等 Eq. (6)）及 Mysteric 总损失辅助。
"""

from __future__ import annotations

from typing import Tuple

import torch


def friction_pinn_loss(
    tau_fri_pred: torch.Tensor,
    tau_fri_target: torch.Tensor,
    tau_fri_physics: torch.Tensor,
    *,
    lambda_physics: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = (1-λ) mean((τ_pred - τ_true)²) + λ mean((τ_pred - τ_physics)²)
    """
    lam = float(lambda_physics)
    l_data = torch.mean((tau_fri_pred - tau_fri_target) ** 2)
    l_phys = torch.mean((tau_fri_pred - tau_fri_physics) ** 2)
    l_total = (1.0 - lam) * l_data + lam * l_phys
    return l_total, l_data, l_phys
