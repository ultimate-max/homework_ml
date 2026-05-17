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
    supervise_friction: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = (1-λ) mean((τ_pred - τ_true)²) + λ mean((τ_pred - τ_physics)²)

    ``supervise_friction=False``：无摩擦真值时仅 ``L = λ mean((τ_pred - τ_physics)²)``。
    """
    lam = float(lambda_physics)
    l_phys = torch.mean((tau_fri_pred - tau_fri_physics) ** 2)
    if supervise_friction:
        l_data = torch.mean((tau_fri_pred - tau_fri_target) ** 2)
        l_total = (1.0 - lam) * l_data + lam * l_phys
    else:
        l_data = torch.zeros((), device=tau_fri_pred.device, dtype=tau_fri_pred.dtype)
        l_total = lam * l_phys
    return l_total, l_data, l_phys
