"""
摩擦 PINN 损失（Hu 等 Eq. (6)）及 Mysteric 总损失辅助。

多关节 / 小力矩关节与总力矩一致，推荐 ``fri_loss='smape'``（同 DeLaN ``torque_loss``）。
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..DeLaN.losses import torque_loss


def friction_supervised_loss(
    tau_fri_pred: torch.Tensor,
    tau_fri_target: torch.Tensor,
    kind: str = "smape",
    *,
    smape_eps: float = 1e-3,
) -> torch.Tensor:
    """单条摩擦监督：MSE 或 SMAPE。"""
    return torque_loss(tau_fri_pred, tau_fri_target, kind, smape_eps=smape_eps)


def friction_pinn_loss(
    tau_fri_pred: torch.Tensor,
    tau_fri_target: torch.Tensor,
    tau_fri_physics: torch.Tensor,
    *,
    lambda_physics: float = 0.5,
    supervise_friction: bool = True,
    fri_loss: str = "smape",
    smape_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hu 等 Eq. (6)，数据项与物理项均可用 SMAPE::

        L = (1-λ) loss(τ_pred, τ_true) + λ loss(τ_pred, τ_physics)

    ``loss`` 为 ``mse`` 或 ``smape``（``fri_loss``）。

    ``supervise_friction=False``：无摩擦真值时仅 ``L = λ loss(τ_pred, τ_physics)``。
    """
    lam = float(lambda_physics)
    l_phys = torque_loss(
        tau_fri_pred, tau_fri_physics, fri_loss, smape_eps=smape_eps
    )
    if supervise_friction:
        l_data = torque_loss(
            tau_fri_pred, tau_fri_target, fri_loss, smape_eps=smape_eps
        )
        l_total = (1.0 - lam) * l_data + lam * l_phys
    else:
        l_data = torch.zeros((), device=tau_fri_pred.device, dtype=tau_fri_pred.dtype)
        l_total = lam * l_phys
    return l_total, l_data, l_phys
