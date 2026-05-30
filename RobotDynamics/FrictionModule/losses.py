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
    detach_physics: bool = False,
    fri_loss: str = "smape",
    smape_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hu 等 Eq. (6)，数据项与物理项均可用 SMAPE::

        L = (1-λ) loss(τ_pred, τ_true) + λ loss(τ_pred, τ_physics)

    ``loss`` 为 ``mse`` 或 ``smape``（``fri_loss``）。

    ``supervise_friction=False``：无摩擦真值时仅 ``L = λ loss(τ_pred, τ_physics)``。

    ``detach_physics=True``：物理支路不参与 ``l_phys`` 反传（SCV 作固定锚，
    仅 MLP 向物理输出对齐；无 ``τ_true`` 时 SCV 改由 ``l_tau`` 经 ``tau_hat`` 监督）。
    """
    lam = float(lambda_physics)
    tau_phys = (
        tau_fri_physics.detach()
        if detach_physics
        else tau_fri_physics
    )
    l_phys = torque_loss(
        tau_fri_pred, tau_phys, fri_loss, smape_eps=smape_eps
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


def friction_pinn_tau_blend_loss(
    tau_core: torch.Tensor,
    tau_pred: torch.Tensor,
    tau_scv: torch.Tensor,
    tau_meas: torch.Tensor,
    *,
    lambda_physics: float = 0.5,
    fri_loss: str = "smape",
    smape_eps: float = 1e-3,
    scv_supervision_loss: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    无 ``τ_fri`` 标签时的 PINN 摩擦损失（``tau_blend`` 模式）::

        l_fri = (1-λ) · l_scv_sup + λ · loss(τ_pred, τ_scv.detach())

    ``l_scv_sup = loss(τ_scv, τ_meas − τ_core.detach())``（摩擦残差，SCV 专用监督）。

    ``loss(τ_pred, τ_scv.detach())`` 仅更新 fo/MLP，**不**把 SCV 拉向接近零的 pred。

    全局 ``l_tau = loss(τ_core + τ_pred, τ_meas)`` 仍单独用于 L-Net + fo，不向 SCV 反传。

    ``scv_supervision_loss`` 默认 ``smape``（有界，避免未训练 L-Net 时 MSE 爆炸）；
    可显式传 ``mse`` 在残差已较小时使用。

    Returns:
        l_total, l_scv_sup, l_consist
    """
    lam = float(lambda_physics)
    tau_fri_target = tau_meas - tau_core.detach()
    l_scv_sup = torque_loss(
        tau_scv,
        tau_fri_target,
        scv_supervision_loss,
        smape_eps=smape_eps,
    )
    l_consist = torque_loss(
        tau_pred,
        tau_scv.detach(),
        fri_loss,
        smape_eps=smape_eps,
    )
    l_total = (1.0 - lam) * l_scv_sup + lam * l_consist
    return l_total, l_scv_sup, l_consist
