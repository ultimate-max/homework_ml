"""
DeLaN 训练用力矩损失（含多关节尺度不均衡时常用的 SMAPE）。
"""

from __future__ import annotations

import torch


def torque_loss_mse(tau_hat: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.mean((tau_hat - tau) ** 2)


def torque_loss_smape(
    tau_hat: torch.Tensor,
    tau: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    对称平均绝对百分比误差（与论文式 (20) 一致，对 batch 内所有关节/样本取平均）::

        mean( 2|τ - τ̂| / (|τ| + |τ̂| + eps) )
    """
    num = 2.0 * (tau_hat - tau).abs()
    den = tau.abs() + tau_hat.abs() + float(eps)
    return torch.mean(num / den)


def torque_loss(
    tau_hat: torch.Tensor,
    tau: torch.Tensor,
    kind: str = "mse",
    *,
    smape_eps: float = 1e-3,
) -> torch.Tensor:
    k = kind.lower()
    if k == "mse":
        return torque_loss_mse(tau_hat, tau)
    if k == "smape":
        return torque_loss_smape(tau_hat, tau, eps=smape_eps)
    raise ValueError(f"未知 tau_loss={kind!r}，可选 mse / smape")
