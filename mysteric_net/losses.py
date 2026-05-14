"""
Paper Eq. (7)-(8) and energy consistency term around Eq. (5)-(6).

  l_total = l_tau + l_E

  l_tau = mean |tau_hat - tau|^2

  l_E = mean | dE_hat_rig/dt - (tau - tau_fri_hat)^T qd |^2

  dE_hat_rig/dt = d/dt ( 1/2 qd^T M_hat qd ) + g_hat^T qd
                = qd^T M qdd + 1/2 * (nabla_q (qd_det^T M qd_det)) · qd + g^T qd
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _d_kinetic_dt(lnet: nn.Module, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
    """沿轨迹的 dT/dt，标量/样本，形状 (B,)。"""
    want_grad = torch.is_grad_enabled()
    q_req = q.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        M = lnet.mass_matrix(q_req)
        term1 = torch.einsum("bi,bij,bj->b", qd, M, qdd)
        S = torch.einsum("bi,bij,bj->b", qd.detach(), M, qd.detach()).sum()
        gradS = torch.autograd.grad(S, q_req, create_graph=True)[0]
        out = term1 + 0.5 * torch.sum(gradS * qd, dim=1)
    return out if want_grad else out.detach()


def mysteric_losses(
    lnet: nn.Module,
    tau_hat: torch.Tensor,
    tau_target: torch.Tensor,
    tau_fri_hat: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    g 由 L-Net 前向得到；能量项内部对质量矩阵再建图。

    Returns (l_total, l_tau, l_E).
    """
    l_tau = torch.mean((tau_hat - tau_target) ** 2)

    dT_dt = _d_kinetic_dt(lnet, q, qd, qdd)
    dV_dt = torch.sum(g * qd, dim=1)
    dE_dt_hat = dT_dt + dV_dt
    power_residual = torch.sum((tau_target - tau_fri_hat) * qd, dim=1)
    l_E = torch.mean((dE_dt_hat - power_residual) ** 2)

    l_total = l_tau + l_E
    return l_total, l_tau, l_E
