"""
Mysteric-Net 能量一致性损失（Yeo 等 Eq. (7)），与 DeLaN 能量率定义对齐。

  l_total = l_tau + l_E
  dE_rig/dt = dT/dt + dV/dt
  dV/dt = g^T qd   （g = nabla_q V，与 deep_lagrangian_networks 一致）
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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
    l_tau = torch.mean((tau_hat - tau_target) ** 2)

    if hasattr(lnet, "dynamics"):
        dyn = lnet.dynamics(q, qd, qdd)
        dE_dt_hat = dyn.dTdt + dyn.dVdt
    else:
        q_req = q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            H_hat = lnet.H_hat_from_q(q_req)
            term1 = torch.einsum("bi,bij,bj->b", qd, H_hat, qdd)
            S = torch.einsum("bi,bij,bj->b", qd.detach(), H_hat, qd.detach()).sum()
            gradS = torch.autograd.grad(S, q_req, create_graph=True)[0]
            dT_dt = term1 + 0.5 * torch.sum(gradS * qd, dim=1)
        dE_dt_hat = dT_dt + torch.sum(g * qd, dim=1)

    power_residual = torch.sum((tau_target - tau_fri_hat) * qd, dim=1)
    l_E = torch.mean((dE_dt_hat - power_residual) ** 2)
    l_total = l_tau + l_E
    return l_total, l_tau, l_E
