"""
Mysteric-Net (Yeo 等, Eq. (4)): 刚体部分用 DeLaN 式 (6) 的 L-Net + 摩擦 H-Net。

  tau_hat = tau_rigid + tau_fri
  tau_rigid = H_hat(q) q_ddot + B_c + g_hat(q)   （与 Lutter 等 2019 式 (6) 一致）
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .hnet import HNetTCN
from .lnet import LNet


class MystericNet(nn.Module):
    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        lnet_hidden: int = 32,
        lnet_layers: int = 2,
        hnet_channels: int = 8,
        hnet_kernel: int = 3,
        mass_diag_eps: float = 1.0e-2,
        *,
        lnet_numerical_H_ridge: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.lnet = LNet(
            dof,
            hidden_dim=lnet_hidden,
            num_hidden_layers=lnet_layers,
            b_diagonal=mass_diag_eps,
            numerical_H_ridge=lnet_numerical_H_ridge,
        )
        self.hnet = HNetTCN(dof, seq_len=seq_len, hidden_channels=hnet_channels, kernel_size=hnet_kernel)

    def forward(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q, qd, qdd: 关节空间状态 (B, dof)，供 L-Net（DeLaN 逆模型）。
        q_seq, qd_seq: (B, L, dof)，供 H-Net（论文式 (8)）。

        Returns:
            tau_hat:   (B, dof)
            tau_core:  (B, dof) 刚体逆动力项（DeLaN 式 (6)）
            tau_fri:   (B, dof)
            H_hat:     (B, dof, dof) 惯性矩阵（论文记号 H；部分文献记作 M）
            g_hat:     (B, dof) 论文第三头 g_hat(q; psi)
        """
        tau_core, H_hat, g_hat = self.lnet(q, qd, qdd)
        tau_fri = self.hnet(q_seq, qd_seq)
        tau_hat = tau_core + tau_fri
        return tau_hat, tau_core, tau_fri, H_hat, g_hat
