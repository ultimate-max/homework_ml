"""
Mysteric-Net (paper Eq. (4)): total torque = rigid-body (L-Net) + hysteretic friction (H-Net).

  tau_hat = M qdd + C qd + g + tau_fri
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
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.lnet = LNet(dof, hidden_dim=lnet_hidden, num_hidden_layers=lnet_layers, mass_diag_eps=mass_diag_eps)
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
        q, qd, qdd: instantaneous rigid-body states (B, dof), used by L-Net.
        q_seq, qd_seq: (B, L, dof) history for H-Net (typically desired trajectory as in paper).

        Returns:
            tau_hat: (B, dof)
            tau_core: (B, dof) rigid-body part
            tau_fri: (B, dof)
            M: (B, dof, dof) mass matrix from L-Net (for energy loss)
            g: (B, dof) gravity term from L-Net (for energy loss)
        """
        tau_core, M, g = self.lnet(q, qd, qdd)
        tau_fri = self.hnet(q_seq, qd_seq)
        tau_hat = tau_core + tau_fri
        return tau_hat, tau_core, tau_fri, M, g
