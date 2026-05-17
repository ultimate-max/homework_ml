"""
Mysteric-Net: DeLaN 刚体 (L-Net) + 摩擦子网络 (H-Net)。

  tau_hat = tau_rigid + tau_fri

摩擦后端 ``friction_backend``:
  - ``tcn``: 原论文 TCN（Yeo 等）
  - ``fo_cascade``: TCN₁→MLP→TCN₂（Xun 图 4 分数阶摩擦的神经化）
  - ``stribeck``: 可学习 SCV 物理模型（Hu 等 Eq. (4)）
  - ``stribeck_pinn``: MLP + SCV 物理约束（Hu 等 PINN, Eq. (6)）
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

from ..DeLaN.lnet import LNet
from ..FrictionModule.fo_cascade import HNetFOCascade
from ..FrictionModule.stribeck import HNetStribeck, HNetStribeckPINN
from ..FrictionModule.tcn import HNetTCN

FrictionBackend = Literal["tcn", "fo_cascade", "stribeck", "stribeck_pinn"]


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
        friction_backend: FrictionBackend = "tcn",
        stribeck_hidden: Tuple[int, ...] = (128, 64),
        stribeck_dropout: float = 0.0,
        scv_variant: Literal["scv", "cv"] = "scv",
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.friction_backend: FrictionBackend = friction_backend
        self.lnet = LNet(
            dof,
            hidden_dim=lnet_hidden,
            num_hidden_layers=lnet_layers,
            b_diagonal=mass_diag_eps,
            numerical_H_ridge=lnet_numerical_H_ridge,
        )
        if friction_backend == "tcn":
            self.hnet = HNetTCN(
                dof, seq_len=seq_len, hidden_channels=hnet_channels, kernel_size=hnet_kernel
            )
        elif friction_backend == "fo_cascade":
            self.hnet = HNetFOCascade(
                dof,
                seq_len=seq_len,
                hidden_channels=hnet_channels,
                kernel_size=hnet_kernel,
            )
        elif friction_backend == "stribeck":
            self.hnet = HNetStribeck(dof, model=scv_variant)
        elif friction_backend == "stribeck_pinn":
            self.hnet = HNetStribeckPINN(
                dof,
                seq_len=seq_len,
                hidden=stribeck_hidden,
                dropout=stribeck_dropout,
            )
        else:
            raise ValueError(f"未知 friction_backend={friction_backend!r}")

    def forward(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            tau_hat, tau_core, tau_fri, H_hat, g_hat, tau_fri_physics
            ``tau_fri_physics`` 仅在 ``stribeck_pinn`` 时为 SCV 输出，否则为 ``None``。
        """
        tau_core, H_hat, g_hat = self.lnet(q, qd, qdd)
        tau_fri_physics: torch.Tensor | None = None

        if self.friction_backend in ("tcn", "fo_cascade"):
            tau_fri = self.hnet(q_seq, qd_seq)
        elif self.friction_backend == "stribeck":
            tau_fri, _ = self.hnet(q_seq, qd_seq)
        else:
            tau_fri, tau_fri_physics = self.hnet(q_seq, qd_seq)

        tau_hat = tau_core + tau_fri
        return tau_hat, tau_core, tau_fri, H_hat, g_hat, tau_fri_physics
