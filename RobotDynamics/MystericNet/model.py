"""
Mysteric-Net: DeLaN 刚体 (L-Net) + 摩擦子网络 (H-Net)。

  tau_hat = tau_rigid + tau_fri

  PINN 后端可通过 ``pinn_friction_output='physics'`` 令 ``tau_hat`` 使用 SCV/GMS
  物理输出（无 ``τ_fri`` 标签时推荐），``tau_fri`` 返回值仍为 MLP/fo 预测供 ``l_fri``。

摩擦后端 ``friction_backend``:
  - ``tcn``: 原论文 TCN（Yeo 等）
  - ``fo_cascade``: TCN₁→两层 tanh MLP→TCN₂（Xun 图 4 简化）
  - ``fo_cascade_pinn``: fo_cascade + SCV 物理约束（Hu 等 PINN, Eq. (6)）
  - ``stribeck``: 可学习 SCV 物理模型（Hu 等 Eq. (4)）
  - ``stribeck_pinn``: MLP + SCV 物理约束（Hu 等 PINN, Eq. (6)）
  - ``gms``: 可学习 GMS 物理模型（并联 Maxwell stick/slip + 粘性）
  - ``gms_pinn``: MLP + GMS 物理约束
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

from ..DeLaN.lnet import LNet
from ..FrictionModule.fo_cascade import HNetFOCascade, HNetFOCascadePINN
from ..FrictionModule.gms import HNetGMS, HNetGMSPINN
from ..FrictionModule.stribeck import HNetStribeck, HNetStribeckPINN
from ..FrictionModule.tcn import HNetTCN

FrictionBackend = Literal[
    "tcn",
    "fo_cascade",
    "fo_cascade_pinn",
    "stribeck",
    "stribeck_pinn",
    "gms",
    "gms_pinn",
]

PINN_FRICTION_BACKENDS = frozenset({"stribeck_pinn", "fo_cascade_pinn", "gms_pinn"})
PinnFrictionOutput = Literal["pred", "physics"]


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
        gms_n_blocks: int | None = None,
        gms_n_elements: int = 3,
        gms_dt: float = 0.001,
        fo_mlp_hidden_dim: int | None = None,
        fo_tcn_layers: int | None = None,
        lnet_zero_cg: bool = False,
        pinn_friction_output: PinnFrictionOutput = "pred",
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.friction_backend: FrictionBackend = friction_backend
        if (
            pinn_friction_output == "physics"
            and friction_backend not in PINN_FRICTION_BACKENDS
        ):
            raise ValueError(
                f"pinn_friction_output='physics' 仅适用于 PINN 后端，"
                f"当前 friction_backend={friction_backend!r}"
            )
        self.pinn_friction_output: PinnFrictionOutput = pinn_friction_output
        gms_blocks = gms_n_blocks if gms_n_blocks is not None else gms_n_elements
        self.gms_n_blocks = gms_blocks
        self.lnet = LNet(
            dof,
            hidden_dim=lnet_hidden,
            num_hidden_layers=lnet_layers,
            b_diagonal=mass_diag_eps,
            numerical_H_ridge=lnet_numerical_H_ridge,
            zero_coriolis_gravity=lnet_zero_cg,
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
                tcn_layers=fo_tcn_layers if fo_tcn_layers is not None else 2,
                mlp_hidden=fo_mlp_hidden_dim,
            )
        elif friction_backend == "fo_cascade_pinn":
            self.hnet = HNetFOCascadePINN(
                dof,
                seq_len=seq_len,
                hidden_channels=hnet_channels,
                kernel_size=hnet_kernel,
                tcn_layers=fo_tcn_layers if fo_tcn_layers is not None else 3,
                mlp_hidden=fo_mlp_hidden_dim,
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
        elif friction_backend == "gms":
            self.hnet = HNetGMS(
                dof,
                seq_len=seq_len,
                n_blocks=gms_blocks,
                dt=gms_dt,
            )
        elif friction_backend == "gms_pinn":
            self.hnet = HNetGMSPINN(
                dof,
                seq_len=seq_len,
                hidden=stribeck_hidden,
                dropout=stribeck_dropout,
                n_blocks=gms_blocks,
                dt=gms_dt,
            )
        else:
            raise ValueError(f"未知 friction_backend={friction_backend!r}")

    def friction_in_total_torque(
        self,
        tau_fri: torch.Tensor,
        tau_fri_physics: torch.Tensor | None,
    ) -> torch.Tensor:
        """``tau_hat`` 中实际使用的摩擦项（PINN 无标签时可为 SCV/GMS 物理输出）。"""
        if (
            self.pinn_friction_output == "physics"
            and tau_fri_physics is not None
        ):
            return tau_fri_physics
        return tau_fri

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
            ``tau_fri_physics`` 在 ``stribeck_pinn`` / ``fo_cascade_pinn`` / ``gms_pinn`` 时为物理摩擦输出，否则为 ``None``。
        """
        tau_core, H_hat, g_hat = self.lnet(q, qd, qdd)
        tau_fri_physics: torch.Tensor | None = None

        if self.friction_backend in ("tcn", "fo_cascade"):
            tau_fri = self.hnet(q_seq, qd_seq)
        elif self.friction_backend == "stribeck":
            tau_fri, _ = self.hnet(q_seq, qd_seq)
        elif self.friction_backend == "gms":
            tau_fri, _ = self.hnet(q_seq, qd_seq)
        elif self.friction_backend in PINN_FRICTION_BACKENDS:
            tau_fri, tau_fri_physics = self.hnet(q_seq, qd_seq)
        else:
            raise RuntimeError(f"未处理的 friction_backend={self.friction_backend!r}")

        tau_for_hat = self.friction_in_total_torque(tau_fri, tau_fri_physics)
        tau_hat = tau_core + tau_for_hat
        return tau_hat, tau_core, tau_fri, H_hat, g_hat, tau_fri_physics
