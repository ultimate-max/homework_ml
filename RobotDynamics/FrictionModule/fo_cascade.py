"""
H-Net（Xun 等分数阶摩擦图 4 的神经化实现）：TCN₁ → MLP → TCN₂。

  [q, q̇]^{t-L:t}  --TCN₁-->  v_seq（等效分数阶微分）
                 --MLP-->    s_seq（两层 tanh MLP，逐时刻权重共享）
                 --TCN₂-->  τ_fri（线性因果积分 / 滞回记忆，无激活）

TCN₁ 输入为位置与速度拼接（与 Yeo H-Net TCN 一致）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stribeck import StribeckSCVParams


class _CausalConv1d(nn.Module):
    """左侧填充的 causal Conv1d：输出时刻 t 仅依赖输入 ≤ t。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class _CausalTCNStack(nn.Module):
    """多层因果 TCN；dilation 逐层翻倍以扩大感受野。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        n_layers: int = 2,
        kernel_size: int = 3,
        use_activation: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for i in range(n_layers):
            dilation = 2**i
            layers.append(
                _CausalConv1d(ch_in, hidden_channels, kernel_size, dilation=dilation)
            )
            if use_activation:
                layers.append(nn.ReLU(inplace=False))
            ch_in = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StribeckMLP(nn.Module):
    """
    逐时刻两层 MLP（对 ``v_seq`` 每个时间步独立、权重共享）::

        Linear(dof → hidden) → tanh → Linear(hidden → dof)

    输出层为线性，力矩尺度由后续 TCN₂ / head 承担。
    """

    def __init__(self, dof: int, hidden_dim: int) -> None:
        super().__init__()
        self.dof = dof
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dof, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dof),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_stribeck_mlp(dof: int, hidden_dim: int) -> StribeckMLP:
    return StribeckMLP(dof, hidden_dim)


def _stack_q_qd(q_seq: torch.Tensor, qd_seq: torch.Tensor) -> torch.Tensor:
    """(B, L, dof)×2 → (B, 2*dof, L)，供 Conv1d 使用。"""
    if q_seq.shape != qd_seq.shape:
        raise ValueError(f"q_seq {q_seq.shape} 与 qd_seq {qd_seq.shape} 不一致")
    return torch.cat([q_seq, qd_seq], dim=-1).transpose(1, 2)


class HNetFOCascade(nn.Module):
    """
    级联摩擦网络：TCN₁([q,q̇]) → 两层 tanh MLP → TCN₂ → τ_fri。
    """

    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        hidden_channels: int = 8,
        kernel_size: int = 3,
        *,
        tcn_layers: int = 2,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        mlp_h = mlp_hidden if mlp_hidden is not None else max(4 * dof, 16)
        self.mlp_hidden = mlp_h

        self.tcn_diff = _CausalTCNStack(
            2 * dof,
            hidden_channels,
            n_layers=tcn_layers,
            kernel_size=kernel_size,
            use_activation=False,
        )
        self.proj_v = nn.Conv1d(hidden_channels, dof, kernel_size=1)

        self.stribeck_mlp = _build_stribeck_mlp(dof, mlp_h)

        self.tcn_int = _CausalTCNStack(
            dof,
            hidden_channels,
            n_layers=tcn_layers,
            kernel_size=kernel_size,
            use_activation=False,
        )
        self.head = nn.Linear(hidden_channels, dof)

        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_seq, qd_seq: (B, L, dof)，时间沿 dim=1 递增，末帧为当前 t。
        Returns tau_fri: (B, dof)
        """
        if q_seq.shape[1] != self.seq_len or qd_seq.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, "
                f"got q {q_seq.shape[1]}, qd {qd_seq.shape[1]}"
            )
        if qd_seq.shape[2] != self.dof:
            raise ValueError(f"Expected dof {self.dof}, got qd {qd_seq.shape}")

        x = _stack_q_qd(q_seq, qd_seq)
        h_v = self.tcn_diff(x)
        v_seq = self.proj_v(h_v).transpose(1, 2)

        s_seq = self.stribeck_mlp(v_seq)

        s_ch = s_seq.transpose(1, 2)
        h_f = self.tcn_int(s_ch)
        tau_fri = self.head(h_f[:, :, -1])
        return tau_fri

    def forward_with_internals(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """调试：返回 (tau_fri, v_last, s_last, s_raw_last, v_seq)。"""
        if q_seq.shape[1] != self.seq_len or qd_seq.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}")
        x = _stack_q_qd(q_seq, qd_seq)
        v_seq = self.proj_v(self.tcn_diff(x)).transpose(1, 2)
        s_seq = self.stribeck_mlp(v_seq)
        s_ch = s_seq.transpose(1, 2)
        tau_fri = self.head(self.tcn_int(s_ch)[:, :, -1])
        return (
            tau_fri,
            v_seq[:, -1, :],
            s_seq[:, -1, :],
            s_seq[:, -1, :],
            v_seq,
        )


class HNetFOCascadePINN(nn.Module):
    """
    fo_cascade + SCV 物理支路（Hu 等 PINN Eq. (6)）。

    - ``fo``：TCN₁([q,q̇])→MLP→TCN₂，输出 τ_pred
    - ``scv``：SCV(q̇_t)，输出 τ_physics（瞬时 Stribeck 形状）
    """

    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        hidden_channels: int = 8,
        kernel_size: int = 3,
        *,
        tcn_layers: int = 2,
        mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.fo = HNetFOCascade(
            dof,
            seq_len=seq_len,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            tcn_layers=tcn_layers,
            mlp_hidden=mlp_hidden,
        )
        self.scv = StribeckSCVParams(dof)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if qd_seq is None:
            raise ValueError("fo_cascade_pinn 需要 qd_seq")
        if qd_seq.shape[1] != self.seq_len or qd_seq.shape[2] != self.dof:
            raise ValueError(
                f"Expected qd_seq (B, {self.seq_len}, {self.dof}), got {qd_seq.shape}"
            )
        tau_pred = self.fo(q_seq, qd_seq)
        tau_physics = self.scv(qd_seq[:, -1, :])
        return tau_pred, tau_physics
