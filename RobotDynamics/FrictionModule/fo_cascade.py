"""
H-Net（Xun 等分数阶摩擦图 4 的神经化实现）：TCN₁ → MLP → TCN₂。

  q^{t-L:t}  --TCN₁-->  v_seq（等效分数阶微分 / 速度型量）
            --MLP-->   s_seq（等效 Stribeck S(v)，逐时刻共享 MLP）
            --TCN₂-->  τ_fri（等效分数阶积分 / 滞回记忆）

TCN 使用因果卷积（只看过去帧），对历史做可学习加权，近似 FO 滤波器。
输入仅用位置序列 q_seq；qd_seq 保留以兼容 MystericNet 接口，默认不使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    """左侧填充的因果 Conv1d：输出时刻 t 仅依赖输入 ≤ t。"""

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
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for i in range(n_layers):
            dilation = 2**i
            layers.append(
                _CausalConv1d(ch_in, hidden_channels, kernel_size, dilation=dilation)
            )
            layers.append(nn.ReLU(inplace=True))
            ch_in = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HNetFOCascade(nn.Module):
    """
    级联摩擦网络，对齐 Xun 图 4：分数阶微分 → Stribeck → 分数阶积分。

    Hyperparameters 默认与 Yeo H-Net TCN 同量级，便于对比实验。
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

        # TCN₁：1/s^α，由 q 历史得到 v_seq
        self.tcn_diff = _CausalTCNStack(
            dof, hidden_channels, n_layers=tcn_layers, kernel_size=kernel_size
        )
        self.proj_v = nn.Conv1d(hidden_channels, dof, kernel_size=1)

        # Stribeck 模块：逐时刻 S(v)
        self.stribeck_mlp = nn.Sequential(
            nn.Linear(dof, mlp_h),
            nn.Tanh(),
            nn.Linear(mlp_h, dof),
        )

        # TCN₂：b/s^β，由 s 历史积分得到 τ_fri
        self.tcn_int = _CausalTCNStack(
            dof, hidden_channels, n_layers=tcn_layers, kernel_size=kernel_size
        )
        self.head = nn.Linear(hidden_channels, dof)

        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for m in self.stribeck_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q_seq: (B, L, dof)，时间沿 dim=1 递增，末帧为当前 t。
        qd_seq: 兼容接口，未使用。
        Returns tau_fri: (B, dof)
        """
        if q_seq.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {q_seq.shape[1]}")
        _ = qd_seq

        x = q_seq.transpose(1, 2)
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
        qd_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """调试：返回 (tau_fri, v_last, s_last, v_seq)。"""
        if q_seq.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {q_seq.shape[1]}")
        _ = qd_seq
        x = q_seq.transpose(1, 2)
        v_seq = self.proj_v(self.tcn_diff(x)).transpose(1, 2)
        s_seq = self.stribeck_mlp(v_seq)
        s_ch = s_seq.transpose(1, 2)
        tau_fri = self.head(self.tcn_int(s_ch)[:, :, -1])
        return tau_fri, v_seq[:, -1, :], s_seq[:, -1, :], v_seq
