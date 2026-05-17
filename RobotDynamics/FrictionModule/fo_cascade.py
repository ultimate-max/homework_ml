"""
H-Net（Xun 等分数阶摩擦图 4 的神经化实现）：TCN₁ → MLP → 1/s → TCN₂。

  [q, q̇]^{t-L:t}  --TCN₁-->  v_seq（等效分数阶微分）
                 --MLP-->   s_raw（Stribeck 非线性）
                 --1/s-->  s_seq（因果积分低通，抑制 MLP 高频）
                 --TCN₂-->  τ_fri（滞回记忆）

TCN₁ 输入为位置与速度拼接（与 Yeo H-Net TCN 一致）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stribeck import StribeckSCVParams


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
            layers.append(nn.ReLU(inplace=False))
            ch_in = hidden_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CausalIntegrator1s(nn.Module):
    """
    因果离散 ``1/s``（后向欧拉 + 泄漏，防漂移）::

        y[t] = leak * y[t-1] + α * x[t]

    ``α`` 为每关节可学习步长；``leak``∈(0,1] 保持因果低通，滤除 MLP 高频分量。
    """

    def __init__(
        self,
        dof: int,
        *,
        init_alpha: float = 0.2,
        init_leak: float = 0.98,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.log_alpha = nn.Parameter(torch.full((dof,), math.log(init_alpha)))
        self.logit_leak = nn.Parameter(
            torch.full((dof,), math.log(init_leak / (1.0 - init_leak)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dof) → y: (B, L, dof)"""
        alpha = F.softplus(self.log_alpha) + 1e-6
        leak = torch.sigmoid(self.logit_leak).clamp(0.5, 0.999)
        state = alpha * x[:, 0, :]
        steps: list[torch.Tensor] = [state]
        for t in range(1, x.shape[1]):
            state = leak * state + alpha * x[:, t, :]
            steps.append(state)
        return torch.stack(steps, dim=1)


def _stack_q_qd(q_seq: torch.Tensor, qd_seq: torch.Tensor) -> torch.Tensor:
    """(B, L, dof)×2 → (B, 2*dof, L)，供 Conv1d 使用。"""
    if q_seq.shape != qd_seq.shape:
        raise ValueError(f"q_seq {q_seq.shape} 与 qd_seq {qd_seq.shape} 不一致")
    return torch.cat([q_seq, qd_seq], dim=-1).transpose(1, 2)


class HNetFOCascade(nn.Module):
    """
    级联摩擦网络，对齐 Xun 图 4：微分 → Stribeck → ``1/s`` 低通 → 记忆积分。

    TCN₁ 在 ``[q, q̇]`` 历史上做因果卷积；MLP 后接因果积分器 ``integrate_1s``。
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

        # TCN₁：在 [q, q̇] 历史上得到 v_seq
        self.tcn_diff = _CausalTCNStack(
            2 * dof, hidden_channels, n_layers=tcn_layers, kernel_size=kernel_size
        )
        self.proj_v = nn.Conv1d(hidden_channels, dof, kernel_size=1)

        self.stribeck_mlp = nn.Sequential(
            nn.Linear(dof, mlp_h),
            nn.Tanh(),
            nn.Linear(mlp_h, dof),
        )
        self.integrate_1s = _CausalIntegrator1s(dof)

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

        s_raw = self.stribeck_mlp(v_seq)
        s_seq = self.integrate_1s(s_raw)

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
        s_raw = self.stribeck_mlp(v_seq)
        s_seq = self.integrate_1s(s_raw)
        s_ch = s_seq.transpose(1, 2)
        tau_fri = self.head(self.tcn_int(s_ch)[:, :, -1])
        return (
            tau_fri,
            v_seq[:, -1, :],
            s_seq[:, -1, :],
            s_raw[:, -1, :],
            v_seq,
        )


class HNetFOCascadePINN(nn.Module):
    """
    fo_cascade + SCV 物理支路（Hu 等 PINN Eq. (6)）。

    - ``fo``：TCN₁([q,q̇])→MLP→1/s→TCN₂，输出 τ_pred（含记忆/滞回）
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
