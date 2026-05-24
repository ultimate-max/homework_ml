"""
Generalized Maxwell-Slip (GMS) 摩擦模型及 PINN 摩擦网络。

并联 N 个 GMS 块（stick/slip 单元）+ 粘性项::

    τ_F = Σ_i F_i + σ₁ v     (i = 1 … N_blocks)

极限面（Stribeck 回归器，见论文 Eq.(15)(18)）::

    λ = exp(-|v/V_s|^δ)
    |s(v)| = λ f_s + (1-λ) f_c
    s(v) = sgn(v) · |s(v)|

Stick:  |F_i| < |s(v)|  →  F_i ← F_i + k_i v Δt
Slip:   |F_i| ≥ |s(v)|  →  F_i ← s(v) - (s(v)-F_i) exp(-C_i Δt)   （解析积分）
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .stribeck import _positive


def stribeck_lambda(
    qd: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """λ = exp(-|v/v_s|^δ)，与 Stribeck 论文 Eq.(16) 一致。"""
    v_s_b = v_s.clamp_min(1e-6)
    while v_s_b.ndim < qd.ndim:
        v_s_b = v_s_b.unsqueeze(0)
        delta = delta.unsqueeze(0)
    ratio = torch.abs(qd / v_s_b)
    return torch.exp(-torch.pow(ratio, delta))


def gms_stribeck_regressor(
    qd: torch.Tensor,
    f_s: torch.Tensor,
    f_c: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    *,
    sign_smooth: float = 50.0,
) -> torch.Tensor:
    """
    极限面 s(v) = sgn(v) · [λ, 1-λ] · [f_s, f_c]^T  （论文 W_f K_f 两分量形式）。

    f_s, f_c, v_s, delta: (dof,) 广播至 qd 任意 shape。
    """
    lam = stribeck_lambda(qd, v_s, delta)
    while f_s.ndim < qd.ndim:
        f_s = f_s.unsqueeze(0)
        f_c = f_c.unsqueeze(0)
    mag = lam * f_s + (1.0 - lam) * f_c
    sign_v = torch.tanh(sign_smooth * qd)
    return sign_v * mag


def gms_limit_surface(
    qd: torch.Tensor,
    v_a: torch.Tensor,
    k_str: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    *,
    sign_smooth: float = 50.0,
) -> torch.Tensor:
    """s(v)，f_c=v_a，f_s=v_a+k_str（与旧参数化等价）。"""
    f_c = v_a
    f_s = v_a + k_str
    return gms_stribeck_regressor(
        qd, f_s, f_c, v_s, delta, sign_smooth=sign_smooth
    )


def _gms_limit_surface_jit(
    qd_seq: torch.Tensor,
    v_a: torch.Tensor,
    k_str: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """JIT 友好：整段 (B,L,dof) Stribeck 回归器 s(v)。"""
    vs = v_s.clamp(min=1e-6).view(1, 1, -1)
    da = delta.view(1, 1, -1)
    va = v_a.view(1, 1, -1)
    ks = k_str.view(1, 1, -1)
    lam = torch.exp(-torch.abs(qd_seq / vs) ** da)
    mag = lam * (va + ks) + (1.0 - lam) * va
    return torch.tanh(50.0 * qd_seq) * mag


@torch.jit.script
def _gms_integrate_jit(
    qd_seq: torch.Tensor,
    k_i: torch.Tensor,
    c_i: torch.Tensor,
    v_a: torch.Tensor,
    k_str: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    sigma_1: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """TorchScript：预计算 s(v) 序列 + stick Euler + slip 解析更新。"""
    b = qd_seq.size(0)
    length = qd_seq.size(1)
    dof = qd_seq.size(2)
    n_elem = k_i.size(1)

    vs = v_s.clamp(min=1e-6).view(1, 1, -1)
    da = delta.view(1, 1, -1)
    va = v_a.view(1, 1, -1)
    ks = k_str.view(1, 1, -1)
    lam = torch.exp(-torch.abs(qd_seq / vs) ** da)
    s_v_seq = torch.tanh(50.0 * qd_seq) * (lam * (va + ks) + (1.0 - lam) * va)
    s_abs_seq = s_v_seq.abs().clamp(min=1e-6)

    f_state = torch.zeros(b, dof, n_elem, device=qd_seq.device, dtype=qd_seq.dtype)
    exp_c = torch.exp(-c_i.unsqueeze(0) * dt)
    k_b = k_i.unsqueeze(0)

    for t in range(length):
        v = qd_seq[:, t, :]
        s_v = s_v_seq[:, t, :]
        s_abs = s_abs_seq[:, t, :]
        s_v_e = s_v.unsqueeze(-1)
        f_stick = f_state + k_b * v.unsqueeze(-1) * dt
        f_slip = s_v_e - (s_v_e - f_state) * exp_c
        stick = f_state.abs() < s_abs.unsqueeze(-1)
        f_state = torch.where(stick, f_stick, f_slip)

    return f_state.sum(dim=-1) + sigma_1.unsqueeze(0) * qd_seq[:, -1, :]


def _gms_integrate_core(
    qd_seq: torch.Tensor,
    k_i: torch.Tensor,
    c_i: torch.Tensor,
    v_a: torch.Tensor,
    k_str: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    sigma_1: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Eager 路径：与 JIT 等价，便于调试。"""
    s_v_seq = _gms_limit_surface_jit(qd_seq, v_a, k_str, v_s, delta)
    s_abs_seq = s_v_seq.abs().clamp_min(1e-6)
    b, length, dof = qd_seq.shape
    n_elem = k_i.shape[1]

    f_state = qd_seq.new_zeros(b, dof, n_elem)
    dt_t = qd_seq.new_tensor(float(dt))
    exp_c = torch.exp(-c_i.unsqueeze(0) * dt_t)
    k_b = k_i.unsqueeze(0)

    for t in range(length):
        v = qd_seq[:, t, :]
        s_v = s_v_seq[:, t, :]
        s_abs = s_abs_seq[:, t, :]
        s_v_e = s_v.unsqueeze(-1)
        f_stick = f_state + k_b * v.unsqueeze(-1) * dt_t
        f_slip = s_v_e - (s_v_e - f_state) * exp_c
        stick = f_state.abs() < s_abs.unsqueeze(-1)
        f_state = torch.where(stick, f_stick, f_slip)

    return f_state.sum(dim=-1) + sigma_1.unsqueeze(0) * qd_seq[:, -1, :]


def gms_integrate(
    qd_seq: torch.Tensor,
    *,
    k_i: torch.Tensor,
    c_i: torch.Tensor,
    v_a: torch.Tensor,
    k_str: torch.Tensor,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    sigma_1: torch.Tensor,
    dt: float,
    use_jit: bool = True,
) -> torch.Tensor:
    """
    对速度序列积分，返回最后一帧摩擦 τ (B, dof)。

    k_i, c_i: (dof, n_elem)；其余 GMS 包络参数: (dof,)。
  use_jit: 默认 True，TorchScript 加速（训练/推理均可）。
    """
    if qd_seq.ndim != 3:
        raise ValueError(f"qd_seq 须为 (B, L, dof)，got {qd_seq.shape}")
    dof = qd_seq.shape[2]
    if k_i.shape[0] != dof or c_i.shape != k_i.shape:
        raise ValueError(f"k_i/c_i 须为 (dof, n_elem)，got {k_i.shape}, {c_i.shape}")

    fn = _gms_integrate_jit if use_jit else _gms_integrate_core
    return fn(
        qd_seq, k_i, c_i, v_a, k_str, v_s, delta, sigma_1, float(dt)
    )


class GmsParams(nn.Module):
    """每关节 ``n_blocks`` 个并联 GMS 块 + Stribeck 型极限面 + 粘性 σ₁。"""

    def __init__(
        self,
        dof: int,
        *,
        n_blocks: int = 3,
        init_v_s: float = 0.05,
        init_delta: float = 1.5,
        use_jit: bool = True,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"gms n_blocks 须 ≥ 1，got {n_blocks}")
        self.dof = dof
        self.n_blocks = n_blocks
        self.n_elements = n_blocks  # 兼容旧名
        self.use_jit = use_jit

        self.log_v_a = nn.Parameter(torch.full((dof,), math.log(0.1)))
        self.log_k_str = nn.Parameter(torch.full((dof,), math.log(0.05)))
        self.log_v_s = nn.Parameter(torch.full((dof,), math.log(init_v_s)))
        self.log_delta = nn.Parameter(torch.full((dof,), math.log(init_delta)))
        self.log_sigma_1 = nn.Parameter(torch.full((dof,), math.log(0.01)))

        k_init = torch.logspace(-1, 1, n_blocks)
        c_init = torch.full((n_blocks,), 2.0)
        self.log_k_i = nn.Parameter(k_init.log().unsqueeze(0).expand(dof, -1).clone())
        self.log_c_i = nn.Parameter(c_init.log().unsqueeze(0).expand(dof, -1).clone())

    def envelope(self) -> tuple[torch.Tensor, ...]:
        return (
            _positive(self.log_v_a),
            _positive(self.log_k_str),
            _positive(self.log_v_s),
            _positive(self.log_delta, floor=0.5),
            _positive(self.log_sigma_1),
        )

    def forward(self, qd_seq: torch.Tensor, *, dt: float) -> torch.Tensor:
        v_a, k_str, v_s, delta, sigma_1 = self.envelope()
        return gms_integrate(
            qd_seq,
            k_i=_positive(self.log_k_i),
            c_i=_positive(self.log_c_i),
            v_a=v_a,
            k_str=k_str,
            v_s=v_s,
            delta=delta,
            sigma_1=sigma_1,
            dt=dt,
            use_jit=self.use_jit,
        )


def warmstart_gms_from_samples(
    gms: GmsParams,
    qd: torch.Tensor,
    tau_fri: torch.Tensor,
    *,
    qd_min: float = 0.02,
    v_a_floor: float = 0.05,
) -> None:
    """用 median(|τ_fri|)（|q̇|>qd_min）粗估极限面 f_c (=v_a)。"""
    if qd.shape != tau_fri.shape or qd.ndim != 2:
        raise ValueError(f"qd {qd.shape} 与 tau_fri {tau_fri.shape} 须同为 (N, dof)")
    dof = gms.dof
    if qd.shape[1] != dof:
        raise ValueError(f"Expected dof {dof}, got qd {qd.shape[1]}")
    with torch.no_grad():
        for j in range(dof):
            mask = qd[:, j].abs() > qd_min
            if int(mask.sum()) < 8:
                continue
            est = tau_fri[mask, j].abs().median().clamp(min=v_a_floor)
            gms.log_v_a[j] = math.log(float(est))


class HNetGMS(nn.Module):
    """纯物理 GMS 摩擦网络：对 qd_seq 滑窗逐步积分，输出最后一帧 τ_fri。"""

    def __init__(
        self,
        dof: int,
        *,
        seq_len: int = 30,
        n_blocks: int = 3,
        dt: float = 0.001,
        use_jit: bool = True,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.dt = float(dt)
        self.gms = GmsParams(dof, n_blocks=n_blocks, use_jit=use_jit)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, length, n = qd_seq.shape
        if n != self.dof:
            raise ValueError(f"Expected dof {self.dof}, got {n}")
        if length != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {length}")
        tau = self.gms(qd_seq, dt=self.dt)
        return tau, tau


class HNetGMSPINN(nn.Module):
    """
    GMS-PINN：MLP(位置/速度历史) → τ_pred，GMS 积分 → τ_physics；
    训练时用 L = (1-λ) loss(τ_pred, τ_target) + λ loss(τ_pred, τ_physics)。
    """

    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
        *,
        n_blocks: int = 3,
        dt: float = 0.001,
        use_jit: bool = True,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        in_dim = seq_len * dof * 2
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, dof))
        self.mlp = nn.Sequential(*layers)
        self.gms = GmsParams(dof, n_blocks=n_blocks, use_jit=use_jit)
        self.dt = float(dt)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, length, n = qd_seq.shape
        if length != self.seq_len or n != self.dof:
            raise ValueError(f"Expected qd_seq (B, {self.seq_len}, {self.dof}), got {qd_seq.shape}")
        x = torch.cat([q_seq.reshape(b, -1), qd_seq.reshape(b, -1)], dim=-1)
        tau_pred = self.mlp(x)
        tau_physics = self.gms(qd_seq, dt=self.dt)
        return tau_pred, tau_physics
