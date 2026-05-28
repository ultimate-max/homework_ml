"""
Stribeck-Coulomb-Viscous (SCV) 摩擦模型（Hu 等, Eq. (3)(4)）及 PINN 摩擦网络。

CV:  τ_F = k_c tanh(k_a q̇) + k_v q̇
SCV: τ_F = k_v q̇ + k_c tanh(k_a q̇) + (k_s - k_c) exp(-|q̇/v_s|^α) tanh(k_a q̇)
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import torch.nn as nn


def _positive(param: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.softplus(param) + floor


class StribeckSCVParams(nn.Module):
    """每关节可学习的 SCV 参数（正值）。"""

    def __init__(self, dof: int, *, init_v_s: float = 0.05, init_alpha: float = 1.5) -> None:
        super().__init__()
        self.dof = dof
        self.log_k_v = nn.Parameter(torch.full((dof,), math.log(10)))
        self.log_k_c = nn.Parameter(torch.full((dof,), math.log(10)))
        self.log_k_a = nn.Parameter(torch.full((dof,), math.log(100.0)))
        self.log_k_s = nn.Parameter(torch.full((dof,), math.log(0.15)))
        self.log_v_s = nn.Parameter(torch.full((dof,), math.log(init_v_s)))
        self.log_alpha = nn.Parameter(torch.full((dof,), math.log(init_alpha)))

    def forward(self, qd: torch.Tensor) -> torch.Tensor:
        """qd: (B, dof) → τ_physics: (B, dof)"""
        return scv_torque(
            qd,
            _positive(self.log_k_v),
            _positive(self.log_k_c),
            _positive(self.log_k_a),
            _positive(self.log_k_s),
            _positive(self.log_v_s),
            _positive(self.log_alpha, floor=0.5),
        )


def warmstart_scv_from_samples(
    scv: StribeckSCVParams,
    qd: torch.Tensor,
    tau_fri: torch.Tensor,
    *,
    qd_min: float = 0.02,
    k_c_floor: float = 0.05,
) -> None:
    """用 ``median(|τ_fri|)``（``|q̇|>qd_min``）粗估 ``k_c,k_s``，避免 SCV 初值过小。"""
    if qd.shape != tau_fri.shape or qd.ndim != 2:
        raise ValueError(f"qd {qd.shape} 与 tau_fri {tau_fri.shape} 须同为 (N, dof)")
    dof = scv.dof
    if qd.shape[1] != dof:
        raise ValueError(f"Expected dof {dof}, got qd {qd.shape[1]}")
    with torch.no_grad():
        for j in range(dof):
            mask = qd[:, j].abs() > qd_min
            if int(mask.sum()) < 8:
                continue
            est = tau_fri[mask, j].abs().median().clamp(min=k_c_floor)
            scv.log_k_c[j] = math.log(float(est))
            scv.log_k_s[j] = math.log(float(est) * 1.05)


def cv_torque(
    qd: torch.Tensor,
    k_v: torch.Tensor,
    k_c: torch.Tensor,
    k_a: torch.Tensor,
) -> torch.Tensor:
    """Coulomb-viscous，符号约定与 (tau - m - c - g) 一致。"""
    return k_v * qd + k_c * torch.tanh(k_a * qd)


def scv_torque(
    qd: torch.Tensor,
    k_v: torch.Tensor,
    k_c: torch.Tensor,
    k_a: torch.Tensor,
    k_s: torch.Tensor,
    v_s: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Stribeck-Coulomb-viscous，符号约定与 (tau - m - c - g) 一致。"""
    coulomb = k_c * torch.tanh(k_a * qd)
    viscous = k_v * qd
    ratio = torch.abs(qd / v_s.unsqueeze(0).clamp_min(1e-8))
    stribeck = (k_s - k_c) * torch.exp(-torch.pow(ratio, alpha.unsqueeze(0))) * torch.tanh(k_a * qd)
    return viscous + coulomb + stribeck


class HNetStribeck(nn.Module):
    """
    纯物理 SCV 摩擦网络（无可学习 MLP，仅每关节 SCV 参数）。
    输入当前关节速度 q̇；若提供 q_seq/qd_seq 则取最后一帧。
    """

    def __init__(self, dof: int, *, model: Literal["scv", "cv"] = "scv") -> None:
        super().__init__()
        self.dof = dof
        self.model = model
        if model == "scv":
            self.scv = StribeckSCVParams(dof)
        else:
            self.log_k_v = nn.Parameter(torch.full((dof,), math.log(0.01)))
            self.log_k_c = nn.Parameter(torch.full((dof,), math.log(0.1)))
            self.log_k_a = nn.Parameter(torch.full((dof,), math.log(10.0)))

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        qd = qd_seq[:, -1, :]
        if self.model == "scv":
            tau = self.scv(qd)
        else:
            tau = cv_torque(
                qd,
                _positive(self.log_k_v),
                _positive(self.log_k_c),
                _positive(self.log_k_a),
            )
        return tau, tau


class HNetStribeckPINN(nn.Module):
    """
    Hu 等 PINN 摩擦网络：MLP(速度/位置历史) → τ_pred，SCV(q̇) → τ_physics；
    训练时用 L = (1-λ) MSE(τ_pred, τ_target) + λ MSE(τ_pred, τ_physics)。
    """

    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
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
        self.scv = StribeckSCVParams(dof)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        q_seq: torch.Tensor,
        qd_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, n = qd_seq.shape
        if L != self.seq_len or n != self.dof:
            raise ValueError(f"Expected qd_seq (B, {self.seq_len}, {self.dof}), got {qd_seq.shape}")
        x = torch.cat([q_seq.reshape(B, -1), qd_seq.reshape(B, -1)], dim=-1)
        tau_pred = self.mlp(x)
        tau_physics = self.scv(qd_seq[:, -1, :])
        return tau_pred, tau_physics
