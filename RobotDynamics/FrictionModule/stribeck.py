"""
Stribeck-Coulomb-Viscous (SCV) 摩擦模型（Hu 等, Eq. (3)(4)）及 PINN 摩擦网络。

CV:  τ_F = −(k_c tanh(k_a q̇) + k_v q̇)
SCV: τ_F = −(k_v q̇ + k_c tanh(k_a q̇) + (k_s - k_c) exp(-|q̇/v_s|^α) tanh(k_a q̇))

符号：τ_fri 与 q̇ 反号（阻力），与 pickle 分解 τ_fri = τ − m − c − g 一致。

约束：k_s = k_c + softplus(log_delta_k_s) ≥ k_c（静摩擦不低于库仑摩擦）。
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import torch.nn as nn

_POS_FLOOR = 1e-6
_ALPHA_FLOOR = 0.5


def _positive(param: torch.Tensor, floor: float = _POS_FLOOR) -> torch.Tensor:
    return torch.nn.functional.softplus(param) + floor


def _positive_inv(value: torch.Tensor, floor: float = _POS_FLOOR) -> torch.Tensor:
    """``_positive(p) ≈ value`` 的逆（用于 warm-start / 旧 checkpoint 迁移）。"""
    y = value.clamp(min=floor + 1e-8)
    return torch.log(torch.expm1(y - floor))


def _scalar_positive_inv(value: float, floor: float = _POS_FLOOR) -> float:
    y = max(float(value) - floor, 1e-8)
    return math.log(math.expm1(y))


def _init_log_positive(target: float, floor: float = _POS_FLOOR) -> float:
    return _scalar_positive_inv(target, floor=floor)


_SCV_DEFAULTS = {
    "k_v": 0.1,
    "k_c": 2.0,
    "k_a": 10.0,
    "delta_k_s": 0.1,
    "v_s": 0.05,
    "alpha": 1.5,
}


class StribeckSCVParams(nn.Module):
    """
    每关节可学习 SCV 参数。

    正值：k_v, k_c, k_a, v_s, α；Stribeck 超额静摩擦 Δk_s = softplus(log_delta_k_s) ≥ 0，
    故 **k_s = k_c + Δk_s ≥ k_c** 在参数化上恒成立。
    """

    def __init__(self, dof: int, *, init_v_s: float | None = None, init_alpha: float | None = None) -> None:
        super().__init__()
        self.dof = dof
        v_s0 = _SCV_DEFAULTS["v_s"] if init_v_s is None else init_v_s
        alpha0 = _SCV_DEFAULTS["alpha"] if init_alpha is None else init_alpha
        self.log_k_v = nn.Parameter(torch.full((dof,), _init_log_positive(_SCV_DEFAULTS["k_v"])))
        self.log_k_c = nn.Parameter(torch.full((dof,), _init_log_positive(_SCV_DEFAULTS["k_c"])))
        self.log_k_a = nn.Parameter(torch.full((dof,), _init_log_positive(_SCV_DEFAULTS["k_a"])))
        self.log_delta_k_s = nn.Parameter(
            torch.full((dof,), _init_log_positive(_SCV_DEFAULTS["delta_k_s"]))
        )
        self.log_v_s = nn.Parameter(torch.full((dof,), _init_log_positive(v_s0)))
        self.log_alpha = nn.Parameter(torch.full((dof,), _init_log_positive(alpha0)))

    def positive_coefficients(self) -> dict[str, torch.Tensor]:
        """返回正参数 dict：k_v, k_c, k_a, k_s, v_s, alpha（每关节 1D）。"""
        k_c = _positive(self.log_k_c)
        k_v = _positive(self.log_k_v)
        k_a = _positive(self.log_k_a)
        k_s = k_c + _positive(self.log_delta_k_s)
        v_s = _positive(self.log_v_s)
        alpha = _positive(self.log_alpha, floor=_ALPHA_FLOOR)
        return {
            "k_v": k_v,
            "k_c": k_c,
            "k_a": k_a,
            "k_s": k_s,
            "v_s": v_s,
            "alpha": alpha,
        }

    def forward(self, qd: torch.Tensor) -> torch.Tensor:
        """qd: (B, dof) → τ_physics: (B, dof)"""
        c = self.positive_coefficients()
        return scv_torque(
            qd,
            c["k_v"],
            c["k_c"],
            c["k_a"],
            c["k_s"],
            c["v_s"],
            c["alpha"],
        )

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """兼容旧 checkpoint 中的 ``log_k_s`` → ``log_delta_k_s``。"""
        converted = dict(state_dict)
        if "log_k_s" in converted and "log_delta_k_s" not in converted:
            log_k_s = converted.pop("log_k_s")
            log_k_c = converted.get("log_k_c")
            if log_k_c is None:
                raise KeyError("旧 SCV checkpoint 含 log_k_s 但缺少 log_k_c，无法迁移")
            with torch.no_grad():
                k_c = _positive(log_k_c)
                k_s = _positive(log_k_s)
                delta = (k_s - k_c).clamp(min=_POS_FLOOR)
                converted["log_delta_k_s"] = _positive_inv(delta)
        return super().load_state_dict(converted, strict=strict)


def warmstart_scv_from_samples(
    scv: StribeckSCVParams,
    qd: torch.Tensor,
    tau_fri: torch.Tensor,
    *,
    qd_min: float = 0.02,
    qd_viscous_min: float = 0.15,
    k_c_floor: float = 0.05,
    k_s_margin_frac: float = 0.05,
    k_v_frac_of_kc: float = 0.05,
    k_v_floor: float = 1e-4,
) -> int:
    """
    用训练样本粗估 SCV：``k_c ≈ median(|τ_fri|)``（``|q̇|>qd_min``），
    ``Δk_s = max(5%·k_c, 0.02)``，``k_v`` 由高速段 ``|τ_fri/q̇|`` 上限截断。

    Returns:
        成功初始化参数的关节数。
    """
    if qd.shape != tau_fri.shape or qd.ndim != 2:
        raise ValueError(f"qd {qd.shape} 与 tau_fri {tau_fri.shape} 须同为 (N, dof)")
    dof = scv.dof
    if qd.shape[1] != dof:
        raise ValueError(f"Expected dof {dof}, got qd {qd.shape[1]}")
    n_set = 0
    with torch.no_grad():
        for j in range(dof):
            mask = qd[:, j].abs() > qd_min
            if int(mask.sum()) < 8:
                continue
            est = float(tau_fri[mask, j].abs().median().clamp(min=k_c_floor))
            scv.log_k_c[j] = _scalar_positive_inv(est)
            margin = max(est * k_s_margin_frac, 0.02)
            scv.log_delta_k_s[j] = _scalar_positive_inv(margin)

            mask_v = qd[:, j].abs() > qd_viscous_min
            if int(mask_v.sum()) >= 8:
                kv = float(
                    (tau_fri[mask_v, j].abs() / qd[mask_v, j].abs())
                    .median()
                    .clamp(min=k_v_floor, max=est * max(k_v_frac_of_kc, 0.01))
                )
                scv.log_k_v[j] = _scalar_positive_inv(kv)
            n_set += 1
    return n_set


def cv_torque(
    qd: torch.Tensor,
    k_v: torch.Tensor,
    k_c: torch.Tensor,
    k_a: torch.Tensor,
) -> torch.Tensor:
    """Coulomb-viscous；τ_fri 与 q̇ 反号（与 τ−m−c−g 一致）。"""
    return -(k_v * qd + k_c * torch.tanh(k_a * qd))


def scv_torque(
    qd: torch.Tensor,
    k_v: torch.Tensor,
    k_c: torch.Tensor,
    k_a: torch.Tensor,
    k_s: torch.Tensor,
    v_s: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Stribeck-Coulomb-viscous；τ_fri 与 q̇ 反号（与 τ−m−c−g 一致）。"""
    coulomb = k_c * torch.tanh(k_a * qd)
    viscous = k_v * qd
    ratio = torch.abs(qd / v_s.unsqueeze(0).clamp_min(1e-8))
    stribeck = (k_s - k_c) * torch.exp(-torch.pow(ratio, alpha.unsqueeze(0))) * torch.tanh(k_a * qd)
    return -(viscous + coulomb + stribeck)


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
            self.log_k_v = nn.Parameter(torch.full((dof,), _init_log_positive(0.01)))
            self.log_k_c = nn.Parameter(torch.full((dof,), _init_log_positive(0.1)))
            self.log_k_a = nn.Parameter(torch.full((dof,), _init_log_positive(10.0)))

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
