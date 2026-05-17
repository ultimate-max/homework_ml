"""
2-DoF 平面连杆 + 论文式 MIMO 迟滞摩擦（与 examples/synthetic_train 一致），用于生成离线数据集。
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch


def build_windows(
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    tau: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = q.shape[0]
    if T < seq_len:
        raise ValueError("Trajectory shorter than seq_len")
    n = T - seq_len + 1
    dof = q.shape[1]
    q_seq = torch.zeros(n, seq_len, dof, device=q.device, dtype=q.dtype)
    qd_seq = torch.zeros_like(q_seq)
    qi = torch.zeros(n, dof, device=q.device, dtype=q.dtype)
    qdi = torch.zeros_like(qi)
    qddi = torch.zeros_like(qi)
    taui = torch.zeros_like(qi)
    for i in range(n):
        sl = slice(i, i + seq_len)
        q_seq[i] = q[sl]
        qd_seq[i] = qd[sl]
        t0 = i + seq_len - 1
        qi[i] = q[t0]
        qdi[i] = qd[t0]
        qddi[i] = qdd[t0]
        taui[i] = tau[t0]
    return qi, qdi, qddi, taui, q_seq, qd_seq


def simulate_2dof_inverse_dynamics(
    T: int = 8000,
    seq_len: int = 30,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    返回原始轨迹 (T,dof) 与滑窗样本 (N,dof)、(N,L,dof)。
    """
    device = torch.device(device)
    dof = 2
    t = torch.linspace(0, 40 * math.pi, T, device=device)
    q = torch.stack([torch.sin(0.3 * t), 0.4 * torch.cos(0.7 * t + 0.2)], dim=1)
    dt = (t[1] - t[0]).item()
    qd = torch.gradient(q, spacing=dt, dim=0)[0]
    qdd = torch.gradient(qd, spacing=dt, dim=0)[0]

    m1, m2 = 1.2, 0.9
    l1, l2 = 0.35, 0.28
    g0 = 9.81
    tau_true_core = torch.zeros_like(q)
    for ti in range(T):
        th1, th2 = q[ti, 0], q[ti, 1]
        d1, d2 = qd[ti, 0], qd[ti, 1]
        dd1, dd2 = qdd[ti, 0], qdd[ti, 1]
        M11 = m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * torch.cos(th2))
        M12 = m2 * (l2**2 + l1 * l2 * torch.cos(th2))
        M21 = M12
        M22 = m2 * l2**2
        h = -m2 * l1 * l2 * torch.sin(th2)
        C1 = h * (2 * d1 + d2) * d2
        C2 = h * d1**2
        g1 = (m1 + m2) * g0 * l1 * torch.cos(th1) + m2 * g0 * l2 * torch.cos(th1 + th2)
        g2 = m2 * g0 * l2 * torch.cos(th1 + th2)
        M = torch.tensor([[M11, M12], [M21, M22]], device=device)
        qddv = torch.tensor([dd1, dd2], device=device)
        tau_true_core[ti] = M @ qddv + torch.tensor([C1, C2], device=device) + torch.tensor([g1, g2], device=device)

    fri = torch.zeros_like(q)
    for ti in range(T):
        past = max(0, ti - seq_len)
        hist = q[past:ti]
        if hist.numel() == 0:
            delta = torch.zeros(1, dof, device=q.device, dtype=q.dtype)
        else:
            delta = hist - q[ti : ti + 1]
        fri[ti, 0] = 0.08 * torch.tanh(torch.mean(delta[:, 0])) + 0.03 * qd[ti, 1]
        fri[ti, 1] = -0.05 * torch.tanh(torch.mean(delta[:, 1])) + 0.02 * qd[ti, 0]

    tau = tau_true_core + fri
    qi, qdi, qddi, taui, q_seq, qd_seq = build_windows(q, qd, qdd, tau, seq_len)

    return {
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "tau": tau,
        "tau_rigid": tau_true_core,
        "tau_fri": fri,
        "t": t,
        "dt": dt,
        "qi": qi,
        "qdi": qdi,
        "qddi": qddi,
        "taui": taui,
        "q_seq": q_seq,
        "qd_seq": qd_seq,
        "seq_len": seq_len,
        "dof": dof,
    }
