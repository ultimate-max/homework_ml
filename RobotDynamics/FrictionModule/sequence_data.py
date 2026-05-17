"""
从 DeLaN pickle 轨迹构建 Mysteric-Net 滑窗样本 (q_seq, qd_seq)。
"""

from __future__ import annotations

from typing import Any

import dill as pickle
import numpy as np
import torch

from .synthetic_plant import build_windows  # noqa: F401 — used by build_mysteric_tensors


def pickle_has_mcg_decomposition(data: dict[str, Any]) -> bool:
    """数据里是否存在非零的 m/c/g 分解（可用于 τ_fri = τ - m - c - g）。"""
    n = len(data["labels"])
    for i in range(n):
        for key in ("m", "c", "g"):
            if key in data and np.any(np.asarray(data[key][i]) != 0):
                return True
    return False


def load_pickle_trajectories(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def stack_trajectories_to_flat(
    data: dict[str, Any],
    *,
    train_labels: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """拼接多条轨迹为 (N, dof) 时间序列（仅训练子集）。"""
    labels = data["labels"]
    qp_list, qv_list, qa_list, tau_list, m_list, c_list, g_list = [], [], [], [], [], [], []
    for i, lab in enumerate(labels):
        if train_labels is not None and lab not in train_labels:
            continue
        qp_list.append(np.asarray(data["qp"][i], dtype=np.float64))
        qv_list.append(np.asarray(data["qv"][i], dtype=np.float64))
        qa_list.append(np.asarray(data["qa"][i], dtype=np.float64))
        tau_list.append(np.asarray(data["tau"][i], dtype=np.float64))
        m_list.append(np.asarray(data["m"][i], dtype=np.float64))
        c_list.append(np.asarray(data["c"][i], dtype=np.float64))
        g_list.append(np.asarray(data["g"][i], dtype=np.float64))
    qp = np.vstack(qp_list)
    qv = np.vstack(qv_list)
    qa = np.vstack(qa_list)
    tau = np.vstack(tau_list)
    m = np.vstack(m_list)
    c = np.vstack(c_list)
    g = np.vstack(g_list)
    tau_rigid = m + c + g
    tau_fri = tau - tau_rigid
    return qp, qv, qa, tau, tau_rigid, tau_fri


def build_mysteric_tensors(
    qp: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    tau: np.ndarray,
    tau_fri: np.ndarray,
    seq_len: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    q = torch.from_numpy(qp).to(device=dev, dtype=torch.float32)
    qd = torch.from_numpy(qv).to(device=dev, dtype=torch.float32)
    qdd = torch.from_numpy(qa).to(device=dev, dtype=torch.float32)
    t = torch.from_numpy(tau).to(device=dev, dtype=torch.float32)
    tf = torch.from_numpy(tau_fri).to(device=dev, dtype=torch.float32)
    qi, qdi, qddi, taui, q_seq, qd_seq = build_windows(q, qd, qdd, t, seq_len)
    _, _, _, tau_fri_i, _, _ = build_windows(q, qd, qdd, tf, seq_len)
    return {
        "qi": qi,
        "qdi": qdi,
        "qddi": qddi,
        "taui": taui,
        "tau_fri": tau_fri_i,
        "q_seq": q_seq,
        "qd_seq": qd_seq,
    }
