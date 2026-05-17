"""
DeLaN 测试集评估（对齐 deep_lagrangian_networks/example_DeLaN.py 训练后流程）。

- 力矩分解: g(q), c(q,qd), m = H(q) qdd
- MSE: tau, m, c, g, 功率守恒 dE/dt
- 可选: 与官方相同的 2×4 子图
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .lnet import LNet  # noqa: F401 — used by evaluate_delan_on_test


@dataclass
class DeLaNEvalResult:
    err_tau: float
    err_m: float
    err_c: float
    err_g: float
    err_dEdt: float
    t_eval_per_sample: float
    delan_tau: np.ndarray
    delan_m: np.ndarray
    delan_c: np.ndarray
    delan_g: np.ndarray
    delan_dEdt: np.ndarray


@torch.no_grad()
def _batched_decomposition(
    lnet: LNet,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量分解 g, c, m（与官方 batched 段一致）。"""
    zeros_qd = torch.zeros_like(qd)
    zeros_qdd = torch.zeros_like(qdd)
    delan_g = lnet.inv_dyn(q, zeros_qd, zeros_qdd).cpu().numpy()
    delan_c = lnet.inv_dyn(q, qd, zeros_qdd).cpu().numpy() - delan_g
    delan_m = lnet.inv_dyn(q, zeros_qd, qdd).cpu().numpy() - delan_g
    return delan_g, delan_c, delan_m


@torch.no_grad()
def evaluate_delan_on_test(
    lnet: LNet,
    test_qp: np.ndarray,
    test_qv: np.ndarray,
    test_qa: np.ndarray,
    test_tau: np.ndarray,
    test_m: np.ndarray,
    test_c: np.ndarray,
    test_g: np.ndarray,
    device: torch.device | str = "cpu",
) -> DeLaNEvalResult:
    """
    在测试集上评估，含逐样本在线力矩计时（与官方 CPU 单样本循环一致）。
    """
    dev = torch.device(device)
    lnet = lnet.to(dev)
    n = test_qp.shape[0]
    n_dof = test_qp.shape[1]

    q_all = torch.from_numpy(test_qp).to(device=dev, dtype=torch.float32)
    qd_all = torch.from_numpy(test_qv).to(device=dev, dtype=torch.float32)
    qdd_all = torch.from_numpy(test_qa).to(device=dev, dtype=torch.float32)

    t0_batch = time.perf_counter()
    delan_g, delan_c, delan_m = _batched_decomposition(lnet, q_all, qd_all, qdd_all)
    _ = (time.perf_counter() - t0_batch) / max(3.0 * n, 1.0)

    lnet_cpu = lnet.cpu()
    delan_tau = np.zeros((n, n_dof), dtype=np.float64)
    delan_dEdt = np.zeros((n, 1), dtype=np.float64)

    t0_eval = time.perf_counter()
    for i in range(n):
        qi = torch.from_numpy(test_qp[i]).float().view(1, -1)
        qdi = torch.from_numpy(test_qv[i]).float().view(1, -1)
        qddi = torch.from_numpy(test_qa[i]).float().view(1, -1)
        dyn = lnet_cpu.dynamics(qi, qdi, qddi)
        delan_tau[i] = dyn.tau.numpy().squeeze()
        delan_dEdt[i] = (dyn.dTdt + dyn.dVdt).numpy().reshape(-1, 1)

    t_eval = (time.perf_counter() - t0_eval) / max(float(n), 1.0)

    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    inv_n = 1.0 / max(float(n), 1.0)
    return DeLaNEvalResult(
        err_tau=inv_n * np.sum((delan_tau - test_tau) ** 2),
        err_m=inv_n * np.sum((delan_m - test_m) ** 2),
        err_c=inv_n * np.sum((delan_c - test_c) ** 2),
        err_g=inv_n * np.sum((delan_g - test_g) ** 2),
        err_dEdt=inv_n * np.sum((delan_dEdt - test_dEdt) ** 2),
        t_eval_per_sample=t_eval,
        delan_tau=delan_tau,
        delan_m=delan_m,
        delan_c=delan_c,
        delan_g=delan_g,
        delan_dEdt=delan_dEdt,
    )


def print_eval_report(result: DeLaNEvalResult, *, has_mcg_ground_truth: bool = True) -> None:
    print("\n################################################")
    print("Evaluating DeLaN (test set):")
    print("\nPerformance:")
    print(f"                Torque MSE = {result.err_tau:.3e}")
    if has_mcg_ground_truth:
        print(f"              Inertial MSE = {result.err_m:.3e}")
        print(f"Coriolis & Centrifugal MSE = {result.err_c:.3e}")
        print(f"         Gravitational MSE = {result.err_g:.3e}")
    else:
        print("  (无 m/c/g 真值，已跳过 Inertial / Coriolis / Gravity MSE)")
    print(f"    Power Conservation MSE = {result.err_dEdt:.3e}")
    hz = 1.0 / result.t_eval_per_sample if result.t_eval_per_sample > 0 else float("inf")
    print(f"      Comp Time per Sample = {result.t_eval_per_sample:.3e}s / {hz:.1f}Hz")


def _joint_ylim(gt_col: np.ndarray, pred_col: np.ndarray) -> tuple[float, float]:
    both = np.concatenate([np.asarray(gt_col).ravel(), np.asarray(pred_col).ravel()])
    lo, hi = float(np.min(both)), float(np.max(both))
    span = hi - lo
    if span < 1e-9:
        return lo - 0.01, hi + 0.01
    margin = 0.12 * span + 1e-6
    return lo - margin, hi + margin


def plot_delan_performance(
    result: DeLaNEvalResult,
    test_labels: Sequence[str],
    test_tau: np.ndarray,
    test_m: np.ndarray,
    test_c: np.ndarray,
    test_g: np.ndarray,
    divider: Sequence[int],
    *,
    show: bool = False,
    save_path: Path | None = None,
    seed: int | None = None,
) -> None:
    """测试集力矩分解图：每行一个关节，四列 tau / m / c / g（任意 n_dof）。"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plot_alpha = 0.8
    color_pred = "r"
    n_dof = int(test_tau.shape[1])
    div = list(divider)
    ticks = (np.array(div[:-1]) + np.array(div[1:])) / 2.0

    panels = (
        ("tau", test_tau, result.delan_tau),
        ("H(q) * q_ddot", test_m, result.delan_m),
        ("c(q, q_dot)", test_c, result.delan_c),
        ("g(q)", test_g, result.delan_g),
    )

    title = f"Seed = {seed}" if seed is not None else "DeLaN L-Net"
    fig_h = max(4.0, 2.2 * n_dof)
    fig, axes = plt.subplots(n_dof, 4, figsize=(20.0, fig_h), dpi=100, squeeze=False)
    fig.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.92, wspace=0.28, hspace=0.35)
    if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)

    legend = [
        mpatches.Patch(color=color_pred, label="DeLaN"),
        mpatches.Patch(color="k", label="Ground Truth"),
    ]

    for j in range(n_dof):
        for col, (col_title, gt, pred) in enumerate(panels):
            ax = axes[j, col]
            ylo, yhi = _joint_ylim(gt[:, j], pred[:, j])
            ax.set_ylim(ylo, yhi)
            ax.set_xlim(div[0], div[-1])
            ax.vlines(div, ylo, yhi, linestyles="--", lw=0.5, alpha=1.0)
            ax.plot(gt[:, j], color="k")
            ax.plot(pred[:, j], color=color_pred, alpha=plot_alpha)
            if j == 0:
                ax.set_title(col_title)
            if j == n_dof - 1:
                ax.set_xticks(ticks)
                ax.set_xticklabels(test_labels)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(f"J{j}\n[Nm]", fontsize=9)
            if j == 0 and col == 0:
                ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=1.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=save_path.suffix.lstrip(".") or "png", bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\n################################################\n")
