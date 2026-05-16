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

from .lnet import LNet


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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plot_alpha = 0.8
    color_pred = "r"

    y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, result.delan_tau)), axis=0), -np.inf, -0.01)
    y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, result.delan_tau)), axis=0), 0.01, np.inf)
    y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, result.delan_m)), axis=0), -np.inf, -0.01)
    y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, result.delan_m)), axis=0), 0.01, np.inf)
    y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, result.delan_c)), axis=0), -np.inf, -0.01)
    y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, result.delan_c)), axis=0), 0.01, np.inf)
    y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, result.delan_g)), axis=0), -np.inf, -0.01)
    y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, result.delan_g)), axis=0), 0.01, np.inf)

    div = list(divider)
    ticks = (np.array(div[:-1]) + np.array(div[1:])) / 2.0

    title = f"Seed = {seed}" if seed is not None else "DeLaN L-Net"
    fig = plt.figure(figsize=(24.0 / 1.54, 8.0 / 1.54), dpi=100)
    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
    fig.canvas.manager.set_window_title(title)

    legend = [
        mpatches.Patch(color=color_pred, label="DeLaN"),
        mpatches.Patch(color="k", label="Ground Truth"),
    ]

    def _style_joint_axes(ax0, ax1, ylo, yhi, panel: str | None = None) -> None:
        for ax, j in zip((ax0, ax1), (0, 1)):
            ax.set_ylim(ylo[j], yhi[j])
            ax.set_xticks(ticks)
            ax.set_xticklabels(test_labels)
            ax.vlines(div, ylo[j], yhi[j], linestyles="--", lw=0.5, alpha=1.0)
            ax.set_xlim(div[0], div[-1])
        if panel:
            ax1.text(
                s=panel,
                x=0.5,
                y=-0.25,
                fontsize=12,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
            )

    # tau
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.set_title("tau")
    ax0.text(
        s="Joint 0",
        x=-0.35,
        y=0.5,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
        transform=ax0.transAxes,
    )
    ax0.set_ylabel("Torque [Nm]")
    ax1 = fig.add_subplot(2, 4, 5)
    ax1.text(
        s="Joint 1",
        x=-0.35,
        y=0.5,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
        transform=ax1.transAxes,
    )
    ax1.set_ylabel("Torque [Nm]")
    _style_joint_axes(ax0, ax1, y_t_low, y_t_max, panel="(a)")
    ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc="upper left", ncol=1, framealpha=1.0)
    ax0.plot(test_tau[:, 0], color="k")
    ax1.plot(test_tau[:, 1], color="k")
    ax0.plot(result.delan_tau[:, 0], color=color_pred, alpha=plot_alpha)
    ax1.plot(result.delan_tau[:, 1], color=color_pred, alpha=plot_alpha)

    # m
    ax0 = fig.add_subplot(2, 4, 2)
    ax0.set_title("H(q) * q_ddot")
    ax0.set_ylabel("Torque [Nm]")
    ax1 = fig.add_subplot(2, 4, 6)
    ax1.set_ylabel("Torque [Nm]")
    _style_joint_axes(ax0, ax1, y_m_low, y_m_max, panel="(b)")
    ax0.plot(test_m[:, 0], color="k")
    ax1.plot(test_m[:, 1], color="k")
    ax0.plot(result.delan_m[:, 0], color=color_pred, alpha=plot_alpha)
    ax1.plot(result.delan_m[:, 1], color=color_pred, alpha=plot_alpha)

    # c
    ax0 = fig.add_subplot(2, 4, 3)
    ax0.set_title("c(q, q_dot)")
    ax0.set_ylabel("Torque [Nm]")
    ax1 = fig.add_subplot(2, 4, 7)
    ax1.set_ylabel("Torque [Nm]")
    _style_joint_axes(ax0, ax1, y_c_low, y_c_max, panel="(c)")
    ax0.plot(test_c[:, 0], color="k")
    ax1.plot(test_c[:, 1], color="k")
    ax0.plot(result.delan_c[:, 0], color=color_pred, alpha=plot_alpha)
    ax1.plot(result.delan_c[:, 1], color=color_pred, alpha=plot_alpha)

    # g
    ax0 = fig.add_subplot(2, 4, 4)
    ax0.set_title("g(q)")
    ax0.set_ylabel("Torque [Nm]")
    ax1 = fig.add_subplot(2, 4, 8)
    ax1.set_ylabel("Torque [Nm]")
    _style_joint_axes(ax0, ax1, y_g_low, y_g_max, panel="(d)")
    ax0.plot(test_g[:, 0], color="k")
    ax1.plot(test_g[:, 1], color="k")
    ax0.plot(result.delan_g[:, 0], color=color_pred, alpha=plot_alpha)
    ax1.plot(result.delan_g[:, 1], color=color_pred, alpha=plot_alpha)

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
