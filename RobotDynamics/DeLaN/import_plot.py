"""
导入后 character_data 可视化：按轨迹标签检查 qp / qv / qa / tau 等。

- n_dof=1：单列时间序列（单电机）
- n_dof>1：按关节分列子图（机械臂等多轴）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .signal_filter import sampling_rate_from_t


def _as_T_by_dof(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _subsample(x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    if n > max_points:
        step = max(1, n // max_points)
        idx = idx[::step]
    return x[idx], idx


def _resolve_joint_indices(n_dof: int, joint_indices: Sequence[int] | None) -> list[int]:
    if joint_indices is None or len(joint_indices) == 0:
        return list(range(n_dof))
    out = sorted({int(j) for j in joint_indices})
    for j in out:
        if j < 0 or j >= n_dof:
            raise ValueError(f"关节下标 {j} 超出 [0, {n_dof - 1}]")
    return out


def _label_stats(
    lab: str,
    t: np.ndarray,
    qp: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    tau: np.ndarray,
    *,
    dt: float,
    joint_indices: Sequence[int],
) -> str:
    lines = [
        f"  [{lab}] T={qp.shape[0]}  n_dof={qp.shape[1]}  dt={dt:g}s  fs={1/dt:g}Hz",
    ]
    for j in joint_indices:
        qa_grad = np.gradient(qv[:, j], dt)
        denom = max(float(np.max(np.abs(qa_grad))), 1e-12)
        ratio = float(np.max(np.abs(qa[:, j]))) / denom
        lines.append(
            f"    j{j}: |q|max={np.max(np.abs(qp[:, j])):.4g}  "
            f"|qv|max={np.max(np.abs(qv[:, j])):.4g}  "
            f"|qa|max={np.max(np.abs(qa[:, j])):.4g}  "
            f"|tau|max={np.max(np.abs(tau[:, j])):.4g}  "
            f"qa/grad(max)={ratio:.3g}"
        )
    return "\n".join(lines)


def _plot_single_dof(
    fig,
    axes,
    *,
    t_plot: np.ndarray,
    sel: np.ndarray,
    qp: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    tau: np.ndarray,
    dt: float,
    has_mcg: bool,
    m: np.ndarray | None,
    c: np.ndarray | None,
    g: np.ndarray | None,
) -> None:
    """n_dof=1：纵轴堆叠子图（与原逻辑一致）。"""
    qp_p = qp[sel]
    qv_p = qv[sel]
    qa_p = qa[sel]
    tau_p = tau[sel]

    j = 0
    axes[j, 0].plot(t_plot, qp_p[:, 0], "C0", lw=0.8)
    axes[j, 0].set_ylabel("q [rad]")
    axes[j, 0].grid(True, alpha=0.3)
    j += 1

    axes[j, 0].plot(t_plot, qv_p[:, 0], "C1", lw=0.8)
    axes[j, 0].set_ylabel("qv [rad/s]")
    axes[j, 0].grid(True, alpha=0.3)
    j += 1

    qa_grad = np.gradient(qv[:, 0], dt)
    axes[j, 0].plot(t_plot, qa_p[:, 0], "C2", lw=0.9, label="qa (data)")
    axes[j, 0].plot(
        t_plot,
        qa_grad[sel],
        "C3",
        lw=0.7,
        ls="--",
        alpha=0.85,
        label="d(qv)/dt",
    )
    axes[j, 0].set_ylabel("qa [rad/s²]")
    axes[j, 0].legend(loc="upper right", fontsize=7)
    axes[j, 0].grid(True, alpha=0.3)
    j += 1

    axes[j, 0].plot(t_plot, tau_p[:, 0], "k", lw=0.8, label="tau")
    if has_mcg and m is not None and c is not None and g is not None:
        rigid = m + c + g
        fri = tau - rigid
        axes[j, 0].plot(
            t_plot, rigid[sel, 0], "g", lw=0.6, alpha=0.7, label="m+c+g"
        )
        axes[j, 0].plot(
            t_plot, fri[sel, 0], "r", lw=0.5, alpha=0.6, label="tau-(m+c+g)"
        )
    axes[j, 0].set_ylabel("tau [N·m]")
    axes[j, 0].legend(loc="upper right", fontsize=7)
    axes[j, 0].grid(True, alpha=0.3)
    j += 1

    if has_mcg and m is not None:
        axes[j, 0].plot(t_plot, m[sel, 0], lw=0.7, label="m")
        axes[j, 0].plot(t_plot, c[sel, 0], lw=0.7, label="c")
        axes[j, 0].plot(t_plot, g[sel, 0], lw=0.7, label="g")
        axes[j, 0].set_ylabel("m,c,g")
        axes[j, 0].legend(loc="upper right", fontsize=7)
        axes[j, 0].grid(True, alpha=0.3)


def _plot_multi_dof(
    fig,
    axes,
    *,
    t_plot: np.ndarray,
    sel: np.ndarray,
    qp: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    tau: np.ndarray,
    dt: float,
    has_mcg: bool,
    m: np.ndarray | None,
    c: np.ndarray | None,
    g: np.ndarray | None,
    joint_indices: Sequence[int],
) -> None:
    """n_dof>1：行=物理量，列=关节。"""
    n_cols = len(joint_indices)
    row_labels = ("q [rad]", "qv [rad/s]", "qa [rad/s²]", "tau [N·m]")
    n_rows = len(row_labels) + (1 if has_mcg and m is not None else 0)

    ylabels = ("q [rad]", "qv [rad/s]", "qa [rad/s²]", "tau [N·m]", "m,c,g")

    for col, ji in enumerate(joint_indices):
        ax = axes[0, col]
        ax.plot(t_plot, qp[sel, ji], f"C{ji % 10}", lw=0.8)
        ax.set_title(f"joint {ji}", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(t_plot, qv[sel, ji], f"C{ji % 10}", lw=0.8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        qa_grad = np.gradient(qv[:, ji], dt)
        ax.plot(t_plot, qa[sel, ji], "C2", lw=0.9, label="qa")
        ax.plot(
            t_plot,
            qa_grad[sel],
            "C3",
            lw=0.7,
            ls="--",
            alpha=0.85,
            label="d(qv)/dt",
        )
        if col == 0:
            ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        ax.plot(t_plot, tau[sel, ji], "k", lw=0.8, label="tau")
        if has_mcg and m is not None and c is not None and g is not None:
            rigid = m + c + g
            fri = tau - rigid
            ax.plot(
                t_plot,
                rigid[sel, ji],
                "g",
                lw=0.6,
                alpha=0.7,
                label="m+c+g",
            )
            ax.plot(
                t_plot,
                fri[sel, ji],
                "r",
                lw=0.5,
                alpha=0.6,
                label="τ_fri",
            )
        if col == 0:
            ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

        if has_mcg and m is not None:
            ax = axes[4, col]
            ax.plot(t_plot, m[sel, ji], lw=0.7, label="m")
            ax.plot(t_plot, c[sel, ji], lw=0.7, label="c")
            ax.plot(t_plot, g[sel, ji], lw=0.7, label="g")
            if col == 0:
                ax.legend(loc="upper right", fontsize=6)
            ax.grid(True, alpha=0.3)

    for r in range(n_rows):
        axes[r, 0].set_ylabel(ylabels[r], fontsize=8)


def plot_character_data(
    data: dict[str, Any],
    figure_dir: str | Path,
    *,
    max_points: int = 12_000,
    show: bool = False,
    dt_hint: float | None = None,
    joint_indices: Sequence[int] | None = None,
) -> Path:
    """
    为每条轨迹标签保存检查图。

    - n_dof=1：单列子图（q, qv, qa, tau, 可选 m/c/g）
    - n_dof>1：行×关节列网格；每关节对比 qa 与 d(qv)/dt

    joint_indices: 仅画这些关节（默认全部）。返回输出目录路径。
    """
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = data["labels"]
    n_traj = len(labels)
    print(f"\n===== 导入数据检查（共 {n_traj} 条轨迹）=====", file=sys.stderr)

    n_dof_global = int(_as_T_by_dof(data["qp"][0]).shape[1])

    for i, lab in enumerate(labels):
        t = np.asarray(data["t"][i], dtype=np.float64).reshape(-1)
        qp = _as_T_by_dof(data["qp"][i])
        qv = _as_T_by_dof(data["qv"][i])
        qa = _as_T_by_dof(data["qa"][i])
        tau = _as_T_by_dof(data["tau"][i])
        n_dof = qp.shape[1]
        joints = _resolve_joint_indices(n_dof, joint_indices)

        try:
            fs = sampling_rate_from_t(t, dt_hint=dt_hint)
            dt = 1.0 / fs
        except ValueError:
            dt = 0.001
            fs = 1.0 / dt

        print(
            _label_stats(lab, t, qp, qv, qa, tau, dt=dt, joint_indices=joints),
            file=sys.stderr,
        )

        has_mcg = any(
            np.any(np.asarray(data[k][i]) != 0) for k in ("m", "c", "g") if k in data
        )
        m = _as_T_by_dof(data["m"][i]) if has_mcg else None
        c = _as_T_by_dof(data["c"][i]) if has_mcg else None
        g = _as_T_by_dof(data["g"][i]) if has_mcg else None

        t_plot, sel = _subsample(t, max_points)

        if n_dof == 1:
            n_rows = 5 if has_mcg else 4
            fig, axes = plt.subplots(
                n_rows, 1, figsize=(12, 2.2 * n_rows), sharex=True, squeeze=False
            )
            fig.suptitle(
                f"Import check — label '{lab}'  (n_dof=1, motor)",
                fontsize=12,
            )
            _plot_single_dof(
                fig,
                axes,
                t_plot=t_plot,
                sel=sel,
                qp=qp,
                qv=qv,
                qa=qa,
                tau=tau,
                dt=dt,
                has_mcg=has_mcg,
                m=m,
                c=c,
                g=g,
            )
        else:
            n_cols = len(joints)
            n_rows = 5 if has_mcg and m is not None else 4
            fig_w = max(12.0, 2.1 * n_cols)
            fig_h = 2.0 * n_rows + 0.8
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                sharex=True,
                squeeze=False,
            )
            shown = (
                f"joints {joints[0]}..{joints[-1]}"
                if len(joints) > 1
                else f"joint {joints[0]}"
            )
            fig.suptitle(
                f"Import check — label '{lab}'  (n_dof={n_dof}, {shown})",
                fontsize=12,
            )
            _plot_multi_dof(
                fig,
                axes,
                t_plot=t_plot,
                sel=sel,
                qp=qp,
                qv=qv,
                qa=qa,
                tau=tau,
                dt=dt,
                has_mcg=has_mcg,
                m=m,
                c=c,
                g=g,
                joint_indices=joints,
            )

        axes[-1, 0].set_xlabel("time [s]")
        if n_dof > 1:
            for col in range(axes.shape[1]):
                axes[-1, col].set_xlabel("time [s]")
        fig.tight_layout()

        out_path = figure_dir / f"import_{lab}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  图已保存: {out_path}", file=sys.stderr)
        if show:
            plt.show()
        else:
            plt.close(fig)

    _plot_tau_overview(
        data,
        figure_dir,
        labels=labels,
        max_points=max_points,
        n_dof=n_dof_global,
        joint_indices=joint_indices,
        show=show,
    )

    print("===== 检查结束 =====\n", file=sys.stderr)
    return figure_dir


def _plot_tau_overview(
    data: dict[str, Any],
    figure_dir: Path,
    *,
    labels: list,
    max_points: int,
    n_dof: int,
    joint_indices: Sequence[int] | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    joints = _resolve_joint_indices(n_dof, joint_indices)

    if n_dof == 1:
        fig, ax = plt.subplots(figsize=(12, 3))
        for i, lab in enumerate(labels):
            tau = _as_T_by_dof(data["tau"][i])[:, 0]
            n = min(len(tau), max_points)
            step = max(1, len(tau) // n)
            y = tau[::step]
            x = np.arange(y.size) + i * (y.size + 50)
            ax.plot(x, y, lw=0.6, label=str(lab))
        ax.set_title("All labels — tau joint 0 (concatenated index)")
        ax.set_ylabel("tau [N·m]")
        ax.legend(ncol=min(6, len(labels)), fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        overview = figure_dir / "import_all_tau.png"
        fig.savefig(overview, dpi=120, bbox_inches="tight")
        print(f"  总览图: {overview}", file=sys.stderr)
        if not show:
            plt.close(fig)
        return

    n_cols = len(joints)
    n_rows = int(np.ceil(n_cols / 3))
    n_cols_grid = min(3, n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols_grid,
        figsize=(4.2 * n_cols_grid, 2.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle("All labels — |tau| per joint (concatenated index)", fontsize=11)

    for plot_idx, ji in enumerate(joints):
        r, c = divmod(plot_idx, n_cols_grid)
        ax = axes[r, c]
        for i, lab in enumerate(labels):
            tau = _as_T_by_dof(data["tau"][i])[:, ji]
            n = min(len(tau), max_points)
            step = max(1, len(tau) // n)
            y = np.abs(tau[::step])
            x = np.arange(y.size) + i * (y.size + 50)
            ax.plot(x, y, lw=0.6, label=str(lab))
        ax.set_title(f"joint {ji}", fontsize=9)
        ax.set_ylabel("|tau| [N·m]")
        ax.grid(True, alpha=0.3)
        if plot_idx == 0:
            ax.legend(ncol=min(3, len(labels)), fontsize=7, loc="upper right")

    for plot_idx in range(len(joints), n_rows * n_cols_grid):
        r, c = divmod(plot_idx, n_cols_grid)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    overview = figure_dir / "import_all_tau.png"
    fig.savefig(overview, dpi=120, bbox_inches="tight")
    print(f"  总览图: {overview}", file=sys.stderr)
    if not show:
        plt.close(fig)
