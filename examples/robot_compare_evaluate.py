#!/usr/bin/env python3
"""
多 MystericNet checkpoint 同图对比评估：叠加 τ / τ_fri 预测并标注 RMSE。

示例:
  python examples/robot_compare_evaluate.py \\
    --data data/robot_fric1.pickle \\
    --test-labels q \\
    --checkpoint checkpoints/fo_cascade_pinn_net_epoch00500.pt:ep500 \\
    --checkpoint checkpoints/fo_cascade_pinn_net.pt:final \\
    --figure-out figures/robot_compare.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from robot_compare_common import (
    ModelEval,
    eval_checkpoints,
    load_test_data,
    parse_checkpoint_arg,
    resolve_seq_len,
)

FO_CASCADE_PINN_COLOR = "#d62728"  # 红
LA_FC_PINN_DISPLAY = "LA-FC-PINN"
_OTHER_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]  # 绿、蓝、…


def _is_la_fc_pinn(r: ModelEval) -> bool:
    return (
        r.backend == "fo_cascade_pinn"
        or r.name in ("fo_cascade_pinn", "fo_cascade_pinn_net", LA_FC_PINN_DISPLAY)
        or "fo_cascade_pinn" in r.name
        or "fo_cascade_pinn" in r.path.stem
    )


def _display_name(r: ModelEval, results: list[ModelEval]) -> str:
    """图中注释：fo_cascade_pinn(_net) → LA-FC-PINN。"""
    if not _is_la_fc_pinn(r):
        return r.name
    n_fo = sum(1 for x in results if _is_la_fc_pinn(x))
    if n_fo == 1:
        return LA_FC_PINN_DISPLAY
    if r.name in ("fo_cascade_pinn", "fo_cascade_pinn_net") or "fo_cascade_pinn_net" in r.path.stem:
        return LA_FC_PINN_DISPLAY
    return f"{LA_FC_PINN_DISPLAY} ({r.name})"


def _annotation_lines(
    results: list[ModelEval], *, joint: int | None = None
) -> list[str]:
    lines: list[str] = []
    for r in results:
        label = _display_name(r, results)
        if joint is None:
            tau_rmse = r.rmse_tau
            fri_rmse = r.rmse_tau_fri
        else:
            tau_rmse = float(r.rmse_tau_j[joint])
            fri_rmse = (
                float(r.rmse_tau_fri_j[joint])
                if r.rmse_tau_fri_j is not None
                else None
            )
        if fri_rmse is not None:
            lines.append(f"{label}: τ {tau_rmse:.3f}  τ_fri {fri_rmse:.3f}")
        else:
            lines.append(f"{label}: τ {tau_rmse:.3f}")
    return lines


def _legend_label(r: ModelEval, results: list[ModelEval]) -> str:
    return _display_name(r, results)


def _model_color(r: ModelEval, idx: int, results: list[ModelEval]) -> str:
    if _is_la_fc_pinn(r):
        return FO_CASCADE_PINN_COLOR
    k = sum(
        1
        for i, x in enumerate(results)
        if i < idx and not _is_la_fc_pinn(x)
    )
    return _OTHER_COLORS[k % len(_OTHER_COLORS)]


def _model_colors(n: int) -> list:
    """备用：按序号红绿蓝（未使用 backend 映射时）。"""
    from matplotlib import cm

    base = ["#d62728", "#2ca02c", "#1f77b4"]
    if n <= len(base):
        return base[:n]
    extra = [cm.tab10(i) for i in range(10) if i not in (0, 1, 2)]
    return base + extra[: n - len(base)]


def plot_compare(
    data: dict[str, np.ndarray],
    results: list[ModelEval],
    traj_labels: list[str],
    divider: list[int],
    *,
    figure_out: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    FS_TITLE = 14
    FS_SUBTITLE = 13
    FS_LABEL = 12
    FS_TICK = 11
    FS_LEGEND = 10
    FS_ANN = 10

    tau = data["tau"]
    tau_fri_true = data["tau_fri_true"]
    has_fri_ref = np.any(np.isfinite(tau_fri_true))
    n_dof = tau.shape[1]
    model_colors = [_model_color(r, i, results) for i, r in enumerate(results)]
    model_labels = [_legend_label(r, results) for r in results]
    x = np.arange(tau.shape[0])
    ticks = [(divider[i] + divider[i + 1]) / 2 for i in range(len(traj_labels))]

    fig, axes = plt.subplots(
        n_dof, 2, figsize=(12, 2.6 * n_dof), squeeze=False
    )
    fig.suptitle(
        "Multi-model compare  |  left: τ_fri  |  right: τ_hat",
        fontsize=FS_TITLE,
    )

    for j in range(n_dof):
        ax_f = axes[j, 0]
        ax_t = axes[j, 1]

        if has_fri_ref:
            ax_f.plot(
                x,
                tau_fri_true[:, j],
                "k",
                lw=1.2,
                ls="-",
                alpha=0.85,
                label="τ_fri ref",
            )
        for r, c, lab in zip(results, model_colors, model_labels):
            ax_f.plot(
                x,
                r.pred["tau_fri"][:, j],
                color=c,
                alpha=0.9,
                lw=1.1,
                ls="--",
                label=lab,
            )

        ax_t.plot(
            x, tau[:, j], "k", lw=1.2, ls="-", alpha=0.85, label="τ meas"
        )
        for r, c, lab in zip(results, model_colors, model_labels):
            ax_t.plot(
                x,
                r.pred["tau_hat"][:, j],
                color=c,
                alpha=0.9,
                lw=1.1,
                ls="--",
                label=lab,
            )

        ax_f.set_ylabel(f"J{j} [Nm]", fontsize=FS_LABEL)
        ax_f.tick_params(axis="both", labelsize=FS_TICK)
        ax_t.tick_params(axis="both", labelsize=FS_TICK)
        if j == 0:
            ax_f.set_title("Friction τ_fri", fontsize=FS_SUBTITLE)
            ax_t.set_title("Total torque τ_hat", fontsize=FS_SUBTITLE)
        if j == n_dof - 1:
            ax_f.set_xticks(ticks)
            ax_f.set_xticklabels(traj_labels, fontsize=FS_TICK)
            ax_t.set_xticks(ticks)
            ax_t.set_xticklabels(traj_labels, fontsize=FS_TICK)
            for d in divider:
                ax_f.axvline(d, color="gray", ls="--", lw=0.4)
                ax_t.axvline(d, color="gray", ls="--", lw=0.4)
        else:
            ax_f.set_xticks([])
            ax_t.set_xticks([])

        ann = "\n".join(_annotation_lines(results, joint=j))
        bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="0.7")
        ax_f.text(
            0.02,
            0.98,
            ann,
            transform=ax_f.transAxes,
            va="top",
            ha="left",
            fontsize=FS_ANN,
            family="monospace",
            bbox=bbox,
        )
        ax_t.text(
            0.02,
            0.98,
            ann,
            transform=ax_t.transAxes,
            va="top",
            ha="left",
            fontsize=FS_ANN,
            family="monospace",
            bbox=bbox,
        )

        if j == 0:
            ax_f.legend(loc="upper right", fontsize=FS_LEGEND)
            ax_t.legend(loc="upper right", fontsize=FS_LEGEND)

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_out, dpi=120, bbox_inches="tight")
    print(f"Figure saved: {figure_out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _print_summary(
    results: list[ModelEval],
    traj_labels: list[str],
    n_samples: int,
    has_fri_ref: bool,
) -> None:
    print(f"测试轨迹: {traj_labels}  样本数: {n_samples}")
    print(f"{'model':<24} {'backend':<18} {'RMSE τ':>10}  {'RMSE τ_fri':>12}")
    print("-" * 70)
    for r in results:
        fri_s = f"{r.rmse_tau_fri:12.5f}" if r.rmse_tau_fri is not None else "         n/a"
        print(f"{r.name:<24} {r.backend:<18} {r.rmse_tau:10.5f}  {fri_s}")
    if not has_fri_ref:
        print("（数据无 m/c/g 分解，τ_fri RMSE 未计算）")


def main() -> None:
    p = argparse.ArgumentParser(
        description="多 MystericNet 同图对比：τ / τ_fri 预测 + RMSE 标注"
    )
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="PATH[:LABEL]",
        help="可重复；例 checkpoints/a.pt:ep500",
    )
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="滑窗长度；默认取各 checkpoint 中最大值",
    )
    p.add_argument(
        "--figure-out",
        type=Path,
        default=ROOT / "figures" / "robot_compare.png",
    )
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if len(args.checkpoint) < 1:
        raise SystemExit("至少指定一个 --checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_specs = [parse_checkpoint_arg(s) for s in args.checkpoint]
    seq_len = resolve_seq_len([p for p, _ in ckpt_specs], args.seq_len)

    _, data, traj_labels, divider, has_fri_ref = load_test_data(
        args.data, list(args.test_labels), seq_len, device
    )
    results = eval_checkpoints(ckpt_specs, data, device)

    _print_summary(results, traj_labels, data["tau"].shape[0], has_fri_ref)
    plot_compare(
        data,
        results,
        traj_labels,
        divider,
        figure_out=args.figure_out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
