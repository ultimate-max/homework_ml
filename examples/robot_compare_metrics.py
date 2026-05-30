#!/usr/bin/env python3
"""
测试集上统计多个 MystericNet 的 τ_hat / τ_fri RMSE。

示例:
  python examples/robot_compare_metrics.py \\
    --data data/robot_fric1.pickle \\
    --test-labels e v q \\
    --checkpoint checkpoints/fo_cascade_pinn_net_epoch00500.pt:ep500 \\
    --checkpoint checkpoints/fo_cascade_pinn_net.pt:final \\
    --per-joint \\
    --csv-out checkpoints/compare_rmse.csv
"""

from __future__ import annotations

import argparse
import csv
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


def print_summary_table(
    results: list[ModelEval],
    traj_labels: list[str],
    n_samples: int,
    has_fri_ref: bool,
) -> None:
    print(f"测试轨迹: {traj_labels}  样本数: {n_samples}  n_dof: {results[0].rmse_tau_j.shape[0]}")
    print()
    hdr = f"{'model':<22} {'backend':<18} {'RMSE τ_hat':>12} {'RMSE τ_fri':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        fri_s = f"{r.rmse_tau_fri:12.5f}" if r.rmse_tau_fri is not None else "         n/a"
        print(f"{r.name:<22} {r.backend:<18} {r.rmse_tau:12.5f} {fri_s}")
    if not has_fri_ref:
        print("\n（数据无 m/c/g 分解，τ_fri RMSE 未计算）")


def print_per_joint_table(results: list[ModelEval], has_fri_ref: bool) -> None:
    n_dof = results[0].rmse_tau_j.shape[0]
    print("\n按关节 RMSE:")
    for j in range(n_dof):
        print(f"\n  Joint {j}")
        print(f"  {'model':<22} {'RMSE τ_hat':>12} {'RMSE τ_fri':>12}")
        print(f"  {'-'*48}")
        for r in results:
            fri_v = (
                f"{r.rmse_tau_fri_j[j]:12.5f}"
                if has_fri_ref and r.rmse_tau_fri_j is not None
                else "         n/a"
            )
            print(f"  {r.name:<22} {r.rmse_tau_j[j]:12.5f} {fri_v}")


def write_csv(
    path: Path,
    results: list[ModelEval],
    traj_labels: list[str],
    n_samples: int,
    has_fri_ref: bool,
) -> None:
    n_dof = results[0].rmse_tau_j.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "backend",
        "checkpoint",
        "test_labels",
        "n_samples",
        "rmse_tau_hat",
        "rmse_tau_fri",
    ]
    for j in range(n_dof):
        fieldnames.append(f"rmse_tau_hat_j{j}")
    if has_fri_ref:
        for j in range(n_dof):
            fieldnames.append(f"rmse_tau_fri_j{j}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        labels_s = " ".join(traj_labels)
        for r in results:
            row = {
                "model": r.name,
                "backend": r.backend,
                "checkpoint": str(r.path.resolve()),
                "test_labels": labels_s,
                "n_samples": n_samples,
                "rmse_tau_hat": f"{r.rmse_tau:.6f}",
                "rmse_tau_fri": (
                    f"{r.rmse_tau_fri:.6f}" if r.rmse_tau_fri is not None else ""
                ),
            }
            for j in range(n_dof):
                row[f"rmse_tau_hat_j{j}"] = f"{r.rmse_tau_j[j]:.6f}"
            if has_fri_ref and r.rmse_tau_fri_j is not None:
                for j in range(n_dof):
                    row[f"rmse_tau_fri_j{j}"] = f"{r.rmse_tau_fri_j[j]:.6f}"
            w.writerow(row)
    print(f"\nCSV 已写入: {path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="测试集多模型 RMSE 统计（τ_hat 与 τ_fri）"
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
        "--per-joint",
        action="store_true",
        help="额外打印各关节 RMSE",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="将结果写入 CSV",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_specs = [parse_checkpoint_arg(s) for s in args.checkpoint]
    seq_len = resolve_seq_len([p for p, _ in ckpt_specs], args.seq_len)

    _, data, traj_labels, _, has_fri_ref = load_test_data(
        args.data, list(args.test_labels), seq_len, device
    )
    n_samples = int(data["tau"].shape[0])

    print(f"device={device}  data={args.data}  seq_len={seq_len}")
    results = eval_checkpoints(ckpt_specs, data, device)

    print()
    print_summary_table(results, traj_labels, n_samples, has_fri_ref)
    if args.per_joint:
        print_per_joint_table(results, has_fri_ref)
    if args.csv_out is not None:
        write_csv(args.csv_out, results, traj_labels, n_samples, has_fri_ref)


if __name__ == "__main__":
    main()
