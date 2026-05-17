#!/usr/bin/env python3
"""
加载已训练 DeLaN（官方格式 checkpoint），测试集评估与绘图。

示例（仅 DeLaN / delan_lnet.pt）:
  python examples/delan_evaluate.py \\
    --checkpoint checkpoints/delan_lnet.pt \\
    --data data/robot.pickle

Mysteric 权重（mysteric_robot.pt）请用 robot_evaluate.py → figures/robot_friction.png
# 默认保存 figures/delan_performance.png；弹窗: --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import (
    build_lnet,
    evaluate_delan_on_test,
    load_dataset,
    plot_delan_performance,
    print_eval_report,
)


def _lnet_depth(state: dict) -> int:
    layer_ids = [
        int(k.split(".")[1])
        for k in state
        if k.startswith("layers.") and k.endswith(".weight")
    ]
    return max(layer_ids) + 1 if layer_ids else 2


def _strip_state_prefix(state: dict, prefix: str) -> dict:
    n = len(prefix)
    return {k[n:]: v for k, v in state.items() if k.startswith(prefix)}


def load_lnet_from_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict")
    if state is None:
        raise ValueError(f"checkpoint 中无 state_dict: {path}")

    is_mysteric = any(k.startswith("lnet.") for k in state) or ckpt.get("friction_backend") is not None
    if is_mysteric:
        lnet_state = _strip_state_prefix(state, "lnet.")
        if "layers.0.weight" not in lnet_state:
            raise ValueError(
                f"checkpoint 似为 Mysteric-Net 但缺少 lnet.* 权重: {path}\n"
                "完整模型评估请用: python examples/robot_evaluate.py --checkpoint ... --data ..."
            )
        w0 = lnet_state["layers.0.weight"]
        dof = int(ckpt.get("dof", w0.shape[1]))
        hyper = {
            "n_width": int(ckpt.get("lnet_hidden", w0.shape[0])),
            "n_depth": int(ckpt.get("lnet_layers", _lnet_depth(lnet_state))),
            "b_diag_init": float(ckpt.get("mass_diag_eps", ckpt.get("b_diagonal", 1.0e-2))),
            "diagonal_epsilon": float(ckpt.get("lnet_numerical_H_ridge", 1.0e-2)),
            "b_init": float(ckpt.get("b_init", 0.1)),
            "activation": str(ckpt.get("activation", "ReLu")),
        }
        model = build_lnet(dof, hyper)
        model.load_state_dict(lnet_state)
        print(
            f"已从 Mysteric checkpoint 加载 L-Net（摩擦后端={ckpt.get('friction_backend', '?')}）。\n"
            "  本脚本输出 figures/delan_performance.png（仅刚体 τ,m,c,g）。\n"
            "  若要 figures/robot_friction.png（τ + τ_fri 曲线），请运行:\n"
            "    python examples/robot_evaluate.py "
            f"--checkpoint {path} --data <robot.pickle>"
        )
    else:
        hyper = ckpt.get("hyper")
        if hyper is not None:
            model = build_lnet(int(ckpt.get("dof", state["layers.0.weight"].shape[1])), hyper)
        else:
            w0 = state["layers.0.weight"]
            model = build_lnet(
                int(ckpt.get("dof", w0.shape[1])),
                {
                    "n_width": int(ckpt.get("hidden_dim", w0.shape[0])),
                    "n_depth": int(ckpt.get("num_hidden_layers", _lnet_depth(state))),
                    "b_diag_init": float(ckpt.get("b_diagonal", 0.001)),
                    "diagonal_epsilon": 0.01,
                    "b_init": float(ckpt.get("b_init", 1e-4)),
                    "activation": str(ckpt.get("activation", "SoftPlus")),
                },
            )
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="DeLaN 测试集评估（官方同款）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--test-labels", nargs="*", default=["e", "q", "v"])
    p.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="保存评估图（默认开启）",
    )
    p.add_argument("--no-plot", dest="plot", action="store_false", help="不保存图")
    p.add_argument("--show", action="store_true", help="弹窗显示（不保存时用）")
    p.add_argument("--figure-out", type=Path, default=ROOT / "figures" / "delan_performance.png")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint 不存在: {args.checkpoint}")
    if not args.data.is_file():
        raise SystemExit(f"数据不存在: {args.data}")

    _train_data, test_data, divider, _dt = load_dataset(
        filename=str(args.data),
        test_label=tuple(args.test_labels),
    )
    test_labels, test_qp, test_qv, test_qa, _p, _pd, test_tau, test_m, test_c, test_g = test_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  #test={test_qp.shape[0]}  labels={test_labels}")

    model, ckpt = load_lnet_from_checkpoint(args.checkpoint, device)
    seed = args.seed if args.seed is not None else ckpt.get("seed")

    result = evaluate_delan_on_test(
        model,
        test_qp,
        test_qv,
        test_qa,
        test_tau,
        test_m,
        test_c,
        test_g,
        device=device,
    )
    has_mcg = bool(
        np.any(test_m != 0) or np.any(test_c != 0) or np.any(test_g != 0)
    )
    print_eval_report(result, has_mcg_ground_truth=has_mcg)

    if args.plot or args.show:
        plot_delan_performance(
            result,
            test_labels,
            test_tau,
            test_m,
            test_c,
            test_g,
            divider,
            show=args.show,
            save_path=None if args.show else args.figure_out,
            seed=seed,
        )


if __name__ == "__main__":
    main()
