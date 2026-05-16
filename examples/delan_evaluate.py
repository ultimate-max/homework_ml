#!/usr/bin/env python3
"""
加载已训练 DeLaN（官方格式 checkpoint），测试集评估与绘图。

示例:
  python examples/delan_evaluate.py \\
    --checkpoint /home/coral/project/deep_lagrangian_networks/data/delan_model.torch \\
    --data /path/to/character_data.pickle.BAK --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.delan_data import load_dataset
from mysteric_net.delan_eval import evaluate_delan_on_test, plot_delan_performance, print_eval_report
from mysteric_net.delan_train_core import build_lnet


def load_lnet_from_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict")
    if state is None:
        raise ValueError(f"checkpoint 中无 state_dict: {path}")
    hyper = ckpt.get("hyper")
    if hyper is not None:
        model = build_lnet(int(ckpt.get("dof", state["layers.0.weight"].shape[1])), hyper)
    else:
        w0 = state["layers.0.weight"]
        layer_ids = [int(k.split(".")[1]) for k in state if k.startswith("layers.") and k.endswith(".weight")]
        model = build_lnet(
            int(ckpt.get("dof", w0.shape[1])),
            {
                "n_width": int(ckpt.get("hidden_dim", w0.shape[0])),
                "n_depth": int(ckpt.get("num_hidden_layers", max(layer_ids) + 1 if layer_ids else 2)),
                "b_diag_init": float(ckpt.get("b_diagonal", 0.001)),
                "diagonal_epsilon": 0.01,
                "b_init": float(ckpt.get("b_init", 1e-4)),
                "activation": str(ckpt.get("activation", "SoftPlus")),
            },
        )
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="DeLaN 测试集评估（官方同款）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--test-labels", nargs="*", default=["e", "q", "v"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--show", action="store_true")
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
    print_eval_report(result)

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
