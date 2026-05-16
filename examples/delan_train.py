#!/usr/bin/env python3
"""
DeLaN L-Net 训练。

- 数据：官方 ``load_dataset``（mysteric_net.delan_data）
- 训练：默认 **标准循环**（每 batch: l_tau + l_E）；可选 ``--loop official`` 复现 example_DeLaN

示例:
  python examples/delan_train.py \\
    --data /home/coral/project/deep_lagrangian_networks/data/character_data.pickle.BAK \\
    -m 1 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.delan_data import init_env, load_dataset
from mysteric_net.delan_eval import evaluate_delan_on_test, plot_delan_performance, print_eval_report
from mysteric_net.delan_train_core import (
    HYPER_DELAN_MODEL,
    HYPER_EXAMPLE,
    build_lnet,
    save_delan_checkpoint,
    train_delan_loop,
    train_delan_official_loop,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeLaN L-Net 训练")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("/home/coral/project/deep_lagrangian_networks/data/character_data.pickle.BAK"),
        help="character_data.pickle 路径",
    )
    p.add_argument(
        "--preset",
        choices=("example", "delan_model"),
        default="delan_model",
        help="delan_model=64x2,lr5e-4（BAK 推荐）；example=128x8,lr5e-3",
    )
    p.add_argument(
        "--loop",
        choices=("standard", "official"),
        default="standard",
        help="standard=每 batch l_tau+l_E（推荐）；official=example_DeLaN 的 replay+累积能量项",
    )
    p.add_argument("--test-labels", nargs="*", default=["e", "q", "v"])
    p.add_argument("--no-energy-loss", action="store_true", help="仅 l_tau，不用功率守恒项")
    p.add_argument("-c", nargs="?", const=1, default=1, type=int, help="使用 CUDA")
    p.add_argument("-i", nargs="?", const=0, default=0, type=int, help="CUDA 设备 id")
    p.add_argument("-s", nargs="?", const=42, default=42, type=int, help="随机种子")
    p.add_argument("-r", nargs="?", const=0, default=0, type=int, help="训练后弹窗显示评估图")
    p.add_argument("-l", nargs="?", const=0, default=0, type=int, help="仅加载权重，跳过训练")
    p.add_argument("-m", nargs="?", const=0, default=0, type=int, help="保存 checkpoint")
    p.add_argument("--load", type=Path, default=None, help="加载 checkpoint")
    p.add_argument("--save", type=Path, default=None, help="保存路径（默认 checkpoints/delan_lnet.pt）")
    p.add_argument("--plot", action="store_true", help="保存评估图")
    p.add_argument("--figure-out", type=Path, default=ROOT / "figures" / "delan_performance.png")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--max-epoch", type=int, default=None, help="覆盖 max_epoch")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise SystemExit(f"数据文件不存在: {args.data}")

    env_args = SimpleNamespace(
        s=[args.s], i=[args.i], c=[args.c], r=[args.r], l=[args.l], m=[args.m],
    )
    seed, cuda, render, load_model, save_model = init_env(env_args)

    train_data, test_data, divider, dt_mean = load_dataset(
        filename=str(args.data),
        test_label=tuple(args.test_labels),
    )
    train_labels, train_qp, train_qv, train_qa, _p, _pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, _tp, _tpd, test_tau, test_m, test_c, test_g = test_data
    n_dof = train_qp.shape[-1]

    print(f"\ndevice={'cuda' if cuda else 'cpu'}  n_dof={n_dof}  dt_mean={dt_mean:.6f}")
    print(f"train ({len(train_labels)}): {train_labels}")
    print(f"test  ({len(test_labels)}): {test_labels}")
    print(f"#train={train_qp.shape[0]}  #test={test_qp.shape[0]}")

    hyper = dict(HYPER_DELAN_MODEL if args.preset == "delan_model" else HYPER_EXAMPLE)
    if args.max_epoch is not None:
        hyper["max_epoch"] = args.max_epoch
    print(f"preset={args.preset}  loop={args.loop}  hyper(width,depth,lr)=({hyper['n_width']},{hyper['n_depth']},{hyper['learning_rate']})")

    load_path = args.load
    if load_model and load_path is None:
        load_path = ROOT / "checkpoints" / "delan_lnet.pt"
        alt = Path("/home/coral/project/deep_lagrangian_networks/data/delan_model.torch")
        if not load_path.is_file() and alt.is_file():
            load_path = alt

    model = build_lnet(n_dof, hyper)
    device = torch.device("cuda" if cuda else "cpu")

    if load_model:
        if load_path is None or not load_path.is_file():
            raise SystemExit(f"未找到 checkpoint: {load_path}")
        state = torch.load(load_path, map_location=device, weights_only=False)
        if "hyper" in state:
            hyper = state["hyper"]
            model = build_lnet(n_dof, hyper)
        model.load_state_dict(state["state_dict"])
        model = model.to(device)
        print(f"已加载: {load_path}")
        epoch_i = int(state.get("epoch", 0))
    elif args.loop == "official":
        epoch_i = train_delan_official_loop(
            model, train_qp, train_qv, train_qa, train_tau, hyper, cuda=cuda,
        )
    else:
        epoch_i = train_delan_loop(
            model,
            train_qp,
            train_qv,
            train_qa,
            train_tau,
            test_qp,
            test_qv,
            test_qa,
            test_tau,
            hyper,
            cuda=cuda,
            use_energy_loss=not args.no_energy_loss,
        )

    if save_model:
        save_to = args.save or (ROOT / "checkpoints" / "delan_lnet.pt")
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_delan_checkpoint(
            save_to,
            model.cpu(),
            hyper,
            epoch_i,
            extra={
                "seed": seed,
                "train_labels": train_labels,
                "test_labels": test_labels,
                "data_path": str(args.data.resolve()),
                "train_loop": args.loop,
            },
        )
        print(f"\n模型已保存: {save_to}")

    if args.skip_eval:
        return

    print("\n################################################")
    print("Evaluating DeLaN:")
    eval_result = evaluate_delan_on_test(
        model, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, device=device,
    )
    print_eval_report(eval_result)

    if args.plot or render:
        plot_delan_performance(
            eval_result,
            test_labels,
            test_tau,
            test_m,
            test_c,
            test_g,
            divider,
            show=bool(render),
            save_path=None if render else args.figure_out,
            seed=seed,
        )


if __name__ == "__main__":
    main()
