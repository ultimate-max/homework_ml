#!/usr/bin/env python3
"""
DeLaN L-Net 训练。

- 数据：官方 ``load_dataset``（RobotDynamics.delan_data）
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import (
    HYPER_DELAN_MODEL,
    HYPER_EXAMPLE,
    build_lnet,
    evaluate_delan_on_test,
    init_env,
    inspect_dataset,
    load_dataset,
    plot_delan_performance,
    print_eval_report,
    save_delan_checkpoint,
    suggest_hyper,
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
        choices=("example", "delan_model", "auto"),
        default="delan_model",
        help="delan_model / example；auto=按 n_dof 与样本量缩放（任意机械臂推荐）",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="仅打印数据集信息（n_dof、轨迹数等）后退出",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=None,
        metavar="F",
        help="无 test-label 匹配时，用最后 F 比例轨迹作测试（如 0.2）",
    )
    p.add_argument(
        "--loop",
        choices=("standard", "official"),
        default="standard",
        help="standard=每 batch l_tau+l_E（推荐）；official=example_DeLaN 的 replay+累积能量项",
    )
    p.add_argument("--test-labels", nargs="*", default=["e", "q", "v"])
    p.add_argument("--no-energy-loss", action="store_true", help="仅 l_tau，不用功率守恒项")
    p.add_argument(
        "--tau-loss",
        choices=("mse", "smape"),
        default="mse",
        help="力矩项：mse=官方 MSE；smape=对称 MAPE，多关节量级差时推荐（论文式 20）",
    )
    p.add_argument(
        "--smape-eps",
        type=float,
        default=1e-3,
        help="SMAPE 分母稳定项 |τ|+|τ̂|+eps（Nm 量级数据常用 1e-3~1e-2）",
    )
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

    if args.inspect:
        print(inspect_dataset(args.data))
        return

    env_args = SimpleNamespace(
        s=[args.s], i=[args.i], c=[args.c], r=[args.r], l=[args.l], m=[args.m],
    )
    seed, cuda, render, load_model, save_model = init_env(env_args)

    train_data, test_data, divider, dt_mean = load_dataset(
        filename=str(args.data),
        test_label=tuple(args.test_labels),
        test_frac=args.test_frac,
    )
    train_labels, train_qp, train_qv, train_qa, _p, _pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, _tp, _tpd, test_tau, test_m, test_c, test_g = test_data
    n_dof = train_qp.shape[-1]

    print(f"\ndevice={'cuda' if cuda else 'cpu'}  n_dof={n_dof}  dt_mean={dt_mean:.6f}")
    print(f"train ({len(train_labels)}): {train_labels}")
    print(f"test  ({len(test_labels)}): {test_labels}")
    print(f"#train={train_qp.shape[0]}  #test={test_qp.shape[0]}")
    if n_dof > 6:
        print(
            f"提示: n_dof={n_dof} 时 H 与 dH/dq 计算量约为 O(n^4)，"
            "可适当减小 --preset auto 给出的 batch 或 n_width。"
        )

    if args.preset == "auto":
        hyper = suggest_hyper(n_dof, train_qp.shape[0], base="delan_model")
    else:
        hyper = dict(HYPER_DELAN_MODEL if args.preset == "delan_model" else HYPER_EXAMPLE)
    if args.max_epoch is not None:
        hyper["max_epoch"] = args.max_epoch
    print(
        f"preset={args.preset}  loop={args.loop}  tau_loss={args.tau_loss}  "
        f"hyper(width,depth,lr)=({hyper['n_width']},{hyper['n_depth']},{hyper['learning_rate']})"
    )

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
        ckpt_dof = state.get("n_dof", state.get("dof"))
        if ckpt_dof is not None and int(ckpt_dof) != n_dof:
            raise SystemExit(
                f"checkpoint n_dof={ckpt_dof} 与数据 n_dof={n_dof} 不一致，请重新训练或换数据。"
            )
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
            tau_loss=args.tau_loss,
            smape_eps=args.smape_eps,
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
                "tau_loss": args.tau_loss,
                "smape_eps": args.smape_eps,
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
    has_mcg = bool(
        np.any(test_m != 0) or np.any(test_c != 0) or np.any(test_g != 0)
    )
    print_eval_report(eval_result, has_mcg_ground_truth=has_mcg)

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
