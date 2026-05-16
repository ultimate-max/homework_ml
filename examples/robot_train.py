#!/usr/bin/env python3
"""
在 robot.pickle（或任意 DeLaN pickle）上训练 Mysteric-Net：L-Net + 摩擦网络。

摩擦后端（Hu 等 SCV / PINN）:
  --friction-backend stribeck       纯可学习 SCV 物理模型
  --friction-backend stribeck_pinn  MLP + SCV 物理损失（论文 Eq. (6)）
  --friction-backend tcn            原 TCN（默认 Yeo 等）

示例:
  python examples/robot_train.py \\
    --data data/robot.pickle \\
    --friction-backend stribeck_pinn \\
    --lambda-physics 0.5 \\
    -m 1
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

from mysteric_net.delan_data import load_dataset
from mysteric_net.delan_hyper import suggest_hyper
from mysteric_net.delan_losses import torque_loss
from mysteric_net.friction_losses import friction_pinn_loss
from mysteric_net.losses import mysteric_losses
from mysteric_net.model import MystericNet
from mysteric_net.sequence_data import (
    build_mysteric_tensors,
    load_pickle_trajectories,
    stack_trajectories_to_flat,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mysteric-Net（L-Net + 摩擦）训练")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument(
        "--friction-backend",
        choices=("tcn", "stribeck", "stribeck_pinn"),
        default="stribeck_pinn",
    )
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lnet-width", type=int, default=None, help="默认用 suggest_hyper")
    p.add_argument("--lnet-depth", type=int, default=None)
    p.add_argument(
        "--lambda-physics",
        type=float,
        default=0.5,
        help="PINN 摩擦物理项权重 λ（Eq. 6），仅 stribeck_pinn",
    )
    p.add_argument("--energy-loss", action="store_true", help="总力矩 + 刚体能量守恒")
    p.add_argument(
        "--tau-loss",
        choices=("mse", "smape"),
        default="smape",
        help="总力矩监督（多关节推荐 smape）",
    )
    p.add_argument("--smape-eps", type=float, default=1e-3)
    p.add_argument("-m", nargs="?", const=0, default=0, type=int, help="保存 checkpoint")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints" / "mysteric_robot.pt")
    p.add_argument("-c", nargs="?", const=1, default=1, type=int)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise SystemExit(f"数据不存在: {args.data}")

    cuda = bool(args.c) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    train_data, test_data, _, _ = load_dataset(
        filename=str(args.data),
        test_label=tuple(args.test_labels),
    )
    train_labels, *_ = train_data
    test_labels, test_qp, test_qv, test_qa, *_rest = test_data
    test_tau = _rest[2]
    n_dof = test_qp.shape[1]

    raw = load_pickle_trajectories(str(args.data))
    train_label_set = set(train_labels)
    qp, qv, qa, tau, _tau_rigid, tau_fri = stack_trajectories_to_flat(
        raw, train_labels=train_label_set
    )
    tensors = build_mysteric_tensors(
        qp, qv, qa, tau, tau_fri, args.seq_len, device=device
    )
    qi, qdi, qddi, taui = tensors["qi"], tensors["qdi"], tensors["qddi"], tensors["taui"]
    tau_fri_t = tensors["tau_fri"]
    q_seq, qd_seq = tensors["q_seq"], tensors["qd_seq"]

    hyper = suggest_hyper(n_dof, qi.shape[0], base="delan_model")
    l_w = args.lnet_width if args.lnet_width is not None else hyper["n_width"]
    l_d = args.lnet_depth if args.lnet_depth is not None else hyper["n_depth"]

    model = MystericNet(
        dof=n_dof,
        seq_len=args.seq_len,
        lnet_hidden=l_w,
        lnet_layers=l_d,
        friction_backend=args.friction_backend,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    N = qi.shape[0]
    B = args.batch

    print(
        f"device={device}  n_dof={n_dof}  friction={args.friction_backend}  "
        f"train N={N}  test N={test_qp.shape[0]}"
    )

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=device)
        loss_acc = steps = 0
        for s in range(0, N, B):
            idx = perm[s : s + B]
            if idx.numel() < 4:
                continue
            qb, qdb, qddb = qi[idx], qdi[idx], qddi[idx]
            taub, tfb = taui[idx], tau_fri_t[idx]
            qs, qds = q_seq[idx], qd_seq[idx]

            tau_hat, _core, tau_fri, _H, g_hat, tau_phys = model(qb, qdb, qddb, qs, qds)

            if args.friction_backend == "stribeck_pinn":
                assert tau_phys is not None
                lf, _, _ = friction_pinn_loss(
                    tau_fri, tfb, tau_phys, lambda_physics=args.lambda_physics
                )
            else:
                lf = torch.mean((tau_fri - tfb) ** 2)

            ltau = torque_loss(tau_hat, taub, args.tau_loss, smape_eps=args.smape_eps)
            loss = ltau + lf

            if args.energy_loss:
                _, _, lE = mysteric_losses(
                    model.lnet, tau_hat, taub, tau_fri, qb, qdb, qddb, g_hat
                )
                loss = loss + lE

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_acc += float(loss.detach())
            steps += 1

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            with torch.no_grad():
                n_test = min(512, test_qp.shape[0])
                qt = torch.from_numpy(test_qp[:n_test]).float().to(device)
                qdt = torch.from_numpy(test_qv[:n_test]).float().to(device)
                qddt = torch.from_numpy(test_qa[:n_test]).float().to(device)
                tt = torch.from_numpy(test_tau[:n_test]).float().to(device)
                qs_t = qt.unsqueeze(1).expand(-1, args.seq_len, -1)
                qds_t = qdt.unsqueeze(1).expand(-1, args.seq_len, -1)
                th, _, _, _, _, _ = model(qt, qdt, qddt, qs_t, qds_t)
                rmse = float(torch.sqrt(torch.mean((th - tt) ** 2)).cpu())
            print(f"epoch {epoch:4d}  loss={loss_acc/max(steps,1):.5f}  RMSE_test≈{rmse:.4f}")

    if args.m:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dof": n_dof,
                "seq_len": args.seq_len,
                "friction_backend": args.friction_backend,
                "lambda_physics": args.lambda_physics,
                "data_path": str(args.data.resolve()),
            },
            args.save,
        )
        print(f"已保存: {args.save}")


if __name__ == "__main__":
    main()
