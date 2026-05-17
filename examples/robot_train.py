#!/usr/bin/env python3
"""
在 robot.pickle（或任意 DeLaN pickle）上训练 Mysteric-Net：L-Net + 摩擦网络。

摩擦后端（Hu 等 SCV / PINN）:
  --friction-backend stribeck       纯可学习 SCV 物理模型
  --friction-backend stribeck_pinn  MLP + SCV 物理损失（论文 Eq. (6)）
  --friction-backend tcn            原 TCN（Yeo 等）
  --friction-backend fo_cascade       TCN₁→MLP→TCN₂（Xun 图 4）
  --friction-backend fo_cascade_pinn  fo_cascade + SCV PINN（Eq. 6）

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
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import load_dataset, suggest_hyper, torque_loss
from RobotDynamics.FrictionModule import (
    build_mysteric_tensors,
    friction_pinn_loss,
    load_pickle_trajectories,
    mysteric_losses,
    pickle_has_mcg_decomposition,
    stack_trajectories_to_flat,
)
from RobotDynamics.MystericNet import MystericNet


def _checkpoint_payload(
    model: MystericNet,
    *,
    n_dof: int,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "dof": n_dof,
        "seq_len": args.seq_len,
        "lnet_hidden": l_w,
        "lnet_layers": l_d,
        "friction_backend": args.friction_backend,
        "lambda_physics": args.lambda_physics,
        "energy_loss": args.energy_loss,
        "tau_loss": args.tau_loss,
        "data_path": str(args.data.resolve()),
    }


def _save_checkpoint(
    path: Path,
    model: MystericNet,
    *,
    n_dof: int,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
    interrupted: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model, n_dof=n_dof, args=args, l_w=l_w, l_d=l_d, epoch=epoch
    )
    payload["interrupted"] = interrupted
    torch.save(payload, path)
    print(f"已保存: {path.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mysteric-Net（L-Net + 摩擦）训练")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument(
        "--friction-backend",
        choices=("tcn", "fo_cascade", "fo_cascade_pinn", "stribeck", "stribeck_pinn"),
        default="stribeck_pinn",
    )
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lnet-width", type=int, default=None, help="默认用 suggest_hyper")
    p.add_argument("--lnet-depth", type=int, default=None)
    p.add_argument(
        "--lambda-physics",
        type=float,
        default=0.5,
        help="PINN 摩擦物理项权重 λ（Eq. 6），用于 stribeck_pinn / fo_cascade_pinn",
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
    p.add_argument(
        "--friction-label",
        choices=("auto", "none", "decomposition"),
        default="auto",
        help="auto=有 m/c/g 分解才监督 τ_fri；none=仅总力矩+（可选）PINN 物理项",
    )
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
    if args.friction_label == "auto":
        supervise_fri = pickle_has_mcg_decomposition(raw)
    else:
        supervise_fri = args.friction_label == "decomposition"
    if not supervise_fri:
        print(
            "摩擦监督: 无 τ_fri 真值 → 仅用总力矩 τ_hat 监督；"
            "PINN 时另加 SCV 物理项（不监督摩擦标签）。"
        )

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

    epoch_times: list[float] = []
    last_epoch = 0

    def _interrupt_save_path() -> Path:
        if args.m:
            return args.save
        stem = args.save.stem
        if not stem.endswith("_interrupt"):
            stem = f"{stem}_interrupt"
        return args.save.with_name(stem + args.save.suffix)

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            t_epoch_start = time.perf_counter()
            perm = torch.randperm(N, device=device)
            loss_acc = steps = 0
            for s in range(0, N, B):
                idx = perm[s : s + B]
                if idx.numel() < 4:
                    continue
                qb, qdb, qddb = qi[idx], qdi[idx], qddi[idx]
                taub, tfb = taui[idx], tau_fri_t[idx]
                qs, qds = q_seq[idx], qd_seq[idx]

                tau_hat, _core, tau_fri, _H, g_hat, tau_phys = model(
                    qb, qdb, qddb, qs, qds
                )

                lf = torch.zeros((), device=device, dtype=qb.dtype)
                if args.friction_backend in ("stribeck_pinn", "fo_cascade_pinn"):
                    assert tau_phys is not None
                    lf, _, _ = friction_pinn_loss(
                        tau_fri,
                        tfb,
                        tau_phys,
                        lambda_physics=args.lambda_physics,
                        supervise_friction=supervise_fri,
                    )
                elif supervise_fri:
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

            epoch_sec = time.perf_counter() - t_epoch_start
            epoch_times.append(epoch_sec)

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
                win = epoch_times[-50:]
                avg_50 = sum(win) / len(win)
                print(
                    f"epoch {epoch:4d}  loss={loss_acc/max(steps,1):.5f}  "
                    f"RMSE_test≈{rmse:.4f}  "
                    f"time/epoch={epoch_sec:.2f}s  avg50={avg_50:.2f}s/epoch"
                )

    except KeyboardInterrupt:
        print(f"\n训练被中断 (Ctrl+C)，保存 epoch={last_epoch} 的权重 …", flush=True)
        _save_checkpoint(
            _interrupt_save_path(),
            model,
            n_dof=n_dof,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            interrupted=True,
        )
        raise SystemExit(130) from None

    if args.m:
        _save_checkpoint(
            args.save,
            model,
            n_dof=n_dof,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            interrupted=False,
        )


if __name__ == "__main__":
    main()
