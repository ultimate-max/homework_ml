#!/usr/bin/env python3
"""
Minimal training loop for Mysteric-Net on synthetic 2-DoF data (sanity check).

Paper defaults (Table I): lr=7e-4, weight_decay=1e-5, L=30, H-Net channels=8, kernel=3.

默认使用纯力矩 MSE 以在 CPU 上快速跑通；加 --energy-loss 启用论文式 l_tau+l_E（明显更慢）。

数据：不传 --data 时内存仿真；传 --data 时从 scripts/generate_dataset.py 生成的 .npz 加载。
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

from RobotDynamics.FrictionModule import (
    friction_pinn_loss,
    friction_supervised_loss,
    mysteric_losses,
    simulate_2dof_inverse_dynamics,
)
from RobotDynamics.MystericNet import MystericNet


def load_dataset_npz(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    z = np.load(path, allow_pickle=False)
    seq_len = int(z["seq_len"])
    dof = int(z["dof"])
    qi = torch.from_numpy(z["qi"]).to(device=device, dtype=torch.float32)
    qdi = torch.from_numpy(z["qdi"]).to(device=device, dtype=torch.float32)
    qddi = torch.from_numpy(z["qddi"]).to(device=device, dtype=torch.float32)
    taui = torch.from_numpy(z["taui"]).to(device=device, dtype=torch.float32)
    q_seq = torch.from_numpy(z["q_seq"]).to(device=device, dtype=torch.float32)
    qd_seq = torch.from_numpy(z["qd_seq"]).to(device=device, dtype=torch.float32)
    return qi, qdi, qddi, taui, q_seq, qd_seq, seq_len, dof


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--energy-loss", action="store_true", help="使用论文 l_tau + l_E（CPU 上很慢）")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--T", type=int, default=4000, help="无 --data 时轨迹长度")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--data", type=Path, default=None, help="generate_dataset.py 输出的 .npz")
    p.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints", help="模型保存目录")
    p.add_argument("--save-name", type=str, default="RobotDynamics", help="模型保存名称")
    p.add_argument(
        "--friction-backend",
        choices=("tcn", "fo_cascade", "fo_cascade_pinn", "stribeck", "stribeck_pinn", "gms", "gms_pinn"),
        default="tcn",
    )
    p.add_argument("--lambda-physics", type=float, default=0.5)
    p.add_argument(
        "--fri-loss",
        choices=("mse", "smape"),
        default="smape",
        help="摩擦监督与 PINN 物理项",
    )
    p.add_argument("--smape-eps", type=float, default=1e-3)
    args = p.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data is not None:
        if not args.data.is_file():
            raise SystemExit(f"数据文件不存在: {args.data}")
        qi, qdi, qddi, taui, q_seq, qd_seq, seq_len, dof = load_dataset_npz(args.data, device)
    else:
        seq_len = 30
        dof = 2
        d = simulate_2dof_inverse_dynamics(T=args.T, seq_len=seq_len, device=device)
        qi, qdi, qddi, taui, q_seq, qd_seq = d["qi"], d["qdi"], d["qddi"], d["taui"], d["q_seq"], d["qd_seq"]

    model = MystericNet(dof=dof, seq_len=seq_len, friction_backend=args.friction_backend).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=7.0e-4, weight_decay=1.0e-5)

    B = args.batch
    N = qi.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        loss_acc = 0.0
        steps = 0
        for s in range(0, N, B):
            idx = perm[s : s + B]
            if idx.numel() < 8:
                continue
            qb = qi[idx]
            qdb = qdi[idx]
            qddb = qddi[idx]
            taub = taui[idx]
            q_seq_b = q_seq[idx]
            qd_seq_b = qd_seq[idx]

            tau_hat, _core, tau_fri_hat, _H_hat, g_hat, tau_phys = model(
                qb, qdb, qddb, q_seq_b, qd_seq_b
            )
            if args.friction_backend in ("stribeck_pinn", "fo_cascade_pinn", "gms_pinn") and tau_phys is not None:
                tau_fri_true = taub - _core.detach()
                lf, _, _ = friction_pinn_loss(
                    tau_fri_hat,
                    tau_fri_true,
                    tau_phys,
                    lambda_physics=args.lambda_physics,
                    fri_loss=args.fri_loss,
                    smape_eps=args.smape_eps,
                )
            else:
                lf = torch.zeros((), device=device, dtype=qb.dtype)
            if args.energy_loss:
                lt, ltau, lE = mysteric_losses(
                    model.lnet, tau_hat, taub, tau_fri_hat, qb, qdb, qddb, g_hat
                )
                lt = lt + lf
            else:
                ltau = torch.mean((tau_hat - taub) ** 2)
                lt = ltau + lf

            opt.zero_grad(set_to_none=True)
            lt.backward()
            opt.step()

            loss_acc += float(lt.detach().cpu())
            steps += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tag = "l_tau+l_E" if args.energy_loss else "l_tau"
            print(f"epoch {epoch+1:03d}  {tag}  l={loss_acc/max(steps,1):.5f}")

    with torch.no_grad():
        tau_hat, _, _, _, _g, _ = model(
            qi[:512], qdi[:512], qddi[:512], q_seq[:512], qd_seq[:512]
        )
        rmse = torch.sqrt(torch.mean((tau_hat - taui[:512]) ** 2)).item()
        print(f"RMSE torque (subset): {rmse:.5f}")

    # 保存模型
    save_path = args.save_dir / f"{args.save_name}.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'dof': dof,
        'seq_len': seq_len,
        'friction_backend': args.friction_backend,
        'rmse': rmse,
    }, save_path)
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
