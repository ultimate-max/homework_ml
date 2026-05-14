#!/usr/bin/env python3
"""生成 2-DoF 逆动力学测试数据集（.npz），用于模型评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.synthetic_plant import simulate_2dof_inverse_dynamics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "test_synthetic_2dof_inverse.npz")
    p.add_argument("--T", type=int, default=2000, help="测试轨迹长度")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--seed", type=int, default=42, help="随机种子以确保与训练集不同")
    args = p.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"生成测试数据集...")
    d = simulate_2dof_inverse_dynamics(T=args.T, seq_len=args.seq_len, device="cpu")

    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float64)

    np.savez_compressed(
        args.out,
        q=to_np(d["q"]),
        qd=to_np(d["qd"]),
        qdd=to_np(d["qdd"]),
        tau=to_np(d["tau"]),
        tau_rigid=to_np(d["tau_rigid"]),
        tau_fri=to_np(d["tau_fri"]),
        t=to_np(d["t"]),
        qi=to_np(d["qi"]),
        qdi=to_np(d["qdi"]),
        qddi=to_np(d["qddi"]),
        taui=to_np(d["taui"]),
        q_seq=to_np(d["q_seq"]),
        qd_seq=to_np(d["qd_seq"]),
        dt=np.float64(d["dt"]),
        seq_len=np.int32(args.seq_len),
        dof=np.int32(d["dof"]),
    )
    print(f"测试集已保存: {args.out}")
    print(f"  样本数: {d['qi'].shape[0]}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  自由度: {d['dof']}")


if __name__ == "__main__":
    main()
