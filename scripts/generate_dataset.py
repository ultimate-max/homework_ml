#!/usr/bin/env python3
"""生成 2-DoF 逆动力学合成数据集（.npz），供 examples/synthetic_train.py --data 使用。"""

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
    p.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic_2dof_inverse.npz")
    p.add_argument("--T", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=30)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"saved: {args.out}  samples={d['qi'].shape[0]}  seq_len={args.seq_len}")


if __name__ == "__main__":
    main()
