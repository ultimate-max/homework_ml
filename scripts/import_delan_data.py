#!/usr/bin/env python3
"""
将机械臂轨迹导入为 DeLaN ``character_data.pickle``（任意 n_dof）。

示例:
  # MATLAB（与官方 tools/mat_to_character_pickle 相同约定）
  python scripts/import_delan_data.py -i robot.mat -o data/robot.pickle --transpose

  # 单条 npz: 键 qp, qv, qa, tau [, t, m, c, g]
  python scripts/import_delan_data.py -i traj.npz -o data/robot.pickle

  # 查看已有 pickle
  python scripts/import_delan_data.py --inspect data/robot.pickle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.delan_data import inspect_dataset
from mysteric_net.delan_hyper import suggest_hyper
from mysteric_net.delan_import import import_mat, import_npz, inspect_pickle_dict, save_pickle


def main() -> int:
    p = argparse.ArgumentParser(description="导入 DeLaN 训练用 character_data.pickle")
    p.add_argument("-i", "--input", type=Path, help="输入 .mat 或 .npz")
    p.add_argument("-o", "--output", type=Path, help="输出 .pickle")
    p.add_argument("--inspect", type=Path, help="仅查看已有 pickle 并退出")
    p.add_argument("--n-dof", type=int, default=None)
    p.add_argument("--root", type=str, default=None, help="MAT struct 名")
    p.add_argument("--transpose", action="store_true", help="MATLAB n_dof×T 时先转置")
    p.add_argument(
        "--suggest-hyper",
        action="store_true",
        help="导入后按样本量打印推荐超参（delan_model 基准）",
    )
    args = p.parse_args()

    if args.inspect:
        print(inspect_dataset(args.inspect))
        return 0

    if args.input is None or args.output is None:
        p.error("导入需同时指定 -i 与 -o（或仅用 --inspect）")

    if not args.input.is_file():
        print(f"找不到: {args.input}", file=sys.stderr)
        return 1

    suf = args.input.suffix.lower()
    if suf == ".mat":
        data = import_mat(
            args.input,
            n_dof=args.n_dof,
            transpose=args.transpose,
            root=args.root,
        )
    elif suf == ".npz":
        data = import_npz(args.input, n_dof=args.n_dof)
    else:
        print("仅支持 .mat / .npz", file=sys.stderr)
        return 1

    save_pickle(data, args.output)
    print(f"已写入 {args.output}")
    print(inspect_pickle_dict(data))

    if args.suggest_hyper:
        n_dof = int(data["qp"][0].shape[1])
        n_samples = sum(int(np.asarray(data["qp"][i]).shape[0]) for i in range(len(data["labels"])))
        h = suggest_hyper(n_dof, n_samples, base="delan_model")
        print(
            f"\n推荐超参 (preset=auto): width={h['n_width']}, depth={h['n_depth']}, "
            f"lr={h['learning_rate']}, batch={h['n_minibatch']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
