#!/usr/bin/env python3
"""
将机械臂轨迹导入为 DeLaN ``character_data.pickle``（任意 n_dof）。

示例:
  # MATLAB（与官方 tools/mat_to_character_pickle 相同约定）
  python scripts/import_delan_data.py -i robot.mat -o data/robot.pickle --transpose

  # 默认对 qp/qv/qa/tau 等做 200 Hz 低通（采样率由 .mat 的 dt 或 t 推断）
  python scripts/import_delan_data.py -i data/motor_character_data.mat -o data/motor.pickle

  # 指定截止频率 / 关闭滤波
  python scripts/import_delan_data.py -i robot.mat -o data/robot.pickle --filter-cutoff 100
  python scripts/import_delan_data.py -i robot.mat -o data/robot.pickle --no-filter

  # 单条 npz: 键 qp, qv, qa, tau [, t, m, c, g]
  python scripts/import_delan_data.py -i traj.npz -o data/robot.pickle

  # 查看已有 pickle
  python scripts/import_delan_data.py --inspect data/robot.pickle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import (
    import_mat,
    import_npz,
    inspect_dataset,
    inspect_pickle_dict,
    save_pickle,
    suggest_hyper,
)
from RobotDynamics.DeLaN.signal_filter import (
    filter_character_data,
    read_mat_scalar_dt,
)


def _parse_args() -> argparse.Namespace:
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
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--filter-cutoff",
        type=float,
        default=200.0,
        metavar="HZ",
        help="导入后低通滤波截止频率 (Hz)，默认 200；采样率 fs=1/dt",
    )
    g.add_argument(
        "--no-filter",
        action="store_true",
        help="不做低通滤波",
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth 滤波器阶数，默认 4",
    )
    p.add_argument(
        "--dt-hint",
        type=float,
        default=None,
        metavar="SEC",
        help="显式采样周期 (s)，覆盖 .mat 内 dt 与各轨迹 t 的推断",
    )
    p.add_argument(
        "--filter-keys",
        nargs="+",
        default=None,
        help="要滤波的字段，默认 qp qv qa tau p pdot",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.inspect:
        print(inspect_dataset(args.inspect))
        return 0

    if args.input is None or args.output is None:
        print("导入需同时指定 -i 与 -o（或仅用 --inspect）", file=sys.stderr)
        return 1

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

    if not args.no_filter:
        dt_hint = args.dt_hint
        if dt_hint is None and suf == ".mat":
            dt_hint = read_mat_scalar_dt(args.input, root=args.root)
        if dt_hint is not None:
            print(f"滤波采样周期 dt={dt_hint:g} s (fs={1.0/dt_hint:g} Hz)", file=sys.stderr)
        filter_character_data(
            data,
            cutoff_hz=float(args.filter_cutoff),
            order=int(args.filter_order),
            keys=tuple(args.filter_keys) if args.filter_keys else None,
            dt_hint=dt_hint,
        )

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
